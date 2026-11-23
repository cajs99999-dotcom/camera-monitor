"""Same RTMPose-L, RGB/float32 preparation, dynamic CUDA batches.

Raw coordinates/scores remain the estimator's output. Ambiguity flags are
separate diagnostics; callers decide whether to hide or mark those joints.
"""
from pathlib import Path
import time

import cv2
import numpy as np


class PoseEstimator:
    def __init__(self,model_path,device="cuda",threads=2,batch_size=8,precision="fp16",ambiguity_threshold=.03,
                 *,backend="onnxruntime",engine_path=None,acceptance_path=None,
                 backend_config_error=None,reuse_preparation=True):
        if not isinstance(batch_size,int) or not 1<=batch_size<=16:
            raise ValueError("batch_size must be an integer from 1 to 16")
        if precision not in {"fp32","fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        if device=="cpu" and precision!="fp32":
            raise ValueError("CPU mode requires fp32")
        if not isinstance(threads,int) or threads<1:
            raise ValueError("threads must be positive")
        if not 0<=ambiguity_threshold<=1:
            raise ValueError("ambiguity_threshold must be between 0 and 1")
        model_path=Path(model_path).resolve()
        if precision=="fp16" and not model_path.stem.endswith(".fp16"):
            model_path=model_path.with_name(model_path.stem+".fp16.onnx")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.backend_requested=backend if isinstance(backend,str) else "invalid"
        self.backend_selected="onnxruntime"
        self.fallback_reason=backend_config_error
        self.closed=False
        session=None
        if self.fallback_reason is None and backend=="tensorrt":
            try:
                if device not in {"cuda","cuda:0"} or precision!="fp16":
                    raise ValueError("TensorRT profile requires cuda:0 and fp16 graph")
                if engine_path is None or acceptance_path is None:
                    raise ValueError("TensorRT engine and acceptance paths are required")
                from .pose_tensorrt import TensorRTPoseSession
                session=TensorRTPoseSession(engine_path,model_path,acceptance_path)
                self.backend_selected="tensorrt"
            except Exception as exc:
                self.fallback_reason=f"{type(exc).__name__}: {exc}"
        elif self.fallback_reason is None and backend!="onnxruntime":
            self.fallback_reason="Unsupported pose backend; using onnxruntime"
        if session is None:
            session=self._create_ort_session(model_path,device,threads)
        actual=session.get_providers()
        shape=session.get_inputs()[0].shape
        if len(shape)!=4 or shape[-2:]!=[384,288]:
            if hasattr(session,"close"):session.close()
            raise ValueError(f"Expected RTMPose-L 384x288, received {shape}")
        if isinstance(shape[0],int) and shape[0]!=batch_size:
            if hasattr(session,"close"):session.close()
            raise ValueError(f"Model batch is fixed at {shape[0]}, requested {batch_size}")
        self._session=session
        self._input_name=session.get_inputs()[0].name
        self._output_names=[x.name for x in session.get_outputs()]
        self.reuse_preparation=bool(reuse_preparation)
        if self.reuse_preparation:
            from .pose_preparation import RTMPosePreparation
            self._preprocess=RTMPosePreparation()
        else:
            from rtmlib import RTMPose
            self._preprocess=RTMPose.__new__(RTMPose)
            self._preprocess.model_input_size=(288,384)
            self._preprocess.mean=np.array((123.675,116.28,103.53),np.float32)
            self._preprocess.std=np.array((58.395,57.12,57.375),np.float32)
        self.model_path=model_path;self.device=device;self.precision=precision
        self.pose_batch_size=batch_size;self.batch_size=batch_size
        self.actual_providers=list(actual);self.providers=self.actual_providers
        self.backend=f"{self.backend_selected} / {precision} / RGB / batch{batch_size}"
        self.ambiguity_threshold=float(ambiguity_threshold)
        self.last_ambiguity_mask=np.empty((0,17),bool)
        self.last_peak_margins=np.empty((0,17),np.float32)
        self.last_crop_border_mask=np.empty((0,17),bool)
        self.last_crop_boxes=np.empty((0,4),np.float32)
        self.last_timings={}

    @staticmethod
    def _create_ort_session(model_path,device,threads):
        # Importing torch first retains the existing CUDA13 DLL setup for ORT.
        import torch
        import onnxruntime as ort
        if device=="cpu":
            providers=["CPUExecutionProvider"]
        elif device=="cuda" or device.startswith("cuda:"):
            device_id=int(device.split(":",1)[1]) if ":" in device else 0
            if device_id<0:raise ValueError("CUDA device index must be nonnegative")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider is unavailable")
            providers=[("CUDAExecutionProvider",{"device_id":device_id,"gpu_mem_limit":2*1024**3})]
        else:raise ValueError("device must be cpu, cuda, or cuda:<index>")
        options=ort.SessionOptions();options.log_severity_level=3
        options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session=ort.InferenceSession(str(model_path),sess_options=options,providers=providers)
        actual=session.get_providers()
        if device!="cpu" and "CUDAExecutionProvider" not in actual:
            raise RuntimeError(f"CUDA failed to initialize: {actual}")
        session.disable_fallback()
        return session

    @staticmethod
    def _decode_and_margin(sx,sy):
        from rtmlib.tools.pose_estimation.rtmpose import get_simcc_maximum
        locs,scores=get_simcc_maximum(sx,sy)
        gaps=[]
        for axis in (sx,sy):
            indices=np.argmax(axis,axis=-1)
            main=np.max(axis,axis=-1)
            outside=np.abs(np.arange(axis.shape[-1])[None,None,:]-indices[:,:,None])>10
            second=np.max(np.where(outside,axis,-np.inf),axis=-1)
            gaps.append(np.clip((main-second)/np.maximum(np.abs(main),1e-6),0,1))
        return locs,scores,np.minimum(*gaps).astype(np.float32)

    def estimate(self,frame,boxes):
        if getattr(self,"closed",False):
            raise RuntimeError("Pose estimator is closed")
        started=time.perf_counter()
        preparation_ms=runtime_ms=decode_ms=0.
        batch_calls=0
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or min(frame.shape[:2])<2:
            raise ValueError("frame must be a nonempty uint8 BGR HxWx3 image")
        boxes=np.asarray(boxes,dtype=np.float32)
        if boxes.size==0:
            self.last_ambiguity_mask=np.empty((0,17),bool)
            self.last_peak_margins=np.empty((0,17),np.float32)
            self.last_crop_border_mask=np.empty((0,17),bool)
            self.last_crop_boxes=np.empty((0,4),np.float32)
            self.last_timings={"prepareMs":0.,"runtimeMs":0.,"decodeMs":0.,"batchCalls":0,"totalMs":(time.perf_counter()-started)*1000}
            return np.empty((0,17,2),np.float32),np.empty((0,17),np.float32)
        if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all():
            raise ValueError("boxes must be finite Nx4 coordinates")
        boxes=boxes.copy();h,w=frame.shape[:2]
        boxes[:,(0,2)]=np.clip(boxes[:,(0,2)],0,w)
        boxes[:,(1,3)]=np.clip(boxes[:,(1,3)],0,h)
        indices=np.flatnonzero((boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]))
        points=np.zeros((len(boxes),17,2),np.float32)
        all_scores=np.zeros((len(boxes),17),np.float32)
        margins=np.zeros((len(boxes),17),np.float32)
        borders=np.zeros((len(boxes),17),bool)
        crop_boxes=np.zeros((len(boxes),4),np.float32)
        if len(indices):
            # Official export metadata specifies Normalize(to_rgb=True).
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            preparation_ms+=(time.perf_counter()-started)*1000
            for offset in range(0,len(indices),self.batch_size):
                prepared_at=time.perf_counter()
                chunk=indices[offset:offset+self.batch_size]
                inputs=np.empty((len(chunk),3,384,288),np.float32)
                centers=[];scales=[]
                for i,index in enumerate(chunk):
                    image,center,scale=self._preprocess.preprocess(rgb,boxes[index].tolist())
                    inputs[i]=image.transpose(2,0,1)
                    centers.append(center);scales.append(scale)
                preparation_ms+=(time.perf_counter()-prepared_at)*1000
                runtime_at=time.perf_counter()
                sx,sy=self._session.run(self._output_names,{self._input_name:inputs})
                runtime_ms+=(time.perf_counter()-runtime_at)*1000
                batch_calls+=1
                decoded_at=time.perf_counter()
                if sx.shape[:2]!=(len(chunk),17) or sy.shape[:2]!=(len(chunk),17):
                    raise RuntimeError(f"Unexpected pose output: {sx.shape}, {sy.shape}")
                if not np.isfinite(sx).all() or not np.isfinite(sy).all():
                    raise RuntimeError("Pose model produced non-finite output")
                locs,scores,gaps=self._decode_and_margin(sx,sy)
                centers=np.asarray(centers);scales=np.asarray(scales)
                coords=locs/2./np.array((288,384))*scales[:,None,:]+centers[:,None,:]-scales[:,None,:]/2.
                points[chunk]=coords.astype(np.float32)
                all_scores[chunk]=scores.astype(np.float32)
                margins[chunk]=gaps
                borders[chunk]=(locs[:,:,0]<=1)|(locs[:,:,0]>=sx.shape[-1]-2)|(locs[:,:,1]<=1)|(locs[:,:,1]>=sy.shape[-1]-2)
                crop_boxes[chunk]=np.concatenate((centers-scales/2.,centers+scales/2.),axis=1)
                decode_ms+=(time.perf_counter()-decoded_at)*1000
        self.last_peak_margins=margins
        self.last_ambiguity_mask=margins<self.ambiguity_threshold
        self.last_crop_border_mask=borders
        self.last_crop_boxes=crop_boxes
        self.last_timings={"prepareMs":preparation_ms,"runtimeMs":runtime_ms,"decodeMs":decode_ms,"batchCalls":batch_calls,"totalMs":(time.perf_counter()-started)*1000}
        return points,all_scores

    def close(self):
        if self.closed:return
        self.closed=True
        session=self._session
        self._session=None
        if hasattr(session,"close"):session.close()
