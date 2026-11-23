"""Shared RF-DETR-L + RTMPose-L backend for all monitor cameras.

Normalized boxes and COCO17 joints carry explicit display states.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
MODEL_DIR=PROJECT/'models/pose'
MODEL_NAME='RF-DETR-L + RTMPose-L'
for name,relative in [('RF_HOME',''),('HF_HOME','huggingface'),('TORCH_HOME','torch')]:
    os.environ.setdefault(name,str(MODEL_DIR/relative))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')


def load_pose_settings(path=None):
    """Read optional local backend config; malformed settings preserve ORT."""
    path=Path(path) if path is not None else PROJECT/'pose_config.json'
    if not path.is_file():return {'backend':'onnxruntime'}
    requested='onnxruntime'
    try:
        settings=json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(settings,dict):raise ValueError('pose config must be an object')
        requested=settings.get('backend','onnxruntime')
        if not isinstance(requested,str) or requested not in {'onnxruntime','tensorrt'}:
            raise ValueError('unsupported pose backend')
        optimized=settings.get('reuse_preparation',True)
        if not isinstance(optimized,bool):raise ValueError('reuse_preparation must be boolean')
        result={'backend':requested,'reuse_preparation':optimized}
        for key,argument in (('engine','engine_path'),('acceptance','acceptance_path')):
            value=settings.get(key)
            if value is None:
                if requested=='tensorrt':raise ValueError(f'{key} is required for tensorrt')
                continue
            if not isinstance(value,str) or not value.strip() or Path(value).is_absolute():
                raise ValueError(f'{key} must be a project-relative path')
            resolved=(PROJECT/value).resolve()
            if not resolved.is_relative_to(PROJECT.resolve()):raise ValueError(f'{key} is outside the project')
            result[argument]=resolved
        return result
    except (OSError,UnicodeError,ValueError,TypeError) as exc:
        return {'backend':requested if isinstance(requested,str) else 'invalid',
                'backend_config_error':f'pose_config_invalid: {type(exc).__name__}: {exc}'}


class PersonPoseDetector:
    supports_pose=True

    def __init__(self, confidence=.35, threads=2, warmup=True, detector_profile=None):
        from .rfdetr_detector import PersonDetector, PROFILES
        from .pose_estimator import PoseEstimator
        from .pose_quality import PoseQualityFilter
        self.model_name=MODEL_NAME
        if detector_profile is None:
            settings=json.loads((PROJECT/'detector_config.json').read_text(encoding='utf-8-sig'))
            detector_profile=settings['profile']
        if detector_profile not in PROFILES:
            raise ValueError(f'Unknown detector profile: {detector_profile!r}')
        self.model_path=MODEL_DIR/PROFILES[detector_profile]['filename']
        self.pose_model_path=MODEL_DIR/'rtmpose-l.fp16.onnx'
        self.confidence=float(confidence)
        self.detector=PersonDetector(self.model_path,confidence=self.confidence,profile=detector_profile)
        self.pose=PoseEstimator(self.pose_model_path,threads=threads,batch_size=8,precision='fp16',**load_pose_settings())
        self.quality=PoseQualityFilter()
        self.device='cuda:0'
        self.device_name=self.detector.device
        self.last_metrics={}
        if warmup:
            # Warm common shapes before opening cameras, so cold initialization
            # is not presented as a stale real camera observation.
            image=np.zeros((720,1280,3),np.uint8)
            self.detector.detect(image)
            for count in (8,1):
                self.pose.estimate(image,np.tile([200,100,500,650],(count,1)))

    def detect(self,frame):
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or not frame.size or min(frame.shape[:2])<2:
            raise ValueError('frame must be nonempty uint8 BGR')
        started=time.perf_counter()
        height,width=frame.shape[:2]
        rows=np.asarray(self.detector.detect(frame),np.float32).reshape(-1,6).copy()
        rows=rows[np.isfinite(rows).all(axis=1)&(rows[:,4]>=self.confidence)]
        rows[:,(0,2)]=np.clip(rows[:,(0,2)],0,width)
        rows[:,(1,3)]=np.clip(rows[:,(1,3)],0,height)
        rows=rows[(rows[:,2]>rows[:,0])&(rows[:,3]>rows[:,1])]
        detected_at=time.perf_counter()
        boxes=rows[:,:4]
        points,scores=self.pose.estimate(frame,boxes)
        posed_at=time.perf_counter()
        quality=self.quality.apply(boxes,points,scores,frame.shape,
            ambiguity_mask=self.pose.last_ambiguity_mask,crop_border_mask=self.pose.last_crop_border_mask)
        states=quality['states']
        output=[]
        for i,row in enumerate(rows):
            record=dict(zip(('x1','y1','x2','y2','score'),map(float,[row[0]/width,row[1]/height,row[2]/width,row[3]/height,row[4]])))
            record['keypoints']=[{'x':float(points[i,j,0]/width),'y':float(points[i,j,1]/height),
                                  'score':float(scores[i,j]),'state':int(states[i,j])} for j in range(17)]
            output.append(record)
        ended=time.perf_counter()
        self.last_metrics={'detectMs':round((detected_at-started)*1000,2),'poseMs':round((posed_at-detected_at)*1000,2),
            'qualityMs':round((ended-posed_at)*1000,2),'poseProcessedPeople':len(rows),
            'poseValidPeople':int(quality['valid_skeleton_count']),
            'reliableKeypoints':int(quality['reliable_count']),'estimatedKeypoints':int(quality['estimated_count']),
            'rejectedKeypoints':int(quality['rejected_count']),
            'poseStages':dict(self.pose.last_timings),'trackingEnabled':False,'temporalSmoothing':False}
        return output

    def close(self):
        """Call only after the shared inference worker has stopped using us."""
        pose=getattr(self,'pose',None)
        if pose is not None and hasattr(pose,'close'):pose.close()
