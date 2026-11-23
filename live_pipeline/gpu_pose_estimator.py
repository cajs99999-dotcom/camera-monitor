"""Isolated RTMPose candidate binding a CUDA affine crop directly to TensorRT."""
from __future__ import annotations

import ctypes as ct
from threading import Lock
import time

import cv2
import numpy as np

from .gpu_pose_cuda import CUDAWarpKernel
from .gpu_pose_warp import HEIGHT, WIDTH, coordinate_terms, crop_geometry, pose_lut


class _PoseWorkspace:
    """Own bounded staging and CUDA buffers, but borrow the existing TRT runner."""

    def __init__(self, runner, max_frame_bytes, kernel_factory=CUDAWarpKernel):
        self.runner = runner
        self.cuda = runner.cuda
        self._allocations = []
        self.kernel = None
        self.closed = False
        try:
            self.activate()
            self.kernel = kernel_factory()
            self.frame = self._alloc((max_frame_bytes,), np.uint8)
            self.terms = self._alloc((16, 2, WIDTH+HEIGHT), np.int32)
            self.lookup = self._alloc((3, 256), np.float32)
            self.destination = self._alloc((16, 3, HEIGHT, WIDTH), np.float32, pinned=False)
            self.outputs = {name: self._alloc((16, 17, length), np.float32)
                            for name, length in (("simcc_x", 576), ("simcc_y", 768))}
            np.copyto(self.lookup["array"], pose_lut())
            self._upload(self.lookup, self.lookup["size"])
            self.sync()
        except Exception as original:
            try:
                self.close()
            except Exception as cleanup:
                original.add_note(f"CUDA crop cleanup: {cleanup}")
            raise

    def activate(self):
        if self.runner.closed:
            raise RuntimeError("Close candidate buffers before closing the TensorRT runner")
        self.cuda.call("cudaSetDevice", 0)

    def _alloc(self, shape, dtype, *, pinned=True):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buffer = {"size": size}
        if pinned:
            host = self.cuda.create("cudaHostAlloc", size, 0)
            self._allocations.append(("cudaFreeHost", host))
            buffer["host"] = host
            memory = (ct.c_uint8 * size).from_address(host)
            buffer["array"] = np.frombuffer(memory, dtype=dtype).reshape(shape)
        device = self.cuda.create("cudaMalloc", size)
        self._allocations.append(("cudaFree", device))
        buffer["device"] = device
        return buffer

    def _upload(self, buffer, size):
        self.cuda.call("cudaMemcpyAsync", buffer["device"], buffer["host"], size, 1, self.runner.stream)

    def sync(self):
        self.cuda.call("cudaStreamSynchronize", self.runner.stream)

    def stage_frame(self, frame, device_frame):
        self.activate()
        if frame.nbytes > self.frame["size"]:
            raise ValueError("source frame exceeds the CUDA crop's configured image size")
        if device_frame is not None:
            import torch
            if (not isinstance(device_frame, torch.Tensor) or not device_frame.is_cuda
                    or device_frame.device.index != 0 or device_frame.dtype != torch.uint8
                    or tuple(device_frame.shape) != frame.shape or not device_frame.is_contiguous()):
                raise ValueError("device_frame must be the same CUDA device 0 contiguous uint8 HWC BGR frame")
            return device_frame.data_ptr()
        np.copyto(self.frame["array"][:frame.nbytes].reshape(frame.shape), frame)
        self._upload(self.frame, frame.nbytes)
        return self.frame["device"]

    def affine(self, source, width, height, terms):
        n = len(terms)
        if not 1 <= n <= 16:
            raise ValueError("CUDA crop batch size must be between 1 and 16")
        np.copyto(self.terms["array"][:n], terms)
        self._upload(self.terms, terms.nbytes)
        self.kernel.launch(self.runner.stream, source, width, height, self.terms["device"],
                           self.lookup["device"], self.destination["device"], n)

    def infer(self, n):
        runner = self.runner
        input_shape = (n, 3, HEIGHT, WIDTH)
        if not runner.context.set_input_shape("input", input_shape):
            raise RuntimeError("TensorRT rejected CUDA crop input shape")
        shapes = {"input": input_shape, "simcc_x": (n, 17, 576), "simcc_y": (n, 17, 768)}
        for name, address in (("input", self.destination["device"]),
                              *((name, buf["device"]) for name, buf in self.outputs.items())):
            if tuple(runner.context.get_tensor_shape(name)) != shapes[name]:
                raise RuntimeError(f"Unexpected runtime tensor shape for {name}")
            if not runner.context.set_tensor_address(name, address):
                raise RuntimeError(f"Could not bind TensorRT tensor {name}")
        self.cuda.call("cudaEventRecord", runner.start_event, runner.stream)
        if not runner.context.execute_async_v3(runner.stream):
            raise RuntimeError("TensorRT execution enqueue failed")
        self.cuda.call("cudaEventRecord", runner.end_event, runner.stream)
        for buffer in self.outputs.values():
            self.cuda.call("cudaMemcpyAsync", buffer["host"], buffer["device"], buffer["array"][:n].nbytes,
                           2, runner.stream)
        self.sync()
        elapsed = ct.c_float()
        self.cuda.call("cudaEventElapsedTime", ct.byref(elapsed), runner.start_event, runner.end_event)
        sx, sy = (buffer["array"][:n] for buffer in self.outputs.values())
        if not np.isfinite(sx).all() or not np.isfinite(sy).all():
            raise RuntimeError("Non-finite TensorRT output from CUDA crop")
        return sx, sy, float(elapsed.value)

    def copy_inputs(self, n):
        """Debug-only device readback. Not used by the measured inference path."""
        shape = (n, 3, HEIGHT, WIDTH)
        size = n * 3 * HEIGHT * WIDTH * 4
        host = self.cuda.create("cudaHostAlloc", size, 0)
        try:
            self.cuda.call("cudaMemcpyAsync", host, self.destination["device"], size, 2, self.runner.stream)
            self.sync()
            memory = (ct.c_uint8 * size).from_address(host)
            return np.frombuffer(memory, np.float32).reshape(shape).copy()
        finally:
            self.cuda.call("cudaFreeHost", host)

    def close(self):
        if self.closed:
            return
        self.activate()
        self.sync()
        failures = []
        if self.kernel is not None:
            try:
                self.kernel.close()
            except Exception as error:
                failures.append(error)
        for operation, address in reversed(self._allocations):
            try:
                self.cuda.call(operation, address)
            except Exception as error:
                failures.append(error)
        self._allocations.clear()
        self.closed = True
        if failures:
            raise ExceptionGroup("CUDA crop release failed", failures)


class GpuCropPoseEstimator:
    """Reuse the original TensorRT model/stream with CUDA NCHW preparation.

    `device_frame`, if provided, is the exact input frame's device-0 HWC uint8
    BGR tensor. Its producer must have completed before estimate starts. The
    detector's same-stream blocking transfer of results to CPU satisfies this.
    We retain the tensor through the TensorRT stream synchronization. The
    host-only mode uploads the original BGR bytes once, even for >16 people.

    The candidate owns its CUDA scratch, not `base_pose`. Close it before
    closing the original pose estimator. Use one pose worker per object.
    """

    def __init__(self, base_pose, *, max_frame_bytes=1920 * 1080 * 3):
        if getattr(base_pose, "backend_selected", None) != "tensorrt" or getattr(base_pose, "closed", True):
            raise ValueError("CUDA crop requires an open validated TensorRT pose estimator")
        if type(max_frame_bytes) is not int or not 12 <= max_frame_bytes <= 16_000_000:
            raise ValueError("Unsupported CUDA crop frame allocation size")
        self.base_pose = base_pose
        self.closed = False
        self._lock = Lock()
        self._workspace = _PoseWorkspace(base_pose._session.runner, max_frame_bytes)

    def __getattr__(self, name):
        return getattr(self.base_pose, name)

    @staticmethod
    def _frame_shape(frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or min(frame.shape[:2]) < 2 or max(frame.shape[:2]) >= 32768):
            raise ValueError("frame must be a bounded uint8 BGR HxWx3 image")
        return frame.shape[:2]

    def estimate(self, frame, boxes, device_frame=None):
        with self._lock:
            if self.closed:
                raise RuntimeError("CUDA crop estimator is closed")
            try:
                return self._estimate(frame, boxes, device_frame)
            except Exception as error:
                try:
                    self._workspace.sync()
                except Exception as cleanup:
                    error.add_note(f"CUDA crop stream cleanup failed: {cleanup}")
                raise

    def _estimate(self, frame, boxes, device_frame):
        started = time.perf_counter()
        height, width = self._frame_shape(frame)
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError("boxes must be finite Nx4 coordinates")
        boxes = boxes.copy()
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height)
        ids = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        points = np.zeros((len(boxes), 17, 2), np.float32)
        scores = np.zeros((len(boxes), 17), np.float32)
        margins = np.zeros((len(boxes), 17), np.float32)
        borders = np.zeros((len(boxes), 17), bool)
        crop_boxes = np.zeros((len(boxes), 4), np.float32)
        preparation_ms = runtime_ms = decode_ms = trt_ms = 0.
        chunks = 0
        if len(ids):
            workspace = self._workspace
            source = workspace.stage_frame(frame, device_frame)
            preparation_ms += (time.perf_counter()-started)*1000
            for offset in range(0, len(ids), self.batch_size):
                chunk = ids[offset:offset+self.batch_size]
                prepare_at = time.perf_counter()
                centers, scales, matrices = crop_geometry(boxes[chunk])
                terms = coordinate_terms(matrices)
                preparation_ms += (time.perf_counter()-prepare_at)*1000
                run_at = time.perf_counter()
                workspace.affine(source, width, height, terms)
                sx, sy, kernel_ms = workspace.infer(len(chunk))
                runtime_ms += (time.perf_counter()-run_at)*1000
                trt_ms += kernel_ms
                chunks += 1
                decode_at = time.perf_counter()
                locs, score, gaps = self.base_pose._decode_and_margin(sx, sy)
                coords = locs/2./np.array((WIDTH, HEIGHT))*scales[:, None, :]+centers[:, None, :]-scales[:, None, :]/2.
                points[chunk] = coords.astype(np.float32)
                scores[chunk] = score.astype(np.float32)
                margins[chunk] = gaps
                borders[chunk] = (locs[:, :, 0] <= 1) | (locs[:, :, 0] >= sx.shape[-1]-2) | (locs[:, :, 1] <= 1) | (locs[:, :, 1] >= sy.shape[-1]-2)
                crop_boxes[chunk] = np.concatenate((centers-scales/2., centers+scales/2.), axis=1)
                decode_ms += (time.perf_counter()-decode_at)*1000
        self.last_peak_margins = margins
        self.last_ambiguity_mask = margins < self.ambiguity_threshold
        self.last_crop_border_mask = borders
        self.last_crop_boxes = crop_boxes
        self.last_timings = {"prepareMs": preparation_ms, "runtimeMs": runtime_ms, "decodeMs": decode_ms,
                             "trtOnlyMs": trt_ms, "totalMs": (time.perf_counter()-started)*1000,
                             "batchCalls": chunks, "reusedDeviceFrame": bool(len(ids) and device_frame is not None)}
        return points, scores

    def verify_inputs(self, frame, boxes, device_frame=None):
        """Read back GPU input values and compare to installed OpenCV semantics."""
        with self._lock:
            if self.closed:
                raise RuntimeError("CUDA crop estimator is closed")
            height, width = self._frame_shape(frame)
            _, _, forward = crop_geometry(boxes)
            terms = coordinate_terms(forward)
            if not len(terms):
                raise ValueError("Need at least one crop for CUDA parity validation")
            workspace = self._workspace
            try:
                source = workspace.stage_frame(frame, device_frame)
                workspace.affine(source, width, height, terms)
                actual = workspace.copy_inputs(len(terms))
            except Exception:
                workspace.sync()
                raise
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lut = pose_lut()
            expected = np.stack([np.stack([lut[channel, crop[:, :, channel]] for channel in range(3)])
                                 for crop in (cv2.warpAffine(rgb, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
                                              for matrix in forward)])
            mismatch = np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32))
            return {"crops": len(terms), "valuesCompared": actual.size, "mismatchedValues": int(mismatch),
                    "maxAbsoluteDifference": float(np.max(np.abs(expected - actual))), "exact": not mismatch}

    def close(self):
        with self._lock:
            if not self.closed:
                self._workspace.close()
                self.closed = True
