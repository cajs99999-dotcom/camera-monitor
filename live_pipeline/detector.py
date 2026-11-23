"""Production R3 detector graph and GPU crop on the staged runtime."""
from __future__ import annotations

import numpy as np

from ai.person_pose import PersonPoseDetector
from .cached_native_graph import CachedNativeDetectorGraph
from .detector_inputs import detect_with_device_frame
from .dual_stage import SplitPersonPose
from .gpu_pose_estimator import GpuCropPoseEstimator

POSE_BATCH = 16


class GpuPipelineDetector(PersonPoseDetector):
    def __init__(self, confidence):
        self._crop = self._graph = None
        super().__init__(confidence=confidence, threads=2, warmup=False)
        try:
            if (self.pose.backend_selected != "tensorrt" or self.pose.precision != "fp16"
                    or self.detector.profile != "classroom_v1" or self.confidence != .45):
                raise RuntimeError("Expected the tested classroom-v1 / TensorRT FP16 models")
            self.pose.batch_size = self.pose.pose_batch_size = POSE_BATCH
            self.pose.backend = f"tensorrt / fp16 / RGB / batch{POSE_BATCH}"
            warm = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.detector.detect(warm)
            self._graph = CachedNativeDetectorGraph(self.detector)
            self._graph.attach()
            self._crop = GpuCropPoseEstimator(self.pose)
            self.stage = SplitPersonPose(self, detect_call=self._detect_and_keep_frame,
                pose_call=self._crop.estimate, pose_metadata=self._crop)
            for count in (16, 14, 1):
                self._crop.estimate(warm, np.tile([200, 100, 500, 650], (count, 1)))
            self._detect_and_keep_frame(warm)
        except Exception:
            self.close()
            raise

    def _detect_and_keep_frame(self, frame):
        return detect_with_device_frame(self.detector, frame)

    def make_detection_context(self):
        import torch

        torch.cuda.set_device(0)
        stream = torch.cuda.Stream(device=0)
        torch.cuda.synchronize(device=0)
        return lambda: torch.cuda.stream(stream)

    def pose_worker_init(self):
        self.pose._session.runner.cuda.call("cudaSetDevice", 0)

    def close(self):
        try:
            if self._crop is not None:
                self._crop.close()
                self._crop = None
        finally:
            try:
                if self._graph is not None:
                    self._graph.detach()
                    self._graph = None
            finally:
                super().close()
