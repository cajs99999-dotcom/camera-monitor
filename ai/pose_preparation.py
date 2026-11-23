"""Same RTMPose-L affine/float32 normalization with reusable HWC scratch.

Caller supplies RGB uint8 pixels, then immediately copies the returned HWC image
to its own NCHW tensor. The image is a borrowed buffer: another preprocess call
on the same instance AND thread overwrites it. Center and scale are independent
arrays and do not share that lifetime. There is no global OpenCV thread setting.
"""
from __future__ import annotations

import threading

import numpy as np


class RTMPosePreparation:
    """A drop-in ``preprocess(rgb, bbox)`` for the current 288x384 RTMPose-L.

    Own one instance per estimator. Each calling thread owns its own scratch,
    including a construction/warmup thread distinct from the inference worker.
    Do not retain the HWC return as a frame result; use e.g.
    ``inputs[i] = image.transpose(2, 0, 1)`` before calling preprocess again.
    Returned NCHW copies made by the caller are never mutated by this helper.
    """

    def __init__(self):
        self.model_input_size = (288, 384)
        self.mean = np.array((123.675, 116.28, 103.53), dtype=np.float32)
        self.std = np.array((58.395, 57.12, 57.375), dtype=np.float32)
        self.mean.setflags(write=False)
        self.std.setflags(write=False)
        self._local = threading.local()

    def preprocess(self, img: np.ndarray, bbox):
        """Return borrowed contiguous float32 HWC, float64 center/scale.

        Bounding boxes are finite positive-area source-pixel xyxy. The enclosing
        estimator retains its existing clipping/invalid-box policy. This method
        intentionally does not clip again or change aspect/padding/interpolation.
        """
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

        if (not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3
                or img.shape[2] != 3 or min(img.shape[:2]) < 2):
            raise ValueError("img must be a nonempty RGB uint8 HxWx3 image")
        try:
            box = np.asarray(bbox, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox must be four finite source-pixel xyxy coordinates") from exc
        if (box.shape != (4,) or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])):
            raise ValueError("bbox must be four finite positive-area xyxy coordinates")
        if self.model_input_size != (288, 384):
            raise ValueError("This helper is fixed to the existing 288x384 RTMPose-L input")
        center, scale = bbox_xyxy2cs(box, padding=1.25)
        image, scale = top_down_affine(self.model_input_size, scale, center, img)
        scratch = getattr(self._local, "scratch", None)
        if scratch is None:
            scratch = np.empty((384, 288, 3), dtype=np.float32)
            self._local.scratch = scratch
        # Preserve the official two distinct float32 ufunc operations. Using
        # out= avoids two new HWC float32 allocations per person without fusing
        # arithmetic, changing rounding, or changing the warp's pixel values.
        np.subtract(image, self.mean, out=scratch)
        np.divide(scratch, self.std, out=scratch)
        return scratch, center, scale
