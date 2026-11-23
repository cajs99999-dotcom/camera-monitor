"""CPU geometry and reference for an exact OpenCV uint8 affine CUDA crop.

OpenCV 4.11.0 uses a 10-bit fixed-point coordinate with separate roundings
for the horizontal delta and row origin, then a 5-bit interpolation table.
The CUDA crop receives those integer terms and the exact FP32 RGB lookup
table. This file also supplies a NumPy parity oracle and an explicit gate
against the installed, unmodified OpenCV path.
"""
from __future__ import annotations

import cv2
import numpy as np

WIDTH = 288
HEIGHT = 384
X_AXIS = np.arange(WIDTH, dtype=np.float64)
Y_AXIS = np.arange(HEIGHT, dtype=np.float64)
FP_SCALE = 1024
FRACTION_COUNT = 32
ROW_ROUND_DELTA = 16
MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)
STD = np.array((58.395, 57.12, 57.375), dtype=np.float32)


def pose_lut() -> np.ndarray:
    """Return `(3, 256)` exact FP32 normalization values in RGB order."""
    pixels = np.arange(256, dtype=np.uint8)[None, :]
    result = np.empty((3, 256), dtype=np.float32)
    np.subtract(pixels, MEAN[:, None], out=result)
    np.divide(result, STD[:, None], out=result)
    return result


def crop_geometry(boxes):
    """Return original centers, adjusted scales and forward affine matrices."""
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, get_warp_matrix

    values = np.asarray(boxes, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all()
            or np.any(values[:, 2:] <= values[:, :2])):
        raise ValueError("boxes must be finite Nx4 XYXY coordinates with positive area")
    centers, scales = bbox_xyxy2cs(values, padding=1.25)
    width, height = scales[:, 0], scales[:, 1]
    aspect_ratio = WIDTH / HEIGHT
    wide = width > height * aspect_ratio
    adjusted = np.column_stack((np.where(wide, width, height * aspect_ratio),
                                np.where(wide, width / aspect_ratio, height)))
    forward = np.empty((len(values), 2, 3), dtype=np.float64)
    for index, (center, scale) in enumerate(zip(centers, adjusted)):
        forward[index] = get_warp_matrix(center, scale, 0, (WIDTH, HEIGHT))
    return centers, adjusted, forward


def coordinate_terms(forward):
    """Return one flat pair of horizontal/vertical coordinate terms per crop."""
    matrices = np.asarray(forward, dtype=np.float64)
    if (matrices.ndim != 3 or matrices.shape[1:] != (2, 3)
            or not np.isfinite(matrices).all() or len(matrices) > 16):
        raise ValueError("expected up to sixteen finite forward affine matrices")
    inverse = np.empty_like(matrices)
    for index, matrix in enumerate(matrices):
        if np.linalg.det(matrix[:, :2]) == 0:
            raise ValueError("degenerate affine crop")
        inverse[index] = cv2.invertAffineTransform(matrix)
    delta_x = np.rint((inverse[:, :, 0, None] * X_AXIS) * FP_SCALE)
    origin_y = np.rint((inverse[:, :, 1, None] * Y_AXIS + inverse[:, :, 2, None]) * FP_SCALE)
    # Input should be normal camera positions. Bounded integers also keep
    # addition and signed floor divisions below the signed 32-bit limits.
    if max(np.abs(delta_x).max(initial=0), np.abs(origin_y).max(initial=0)) >= 2**27:
        raise ValueError("affine crop outside supported fixed-point coordinate range")
    return np.ascontiguousarray(np.concatenate((delta_x, origin_y + ROW_ROUND_DELTA), axis=2),
                                dtype=np.int32)


def reference_warp(bgr, terms):
    """NumPy port of the CUDA integer warp for one crop, returned as RGB uint8."""
    fixed = (terms[:, :WIDTH][:, None, :] + terms[:, WIDTH:][:, :, None]) >> 5
    x, y = np.clip(fixed >> 5, -32768, 32767)
    frac_x, frac_y = fixed & (FRACTION_COUNT - 1)
    width, height = bgr.shape[1], bgr.shape[0]
    acc = np.zeros((HEIGHT, WIDTH, 3), dtype=np.int32)
    for i, wx in enumerate((FRACTION_COUNT - frac_x, frac_x)):
        for j, wy in enumerate((FRACTION_COUNT - frac_y, frac_y)):
            xx, yy = x + i, y + j
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            pixels = bgr[np.clip(yy, 0, height-1), np.clip(xx, 0, width-1)]
            acc += pixels * (valid * wx * wy)[:, :, None]
    return ((acc + 512) >> 10).astype(np.uint8)[:, :, ::-1]


def check_opencv_reference(bgr, forward, terms):
    """Require exact parity without changing process-wide OpenCV IPP policy."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mismatch_count = different_crops = 0
    for matrix, coordinates in zip(forward, terms):
        reference = cv2.warpAffine(rgb, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        warped = reference_warp(bgr, coordinates)
        changed = np.count_nonzero(reference != warped)
        different_crops += int(changed > 0)
        mismatch_count += int(changed)
    return different_crops, mismatch_count
