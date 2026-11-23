"""Isolated RF-DETR input experiments with unchanged postprocessing."""

import numpy as np


def native_rows(prediction, person_ids):
    keep = np.isin(prediction.class_id, person_ids)
    boxes = np.asarray(prediction.xyxy[keep], dtype=np.float32)
    scores = np.asarray(prediction.confidence[keep], dtype=np.float32)
    if not len(scores):
        return np.empty((0, 6), dtype=np.float32)
    return np.column_stack((boxes, scores, np.zeros(len(scores), dtype=np.float32))).astype(np.float32)


def detect_with_device_frame(detector, frame):
    """Keep the same BGR bytes alive on device after synchronous result decode."""
    import torch
    from rfdetr._prediction.core import _uint8_chw_to_float, _uint8_scale

    from .dual_stage import DetectionOutput

    if frame.dtype != np.uint8 or not frame.flags.c_contiguous:
        raise ValueError("Need the original contiguous BGR bytes")
    with torch.inference_mode():
        device_frame = torch.from_numpy(frame).pin_memory().to("cuda:0", non_blocking=True)
        rgb = device_frame.flip(-1).permute(2, 0, 1)
        rgb = _uint8_chw_to_float(rgb, _uint8_scale(rgb.device, torch.get_default_dtype()))
        result = detector.model.predict(rgb, threshold=detector.confidence,
            shape=(detector.input_size, detector.input_size), antialias=detector.antialias, include_source_image=False)
        rows = native_rows(result, detector.person_ids)
        # Supervision outputs are CPU arrays and native result decode has
        # already waited for this stream. Explicitly confirm the handoff,
        # even if a later RF-DETR API optimizes the empty-result case.
        torch.cuda.current_stream(device_frame.device).synchronize()
    return DetectionOutput(rows, device_frame)
