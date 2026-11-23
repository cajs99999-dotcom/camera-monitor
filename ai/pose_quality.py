"""Single-frame quality states for COCO's 17 pose joints, in source pixels.

This module preserves the estimator's coordinates; it does not track people,
smooth poses, or infer visibility from a confidence score. States are 0 rejected,
1 estimated, and 2 reliable under the configured heuristics. A rejected point
must not be drawn again using its raw score. Draw a segment only when BOTH
endpoints have a nonzero state.

``reasons`` is an N x 17 list of strings, aligned with ``states``. Multiple
reasons are joined with ``|``. ``valid_skeleton_count`` counts rows with both
shoulders reliable and at least five reliable joints; it is a quality summary,
not the number of accurate, visible, or complete full-body skeletons.
"""
from __future__ import annotations

import numpy as np


REJECTED, ESTIMATED, RELIABLE = 0, 1, 2
_CHAINS = ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16))


class PoseQualityFilter:
    """Apply conservative, stateless display checks without changing raw data.

    Boxes are Nx4 xyxy source-image pixels, points Nx17x2, scores Nx17.
    Optional masks are Nx17 booleans. ``crop_border_mask`` should come from
    the estimator's original SimCC/crop coordinates, not detection-box edges.
    Such points are hard rejected; ambiguity alone makes a point estimated.
    Thresholds are score cutoffs, not calibrated visibility probabilities.
    """

    def __init__(self, threshold=.60, weak_threshold=.10):
        threshold, weak_threshold = float(threshold), float(weak_threshold)
        if not (np.isfinite(threshold) and np.isfinite(weak_threshold)
                and 0 <= weak_threshold <= threshold <= 1):
            raise ValueError("Require 0 <= weak_threshold <= threshold <= 1")
        self.threshold = threshold
        self.weak_threshold = weak_threshold

    @staticmethod
    def _mask(value, shape, name):
        if value is None:
            return np.zeros(shape, dtype=bool)
        result = np.asarray(value)
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
        if result.dtype != np.bool_ and not np.isin(result, (0, 1)).all():
            raise ValueError(f"{name} must contain booleans or 0/1 values")
        return result.astype(bool, copy=True)

    def apply(self, boxes, points, scores, frame_shape,
              ambiguity_mask=None, crop_border_mask=None):
        boxes = np.asarray(boxes, dtype=np.float32)
        copied = np.array(points, dtype=np.float32, copy=True)
        raw = np.asarray(scores, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1:] != (4,):
            raise ValueError("boxes must have shape Nx4")
        n = len(boxes)
        if copied.shape != (n, 17, 2) or raw.shape != (n, 17):
            raise ValueError("points/scores must have shapes Nx17x2 and Nx17")
        if (not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("boxes must be finite, positive-area xyxy boxes")
        try:
            height, width = frame_shape[:2]
            valid_shape = (np.isfinite(height) and np.isfinite(width)
                           and int(height) == height and int(width) == width
                           and height > 0 and width > 0)
        except (TypeError, ValueError, OverflowError):
            valid_shape = False
        if not valid_shape:
            raise ValueError("frame_shape must start with positive integer height, width")
        ambiguity = self._mask(ambiguity_mask, raw.shape, "ambiguity_mask")
        border = self._mask(crop_border_mask, raw.shape, "crop_border_mask")
        finite_point = np.isfinite(copied).all(axis=2)
        finite_score = np.isfinite(raw)
        in_frame = (finite_point & (copied[:, :, 0] >= 0)
                    & (copied[:, :, 0] < width) & (copied[:, :, 1] >= 0)
                    & (copied[:, :, 1] < height))
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        # Modest tolerance accepts detector-box jitter. It cannot establish
        # occlusion/ownership; no aspect-ratio rule suppresses hips or legs.
        left = boxes[:, 0] - .15 * widths
        right = boxes[:, 2] + .15 * widths
        top = boxes[:, 1] - .15 * heights
        bottom = boxes[:, 3] + .10 * heights
        in_box = (finite_point & (copied[:, :, 0] >= left[:, None])
                  & (copied[:, :, 0] <= right[:, None])
                  & (copied[:, :, 1] >= top[:, None])
                  & (copied[:, :, 1] <= bottom[:, None]))

        causes = [[[] for _ in range(17)] for _ in range(n)]
        hard = np.zeros(raw.shape, dtype=bool)

        def reject(mask, reason):
            hard[mask] = True
            for i, joint in np.argwhere(mask):
                causes[i][joint].append(reason)

        reject(~finite_point, "nonfinite_point")
        reject(~finite_score, "nonfinite_score")
        reject(finite_score & (raw < self.weak_threshold), "below_weak_threshold")
        reject(finite_point & ~in_frame, "out_of_frame")
        reject(finite_point & ~in_box, "outside_box")
        reject(border, "crop_border")

        # Look only for a compact, strong head cluster that is clearly outside
        # its own permitted box and in another box's upper head region. Merely
        # overlapping two person boxes is not evidence of a wrong assignment.
        ownership = np.zeros(raw.shape, dtype=bool)
        for i in range(n):
            strong_head = (in_frame[i, :5] & finite_score[i, :5]
                           & (raw[i, :5] >= self.threshold) & ~border[i, :5])
            if np.count_nonzero(strong_head) < 3:
                continue
            head = copied[i, :5][strong_head]
            center = np.median(head, axis=0)
            if left[i] <= center[0] <= right[i] and top[i] <= center[1] <= bottom[i]:
                continue
            if np.max(np.linalg.norm(head - center, axis=1)) > max(8., .35 * min(widths[i], heights[i])):
                continue
            for other in range(n):
                if other == i:
                    continue
                b = boxes[other]
                in_other_head = ((head[:, 0] >= b[0] - .05 * widths[other])
                                 & (head[:, 0] <= b[2] + .05 * widths[other])
                                 & (head[:, 1] >= b[1] - .10 * heights[other])
                                 & (head[:, 1] <= b[1] + .55 * heights[other]))
                if np.count_nonzero(in_other_head) >= 3:
                    ownership[i, :5] = True
                    break
        reject(ownership, "head_ownership_conflict")

        # Evaluate segment lengths against a frozen set of spatially plausible
        # endpoints. Rejecting an elbow must not skip the elbow-wrist check.
        # A newly rejected implausible proximal segment also rejects that
        # chain's distal joint rather than leaving a floating confident wrist.
        candidate = ~hard
        long_limb = np.zeros(raw.shape, dtype=bool)
        broken_chain = np.zeros(raw.shape, dtype=bool)
        limits = .65 * np.hypot(widths, heights)
        for proximal, middle, distal in _CHAINS:
            for a, b in ((proximal, middle), (middle, distal)):
                both = candidate[:, a] & candidate[:, b]
                rows = np.flatnonzero(both)
                distances = np.linalg.norm(copied[rows, a] - copied[rows, b], axis=1)
                bad_rows = rows[distances > limits[rows]]
                long_limb[bad_rows, b] = True
            broken_chain[:, distal] |= long_limb[:, middle] & candidate[:, distal]
        reject(long_limb, "implausible_limb")
        reject(broken_chain, "broken_chain")

        states = np.where(raw >= self.threshold, RELIABLE, ESTIMATED).astype(np.uint8)
        states[ambiguity] = ESTIMATED
        states[hard] = REJECTED
        reasons = []
        for i in range(n):
            row = []
            for joint in range(17):
                if states[i, joint] == ESTIMATED:
                    if raw[i, joint] < self.threshold:
                        causes[i][joint].append("low_confidence")
                    if ambiguity[i, joint]:
                        causes[i][joint].append("ambiguous")
                row.append("|".join(causes[i][joint]) or "reliable")
            reasons.append(row)
        reliable_per_person = np.sum(states == RELIABLE, axis=1)
        valid_skeletons = ((states[:, 5] == RELIABLE) & (states[:, 6] == RELIABLE)
                           & (reliable_per_person >= 5))
        return {
            "points": copied,
            "states": states,
            "reasons": reasons,
            "reliable_count": int(np.count_nonzero(states == RELIABLE)),
            "estimated_count": int(np.count_nonzero(states == ESTIMATED)),
            "rejected_count": int(np.count_nonzero(states == REJECTED)),
            "valid_skeleton_count": int(np.count_nonzero(valid_skeletons)),
        }
