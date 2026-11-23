"""Isolated two-stage inference with exact-frame FIFO publication.

The dispatcher owns detection. One worker owns pose and submits to the
previously tested CPU publication worker. No cameras or models are started here.
Prepared detector results can optionally carry the same frame's device buffer.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import math
import time

import numpy as np


from .cpu_publication import CpuPublicationPipeline, Publication


@dataclass(frozen=True)
class DetectionOutput:
    rows: np.ndarray
    device_frame: object = None


@dataclass(frozen=True)
class PoseOutput:
    boxes: list[dict]
    metrics: dict


def native_batch_detector(detector):
    """Opt-in RF-DETR eager microbatch with the original input/output rules.

    `detector` is the inner PersonDetector (`PersonPoseDetector.detector`).
    Batched forward kernels can differ numerically, so evaluate against native
    single-frame output before considering this candidate.
    """
    import cv2
    import torch

    def detect(frames):
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        with torch.inference_mode():
            predictions = detector.model.predict(rgb, threshold=detector.confidence,
                shape=(detector.input_size, detector.input_size), antialias=detector.antialias,
                include_source_image=False)
        if not isinstance(predictions, list) or len(predictions) != len(frames):
            raise ValueError("Native microbatch returned the wrong number of frame results")
        results = []
        for prediction in predictions:
            if not len(prediction):
                results.append(np.empty((0, 6), dtype=np.float32))
                continue
            keep = np.isin(prediction.class_id, detector.person_ids)
            boxes = np.asarray(prediction.xyxy[keep], dtype=np.float32)
            scores = np.asarray(prediction.confidence[keep], dtype=np.float32)
            results.append(np.column_stack((boxes, scores, np.zeros(len(boxes), np.float32))).astype(np.float32))
        return results

    return detect


class SplitPersonPose:
    """The production PersonPoseDetector processing split at its box boundary.

    `detect_call(frame)` may return native rows or a DetectionOutput that keeps
    an uploaded frame alive. An optional `pose_call(frame, boxes, device_frame)`
    can consume that buffer. `detect_batch_call`, if supplied, returns one row
    matrix or DetectionOutput per input frame in the same order.
    """

    def __init__(self, detector, *, detect_call=None, pose_call=None, detect_batch_call=None, pose_metadata=None):
        self._model = detector
        self._detect_call = detect_call if detect_call is not None else detector.detector.detect
        self._pose_call = pose_call
        self._detect_batch_call = detect_batch_call
        self._pose_metadata = detector.pose if pose_metadata is None else pose_metadata

    @property
    def supports_detector_batch(self):
        return self._detect_batch_call is not None

    @staticmethod
    def _validate_frame(frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or not frame.size or min(frame.shape[:2]) < 2):
            raise ValueError("frame must be nonempty uint8 BGR")

    def _filter_rows(self, frame, result):
        if not isinstance(result, DetectionOutput):
            result = DetectionOutput(result)
        height, width = frame.shape[:2]
        rows = np.asarray(result.rows, np.float32).reshape(-1, 6).copy()
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= self._model.confidence)]
        rows[:, (0, 2)] = np.clip(rows[:, (0, 2)], 0, width)
        rows[:, (1, 3)] = np.clip(rows[:, (1, 3)], 0, height)
        rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
        return DetectionOutput(rows, result.device_frame)

    def detect(self, frame):
        self._validate_frame(frame)
        return self._filter_rows(frame, self._detect_call(frame))

    def detect_batch(self, frames):
        if not self.supports_detector_batch:
            raise RuntimeError("No detector batch callable installed")
        for frame in frames:
            self._validate_frame(frame)
        results = tuple(self._detect_batch_call(frames))
        if len(results) != len(frames):
            raise ValueError("Batched detection must provide one result per input frame")
        return [self._filter_rows(frame, result) for frame, result in zip(frames, results)]

    def pose(self, frame, detection):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        rows = detection.rows
        boxes = rows[:, :4]
        estimator = self._pose_metadata
        if self._pose_call is None:
            points, scores = estimator.estimate(frame, boxes)
        else:
            points, scores = self._pose_call(frame, boxes, detection.device_frame)
        posed_at = time.perf_counter()
        quality = self._model.quality.apply(boxes, points, scores, frame.shape,
            ambiguity_mask=estimator.last_ambiguity_mask, crop_border_mask=estimator.last_crop_border_mask)
        states = quality["states"]
        output = []
        for i, row in enumerate(rows):
            record = dict(zip(("x1", "y1", "x2", "y2", "score"),
                              map(float, [row[0]/width, row[1]/height, row[2]/width, row[3]/height, row[4]])))
            record["keypoints"] = [{"x": float(points[i, j, 0]/width), "y": float(points[i, j, 1]/height),
                                    "score": float(scores[i, j]), "state": int(states[i, j])} for j in range(17)]
            output.append(record)
        ended = time.perf_counter()
        metrics = {"poseMs": round((posed_at-started)*1000, 2),
            "qualityMs": round((ended-posed_at)*1000, 2), "poseProcessedPeople": len(rows),
            "poseValidPeople": int(quality["valid_skeleton_count"]), "reliableKeypoints": int(quality["reliable_count"]),
            "estimatedKeypoints": int(quality["estimated_count"]), "rejectedKeypoints": int(quality["rejected_count"]),
            "poseStages": dict(estimator.last_timings), "trackingEnabled": False, "temporalSmoothing": False}
        return PoseOutput(output, metrics)


@dataclass(frozen=True)
class StageResult:
    output: PoseOutput | None
    publication: Future[Publication]
    detection_started_at: float
    detected_at: float
    pose_started_at: float
    pose_finished_at: float
    detector_batch_size: int


class DualStagePipeline:
    """Single dispatcher with bounded pose and CPU queues, both FIFO.

    After detection the usual call may hold one extra result while waiting
    for a pose slot. A microbatch may additionally hold its own small set of
    detections. Neither path waits to form a batch or queues arbitrary frames.
    """

    def __init__(self, stages, *, max_pose_inflight=2, max_cpu_inflight=2,
                 detection_context=None, pose_initializer=None, on_published=None, result_ttl=1.2, clock=time.perf_counter):
        if type(max_pose_inflight) is not int or max_pose_inflight < 1:
            raise ValueError("max_pose_inflight must be a positive integer")
        if not math.isfinite(result_ttl) or result_ttl <= 0:
            raise ValueError("result_ttl must be finite and positive")
        self._stages = stages
        self._pose_limit = max_pose_inflight
        self._result_ttl = result_ttl
        self._clock = clock
        if on_published is not None and not callable(on_published):
            raise TypeError("on_published must be a callable")
        self._on_published = on_published
        self._detection_context = detection_context if detection_context is not None else nullcontext
        self._publication = CpuPublicationPipeline(max_inflight=max_cpu_inflight, result_ttl=result_ttl, clock=clock)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose-stage", initializer=pose_initializer)
        self._pending: deque[Future[StageResult]] = deque()
        self._last_keys = {}
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._shutdown(cancel_pending=exc_type is not None)
        except Exception as failure:
            if exc is None:
                raise
            exc.add_note(f"Two-stage shutdown: {failure!r}")

    def _check_frame_keys(self, jobs):
        if self._closed:
            raise RuntimeError("Cannot submit to a closed two-stage pipeline")
        proposed = {}
        for reader, token in jobs:
            if reader.camera_id in proposed:
                raise ValueError("A detector batch can contain at most one frame per camera")
            key = token[1:3]
            if key <= self._last_keys.get(reader.camera_id, (-1, -1)):
                raise ValueError(f"Non-increasing submitted frame key for {reader.camera_id}")
            proposed[reader.camera_id] = key
        return proposed

    def submit(self, reader, token):
        proposed = self._check_frame_keys([(reader, token)])
        started = self._clock()
        with self._detection_context():
            detection = self._stages.detect(token[0])
        detected = self._clock()
        future = self._enqueue_pose(reader, token, detection, started, detected, 1)
        self._last_keys.update(proposed)
        return future

    def submit_many(self, jobs):
        """Submit a small caller-selected batch without delaying available work."""
        jobs = tuple(jobs)
        if not 1 <= len(jobs) <= self._pose_limit:
            raise ValueError("Detector batch must fit inside the bounded pose queue")
        proposed = self._check_frame_keys(jobs)
        if len(jobs) == 1 or not self._stages.supports_detector_batch:
            return [self.submit(reader, token) for reader, token in jobs]
        started = self._clock()
        with self._detection_context():
            detections = self._stages.detect_batch([token[0] for _, token in jobs])
        detected = self._clock()
        futures = []
        for (reader, token), detection in zip(jobs, detections):
            futures.append(self._enqueue_pose(reader, token, detection, started, detected, len(jobs)))
            self._last_keys[reader.camera_id] = proposed[reader.camera_id]
        return futures

    def _enqueue_pose(self, reader, token, detection, started, detected, batch_size):
        while self._pending and (self._pending[0].done() or len(self._pending) >= self._pose_limit):
            self._pending.popleft().result()
        future = self._executor.submit(self._run_pose, reader, token, detection, started, detected, batch_size)
        self._pending.append(future)
        return future

    def _run_pose(self, reader, token, detection, started, detected, batch_size):
        pose_started = self._clock()
        latest = reader.latest(pose_started)
        detect_ms = round((detected-started)*1000, 2)
        if (latest is None or reader.stop_event.is_set() or not 0 <= pose_started-token[3] <= self._result_ttl
                or latest[1] != token[1] or latest[5:7] != token[5:7]):
            skipped = Future()
            skipped.set_result(Publication(False, {"detectMs": detect_ms}, pose_started, pose_started,
                                           "expired_or_reconnected_before_pose"))
            return StageResult(None, skipped, started, detected, pose_started, pose_started, batch_size)
        output = self._stages.pose(token[0], detection)
        pose_finished = self._clock()
        output.metrics["detectMs"] = detect_ms
        publication = self._publication.submit(reader, token, output.boxes, output.metrics,
                                               inference_started_at=started)
        if self._on_published is not None:
            publication.add_done_callback(self._notify_accepted)
        return StageResult(output, publication, started, detected, pose_started, pose_finished, batch_size)

    def _notify_accepted(self, publication):
        if (not publication.cancelled() and publication.exception() is None
                and publication.result().accepted):
            self._on_published()

    def close(self):
        self._shutdown(cancel_pending=False)

    def _shutdown(self, *, cancel_pending):
        if self._closed:
            return
        self._closed = True
        pending, self._pending = self._pending, deque()
        try:
            self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
            if not cancel_pending:
                for future in pending:
                    future.result()
        finally:
            self._publication.close()


class FairFrameSelector:
    """Round-robin eligible frame heads. A slow camera never blocks the rest."""

    def __init__(self, readers, *, minimum_interval=0.0, result_ttl=1.2):
        self._readers = tuple(readers)
        if len({reader.camera_id for reader in self._readers}) != len(self._readers):
            raise ValueError("Camera IDs must be unique")
        if not math.isfinite(minimum_interval) or minimum_interval < 0:
            raise ValueError("minimum_interval must be finite and nonnegative")
        if not math.isfinite(result_ttl) or result_ttl <= 0:
            raise ValueError("result_ttl must be finite and positive")
        self._minimum_interval = minimum_interval
        self._result_ttl = result_ttl
        self._cursor = 0
        self._previous = {}

    def take(self, now, *, limit=1):
        """Take at most the requested ready heads, no more than one per camera."""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        selected = []
        size = len(self._readers)
        cursor = self._cursor
        for offset in range(size):
            index = (cursor + offset) % size
            reader = self._readers[index]
            if reader.stop_event.is_set():
                continue
            token = reader.latest(now)
            if token is None or not 0 <= now-token[3] <= self._result_ttl:
                continue
            key, previous_time = self._previous.get(reader.camera_id, ((-1, -1), -float("inf")))
            if token[1:3] <= key or (token[1] == key[0] and now - previous_time < self._minimum_interval):
                continue
            self._previous[reader.camera_id] = (token[1:3], now)
            self._cursor = (index + 1) % size
            selected.append((reader, token))
            if len(selected) >= limit:
                break
        return selected
