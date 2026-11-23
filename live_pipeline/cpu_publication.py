"""Bounded FIFO publication of anonymous boxes and pose results."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class Publication:
    accepted: bool
    metrics: dict[str, Any]
    published_at: float
    started_at: float
    skipped_reason: str | None = None


class CpuPublicationPipeline:
    """Use from one inference dispatcher. A full queue blocks that dispatcher.

    At most `max_inflight` futures retain queued frame references, including the
    running job. Neither a slow CPU nor a caller collecting results later can
    cause this worker to queue an arbitrary number of capture frames.
    """

    def __init__(self, *, max_inflight: int = 2,
                 result_ttl: float = 1.2, clock=time.perf_counter):
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        if not 0 < result_ttl < float("inf"):
            raise ValueError("result_ttl must be finite and positive")
        self._limit = max_inflight
        self._result_ttl = result_ttl
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-publication")
        self._inflight: deque[Future[Publication]] = deque()
        self._closed = False

    def __enter__(self) -> CpuPublicationPipeline:
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self._shutdown(cancel_pending=True)

    def submit(self, reader, token, boxes, metrics, *, inference_started_at=None) -> Future[Publication]:
        if self._closed:
            raise RuntimeError("Cannot submit to a closed CPU publication pipeline")
        while self._inflight and (self._inflight[0].done() or len(self._inflight) >= self._limit):
            self._inflight.popleft().result()
        if inference_started_at is None:
            inference_started_at = token[3]
        future = self._executor.submit(self._publish, reader, token, deepcopy(boxes), deepcopy(metrics),
                                       inference_started_at)
        self._inflight.append(future)
        return future

    def close(self) -> None:
        self._shutdown(cancel_pending=False)

    def _shutdown(self, *, cancel_pending: bool) -> None:
        if self._closed:
            return
        self._closed = True
        pending, self._inflight = self._inflight, deque()
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
        if not cancel_pending:
            for future in pending:
                future.result()

    def _current_epoch(self, reader, token, now: float) -> bool:
        if reader.stop_event.is_set() or not 0 <= now - token[3] <= self._result_ttl:
            return False
        latest = reader.latest(now)
        return latest is not None and latest[1] == token[1] and latest[5:7] == token[5:7]

    def _publish(self, reader, token, boxes, metrics, inference_started_at: float) -> Publication:
        started_at = self._clock()
        if not self._current_epoch(reader, token, started_at):
            return Publication(False, metrics, started_at, started_at, "expired_or_reconnected")
        now = self._clock()
        accepted = self._current_epoch(reader, token, now) and reader.publish(
            token, boxes, (now - inference_started_at) * 1000, metrics)
        return Publication(accepted, metrics, self._clock(), started_at, None if accepted else "publication_rejected")
