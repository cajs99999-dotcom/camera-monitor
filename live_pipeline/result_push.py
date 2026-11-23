"""Bounded, thread-safe wakeups for the existing detection WebSocket protocol."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import math
import threading
import time


class LatestResultPush:
    """Wake each subscriber on publication and take a fresh snapshot at send time.

    Bind to the serving event loop before starting the inference workers. Call
    notify_published() after a result has been committed to the engine. A burst
    schedules at most one pending loop callback and never queues result data.
    """

    def __init__(self, engine, session_id, *, min_interval=.10, heartbeat=.25, send_timeout=1.0):
        if (not all(math.isfinite(value) and value > 0
                    for value in (min_interval, heartbeat, send_timeout))
                or heartbeat < min_interval):
            raise ValueError("Require 0 < min_interval <= heartbeat and a positive send timeout")
        self._engine = engine
        self._session_id = session_id
        self._min_interval = min_interval
        self._heartbeat = heartbeat
        self._send_timeout = send_timeout
        self._lock = threading.Lock()
        self._loop = None
        self._change = None
        self._revision = 0
        self._scheduled = self._closed = False

    def start(self):
        """Bind to the serving event loop on application startup."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed or self._loop not in (None, loop):
                raise RuntimeError("Publication wakeup cannot be rebound")
            if self._loop is None:
                self._loop = loop
                self._change = asyncio.Event()

    def notify_published(self):
        """Schedule at most one pending wakeup from any publisher thread."""
        with self._lock:
            if self._closed or self._scheduled or self._loop is None:
                return
            self._scheduled = True
            try:
                self._loop.call_soon_threadsafe(self._advance)
            except RuntimeError:
                # A worker can finish after the serving loop has closed.
                self._scheduled = False

    def _advance(self):
        with self._lock:
            self._scheduled = False
            if self._closed:
                return
        previous = self._change
        self._change = asyncio.Event()
        self._revision += 1
        previous.set()

    def close(self):
        """Wake subscribers during application shutdown on the serving loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._loop not in (None, loop):
                raise RuntimeError("Close publication wakeup on its serving loop")
            self._closed = True
        if self._change is not None:
            self._change.set()

    async def push_latest(self, ws):
        """Use after the ordinary ready message has been sent to this client."""
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            raise RuntimeError("Start publication wakeup before serving subscribers")
        try:
            while not ws.closed and not self._closed:
                revision = self._revision
                message = {"type": "detections", "sessionId": self._session_id,
                           **self._engine.snapshot(), "serverTime": time.time() * 1000}
                await asyncio.wait_for(ws.send_json(message), timeout=self._send_timeout)
                if ws.closed or self._closed:
                    break
                sent = time.perf_counter()

                # The idle deadline emits another fresh snapshot even when
                # there are no results, so disconnects and TTL expiry propagate.
                if self._revision == revision:
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._change.wait(), timeout=self._heartbeat)
                # Some Windows event-loop timers fire early. Recheck the real
                # deadline so subscriber output still has a strict rate cap.
                while not self._closed:
                    until_allowed = sent + self._min_interval - time.perf_counter()
                    if until_allowed <= 0:
                        break
                    await asyncio.sleep(until_allowed)
        except (ConnectionError, OSError, RuntimeError, asyncio.TimeoutError):
            if not ws.closed:
                with suppress(ConnectionError, OSError, RuntimeError):
                    await ws.close(code=1001, message=b"subscriber unavailable")
