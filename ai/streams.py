"""从 MediaMTX 直接取六路视频，并在后台持续做人体检测。

每路一个采集线程：持续读取、只保留最新一帧。
六路共用一个推理线程：轮流检测最新帧，不依赖是否有人打开网页。
网页只读取 snapshot() 的坐标和人数，不再上传取样图片。
"""

from __future__ import annotations

import os
import copy
from collections import deque
import threading
import time
from typing import Any
from uuid import uuid4

# 只影响本 Python 进程：读取本机 MediaMTX 的 RTSP 时使用 TCP。
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2


CAMERA_IDS = tuple(f"cam{i}" for i in range(1, 7))
DEFAULT_URLS = {camera_id: f"rtsp://127.0.0.1:8554/{camera_id}" for camera_id in CAMERA_IDS}
RESULT_TTL = 1.2
CAPTURE_TIMEOUT_MS = 3000
RECONNECT_SECONDS = 2.0
MAX_INFERENCE_WIDTH = 1920
MAX_INFERENCE_HEIGHT = 1080


class _CameraReader:
    """采集线程拥有自己的 VideoCapture，其他线程只读取最新帧引用。"""

    def __init__(self, camera_id: str, url: str, stop_event: threading.Event):
        self.camera_id, self.url, self.stop_event = camera_id, url, stop_event
        self.lock = threading.Lock()
        self.generation = 0
        self.sequence = 0
        self.frames_read = 0
        self.detections = 0
        self.width = self.height = None
        self.frame = None
        self.received_at = None
        self.received_wall = None
        self.result = None
        self.result_times = deque(maxlen=20)
        self.receive_times = deque(maxlen=60)
        self.status = "connecting"
        self.inference_error = False
        self.last_warning_at = -float("inf")
        self.thread = threading.Thread(target=self._read_loop, name=f"capture-{camera_id}", daemon=True)

    def _invalidate(self, status: str) -> None:
        with self.lock:
            # 先失效代次，推理线程即使稍后完成，也不能写回旧连接的结果。
            self.generation += 1
            self.frame = self.result = None
            self.result_times.clear()
            self.receive_times.clear()
            self.received_at = self.received_wall = None
            self.width = self.height = None
            self.inference_error = False
            self.status = status

    def _warn(self, detail: str) -> None:
        now = time.perf_counter()
        if now - self.last_warning_at >= 30:
            self.last_warning_at = now
            # 只输出通道编号和固定提示，不输出可能含凭据的 URL 或异常详情。
            print(f"[AI][{self.camera_id}] {detail}", flush=True)

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            self._invalidate("reconnecting" if self.frames_read else "connecting")
            capture = None
            try:
                capture = cv2.VideoCapture()
                # 这些参数要在 open() 时传入。各路独立采集，单路超时不阻塞其他通道。
                params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS,
                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAPTURE_TIMEOUT_MS]
                if hasattr(cv2, "CAP_PROP_N_THREADS"):
                    params += [cv2.CAP_PROP_N_THREADS, 1]
                opened = capture.open(self.url, cv2.CAP_FFMPEG, params)
                if not opened:
                    self._invalidate("reconnecting")
                    self._warn("暂未取到视频，后台会自动重连")
                else:
                    with self.lock:
                        self.status = "waiting"
                    while not self.stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None or not getattr(frame, "size", 0):
                            self._invalidate("reconnecting")
                            self._warn("视频读取中断，后台会自动重连")
                            break
                        received_at, received_wall = time.perf_counter(), time.time() * 1000
                        height, width = frame.shape[:2]
                        # 让 4K 等视频适配姿态 GPU 工作区，并保持视野和宽高比。
                        scale = min(MAX_INFERENCE_WIDTH / width, MAX_INFERENCE_HEIGHT / height)
                        if scale < 1:
                            width, height = max(2, round(width * scale)), max(2, round(height * scale))
                            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                        with self.lock:
                            # 覆盖单槽，不排队。推理中的旧帧由其局部引用持有。
                            self.frame = frame
                            self.sequence += 1
                            self.frames_read += 1
                            self.width, self.height = int(width), int(height)
                            self.received_at, self.received_wall = received_at, received_wall
                            self.receive_times.append(received_at)
                            self.status = "waiting"
            except Exception:
                self._invalidate("reconnecting")
                self._warn("视频连接暂不可用，后台会自动重连")
            finally:
                # release 与 read 在同一个采集线程执行，避免跨线程释放竞争。
                if capture is not None:
                    capture.release()
            self.stop_event.wait(RECONNECT_SECONDS)
        self._invalidate("waiting")

    def latest(self, now: float) -> tuple | None:
        with self.lock:
            if self.frame is None or self.received_at is None or now - self.received_at > RESULT_TTL:
                return None
            return (self.frame, self.generation, self.sequence, self.received_at,
                    self.received_wall, self.width, self.height)

    def publish(self, token: tuple, boxes: list[dict], inference_ms: float, metrics: dict | None = None) -> bool:
        _frame, generation, sequence, captured_at, captured_wall, width, height = token
        now = time.perf_counter()
        with self.lock:
            if (self.stop_event.is_set() or self.frame is None or generation != self.generation
                    or now - captured_at > RESULT_TTL or (width, height) != (self.width, self.height)):
                return False
            self.result_times.append(now)
            processing_fps=((len(self.result_times)-1)/(self.result_times[-1]-self.result_times[0])
                            if len(self.result_times)>1 and self.result_times[-1]>self.result_times[0] else 0.)
            self.result = {
                "generation": generation, "sequence": sequence, "capturedMono": captured_at,
                "capturedAt": captured_wall, "processedAt": time.time() * 1000,
                "width": width, "height": height, "inferenceMs": inference_ms,
                "boxes": copy.deepcopy(boxes),
                "metrics": copy.deepcopy(metrics or {}), "processingFps": round(processing_fps,2),
                "backendAgeMs": round((now-captured_at)*1000,2),
            }
            self.inference_error = False
            self.detections += 1
            return True

    def mark_inference_error(self, generation: int) -> None:
        with self.lock:
            if generation == self.generation:
                self.result = None
                self.inference_error = True

    def snapshot(self, now: float) -> dict[str, Any]:
        with self.lock:
            result = self.result
            age_ms = max(0.0, (now - result["capturedMono"]) * 1000) if result else None
            fresh_frame = self.frame is not None and self.received_at is not None and now - self.received_at <= RESULT_TTL
            valid = (fresh_frame and result is not None and result["generation"] == self.generation
                     and age_ms <= RESULT_TTL * 1000 and
                     (result["width"], result["height"]) == (self.width, self.height))
            status = "live" if valid else ("error" if self.inference_error and fresh_frame else self.status)
            boxes = copy.deepcopy(result["boxes"]) if valid else []
            metrics = copy.deepcopy(result.get("metrics", {})) if valid else {}
            capture_fps=((len(self.receive_times)-1)/(self.receive_times[-1]-self.receive_times[0])
                         if fresh_frame and len(self.receive_times)>1 and self.receive_times[-1]>self.receive_times[0] else 0.)
            return {
                "cameraId": self.camera_id, "status": status,
                "generation": self.generation,
                "sequence": result["sequence"] if result else self.sequence,
                "width": self.width, "height": self.height,
                # age 由原始取帧时刻计算；重复推送 snapshot 不会给旧结果续期。
                "ageMs": round(age_ms, 1) if age_ms is not None else None,
                "count": len(boxes) if valid else None, "boxes": boxes,
                "inferenceMs": result["inferenceMs"] if valid else None,
                "capturedAt": result["capturedAt"] if result else None,
                "capturedMono": result["capturedMono"] if result else None,
                "processedAt": result["processedAt"] if result else None,
                "framesRead": self.frames_read, "detections": self.detections,
                "processingFps": result.get("processingFps",0.) if valid else 0.,
                "captureFps": round(capture_fps,2), "resultTtlMs": int(RESULT_TTL*1000),
                "backendAgeMs": result.get("backendAgeMs") if valid else None,
                "timingScope": "backend decoded frame; asynchronous WebRTC display",
                **metrics,
            }


class MonitoringEngine:
    """Own capture threads and lifecycle for the shared inference dispatcher."""

    def __init__(self, detector, urls=None):
        sources = dict(DEFAULT_URLS if urls is None else urls)
        if set(sources) != set(CAMERA_IDS) or any(not isinstance(url, str) or not url for url in sources.values()):
            raise ValueError(f"urls 必须提供以下机位的视频地址：{', '.join(CAMERA_IDS)}")
        self.detector = detector
        self.clock_session_id = str(uuid4())
        self.stop_event = threading.Event()
        self.cameras = [_CameraReader(camera_id, sources[camera_id], self.stop_event) for camera_id in CAMERA_IDS]
        self.lock = threading.Lock()
        self.processed_frames = 0
        self.started = False
        self.model_closed = False
        self.inference_thread = threading.Thread(target=self._infer_loop, name="person-inference", daemon=True)

    def start(self):
        if self.started:
            return
        if self.stop_event.is_set():
            raise RuntimeError("已停止的 MonitoringEngine 不能重启，请新建实例")
        self.started = True
        for camera in self.cameras:
            camera.thread.start()
        self.inference_thread.start()

    def _infer_loop(self):
        raise NotImplementedError("The inference dispatcher must implement _infer_loop")

    def snapshot(self):
        with self.lock:
            processed = self.processed_frames
        now = time.perf_counter()
        records = [{**camera.snapshot(now), "clockSessionId": self.clock_session_id} for camera in self.cameras]
        return {"processedFrames": processed, "running": self.started and not self.stop_event.is_set(),
                "resultTtlMs": int(RESULT_TTL * 1000), "clockSessionId": self.clock_session_id,
                "cameras": records, "poseEnabled": bool(getattr(self.detector, "supports_pose", False)),
                "trackingEnabled": False}

    def stop(self) -> bool:
        """通知各线程退出；共同使用5秒总期限，不逐路叠加超时。"""
        self.stop_event.set()
        if not self.started:
            return self._close_model()
        deadline = time.perf_counter() + 5.0
        threads = [camera.thread for camera in self.cameras] + [self.inference_thread]
        for thread in threads:
            if thread is not threading.current_thread() and thread.ident is not None:
                thread.join(max(0.0, deadline - time.perf_counter()))
        for camera in self.cameras:
            camera._invalidate("waiting")
        stopped = not any(thread.is_alive() for thread in threads)
        return self._close_model() if stopped else False

    def _close_model(self) -> bool:
        """Release native inference resources only after its worker is finished."""
        if self.model_closed:
            return True
        closer = getattr(self.detector, "close", None)
        try:
            if callable(closer):
                closer()
            self.model_closed = True
            return True
        except Exception:
            print("[AI] 推理资源释放未完成，进程退出时会回收。", flush=True)
            return False
