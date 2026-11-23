"""Camera preview and real-time detection results."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

from aiohttp import WSMsgType, web

from ai.detection_settings import load_detection_settings
from ai.person_pose import MODEL_NAME
from live_pipeline.detector import GpuPipelineDetector
from live_pipeline.engine import GpuPipelineEngine
from live_pipeline.result_push import LatestResultPush


PROJECT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT / "web"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
MONITOR_HOSTS = frozenset({"127.0.0.1", "localhost"}).union(
    host.strip() for host in os.environ.get("CAMERA_MONITOR_HOSTS", "").split(",") if host.strip()
)
PIPELINE_VERSION = "pose-slim-v1-gpu-pipeline-r3"
MAX_CLIENTS = 8
COMPONENTS = frozenset({
    "cached_native_graph.py", "cpu_publication.py", "detector.py", "detector_fast_path.py",
    "detector_inputs.py", "dual_stage.py", "engine.py", "gpu_pose_cuda.py",
    "gpu_pose_estimator.py", "gpu_pose_kernel.cu", "gpu_pose_warp.py",
    "native_detector_graph.py", "result_push.py",
})


class DetectionService:
    def __init__(self, engine, confidence):
        self.engine = engine
        self.confidence = confidence
        self.session_id = str(uuid4())
        self.clients = set()
        self.publications = LatestResultPush(engine, self.session_id)

    def ready_message(self):
        snapshot = self.engine.snapshot()
        detector = getattr(self.engine.detector, "detector", None)
        return {"type": "ready", "mode": "server-rtsp", "sessionId": self.session_id,
                "serverTime": time.time() * 1000, "model": MODEL_NAME,
                "threshold": self.confidence, "poseEnabled": snapshot["poseEnabled"],
                "trackingEnabled": False, "clockSessionId": snapshot["clockSessionId"],
                "detectorInfo": getattr(detector, "info", {}), "pipelineVersion": PIPELINE_VERSION,
                "jointCount": 17, "keypointStates": {"rejected": 0, "estimated": 1, "reliable": 2}}

    async def health(self, _request):
        detector = self.engine.detector
        pose = getattr(detector, "pose", None)
        snapshot = self.engine.snapshot()
        return web.json_response({
            "status": "ready" if snapshot["running"] else "waiting",
            "mode": "server-rtsp", "sessionId": self.session_id, "model": MODEL_NAME,
            "threshold": self.confidence, "pipelineVersion": PIPELINE_VERSION,
            "device": getattr(detector, "device", "unknown"),
            "deviceName": getattr(detector, "device_name", "unknown"),
            "detectorInfo": getattr(getattr(detector, "detector", None), "info", {}),
            "poseProviders": getattr(pose, "actual_providers", []),
            "poseBackendRequested": getattr(pose, "backend_requested", None),
            "poseBackendSelected": getattr(pose, "backend_selected", None),
            "poseFallbackReason": getattr(pose, "fallback_reason", None),
            "posePreparationReuse": getattr(pose, "reuse_preparation", False),
            "posePrecision": getattr(pose, "precision", None),
            "poseBatchSize": getattr(pose, "pose_batch_size", None),
            "clients": len(self.clients), "maxClients": MAX_CLIENTS, **snapshot,
        }, headers={"Cache-Control": "no-store"})

    async def websocket(self, request):
        # Only the monitor's own HTTP page may subscribe, including on LAN.
        if request.host not in request.app[WS_AUTHORITIES] or request.headers.get("Origin") != f"http://{request.host}":
            raise web.HTTPForbidden(text="仅接受同源监控网页")
        if len(self.clients) >= MAX_CLIENTS:
            raise web.HTTPServiceUnavailable(text="请关闭多余的监控页面")
        ws = web.WebSocketResponse(max_msg_size=4096, heartbeat=15)
        await ws.prepare(request)
        self.clients.add(ws)
        sender = None
        try:
            await ws.send_json(self.ready_message())
            sender = asyncio.create_task(self.publications.push_latest(ws))
            async for message in ws:
                if message.type in (WSMsgType.TEXT, WSMsgType.BINARY):
                    await ws.close(code=1008, message=b"results subscription only")
                    break
                if message.type == WSMsgType.ERROR:
                    break
        except (ConnectionError, OSError, RuntimeError):
            pass
        finally:
            if sender is not None:
                sender.cancel()
                with suppress(asyncio.CancelledError):
                    await sender
            self.clients.discard(ws)
            with suppress(ConnectionError, OSError, RuntimeError):
                await ws.close()
        return ws

    async def startup(self, _app):
        self.publications.start()
        self.engine.on_published = self.publications.notify_published
        try:
            self.engine.start()
        except BaseException:
            self.engine.on_published = None
            self.publications.close()
            await asyncio.to_thread(self.engine.stop)
            raise

    async def shutdown(self, _app):
        self.engine.on_published = None
        self.publications.close()
        await asyncio.gather(*(client.close(code=1001, message=b"server stopping")
                               for client in tuple(self.clients)), return_exceptions=True)

    async def cleanup(self, _app):
        await asyncio.to_thread(self.engine.stop)


WS_AUTHORITIES = web.AppKey("ws_authorities", frozenset)


def create_app(engine, *, web_root=WEB_ROOT, port=DEFAULT_PORT, confidence=None):
    if confidence is None:
        confidence = load_detection_settings()["confidence"]
    service = DetectionService(engine, confidence)
    app = web.Application(client_max_size=4096)
    app[WS_AUTHORITIES] = frozenset(f"{host}:{port}" for host in MONITOR_HOSTS)
    app.router.add_get("/health", service.health)
    app.router.add_get("/ws", service.websocket)

    async def monitor_index(_request):
        raise web.HTTPFound("/web/index.html")

    app.router.add_get("/", monitor_index)
    app.router.add_static("/web/", web_root, show_index=False, follow_symlinks=False)
    app.on_startup.append(service.startup)
    app.on_shutdown.append(service.shutdown)
    app.on_cleanup.append(service.cleanup)
    return app


def checked_components():
    location = PROJECT / "live_pipeline"
    manifest = json.loads((location / "sha256.json").read_text(encoding="utf-8"))
    if set(manifest) != COMPONENTS:
        raise RuntimeError("Incomplete live pipeline component manifest")
    for name, expected in manifest.items():
        if hashlib.sha256((location / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Live pipeline component changed: {name}")


def check_app(port):
    checked_components()
    engine = GpuPipelineEngine(object())
    paths = {route.resource.canonical for route in create_app(engine, port=port).router.routes()}
    if paths != {"/", "/web", "/health", "/ws"} or engine.started:
        raise RuntimeError("Unexpected monitor routes or active capture")
    missing = [path for path in ("index.html", "classroom/index.html") if not (WEB_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing monitor pages: {', '.join(missing)}")
    print("App check passed. No sources or models started.", flush=True)


def port_available(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            if error.errno in (errno.EADDRINUSE, 10048) or getattr(error, "winerror", None) == 10048:
                return False
            raise
    return True


def run(host, port):
    if not port_available(host, port):
        raise RuntimeError(f"端口 {port} 已由其他进程监听")
    checked_components()
    import cv2

    cv2.setNumThreads(2)
    confidence = load_detection_settings()["confidence"]
    detector = GpuPipelineDetector(confidence)
    engine = None
    try:
        engine = GpuPipelineEngine(detector)
        print(f"[AI] {PIPELINE_VERSION} · RF-DETR-L · TensorRT FP16 batch16 · {detector.device_name}", flush=True)
        web.run_app(create_app(engine, port=port, confidence=confidence), host=host, port=port, access_log=None)
    finally:
        if not detector.pose.closed and (engine is None or not engine.inference_thread.is_alive()):
            detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check-app", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.check_app:
        check_app(args.port)
    else:
        run(args.host, args.port)
