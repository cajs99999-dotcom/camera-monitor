"""Hidden Windows FFmpeg relay for a MediaMTX runOnDemand hook.

Run with .venv-pose/Scripts/pythonw.exe. The child does not break away from
MediaMTX's inherited Windows Job Object, so closing that job stops the tree.
Exit codes: 0 = FFmpeg completed, 1 = FFmpeg failed, 2 = setup/start failure,
130 = interrupted. The original FFmpeg status is retained in the private log.
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import subprocess
import threading


def command(ffmpeg: Path, camera: str, rtsp_port: int = 8554, *, transcode_h264: bool = False) -> list[str]:
    video_options = ["-c:v", "copy"]
    if transcode_h264:
        video_options = ["-c:v", "libx264", "-threads", "2", "-preset", "veryfast",
                         "-tune", "zerolatency", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
                         "-bf", "0", "-g", "30"]
    return [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "warning",
            "-rtsp_transport", "tcp", "-timeout", "5000000",
            "-use_wallclock_as_timestamps", "1",
            "-i", f"rtsp://127.0.0.1:{rtsp_port}/{camera}",
            "-map", "0:v:0", "-an", *video_options, "-f", "rtsp",
            "-rtsp_transport", "tcp", f"rtsp://127.0.0.1:{rtsp_port}/{camera}_browser"]


def private_logger(camera: str) -> logging.Logger:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is required for private logs")
    directory = Path(local_app_data) / "CameraMonitor" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"browser_remux.{camera}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Two files per camera: the active file and one rotated backup, 1 MiB each.
    handler = RotatingFileHandler(directory / f"{camera}_browser_remux.log",
                                  maxBytes=1024 * 1024, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def drain_stderr(pipe, logger: logging.Logger) -> None:
    try:
        # Bound each read even if a child emits an unusually long stderr line.
        while chunk := pipe.readline(8192):
            text = chunk.decode("utf-8", errors="replace").rstrip("\r\n")
            if text:
                logger.warning("ffmpeg: %s", text)
    except (OSError, ValueError):
        logger.info("stderr pipe closed")
    finally:
        pipe.close()


def stop_child(child: subprocess.Popen, logger: logging.Logger) -> None:
    if child.poll() is not None:
        return
    logger.info("interrupt: terminating FFmpeg pid=%s", child.pid)
    try:
        child.terminate()
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg did not terminate within 5 seconds; killing")
        child.kill()
        child.wait(timeout=5)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", choices=[f"cam{i}" for i in range(1, 7)])
    parser.add_argument("--ffmpeg", type=Path,
                        help="Absolute FFmpeg executable path (defaults to the PATH lookup)")
    parser.add_argument("--transcode-h264", action="store_true",
                        help="Encode H.264 for browsers when the upstream codec is HEVC")
    parser.add_argument("--rtsp-port", type=int, default=os.environ.get("RTSP_PORT", "8554"),
                        help="Local RTSP port (defaults to MediaMTX RTSP_PORT or 8554)")
    args = parser.parse_args()
    if args.ffmpeg is not None and not args.ffmpeg.is_absolute():
        parser.error("--ffmpeg must be an absolute path")
    if not 1024 <= args.rtsp_port <= 65535:
        parser.error("--rtsp-port must be between 1024 and 65535")
    try:
        logger = private_logger(args.camera)
    except (OSError, RuntimeError):
        # pythonw has no reliable stderr. Do not fall back to the served project.
        return 2
    child = None
    drainer = None
    try:
        executable = args.ffmpeg or shutil.which("ffmpeg")
        if executable is None or not Path(executable).is_file():
            logger.error("FFmpeg executable not found; add it to PATH or use --ffmpeg")
            return 2
        logger.info("starting %s -> %s_browser on localhost:%s using %s",
                    args.camera, args.camera, args.rtsp_port,
                    "H.264 encoding" if args.transcode_h264 else "stream copy")
        child = subprocess.Popen(command(Path(executable), args.camera, args.rtsp_port,
                                         transcode_h264=args.transcode_h264), shell=False,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, close_fds=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        logger.info("FFmpeg started pid=%s", child.pid)
        drainer = threading.Thread(target=drain_stderr, args=(child.stderr, logger),
                                   name=f"stderr-{args.camera}", daemon=True)
        drainer.start()
        status = child.wait()
        logger.info("FFmpeg exited status=%s unsigned_hex=0x%08X", status, status & 0xFFFFFFFF)
        return 0 if status == 0 else 1
    except KeyboardInterrupt:
        if child is not None:
            stop_child(child, logger)
        return 130
    except (OSError, subprocess.SubprocessError):
        logger.exception("FFmpeg setup or process control failed")
        if child is not None:
            stop_child(child, logger)
        return 2
    finally:
        if drainer is not None:
            drainer.join(timeout=2)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
