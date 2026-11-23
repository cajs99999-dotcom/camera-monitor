"""Fingerprint and manifest checks shared by the live TensorRT entry."""
from __future__ import annotations

import csv
import ctypes
import hashlib
import io
from pathlib import Path
import platform
import subprocess
import sys

MODEL_SHA256 = "9acef0f3d6958bdd2427441967397ba22827f913f5b31a838e923dc0e1f3bdfd"
EXPECTED_TRT = "11.4.0.106"
SCHEMA = "rtmpose-l-tensorrt-local-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loaded_dll(name: str) -> Path:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint32
    handle = kernel.GetModuleHandleW(name)
    buffer = ctypes.create_unicode_buffer(32768)
    if not handle or not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise RuntimeError(f"Cannot identify the actually loaded {name}")
    return Path(buffer.value).resolve(strict=True)


def runtime_fingerprint() -> tuple[dict, dict]:
    if platform.system() != "Windows" or platform.machine().upper() not in {"AMD64", "X86_64"}:
        raise RuntimeError("This fixed tool targets this Windows x64 installation")
    import tensorrt as trt

    if trt.__version__ != EXPECTED_TRT:
        raise RuntimeError(f"Use the isolated TensorRT {EXPECTED_TRT} environment; found {trt.__version__}")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    devices = [row for row in csv.reader(io.StringIO(completed.stdout)) if row]
    if len(devices) != 1 or devices[0][0].strip() != "0":
        raise RuntimeError("Expected the single GPU at CUDA ordinal 0; explicit multi-GPU selection is not implemented")
    row = [part.strip() for part in devices[0]]
    dll = loaded_dll("nvinfer_11.dll")
    fingerprint = {"platform": platform.system(), "architecture": platform.machine(),
                   "tensorrt": trt.__version__, "nvinferSha256": sha256_file(dll),
                   "gpu": {"index": 0, "name": row[1], "uuid": row[2],
                           "driverVersion": row[3], "computeCapability": row[4]}}
    details = {"python": sys.executable, "pythonVersion": sys.version, "trtModule": trt.__file__,
               "nvinferDll": str(dll), "networkTyping": "TensorRT11_strongly_typed",
               "cudaContextCreatedByInspection": False}
    return fingerprint, details


def build_settings(workspace_mib: int, opt_batch: int, optimization_level: int = 3) -> dict:
    if workspace_mib not in {256, 512} or not 1 <= opt_batch <= 16 or not 0 <= optimization_level <= 5:
        raise ValueError("Invalid workspace/profile/optimization level")
    return {"workspaceMiB": workspace_mib, "optimizationLevel": optimization_level,
            "tf32": False, "precisionConversion": False,
            "profile": {"input": {"min": [1, 3, 384, 288], "opt": [opt_batch, 3, 384, 288],
                                  "max": [16, 3, 384, 288]}}}


def check_manifest(manifest: dict, engine_sha256: str, model: dict, fingerprint: dict) -> None:
    """CPU-only guard; call before any future Runtime/deserialization operation."""
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "built_unvalidated":
        raise ValueError("Missing or unsupported local engine manifest")
    if manifest.get("engineSha256") != engine_sha256:
        raise ValueError("Engine SHA256 mismatch")
    if manifest.get("model", {}).get("sha256") != model["sha256"]:
        raise ValueError("Source model SHA256 mismatch")
    if manifest.get("runtimeFingerprint") != fingerprint:
        raise ValueError("TensorRT/platform/DLL/GPU/driver fingerprint mismatch; rebuild required")
    settings = manifest.get("buildSettings", {})
    try:
        expected = build_settings(settings["workspaceMiB"], settings["profile"]["input"]["opt"][0],
                                  settings["optimizationLevel"])
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete engine build settings") from exc
    if settings != expected:
        raise ValueError("Unexpected precision/profile/build settings")
