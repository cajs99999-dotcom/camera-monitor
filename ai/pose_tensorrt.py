"""Optional, guarded RTMPose TensorRT session; importing this creates no GPU work.

The ORT-shaped interface lets the estimator retain its exact preparation,
batching, SimCC decoding, and quality decisions. Only construction of a fully
validated session may deserialize the local engine.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .tensorrt_evidence import MODEL_SHA256, check_manifest, runtime_fingerprint, sha256_file


SDK = Path(r"D:\NVIDIA\TensorRT-11.4.0.106")
CUDA_BIN = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
CUDA_DLL = CUDA_BIN / "cudart64_12.dll"
CUDA_DLL_SHA256 = "3df50cf48718712f17b8c77ad72c45a1f43ff94781e302a4be7f1d7a9b66c746"
# Locks the reviewed numerical report, whose manifest hashes bind the two
# backends, source ONNX, prepared inputs, and this particular engine.
COMPARISON_SHA256 = "bc234c1c8a5c8ab4858cdcb9bd23cd12dbffa4db4e8bc2b71094095600e142c5"
EXPECTED_CASES = {"batch1", "batch2", "batch4", "batch8", "batch16", "people15_b8", "people17_b8"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_numerical_evidence(engine_path, model_path, acceptance_path):
    """Only file/JSON/hash checks: no ONNX parser, TRT import, or CUDA context."""
    engine_path = Path(engine_path).resolve(strict=True)
    model_path = Path(model_path).resolve(strict=True)
    acceptance_path = Path(acceptance_path).resolve(strict=True)
    engine_sha = sha256_file(engine_path)
    model_sha = sha256_file(model_path)
    if model_sha != MODEL_SHA256:
        raise ValueError("Unvalidated RTMPose source model SHA256")
    manifest = _read_json(engine_path.with_suffix(engine_path.suffix + ".json"))
    if manifest.get("engineSha256") != engine_sha or manifest.get("model", {}).get("sha256") != model_sha:
        raise ValueError("Engine/source manifest SHA256 mismatch")
    if sha256_file(acceptance_path) != COMPARISON_SHA256:
        raise ValueError("Numerical acceptance report SHA256 mismatch")
    report = _read_json(acceptance_path)
    cases = report.get("cases", [])
    if (report.get("passed") is not True or len(cases) != len(EXPECTED_CASES)
            or {case.get("name") for case in cases} != EXPECTED_CASES
            or any(case.get("passed") is not True for case in cases)
            or report.get("criteria") != {"stablePixelErrorStrictlyLessThan": 2.0,
                                          "stableScoreErrorAtMost": 0.02,
                                          "noReliableRejectedStateFlips": True}):
        raise ValueError("Incomplete or failed numerical acceptance")
    backends = {}
    for name, key in (("tensorrt", "candidateManifestSha256"), ("ort_cuda", "referenceManifestSha256")):
        path = acceptance_path.parent / name / "manifest.json"
        if sha256_file(path) != report.get(key):
            raise ValueError(f"Numerical {name} manifest SHA256 mismatch")
        result = _read_json(path)
        if (result.get("schema") != "rtmpose-backend-results-v1" or result.get("status") != "completed"
                or result.get("backend") != name or result.get("modelSha256") != model_sha
                or len(result.get("cases", [])) != len(EXPECTED_CASES)
                or {case.get("name") for case in result["cases"]} != EXPECTED_CASES):
            raise ValueError(f"Numerical {name} results do not match the model/cases")
        backends[name] = result
    candidate, reference = backends["tensorrt"], backends["ort_cuda"]
    if (candidate.get("environment", {}).get("engineSha256") != engine_sha
            or candidate.get("environment", {}).get("runtimeFingerprint") != manifest.get("runtimeFingerprint")
            or not candidate.get("inputsSha256") or not candidate.get("preparedManifestSha256")
            or candidate["inputsSha256"] != reference.get("inputsSha256")
            or candidate["preparedManifestSha256"] != reference.get("preparedManifestSha256")):
        raise ValueError("Numerical evidence is not bound to this engine/runtime/input set")
    return manifest, {"engineSha256": engine_sha, "modelSha256": model_sha,
                      "comparisonSha256": COMPARISON_SHA256,
                      "numericalCasesPassed": len(cases),
                      "scope": "Same-ONNX numerical agreement; live mixed CUDA runtime smoke remains separate"}


def _open_dll_directories():
    """Retain per-process handles; do not edit PATH or any machine settings."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        raise RuntimeError("The validated TensorRT adapter requires Windows DLL directories")
    if not CUDA_DLL.is_file() or sha256_file(CUDA_DLL) != CUDA_DLL_SHA256:
        raise ValueError("Validated CUDA12.6 runtime DLL is missing or changed")
    directories = (SDK / "lib", SDK / "bin", CUDA_BIN)
    handles = []
    try:
        for directory in directories:
            if not directory.is_dir():
                raise FileNotFoundError(f"TensorRT/CUDA DLL directory missing: {directory}")
            handles.append(os.add_dll_directory(str(directory)))
        return handles
    except Exception:
        for handle in reversed(handles):
            handle.close()
        raise


@contextmanager
def _sdk_import_path():
    """Official wheel's find_lib scans PATH; change only this process briefly."""
    previous = os.environ.get("PATH")
    os.environ["PATH"] = str(SDK / "bin") + (os.pathsep + previous if previous else "")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous


class TensorRTPoseSession:
    """Minimal ORT session contract with explicit actual provider reporting."""

    def __init__(self, engine_path, model_path, acceptance_path):
        self.runner = None
        self._dll_handles = []
        self.closed = False
        self.last_timings = {}
        try:
            manifest, self.evidence = validate_numerical_evidence(engine_path, model_path, acceptance_path)
            self._dll_handles = _open_dll_directories()
            with _sdk_import_path():
                fingerprint, _ = runtime_fingerprint()  # nvidia-smi/DLL inspection, no context
                check_manifest(manifest, self.evidence["engineSha256"],
                               {"sha256": self.evidence["modelSha256"]}, fingerprint)
                from .tensorrt_runtime import TensorRTPoseRunner
                # Repeat the pre-allocation guard without an ONNX parser.
                def validated(engine, model):
                    fresh, _ = validate_numerical_evidence(engine, model, acceptance_path)
                    check_manifest(fresh, sha256_file(engine), {"sha256": sha256_file(model)}, fingerprint)
                    return fresh
                self.runner = TensorRTPoseRunner(engine_path, model_path, cuda_dll=CUDA_DLL,
                                                manifest_validator=validated)
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"TensorRT initialization cleanup: {cleanup}")
            raise

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=["batch", 3, 384, 288], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name=name, shape=["batch", 17, length], type="tensor(float)")
                for name, length in (("simcc_x", 576), ("simcc_y", 768))]

    def get_providers(self):
        return ["TensorRT11_CUDA12"]

    def run(self, output_names, inputs):
        if self.closed:
            raise RuntimeError("TensorRT pose session is closed")
        names = ["simcc_x", "simcc_y"] if output_names is None else list(output_names)
        if not names or any(name not in {"simcc_x", "simcc_y"} for name in names):
            raise ValueError("Unsupported RTMPose output name")
        if not isinstance(inputs, dict) or set(inputs) != {"input"}:
            raise ValueError("Expected one input named input")
        value = inputs["input"]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 4
                or value.shape[1:] != (3, 384, 288) or not 1 <= len(value) <= 16
                or not np.isfinite(value).all()):
            raise ValueError("Expected finite float32 Nx3x384x288 input, batch1..16")
        outputs, self.last_timings = self.runner.run_batch(value)
        for name, length in (("simcc_x", 576), ("simcc_y", 768)):
            result = outputs[name]
            if result.dtype != np.float32 or result.shape != (len(value), 17, length) or not np.isfinite(result).all():
                raise RuntimeError("Unexpected TensorRT pose output shape/type/value")
        return [outputs[name] for name in names]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.runner is not None:
                self.runner.close()
        finally:
            self.runner = None
            for handle in reversed(self._dll_handles):
                handle.close()
            self._dll_handles.clear()
