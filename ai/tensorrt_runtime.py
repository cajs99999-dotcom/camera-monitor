"""Independent TensorRT validation runner; not used by the production estimator.

Imports create no GPU objects. Construction is intentionally a GPU operation.
CUDA 12.6 function declarations are matched to the installed official headers.
"""
from __future__ import annotations

import ctypes as ct
from pathlib import Path
import time

import numpy as np

CUDA12_DLL = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudart64_12.dll")


class Cuda12Runtime:
    """Checked CUDA Runtime ABI, with explicit pointer/size/stream/event types."""

    def __init__(self, dll_path=CUDA12_DLL, *, library=None):
        self.path = Path(dll_path)
        self.lib = library if library is not None else ct.WinDLL(str(self.path.resolve(strict=True)))
        p, size, uint, integer = ct.c_void_p, ct.c_size_t, ct.c_uint, ct.c_int
        signatures = {
            "cudaGetErrorString": ([integer], ct.c_char_p),
            "cudaSetDevice": ([integer], integer),
            "cudaRuntimeGetVersion": ([ct.POINTER(integer)], integer),
            "cudaDriverGetVersion": ([ct.POINTER(integer)], integer),
            "cudaMalloc": ([ct.POINTER(p), size], integer),
            "cudaFree": ([p], integer),
            "cudaHostAlloc": ([ct.POINTER(p), size, uint], integer),
            "cudaFreeHost": ([p], integer),
            "cudaMemcpyAsync": ([p, p, size, integer, p], integer),
            "cudaStreamCreateWithFlags": ([ct.POINTER(p), uint], integer),
            "cudaStreamSynchronize": ([p], integer),
            "cudaStreamDestroy": ([p], integer),
            "cudaEventCreateWithFlags": ([ct.POINTER(p), uint], integer),
            "cudaEventRecord": ([p, p], integer),
            "cudaEventElapsedTime": ([ct.POINTER(ct.c_float), p, p], integer),
            "cudaEventDestroy": ([p], integer),
            "cudaMemGetInfo": ([ct.POINTER(size), ct.POINTER(size)], integer),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, result

    def call(self, name, *args):
        code = int(getattr(self.lib, name)(*args))
        if code:
            text = self.lib.cudaGetErrorString(code)
            raise RuntimeError(f"{name} failed ({code}): {text.decode(errors='replace') if text else 'unknown CUDA error'}")

    def create(self, name, *args):
        pointer = ct.c_void_p()
        self.call(name, ct.byref(pointer), *args)
        if not pointer.value:
            raise RuntimeError(f"{name} returned a null handle")
        return int(pointer.value)

    def versions(self):
        values = {}
        for name in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
            value = ct.c_int()
            self.call(name, ct.byref(value))
            values[name] = value.value
        if not 12000 <= values["cudaRuntimeGetVersion"] < 13000:
            raise RuntimeError("The validation runner requires the explicitly selected CUDA12 runtime")
        return values

    def memory_info(self):
        free, total = ct.c_size_t(), ct.c_size_t()
        self.call("cudaMemGetInfo", ct.byref(free), ct.byref(total))
        return {"freeBytes": free.value, "totalBytes": total.value}


class TensorRTPoseRunner:
    """One synchronous host-facing call; cached pinned/device buffers by batch.

    callMs includes host staging, dynamic shape/address binding, H2D, execution,
    D2H, synchronization and detached NumPy outputs. gpuKernelMs uses CUDA events
    around execution only, and must not be compared with ORT's host call time.
    """

    def __init__(self, engine_path, model_path, *, cuda_dll=CUDA12_DLL,
                 cuda=None, trt_module=None, manifest_validator=None):
        self.cuda = None
        self.runtime = self.engine = self.context = self.logger = None
        self.stream = self.start_event = self.end_event = None
        self.device_buffers, self.host_buffers, self.batches = [], [], {}
        self.closed = False
        try:
            if manifest_validator is None:
                raise ValueError("TensorRT deserialization requires the verified engine manifest validator")
            # This must precede any CUDA allocation or engine deserialization.
            self.manifest = manifest_validator(Path(engine_path), Path(model_path))
            if trt_module is None:
                import tensorrt as trt_module
            self.trt = trt_module
            self.cuda = cuda if cuda is not None else Cuda12Runtime(cuda_dll)
            self.versions = self.cuda.versions()
            self.cuda.call("cudaSetDevice", 0)
            self.memory_before = self.cuda.memory_info()
            self.logger = self.trt.Logger(self.trt.Logger.WARNING)
            self.runtime = self.trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
            if self.engine is None:
                raise RuntimeError("TensorRT deserialization failed")
            expected = {"input": self.trt.TensorIOMode.INPUT, "simcc_x": self.trt.TensorIOMode.OUTPUT,
                        "simcc_y": self.trt.TensorIOMode.OUTPUT}
            names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
            if names != set(expected) or self.engine.num_optimization_profiles != 1:
                raise RuntimeError("Unexpected engine I/O/profile count")
            for name, mode in expected.items():
                if self.engine.get_tensor_mode(name) != mode or self.engine.get_tensor_dtype(name) != self.trt.float32:
                    raise RuntimeError(f"Expected float32 {mode} tensor {name}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Cannot create TensorRT execution context")
            self.stream = self.cuda.create("cudaStreamCreateWithFlags", 1)
            self.start_event = self.cuda.create("cudaEventCreateWithFlags", 0)
            self.end_event = self.cuda.create("cudaEventCreateWithFlags", 0)
        except Exception as exc:
            try:
                self.close()
            except Exception as cleanup:
                exc.add_note(f"Cleanup after initialization error: {cleanup}")
            raise

    def _buffer(self, shape):
        elements = int(np.prod(shape))
        size = elements * np.dtype(np.float32).itemsize
        host = self.cuda.create("cudaHostAlloc", size, 0)
        self.host_buffers.append(host)
        device = self.cuda.create("cudaMalloc", size)
        self.device_buffers.append(device)
        array = np.ctypeslib.as_array((ct.c_float * elements).from_address(host)).reshape(shape)
        return {"host": host, "device": device, "array": array, "bytes": size}

    def run_batch(self, inputs):
        if self.closed:
            raise RuntimeError("TensorRT runner is closed")
        inputs = np.asarray(inputs)
        if (inputs.dtype != np.float32 or inputs.ndim != 4 or inputs.shape[1:] != (3, 384, 288)
                or not 1 <= len(inputs) <= 16 or not np.isfinite(inputs).all()):
            raise ValueError("Expected finite float32 Nx3x384x288 input, batch1..16")
        started = time.perf_counter()
        n = len(inputs)
        if not self.context.set_input_shape("input", tuple(inputs.shape)):
            raise RuntimeError("TensorRT rejected input shape")
        shapes = {"input": tuple(inputs.shape), "simcc_x": (n, 17, 576), "simcc_y": (n, 17, 768)}
        for name, shape in shapes.items():
            if tuple(self.context.get_tensor_shape(name)) != shape:
                raise RuntimeError(f"Unexpected runtime tensor shape for {name}")
        if n not in self.batches:
            self.batches[n] = {name: self._buffer(shape) for name, shape in shapes.items()}
        buffers = self.batches[n]
        for name, buffer in buffers.items():
            if not self.context.set_tensor_address(name, buffer["device"]):
                raise RuntimeError(f"Could not bind {name}")
        np.copyto(buffers["input"]["array"], inputs)
        source = buffers["input"]
        self.cuda.call("cudaMemcpyAsync", source["device"], source["host"], source["bytes"], 1, self.stream)
        self.cuda.call("cudaEventRecord", self.start_event, self.stream)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution enqueue failed")
        self.cuda.call("cudaEventRecord", self.end_event, self.stream)
        for name in ("simcc_x", "simcc_y"):
            buffer = buffers[name]
            self.cuda.call("cudaMemcpyAsync", buffer["host"], buffer["device"], buffer["bytes"], 2, self.stream)
        self.cuda.call("cudaStreamSynchronize", self.stream)
        elapsed = ct.c_float()
        self.cuda.call("cudaEventElapsedTime", ct.byref(elapsed), self.start_event, self.end_event)
        outputs = {name: buffers[name]["array"].copy() for name in ("simcc_x", "simcc_y")}
        if not all(np.isfinite(array).all() for array in outputs.values()):
            raise RuntimeError("Non-finite TensorRT output")
        return outputs, {"callMs": (time.perf_counter() - started) * 1000,
                         "gpuKernelMs": float(elapsed.value)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []

        def cleanup(name, *args):
            try:
                self.cuda.call(name, *args)
            except Exception as exc:
                errors.append(str(exc))

        if self.cuda is not None:
            if self.stream is not None:
                cleanup("cudaStreamSynchronize", self.stream)
            self.context = None
            for pointer in reversed(self.device_buffers):
                cleanup("cudaFree", pointer)
            for pointer in reversed(self.host_buffers):
                cleanup("cudaFreeHost", pointer)
            for event in (self.end_event, self.start_event):
                if event is not None:
                    cleanup("cudaEventDestroy", event)
            if self.stream is not None:
                cleanup("cudaStreamDestroy", self.stream)
        self.batches.clear()
        self.device_buffers.clear()
        self.host_buffers.clear()
        self.engine = self.runtime = self.logger = None
        if errors:
            raise RuntimeError("Cleanup errors: " + " | ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"TensorRT cleanup: {cleanup}")
        return False
