"""NVRTC-compiled crop launched on the existing TensorRT CUDA 12 stream."""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path

from ai.tensorrt_runtime import CUDA12_DLL

KERNEL_PATH = Path(__file__).with_name("gpu_pose_kernel.cu")
NVRTC_PATH = CUDA12_DLL.with_name("nvrtc64_120_0.dll")


def _bind(lib, name, args, result=ct.c_int):
    method = getattr(lib, name)
    method.argtypes, method.restype = args, result
    return method


def compile_ptx(*, architecture="compute_89") -> bytes:
    """Compile device-only PTX. This does not initialize CUDA or use a GPU."""
    p, size, i, text = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_char_p
    with os.add_dll_directory(str(NVRTC_PATH.parent)):
        lib = ct.WinDLL(str(NVRTC_PATH.resolve(strict=True)))
        _bind(lib, "nvrtcGetErrorString", [i], text)
        _bind(lib, "nvrtcCreateProgram", [ct.POINTER(p), text, text, i, ct.POINTER(text), ct.POINTER(text)])
        _bind(lib, "nvrtcCompileProgram", [p, i, ct.POINTER(text)])
        for suffix in ("ProgramLog", "PTX"):
            _bind(lib, f"nvrtcGet{suffix}Size", [p, ct.POINTER(size)])
            _bind(lib, f"nvrtcGet{suffix}", [p, p])
        _bind(lib, "nvrtcDestroyProgram", [ct.POINTER(p)])

        def get_bytes(program, suffix):
            n = size()
            _check(getattr(lib, f"nvrtcGet{suffix}Size")(program, ct.byref(n)))
            out = ct.create_string_buffer(n.value)
            _check(getattr(lib, f"nvrtcGet{suffix}")(program, out))
            return out.raw

        def _check(result, details=b""):
            if result:
                name = lib.nvrtcGetErrorString(result) or str(result).encode()
                detail = (name + b"\n" + details).rstrip(b"\0\n").decode(errors="replace")
                raise RuntimeError(f"CUDA crop compile failed: {detail}")

        program = p()
        _check(lib.nvrtcCreateProgram(ct.byref(program), KERNEL_PATH.read_bytes(), b"gpu_pose_kernel.cu", 0, None, None))
        try:
            options = (text * 2)(f"--gpu-architecture={architecture}".encode(), b"--std=c++14")
            result = lib.nvrtcCompileProgram(program, len(options), options)
            if result:
                _check(result, get_bytes(program, "ProgramLog"))
            return get_bytes(program, "PTX")
        finally:
            _check(lib.nvrtcDestroyProgram(ct.byref(program)))


class CUDAWarpKernel:
    """Use the active device-0 primary context, without creating another context."""

    def __init__(self, ptx: bytes | None = None):
        self._module = ct.c_void_p()
        self._function = ct.c_void_p()
        self._context = None
        self._closed = False
        p, u = ct.c_void_p, ct.c_uint
        lib = self._lib = ct.WinDLL("nvcuda.dll")
        _bind(lib, "cuInit", [u])
        _bind(lib, "cuCtxGetCurrent", [ct.POINTER(p)])
        _bind(lib, "cuGetErrorName", [ct.c_int, ct.POINTER(ct.c_char_p)])
        _bind(lib, "cuModuleLoadData", [ct.POINTER(p), p])
        _bind(lib, "cuModuleGetFunction", [ct.POINTER(p), p, ct.c_char_p])
        _bind(lib, "cuModuleUnload", [p])
        _bind(lib, "cuLaunchKernel", [p, u, u, u, u, u, u, u, p, ct.POINTER(p), p])
        self._call("cuInit", 0)
        self._context = self._current_context()
        if self._context is None:
            raise RuntimeError("TensorRT CUDA context must be current before loading the crop kernel")
        try:
            blob = ct.create_string_buffer(compile_ptx() if ptx is None else ptx)
            self._call("cuModuleLoadData", ct.byref(self._module), blob)
            self._call("cuModuleGetFunction", ct.byref(self._function), self._module, b"rtmpose_warp_rgb_chw")
        except Exception:
            self.close()
            raise

    def _call(self, operation, *args):
        result = getattr(self._lib, operation)(*args)
        if result:
            name = ct.c_char_p()
            self._lib.cuGetErrorName(result, ct.byref(name))
            detail = name.value.decode(errors="replace") if name.value else str(result)
            raise RuntimeError(f"{operation} failed for CUDA pose crop ({detail})")

    def _current_context(self):
        result = ct.c_void_p()
        self._call("cuCtxGetCurrent", ct.byref(result))
        return result.value

    def launch(self, stream, source, width, height, terms, lut, destination, count):
        if self._closed or self._current_context() != self._context:
            raise RuntimeError("CUDA pose crop needs its original active device context")
        values = (ct.c_uint64(source), ct.c_int(width), ct.c_int(height),
                  ct.c_uint64(terms), ct.c_uint64(lut), ct.c_uint64(destination))
        parameters = (ct.c_void_p * len(values))(*(ct.addressof(value) for value in values))
        block_size = 128
        blocks = (288 * 384 + block_size - 1) // block_size
        self._call("cuLaunchKernel", self._function, blocks, count, 1, block_size, 1, 1, 0,
                   stream, parameters, None)

    def close(self):
        if self._closed:
            return
        if self._module.value:
            if self._current_context() != self._context:
                raise RuntimeError("Cannot release CUDA crop module in a different device context")
            self._call("cuModuleUnload", self._module)
        self._closed = True
