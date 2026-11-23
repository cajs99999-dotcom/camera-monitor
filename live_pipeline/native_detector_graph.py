"""Isolated RF-DETR CUDA Graph using the unmodified production forward."""
from dataclasses import replace
from importlib.metadata import version

from .detector_fast_path import MODEL_SHAPE, PACKAGE_VERSION, _require_classroom_detector, _context_fingerprint

_CONTEXT_NAME = "_prediction_context"
_INPUT_SHAPE = (1, 3, *MODEL_SHAPE)


def _numeric_policy(torch):
    return (torch.get_default_dtype(), torch.is_autocast_enabled("cuda"),
            torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled())


def _capture_native_run(detector, get_context):
    import numpy as np
    import torch
    from rfdetr._prediction.core import _prepare_batch
    from rfdetr.detr import _CUDAGraphInferenceModel

    # Match the tensor that the ordinary detector preprocessing creates. The
    # test sample selects sizes/strides; each replay copies the live input.
    policy = _numeric_policy(torch)
    if policy[0] != torch.float32 or policy[1]:
        raise ValueError("Native graph requires FP32 with CUDA autocast disabled")
    image = np.zeros((*MODEL_SHAPE, 3), dtype=np.uint8)
    with torch.inference_mode():
        prepared = _prepare_batch(get_context, image, detector.confidence, MODEL_SHAPE,
                                  None, False, detector.antialias)
        sample = prepared.batch_tensor
        if (tuple(sample.shape) != _INPUT_SHAPE or sample.dtype != torch.float32
                or sample.device.type != "cuda"):
            raise ValueError("Unexpected native detector input shape, dtype or device")
        # RFDETR's graph helper accepts a callable and preserves all returned
        # tensors. Capture the *eager* callable with its existing model weights.
        graph = _CUDAGraphInferenceModel(prepared.context.run, sample, sample.device)
    strides = sample.stride()

    def replay(value):
        if _numeric_policy(torch) != policy:
            raise RuntimeError("Native graph numeric policy changed since capture")
        if not torch.is_inference_mode_enabled() or value.stride() != strides:
            raise ValueError("Native graph requires inference mode and the captured input strides")
        # The graph helper additionally verifies shape, dtype and device.
        return graph(value)

    return replay


class NativeDetectorGraph:
    """Capture before workers start, attach only between benchmark arms.

    Only the loaded source instance's prediction-context `run` is replaced.
    The ordinary RGB conversion, resize, score threshold and result decode
    remain executed by their original code paths. A CUDA Graph capture error
    must end the benchmark process before trying any other candidate.
    """

    def __init__(self, detector):
        if version("rfdetr") != PACKAGE_VERSION:
            raise RuntimeError(f"Native graph requires RF-DETR {PACKAGE_VERSION}")
        _require_classroom_detector(detector)
        source = detector.model
        if (_CONTEXT_NAME in vars(source) or getattr(source, "_is_optimized_for_inference", None) is not False
                or _context_fingerprint(source).batch_size is not None):
            raise ValueError("Native graph must be created from an unmodified native predictor")
        self._detector = detector
        self._source = source
        self._native_context = source._prediction_context
        self._replay = _capture_native_run(detector, self._native_context)
        self._override = self._graph_context

    def _graph_context(self):
        return replace(self._native_context(), run=self._replay)

    def attach(self):
        if self._detector.model is not self._source:
            raise RuntimeError("Restore the original detector before attaching its native graph")
        current = vars(self._source).get(_CONTEXT_NAME)
        if current is self._override:
            return
        if _CONTEXT_NAME in vars(self._source):
            raise RuntimeError("The original detector context was replaced after capture")
        setattr(self._source, _CONTEXT_NAME, self._override)

    def detach(self):
        attributes = vars(self._source)
        if _CONTEXT_NAME not in attributes:
            return
        if attributes[_CONTEXT_NAME] is not self._override:
            raise RuntimeError("Cannot detach a detector context installed by another caller")
        delattr(self._source, _CONTEXT_NAME)
