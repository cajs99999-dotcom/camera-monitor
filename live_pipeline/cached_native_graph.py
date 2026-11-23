"""Capture the same RF-DETR eager forward with its built-in shape cache.

The ordinary native branch copies a CPU integer shape tensor to CUDA every
forward. RF-DETR 1.12.1 supplies an instance-local GPU cache for this exact
integer geometry. Its warmup populates the cache before the graph capture.
The original transformer setting is restored even when capture fails.
"""
from .native_detector_graph import NativeDetectorGraph


class CachedNativeDetectorGraph(NativeDetectorGraph):
    def __init__(self, detector):
        model = detector.model.model.model
        transformer = model.transformer
        if (type(transformer).__module__ != "rfdetr.models.transformer"
                or getattr(transformer, "_cuda_graph_spatial_shapes", False) is not None
                or not callable(getattr(transformer, "enable_cuda_graph_capture", None))):
            raise ValueError("Expected the native transformer with an unused CUDA Graph shape cache")

        try:
            transformer.enable_cuda_graph_capture()
            # The installed capture helper makes three real warmup forwards
            # on the capture stream before recording the fixed B1 forward.
            super().__init__(detector)
            shapes = transformer._cuda_graph_spatial_shapes
            if not shapes:
                raise RuntimeError("The native warmup did not populate the spatial shape cache")
            for (device, geometry), tensor in shapes.items():
                if (tensor.device != device or str(tensor.dtype) != "torch.int64"
                        or tensor.detach().cpu().tolist() != [list(hw) for hw in geometry]):
                    raise RuntimeError("Cached spatial shape tensors differ from the native geometry")
            # Graph kernels read this storage directly. Keep it alive while
            # the graph can replay, even when the native path is restored.
            self._captured_spatial_shapes = shapes
        finally:
            transformer._cuda_graph_spatial_shapes = None
