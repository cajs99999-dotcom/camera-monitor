"""Review-only RF-DETR inference snapshot with pinned classroom-v1 semantics."""
from importlib.metadata import version
from types import SimpleNamespace


PACKAGE_VERSION = "1.12.1"
CHECKPOINT_SHA256 = "a46f3f687bbcae50a4a671ec5e562ba0f888d8a068f4fa2c9a5c6e9cd02b3b30"
MODEL_SHAPE = (704, 704)
MODES = frozenset({"eager", "torchscript", "cudagraph"})


def _require_classroom_detector(detector):
    if not (detector.profile == "classroom_v1" and detector.info.get("sha256") == CHECKPOINT_SHA256
            and detector.input_size == MODEL_SHAPE[0] and detector.antialias is True
            and detector.person_ids == [0] and detector.confidence == .45):
        raise ValueError("Review requires the pinned classroom-v1 detector with 704/AA/person0/.45")


def _context_fingerprint(predictor):
    context = predictor._prediction_context()
    device = str(context.device)
    if (tuple(context.default_shape) != MODEL_SHAPE or context.class_id_to_name != {0: "person"}
            or context.config.num_channels != 3 or device.split(":", 1)[0] != "cuda"):
        raise ValueError("Prediction context changed shape, class map, channels, or device")
    if device == "cuda":
        from torch.cuda import current_device

        device = f"cuda:{current_device()}"
    return SimpleNamespace(device=device, channels=context.config.num_channels,
        shape=tuple(context.default_shape), means=tuple(context.means), stds=tuple(context.stds),
        patch_size=context.config.patch_size, num_windows=context.config.num_windows,
        batch_size=context.batch_size)


def create_predictor(detector, *, mode="cudagraph"):
    """Return an independent FP32 predictor without replacing the detector's native model.

    Create this before worker threads start. A CUDA Graph capture failure must end
    that experimental process, as the installed RF-DETR runtime documents.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown review detector mode: {mode!r}")
    if version("rfdetr") != PACKAGE_VERSION:
        raise RuntimeError(f"Review requires RF-DETR {PACKAGE_VERSION}")
    _require_classroom_detector(detector)
    source = detector.model
    expected = _context_fingerprint(source)
    if expected.batch_size is not None:
        raise ValueError("Build the review predictor from the native detector")
    options = {"compile": mode != "eager", "batch_size": 1, "dtype": "float32"}
    if mode != "eager":
        options["compile_backend"] = mode
    predictor = source.inference(**options)
    actual = _context_fingerprint(predictor)
    expected.batch_size = None if mode == "eager" else 1
    if actual != expected:
        raise ValueError("Review predictor normalization, patching, or compiled batch contract changed")
    return predictor


def select_predictor(detector, predictor, *, mode):
    """Use a previously validated snapshot, or restore the saved native model."""
    if mode not in MODES | {"native"}:
        raise ValueError(f"Unknown review detector mode: {mode!r}")
    _require_classroom_detector(detector)
    expected_batch_size = None if mode in ("native", "eager") else 1
    if _context_fingerprint(predictor).batch_size != expected_batch_size:
        raise ValueError("Selected predictor has the wrong batch contract")
    detector.model = predictor
