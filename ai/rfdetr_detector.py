"""Local RF-DETR-L person detector with explicit checkpoint preprocessing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "models/pose"
PROFILES = {
    "generic": {
        "filename": "rf-detr-large-2026.pth",
        "sha256": "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38",
        "num_classes": 90, "person_id": 1, "antialias": False,
        "label": "official COCO",
    },
    "classroom_v1": {
        "filename": "rf-detr-large-classroom-v1.pth",
        "sha256": "a46f3f687bbcae50a4a671ec5e562ba0f888d8a068f4fa2c9a5c6e9cd02b3b30",
        "num_classes": 1, "person_id": 0, "antialias": True,
        "label": "classroom v1 EMA",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PersonDetector:
    def __init__(self, model_path: str | Path | None = None, confidence: float = .10,
                 *, profile: str = "generic"):
        from rfdetr import RFDETR, RFDETRLarge

        if profile not in PROFILES:
            raise ValueError(f"Unknown detector profile: {profile!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("RF-DETR-L requires the configured NVIDIA CUDA environment.")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        config = PROFILES[profile]
        self.model_path = Path(model_path or MODEL_DIR / config["filename"]).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Local detector checkpoint missing: {self.model_path}")
        digest = sha256_file(self.model_path)
        if digest != config["sha256"]:
            raise ValueError(f"Checkpoint SHA256 does not match detector profile {profile!r}")
        torch.set_num_threads(2)
        self.confidence = float(confidence)
        self.device = torch.cuda.get_device_name(0)
        self.profile = profile
        self.input_size = 704
        self.antialias = config["antialias"]
        if profile == "generic":
            self.model = RFDETRLarge(pretrain_weights=str(self.model_path), device="cuda")
        else:
            # The verified local EMA export is already person0. Preserve its
            # trained head rather than reconstructing a 90-class COCO model.
            self.model = RFDETR.from_checkpoint(str(self.model_path), device="cuda", trust_checkpoint=True)
        mc = self.model.model_config
        # Official constructor configs may leave the optional model_name unset;
        # the pinned checkpoint hash and explicit Large constructor identify it.
        valid_names = (None, "RFDETRLarge") if profile == "generic" else ("RFDETRLarge",)
        if mc.model_name not in valid_names or mc.num_classes != config["num_classes"] or mc.resolution != 704:
            raise ValueError(f"Unexpected Large/class/resolution metadata for {profile!r}: "
                             f"{mc.model_name!r}/{mc.num_classes}/{mc.resolution}")
        mapping = self.model._prediction_context().class_id_to_name
        self.person_ids = [int(i) for i, name in mapping.items() if name == "person"]
        if self.person_ids != [config["person_id"]]:
            raise ValueError(f"Unexpected person class mapping for {profile!r}: {self.person_ids}")
        self.backend = f"RF-DETR-L / {config['label']} / PyTorch CUDA"
        self.info = {
            "profile": profile, "label": config["label"], "architecture": "RFDETRLarge",
            "weights": self.model_path.name, "sha256": digest,
            "inputSize": [704, 704], "antialias": self.antialias,
            "personClassId": self.person_ids[0], "confidence": self.confidence,
        }

    def detect(self, frame: np.ndarray) -> np.ndarray:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or not frame.size or min(frame.shape[:2]) < 2):
            raise ValueError("Expected a nonempty uint8 BGR frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            result = self.model.predict(rgb, threshold=self.confidence,
                                        shape=(self.input_size, self.input_size),
                                        antialias=self.antialias, include_source_image=False)
        if not len(result):
            return np.empty((0, 6), dtype=np.float32)
        keep = np.isin(result.class_id, self.person_ids)
        boxes = np.asarray(result.xyxy[keep], dtype=np.float32)
        scores = np.asarray(result.confidence[keep], dtype=np.float32)
        if not len(boxes):
            return np.empty((0, 6), dtype=np.float32)
        # Existing pose/occupancy callers receive one external person0 label.
        return np.column_stack((boxes, scores, np.zeros(len(boxes), np.float32))).astype(np.float32)
