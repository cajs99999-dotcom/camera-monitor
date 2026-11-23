"""Small shared detector settings, without importing a GPU framework."""
import json
import math
from pathlib import Path

CONFIG=Path(__file__).resolve().parents[1]/"detector_config.json"


def load_detection_settings(path=CONFIG):
    settings=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(settings,dict) or set(settings)-{"profile","confidence"}:
        raise ValueError("detector_config.json only accepts profile and optional confidence")
    profile=settings.get("profile")
    if profile not in {"classroom_v1","generic"}:
        raise ValueError("Unknown detector profile")
    confidence=settings.get("confidence",.35)
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<confidence<1:
        raise ValueError("Detection confidence must be a finite number between 0 and 1")
    return {"profile":profile,"confidence":float(confidence)}
