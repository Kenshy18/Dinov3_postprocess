"""Small runtime health check."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

from .legacy import engine_exists
from .models import model_status


def _module_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run(model_root: Path | None = None) -> int:
    status = model_status(model_root)
    checks = {
        "legacy_engine": engine_exists(),
        "numpy": _module_ok("numpy"),
        "cv2": _module_ok("cv2"),
        "torch": _module_ok("torch"),
        "orjson": _module_ok("orjson"),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "k2_checkpoint": bool(status["k2_checkpoint"]),
        "polygon_checkpoint": bool(status["polygon_checkpoint"]),
    }
    for name, ok in checks.items():
        print(f"{name}: {'ok' if ok else 'missing'}")
    print(f"model_root: {status['root']}")
    print(f"k2_dir: {status['k2_dir']}")
    print(f"polygon_point_predictor_dir: {status['polygon_point_predictor_dir']}")
    return 0 if all(checks.values()) else 1
