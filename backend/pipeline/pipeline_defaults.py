#!/usr/bin/env python3
"""Shared defaults for integrated pipeline entrypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = SCRIPT_DIR.parents[1]
RUNTIME_PROFILE = Path(os.environ.get("DINOV3_RUNTIME_PROFILE", INTEGRATION_ROOT / "configs" / "runtime_profile.json"))


def _runtime_profile() -> dict:
    try:
        if RUNTIME_PROFILE.is_file():
            return json.loads(RUNTIME_PROFILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _profile_int(section: str, key: str, default: int) -> int:
    try:
        value = _runtime_profile().get("recommendations", {}).get(section, {}).get(key)
        return int(value) if value is not None else default
    except Exception:
        return default


def _profile_path(section: str, key: str, default: Path) -> Path:
    try:
        value = _runtime_profile().get("recommendations", {}).get(section, {}).get(key)
        if value:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else INTEGRATION_ROOT / path
    except Exception:
        pass
    return default


DEFAULT_DINOV3_RUNTIME = INTEGRATION_ROOT / "backend" / "detectors" / "dinov3" / "runtime"
DEFAULT_EVA02_RUNTIME = INTEGRATION_ROOT / "backend" / "detectors" / "eva02" / "runtime"
LOCAL_ATOSYORI_REPO = INTEGRATION_ROOT / "external" / "atosyori-pipeline-dev"
DEFAULT_MODEL_ROOT = INTEGRATION_ROOT / "checkpoints" / "postprocess"
DEFAULT_DETECTOR_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "detector" / "model_final.pth"
DEFAULT_EVA02_DETECTOR_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "eva02" / "detector" / "model_final.pth"
DEFAULT_DINOV3_WEIGHTS = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "dinov3"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_CLASSIFIER_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "classifier" / "best.pt"
DEFAULT_EVA02_CLASSIFIER_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "eva02" / "classifier" / "best.pt"
FALLBACK_TRT_BACKBONE_ENGINE = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "trt"
    / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"
)
DEFAULT_TRT_BACKBONE_ENGINE = _profile_path("dinov3", "trt_backbone_engine", FALLBACK_TRT_BACKBONE_ENGINE)
DEFAULT_POLICY = INTEGRATION_ROOT / "configs" / "class_policy_default.json"

DINO_DEFAULT_WARMUP_FRAMES = _profile_int("dinov3", "warmup_frames", 300)
DINO_DEFAULT_BATCH_SIZE = _profile_int("dinov3", "batch_size", 8)

EVA02_DEFAULT_TARGET_SIZE = 1280
EVA02_DEFAULT_SCORE_THRESH = 0.1
EVA02_DEFAULT_NMS_THRESH = 0.5
EVA02_DEFAULT_TOPK = 80
EVA02_DEFAULT_BATCH_SIZE = _profile_int("eva02", "batch_size", 1)
EVA02_DEFAULT_WARMUP_FRAMES = _profile_int("eva02", "warmup_frames", 0)
EVA02_DEFAULT_CLASSIFIER_BATCH_SIZE = _profile_int("eva02", "classifier_batch_size", 1024)
EVA02_DEFAULT_JSON_BACKEND = "json"
EVA02_DEFAULT_MASK_APPROX = "simple"
EVA02_DEFAULT_ASYNC_WRITER = False
