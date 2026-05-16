#!/usr/bin/env python3
"""Shared defaults for integrated pipeline entrypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUNTIME_PROFILE = INTEGRATION_ROOT / ".runtime" / "runtime_profile.json"
LEGACY_RUNTIME_PROFILE = INTEGRATION_ROOT / "configs" / "runtime_profile.json"


def _default_runtime_profile_path() -> Path:
    raw = os.environ.get("DINOV3_RUNTIME_PROFILE")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else INTEGRATION_ROOT / path
    return DEFAULT_RUNTIME_PROFILE if DEFAULT_RUNTIME_PROFILE.is_file() else LEGACY_RUNTIME_PROFILE


RUNTIME_PROFILE = _default_runtime_profile_path()


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


def _profile_float(section: str, key: str, default: float) -> float:
    try:
        value = _runtime_profile().get("recommendations", {}).get(section, {}).get(key)
        return float(value) if value is not None else default
    except Exception:
        return default


def _profile_str(section: str, key: str, default: str) -> str:
    try:
        value = _runtime_profile().get("recommendations", {}).get(section, {}).get(key)
        return str(value) if value is not None else default
    except Exception:
        return default


def _profile_bool(section: str, key: str, default: bool) -> bool:
    try:
        value = _runtime_profile().get("recommendations", {}).get(section, {}).get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value) if value is not None else default
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


def _profile_optional_path(section: str, key: str) -> Path | None:
    try:
        value = _runtime_profile().get("recommendations", {}).get(section, {}).get(key)
        if value:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else INTEGRATION_ROOT / path
    except Exception:
        pass
    return None


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else INTEGRATION_ROOT / path


def _prefer_existing(local: Path, fallback: Path) -> Path:
    return local if local.is_file() else fallback


DEFAULT_DINOV3_RUNTIME = INTEGRATION_ROOT / "backend" / "detectors" / "dinov3" / "runtime"
DEFAULT_EVA02_RUNTIME = INTEGRATION_ROOT / "backend" / "detectors" / "eva02" / "runtime"
DEFAULT_CODINO_RUNTIME = INTEGRATION_ROOT / "backend" / "detectors" / "codino" / "runtime"
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
CODINO_EXTERNAL_ROOT = _env_path(
    "CODINO_EXTERNAL_ROOT",
    Path("/home/kenke/workspace/CV/unified_training_codino_eva02"),
)
CODINO_EXTERNAL_CODINO_ROOT = _env_path("CODINO_ROOT", CODINO_EXTERNAL_ROOT / "codino")
CODINO_EXTERNAL_RUN_DIR = (
    CODINO_EXTERNAL_CODINO_ROOT
    / "work_dirs"
    / "dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019"
)
CODINO_EXTERNAL_TWO_STAGE_ROOT = (
    CODINO_EXTERNAL_ROOT / "inference" / "dinov3_cascade_unified" / "two_stage_multiclass_20260426"
)
LOCAL_CODINO_DETECTOR_DIR = INTEGRATION_ROOT / "checkpoints" / "codino" / "detector"
LOCAL_CODINO_CLASSIFIER_DIR = INTEGRATION_ROOT / "checkpoints" / "codino" / "classifier"
LOCAL_CODINO_TRT_DIR = INTEGRATION_ROOT / "checkpoints" / "codino" / "trt"
DEFAULT_CODINO_RUNTIME_SCRIPT = _profile_path(
    "codino",
    "runtime_script",
    CODINO_EXTERNAL_CODINO_ROOT / "tools" / "infer_dinov3_codino_video_fast.py",
)
DEFAULT_CODINO_CONFIG = _profile_path(
    "codino",
    "config",
    _prefer_existing(LOCAL_CODINO_DETECTOR_DIR / "resolved_config.py", CODINO_EXTERNAL_RUN_DIR / "resolved_config.py"),
)
DEFAULT_CODINO_CHECKPOINT = _profile_path(
    "codino",
    "checkpoint",
    _prefer_existing(LOCAL_CODINO_DETECTOR_DIR / "epoch_2.pth", CODINO_EXTERNAL_RUN_DIR / "epoch_2.pth"),
)
DEFAULT_CODINO_CLASSIFIER_CHECKPOINT = _profile_path(
    "codino",
    "classifier_checkpoint",
    _prefer_existing(
        LOCAL_CODINO_CLASSIFIER_DIR / "best.pt",
        CODINO_EXTERNAL_TWO_STAGE_ROOT
        / "outputs"
        / "roi_classifier_codino_maskroi_spatial_gap_full_fp16"
        / "run_20260516_041219"
        / "checkpoints"
        / "best.pt",
    ),
)
FALLBACK_TRT_BACKBONE_ENGINE = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "trt"
    / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"
)
DEFAULT_TRT_BACKBONE_ENGINE = _profile_path("dinov3", "trt_backbone_engine", FALLBACK_TRT_BACKBONE_ENGINE)
DEFAULT_CODINO_TRT_FEATURE_ENGINE = _profile_optional_path("codino", "trt_feature_engine")
DEFAULT_CODINO_TRT_BACKBONE_ENGINE = None if DEFAULT_CODINO_TRT_FEATURE_ENGINE is not None else _profile_path(
    "codino",
    "trt_backbone_engine",
    _prefer_existing(
        LOCAL_CODINO_TRT_DIR / "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",
        CODINO_EXTERNAL_ROOT / "outputs" / "trt" / "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",
    ),
)
DEFAULT_CODINO_TRT_QUERY_ENCODER_ENGINE = _profile_path(
    "codino",
    "trt_query_encoder_engine",
    _prefer_existing(
        LOCAL_CODINO_TRT_DIR / "codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",
        CODINO_EXTERNAL_ROOT / "outputs" / "trt" / "codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",
    ),
)
DEFAULT_CODINO_TRT_DECODER_ENGINE = _profile_path(
    "codino",
    "trt_decoder_engine",
    _prefer_existing(
        LOCAL_CODINO_TRT_DIR / "codino_decoder_b2_736x1280_msda_plugin_fp16.engine",
        CODINO_EXTERNAL_ROOT / "outputs" / "trt" / "codino_decoder_b2_736x1280_msda_plugin_fp16.engine",
    ),
)
DEFAULT_CODINO_TRT_MASK_HEAD_ENGINE = _profile_path(
    "codino",
    "trt_mask_head_engine",
    _prefer_existing(
        LOCAL_CODINO_TRT_DIR / "codino_mask_head_core_n1_736x1280_fp16.engine",
        CODINO_EXTERNAL_ROOT / "outputs" / "trt" / "codino_mask_head_core_n1_736x1280_fp16.engine",
    ),
)
DEFAULT_CODINO_TRT_EXTRA_SITE_PACKAGES = _profile_path(
    "codino",
    "trt_extra_site_packages",
    CODINO_EXTERNAL_ROOT / "inference" / "eva02_cascade_experimental" / "venv" / "lib" / "python3.10" / "site-packages",
)
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

CODINO_DEFAULT_TARGET_SIZE = _profile_str("codino", "target_size", "1280x720")
CODINO_DEFAULT_SCORE_THRESH = _profile_float("codino", "score_thresh", 0.30)
CODINO_DEFAULT_MODEL_SCORE_THR = _profile_float("codino", "model_score_thr", 0.05)
CODINO_DEFAULT_BATCH_SIZE = _profile_int("codino", "batch_size", 2)
CODINO_DEFAULT_WARMUP_FRAMES = _profile_int("codino", "warmup_frames", 60)
CODINO_DEFAULT_JSON_BACKEND = _profile_str("codino", "json_backend", "orjson")
CODINO_DEFAULT_MASK_APPROX = _profile_str("codino", "mask_approx", "none")
CODINO_DEFAULT_ASYNC_WRITER = _profile_bool("codino", "async_writer", True)
CODINO_DEFAULT_AMP = _profile_str("codino", "amp", "fp16")
CODINO_DEFAULT_TF32 = _profile_bool("codino", "tf32", True)
CODINO_DEFAULT_DISABLE_MASK_IOU_HEAD = _profile_bool("codino", "disable_mask_iou_head", True)
CODINO_DEFAULT_TRT_QUERY_ENCODER_SHAPES = _profile_str(
    "codino",
    "trt_query_encoder_shapes",
    "184x320,92x160,46x80,23x40,12x20",
)
