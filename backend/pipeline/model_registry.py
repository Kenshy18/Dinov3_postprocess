"""Detector runtime, artifact, and speed-profile registry.

This module is intentionally small and data-oriented. New detector families
should add one registry entry here instead of scattering paths across CLI, UI,
and runtime defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_PROFILE = ROOT / ".runtime" / "runtime_profile.json"
LEGACY_RUNTIME_PROFILE = ROOT / "configs" / "runtime_profile.json"


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    label: str
    runtime_dir: Path
    artifacts: dict[str, Path | None]
    defaults: dict[str, Any]

    def artifact(self, key: str) -> Path | None:
        return self.artifacts.get(key)

    def default(self, key: str) -> Any:
        return self.defaults[key]


def _runtime_profile_path() -> Path:
    raw = os.environ.get("DINOV3_RUNTIME_PROFILE")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return DEFAULT_RUNTIME_PROFILE if DEFAULT_RUNTIME_PROFILE.is_file() else LEGACY_RUNTIME_PROFILE


def _runtime_profile() -> dict[str, Any]:
    path = _runtime_profile_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _profile_value(section: str, key: str) -> Any:
    rec = _runtime_profile().get("recommendations", {})
    if not isinstance(rec, dict):
        return None
    section_data = rec.get(section, {})
    if not isinstance(section_data, dict):
        return None
    return section_data.get(key)


def _profile_str(section: str, key: str, default: str) -> str:
    value = _profile_value(section, key)
    return str(value) if value is not None else default


def _profile_int(section: str, key: str, default: int) -> int:
    try:
        value = _profile_value(section, key)
        return int(value) if value is not None else default
    except Exception:
        return default


def _profile_float(section: str, key: str, default: float) -> float:
    try:
        value = _profile_value(section, key)
        return float(value) if value is not None else default
    except Exception:
        return default


def _profile_bool(section: str, key: str, default: bool) -> bool:
    value = _profile_value(section, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value) if value is not None else default


def _profile_path(section: str, key: str, default: Path | None) -> Path | None:
    value = _profile_value(section, key)
    if value:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else ROOT / path
    return default


def _local(path: str) -> Path:
    return ROOT / path


def _default_trt_site_packages() -> Path | None:
    raw = os.environ.get("TENSORRT_SITE_PACKAGES")
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    return None


def _build_registry() -> dict[str, DetectorSpec]:
    codino_trt_feature = _profile_path("codino", "trt_feature_engine", None)
    codino_trt_backbone = None if codino_trt_feature is not None else _profile_path(
        "codino",
        "trt_backbone_engine",
        _local("checkpoints/codino/trt/codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine"),
    )
    return {
        "dinov3": DetectorSpec(
            name="dinov3",
            label="DINOv3 Cascade Mask R-CNN",
            runtime_dir=_local("backend/detectors/dinov3/runtime"),
            artifacts={
                "detector_checkpoint": _local("checkpoints/detector/model_final.pth"),
                "classifier_checkpoint": _local("checkpoints/classifier/best.pt"),
                "trt_backbone_engine": _profile_path(
                    "dinov3",
                    "trt_backbone_engine",
                    _local("checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"),
                ),
            },
            defaults={
                "batch_size": _profile_int("dinov3", "batch_size", 8),
                "warmup_frames": _profile_int("dinov3", "warmup_frames", 300),
            },
        ),
        "eva02": DetectorSpec(
            name="eva02",
            label="EVA02 Cascade Mask R-CNN",
            runtime_dir=_local("backend/detectors/eva02/runtime"),
            artifacts={
                "detector_checkpoint": _local("checkpoints/eva02/detector/model_final.pth"),
                "classifier_checkpoint": _local("checkpoints/eva02/classifier/best.pt"),
            },
            defaults={
                "target_size": 1280,
                "score_thresh": 0.1,
                "nms_thresh": 0.5,
                "topk": 80,
                "batch_size": _profile_int("eva02", "batch_size", 1),
                "warmup_frames": _profile_int("eva02", "warmup_frames", 0),
                "classifier_batch_size": _profile_int("eva02", "classifier_batch_size", 1024),
                "json_backend": "json",
                "mask_approx": "simple",
                "async_writer": False,
            },
        ),
        "codino": DetectorSpec(
            name="codino",
            label="DINOv3 Co-DINO",
            runtime_dir=_local("backend/detectors/codino/runtime"),
            artifacts={
                "runtime_script": _profile_path(
                    "codino",
                    "runtime_script",
                    _local("backend/detectors/codino/runtime/codino_video_fast_runtime.py"),
                ),
                "config": _profile_path("codino", "config", _local("checkpoints/codino/detector/resolved_config.py")),
                "checkpoint": _profile_path("codino", "checkpoint", _local("checkpoints/codino/detector/epoch_2.pth")),
                "classifier_checkpoint": _profile_path(
                    "codino",
                    "classifier_checkpoint",
                    _local("checkpoints/codino/classifier/best.pt"),
                ),
                "trt_feature_engine": codino_trt_feature,
                "trt_backbone_engine": codino_trt_backbone,
                "trt_query_encoder_engine": _profile_path(
                    "codino",
                    "trt_query_encoder_engine",
                    _local("checkpoints/codino/trt/codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine"),
                ),
                "trt_decoder_engine": _profile_path(
                    "codino",
                    "trt_decoder_engine",
                    _local("checkpoints/codino/trt/codino_decoder_b2_736x1280_msda_plugin_fp16.engine"),
                ),
                "trt_mask_head_engine": _profile_path(
                    "codino",
                    "trt_mask_head_engine",
                    _local("checkpoints/codino/trt/codino_mask_head_core_n1_736x1280_fp16.engine"),
                ),
                "trt_extra_site_packages": _profile_path("codino", "trt_extra_site_packages", _default_trt_site_packages()),
            },
            defaults={
                "target_size": _profile_str("codino", "target_size", "1280x720"),
                "score_thresh": _profile_float("codino", "score_thresh", 0.30),
                "model_score_thr": _profile_float("codino", "model_score_thr", 0.05),
                "batch_size": _profile_int("codino", "batch_size", 2),
                "warmup_frames": _profile_int("codino", "warmup_frames", 60),
                "json_backend": _profile_str("codino", "json_backend", "orjson"),
                "mask_approx": _profile_str("codino", "mask_approx", "none"),
                "async_writer": _profile_bool("codino", "async_writer", True),
                "amp": _profile_str("codino", "amp", "fp16"),
                "tf32": _profile_bool("codino", "tf32", True),
                "disable_mask_iou_head": _profile_bool("codino", "disable_mask_iou_head", True),
                "trt_feature_names": _profile_str("codino", "trt_feature_names", "feat0,feat1,feat2,feat3,feat4"),
                "trt_query_encoder_shapes": _profile_str(
                    "codino",
                    "trt_query_encoder_shapes",
                    "184x320,92x160,46x80,23x40,12x20",
                ),
            },
        ),
    }


REGISTRY = _build_registry()


def detector_choices() -> tuple[str, ...]:
    return tuple(REGISTRY)


def detector_spec(name: str) -> DetectorSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown detector: {name}") from exc
