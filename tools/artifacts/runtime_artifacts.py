#!/usr/bin/env python3
"""Shared runtime artifact layout definitions.

Keep artifact names, local destinations, and accepted source layouts in one
place so setup, download, and verification tools cannot drift apart.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SourceSpec = str | tuple[str, ...]
Mapping = tuple[SourceSpec, str]
_RUNTIME_ENV_CACHE: dict[str, str] | None = None


def runtime_env_values() -> dict[str, str]:
    global _RUNTIME_ENV_CACHE
    if _RUNTIME_ENV_CACHE is not None:
        return _RUNTIME_ENV_CACHE
    values: dict[str, str] = {}
    env_file = ROOT / ".runtime" / "gui_runtime.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            try:
                parts = shlex.split(stripped, comments=True, posix=True)
            except ValueError:
                continue
            if len(parts) != 1 or "=" not in parts[0]:
                continue
            key, value = parts[0].split("=", 1)
            values[key] = value
    _RUNTIME_ENV_CACHE = values
    return values


def env_dest(name: str, default: str) -> str:
    return os.environ.get(name) or runtime_env_values().get(name) or default


@dataclass(frozen=True)
class RuntimeArtifact:
    name: str
    dest: str
    sources: tuple[str, ...]

    @property
    def path(self) -> Path:
        return ROOT / self.dest

    @property
    def source_spec(self) -> SourceSpec:
        if len(self.sources) == 1:
            return self.sources[0]
        return self.sources

    @property
    def mapping(self) -> Mapping:
        return (self.source_spec, self.dest)


DETECTOR = RuntimeArtifact(
    "detector checkpoint",
    "checkpoints/detector/model_final.pth",
    (
        "checkpoints/dinov3/detector/model_final.pth",
        "checkpoints/detector/model_final.pth",
    ),
)
DINO_WEIGHTS = RuntimeArtifact(
    "DINOv3 pretrained weights",
    "checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    ("checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",),
)
CLASSIFIER = RuntimeArtifact(
    "ROI classifier checkpoint",
    "checkpoints/classifier/best.pt",
    (
        "checkpoints/dinov3/classifier/best.pt",
        "checkpoints/classifier/best.pt",
    ),
)
EVA02_DETECTOR = RuntimeArtifact(
    "EVA02 detector checkpoint",
    "checkpoints/eva02/detector/model_final.pth",
    (
        "checkpoints/Eva02/detector/model_final.pth",
        "checkpoints/eva02/detector/model_final.pth",
    ),
)
EVA02_CLASSIFIER = RuntimeArtifact(
    "EVA02 ROI classifier checkpoint",
    "checkpoints/eva02/classifier/best.pt",
    (
        "checkpoints/Eva02/classifier/best.pt",
        "checkpoints/eva02/classifier/best.pt",
    ),
)
TRT_BACKBONE = RuntimeArtifact(
    "TensorRT backbone engine",
    env_dest(
        "DINOV3_TRT_BACKBONE_ENGINE",
        "checkpoints/trt/dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine",
    ),
    ("checkpoints/trt/dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine",),
)
CODINO_CONFIG = RuntimeArtifact(
    "Co-DINO config",
    "checkpoints/codino/detector/resolved_config.py",
    (
        "checkpoints/codino/detector/resolved_config.py",
        "checkpoints/CO-DINO/detector/resolved_config.py",
    ),
)
CODINO_CHECKPOINT = RuntimeArtifact(
    "Co-DINO detector checkpoint",
    "checkpoints/codino/detector/epoch_2.pth",
    (
        "checkpoints/codino/detector/epoch_2.pth",
        "checkpoints/CO-DINO/detector/epoch_2.pth",
    ),
)
CODINO_CLASSIFIER = RuntimeArtifact(
    "Co-DINO ROI classifier checkpoint",
    "checkpoints/codino/classifier/best.pt",
    (
        "checkpoints/codino/classifier/best.pt",
        "checkpoints/CO-DINO/classifier/best.pt",
    ),
)
RTDETR_CHECKPOINT = RuntimeArtifact(
    "RT-DETR Head/Face checkpoint",
    "checkpoints/rtdetr/head_face_best_stg1.pth",
    (
        "checkpoints/rtdetr/head_face_best_stg1.pth",
        "checkpoints/RT-DETR/head_face_best_stg1.pth",
        "checkpoints/RT-DETR/HeadFace/head_face_best_stg1.pth",
        "rtdetr/head_face_best_stg1.pth",
        "RT-DETR/head_face_best_stg1.pth",
        "outputs/pth/rtv2-r18-vhf-512x896-80e-bs16-20260518-010638/best_stg1.pth",
    ),
)
CODINO_TRT_BACKBONE = RuntimeArtifact(
    "Co-DINO TensorRT DINOv3 backbone engine",
    env_dest(
        "CODINO_TRT_BACKBONE_ENGINE",
        "checkpoints/codino/trt/codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",
    ),
    ("checkpoints/codino/trt/codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",),
)
CODINO_TRT_QUERY_ENCODER = RuntimeArtifact(
    "Co-DINO TensorRT query encoder engine",
    env_dest(
        "CODINO_TRT_QUERY_ENCODER_ENGINE",
        "checkpoints/codino/trt/codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",
    ),
    ("checkpoints/codino/trt/codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",),
)
CODINO_TRT_DECODER = RuntimeArtifact(
    "Co-DINO TensorRT decoder engine",
    env_dest(
        "CODINO_TRT_DECODER_ENGINE",
        "checkpoints/codino/trt/codino_decoder_b2_736x1280_msda_plugin_fp16.engine",
    ),
    ("checkpoints/codino/trt/codino_decoder_b2_736x1280_msda_plugin_fp16.engine",),
)
CODINO_TRT_MASK_HEAD = RuntimeArtifact(
    "Co-DINO TensorRT mask head engine",
    env_dest(
        "CODINO_TRT_MASK_HEAD_ENGINE",
        "checkpoints/codino/trt/codino_mask_head_core_n1_736x1280_fp16.engine",
    ),
    ("checkpoints/codino/trt/codino_mask_head_core_n1_736x1280_fp16.engine",),
)
POSTPROCESS_K2 = RuntimeArtifact(
    "postprocess K2 checkpoint",
    "checkpoints/postprocess/k2_v5/best_exact.pt",
    ("checkpoints/postprocess/k2_v5/best_exact.pt",),
)
POSTPROCESS_POLYGON = RuntimeArtifact(
    "postprocess polygon checkpoint",
    "checkpoints/postprocess/polygon_point_predictor/best.pt",
    ("checkpoints/postprocess/polygon_point_predictor/best.pt",),
)
POSTPROCESS_POLYGON_STATS = RuntimeArtifact(
    "postprocess polygon feature stats",
    "checkpoints/postprocess/polygon_point_predictor/feature_stats.npz",
    ("checkpoints/postprocess/polygon_point_predictor/feature_stats.npz",),
)

PORTABLE_REQUIRED_ARTIFACTS = (
    DETECTOR,
    DINO_WEIGHTS,
    CLASSIFIER,
    EVA02_DETECTOR,
    EVA02_CLASSIFIER,
    CODINO_CONFIG,
    CODINO_CHECKPOINT,
    CODINO_CLASSIFIER,
    RTDETR_CHECKPOINT,
    POSTPROCESS_K2,
    POSTPROCESS_POLYGON,
    POSTPROCESS_POLYGON_STATS,
)
TRT_ARTIFACTS = (
    TRT_BACKBONE,
    CODINO_TRT_BACKBONE,
    CODINO_TRT_QUERY_ENCODER,
    CODINO_TRT_DECODER,
    CODINO_TRT_MASK_HEAD,
)
REQUIRED_ARTIFACTS = PORTABLE_REQUIRED_ARTIFACTS
ALL_RUNTIME_ARTIFACTS = (*PORTABLE_REQUIRED_ARTIFACTS, *TRT_ARTIFACTS)

DETECTRON2_EXTENSION_ROOT = ROOT / "eva02" / "eva02_det" / "detectron2"
RTDETR_SOURCE_ROOT = ROOT / "external" / "RT-DETR" / "RT-DETRv4"
RTDETR_RUNTIME_SCRIPT = RTDETR_SOURCE_ROOT / "tools" / "inference" / "video_sqlite_inf.py"
RTDETR_DEFAULT_CONFIG = RTDETR_SOURCE_ROOT / "configs" / "rtv2" / "rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml"
RTDETR_DEFAULT_CHECKPOINT = RTDETR_CHECKPOINT.path

POSTPROCESS_MAPPINGS: tuple[Mapping, ...] = (
    ("k2_v5/best_exact.pt", "checkpoints/postprocess/k2_v5/best_exact.pt"),
    ("k2_v5/run_config.json", "checkpoints/postprocess/k2_v5/run_config.json"),
    (
        "k2_v5/train_k2_slot_set_spd_standalone_v5.py",
        "checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py",
    ),
    ("polygon_point_predictor/best.pt", "checkpoints/postprocess/polygon_point_predictor/best.pt"),
    (
        "polygon_point_predictor/feature_stats.npz",
        "checkpoints/postprocess/polygon_point_predictor/feature_stats.npz",
    ),
    (
        "polygon_point_predictor/run_config.json",
        "checkpoints/postprocess/polygon_point_predictor/run_config.json",
    ),
    (
        "polygon_point_predictor/train_mask_point_predictor.py",
        "checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py",
    ),
)

DINOV3_MAPPINGS: tuple[Mapping, ...] = (
    ("detector/model_final.pth", DETECTOR.dest),
    (
        (
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        ),
        DINO_WEIGHTS.dest,
    ),
    ("classifier/best.pt", CLASSIFIER.dest),
)

RUNTIME_MAPPINGS: tuple[Mapping, ...] = (
    DETECTOR.mapping,
    DINO_WEIGHTS.mapping,
    CLASSIFIER.mapping,
    EVA02_DETECTOR.mapping,
    EVA02_CLASSIFIER.mapping,
    CODINO_CHECKPOINT.mapping,
    CODINO_CLASSIFIER.mapping,
    RTDETR_CHECKPOINT.mapping,
    *POSTPROCESS_MAPPINGS,
)
TRT_MAPPINGS: tuple[Mapping, ...] = (
    TRT_BACKBONE.mapping,
    CODINO_TRT_BACKBONE.mapping,
    CODINO_TRT_QUERY_ENCODER.mapping,
    CODINO_TRT_DECODER.mapping,
    CODINO_TRT_MASK_HEAD.mapping,
)
