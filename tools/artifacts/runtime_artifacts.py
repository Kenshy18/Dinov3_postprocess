#!/usr/bin/env python3
"""Shared runtime artifact layout definitions.

Keep artifact names, local destinations, and accepted source layouts in one
place so setup, download, and verification tools cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SourceSpec = str | tuple[str, ...]
Mapping = tuple[SourceSpec, str]


def env_dest(name: str, default: str) -> str:
    return os.environ.get(name, default)


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
        "checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine",
    ),
    ("checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine",),
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
    CODINO_CONFIG.mapping,
    CODINO_CHECKPOINT.mapping,
    CODINO_CLASSIFIER.mapping,
    *POSTPROCESS_MAPPINGS,
)
TRT_MAPPINGS: tuple[Mapping, ...] = (
    TRT_BACKBONE.mapping,
    CODINO_TRT_BACKBONE.mapping,
    CODINO_TRT_QUERY_ENCODER.mapping,
    CODINO_TRT_DECODER.mapping,
    CODINO_TRT_MASK_HEAD.mapping,
)
