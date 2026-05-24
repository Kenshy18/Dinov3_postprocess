#!/usr/bin/env python3
"""Shared defaults for integrated pipeline entrypoints."""

from __future__ import annotations

from pathlib import Path

from .model_registry import detector_choices, detector_spec


SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUNTIME_PROFILE = INTEGRATION_ROOT / ".runtime" / "runtime_profile.json"
LEGACY_RUNTIME_PROFILE = INTEGRATION_ROOT / "configs" / "runtime_profile.json"
DETECTOR_CHOICES = detector_choices()
DINOV3_SPEC = detector_spec("dinov3")
EVA02_SPEC = detector_spec("eva02")
CODINO_SPEC = detector_spec("codino")


DEFAULT_DINOV3_RUNTIME = DINOV3_SPEC.runtime_dir
DEFAULT_EVA02_RUNTIME = EVA02_SPEC.runtime_dir
DEFAULT_CODINO_RUNTIME = CODINO_SPEC.runtime_dir
LOCAL_ATOSYORI_REPO = INTEGRATION_ROOT / "external" / "atosyori-pipeline-dev"
DEFAULT_MODEL_ROOT = INTEGRATION_ROOT / "checkpoints" / "postprocess"
DEFAULT_DETECTOR_CHECKPOINT = DINOV3_SPEC.artifact("detector_checkpoint")
DEFAULT_EVA02_DETECTOR_CHECKPOINT = EVA02_SPEC.artifact("detector_checkpoint")
DEFAULT_DINOV3_WEIGHTS = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "dinov3"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_CLASSIFIER_CHECKPOINT = DINOV3_SPEC.artifact("classifier_checkpoint")
DEFAULT_EVA02_CLASSIFIER_CHECKPOINT = EVA02_SPEC.artifact("classifier_checkpoint")
LOCAL_CODINO_DETECTOR_DIR = INTEGRATION_ROOT / "checkpoints" / "codino" / "detector"
LOCAL_CODINO_CLASSIFIER_DIR = INTEGRATION_ROOT / "checkpoints" / "codino" / "classifier"
LOCAL_CODINO_TRT_DIR = INTEGRATION_ROOT / "checkpoints" / "codino" / "trt"
DEFAULT_CODINO_RUNTIME_SCRIPT = CODINO_SPEC.artifact("runtime_script")
DEFAULT_CODINO_CONFIG = CODINO_SPEC.artifact("config")
DEFAULT_CODINO_CHECKPOINT = CODINO_SPEC.artifact("checkpoint")
DEFAULT_CODINO_CLASSIFIER_CHECKPOINT = CODINO_SPEC.artifact("classifier_checkpoint")
FALLBACK_TRT_BACKBONE_ENGINE = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "trt"
    / "dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine"
)
DEFAULT_TRT_BACKBONE_ENGINE = DINOV3_SPEC.artifact("trt_backbone_engine")
DEFAULT_CODINO_TRT_FEATURE_ENGINE = CODINO_SPEC.artifact("trt_feature_engine")
DEFAULT_CODINO_TRT_BACKBONE_ENGINE = CODINO_SPEC.artifact("trt_backbone_engine")
DEFAULT_CODINO_TRT_QUERY_ENCODER_ENGINE = CODINO_SPEC.artifact("trt_query_encoder_engine")
DEFAULT_CODINO_TRT_DECODER_ENGINE = CODINO_SPEC.artifact("trt_decoder_engine")
DEFAULT_CODINO_TRT_MASK_HEAD_ENGINE = CODINO_SPEC.artifact("trt_mask_head_engine")
DEFAULT_CODINO_TRT_EXTRA_SITE_PACKAGES = CODINO_SPEC.artifact("trt_extra_site_packages")
DEFAULT_POLICY = INTEGRATION_ROOT / "configs" / "class_policy_default.json"

DINO_DEFAULT_WARMUP_FRAMES = DINOV3_SPEC.default("warmup_frames")
DINO_DEFAULT_BATCH_SIZE = DINOV3_SPEC.default("batch_size")

EVA02_DEFAULT_TARGET_SIZE = EVA02_SPEC.default("target_size")
EVA02_DEFAULT_SCORE_THRESH = EVA02_SPEC.default("score_thresh")
EVA02_DEFAULT_NMS_THRESH = EVA02_SPEC.default("nms_thresh")
EVA02_DEFAULT_TOPK = EVA02_SPEC.default("topk")
EVA02_DEFAULT_BATCH_SIZE = EVA02_SPEC.default("batch_size")
EVA02_DEFAULT_WARMUP_FRAMES = EVA02_SPEC.default("warmup_frames")
EVA02_DEFAULT_CLASSIFIER_BATCH_SIZE = EVA02_SPEC.default("classifier_batch_size")
EVA02_DEFAULT_JSON_BACKEND = EVA02_SPEC.default("json_backend")
EVA02_DEFAULT_MASK_APPROX = EVA02_SPEC.default("mask_approx")
EVA02_DEFAULT_ASYNC_WRITER = EVA02_SPEC.default("async_writer")

CODINO_DEFAULT_TARGET_SIZE = CODINO_SPEC.default("target_size")
CODINO_DEFAULT_SCORE_THRESH = CODINO_SPEC.default("score_thresh")
CODINO_DEFAULT_MODEL_SCORE_THR = CODINO_SPEC.default("model_score_thr")
CODINO_DEFAULT_BATCH_SIZE = CODINO_SPEC.default("batch_size")
CODINO_DEFAULT_WARMUP_FRAMES = CODINO_SPEC.default("warmup_frames")
CODINO_DEFAULT_JSON_BACKEND = CODINO_SPEC.default("json_backend")
CODINO_DEFAULT_MASK_APPROX = CODINO_SPEC.default("mask_approx")
CODINO_DEFAULT_ASYNC_WRITER = CODINO_SPEC.default("async_writer")
CODINO_DEFAULT_AMP = CODINO_SPEC.default("amp")
CODINO_DEFAULT_TF32 = CODINO_SPEC.default("tf32")
CODINO_DEFAULT_DISABLE_MASK_IOU_HEAD = CODINO_SPEC.default("disable_mask_iou_head")
CODINO_DEFAULT_TRT_FEATURE_NAMES = CODINO_SPEC.default("trt_feature_names")
CODINO_DEFAULT_TRT_QUERY_ENCODER_SHAPES = CODINO_SPEC.default("trt_query_encoder_shapes")
