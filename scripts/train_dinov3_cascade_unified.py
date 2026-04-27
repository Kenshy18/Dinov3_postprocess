#!/usr/bin/env python
"""DINOv3 ViT-L/16 + Cascade Mask R-CNN training script.

train_eva02_cascade_unified.py と同一のパイプライン（FPN, RPN/ATSS, Cascade ROI Heads,
データ拡張, 評価, WandB ログ等）を使い、バックボーンのみを EVA-02 ViT-L から
DINOv3 ViT-L/16 に差し替えたバリアントです。バックボーン以外のコードは
EVA-02 版と完全に一致させ、公平な比較実験を可能にしています。

"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch

# Allowlist OmegaConf classes for torch.load(weights_only=True)
try:  # pragma: no cover - safety config
    from omegaconf import DictConfig, ListConfig  # type: ignore
    from omegaconf.base import ContainerMetadata  # type: ignore
    from typing import Any as TypingAny

    torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, TypingAny])
except Exception:
    pass

# Trust local checkpoints: allow full checkpoint loading (PyTorch 2.6 defaults weights_only=True)
_torch_load_orig = torch.load


def _torch_load_full(*args, **kwargs):  # pragma: no cover - I/O wrapper
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_full


# -----------------------------------------------------------------------------
# Environment preparation
# -----------------------------------------------------------------------------


def suppress_all_warnings() -> None:
    """Disable noisy warnings and lower verbose loggers."""

    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    logging.getLogger("detectron2").setLevel(logging.WARNING)


suppress_all_warnings()

os.environ.setdefault("EVA02_XATTN", "1")
os.environ.setdefault("DINOV3_USE_XFORMERS", "1")


# Detectron2 path setup -------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
LEGACY_REPO_ROOT = Path("/home/kenke/unified_training_codino_eva02")
DEFAULT_SUPERSET_V2_DIR = ROOT_DIR / "datasets" / "0226_gtmask_full_superset_cc_v2"
DEFAULT_SUPERSET_V2_TRAIN_JSON = DEFAULT_SUPERSET_V2_DIR / "annotations_train_cc.json"
DEFAULT_SUPERSET_V2_VAL_JSON = DEFAULT_SUPERSET_V2_DIR / "annotations_val_cc.json"
DEFAULT_INCOMING_ADDED_DIR = ROOT_DIR / "datasets" / "0414_incoming_added_cc_v1"
DEFAULT_INCOMING_ADDED_TRAIN_JSON = DEFAULT_INCOMING_ADDED_DIR / "annotations_train_cc.json"
DEFAULT_INCOMING_ADDED_VAL_JSON = DEFAULT_INCOMING_ADDED_DIR / "annotations_val_cc.json"
DINOV3_PACKAGE_ROOT = ROOT_DIR / "dinov3"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DINOV3_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOV3_PACKAGE_ROOT))
from configs import paths as unified_paths

DEFAULT_EVA02_DET_PATH = str(ROOT_DIR / "eva02" / "eva02_det")
EVA02_DET_PATH = os.environ.get("EVA02_DET_PATH", DEFAULT_EVA02_DET_PATH)
if not Path(EVA02_DET_PATH).is_dir():
    candidate = Path("/home/kenke/EVA02/EVA/EVA-02/det")
    if candidate.is_dir():
        EVA02_DET_PATH = str(candidate)
    else:
        raise FileNotFoundError(
            f"EVA-02 det path not found. Set EVA02_DET_PATH or ensure {candidate} exists."
        )
sys.path.insert(0, EVA02_DET_PATH)
os.chdir(EVA02_DET_PATH)


# Third-party imports after path adjustment ----------------------------------

import cv2  # type: ignore
import detectron2  # type: ignore
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.transforms import Transform
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    TrainerBase,
    hooks,
)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, ema
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.structures import Boxes, Instances
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, EventStorage, EventWriter, JSONWriter, TensorboardXWriter, get_event_storage
from detectron2.utils.visualizer import Visualizer
import detectron2.data.transforms as T


try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)


def get_output_layout(output_dir: str | Path) -> unified_paths.OutputLayout:
    return unified_paths.build_output_layout(output_dir)


def build_event_writers(output_dir: str | Path, max_iter: int) -> list[EventWriter]:
    layout = get_output_layout(output_dir)
    layout.json_dir.mkdir(parents=True, exist_ok=True)
    layout.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return [
        CommonMetricPrinter(max_iter),
        JSONWriter(str(layout.json_dir / "metrics.json")),
        TensorboardXWriter(str(layout.tensorboard_dir)),
    ]


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    train_json: str
    train_dir: str
    val_json: str
    val_dir: str
    class_name: str = "foreground"


@dataclass
class SplitConfig:
    ratio: Optional[float] = None
    seed: int = 42
    sources: Optional[List[str]] = None


@dataclass
class ATSSConfig:
    enabled: bool = True
    topk: int = 80
    center_radius: float = 2.0


@dataclass
class RPNConfig:
    batch_size_per_image: int = 256
    positive_fraction: float = 0.7
    pre_nms_topk_train: int = 20_000
    pre_nms_topk_test: int = 10_000
    post_nms_topk_train: int = 3_000
    post_nms_topk_test: int = 1_500
    nms_thresh: float = 0.9
    min_box_size: float = 0.0


@dataclass
class CascadeConfig:
    iou_thresholds: List[float] = field(default_factory=lambda: [0.45, 0.55, 0.65])
    test_score_thresh: float = 0.01
    test_nms_thresh: float = 0.4
    test_topk_per_image: int = 200


@dataclass
class ROIConfig:
    batch_size_per_image: int = 512
    positive_fraction: float = 0.7
    proposal_append_gt: bool = True


def _unpack_size(size) -> Tuple[int, int]:
    """Unpack a size that can be int (square) or (height, width) sequence.

    Handles plain int, tuple, list, and omegaconf ListConfig.
    """
    if isinstance(size, int):
        return size, size
    # tuple, list, omegaconf.ListConfig, etc.
    try:
        h, w = size[0], size[1]
        return int(h), int(w)
    except (TypeError, IndexError, KeyError):
        v = int(size)
        return v, v


@dataclass
class AugmentationConfig:
    train_size: Union[int, Tuple[int, int]] = 1280
    test_size: Union[int, Tuple[int, int]] = 1280
    use_random_flip: bool = True


@dataclass
class OptimizationConfig:
    epochs: int = 20
    batch_size_per_gpu: int = 2
    gradient_accumulation_steps: int = 1
    base_lr: float = 3e-4
    warmup_epochs: float = 0.5
    cosine_end_value: float = 0.05


@dataclass
class TrainerConfig:
    num_gpus: int = 2
    num_workers: int = 2
    use_amp: bool = True
    use_ema: bool = False
    ddp_timeout_minutes: int = 360
    output_dir: str = "./output_atss_refactored"
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 2
    max_checkpoints_to_keep: int = 5


@dataclass
class LoggingConfig:
    log_period: int = 10
    enable_wandb: bool = True
    wandb_project: str = "EVA02-Cascade_unified"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    log_gradients: bool = False


@dataclass
class VisualizationConfig:
    train_samples: bool = False
    eval_samples: bool = False
    max_images: Optional[int] = None
    vis_num_images: int = 16
    score_thresh: float = 0.3
    image_format: str = "png"
    dump_only: bool = False
    train_dir: Optional[str] = None
    eval_dir: Optional[str] = None


@dataclass
class DebugConfig:
    debug_mode: bool = False
    save_debug_images: bool = True
    max_debug_images: int = 20
    run_letterbox_test: bool = False
    train_samples: int = 500
    val_samples: int = 200
    seed: int = 42


class StableWandBWriter(EventWriter):
    """W&B writer that logs all scalars at the current iteration to avoid step regressions."""

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size

    def write(self) -> None:
        if wandb is None or wandb.run is None:
            return
        storage = get_event_storage()
        step = int(storage.iter) + 1
        payload: Dict[str, float] = {}
        for key, (value, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            if isinstance(value, (int, float, np.floating)):
                payload[key] = float(value)
        if payload:
            wandb.log(payload, step=step)

    def close(self) -> None:
        return


@dataclass
class QuickEvalConfig:
    enabled: bool = True
    train_metrics_samples: int = 400
    val_loss_samples: int = 400
    interval: int = 1
    seed: int = 42
    samples_per_gpu: int = 1
    num_workers: int = 2
    metrics: Tuple[str, ...] = ("bbox", "segm")


@dataclass
class NegativeSamplingConfig:
    enabled: bool = False
    directory: str = "/home/kenke/EVA02/1220_datasets_original/negative_sampling"
    recursive: bool = True
    add_to_val: bool = False
    max_images: Optional[int] = None
    seed: int = 42
    extensions: Tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "webp")


@dataclass
class CheckpointConfig:
    backbone_checkpoint: str = (
        "/home/kenke/EVA02/checkpoints/eva02_L_coco_det_sys_o365.pth"
    )
    use_pretrained: bool = True
    pretrained_path: Optional[str] = (
        "/home/kenke/EVA02/EVA/EVA-02/det/output_atss_5/model_0259025.pth"
    )
    resume_from_latest: bool = False
    resume_path: Optional[str] = None


@dataclass
class TrainingConfig:
    model_size: str = "eva02_L_8attn_1280"
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            train_json="/home/kenke/EVA02/dataset1027_tentative_combined_processed/annotations/annotations_all.json",
            train_dir="/home/kenke/EVA02/dataset1027_tentative_combined",
            val_json="/home/kenke/EVA02/dataset1027_tentative_combined_processed/annotations/annotations_all.json",
            val_dir="/home/kenke/EVA02/dataset1027_tentative_combined",
        )
    )
    split: SplitConfig = field(default_factory=lambda: SplitConfig(ratio=0.93))
    atss: ATSSConfig = field(default_factory=ATSSConfig)
    rpn: RPNConfig = field(default_factory=RPNConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    recall_iou_thrs: Tuple[float, ...] = (
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    )
    recall_max_dets: int = 100
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    quick_eval: QuickEvalConfig = field(default_factory=QuickEvalConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_repo_path(path_value: os.PathLike[str] | str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        repo_relative = (ROOT_DIR / candidate).resolve()
        if repo_relative.exists():
            return repo_relative
        return candidate

    try:
        relative_to_legacy = candidate.relative_to(LEGACY_REPO_ROOT)
    except ValueError:
        return candidate

    migrated = (ROOT_DIR / relative_to_legacy).resolve()
    if migrated.exists():
        logger.info("Resolved legacy repo path: %s -> %s", candidate, migrated)
        return migrated
    return candidate


def normalize_dataset_paths(config: TrainingConfig) -> None:
    missing: list[str] = []
    for attr in ("train_json", "train_dir", "val_json", "val_dir"):
        resolved = resolve_repo_path(getattr(config.dataset, attr))
        setattr(config.dataset, attr, str(resolved))
        if not resolved.exists():
            missing.append(f"{attr}={resolved}")

    if missing:
        joined = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Dataset paths not found after resolution:\n"
            f"{joined}\n"
            "Set UNIFIED_TRAIN_JSON / UNIFIED_TRAIN_DIR / UNIFIED_VAL_JSON / UNIFIED_VAL_DIR as needed."
        )


def read_last_checkpoint(output_dir: Path) -> Optional[Path]:
    last_checkpoint_file = output_dir / "last_checkpoint"
    if not last_checkpoint_file.is_file():
        return None
    checkpoint_name = last_checkpoint_file.read_text().strip()
    if not checkpoint_name:
        return None
    candidate = Path(checkpoint_name)
    if not candidate.is_absolute():
        candidate = output_dir / checkpoint_name
    if candidate.is_file():
        return candidate.resolve()
    return None


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    latest = read_last_checkpoint(output_dir)
    if latest is not None:
        return latest
    if not output_dir.is_dir():
        return None
    candidates: List[Path] = []
    candidates.extend(output_dir.glob("model_*.pth"))
    candidates.extend(output_dir.glob("model_final*.pth"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1].resolve()


def resolve_checkpoint_reference(path_value: os.PathLike[str] | str) -> Optional[Path]:
    candidate = Path(path_value).expanduser().resolve()
    if candidate.is_dir():
        return find_latest_checkpoint(candidate)
    if not candidate.is_file():
        return None
    if candidate.name != "last_checkpoint":
        return candidate

    checkpoint_name = candidate.read_text().strip()
    if not checkpoint_name:
        return None
    resolved = Path(checkpoint_name)
    if not resolved.is_absolute():
        resolved = candidate.parent / checkpoint_name
    if resolved.is_file():
        return resolved.resolve()
    return None


def resolve_checkpoint_settings(config: TrainingConfig, output_dir: Path) -> Tuple[str, bool]:
    checkpoint_path = config.checkpoint.backbone_checkpoint
    if config.checkpoint.use_pretrained and config.checkpoint.pretrained_path:
        candidate = resolve_checkpoint_reference(config.checkpoint.pretrained_path)
        if candidate is not None:
            checkpoint_path = str(candidate)
            logger.info("Using pretrained checkpoint: %s", checkpoint_path)
        else:
            logger.warning(
                "Pretrained checkpoint not found or unresolved at %s, falling back to backbone checkpoint",
                config.checkpoint.pretrained_path,
            )

    resume = False
    resume_candidate: Optional[Path] = None
    if config.checkpoint.resume_path:
        explicit = resolve_checkpoint_reference(config.checkpoint.resume_path)
        if explicit is not None:
            resume_candidate = explicit
        else:
            logger.warning("Resume checkpoint not found or unresolved at %s", config.checkpoint.resume_path)

    if config.checkpoint.resume_from_latest and resume_candidate is None:
        resume_candidate = find_latest_checkpoint(output_dir)
        if resume_candidate is None:
            logger.warning("Resume enabled but no checkpoint found in %s", output_dir)

    if resume_candidate is not None:
        checkpoint_path = str(resume_candidate)
        resume = True
        logger.info("Resuming from checkpoint: %s", checkpoint_path)

    return checkpoint_path, resume


# -----------------------------------------------------------------------------
# ATSS implementation
# -----------------------------------------------------------------------------


@torch.no_grad()
def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def atss_assign_single_image(
    anchors_per_level: List[torch.Tensor],
    gt_boxes: torch.Tensor,
    strides: List[int],
    topk: int = 9,
    center_radius: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = gt_boxes.device
    anchors = torch.cat(anchors_per_level, dim=0)
    total_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]

    labels = anchors.new_full((total_anchors,), 0, dtype=torch.int8)
    matched = anchors.new_full((total_anchors,), -1, dtype=torch.int64)

    if num_gt == 0:
        return labels, matched

    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    ious = pairwise_iou(gt_boxes, anchors)

    level_offsets: List[Tuple[int, int]] = []
    start = 0
    for per_level in anchors_per_level:
        end = start + per_level.shape[0]
        level_offsets.append((start, end))
        start = end

    positives = torch.zeros((num_gt, total_anchors), dtype=torch.bool, device=device)

    for gt_idx in range(num_gt):
        gx = (gt_boxes[gt_idx, 0] + gt_boxes[gt_idx, 2]) * 0.5
        gy = (gt_boxes[gt_idx, 1] + gt_boxes[gt_idx, 3]) * 0.5

        candidates: List[torch.Tensor] = []
        for level, (st, ed) in enumerate(level_offsets):
            if st >= ed:
                continue
            idx = torch.arange(st, ed, device=device)
            dist = (ax[idx] - gx).abs() + (ay[idx] - gy).abs()
            topk_level = min(topk, idx.numel())
            if topk_level == 0:
                continue
            order = dist.topk(k=topk_level, largest=False).indices
            candidates.append(idx[order])

        if not candidates:
            continue

        cand_inds = torch.cat(candidates, dim=0)
        candidate_ious = ious[gt_idx, cand_inds]
        threshold = candidate_ious.mean() + candidate_ious.std()

        if center_radius > 0:
            center_mask = torch.zeros_like(cand_inds, dtype=torch.bool)
            for level, (st, ed) in enumerate(level_offsets):
                level_mask = (cand_inds >= st) & (cand_inds < ed)
                if not level_mask.any():
                    continue
                radius = strides[level] * center_radius
                level_inds = cand_inds[level_mask]
                cx_ok = (ax[level_inds] >= gx - radius) & (ax[level_inds] <= gx + radius)
                cy_ok = (ay[level_inds] >= gy - radius) & (ay[level_inds] <= gy + radius)
                center_mask[level_mask] = cx_ok & cy_ok
        else:
            center_mask = torch.ones_like(cand_inds, dtype=torch.bool)

        pos_inds = cand_inds[(ious[gt_idx, cand_inds] >= threshold) & center_mask]
        positives[gt_idx, pos_inds] = True

    any_pos = positives.any(dim=0)
    if any_pos.any():
        pos_indices = any_pos.nonzero(as_tuple=False).squeeze(1)
        matched_iou, matched_gt = ious[:, pos_indices].max(dim=0)
        labels[pos_indices] = 1
        matched[pos_indices] = matched_gt

    return labels, matched


@PROPOSAL_GENERATOR_REGISTRY.register()
class ATSSRPN(RPN):
    def __init__(self, strides=None, topk=9, center_radius=0.0, **kwargs):
        super().__init__(**kwargs)
        if strides is None and hasattr(self.anchor_generator, "strides"):
            strides = list(self.anchor_generator.strides)
        self.strides = strides if strides is not None else []
        self.topk = topk
        self.center_radius = center_radius

        if comm.is_main_process():
            logger.info(
                "Using ATSS RPN (topk=%s, center_radius=%s, strides=%s)",
                self.topk,
                self.center_radius,
                self.strides,
            )

    @classmethod
    def from_config(cls, cfg, input_shape):
        params = super().from_config(cfg, input_shape)
        in_features = params.get("in_features", cfg.MODEL.RPN.IN_FEATURES)
        params["strides"] = [input_shape[f].stride for f in in_features]
        params["topk"] = getattr(cfg.MODEL.RPN, "ATSS_TOPK", 9)
        params["center_radius"] = getattr(cfg.MODEL.RPN, "ATSS_CENTER_RADIUS", 0.0)
        if comm.is_main_process():
            logger.info("ATSS from_config: features=%s, strides=%s", in_features, params["strides"])
        return params

    def set_atss_params(self, topk: int, center_radius: float) -> None:
        self.topk = topk
        self.center_radius = center_radius

    def label_and_sample_anchors(self, anchors, gt_instances):
        anchors_cat = Boxes.cat(anchors)
        num_per_level = [len(level.tensor) for level in anchors]
        gt_boxes = [instance.gt_boxes for instance in gt_instances]
        image_sizes = [instance.image_size for instance in gt_instances]
        gt_labels: List[torch.Tensor] = []
        matched_boxes: List[torch.Tensor] = []

        for image_size, gt_boxes_i in zip(image_sizes, gt_boxes):
            anchors_per_level = []
            start = 0
            for count in num_per_level:
                end = start + count
                anchors_per_level.append(anchors_cat.tensor[start:end])
                start = end

            if len(gt_boxes_i) > 0:
                gt_tensor = gt_boxes_i.tensor
            else:
                gt_tensor = anchors_cat.tensor.new_zeros((0, 4))

            if anchors_per_level and len(gt_tensor) > 0:
                usable_levels = min(len(anchors_per_level), len(self.strides))
                labels_atss, matched_atss = atss_assign_single_image(
                    anchors_per_level[:usable_levels],
                    gt_tensor,
                    strides=self.strides[:usable_levels],
                    topk=self.topk,
                    center_radius=self.center_radius,
                )

                if usable_levels < len(anchors_per_level):
                    remaining = torch.cat(anchors_per_level[usable_levels:], dim=0)
                    from detectron2.modeling.matcher import Matcher
                    from detectron2.structures import pairwise_iou as d2_pairwise_iou

                    matrix = d2_pairwise_iou(gt_boxes_i, Boxes(remaining))
                    match_ids, labels_remain = self.anchor_matcher(matrix)
                    labels_i = torch.cat([labels_atss, labels_remain])
                    matched_ids = torch.cat([matched_atss, match_ids])
                else:
                    labels_i, matched_ids = labels_atss, matched_atss
            else:
                labels_i = torch.zeros(len(anchors_cat), dtype=torch.long, device=anchors_cat.tensor.device)
                matched_ids = torch.zeros(len(anchors_cat), dtype=torch.long, device=anchors_cat.tensor.device)

            if self.anchor_boundary_thresh >= 0:
                inside = anchors_cat.inside_box(image_size, self.anchor_boundary_thresh)
                labels_i[~inside] = -1

            if len(gt_boxes_i) == 0:
                matched_boxes_i = torch.zeros_like(anchors_cat.tensor)
            else:
                clamped = matched_ids.clamp(min=0, max=len(gt_boxes_i) - 1)
                matched_boxes_i = gt_boxes_i[clamped].tensor
                matched_boxes_i[labels_i != 1] = 0

            labels_i = super()._subsample_labels(labels_i)
            gt_labels.append(labels_i)
            matched_boxes.append(matched_boxes_i)

        return gt_labels, matched_boxes

    def _subsample_labels(self, labels, matched_gt_boxes):
        batch_size = self.batch_size_per_image
        positive_fraction = self.positive_fraction

        sampled_labels: List[torch.Tensor] = []
        sampled_boxes: List[torch.Tensor] = []

        for labels_per_image, boxes_per_image in zip(labels, matched_gt_boxes):
            pos_idx = (labels_per_image == 1).nonzero(as_tuple=False).squeeze(1)
            neg_idx = (labels_per_image == 0).nonzero(as_tuple=False).squeeze(1)

            num_pos = min(int(batch_size * positive_fraction), pos_idx.numel())
            if num_pos > 0:
                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:num_pos]]

            num_neg = min(batch_size - num_pos, neg_idx.numel())
            if num_neg > 0:
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]]

            sampled = labels_per_image.new_full(labels_per_image.shape, -1)
            sampled[pos_idx] = 1
            sampled[neg_idx] = 0

            sampled_labels.append(sampled)
            sampled_boxes.append(boxes_per_image)

        return sampled_labels, sampled_boxes


# -----------------------------------------------------------------------------
# Custom transforms and augmentations
# -----------------------------------------------------------------------------


class LetterboxResize(T.Augmentation):
    def __init__(self, target_size: Union[int, Tuple[int, int]], pad_value: int = 128):
        super().__init__()
        self.target_size = target_size
        self.pad_value = pad_value
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> "LetterboxTransform":
        return LetterboxTransform(img.shape[:2], self.target_size, self.pad_value)

    def __call__(self, aug_input: T.AugInput) -> Transform:
        transform = super().__call__(aug_input)
        target_h, target_w = _unpack_size(self.target_size)
        aug_input.height = target_h
        aug_input.width = target_w
        return transform


class LetterboxTransform(Transform):
    def __init__(self, src_shape: Tuple[int, int], target_size: Union[int, Tuple[int, int]], pad_value: int = 128):
        super().__init__()
        h, w = src_shape
        target_h, target_w = _unpack_size(target_size)
        self.scale = min(target_h / h, target_w / w)
        self.new_h = int(h * self.scale)
        self.new_w = int(w * self.scale)
        pad_h = target_h - self.new_h
        pad_w = target_w - self.new_w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left
        self.target_size = target_size
        self.target_h = target_h
        self.target_w = target_w
        self.pad_value = pad_value

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        border_value = (self.pad_value,) * img.shape[2] if img.ndim == 3 else self.pad_value
        return cv2.copyMakeBorder(
            resized,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords = coords.astype(np.float32)
        coords[:, 0] = coords[:, 0] * self.scale + self.pad_left
        coords[:, 1] = coords[:, 1] * self.scale + self.pad_top
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        reshaped = box.reshape(-1, 2)
        return self.apply_coords(reshaped).reshape(box.shape)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if isinstance(segmentation, list):
            return [self.apply_coords(np.array(poly).reshape(-1, 2)).flatten().tolist() for poly in segmentation]
        if isinstance(segmentation, np.ndarray):
            # For binary masks, pad with background (0) and use nearest interpolation.
            mask = segmentation
            is_bool = mask.dtype == np.bool_
            if mask.ndim == 2:
                mask_u8 = mask.astype(np.uint8)
                resized = cv2.resize(mask_u8, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST)
                padded = cv2.copyMakeBorder(
                    resized,
                    self.pad_top,
                    self.pad_bottom,
                    self.pad_left,
                    self.pad_right,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
                return padded.astype(np.bool_) if is_bool else padded
            if mask.ndim == 3:
                resized = cv2.resize(mask, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST)
                border_value = (0,) * resized.shape[2]
                return cv2.copyMakeBorder(
                    resized,
                    self.pad_top,
                    self.pad_bottom,
                    self.pad_left,
                    self.pad_right,
                    cv2.BORDER_CONSTANT,
                    value=border_value,
                )
        return self.apply_image(segmentation)


def _motion_kernel(ksize: int, angle_deg: float) -> np.ndarray:
    ksize = max(3, int(ksize) | 1)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    return kernel / max(kernel.sum(), 1e-6)


class MotionBlurTransform(Transform):
    def __init__(self, ksize: int, angle: float):
        super().__init__()
        self.ksize = int(ksize)
        self.angle = float(angle)
        self.kernel = _motion_kernel(self.ksize, self.angle)

    def apply_image(self, img: np.ndarray) -> np.ndarray:  # pragma: no cover - cv2 dependent
        return cv2.filter2D(img, -1, self.kernel)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        # photometric transform: do not alter masks
        return segmentation


class RandomMotionBlur(T.Augmentation):
    def __init__(self, prob: float = 0.35, k_min: int = 7, k_max: int = 25, angle_range: Tuple[int, int] = (-90, 90)):
        super().__init__()
        self.prob = prob
        self.k_min = k_min
        self.k_max = k_max
        self.angle_range = angle_range
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        if self._rand_range() >= self.prob:
            return T.NoOpTransform()
        k = int(np.random.randint(self.k_min, self.k_max + 1))
        angle = float(np.random.uniform(*self.angle_range))
        return MotionBlurTransform(k, angle)


class GammaTransform(Transform):
    def __init__(self, gamma: float, gain: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.gain = float(gain)

    def apply_image(self, img: np.ndarray) -> np.ndarray:  # pragma: no cover - cv2 dependent
        x = img.astype(np.float32) / 255.0
        y = self.gain * np.power(x, self.gamma)
        out = np.clip(y, 0.0, 1.0) * 255.0 + 0.5
        return out.astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        # photometric transform: do not alter masks
        return segmentation


class RandomGamma(T.Augmentation):
    def __init__(self, prob: float = 0.2, gamma_range: Tuple[float, float] = (0.6, 0.95), gain: float = 1.0):
        super().__init__()
        self.prob = prob
        self.gamma_range = gamma_range
        self.gain = gain
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        if self._rand_range() >= self.prob:
            return T.NoOpTransform()
        gamma = float(np.random.uniform(*self.gamma_range))
        return GammaTransform(gamma=gamma, gain=self.gain)


class CLAHETransform(Transform):
    def __init__(self, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)):
        super().__init__()
        self.clip_limit = float(clip_limit)
        self.tile_grid = tile_grid

    def apply_image(self, img: np.ndarray) -> np.ndarray:  # pragma: no cover - cv2 dependent
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        # photometric transform: do not alter masks
        return segmentation


class RandomCLAHE(T.Augmentation):
    def __init__(self, prob: float = 0.2, clip_min: float = 1.8, clip_max: float = 3.2,
                 tile_choices: Tuple[Tuple[int, int], ...] = ((4, 4), (8, 8), (12, 12))):
        super().__init__()
        self.prob = prob
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.tile_choices = tile_choices
        self._init(locals())

    def get_transform(self, img: np.ndarray) -> Transform:
        if self._rand_range() >= self.prob:
            return T.NoOpTransform()
        clip = float(np.random.uniform(self.clip_min, self.clip_max))
        tile = random.choice(self.tile_choices)
        return CLAHETransform(clip_limit=clip, tile_grid=tile)


class LetterboxDatasetMapper(detectron2.data.DatasetMapper):
    def __init__(
        self,
        *args,
        save_visualizations: bool = False,
        visualization_root: Optional[str] = None,
        metadata_name: Optional[str] = None,
        visualization_prefix: str = "train",
        visualization_format: str = "png",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_visualizations = save_visualizations
        self.visualization_format = visualization_format
        self.visualization_prefix = visualization_prefix
        self.visualization_root = Path(visualization_root).resolve() if visualization_root else None
        self.metadata = MetadataCatalog.get(metadata_name) if metadata_name else None

        if self.save_visualizations:
            if self.visualization_root is None:
                raise ValueError("visualization_root must be provided when save_visualizations is True")
            self.visualization_root.mkdir(parents=True, exist_ok=True)
            logging.getLogger(__name__).info(
                "[LetterboxDatasetMapper] saving samples to %s", self.visualization_root
            )

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - heavy I/O
        dataset_dict = copy.deepcopy(dataset_dict)
        image = detectron2.data.detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        detectron2.data.detection_utils.check_image_size(dataset_dict, image)

        h_orig, w_orig = image.shape[:2]
        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        h_trans, w_trans = image.shape[:2]

        dataset_dict["height"] = h_trans
        dataset_dict["width"] = w_trans
        dataset_dict["original_height"] = h_orig
        dataset_dict["original_width"] = w_orig

        if isinstance(transforms, T.TransformList):
            for tfm in transforms.transforms:
                if isinstance(tfm, LetterboxTransform):
                    dataset_dict["letterbox_meta"] = {
                        "scale": tfm.scale,
                        "pad_top": tfm.pad_top,
                        "pad_left": tfm.pad_left,
                        "new_h": tfm.new_h,
                        "new_w": tfm.new_w,
                        "target": tfm.target_size,
                    }
                    break

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        annotations = dataset_dict.get("annotations")
        if annotations is not None:
            filtered = [obj for obj in annotations if obj.get("iscrowd", 0) == 0]
            instances = []
            image_shape = image.shape[:2]
            for obj in filtered:
                instances.append(
                    detectron2.data.detection_utils.transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    )
                )
            dataset_dict["instances"] = detectron2.data.detection_utils.annotations_to_instances(
                instances,
                image_shape,
                mask_format=self.instance_mask_format,
            )
            if self.recompute_boxes and dataset_dict["instances"].has("gt_masks"):
                dataset_dict["instances"].gt_boxes = dataset_dict["instances"].gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = detectron2.data.detection_utils.filter_empty_instances(
                dataset_dict["instances"]
            )

        if self.save_visualizations:
            self._save_visualization(image, instances or [], dataset_dict, h_orig, w_orig)

        dataset_dict.pop("annotations", None)
        return dataset_dict

    def _save_visualization(
        self,
        image: np.ndarray,
        annotations: Iterable[Dict[str, Any]],
        dataset_dict: Dict[str, Any],
        h_orig: int,
        w_orig: int,
    ) -> None:  # pragma: no cover - heavy I/O
        try:
            vis_dict = {
                "height": image.shape[0],
                "width": image.shape[1],
                "image_id": dataset_dict.get("image_id"),
                "annotations": annotations,
            }
            visualizer = Visualizer(image, metadata=self.metadata, scale=1.0)
            drawn = visualizer.draw_dataset_dict(vis_dict).get_image()

            file_stem = Path(dataset_dict.get("file_name", "image")).stem
            unique_id = uuid4().hex
            filename = f"{self.visualization_prefix}_{file_stem}_{unique_id}.{self.visualization_format}"
            output_path = self.visualization_root / filename

            info_text = f"orig:{w_orig}x{h_orig} -> aug:{image.shape[1]}x{image.shape[0]}"
            annotated = drawn.copy()
            cv2.putText(
                annotated,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        except Exception as exc:  # pragma: no cover - logging path
            logging.getLogger(__name__).warning(
                "Failed to save visualization for %s: %s", dataset_dict.get("file_name"), exc
            )


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------


def prepare_dataset_split(config: TrainingConfig) -> Tuple[str, str]:
    ratio = config.split.ratio
    if ratio is None:
        return config.dataset.train_json, config.dataset.val_json

    if not 0 < ratio < 1:
        raise ValueError("dataset_split_ratio must be between 0 and 1")

    base_jsons = config.split.sources or [config.dataset.train_json, config.dataset.val_json]
    base_jsons = [str(Path(p).resolve()) for p in base_jsons]
    config.split.sources = base_jsons

    logger.info(
        "[Dataset Split] ratio=%.3f/%.3f seed=%d sources=%s",
        ratio,
        1 - ratio,
        config.split.seed,
        base_jsons,
    )

    combined_images: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
    image_to_annotations: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    categories = None
    licenses = None
    info = None

    for src_idx, json_path in enumerate(base_jsons):
        path_obj = Path(json_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"annotation not found: {json_path}")

        with path_obj.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if categories is None:
            categories = copy.deepcopy(data.get("categories", []))
            licenses = copy.deepcopy(data.get("licenses")) if data.get("licenses") else None
            info = copy.deepcopy(data.get("info")) if data.get("info") else None
        elif data.get("categories") and data.get("categories") != categories:
            logger.warning("[Dataset Split] category mismatch detected for %s", json_path)

        for img in data.get("images", []):
            key = (src_idx, img["id"])
            combined_images.append((key, copy.deepcopy(img)))
            image_to_annotations.setdefault(key, [])

        for ann in data.get("annotations", []):
            key = (src_idx, ann.get("image_id"))
            image_to_annotations.setdefault(key, [])
            image_to_annotations[key].append(copy.deepcopy(ann))

    total_images = len(combined_images)
    if total_images == 0:
        raise ValueError("no images found for splitting")

    rng = random.Random(config.split.seed)
    rng.shuffle(combined_images)

    desired_train = int(total_images * ratio)
    desired_train = max(1, min(total_images - 1, desired_train))
    train_entries = combined_images[:desired_train]
    val_entries = combined_images[desired_train:]

    if not val_entries:
        val_entries = train_entries[-1:]
        train_entries = train_entries[:-1]

    def build_subset(entries: List[Tuple[Tuple[int, int], Dict[str, Any]]], split_name: str) -> Dict[str, Any]:
        images_subset: List[Dict[str, Any]] = []
        annotations_subset: List[Dict[str, Any]] = []
        next_ann_id = 1
        for new_image_id, (key, img) in enumerate(entries, start=1):
            new_img = copy.deepcopy(img)
            new_img["id"] = new_image_id
            images_subset.append(new_img)

            for ann in image_to_annotations.get(key, []):
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = next_ann_id
                next_ann_id += 1
                new_ann["image_id"] = new_image_id
                annotations_subset.append(new_ann)

        subset_info = copy.deepcopy(info) if info is not None else {}
        if isinstance(subset_info, dict):
            subset_info.setdefault("description", "Generated by prepare_dataset_split")
            subset_info.setdefault("split", split_name)
            subset_info.setdefault("split_ratio", ratio)
            subset_info.setdefault("seed", config.split.seed)

        subset: Dict[str, Any] = {
            "images": images_subset,
            "annotations": annotations_subset,
            "categories": copy.deepcopy(categories) if categories is not None else [],
            "info": subset_info,
        }
        if licenses is not None:
            subset["licenses"] = copy.deepcopy(licenses)
        return subset

    def _iter_negative_image_paths(ns_cfg: NegativeSamplingConfig) -> List[Path]:
        root = Path(ns_cfg.directory).expanduser()
        if ns_cfg.recursive:
            candidates = list(root.rglob("*"))
        else:
            candidates = list(root.glob("*"))
        exts = {f".{ext.lower().lstrip('.')}" for ext in ns_cfg.extensions}
        paths = [p for p in candidates if p.is_file() and p.suffix.lower() in exts]
        rng_local = random.Random(ns_cfg.seed)
        rng_local.shuffle(paths)
        if ns_cfg.max_images is not None:
            paths = paths[: max(0, int(ns_cfg.max_images))]
        return paths

    def _read_image_size(path: Path) -> Optional[Tuple[int, int]]:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2.imread returned None")
            h, w = img.shape[:2]
            return int(w), int(h)
        except Exception:
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                img2 = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                if img2 is None:
                    return None
                h2, w2 = img2.shape[:2]
                return int(w2), int(h2)
            except Exception:
                return None

    def _add_negative_images(subset: Dict[str, Any], ns_cfg: NegativeSamplingConfig, split_name: str) -> int:
        root = Path(ns_cfg.directory).expanduser()
        if not root.exists():
            logger.warning("[Negative Sampling] directory not found: %s", root)
            return 0

        image_paths = _iter_negative_image_paths(ns_cfg)
        if not image_paths:
            logger.warning("[Negative Sampling] no images found under %s", root)
            return 0

        images_list = subset.setdefault("images", [])
        existing_ids = {img.get("id") for img in images_list if isinstance(img, dict)}
        next_id = max([i for i in existing_ids if isinstance(i, int)] + [0]) + 1

        added = 0
        skipped = 0
        for p in image_paths:
            size = _read_image_size(p)
            if size is None:
                skipped += 1
                continue
            w, h = size
            images_list.append(
                {
                    "id": next_id,
                    "file_name": str(p.resolve()),
                    "width": w,
                    "height": h,
                }
            )
            next_id += 1
            added += 1

        if skipped:
            logger.warning("[Negative Sampling] skipped %d image(s) with unreadable size in %s", skipped, root)
        logger.info("[Negative Sampling] added %d negative image(s) to %s split", added, split_name)
        return added

    train_data = build_subset(train_entries, "train")
    val_data = build_subset(val_entries, "val")

    if config.negative_sampling.enabled:
        _add_negative_images(train_data, config.negative_sampling, "train")
        if config.negative_sampling.add_to_val:
            _add_negative_images(val_data, config.negative_sampling, "val")

    split_root = get_output_layout(config.trainer.output_dir).meta_dir / "dataset_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    ratio_tag = f"{ratio:.3f}".replace(".", "p")
    train_path = split_root / f"train_ratio{ratio_tag}_seed{config.split.seed}.json"
    val_path = split_root / f"val_ratio{ratio_tag}_seed{config.split.seed}.json"

    with train_path.open("w", encoding="utf-8") as handle:
        json.dump(train_data, handle, indent=2)
    with val_path.open("w", encoding="utf-8") as handle:
        json.dump(val_data, handle, indent=2)

    logger.info(
        "[Dataset Split] train=%d val=%d -> %s | %s",
        len(train_data["images"]),
        len(val_data["images"]),
        train_path,
        val_path,
    )

    return str(train_path), str(val_path)


def setup_custom_dataset(config: TrainingConfig) -> Tuple[int, int]:
    logger.info("Registering dataset for class=%s", config.dataset.class_name)
    normalize_dataset_paths(config)

    train_json, val_json = prepare_dataset_split(config)
    train_json, val_json = apply_debug_subset(config, train_json, val_json)
    config.dataset.train_json = train_json
    config.dataset.val_json = val_json

    for name in ("custom_train", "custom_val"):
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        if name in MetadataCatalog:
            MetadataCatalog.remove(name)

    register_coco_instances("custom_train", {}, train_json, config.dataset.train_dir)
    register_coco_instances("custom_val", {}, val_json, config.dataset.val_dir)

    MetadataCatalog.get("custom_train").thing_classes = [config.dataset.class_name]
    MetadataCatalog.get("custom_train").thing_dataset_id_to_contiguous_id = {1: 0}
    MetadataCatalog.get("custom_val").thing_classes = [config.dataset.class_name]
    MetadataCatalog.get("custom_val").thing_dataset_id_to_contiguous_id = {1: 0}

    from detectron2.data import get_detection_dataset_dicts

    train_size = len(get_detection_dataset_dicts("custom_train", filter_empty=False))
    val_size = len(get_detection_dataset_dicts("custom_val", filter_empty=False))

    logger.info("Dataset registered: train=%d val=%d", train_size, val_size)
    return train_size, val_size


def ensure_subset_json(
    src_json: str,
    dst_json: Path,
    count: int,
    seed: int,
    label: str,
) -> Optional[Path]:
    if count <= 0:
        return None
    if dst_json.is_file():
        try:
            with dst_json.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and data.get("images"):
                return dst_json
        except Exception:
            pass
    with open(src_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not images:
        logger.warning("[Quick Eval] no images in %s (skip %s subset)", src_json, label)
        return None
    rng = random.Random(seed)
    ids = [img["id"] for img in images]
    rng.shuffle(ids)
    take = min(int(count), len(ids))
    subset_ids = set(ids[:take])
    subset_images = [img for img in images if img["id"] in subset_ids]
    subset_anns = [ann for ann in annotations if ann.get("image_id") in subset_ids]
    payload = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": subset_images,
        "annotations": subset_anns,
    }
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("[Quick Eval] wrote %s subset to %s (%d images)", label, dst_json, len(subset_images))
    return dst_json


def apply_debug_subset(config: TrainingConfig, train_json: str, val_json: str) -> Tuple[str, str]:
    if not config.debug.debug_mode:
        return train_json, val_json

    train_samples = int(config.debug.train_samples)
    val_samples = int(config.debug.val_samples)
    if train_samples <= 0 and val_samples <= 0:
        logger.warning("[Debug Mode] enabled but no subset sizes provided; using full dataset.")
        return train_json, val_json

    split_root = get_output_layout(config.trainer.output_dir).meta_dir / "dataset_splits" / "debug"
    if train_samples > 0:
        train_subset = ensure_subset_json(
            train_json,
            split_root / f"train_debug_{train_samples}_seed{config.debug.seed}.json",
            train_samples,
            config.debug.seed,
            label="debug_train",
        )
        if train_subset is not None:
            train_json = str(train_subset)

    if val_samples > 0:
        val_subset = ensure_subset_json(
            val_json,
            split_root / f"val_debug_{val_samples}_seed{config.debug.seed}.json",
            val_samples,
            config.debug.seed + 7,
            label="debug_val",
        )
        if val_subset is not None:
            val_json = str(val_subset)

    logger.info("[Debug Mode] using subset train=%s val=%s", train_json, val_json)
    return train_json, val_json


def calculate_iterations(epochs: int, dataset_size: int, batch_size: int, num_gpus: int) -> Tuple[int, int]:
    import math

    total_batch = max(1, batch_size * max(1, num_gpus))
    iters_per_epoch = math.ceil(dataset_size / total_batch)
    max_iter = epochs * iters_per_epoch
    return max_iter, iters_per_epoch


def build_letterbox_transforms(config: TrainingConfig, *, is_train: bool) -> List[Any]:
    transforms: List[Any] = []
    if is_train:
        if config.augmentation.use_random_flip:
            transforms.append(L(T.RandomFlip)(horizontal=True))
        transforms.append(
            L(T.RandomApply)(tfm_or_aug=L(T.RandomRotation)(angle=[-180, 180], expand=True), prob=0.2)
        )
        transforms.append(
            L(T.RandomApply)(
                tfm_or_aug=L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.2),
                prob=0.2,
            )
        )
        transforms.append(
            L(T.RandomApply)(
                tfm_or_aug=L(T.RandomBrightness)(intensity_min=0.4, intensity_max=1.0),
                prob=0.25,
            )
        )
        transforms.append(
            L(T.RandomApply)(
                tfm_or_aug=L(T.RandomBrightness)(intensity_min=0.3, intensity_max=0.7),
                prob=0.10,
            )
        )
        transforms.append(
            L(T.RandomApply)(
                tfm_or_aug=L(T.RandomContrast)(intensity_min=0.7, intensity_max=1.3),
                prob=0.2,
            )
        )
        transforms.append(
            L(T.RandomApply)(
                tfm_or_aug=L(T.RandomContrast)(intensity_min=0.4, intensity_max=1.0),
                prob=0.25,
            )
        )
        transforms.append(L(RandomGamma)(prob=0.2, gamma_range=(0.6, 0.95), gain=1.0))
        transforms.append(
            L(T.RandomApply)(
                tfm_or_aug=L(T.RandomSaturation)(intensity_min=0.7, intensity_max=1.3),
                prob=0.2,
            )
        )
        transforms.append(
            L(T.RandomApply)(tfm_or_aug=L(T.RandomLighting)(scale=0.7), prob=0.1)
        )
        transforms.append(
            L(RandomCLAHE)(prob=0.2, clip_min=1.8, clip_max=3.2, tile_choices=((4, 4), (8, 8), (12, 12)))
        )
        transforms.append(L(RandomMotionBlur)(prob=0.35, k_min=7, k_max=25, angle_range=(-90, 90)))

    transforms.append(L(LetterboxResize)(target_size=config.augmentation.train_size, pad_value=128))
    return transforms


def create_detectron_config(config: TrainingConfig, train_size: int) -> Tuple[Any, int, int, bool]:
    cfg = LazyConfig.load(
        "projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1280_lrd0p8.py"
    )

    optimization = config.optimization
    trainer_cfg = config.trainer
    max_iter, iters_per_epoch = calculate_iterations(
        optimization.epochs,
        train_size,
        optimization.batch_size_per_gpu,
        trainer_cfg.num_gpus,
    )
    eval_period = max(1, int(trainer_cfg.eval_every_epochs * iters_per_epoch))
    checkpoint_period = trainer_cfg.checkpoint_every_epochs * iters_per_epoch
    warmup_iters = int(optimization.warmup_epochs * iters_per_epoch)

    logger.info("\n%s", "=" * 60)
    logger.info("Training configuration summary")
    logger.info("=" * 60)
    logger.info("Model: %s", config.model_size)
    logger.info("Train images: %d", train_size)
    logger.info("Epochs: %d", optimization.epochs)
    logger.info("Batch/GPU: %d", optimization.batch_size_per_gpu)
    logger.info("Gradient accumulation: %d", optimization.gradient_accumulation_steps)
    effective_batch = (
        optimization.batch_size_per_gpu
        * trainer_cfg.num_gpus
        * optimization.gradient_accumulation_steps
    )
    logger.info("Effective batch size: %d", effective_batch)
    logger.info("Base LR: %.4g", optimization.base_lr)
    logger.info("Letterbox size: %s", config.augmentation.train_size)
    logger.info("ATSS enabled: %s", config.atss.enabled)
    logger.info("Iterations: %d (%d/epoch)", max_iter, iters_per_epoch)
    logger.info("=" * 60)

    # --- DINOv3 backbone (差し替え箇所: EVA-02 ViT-L → DINOv3 ViT-L/16) -------
    from detectron2.modeling import DINOv3Backbone

    cfg.model.roi_heads.num_classes = 1
    _train_h, _train_w = _unpack_size(config.augmentation.train_size)
    if _train_h == _train_w:
        cfg.model.backbone.square_pad = _train_h
    else:
        # 矩形入力時はsquare paddingを無効化。
        # LetterboxResizeが正確なターゲットサイズを生成するため、
        # ImageList.from_tensorsはsize_divisibility(=16)のみ適用。
        cfg.model.backbone.square_pad = 0
    cfg.model.backbone.net = L(DINOv3Backbone)(
        img_size=max(_train_h, _train_w),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layers_to_use=1,
        out_feature="last_feat",
        weights=unified_paths.DINOv3_WEIGHTS,
        pretrained=True,
    )
    cfg.model.backbone.in_feature = "last_feat"
    # --- DINOv3 backbone ここまで -----------------------------------------------

    cfg.dataloader.train = L(detectron2.data.build_detection_train_loader)(
        dataset=L(detectron2.data.get_detection_dataset_dicts)(names="custom_train", filter_empty=False),
        mapper=L(LetterboxDatasetMapper)(
            is_train=True,
            augmentations=build_letterbox_transforms(config, is_train=True),
            image_format="RGB",
            use_instance_mask=True,
            recompute_boxes=True,
            instance_mask_format="bitmask",
            save_visualizations=config.viz.train_samples,
            visualization_root=config.viz.train_dir,
            metadata_name="custom_train",
            visualization_prefix="train",
            visualization_format=config.viz.image_format,
        ),
        total_batch_size=optimization.batch_size_per_gpu * trainer_cfg.num_gpus,
        num_workers=trainer_cfg.num_workers,
    )

    cfg.dataloader.test = L(detectron2.data.build_detection_test_loader)(
        dataset=L(detectron2.data.get_detection_dataset_dicts)(names="custom_val", filter_empty=False),
        mapper=L(LetterboxDatasetMapper)(
            is_train=False,
            augmentations=build_letterbox_transforms(config, is_train=False),
            image_format="RGB",
            instance_mask_format="bitmask",
            save_visualizations=config.viz.eval_samples,
            visualization_root=config.viz.eval_dir,
            metadata_name="custom_val",
            visualization_prefix="val",
            visualization_format=config.viz.image_format,
        ),
        num_workers=trainer_cfg.num_workers,
    )

    cfg.dataloader.evaluator = L(LetterboxCOCOEvaluator)(
        dataset_name="custom_val",
        output_dir=trainer_cfg.output_dir,
        target_size=config.augmentation.train_size,
        recall_iou_thrs=config.recall_iou_thrs,
        recall_max_dets=config.recall_max_dets,
    )

    cfg.model.proposal_generator.pre_nms_topk = (
        config.rpn.pre_nms_topk_train,
        config.rpn.pre_nms_topk_test,
    )
    cfg.model.proposal_generator.post_nms_topk = (
        config.rpn.post_nms_topk_train,
        config.rpn.post_nms_topk_test,
    )
    cfg.model.proposal_generator.batch_size_per_image = config.rpn.batch_size_per_image
    cfg.model.proposal_generator.nms_thresh = config.rpn.nms_thresh
    cfg.model.proposal_generator.min_box_size = config.rpn.min_box_size

    if config.atss.enabled:
        original_pg = cfg.model.proposal_generator
        cfg.model.proposal_generator = L(ATSSRPN)(
            topk=config.atss.topk,
            center_radius=config.atss.center_radius,
            in_features=original_pg.in_features,
            head=original_pg.head,
            anchor_generator=original_pg.anchor_generator,
            anchor_matcher=None,
            box2box_transform=original_pg.box2box_transform,
            batch_size_per_image=config.rpn.batch_size_per_image,
            positive_fraction=config.rpn.positive_fraction,
            pre_nms_topk=(config.rpn.pre_nms_topk_train, config.rpn.pre_nms_topk_test),
            post_nms_topk=(config.rpn.post_nms_topk_train, config.rpn.post_nms_topk_test),
            nms_thresh=config.rpn.nms_thresh,
            min_box_size=config.rpn.min_box_size,
            anchor_boundary_thresh=getattr(original_pg, "anchor_boundary_thresh", -1),
            loss_weight=getattr(original_pg, "loss_weight", 1.0),
            box_reg_loss_type=getattr(original_pg, "box_reg_loss_type", "smooth_l1"),
            smooth_l1_beta=getattr(original_pg, "smooth_l1_beta", 0.0),
        )
    else:
        cfg.model.proposal_generator.anchor_matcher = L(detectron2.modeling.matcher.Matcher)(
            thresholds=[0.2, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        )

    cfg.model.roi_heads.batch_size_per_image = config.roi.batch_size_per_image
    cfg.model.roi_heads.positive_fraction = config.roi.positive_fraction
    cfg.model.roi_heads.proposal_append_gt = config.roi.proposal_append_gt

    if hasattr(cfg.model.roi_heads, "box_predictors"):
        for predictor in cfg.model.roi_heads.box_predictors:
            predictor.test_score_thresh = config.cascade.test_score_thresh
            predictor.test_nms_thresh = config.cascade.test_nms_thresh
            predictor.test_topk_per_image = config.cascade.test_topk_per_image

    if hasattr(cfg.model.roi_heads, "proposal_matchers"):
        cfg.model.roi_heads.proposal_matchers = [
            L(detectron2.modeling.matcher.Matcher)(
                thresholds=[config.cascade.iou_thresholds[idx]],
                labels=[0, 1],
                allow_low_quality_matches=False,
            )
            for idx in range(len(config.cascade.iou_thresholds))
        ]

    cfg.train.max_iter = max_iter
    cfg.train.eval_period = eval_period
    cfg.train.log_period = config.logging.log_period
    cfg.train.checkpointer = {
        "period": checkpoint_period,
        "max_to_keep": trainer_cfg.max_checkpoints_to_keep,
    }
    layout = get_output_layout(trainer_cfg.output_dir)
    cfg.train.output_dir = str(layout.pth_dir)
    cfg.train.device = "cuda"
    cfg.train.amp.enabled = trainer_cfg.use_amp
    cfg.train.ddp.fp16_compression = True

    output_dir = layout.pth_dir.expanduser().resolve()
    checkpoint_path, resume = resolve_checkpoint_settings(config, output_dir)
    cfg.train.init_checkpoint = checkpoint_path
    cfg.train.resume = resume

    cfg.train.model_ema = {
        "enabled": trainer_cfg.use_ema,
        "device": "cuda",
        "decay": 0.9999,
        "use_ema_weights_for_eval_only": True,
    }

    from functools import partial
    from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
    from fvcore.common.param_scheduler import CosineParamScheduler
    from detectron2.solver import WarmupParamScheduler

    cfg.optimizer = L(torch.optim.AdamW)(
        params=L(detectron2.solver.get_default_optimizer_params)(
            model="${...model}",
            base_lr="${..lr}",
            weight_decay_norm=None,
            bias_lr_factor=1.0,
            weight_decay_bias=None,
            lr_factor_func=partial(get_vit_lr_decay_rate, num_layers=24, lr_decay_rate=0.8),
            overrides={},
        ),
        lr=optimization.base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )

    cfg.lr_multiplier = L(WarmupParamScheduler)(
        scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=optimization.cosine_end_value),
        warmup_length=warmup_iters / max_iter if warmup_iters > 0 else 0.01,
        warmup_factor=0.001,
    )

    return cfg, max_iter, iters_per_epoch, resume


class GradientAccumulationTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, *, accumulation_steps: int = 1):
        super().__init__(model, data_loader, optimizer)
        self.accumulation_steps = max(1, accumulation_steps)
        self._step_count = 0
        logger.info("Gradient accumulation enabled: %d steps", self.accumulation_steps)

    def run_step(self) -> None:  # pragma: no cover - requires training loop
        assert self.model.training
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
        losses = losses / self.accumulation_steps
        losses.backward()

        self._step_count += 1
        if self._step_count % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)


class AMPGradientAccumulationTrainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, *, accumulation_steps: int = 1, grad_scaler=None):
        super().__init__(model, data_loader, optimizer, grad_scaler)
        self.accumulation_steps = max(1, accumulation_steps)
        self._step_count = 0
        logger.info("AMP + gradient accumulation: %d steps", self.accumulation_steps)

    def run_step(self) -> None:  # pragma: no cover - requires training loop
        assert self.model.training
        assert torch.cuda.is_available(), "AMP requires CUDA"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            losses = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict
            losses = losses / self.accumulation_steps

        self.grad_scaler.scale(losses).backward()
        self._step_count += 1

        if self._step_count % self.accumulation_steps == 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)


def compute_letterbox_params(h: int, w: int, target: Union[int, Tuple[int, int]]) -> Dict[str, float]:
    target_h, target_w = _unpack_size(target)
    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    return {
        "scale": scale,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }


def unletterbox_instances(instances: Instances, lb: Dict[str, float], orig_h: int, orig_w: int, target_size: Union[int, Tuple[int, int]]) -> Instances:
    boxes = instances.pred_boxes.tensor
    boxes[:, [0, 2]] -= lb["pad_left"]
    boxes[:, [1, 3]] -= lb["pad_top"]
    boxes /= lb["scale"]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h - 1)
    instances.pred_boxes.tensor = boxes

    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        mask_shape = masks.shape
        _tgt_h, _tgt_w = _unpack_size(target_size)
        if len(mask_shape) == 3 and mask_shape[1] == _tgt_h and mask_shape[2] == _tgt_w:
            cropped = masks[
                :,
                lb["pad_top"] : lb["pad_top"] + lb["new_h"],
                lb["pad_left"] : lb["pad_left"] + lb["new_w"],
            ]
            if cropped.shape[1] > 0 and cropped.shape[2] > 0:
                resized = []
                for mask in cropped.cpu().numpy().astype(np.uint8):
                    resized.append(cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST))
                if resized:
                    instances.pred_masks = torch.from_numpy(np.stack(resized, axis=0) > 0)
            else:
                instances.pred_masks = torch.zeros((len(masks), orig_h, orig_w), dtype=torch.bool)

    return instances


class LetterboxCOCOEvaluator(COCOEvaluator):
    def __init__(
        self,
        dataset_name: str,
        output_dir: Optional[str] = None,
        target_size: Union[int, Tuple[int, int]] = 1280,
        recall_iou_thrs: Optional[Tuple[float, ...]] = None,
        recall_max_dets: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name, output_dir=output_dir, *args, **kwargs)
        self.target_size = target_size
        self.recall_iou_thrs = tuple(recall_iou_thrs) if recall_iou_thrs else None
        self.recall_max_dets = int(recall_max_dets)

    def process(self, inputs, outputs):  # pragma: no cover - evaluation runtime
        for inp, out in zip(inputs, outputs):
            h_orig = inp.get("original_height", inp["height"])
            w_orig = inp.get("original_width", inp["width"])
            lb_meta = inp.get("letterbox_meta")
            if lb_meta is None:
                lb_meta = compute_letterbox_params(h_orig, w_orig, self.target_size)
            if "instances" in out:
                out["instances"] = unletterbox_instances(out["instances"], lb_meta, h_orig, w_orig, self.target_size)
        super().process(inputs, outputs)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None, use_custom_ranges=False):
        results = super()._derive_coco_results(
            coco_eval,
            iou_type,
            class_names=class_names,
            use_custom_ranges=use_custom_ranges,
        )
        if coco_eval is None or not self.recall_iou_thrs:
            return results
        recalls = coco_eval.eval.get("recall", None)
        if recalls is None:
            return results
        eval_iou_thrs = np.array(coco_eval.params.iouThrs)
        target_iou_thrs = np.array(self.recall_iou_thrs, dtype=float)
        iou_indices = []
        missing = []
        for thr in target_iou_thrs:
            idx = np.where(np.isclose(eval_iou_thrs, thr, atol=1e-6))[0]
            if idx.size == 0:
                missing.append(float(thr))
            else:
                iou_indices.append(int(idx[0]))
        if missing:
            self._logger.warning("Missing recall IoU thresholds in COCOeval: %s", missing)
        max_dets = list(coco_eval.params.maxDets)
        if self.recall_max_dets not in max_dets:
            selected_max_dets = max_dets[-1]
            self._logger.warning(
                "recall_max_dets=%s not in maxDets=%s; using %s instead.",
                self.recall_max_dets,
                max_dets,
                selected_max_dets,
            )
        else:
            selected_max_dets = self.recall_max_dets
        if iou_indices:
            m_index = max_dets.index(selected_max_dets)
            for thr_idx in iou_indices:
                thr = float(eval_iou_thrs[thr_idx])
                rec = recalls[thr_idx, :, 0, m_index]
                valid = rec > -1
                rec_val = float(np.mean(rec[valid])) if valid.any() else float("nan")
                results[f"{iou_type}_recall@{selected_max_dets}@{thr:.2f}"] = rec_val
        return results


def visualize_predictions(
    cfg,
    model,
    dataset_name: str,
    config: TrainingConfig,
    current_iter: int,
    current_epoch: int,
    output_dir: Path,
) -> None:
    if not (comm.is_main_process() and config.logging.enable_wandb and wandb is not None and cv2 is not None):
        return

    if wandb.run is None:
        logger.warning("WandB run inactive, skipping visualization")
        return

    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    if not dataset:
        logger.warning("Dataset %s is empty, skip visualization", dataset_name)
        return

    vis_dir = output_dir / "visualizations" / f"epoch_{current_epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    mapper = LetterboxDatasetMapper(
        is_train=False,
        augmentations=instantiate(cfg.dataloader.test.mapper.augmentations),
        image_format="RGB",
        use_instance_mask=False,
        instance_mask_format="bitmask",
    )

    model_for_eval = model.module if hasattr(model, "module") else model
    was_training = model_for_eval.training
    model_for_eval.eval()
    use_amp = bool(getattr(getattr(cfg, "train", None), "amp", None) and cfg.train.amp.enabled)
    amp_dtype = None
    if use_amp and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    indices = np.linspace(0, len(dataset) - 1, num=min(config.viz.vis_num_images, len(dataset)), dtype=int)
    recorded: List[Any] = []

    try:
        with torch.inference_mode():
            for idx in indices[:4]:
                data = dataset[idx]
                img_path = data["file_name"]
                image_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if image_bgr is None:
                    continue

                if image_bgr.ndim == 2:
                    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
                elif image_bgr.shape[2] == 4:
                    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                h_orig, w_orig = image_rgb.shape[:2]

                vis_gt = Visualizer(image_rgb, metadata=metadata, scale=1.0)
                gt_record = {
                    "annotations": data.get("annotations", []),
                    "image_id": data.get("image_id"),
                    "height": h_orig,
                    "width": w_orig,
                }
                gt_image = vis_gt.draw_dataset_dict(gt_record).get_image()

                mapped = mapper(data)
                inputs = {
                    "image": mapped["image"].to(cfg.train.device),
                    "height": mapped["height"],
                    "width": mapped["width"],
                }

                if amp_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        outputs = model_for_eval([inputs])[0]
                else:
                    outputs = model_for_eval([inputs])[0]
                instances = outputs["instances"]
                keep = instances.scores >= config.viz.score_thresh
                instances = instances[keep]
                lb_meta = mapped.get("letterbox_meta") or compute_letterbox_params(
                    mapped.get("original_height", h_orig),
                    mapped.get("original_width", w_orig),
                    config.augmentation.train_size,
                )
                instances = unletterbox_instances(
                    instances,
                    lb_meta,
                    mapped.get("original_height", h_orig),
                    mapped.get("original_width", w_orig),
                    config.augmentation.train_size,
                )

                vis_pred = Visualizer(image_rgb, metadata=metadata, scale=1.0)
                pred_image = vis_pred.draw_instance_predictions(instances.to("cpu")).get_image()

                combined = np.concatenate([gt_image, pred_image], axis=1)
                caption = f"Epoch {current_epoch} | {Path(img_path).name}"
                recorded.append(wandb.Image(combined, caption=caption))

                out_file = vis_dir / f"{idx:04d}.png"
                cv2.imwrite(str(out_file), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if recorded:
            wandb.log({"val/predictions": recorded}, step=current_iter)
    finally:
        if was_training:
            model_for_eval.train()


def dump_training_visualizations(config: TrainingConfig) -> None:
    logger.info("[Visualization Dump] Exporting augmented training samples")
    original_gpus = config.trainer.num_gpus
    original_flag = config.viz.train_samples
    try:
        config.trainer.num_gpus = 1
        config.viz.train_samples = True

        if not config.viz.train_dir:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            config.viz.train_dir = str(get_output_layout(config.trainer.output_dir).visualizations_dir / "data_visualizations" / timestamp / "train")
        ensure_dir(config.viz.train_dir)

        train_size, _ = setup_custom_dataset(config)
        cfg, _, _, _ = create_detectron_config(config, train_size)
        data_loader = instantiate(cfg.dataloader.train)

        total = 0
        limit = config.viz.max_images
        for batch in data_loader:
            total += len(batch)
            if limit and total >= limit:
                break

        logger.info("[Visualization Dump] Saved %d samples to %s", total, config.viz.train_dir)
    finally:
        config.trainer.num_gpus = original_gpus
        config.viz.train_samples = original_flag


class EpochTrainerHook(hooks.HookBase):
    def __init__(self, total_epochs: int, iters_per_epoch: int):
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch

    def before_train(self):  # pragma: no cover - logging hook
        logger.info("\n%s", "=" * 80)
        logger.info("Training for %d epochs (%d iters/epoch)", self.total_epochs, self.iters_per_epoch)
        logger.info("=" * 80)

    def after_step(self):  # pragma: no cover - hook runtime
        current_iter = self.trainer.iter + 1
        current_epoch = (current_iter - 1) // self.iters_per_epoch + 1
        iter_in_epoch = ((current_iter - 1) % self.iters_per_epoch) + 1
        storage: EventStorage = self.trainer.storage
        storage.put_scalar("epoch", current_epoch, smoothing_hint=False)
        storage.put_scalar("iter_in_epoch", iter_in_epoch, smoothing_hint=False)
        storage.put_scalar("epoch_progress", (iter_in_epoch / self.iters_per_epoch) * 100, smoothing_hint=False)


class CustomEvalHook(hooks.EvalHook):
    def __init__(self, eval_period: int, eval_function, user_cfg: TrainingConfig, trainer: Optional[TrainerBase]):
        super().__init__(eval_period, eval_function)
        self.user_cfg = user_cfg
        self.trainer = trainer
        self.metrics_dir: Optional[str] = None
        self._quick_eval_ready = False
        self._train_metrics_loader = None
        self._train_metrics_evaluator = None
        self._val_loss_loader = None
        self._train_metrics_size = 0
        self._val_loss_size = 0

    @staticmethod
    def _collect_recall_metrics(results: Dict[str, Any]) -> Dict[str, float]:
        recall_metrics: Dict[str, float] = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if "recall@" in sub_key and isinstance(sub_val, (int, float, np.floating)):
                        recall_metrics[f"{key}/{sub_key}"] = float(sub_val)
            elif "recall@" in key and isinstance(value, (int, float, np.floating)):
                recall_metrics[key] = float(value)
        return recall_metrics

    def _dump_recall_metrics(self, results: Dict[str, Any], current_iter: int, current_epoch: int) -> None:
        if self.metrics_dir is None:
            return
        recall_metrics = self._collect_recall_metrics(results)
        if not recall_metrics:
            return
        payload = {
            "iter": current_iter,
            "epoch": current_epoch,
            "recall_iou_thrs": list(self.user_cfg.recall_iou_thrs),
            "recall_max_dets": int(self.user_cfg.recall_max_dets),
            "metrics": recall_metrics,
        }
        out_path = os.path.join(self.metrics_dir, f"recall_iou_iter_{current_iter:07d}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
        except OSError as exc:
            logger.warning("Failed to write recall metrics: %s", exc)

    def _prepare_quick_eval(self) -> None:
        if self._quick_eval_ready:
            return
        qcfg = self.user_cfg.quick_eval
        if not qcfg.enabled:
            return
        if qcfg.train_metrics_samples <= 0 and qcfg.val_loss_samples <= 0:
            return

        split_root = get_output_layout(self.user_cfg.trainer.output_dir).meta_dir / "dataset_splits" / "quick_eval"
        train_json = None
        val_json = None
        if qcfg.train_metrics_samples > 0:
            train_json = ensure_subset_json(
                self.user_cfg.dataset.train_json,
                split_root / f"train_metrics_{qcfg.train_metrics_samples}_seed{qcfg.seed}.json",
                qcfg.train_metrics_samples,
                qcfg.seed,
                label="train_metrics",
            )
        if qcfg.val_loss_samples > 0:
            val_json = ensure_subset_json(
                self.user_cfg.dataset.val_json,
                split_root / f"val_loss_{qcfg.val_loss_samples}_seed{qcfg.seed}.json",
                qcfg.val_loss_samples,
                qcfg.seed + 17,
                label="val_loss",
            )

        if train_json is not None:
            name = "custom_train_metrics"
            if name in DatasetCatalog:
                DatasetCatalog.remove(name)
            if name in MetadataCatalog:
                MetadataCatalog.remove(name)
            register_coco_instances(name, {}, str(train_json), self.user_cfg.dataset.train_dir)
            MetadataCatalog.get(name).thing_classes = [self.user_cfg.dataset.class_name]
            MetadataCatalog.get(name).thing_dataset_id_to_contiguous_id = {1: 0}
            eval_transforms = instantiate(build_letterbox_transforms(self.user_cfg, is_train=False))
            mapper = LetterboxDatasetMapper(
                is_train=False,
                augmentations=eval_transforms,
                image_format="RGB",
                instance_mask_format="bitmask",
            )
            dataset = DatasetCatalog.get(name)
            self._train_metrics_loader = build_detection_test_loader(
                dataset,
                mapper=mapper,
                batch_size=qcfg.samples_per_gpu,
                num_workers=qcfg.num_workers,
            )
            self._train_metrics_evaluator = LetterboxCOCOEvaluator(
                dataset_name=name,
                output_dir=None,
                target_size=self.user_cfg.augmentation.train_size,
                recall_iou_thrs=self.user_cfg.recall_iou_thrs,
                recall_max_dets=self.user_cfg.recall_max_dets,
            )
            self._train_metrics_size = qcfg.train_metrics_samples

        if val_json is not None:
            name = "custom_val_loss"
            if name in DatasetCatalog:
                DatasetCatalog.remove(name)
            if name in MetadataCatalog:
                MetadataCatalog.remove(name)
            register_coco_instances(name, {}, str(val_json), self.user_cfg.dataset.val_dir)
            MetadataCatalog.get(name).thing_classes = [self.user_cfg.dataset.class_name]
            MetadataCatalog.get(name).thing_dataset_id_to_contiguous_id = {1: 0}
            loss_transforms = instantiate(build_letterbox_transforms(self.user_cfg, is_train=False))
            mapper = LetterboxDatasetMapper(
                is_train=True,
                augmentations=loss_transforms,
                image_format="RGB",
                use_instance_mask=True,
                recompute_boxes=True,
                instance_mask_format="bitmask",
            )
            dataset = DatasetCatalog.get(name)
            self._val_loss_loader = build_detection_test_loader(
                dataset,
                mapper=mapper,
                batch_size=qcfg.samples_per_gpu,
                num_workers=qcfg.num_workers,
            )
            self._val_loss_size = qcfg.val_loss_samples

        self._quick_eval_ready = True

    def _run_train_metrics(self, model) -> Dict[str, Any]:
        if self._train_metrics_loader is None or self._train_metrics_evaluator is None:
            return {}
        amp_dtype = None
        cfg_obj = getattr(self.trainer, "cfg", None)
        use_amp = bool(getattr(getattr(cfg_obj, "train", None), "amp", None) and cfg_obj.train.amp.enabled)
        if use_amp and torch.cuda.is_available():
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                return inference_on_dataset(
                    model, self._train_metrics_loader, self._train_metrics_evaluator
                )
        return inference_on_dataset(model, self._train_metrics_loader, self._train_metrics_evaluator)

    def _compute_val_loss(self, model, use_amp: bool) -> Optional[float]:
        if self._val_loss_loader is None:
            return None
        was_training = model.training
        model.train()
        total_loss = 0.0
        total_samples = 0
        amp_dtype = None
        if use_amp and torch.cuda.is_available():
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        for batch in self._val_loss_loader:
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        loss_dict = model(batch)
                else:
                    loss_dict = model(batch)
            if not loss_dict:
                continue
            batch_loss = sum(value for value in loss_dict.values())
            total_loss += float(batch_loss.detach().cpu()) * len(batch)
            total_samples += len(batch)
        if not was_training:
            model.eval()
        if total_samples == 0:
            return 0.0
        return total_loss / float(total_samples)

    def _do_eval(self):  # pragma: no cover - evaluation runtime
        if self.metrics_dir is None and comm.is_main_process():
            self.metrics_dir = str(get_output_layout(self.user_cfg.trainer.output_dir).json_dir / "eval_metrics_json")
            os.makedirs(self.metrics_dir, exist_ok=True)
            logger.info("[Eval Metrics] directory: %s", self.metrics_dir)

        results = self._func()
        current_iter = self.trainer.iter + 1 if self.trainer else 0
        iters_per_epoch = (
            max(1, self.trainer.max_iter // self.user_cfg.optimization.epochs)
            if self.trainer
            else 1
        )
        current_epoch = (current_iter - 1) // iters_per_epoch + 1 if current_iter > 0 else 0

        if comm.is_main_process():
            self._dump_recall_metrics(results, current_iter, current_epoch)
            self._prepare_quick_eval()

        quick_log_vars: Dict[str, float] = {}
        if (
            comm.is_main_process()
            and self.trainer is not None
            and self.user_cfg.quick_eval.enabled
            and self.user_cfg.quick_eval.interval > 0
            and current_epoch > 0
            and (current_epoch % self.user_cfg.quick_eval.interval == 0)
        ):
            train_metrics = self._run_train_metrics(self.trainer.model)
            for metric_name in self.user_cfg.quick_eval.metrics:
                metric_results = train_metrics.get(metric_name, {})
                if isinstance(metric_results, dict) and "AP" in metric_results:
                    quick_log_vars[f"train{self._train_metrics_size}_{metric_name}_mAP"] = float(metric_results["AP"])
            cfg_obj = getattr(self.trainer, "cfg", None)
            use_amp = bool(getattr(getattr(cfg_obj, "train", None), "amp", None) and cfg_obj.train.amp.enabled)
            val_loss = self._compute_val_loss(self.trainer.model, use_amp=use_amp)
            if val_loss is not None:
                quick_log_vars[f"val{self._val_loss_size}_loss"] = float(val_loss)
            if quick_log_vars:
                storage: EventStorage = self.trainer.storage
                for key, value in quick_log_vars.items():
                    storage.put_scalar(key, value, smoothing_hint=False)
                logger.info(
                    "QuickEval: " + ", ".join([f"{k}={v:.4f}" for k, v in quick_log_vars.items()])
                )

        if comm.is_main_process() and self.user_cfg.logging.enable_wandb and wandb is not None:
            try:
                log_data: Dict[str, float] = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, (int, float)):
                                log_data[f"val/{key}/{sub_key}"] = float(sub_val)
                    elif isinstance(value, (int, float)):
                        log_data[f"val/{key}"] = float(value)
                log_data.update(quick_log_vars)
                if log_data and wandb.run is not None:
                    wandb.log(log_data, step=current_iter)
            except Exception as exc:
                logger.warning("Failed to log evaluation metrics to WandB: %s", exc)

        if comm.is_main_process() and self.user_cfg.logging.enable_wandb and wandb is not None and self.trainer is not None:
            try:
                output_dir = get_output_layout(self.user_cfg.trainer.output_dir).visualizations_dir
                cfg_obj = getattr(self.trainer, "cfg", None)
                if cfg_obj is not None:
                    visualize_predictions(cfg_obj, self.trainer.model, "custom_val", self.user_cfg, current_iter, current_epoch, output_dir)
            except Exception as exc:
                logger.warning("Visualization during evaluation failed: %s", exc)

        # Evaluation may take different time among workers.
        # A barrier makes them start the next iteration together.
        comm.synchronize()
        return results


def do_test(cfg, model):  # pragma: no cover - evaluation execution
    if "evaluator" not in cfg.dataloader:
        return {}

    evaluator = instantiate(cfg.dataloader.evaluator)
    use_amp = bool(getattr(getattr(cfg.train, "amp", None), "enabled", False))
    if use_amp and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            results = inference_on_dataset(model, instantiate(cfg.dataloader.test), evaluator)
    else:
        results = inference_on_dataset(model, instantiate(cfg.dataloader.test), evaluator)

    if comm.is_main_process():
        logger.info("\n%s", "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        logger.info("%s/%s: %.4f", key, sub_key, sub_val)
            elif isinstance(value, (int, float)):
                logger.info("%s: %.4f", key, value)
        logger.info("=" * 60)

    return results


def do_train(
    cfg,
    config: TrainingConfig,
    max_iter: int,
    iters_per_epoch: int,
    resume: bool,
) -> None:  # pragma: no cover - training loop
    writers = []
    if comm.is_main_process():
        writers = build_event_writers(config.trainer.output_dir, max_iter)

        if config.logging.enable_wandb and wandb is not None:
            try:
                wandb_config = {
                    "model": config.model_size,
                    "epochs": config.optimization.epochs,
                    "batch_size": config.optimization.batch_size_per_gpu,
                    "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
                    "effective_batch_size": config.optimization.batch_size_per_gpu * config.trainer.num_gpus * config.optimization.gradient_accumulation_steps,
                    "base_lr": config.optimization.base_lr,
                    "train_size": config.augmentation.train_size,
                    "letterbox": True,
                    "atss_enabled": config.atss.enabled,
                }
                wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    name=config.logging.wandb_run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=wandb_config,
                    mode=config.logging.wandb_mode,
                )
                writers.append(StableWandBWriter())
                logger.info("WandB logging enabled")
            except Exception as exc:
                logger.warning("WandB initialization failed: %s", exc)
                config.logging.enable_wandb = False

    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    # Ensure initial_lr is set for scheduler when resuming.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    train_loader = instantiate(cfg.dataloader.train)

    ema.may_build_model_ema(cfg, model)

    accumulation = config.optimization.gradient_accumulation_steps
    if accumulation > 1:
        trainer_class = AMPGradientAccumulationTrainer if cfg.train.amp.enabled else GradientAccumulationTrainer
        trainer = trainer_class(model, train_loader, optimizer, accumulation_steps=accumulation)
    else:
        trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optimizer)

    trainer.cfg = cfg
    trainer.max_iter = max_iter

    layout = get_output_layout(config.trainer.output_dir)
    checkpointer = DetectionCheckpointer(
        model,
        str(layout.pth_dir),
        trainer=trainer,
        **ema.may_get_ema_checkpointer(cfg, model),
    )

    from detectron2.data import get_detection_dataset_dicts

    train_dataset_size = len(get_detection_dataset_dicts("custom_train", filter_empty=False))
    total_batch = config.optimization.batch_size_per_gpu * config.trainer.num_gpus
    computed_iters_per_epoch = (train_dataset_size + total_batch - 1) // total_batch
    logger.info("Dataset size: %d | total batch: %d | iters/epoch: %d", train_dataset_size, total_batch, computed_iters_per_epoch)

    scheduler = instantiate(cfg.lr_multiplier)

    eval_hook = CustomEvalHook(cfg.train.eval_period, lambda c=cfg, m=model: do_test(c, m), config, trainer)

    hooks_to_register = [
        hooks.IterationTimer(),
        ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
        hooks.LRScheduler(scheduler=scheduler),
        hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer) if comm.is_main_process() else None,
        eval_hook,
        EpochTrainerHook(config.optimization.epochs, iters_per_epoch),
        hooks.PeriodicWriter(writers, period=cfg.train.log_period) if comm.is_main_process() else None,
    ]

    trainer.register_hooks([hook for hook in hooks_to_register if hook is not None])

    checkpoint_state = checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=resume)
    start_iter = checkpoint_state.get("iteration", -1) + 1 if resume else 0
    logger.info("Starting training loop (max_iter=%d)", max_iter)
    trainer.train(start_iter, max_iter)

    if comm.is_main_process() and config.logging.enable_wandb and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def run_training_process(config: TrainingConfig) -> None:  # pragma: no cover - orchestrator
    set_seed(42)
    layout = get_output_layout(config.trainer.output_dir)
    for path in (layout.pth_dir, layout.json_dir, layout.tensorboard_dir, layout.visualizations_dir, layout.meta_dir):
        ensure_dir(str(path))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_vis_dir = layout.visualizations_dir / "data_visualizations" / timestamp
    if config.viz.train_samples and not config.viz.train_dir:
        config.viz.train_dir = str(base_vis_dir / "train")
    if config.viz.eval_samples and not config.viz.eval_dir:
        config.viz.eval_dir = str(base_vis_dir / "val")
    if config.viz.train_dir:
        ensure_dir(config.viz.train_dir)
    if config.viz.eval_dir:
        ensure_dir(config.viz.eval_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(layout.meta_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )

    if config.viz.dump_only:
        dump_training_visualizations(config)
        logger.info("Visualization dump completed (training skipped)")
        return

    train_size, _ = setup_custom_dataset(config)
    cfg, max_iter, iters_per_epoch, resume = create_detectron_config(config, train_size)

    import pickle

    with open(layout.meta_dir / "config.pkl", "wb") as handle:
        pickle.dump(cfg, handle)

    with open(layout.meta_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_size": config.model_size,
                "epochs": config.optimization.epochs,
                "batch_size": config.optimization.batch_size_per_gpu,
                "base_lr": config.optimization.base_lr,
                "letterbox": True,
                "atss_enabled": config.atss.enabled,
                "train_json": config.dataset.train_json,
                "val_json": config.dataset.val_json,
                "use_wandb": config.logging.enable_wandb,
                "negative_sampling_enabled": config.negative_sampling.enabled,
                "negative_sampling_dir": config.negative_sampling.directory if config.negative_sampling.enabled else None,
                "negative_sampling_add_to_val": config.negative_sampling.add_to_val if config.negative_sampling.enabled else None,
                "logical_output_dir": str(layout.logical_dir),
                "artifact_dirs": {
                    "pth": str(layout.pth_dir),
                    "json": str(layout.json_dir),
                    "tensorboard": str(layout.tensorboard_dir),
                    "visualizations": str(layout.visualizations_dir),
                    "meta": str(layout.meta_dir),
                },
            },
            handle,
            indent=2,
        )

    do_train(cfg, config, max_iter, iters_per_epoch, resume)
    logger.info("Training finished successfully")


def test_letterbox_transform(target_size: Union[int, Tuple[int, int]] = 1280) -> None:
    logger.info("Testing Letterbox Transform with target size %s", target_size)
    letterbox = LetterboxResize(target_size=target_size, pad_value=128)
    _expected_h, _expected_w = _unpack_size(target_size)
    test_cases = [
        (1920, 1080, "Landscape 16:9"),
        (1080, 1920, "Portrait 9:16"),
        (1000, 1000, "Square"),
        (2000, 500, "Wide"),
    ]

    for h, w, desc in test_cases:
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        transform = letterbox.get_transform(img)
        transformed = transform.apply_image(img)
        bbox = np.array([[100, 100, 200, 200]], dtype=np.float32)
        transformed_bbox = transform.apply_box(bbox)
        logger.info(
            "%s: %dx%d -> %dx%d | scale=%.3f | pad=(%d,%d)",
            desc,
            h,
            w,
            transformed.shape[0],
            transformed.shape[1],
            transform.scale,
            transform.pad_top,
            transform.pad_left,
        )
        assert transformed.shape[0] == _expected_h and transformed.shape[1] == _expected_w
        logger.info("bbox %s -> %s", bbox[0], transformed_bbox[0])

    logger.info("Letterbox transform test passed")


def launch_worker(config: TrainingConfig) -> None:  # pragma: no cover - multiprocessing entry
    run_training_process(config)


def training_entry(config: TrainingConfig) -> None:
    available_gpus = torch.cuda.device_count()
    if available_gpus < config.trainer.num_gpus:
        logger.warning(
            "Only %d GPU(s) available, adjusting num_gpus from %d",
            available_gpus,
            config.trainer.num_gpus,
        )
        config.trainer.num_gpus = available_gpus

    if config.debug.run_letterbox_test:
        test_letterbox_transform(config.augmentation.train_size)
        return

    if config.trainer.num_gpus > 1:
        from detectron2.engine import launch

        launch(
            launch_worker,
            config.trainer.num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="tcp://127.0.0.1:12357",
            timeout=timedelta(minutes=config.trainer.ddp_timeout_minutes),
            args=(config,),
        )
    else:
        run_training_process(config)


if __name__ == "__main__":
    # =====================================================================
    # ユーザー設定セクション
    #   - 下記の値を編集してハイパーパラメータやパスを調整してください。
    #   - CLI 引数は不要です。
    # =====================================================================

    TRAINING_CONFIG = TrainingConfig()
    TRAINING_CONFIG.augmentation.train_size = (1280, 720)
    TRAINING_CONFIG.augmentation.test_size = (1280, 720)
    TRAINING_CONFIG.model_size = "dinov3_vitl16_1280x720"

    # チェックポイント設定
    # DINOv3 はバックボーン事前学習のみ。FPN/RPN/ROI はランダム初期化から学習する。
    TRAINING_CONFIG.checkpoint.backbone_checkpoint = unified_paths.DINOv3_WEIGHTS
    TRAINING_CONFIG.checkpoint.use_pretrained = False
    TRAINING_CONFIG.checkpoint.pretrained_path = None
    TRAINING_CONFIG.checkpoint.resume_path = None

    # データセットパス
    # 既定では 0414 追加分込みの最新データセットを使用する。
    TRAINING_CONFIG.dataset.train_json = str(DEFAULT_INCOMING_ADDED_TRAIN_JSON)
    TRAINING_CONFIG.dataset.train_dir = str(DEFAULT_INCOMING_ADDED_DIR)
    TRAINING_CONFIG.dataset.val_json = str(DEFAULT_INCOMING_ADDED_VAL_JSON)
    TRAINING_CONFIG.dataset.val_dir = str(DEFAULT_INCOMING_ADDED_DIR)
    TRAINING_CONFIG.split.ratio = None
    TRAINING_CONFIG.split.sources = None

    # 基本ハイパーパラメータ（EVA-02 版と同一）
    TRAINING_CONFIG.optimization.epochs = 30
    TRAINING_CONFIG.optimization.batch_size_per_gpu = 6
    TRAINING_CONFIG.optimization.base_lr = 4e-5
    TRAINING_CONFIG.optimization.gradient_accumulation_steps = 8

    # 出力ディレクトリ
    TRAINING_CONFIG.trainer.output_dir = str(unified_paths.OUTPUT_ROOT / "dinov3_cascade_0403_1280x720")

    # ATSS / RPN 設定
    TRAINING_CONFIG.atss.enabled = True
    TRAINING_CONFIG.atss.topk = 20
    TRAINING_CONFIG.atss.center_radius = 2.0
    TRAINING_CONFIG.rpn.batch_size_per_image = 256
    TRAINING_CONFIG.rpn.positive_fraction = 0.7
    TRAINING_CONFIG.rpn.pre_nms_topk_train = 20_000
    TRAINING_CONFIG.rpn.pre_nms_topk_test = 10_000
    TRAINING_CONFIG.rpn.post_nms_topk_train = 2_000
    TRAINING_CONFIG.rpn.post_nms_topk_test = 1_000
    TRAINING_CONFIG.rpn.nms_thresh = 0.9
    TRAINING_CONFIG.rpn.min_box_size = 0

    # ROI / Cascade 設定
    TRAINING_CONFIG.roi.batch_size_per_image = 512
    TRAINING_CONFIG.roi.positive_fraction = 0.7
    TRAINING_CONFIG.roi.proposal_append_gt = True
    TRAINING_CONFIG.cascade.iou_thresholds = [0.45, 0.55, 0.65]
    TRAINING_CONFIG.cascade.test_score_thresh = 0.01
    TRAINING_CONFIG.cascade.test_nms_thresh = 0.4
    TRAINING_CONFIG.cascade.test_topk_per_image = 200

    # ログ / デバッグ
    TRAINING_CONFIG.logging.wandb_project = "DINOv3-Cascade_unified"
    TRAINING_CONFIG.logging.enable_wandb = True
    TRAINING_CONFIG.trainer.use_ema = True
    TRAINING_CONFIG.debug.run_letterbox_test = False
    TRAINING_CONFIG.debug.debug_mode = False
    TRAINING_CONFIG.debug.train_samples = 500
    TRAINING_CONFIG.debug.val_samples = 200
    TRAINING_CONFIG.debug.seed = 43
    TRAINING_CONFIG.quick_eval.seed = 43
    TRAINING_CONFIG.trainer.ddp_timeout_minutes = 60

    # 可視化設定（必要に応じて有効化）
    TRAINING_CONFIG.viz.train_samples = False
    TRAINING_CONFIG.viz.eval_samples = False
    TRAINING_CONFIG.viz.dump_only = False

    # ネガティブサンプリング（負例画像; マスク/アノテ無し）を学習に追加
    # - 画像は COCO JSON に追記され、annotations は付与しません（空アノテ画像として扱う）
    # - 評価を汚さないため、デフォルトでは train のみに追加します
    # NOTE: 新データJSONにnegative_sampling画像を含めたため、二重追加防止で無効化
    TRAINING_CONFIG.negative_sampling.enabled = False
    TRAINING_CONFIG.negative_sampling.directory = os.path.join(unified_paths.DATA_ROOT, "negative_sampling")
    TRAINING_CONFIG.negative_sampling.recursive = True
    TRAINING_CONFIG.negative_sampling.add_to_val = False
    TRAINING_CONFIG.negative_sampling.max_images = None

    # 環境変数による上書き（動作確認や短時間検証用）
    env_epochs = os.environ.get("UNIFIED_EPOCHS")
    if env_epochs:
        TRAINING_CONFIG.optimization.epochs = int(env_epochs)
    env_batch = os.environ.get("UNIFIED_BATCH_SIZE_PER_GPU")
    if env_batch:
        TRAINING_CONFIG.optimization.batch_size_per_gpu = int(env_batch)
    env_accum = os.environ.get("UNIFIED_GRAD_ACCUM_STEPS")
    if env_accum:
        TRAINING_CONFIG.optimization.gradient_accumulation_steps = int(env_accum)
    env_gpus = os.environ.get("UNIFIED_NUM_GPUS")
    if env_gpus:
        TRAINING_CONFIG.trainer.num_gpus = int(env_gpus)
    env_num_workers = os.environ.get("UNIFIED_NUM_WORKERS")
    if env_num_workers:
        TRAINING_CONFIG.trainer.num_workers = int(env_num_workers)
    if os.environ.get("UNIFIED_DISABLE_WANDB") == "1":
        TRAINING_CONFIG.logging.enable_wandb = False
    if os.environ.get("UNIFIED_DISABLE_NEGATIVE_SAMPLING") == "1":
        TRAINING_CONFIG.negative_sampling.enabled = False

    # 追加の環境変数上書き（データデバッグ・可視化運用向け）
    env_train_json = os.environ.get("UNIFIED_TRAIN_JSON")
    if env_train_json:
        TRAINING_CONFIG.dataset.train_json = env_train_json
    env_train_dir = os.environ.get("UNIFIED_TRAIN_DIR")
    if env_train_dir:
        TRAINING_CONFIG.dataset.train_dir = env_train_dir
    env_val_json = os.environ.get("UNIFIED_VAL_JSON")
    if env_val_json:
        TRAINING_CONFIG.dataset.val_json = env_val_json
    env_val_dir = os.environ.get("UNIFIED_VAL_DIR")
    if env_val_dir:
        TRAINING_CONFIG.dataset.val_dir = env_val_dir
    env_output_dir = os.environ.get("UNIFIED_OUTPUT_DIR")
    if env_output_dir:
        TRAINING_CONFIG.trainer.output_dir = env_output_dir

    if os.environ.get("UNIFIED_VIZ_DUMP_ONLY") == "1":
        TRAINING_CONFIG.viz.dump_only = True
    if os.environ.get("UNIFIED_VIZ_TRAIN_SAMPLES") == "1":
        TRAINING_CONFIG.viz.train_samples = True
    env_viz_max_images = os.environ.get("UNIFIED_VIZ_MAX_IMAGES")
    if env_viz_max_images:
        TRAINING_CONFIG.viz.max_images = int(env_viz_max_images)
    env_viz_train_dir = os.environ.get("UNIFIED_VIZ_TRAIN_DIR")
    if env_viz_train_dir:
        TRAINING_CONFIG.viz.train_dir = env_viz_train_dir

    training_entry(TRAINING_CONFIG)
