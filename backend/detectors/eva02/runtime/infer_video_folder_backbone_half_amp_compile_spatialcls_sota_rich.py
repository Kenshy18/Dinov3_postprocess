#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch video inference runner (folder input) for:
  detector: backbone-half + AMP + torch.compile (max-autotune) with
            activation-checkpoint disabled, SDPA-forced attention,
            and structural ViT block drop
  classifier: two-stage ROI-feature multiclass head (supports rich_spatial_attn_fusion)

Edit the constants below to change paths/parameters.
"""

from __future__ import annotations

import os
import sys
import time
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

# Prefer SDPA and avoid xFormers import failures (same as base script).
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("EVA2_PREFER_SDPA", "1")

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from configs import paths as unified_paths
TWO_STAGE_DIR = BASE_DIR
TWO_STAGE_SCRIPTS_DIR = BASE_DIR

def _import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load base infer module explicitly from standalone dir to avoid name collision
# with experimental scripts/infer_video_jsonl_singleclass.py.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
base = _import_module_from_file(
    "_standalone_infer_video_jsonl_singleclass",
    BASE_DIR / "infer_video_jsonl_singleclass.py",
)

# Import rich ROI-classifier utilities from experimental two-stage scripts.
if str(TWO_STAGE_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(TWO_STAGE_SCRIPTS_DIR))

# =========================
# Hardcoded settings (edit)
# =========================
from two_stage_roi_classifier import (  # noqa: E402
    build_roi_feature_tensor_from_instances,
    classifier_from_checkpoint,
    extract_box_head_features_from_instances,
    extract_box_pooler_features_expanded_from_instances,
    extract_mask_pooler_features_from_instances,
    normalize_meta_feature_set,
    normalize_roi_feature_source,
    roi_feature_source_needs,
)

INPUT_DIR = str(BASE_DIR / "input")
OUTPUT_DIR = str(BASE_DIR / "output")

RECURSIVE = False
OVERWRITE = False

CHECKPOINT = REPO_ROOT / "checkpoints" / "eva02" / "detector" / "model_final.pth"
CONFIG = (
    REPO_ROOT
    / "eva02"
    / "eva02_det"
    / "projects"
    / "ViTDet"
    / "configs"
    / "eva2_o365_to_coco"
    / "eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1280_lrd0p8.py"
)

# Multiclass ROI classifier checkpoint:
# - If set to None, the latest best.pt is selected from CLASSIFIER_SEARCH_DIRS.
CLASSIFIER_CHECKPOINT: Optional[Path] = REPO_ROOT / "checkpoints" / "eva02" / "classifier" / "best.pt"
CLASSIFIER_SEARCH_DIRS = [
    REPO_ROOT / "checkpoints" / "eva02" / "classifier",
    BASE_DIR / "checkpoints",
]

# If "auto", read model_cfg.feature_source from classifier checkpoint.
ROI_FEATURE_SOURCE = "auto"  # auto|box_head|pooler_gap|pooler_flatten|box_head_plus_pooler_gap

TARGET_SIZE = 1280
SCORE_THRESH = 0.1
NMS_THRESH = 0.5
TOPK = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = True
MODEL_HALF = False
BACKBONE_HALF = True
GPU_PREPROCESS_FLOAT = True
PIN_INPUTS = False
PACK_INPUTS = True
RAW_DETECTOR_POSTPROCESS = True
COMPACT_SERIALIZATION_PAYLOAD = False
ASYNC_SERIALIZATION_TRANSFER = False
CHANNELS_LAST = False
EARLY_SCORE_PREFILTER = False
DEFER_CPU_SERIALIZATION = False
DEFER_CPU_SERIALIZATION_SNAPSHOT = False

COMPILE_BACKBONE = "max-autotune"
COMPILE_HEADS = "none"
INDUCTOR_DISABLE_CUDAGRAPHS = False
DISABLE_ACT_CHECKPOINT = True
PREFER_SDPA = True
DROP_BLOCK_INDICES = "19,21,22"
ONNX_BACKBONE: Optional[str] = None
ONNX_MAX_BATCH = 12
ONNX_DTYPE = "fp16"
ONNX_IOBIND = False
ONNX_DISABLE_TRT = False
RPN_NMS_IMPL = "batched"
RPN_PRE_NMS_MULTIPLIER = 10

BATCH_SIZE = 20
MAX_FRAMES = None
WARMUP_FRAMES = 10  # Reduced for 100-frame baseline measurement
MEASURE = True
MEASURE_BATCH_SYNC = True

JSON_BACKEND = "json"
FLUSH_EVERY = 50
MASK_APPROX = "simple"
MASK_RETR_MODE = "ccomp"
ASYNC_WRITER = False
PREFETCH = False
PREFETCH_QUEUE = 64

# Classifier runtime (for speed):
CLASSIFIER_DEVICE = DEVICE
CLASSIFIER_AMP = True
CLASSIFIER_BATCH_SIZE = 2048  # instance batch (across multiple frames)
CLASSIFIER_OUTPUT_PROBS = False
CLASSIFIER_COMPILE_MODE = "none"  # none|reduce-overhead|max-autotune-no-cudagraphs|max-autotune
CLASSIFIER_TIMING_SYNC = True
CLASSIFIER_FAST_CONFIDENCE = False
CASCADE_SINGLE_CLASS_STREAMING_SCORES = False
RPN_ANCHOR_CACHE = False
UNLETTERBOX_BOX_INPLACE = False
TENSOR_SERIALIZATION_PAYLOAD = False
RAW_TO_ORIG_MASK_POSTPROCESS = True
FAST_RCNN_ONECLASS_FASTPATH = False

LOG_EVERY = 50  # frames

SAVE_OVERLAY_VIDEO = True
OVERLAY_SUFFIX = "_overlay.mp4"
OVERLAY_FOURCC = "mp4v"
OVERLAY_ALPHA = 0.35
OVERLAY_COLOR_BGR = (40, 255, 40)
OVERLAY_DRAW_BOXES = True
OVERLAY_DRAW_CLASS = True
OVERLAY_DRAW_CLS_SCORE = True
OVERLAY_DRAW_DET_SCORE = True
OVERLAY_BOX_COLOR_BGR = (20, 220, 255)
OVERLAY_BOX_THICKNESS = 2
OVERLAY_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_SCALE = 0.55
OVERLAY_TEXT_THICKNESS = 1
OVERLAY_TEXT_COLOR_BGR = (255, 255, 255)
OVERLAY_TEXT_USE_UTF8 = True
OVERLAY_PIL_FONT_SIZE = 20
OVERLAY_PIL_FONT_PATH: Optional[Path] = BASE_DIR / "fonts" / "NotoSansCJK-Medium.ttc"
OVERLAY_PIL_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
]

_PIL_FONT_CACHE = None
_WARNED_UTF8_UNAVAILABLE = False


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*OVERLAY_FOURCC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open overlay video writer: {output_path}")
    return writer


def _contains_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _resolve_overlay_font_path() -> Optional[Path]:
    if OVERLAY_PIL_FONT_PATH is not None:
        p = _as_path(OVERLAY_PIL_FONT_PATH).expanduser().resolve()
        return p if p.is_file() else None
    for p in OVERLAY_PIL_FONT_CANDIDATES:
        if p.is_file():
            return p
    return None


def _get_overlay_pil_font():
    global _PIL_FONT_CACHE
    if _PIL_FONT_CACHE is not None:
        return _PIL_FONT_CACHE
    if ImageFont is None:
        return None
    font_path = _resolve_overlay_font_path()
    if font_path is None:
        return None
    try:
        _PIL_FONT_CACHE = ImageFont.truetype(str(font_path), int(OVERLAY_PIL_FONT_SIZE))
    except Exception:
        _PIL_FONT_CACHE = None
    return _PIL_FONT_CACHE


def _draw_label_cv2(out: np.ndarray, label: str, x: int, y: int) -> None:
    cv2.putText(
        out,
        label,
        (int(x), int(y)),
        OVERLAY_FONT_FACE,
        OVERLAY_FONT_SCALE,
        (0, 0, 0),
        OVERLAY_TEXT_THICKNESS + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        label,
        (int(x), int(y)),
        OVERLAY_FONT_FACE,
        OVERLAY_FONT_SCALE,
        OVERLAY_TEXT_COLOR_BGR,
        OVERLAY_TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_overlay_labels(out: np.ndarray, text_items: list[tuple[str, int, int]]) -> np.ndarray:
    global _WARNED_UTF8_UNAVAILABLE
    if not text_items:
        return out

    need_utf8 = bool(OVERLAY_TEXT_USE_UTF8 and any(_contains_non_ascii(t[0]) for t in text_items))
    if not need_utf8:
        for label, x, y in text_items:
            _draw_label_cv2(out, label, x, y)
        return out

    font = _get_overlay_pil_font()
    if Image is None or ImageDraw is None or font is None:
        if not _WARNED_UTF8_UNAVAILABLE:
            _WARNED_UTF8_UNAVAILABLE = True
            print(
                "[WARN] UTF-8 overlay requested but Pillow/font unavailable. "
                "Install pillow and ensure a CJK font exists (e.g. NotoSansCJK)."
            )
        for label, x, y in text_items:
            _draw_label_cv2(out, label, x, y)
        return out

    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    text_color_rgb = (int(OVERLAY_TEXT_COLOR_BGR[2]), int(OVERLAY_TEXT_COLOR_BGR[1]), int(OVERLAY_TEXT_COLOR_BGR[0]))
    shadow_rgb = (0, 0, 0)

    for label, x, y in text_items:
        x_i = int(x)
        y_i = int(y)
        # Thin shadow/outline for readability.
        draw.text((x_i - 1, y_i), label, font=font, fill=shadow_rgb)
        draw.text((x_i + 1, y_i), label, font=font, fill=shadow_rgb)
        draw.text((x_i, y_i - 1), label, font=font, fill=shadow_rgb)
        draw.text((x_i, y_i + 1), label, font=font, fill=shadow_rgb)
        draw.text((x_i, y_i), label, font=font, fill=text_color_rgb)

    out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return out


def _resolve_classifier_checkpoint() -> Path:
    if CLASSIFIER_CHECKPOINT is not None:
        p = _as_path(CLASSIFIER_CHECKPOINT).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"classifier checkpoint not found: {p}")
        return p

    cands: list[Path] = []
    for d in CLASSIFIER_SEARCH_DIRS:
        d = _as_path(d).expanduser().resolve()
        if not d.is_dir():
            continue
        cands.extend(d.glob("run_*/checkpoints/best.pt"))

    if not cands:
        raise FileNotFoundError(
            "No classifier checkpoint found. Set CLASSIFIER_CHECKPOINT or prepare outputs under: "
            + ", ".join(str(_as_path(x).expanduser().resolve()) for x in CLASSIFIER_SEARCH_DIRS)
        )

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _filter_instances_by_score(instances, score_thresh: float):
    # Keep current device here; move to CPU only when serializing outputs.
    if score_thresh > 0 and instances.has("scores"):
        keep = instances.scores >= float(score_thresh)
        instances = instances[keep]
    return instances


_AUX_INSTANCE_FIELDS_TO_DROP = (
    "pred_class_logits",
    "pred_box_features",
    "pred_box_pooler_features",
    "pred_box_pooler_features_expanded",
    "pred_mask_pooler_features",
)


def _strip_aux_instance_fields(instances) -> None:
    for name in _AUX_INSTANCE_FIELDS_TO_DROP:
        if instances.has(name):
            instances.remove(name)


def _build_meta_features(instances, width: int, height: int, meta_feature_set: str = "legacy_iou") -> torch.Tensor:
    num_inst = len(instances)
    if num_inst == 0:
        return torch.zeros((0, 5), dtype=torch.float32)

    img_area = float(max(1, int(width) * int(height)))

    boxes = instances.pred_boxes.tensor
    if boxes.dtype != torch.float32:
        boxes = boxes.float()
    feat_device = boxes.device

    if instances.has("scores"):
        det_scores = instances.scores
        if det_scores.dtype != torch.float32:
            det_scores = det_scores.float()
        if det_scores.device != feat_device:
            det_scores = det_scores.to(feat_device, non_blocking=True)
    else:
        det_scores = torch.ones((num_inst,), dtype=torch.float32, device=feat_device)

    bw = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
    bh = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    bbox_area_ratio = (bw * bh) / img_area

    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        if masks.ndim == 4:
            masks = masks[:, 0]
        if hasattr(masks, "device") and masks.device != feat_device:
            # Avoid large cross-device copies for meta-only mask ratio.
            mask_area = torch.zeros((num_inst,), dtype=torch.float32, device=feat_device)
            mask_area_ratio = torch.zeros((num_inst,), dtype=torch.float32, device=feat_device)
        else:
            masks_f = masks.float()
            if masks_f.ndim == 3 and int(masks_f.shape[-2]) == int(height) and int(masks_f.shape[-1]) == int(width):
                mask_area = masks_f.flatten(start_dim=1).sum(dim=1)
                mask_area_ratio = mask_area / img_area
                if mask_area_ratio.device != feat_device:
                    mask_area_ratio = mask_area_ratio.to(feat_device, non_blocking=True)
            else:
                # Raw ROI mask path: approximate mask area in image space via
                # positive fraction inside the box times box area.
                mask_fill_ratio = (masks_f > 0.5).flatten(start_dim=1).float().mean(dim=1)
                mask_area = mask_fill_ratio * (bw * bh)
                mask_area_ratio = mask_area / img_area
    else:
        mask_area = torch.zeros((num_inst,), dtype=torch.float32, device=feat_device)
        mask_area_ratio = torch.zeros((num_inst,), dtype=torch.float32, device=feat_device)

    bbox_area = bw * bh
    mset = normalize_meta_feature_set(meta_feature_set)
    if mset == "legacy_iou":
        meta4 = torch.zeros((num_inst,), dtype=torch.float32, device=feat_device)
        meta5 = torch.zeros((num_inst,), dtype=torch.float32, device=feat_device)
    elif mset == "geo_v2":
        meta4 = torch.log((bw + 1e-6) / (bh + 1e-6))
        meta5 = mask_area / torch.clamp(bbox_area, min=1e-6)
    else:
        raise ValueError(f"unsupported meta_feature_set: {meta_feature_set}")
    return torch.stack([det_scores, bbox_area_ratio, mask_area_ratio, meta4, meta5], dim=1)


def _split_by_counts(x: np.ndarray, counts: Sequence[int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    s = 0
    for c in counts:
        e = s + int(c)
        out.append(x[s:e])
        s = e
    return out


def _classify_roi_batch(
    roi_feats_by_frame: list[torch.Tensor],
    meta_by_frame: list[torch.Tensor],
    extra_feats_by_frame: Optional[Dict[str, list[torch.Tensor]]],
    classifier: torch.nn.Module,
    classifier_device: str,
    classifier_amp: bool,
    classifier_batch_size: int,
    classifier_use_meta: bool,
    feature_l2norm: bool,
    output_probs: bool,
    synchronize_timing: bool,
):
    counts = [int(x.shape[0]) for x in roi_feats_by_frame]
    total = int(sum(counts))
    if total == 0:
        empty_i = [np.zeros((0,), dtype=np.int64) for _ in counts]
        empty_f = [np.zeros((0,), dtype=np.float32) for _ in counts]
        empty_p = [np.zeros((0, 0), dtype=np.float32) for _ in counts]
        return empty_i, empty_f, empty_p, 0.0

    preds_split: list[np.ndarray] = []
    confs_split: list[np.ndarray] = []
    probs_split: list[np.ndarray] = []
    pending_feat: list[torch.Tensor] = []
    pending_meta: list[torch.Tensor] = []
    pending_counts: list[int] = []
    pending_total = 0
    pending_extra: Dict[str, list[torch.Tensor]] = {
        name: [] for name, by_frame in (extra_feats_by_frame or {}).items() if by_frame
    }

    use_cuda = classifier_device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda and synchronize_timing:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    classifier.eval()

    def flush_pending() -> None:
        nonlocal pending_total
        if not pending_counts:
            return

        x = torch.cat(pending_feat, dim=0).contiguous().to(
            classifier_device,
            non_blocking=True,
            dtype=torch.float32,
        )
        m = torch.cat(pending_meta, dim=0).contiguous().to(
            classifier_device,
            non_blocking=True,
            dtype=torch.float32,
        )
        extra_kwargs: Dict[str, torch.Tensor] = {}
        for name, chunks in pending_extra.items():
            if not chunks:
                continue
            extra_kwargs[name] = torch.cat(chunks, dim=0).contiguous().to(
                classifier_device,
                non_blocking=True,
                dtype=torch.float32,
            )

        if feature_l2norm:
            x = F.normalize(x, p=2, dim=1)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(classifier_amp and classifier_device.startswith("cuda")),
        ):
            logits = classifier(x, m if classifier_use_meta else None, **extra_kwargs)
            if output_probs:
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
            elif CLASSIFIER_FAST_CONFIDENCE:
                max_logits, pred = torch.max(logits, dim=1)
                conf = torch.exp(max_logits - torch.logsumexp(logits, dim=1))
            else:
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

        preds_split.extend(_split_by_counts(pred.detach().cpu().numpy().astype(np.int64), pending_counts))
        confs_split.extend(_split_by_counts(conf.detach().cpu().numpy().astype(np.float32), pending_counts))
        if output_probs:
            probs_split.extend(_split_by_counts(probs.detach().cpu().numpy().astype(np.float32), pending_counts))
        else:
            probs_split.extend([np.zeros((0, 0), dtype=np.float32) for _ in pending_counts])

        pending_feat.clear()
        pending_meta.clear()
        pending_counts.clear()
        pending_total = 0
        for chunks in pending_extra.values():
            chunks.clear()

    with torch.no_grad():
        for idx, roi_feat in enumerate(roi_feats_by_frame):
            frame_count = int(counts[idx])
            if frame_count <= 0:
                continue
            if pending_counts and (pending_total + frame_count) > int(classifier_batch_size):
                flush_pending()

            pending_feat.append(roi_feat)
            pending_meta.append(meta_by_frame[idx])
            pending_counts.append(frame_count)
            pending_total += frame_count

            for name, by_frame in (extra_feats_by_frame or {}).items():
                if not by_frame:
                    continue
                pending_extra[name].append(by_frame[idx])

        flush_pending()

    if use_cuda and synchronize_timing:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return preds_split, confs_split, probs_split, elapsed


def _render_mask_overlay_frame(
    frame_bgr: np.ndarray,
    instances,
    class_labels: Sequence[str],
    cls_scores: Optional[Sequence[float]],
) -> np.ndarray:
    out = frame_bgr.copy()
    num_inst = len(instances)
    if num_inst == 0:
        return out

    if instances.has("pred_masks"):
        masks = instances.pred_masks
        if isinstance(masks, torch.Tensor):
            if masks.numel() > 0:
                masks_np = masks.cpu().numpy()
            else:
                masks_np = np.empty((0,), dtype=np.uint8)
        else:
            masks_np = np.asarray(masks)

        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]

        if masks_np.ndim == 3 and masks_np.size > 0:
            mask_any = np.any(masks_np > 0, axis=0)
            if np.any(mask_any):
                h, w = out.shape[:2]
                if mask_any.shape[0] != h or mask_any.shape[1] != w:
                    mask_any = cv2.resize(
                        mask_any.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                if OVERLAY_ALPHA > 0:
                    alpha = min(1.0, max(0.0, float(OVERLAY_ALPHA)))
                    color = np.asarray(OVERLAY_COLOR_BGR, dtype=np.float32)
                    pix = out[mask_any].astype(np.float32, copy=False)
                    out[mask_any] = (pix * (1.0 - alpha) + color * alpha).astype(np.uint8)

    if OVERLAY_DRAW_BOXES and instances.has("pred_boxes"):
        boxes_np = instances.pred_boxes.tensor.cpu().numpy()
        det_scores = instances.scores.cpu().numpy() if instances.has("scores") else None
        text_items: list[tuple[str, int, int]] = []

        h, w = out.shape[:2]
        for idx, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box.tolist()
            x1i = int(np.clip(round(x1), 0, w - 1))
            y1i = int(np.clip(round(y1), 0, h - 1))
            x2i = int(np.clip(round(x2), 0, w - 1))
            y2i = int(np.clip(round(y2), 0, h - 1))
            if x2i <= x1i or y2i <= y1i:
                continue

            cv2.rectangle(out, (x1i, y1i), (x2i, y2i), OVERLAY_BOX_COLOR_BGR, OVERLAY_BOX_THICKNESS)

            if OVERLAY_DRAW_CLASS:
                cls_name = str(class_labels[idx]) if idx < len(class_labels) else "unknown"
                txt_parts = [cls_name]
                if OVERLAY_DRAW_CLS_SCORE and cls_scores is not None and idx < len(cls_scores):
                    txt_parts.append(f"c:{float(cls_scores[idx]):.2f}")
                if OVERLAY_DRAW_DET_SCORE and det_scores is not None and idx < len(det_scores):
                    txt_parts.append(f"d:{float(det_scores[idx]):.2f}")
                label = " ".join(txt_parts)

                text_x = x1i
                text_y = y1i - 6
                if text_y < 12:
                    text_y = min(h - 2, y1i + 14)
                text_items.append((label, text_x, text_y))

        out = _draw_overlay_labels(out, text_items)

    return out


def _infer_video_jsonl_with_progress(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    class_names: list[str],
    class_ids: list[int],
    roi_feature_source: str,
    classifier_use_meta: bool,
    classifier_feature_l2norm: bool,
    classifier_meta_feature_set: str,
    need_classifier_box_head_feat: bool,
    need_classifier_box_pooler_feat_expanded: bool,
    need_classifier_mask_pooler_feat: bool,
    need_classifier_box_pooler_gap_expanded: bool,
    need_classifier_mask_pooler_gap: bool,
    video_path: Path,
    output_path: Path,
    overlay_output_path: Optional[Path] = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ASYNC_WRITER:
        writer = base._AsyncJsonlWriter(output_path=output_path, backend=JSON_BACKEND)
    else:
        writer = base._JsonlWriter(output_path=output_path, backend=JSON_BACKEND)

    overlay_writer: Optional[cv2.VideoWriter] = None
    if overlay_output_path is not None:
        overlay_output_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = _open_video_writer(overlay_output_path, fps, width, height)

    total_time = 0.0
    measured_frames = 0
    interval_time = 0.0
    interval_frames = 0
    processed_frames = 0

    total_instances = 0
    cls_total_time = 0.0
    interval_instances = 0
    interval_cls_time = 0.0

    frames_bgr: list[np.ndarray] = []
    frame_ids: list[int] = []
    prefetcher = base._FramePrefetcher(cap, max_queue=PREFETCH_QUEUE, max_frames=MAX_FRAMES) if PREFETCH else None
    serialization_stream: Optional[torch.cuda.Stream] = None
    if (
        ASYNC_SERIALIZATION_TRANSFER
        and overlay_writer is None
        and DEVICE.startswith("cuda")
        and torch.cuda.is_available()
    ):
        serialization_stream = torch.cuda.Stream()

    try:
        while True:
            end_of_stream = False
            if prefetcher is not None:
                item = prefetcher.get()
                if item is prefetcher._stop:
                    end_of_stream = True
                else:
                    frame_id, frame_bgr = item  # type: ignore[misc]
                    frames_bgr.append(frame_bgr)
                    frame_ids.append(frame_id)
                    processed_frames += 1
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    end_of_stream = True
                else:
                    frames_bgr.append(frame_bgr)
                    frame_ids.append(processed_frames)
                    processed_frames += 1
                    if MAX_FRAMES is not None and processed_frames >= MAX_FRAMES:
                        end_of_stream = True

            if len(frames_bgr) < BATCH_SIZE and not end_of_stream:
                continue
            if not frames_bgr:
                break

            if MEASURE and MEASURE_BATCH_SYNC and DEVICE.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter() if MEASURE else None

            inputs, metas = base._prepare_batch_inputs(
                frames_bgr=frames_bgr,
                target_size=TARGET_SIZE,
                device=DEVICE,
                gpu_preprocess_float=GPU_PREPROCESS_FLOAT,
                model_half=MODEL_HALF,
                pin_inputs=PIN_INPUTS,
                pack_inputs=PACK_INPUTS,
            )
            outputs = base._run_model(
                model,
                inputs,
                DEVICE,
                AMP,
                MODEL_HALF,
                postprocess=not RAW_DETECTOR_POSTPROCESS,
            )

            batch_instances: list = []
            batch_roi_feats: list[torch.Tensor] = []
            batch_meta_feats: list[torch.Tensor] = []
            batch_extra_feats: Dict[str, list[torch.Tensor]] = {}
            if need_classifier_box_head_feat:
                batch_extra_feats["box_head_feat"] = []
            if need_classifier_box_pooler_feat_expanded:
                batch_extra_feats["box_pooler_feat_expanded"] = []
            if need_classifier_mask_pooler_feat:
                batch_extra_feats["mask_pooler_feat"] = []
            if need_classifier_box_pooler_gap_expanded:
                batch_extra_feats["box_pooler_gap_expanded"] = []
            if need_classifier_mask_pooler_gap:
                batch_extra_feats["mask_pooler_gap"] = []

            for out_item, (h_src, w_src, lb_meta) in zip(outputs, metas):
                instances = out_item["instances"]
                if EARLY_SCORE_PREFILTER and SCORE_THRESH > 0 and instances.has("scores") and len(instances) > 0:
                    pre_keep = instances.scores >= float(SCORE_THRESH)
                    if pre_keep.numel() == len(instances):
                        instances = instances[pre_keep]
                instances = base.unletterbox_instances(instances, lb_meta, h_src, w_src, TARGET_SIZE)
                if len(instances) > 0:
                    roi_feat = build_roi_feature_tensor_from_instances(instances=instances, feature_source=roi_feature_source)
                    if not isinstance(roi_feat, torch.Tensor):
                        roi_feat = torch.as_tensor(roi_feat)
                    if roi_feat.ndim > 2:
                        roi_feat = torch.flatten(roi_feat, start_dim=1)

                    extra_classifier_feats: Dict[str, torch.Tensor] = {}
                    if need_classifier_box_head_feat:
                        extra_classifier_feats["box_head_feat"] = extract_box_head_features_from_instances(instances)
                    if need_classifier_box_pooler_feat_expanded:
                        extra_classifier_feats["box_pooler_feat_expanded"] = extract_box_pooler_features_expanded_from_instances(
                            instances, mode="raw"
                        )
                    if need_classifier_mask_pooler_feat:
                        extra_classifier_feats["mask_pooler_feat"] = extract_mask_pooler_features_from_instances(
                            instances, mode="raw"
                        )
                    if need_classifier_box_pooler_gap_expanded:
                        extra_classifier_feats["box_pooler_gap_expanded"] = extract_box_pooler_features_expanded_from_instances(
                            instances, mode="gap"
                        )
                    if need_classifier_mask_pooler_gap:
                        extra_classifier_feats["mask_pooler_gap"] = extract_mask_pooler_features_from_instances(
                            instances, mode="gap"
                        )

                    meta_feat = _build_meta_features(
                        instances,
                        width=width,
                        height=height,
                        meta_feature_set=classifier_meta_feature_set,
                    )
                    if roi_feat.shape[0] > 0:
                        batch_roi_feats.append(roi_feat.contiguous())
                        batch_meta_feats.append(meta_feat.contiguous())
                        for name, feat in extra_classifier_feats.items():
                            if name in batch_extra_feats:
                                batch_extra_feats[name].append(feat.contiguous())

                _strip_aux_instance_fields(instances)
                if overlay_writer is None and COMPACT_SERIALIZATION_PAYLOAD:
                    serialization_item = base._build_serialization_payload(instances, height=height, width=width)
                elif overlay_writer is None and TENSOR_SERIALIZATION_PAYLOAD:
                    serialization_item = base._build_tensor_serialization_payload(
                        instances,
                        height=height,
                        width=width,
                        pin_memory=False,
                        non_blocking=False,
                    )
                elif overlay_writer is None and serialization_stream is not None:
                    serialization_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(serialization_stream):
                        serialization_item = base._build_tensor_serialization_payload(
                            instances,
                            height=height,
                            width=width,
                            pin_memory=True,
                            non_blocking=True,
                        )
                elif overlay_writer is None and DEFER_CPU_SERIALIZATION_SNAPSHOT:
                    serialization_item = base._build_device_serialization_payload(
                        instances,
                        height=height,
                        width=width,
                    )
                elif overlay_writer is None and DEFER_CPU_SERIALIZATION:
                    serialization_item = instances
                else:
                    # Overlay rendering still expects a full Instances object on CPU.
                    serialization_item = instances.to("cpu")
                batch_instances.append(serialization_item)

            if batch_roi_feats:
                pred_split_nonempty, score_split_nonempty, probs_split_nonempty, cls_elapsed = _classify_roi_batch(
                    roi_feats_by_frame=batch_roi_feats,
                    meta_by_frame=batch_meta_feats,
                    extra_feats_by_frame=batch_extra_feats if batch_extra_feats else None,
                    classifier=classifier,
                    classifier_device=CLASSIFIER_DEVICE,
                    classifier_amp=CLASSIFIER_AMP,
                    classifier_batch_size=CLASSIFIER_BATCH_SIZE,
                    classifier_use_meta=classifier_use_meta,
                    feature_l2norm=classifier_feature_l2norm,
                    output_probs=CLASSIFIER_OUTPUT_PROBS,
                    synchronize_timing=CLASSIFIER_TIMING_SYNC,
                )
                cls_total_time += float(cls_elapsed)
                interval_cls_time += float(cls_elapsed)
            else:
                pred_split_nonempty, score_split_nonempty, probs_split_nonempty = [], [], []

            if serialization_stream is not None:
                serialization_stream.synchronize()

            # Re-expand predictions to per-frame list (including empty frames)
            pred_by_frame: list[np.ndarray] = []
            cls_score_by_frame: list[np.ndarray] = []
            cls_prob_by_frame: list[np.ndarray] = []
            nonempty_ptr = 0
            for inst in batch_instances:
                n = base._serialization_item_len(inst)
                if n == 0:
                    pred_by_frame.append(np.zeros((0,), dtype=np.int64))
                    cls_score_by_frame.append(np.zeros((0,), dtype=np.float32))
                    cls_prob_by_frame.append(np.zeros((0, 0), dtype=np.float32))
                else:
                    pred_by_frame.append(pred_split_nonempty[nonempty_ptr])
                    cls_score_by_frame.append(score_split_nonempty[nonempty_ptr])
                    cls_prob_by_frame.append(probs_split_nonempty[nonempty_ptr])
                    nonempty_ptr += 1

            for frame_id, frame_bgr, instances, pred_idx, cls_score, cls_probs in zip(
                frame_ids,
                frames_bgr,
                batch_instances,
                pred_by_frame,
                cls_score_by_frame,
                cls_prob_by_frame,
            ):
                inst_list = base._serialization_item_to_json(
                    item=instances,
                    class_name="foreground",
                    score_thresh=0.0,
                    height=height,
                    width=width,
                    mask_approx=MASK_APPROX,
                )

                class_labels: list[str] = []
                for i, item in enumerate(inst_list):
                    idx = int(pred_idx[i]) if i < len(pred_idx) else 0
                    if idx < 0 or idx >= len(class_names):
                        idx = 0

                    item["class_name"] = str(class_names[idx])
                    item["category_id"] = int(class_ids[idx])
                    item["cls_score"] = float(cls_score[i]) if i < len(cls_score) else 0.0
                    if CLASSIFIER_OUTPUT_PROBS and i < len(cls_probs):
                        item["cls_probs"] = [float(v) for v in cls_probs[i].tolist()]

                    class_labels.append(str(class_names[idx]))

                total_instances += int(len(inst_list))
                interval_instances += int(len(inst_list))

                record = {
                    "frame_idx": int(frame_id),
                    "time_sec": float(frame_id / fps),
                    "width": width,
                    "height": height,
                    "instances": inst_list,
                }
                writer.write(record)

                if overlay_writer is not None:
                    overlay_instances = base._materialize_full_masks(instances, height=height, width=width)
                    overlay_frame = _render_mask_overlay_frame(
                        frame_bgr=frame_bgr,
                        instances=overlay_instances,
                        class_labels=class_labels,
                        cls_scores=cls_score.tolist() if len(cls_score) > 0 else None,
                    )
                    overlay_writer.write(overlay_frame)

            if MEASURE and MEASURE_BATCH_SYNC and DEVICE.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            if MEASURE and t0 is not None:
                t1 = time.perf_counter()
                batch_measured = sum(1 for fid in frame_ids if fid >= WARMUP_FRAMES)
                if batch_measured > 0:
                    elapsed = (t1 - t0) * (batch_measured / len(frame_ids))
                    total_time += elapsed
                    measured_frames += batch_measured
                    interval_time += elapsed
                    interval_frames += batch_measured

            if processed_frames % max(1, FLUSH_EVERY) == 0:
                writer.flush()

            if processed_frames % max(1, LOG_EVERY) == 0 or end_of_stream:
                fps_total = (measured_frames / total_time) if total_time > 0 else 0.0
                fps_interval = (interval_frames / interval_time) if interval_time > 0 else 0.0
                cls_ips_total = (total_instances / cls_total_time) if cls_total_time > 0 else 0.0
                cls_ips_interval = (interval_instances / interval_cls_time) if interval_cls_time > 0 else 0.0
                print(
                    f"[INFO] {video_path.name}: processed={processed_frames} "
                    f"fps_total={fps_total:.4f} fps_interval={fps_interval:.4f} "
                    f"instances={total_instances} cls_ips_total={cls_ips_total:.2f} cls_ips_interval={cls_ips_interval:.2f}"
                )
                interval_time = 0.0
                interval_frames = 0
                interval_instances = 0
                interval_cls_time = 0.0

            frames_bgr.clear()
            frame_ids.clear()
            if end_of_stream:
                break
    finally:
        cap.release()
        if prefetcher is not None:
            prefetcher.close()
        writer.flush()
        writer.close()
        if overlay_writer is not None:
            overlay_writer.release()

    print(f"[DONE] wrote {processed_frames} frames -> {output_path}")
    if overlay_output_path is not None:
        print(f"[DONE] wrote overlay video -> {overlay_output_path}")
    if MEASURE and measured_frames > 0:
        fps_e2e = measured_frames / total_time if total_time > 0 else 0.0
        cls_ips = total_instances / cls_total_time if cls_total_time > 0 else 0.0
        print(f"[MEASURE] e2e_fps={fps_e2e:.4f} over {measured_frames} frames (warmup={WARMUP_FRAMES})")
        print(f"[MEASURE] classifier_inst_per_sec={cls_ips:.2f} over {total_instances} instances")


def main() -> int:
    input_path = _as_path(INPUT_DIR).expanduser().resolve()
    output_dir = _as_path(OUTPUT_DIR).expanduser().resolve()
    if output_dir.name == "output":
        output_root = output_dir
        run_name = "manual_run"
    else:
        output_root = output_dir.parent
        run_name = output_dir.name
    jsonl_dir = output_root / "jsonl" / run_name
    video_dir = output_root / "video" / run_name
    checkpoint_path = _as_path(CHECKPOINT).expanduser().resolve()
    config_path = _as_path(CONFIG).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_OVERLAY_VIDEO:
        video_dir.mkdir(parents=True, exist_ok=True)

    model = base._build_model(
        config_path=config_path,
        checkpoint=checkpoint_path,
        num_classes=1,
        target_size=TARGET_SIZE,
        score_thresh=SCORE_THRESH,
        nms_thresh=NMS_THRESH,
        topk_per_image=TOPK,
        device=DEVICE,
        compile_backbone_mode=COMPILE_BACKBONE,
        model_half=MODEL_HALF,
        backbone_half=BACKBONE_HALF,
        onnx_backbone_path=Path(ONNX_BACKBONE).expanduser().resolve() if ONNX_BACKBONE else None,
        onnx_max_batch=ONNX_MAX_BATCH,
        onnx_dtype=ONNX_DTYPE,
        onnx_iobind=ONNX_IOBIND,
        onnx_disable_trt=ONNX_DISABLE_TRT,
        clean_sys_path=False,
        compile_heads_mode=COMPILE_HEADS,
        inductor_disable_cudagraphs=INDUCTOR_DISABLE_CUDAGRAPHS,
        disable_act_checkpoint=DISABLE_ACT_CHECKPOINT,
        prefer_sdpa=PREFER_SDPA,
        drop_block_indices=base.parse_block_indices(DROP_BLOCK_INDICES),
        channels_last=CHANNELS_LAST,
        rpn_pre_nms_multiplier=RPN_PRE_NMS_MULTIPLIER,
    )

    classifier_ckpt = _resolve_classifier_checkpoint()
    classifier, cls_ckpt = classifier_from_checkpoint(classifier_ckpt, map_location="cpu")
    classifier = classifier.to(CLASSIFIER_DEVICE).eval()
    base.MASK_RETR_MODE = MASK_RETR_MODE

    if CLASSIFIER_COMPILE_MODE != "none" and hasattr(torch, "compile"):
        try:
            classifier = torch.compile(classifier, mode=CLASSIFIER_COMPILE_MODE)
            print(f"[INFO] compiled classifier with mode={CLASSIFIER_COMPILE_MODE}")
        except Exception as exc:
            print(f"[WARN] classifier compile failed ({CLASSIFIER_COMPILE_MODE}): {exc}")

    class_names = [str(x) for x in cls_ckpt.get("class_names", [])]
    class_ids = [int(x) for x in cls_ckpt.get("class_ids", [])]
    if not class_names or len(class_names) != len(class_ids):
        raise RuntimeError("classifier checkpoint must contain class_names/class_ids with same length")

    model_cfg = cls_ckpt.get("model_cfg") or {}
    classifier_use_meta = bool(model_cfg.get("use_meta", True))
    classifier_feature_l2norm = bool(model_cfg.get("feature_l2norm", False))
    classifier_meta_feature_set = normalize_meta_feature_set(str(model_cfg.get("meta_feature_set", "legacy_iou")))

    if ROI_FEATURE_SOURCE == "auto":
        ckpt_feature_source = str(model_cfg.get("feature_source", "box_head"))
        roi_feature_source = normalize_roi_feature_source(ckpt_feature_source)
    else:
        roi_feature_source = normalize_roi_feature_source(ROI_FEATURE_SOURCE)

    need_box_head_feat, need_pooler_feat = roi_feature_source_needs(roi_feature_source)
    model_type = str(model_cfg.get("model_type", "mlp")).strip().lower()
    rich_spatial_types = {
        "rich_spatial_fusion",
        "roi_rich_spatial_fusion",
        "rich_spatial_gated_fusion",
        "roi_rich_spatial_gated_fusion",
        "rich_spatial_attn_fusion",
        "roi_rich_spatial_attn_fusion",
        "rich_spatial_mask_guided_fusion",
        "roi_rich_spatial_mask_guided_fusion",
    }
    need_classifier_box_head_feat = bool(model_type in rich_spatial_types)
    need_classifier_box_pooler_feat_expanded = bool(model_type in rich_spatial_types)
    need_classifier_mask_pooler_feat = bool(model_type in rich_spatial_types)
    need_classifier_box_pooler_gap_expanded = False
    need_classifier_mask_pooler_gap = False

    if model_type in ("rich_gap_fusion", "roi_rich_gap_fusion"):
        need_classifier_box_head_feat = bool(model_cfg.get("use_box_head_feat", True))
        need_classifier_box_pooler_gap_expanded = bool(model_cfg.get("use_box_pooler_gap_expanded", True))
        need_classifier_mask_pooler_gap = bool(model_cfg.get("use_mask_pooler_gap", True))

    need_box_head_export = bool(need_box_head_feat or need_classifier_box_head_feat)
    need_pooler_export = bool(
        need_pooler_feat or need_classifier_box_pooler_feat_expanded or need_classifier_box_pooler_gap_expanded
    )
    need_pooler_expanded_export = bool(
        need_classifier_box_pooler_feat_expanded or need_classifier_box_pooler_gap_expanded
    )
    need_mask_pooler_export = bool(need_classifier_mask_pooler_feat or need_classifier_mask_pooler_gap)
    if hasattr(model, "roi_heads"):
        setattr(model.roi_heads, "return_box_features", need_box_head_export)
        setattr(model.roi_heads, "return_box_pooler_features", need_pooler_export)
        setattr(model.roi_heads, "return_box_pooler_features_expanded", need_pooler_expanded_export)
        setattr(model.roi_heads, "return_mask_pooler_features", need_mask_pooler_export)
        print(
            "[INFO] enabled ROI feature export from detector.roi_heads "
            f"(source={roi_feature_source}, box_head={need_box_head_export}, pooler={need_pooler_export}, "
            f"pooler_expanded={need_pooler_expanded_export}, mask_pooler={need_mask_pooler_export})"
        )

    print(
        f"[INFO] classifier={classifier_ckpt} "
        f"classes={len(class_names)} use_meta={classifier_use_meta} "
        f"feature_l2norm={classifier_feature_l2norm} meta={classifier_meta_feature_set} "
        f"model_type={model_type} classifier_device={CLASSIFIER_DEVICE}"
    )

    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    video_paths = base._collect_videos(input_path, RECURSIVE)
    if not video_paths:
        raise RuntimeError(f"No videos found under: {input_path}")

    processed = 0
    for vid_path in video_paths:
        out_name = vid_path.stem + ".jsonl"
        out_path = jsonl_dir / out_name
        overlay_path = video_dir / f"{vid_path.stem}{OVERLAY_SUFFIX}" if SAVE_OVERLAY_VIDEO else None
        all_outputs_exist = out_path.exists() and (overlay_path is None or overlay_path.exists())
        if all_outputs_exist and not OVERWRITE:
            print(f"[SKIP] {vid_path.name} (exists)")
            continue

        print(f"[START] {vid_path.name}")
        _infer_video_jsonl_with_progress(
            model=model,
            classifier=classifier,
            class_names=class_names,
            class_ids=class_ids,
            roi_feature_source=roi_feature_source,
            classifier_use_meta=classifier_use_meta,
            classifier_feature_l2norm=classifier_feature_l2norm,
            classifier_meta_feature_set=classifier_meta_feature_set,
            need_classifier_box_head_feat=need_classifier_box_head_feat,
            need_classifier_box_pooler_feat_expanded=need_classifier_box_pooler_feat_expanded,
            need_classifier_mask_pooler_feat=need_classifier_mask_pooler_feat,
            need_classifier_box_pooler_gap_expanded=need_classifier_box_pooler_gap_expanded,
            need_classifier_mask_pooler_gap=need_classifier_mask_pooler_gap,
            video_path=vid_path,
            output_path=out_path,
            overlay_output_path=overlay_path,
        )
        processed += 1

    print(f"[DONE] processed {processed} videos -> jsonl:{jsonl_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
