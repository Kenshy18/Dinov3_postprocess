#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast DINOv3 Cascade video inference with optional ROI classifier and JSONL output.

Default acceleration matches infer_video_trt_bf16_cascade1_rpn100_40.py:

- native TensorRT BF16 DINOv3 backbone
- Cascade box head reduced to 1 stage
- RPN test top-k reduced to pre/post NMS 100/40
- batch size 8
- BF16 autocast
- decode/preprocess prefetch

Output is JSONL by default. Overlay MP4 can be enabled with --write-overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
BUNDLE_CHECKPOINTS = REPO_ROOT / "checkpoints"
DEFAULT_TRT_ENGINE = (
    BUNDLE_CHECKPOINTS
    / "trt"
    / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"
)
DEFAULT_CLASSIFIER_CKPT = BUNDLE_CHECKPOINTS / "classifier" / "best.pt"
DEFAULT_DINOV3_WEIGHTS = (
    BUNDLE_CHECKPOINTS
    / "dinov3"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import orjson  # type: ignore
except Exception:
    orjson = None

from infer_images_singleclass import (  # noqa: E402
    DEFAULT_CONFIG,
    _AsyncBatchProducer,
    _CudaBatchPrefetcher,
    _autocast_context,
    _build_model,
    _parse_target_size,
    _resolve_checkpoint,
    _unletterbox_instances,
    _unpack_size,
    unified_paths,
)
try:
    from backend.classifiers.dinov3_roi.runtime.two_stage_roi_classifier import (  # noqa: E402
        classifier_from_checkpoint,
        extract_box_head_features_from_instances,
        extract_box_pooler_features_expanded_from_instances,
        extract_box_pooler_features_from_instances,
        extract_mask_pooler_features_from_instances,
    )

    CLASSIFIER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - only used when classifier deps are unavailable
    classifier_from_checkpoint = None  # type: ignore[assignment]
    extract_box_head_features_from_instances = None  # type: ignore[assignment]
    extract_box_pooler_features_expanded_from_instances = None  # type: ignore[assignment]
    extract_box_pooler_features_from_instances = None  # type: ignore[assignment]
    extract_mask_pooler_features_from_instances = None  # type: ignore[assignment]
    CLASSIFIER_IMPORT_ERROR = exc


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
AUX_INSTANCE_FIELDS = {
    "pred_box_features",
    "pred_box_pooler_features",
    "pred_box_pooler_features_expanded",
    "pred_mask_pooler_features",
    "pred_multiclass_logits",
}
CLASS_COLORS = [
    (0, 220, 255),
    (80, 180, 255),
    (80, 255, 120),
    (255, 180, 80),
    (220, 120, 255),
]


@dataclass(frozen=True)
class RoiExportRequirements:
    return_box_features: bool = False
    return_box_pooler_features: bool = False
    return_box_pooler_features_expanded: bool = False
    return_mask_pooler_features: bool = False


def _torch_load_metadata(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _classifier_export_requirements(raw_checkpoint: dict) -> RoiExportRequirements:
    cfg = raw_checkpoint.get("model_cfg") or {}
    model_type = str(cfg.get("model_type", "")).strip().lower()
    if model_type in {
        "rich_spatial_attn_no_expanded_fusion",
        "roi_rich_spatial_attn_no_expanded_fusion",
    }:
        return RoiExportRequirements(
            return_box_features=True,
            return_box_pooler_features=True,
            return_box_pooler_features_expanded=False,
            return_mask_pooler_features=True,
        )
    if model_type in {
        "rich_spatial_fusion",
        "roi_rich_spatial_fusion",
        "rich_spatial_gated_fusion",
        "roi_rich_spatial_gated_fusion",
        "rich_spatial_attn_fusion",
        "roi_rich_spatial_attn_fusion",
        "rich_spatial_mask_guided_fusion",
        "roi_rich_spatial_mask_guided_fusion",
    }:
        return RoiExportRequirements(
            return_box_features=True,
            return_box_pooler_features=True,
            return_box_pooler_features_expanded=True,
            return_mask_pooler_features=True,
        )
    if model_type in {"rich_gap_fusion", "roi_rich_gap_fusion"}:
        feature_source = str(cfg.get("feature_source", "")).strip().lower()
        uses_pooler_roi = bool(cfg.get("use_roi_feat", True)) and feature_source in {
            "pooler_flatten",
            "pooler_gap",
            "",
        }
        return RoiExportRequirements(
            return_box_features=bool(cfg.get("use_box_head_feat", False)),
            return_box_pooler_features=uses_pooler_roi,
            return_box_pooler_features_expanded=bool(cfg.get("use_box_pooler_gap_expanded", False)),
            return_mask_pooler_features=bool(cfg.get("use_mask_pooler_gap", False)),
        )
    return RoiExportRequirements(
        return_box_features=True,
        return_box_pooler_features=True,
        return_box_pooler_features_expanded=True,
        return_mask_pooler_features=True,
    )


def _configure_roi_feature_export(model: torch.nn.Module, requirements: RoiExportRequirements) -> None:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None:
        raise RuntimeError("model.roi_heads not found")
    setattr(roi_heads, "return_box_features", bool(requirements.return_box_features))
    setattr(roi_heads, "return_box_pooler_features", bool(requirements.return_box_pooler_features))
    setattr(
        roi_heads,
        "return_box_pooler_features_expanded",
        bool(requirements.return_box_pooler_features_expanded),
    )
    setattr(roi_heads, "box_pooler_expanded_scale", 2.0)
    setattr(roi_heads, "return_mask_pooler_features", bool(requirements.return_mask_pooler_features))


def _collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTS:
            return [input_path]
        raise RuntimeError(f"Input file is not a video: {input_path}")
    if not input_path.is_dir():
        raise RuntimeError(f"Input path not found: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _read_video_meta(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def _build_geo_v2_meta(instances, target_size: int | tuple[int, int]) -> torch.Tensor:
    if not instances.has("scores") or not instances.has("pred_boxes"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.empty((0, 5), device=device, dtype=torch.float32)
    scores = instances.scores.float()
    boxes = instances.pred_boxes.tensor.float()
    n = int(boxes.shape[0])
    if n == 0:
        return torch.empty((0, 5), device=boxes.device, dtype=torch.float32)

    target_h, target_w = _unpack_size(target_size)
    img_area = float(max(1, int(target_h) * int(target_w)))
    w = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
    h = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    bbox_area = w * h
    bbox_area_ratio = bbox_area / img_area
    log_aspect_ratio = torch.log((w + 1e-6) / (h + 1e-6))

    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        mask_area = masks.flatten(1).float().sum(dim=1)
    else:
        mask_area = torch.zeros((n,), device=boxes.device, dtype=torch.float32)
    mask_area_ratio = mask_area / img_area
    mask_over_bbox = mask_area / torch.clamp(bbox_area, min=1e-6)
    return torch.stack(
        [scores, bbox_area_ratio, mask_area_ratio, log_aspect_ratio, mask_over_bbox],
        dim=1,
    )


def _classify_outputs_batched(
    classifier: torch.nn.Module,
    outputs: list[dict[str, object]],
    target_size: int | tuple[int, int],
    requirements: RoiExportRequirements,
    pad_to: int = 0,
    pad_multiple: int = 0,
) -> int:
    if (
        extract_box_pooler_features_from_instances is None
        or extract_box_head_features_from_instances is None
        or extract_box_pooler_features_expanded_from_instances is None
        or extract_mask_pooler_features_from_instances is None
    ):
        raise RuntimeError(f"classifier dependencies are unavailable: {CLASSIFIER_IMPORT_ERROR}")

    records: list[tuple[object, int]] = []
    roi_feats: list[torch.Tensor] = []
    metas: list[torch.Tensor] = []
    box_heads: list[torch.Tensor] = []
    expanded_feats: list[torch.Tensor] = []
    mask_feats: list[torch.Tensor] = []

    for output in outputs:
        instances = output["instances"]
        n = int(len(instances))
        if n == 0:
            continue
        if requirements.return_box_pooler_features:
            box_pooler = extract_box_pooler_features_from_instances(instances, mode="raw")
            roi_feats.append(box_pooler.flatten(start_dim=1))
        metas.append(_build_geo_v2_meta(instances, target_size=target_size))
        if requirements.return_box_features:
            box_heads.append(extract_box_head_features_from_instances(instances))
        if requirements.return_box_pooler_features_expanded:
            expanded_feats.append(extract_box_pooler_features_expanded_from_instances(instances, mode="raw"))
        if requirements.return_mask_pooler_features:
            mask_feats.append(extract_mask_pooler_features_from_instances(instances, mode="raw"))
        records.append((instances, n))

    total = sum(n for _, n in records)
    if total == 0:
        return 0
    if not roi_feats:
        raise RuntimeError("current classifier path requires box pooler ROI features")

    roi_feat = torch.cat(roi_feats, dim=0)
    meta = torch.cat(metas, dim=0)
    kwargs: dict[str, torch.Tensor] = {}
    if box_heads:
        kwargs["box_head_feat"] = torch.cat(box_heads, dim=0)
    if expanded_feats:
        kwargs["box_pooler_feat_expanded"] = torch.cat(expanded_feats, dim=0)
    if mask_feats:
        kwargs["mask_pooler_feat"] = torch.cat(mask_feats, dim=0)

    target_total = int(total)
    if pad_to > 0:
        target_total = max(target_total, int(pad_to))
    if pad_multiple > 0 and target_total > 0:
        m = int(pad_multiple)
        target_total = ((target_total + m - 1) // m) * m
    if target_total > total:
        pad_n = int(target_total - total)

        def _pad_first_dim(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, x.new_zeros((pad_n, *x.shape[1:]))], dim=0)

        roi_feat = _pad_first_dim(roi_feat)
        meta = _pad_first_dim(meta)
        kwargs = {name: _pad_first_dim(value) for name, value in kwargs.items()}

    logits = classifier(roi_feat, meta, **kwargs)[:total]
    probs = torch.softmax(logits.float(), dim=1)
    class_scores, classes = torch.max(probs, dim=1)

    offset = 0
    for instances, n in records:
        next_offset = offset + n
        instances.pred_multiclass_logits = logits[offset:next_offset]
        instances.pred_multiclass_classes = classes[offset:next_offset]
        instances.pred_multiclass_scores = class_scores[offset:next_offset]
        offset = next_offset
    return int(total)


def _drop_aux_instance_fields(instances) -> None:
    fields = getattr(instances, "_fields", None)
    if not isinstance(fields, dict):
        return
    for name in AUX_INSTANCE_FIELDS:
        fields.pop(name, None)


def _mask_to_polygons(mask: np.ndarray, approx_mode: str) -> list[list[float]]:
    mask = np.ascontiguousarray(mask.astype("uint8", copy=False))
    chain_mode = cv2.CHAIN_APPROX_SIMPLE if approx_mode == "simple" else cv2.CHAIN_APPROX_NONE
    res = cv2.findContours(mask, cv2.RETR_CCOMP, chain_mode)
    hierarchy = res[-1]
    if hierarchy is None:
        return []
    polygons: list[list[float]] = []
    for contour in res[-2]:
        flat = contour.flatten()
        if len(flat) >= 6:
            polygons.append((flat.astype(np.float32) + 0.5).tolist())
    return polygons


def _instances_to_json(
    instances,
    *,
    class_names: list[str],
    class_ids: list[int],
    score_thresh: float,
    mask_approx: str,
) -> list[dict[str, object]]:
    if instances.has("scores") and score_thresh > 0:
        instances = instances[instances.scores >= score_thresh]
    if len(instances) == 0:
        return []

    boxes = instances.pred_boxes.tensor.cpu().numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.float().cpu().numpy() if instances.has("scores") else None
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
    class_idx = (
        instances.pred_multiclass_classes.long().cpu().numpy()
        if instances.has("pred_multiclass_classes")
        else None
    )
    class_scores = (
        instances.pred_multiclass_scores.float().cpu().numpy()
        if instances.has("pred_multiclass_scores")
        else None
    )

    rows: list[dict[str, object]] = []
    for det_idx in range(len(instances)):
        if class_idx is not None:
            idx = int(class_idx[det_idx])
            label = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            category_id = int(class_ids[idx]) if 0 <= idx < len(class_ids) else idx
        else:
            idx = 0
            label = class_names[0] if class_names else "foreground"
            category_id = int(class_ids[0]) if class_ids else 0

        item: dict[str, object] = {
            "label": label,
            "class_name": label,
            "category_id": category_id,
            "category_index": idx,
        }
        if boxes is not None:
            x1, y1, x2, y2 = boxes[det_idx].tolist()
            item["bbox_xyxy"] = [float(x1), float(y1), float(x2), float(y2)]
            item["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if scores is not None:
            item["score"] = float(scores[det_idx])
            item["detector_score"] = float(scores[det_idx])
        if class_scores is not None:
            item["class_score"] = float(class_scores[det_idx])
        if masks is not None:
            mask = masks[det_idx]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask_bin = mask.astype(np.uint8, copy=False)
            if mask_bin.max() > 1:
                mask_bin = (mask_bin > 0.5).astype(np.uint8)
            polygons = _mask_to_polygons(mask_bin, mask_approx)
            item["segmentation"] = polygons
            item["polygons"] = polygons
        rows.append(item)
    return rows


def _draw_overlay(
    image_bgr: np.ndarray,
    instances,
    *,
    class_names: list[str],
    score_thresh: float,
) -> np.ndarray:
    if instances.has("scores") and score_thresh > 0:
        instances = instances[instances.scores >= score_thresh]
    img = image_bgr.copy()
    boxes = instances.pred_boxes.tensor.float().cpu().numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.float().cpu().numpy() if instances.has("scores") else None
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
    class_idx = (
        instances.pred_multiclass_classes.long().cpu().numpy()
        if instances.has("pred_multiclass_classes")
        else None
    )
    class_scores = (
        instances.pred_multiclass_scores.float().cpu().numpy()
        if instances.has("pred_multiclass_scores")
        else None
    )
    for idx in range(len(instances)):
        cls = int(class_idx[idx]) if class_idx is not None else 0
        label = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        color_arr = np.asarray(color, dtype=np.float32)
        if masks is not None:
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask_bool = mask.astype(bool, copy=False)
            if mask_bool.any():
                pixels = img[mask_bool].astype(np.float32, copy=False)
                img[mask_bool] = np.rint(pixels * 0.6 + color_arr * 0.4).clip(0, 255).astype(np.uint8)
        if boxes is not None:
            x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            det_s = float(scores[idx]) if scores is not None else 0.0
            if class_scores is not None:
                cls_s = float(class_scores[idx])
                text = f"{label} {det_s:.2f}/{cls_s:.2f}"
            else:
                text = f"{label} {det_s:.2f}"
            cv2.putText(
                img,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return img


class _JsonlWriter:
    def __init__(self, path: Path, backend: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = str(backend)
        if self.backend == "orjson":
            if orjson is None:
                raise RuntimeError("orjson backend requested but orjson is not installed")
            self._bin = path.open("wb")
            self._txt = None
        else:
            self._txt = path.open("w", encoding="utf-8")
            self._bin = None

    def write(self, record: dict[str, object]) -> None:
        if self.backend == "orjson":
            self._bin.write(orjson.dumps(record))  # type: ignore[union-attr]
            self._bin.write(b"\n")  # type: ignore[union-attr]
        else:
            self._txt.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]

    def close(self) -> None:
        if self.backend == "orjson":
            self._bin.flush()  # type: ignore[union-attr]
            self._bin.close()  # type: ignore[union-attr]
        else:
            self._txt.flush()  # type: ignore[union-attr]
            self._txt.close()  # type: ignore[union-attr]


class _AsyncJsonlWriter:
    def __init__(self, path: Path, backend: str, queue_size: int) -> None:
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._writer = _JsonlWriter(path, backend)
        self._thread = threading.Thread(target=self._worker, name="jsonl_writer", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._writer.write(item)
            finally:
                self._queue.task_done()

    def write(self, record: dict[str, object]) -> None:
        self._queue.put(record)

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        self._writer.close()


def _make_writer(path: Path, backend: str, async_writer: bool, queue_size: int):
    if async_writer:
        return _AsyncJsonlWriter(path, backend, queue_size)
    return _JsonlWriter(path, backend)


def _infer_one_video(
    *,
    model: torch.nn.Module,
    classifier: torch.nn.Module | None,
    classifier_requirements: RoiExportRequirements,
    video_path: Path,
    jsonl_path: Path,
    overlay_path: Path | None,
    class_names: list[str],
    class_ids: list[int],
    target_size: int | tuple[int, int],
    device: str,
    amp: bool,
    amp_dtype: str,
    score_thresh: float,
    max_frames: int | None,
    warmup_frames: int,
    batch_size: int,
    io_prefetch: bool,
    prefetch_batches: int,
    gpu_prefetch: bool,
    json_backend: str,
    async_writer: bool,
    writer_queue_size: int,
    mask_approx: str,
    classifier_pad_to: int,
    classifier_pad_multiple: int,
    overlay_fourcc: str,
) -> dict[str, object]:
    if not io_prefetch:
        raise NotImplementedError("this script currently expects --io-prefetch")

    video_meta = _read_video_meta(video_path)
    fps = float(video_meta["fps"]) if float(video_meta["fps"]) > 0 else 30.0
    writer = _make_writer(jsonl_path, json_backend, async_writer, writer_queue_size)
    overlay_writer = None
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*overlay_fourcc),
            fps,
            (int(video_meta["width"]), int(video_meta["height"])),
            True,
        )
        if not overlay_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {overlay_path}")

    processed = 0
    detections = 0
    measured_frames = 0
    measured_time = 0.0
    pin_memory = device.startswith("cuda") and torch.cuda.is_available()

    producer = _AsyncBatchProducer(
        video_path=video_path,
        target_size=target_size,
        batch_size=max(1, int(batch_size)),
        max_frames=max_frames,
        pin_memory=pin_memory,
        prefetch_batches=max(1, int(prefetch_batches)),
    )
    producer.start()
    prefetcher = _CudaBatchPrefetcher(producer, device=device, enabled=gpu_prefetch)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    try:
        with torch.inference_mode():
            source: Iterable = iter(lambda: prefetcher.next(), None)
            for batch in source:
                if batch is None:
                    break
                frame_ids = list(range(processed, processed + len(batch.items)))
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_start = time.perf_counter()
                with _autocast_context(device, amp, amp_dtype):
                    outputs = model(batch.inputs)
                    if classifier is not None:
                        detections += _classify_outputs_batched(
                            classifier,
                            outputs,
                            target_size,
                            classifier_requirements,
                            pad_to=max(0, int(classifier_pad_to)),
                            pad_multiple=max(0, int(classifier_pad_multiple)),
                        )
                    else:
                        detections += sum(int(len(output["instances"])) for output in outputs)

                for output, frame_id, (image_bgr, meta) in zip(outputs, frame_ids, batch.items):
                    instances = output["instances"]
                    _drop_aux_instance_fields(instances)
                    instances = _unletterbox_instances(
                        instances,
                        meta["lb"],  # type: ignore[arg-type]
                        int(meta["orig_h"]),
                        int(meta["orig_w"]),
                        target_size,
                    ).to("cpu")
                    det_rows = _instances_to_json(
                        instances,
                        class_names=class_names,
                        class_ids=class_ids,
                        score_thresh=score_thresh,
                        mask_approx=mask_approx,
                    )
                    writer.write(
                        {
                            "frame_index": int(frame_id),
                            "frame_idx": int(frame_id),
                            "time_sec": float(frame_id / fps),
                            "width": int(video_meta["width"]),
                            "height": int(video_meta["height"]),
                            "detections": det_rows,
                            "instances": det_rows,
                        }
                    )
                    if overlay_writer is not None:
                        overlay_writer.write(
                            _draw_overlay(
                                image_bgr,
                                instances,
                                class_names=class_names,
                                score_thresh=score_thresh,
                            )
                        )

                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_elapsed = time.perf_counter() - batch_start
                batch_measured = sum(1 for frame_id in frame_ids if frame_id >= warmup_frames)
                if batch_measured > 0:
                    measured_time += batch_elapsed * (batch_measured / len(frame_ids))
                    measured_frames += batch_measured
                processed += len(batch.items)
    finally:
        prefetcher.close()
        writer.close()
        if overlay_writer is not None:
            overlay_writer.release()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_elapsed = time.perf_counter() - wall_start
    wall_fps = processed / wall_elapsed if wall_elapsed > 0 else 0.0
    measured_fps = measured_frames / measured_time if measured_time > 0 else 0.0
    return {
        "video": str(video_path),
        "output_jsonl": str(jsonl_path),
        "output_overlay": None if overlay_path is None else str(overlay_path),
        "video_meta": video_meta,
        "processed_frames": int(processed),
        "detections": int(detections),
        "detections_per_frame": float(detections / max(1, processed)),
        "e2e_fps": float(wall_fps),
        "e2e_ms_per_frame": float(1000.0 / wall_fps) if wall_fps > 0 else 0.0,
        "wall_elapsed_sec": float(wall_elapsed),
        "wall_fps": float(wall_fps),
        "wall_ms_per_frame": float(1000.0 / wall_fps) if wall_fps > 0 else 0.0,
        "warmup_frames": int(warmup_frames),
        "measured_frames": int(measured_frames),
        "measured_time_sec": float(measured_time),
        "compute_fps": float(measured_fps),
        "compute_ms_per_frame": float(1000.0 / measured_fps) if measured_fps > 0 else 0.0,
        "measured_fps": float(measured_fps),
        "measured_ms_per_frame": float(1000.0 / measured_fps) if measured_fps > 0 else 0.0,
        "fps_note": (
            "e2e_fps/wall_fps includes decode/preprocess wait, warmup frames, and writer flush. "
            "compute_fps/measured_fps excludes warmup and starts timing after each prefetched batch is available."
        ),
        "jsonl_size_bytes": int(jsonl_path.stat().st_size if jsonl_path.is_file() else 0),
        "overlay_size_bytes": int(overlay_path.stat().st_size if overlay_path is not None and overlay_path.is_file() else 0),
    }


def _format_size_for_summary(target_size: int | tuple[int, int]) -> str:
    h, w = _unpack_size(target_size)
    return f"{w}x{h}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast DINOv3 Cascade + optional ROI classifier JSONL inference")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-checkpoint", default=str(DEFAULT_CLASSIFIER_CKPT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backbone-weights", default=None)
    parser.add_argument("--trt-backbone-engine", default=str(DEFAULT_TRT_ENGINE))
    parser.add_argument("--target-size", type=_parse_target_size, default=(1280, 720))
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--rpn-pre-nms-topk-test", type=int, default=100)
    parser.add_argument("--rpn-post-nms-topk-test", type=int, default=40)
    parser.add_argument("--rpn-nms-thresh", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=300)
    parser.add_argument("--io-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-batches", type=int, default=1)
    parser.add_argument("--gpu-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json-backend", choices=["json", "orjson"], default="orjson")
    parser.add_argument("--async-writer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--writer-queue-size", type=int, default=512)
    parser.add_argument("--mask-approx", choices=["none", "simple"], default="none")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--overlay-dir", default=None)
    parser.add_argument("--overlay-ext", default=".mp4")
    parser.add_argument("--overlay-fourcc", default="mp4v")
    parser.add_argument("--compile-classifier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--classifier-compile-mode", default="reduce-overhead")
    parser.add_argument("--classifier-compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-pad-to", type=int, default=0)
    parser.add_argument("--classifier-pad-multiple", type=int, default=0)
    args = parser.parse_args()

    os.environ["EVA_CASCADE_NUM_STAGES"] = "1"
    os.environ.setdefault("EVA_PYRAMID_CHANNELS_LAST", "1")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    jsonl_dir = output_dir / "jsonl"
    overlay_dir = Path(args.overlay_dir).expanduser().resolve() if args.overlay_dir else output_dir / "overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    if args.write_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    classifier_enabled = bool(args.classifier)
    classifier_ckpt = Path(args.classifier_checkpoint).expanduser().resolve() if classifier_enabled else None
    trt_engine = Path(args.trt_backbone_engine).expanduser().resolve() if args.trt_backbone_engine else None
    if classifier_enabled and classifier_ckpt is not None and not classifier_ckpt.is_file():
        raise FileNotFoundError(classifier_ckpt)
    if trt_engine is not None and not trt_engine.is_file():
        raise FileNotFoundError(trt_engine)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    backbone_weights = args.backbone_weights or (
        str(DEFAULT_DINOV3_WEIGHTS) if DEFAULT_DINOV3_WEIGHTS.is_file() else unified_paths.DINOv3_WEIGHTS
    )

    raw_classifier: dict[str, object] = {}
    classifier_cfg: dict[str, object] = {}
    classifier: torch.nn.Module | None = None
    loaded: dict[str, object] = {}
    if classifier_enabled and classifier_ckpt is not None:
        raw_classifier = _torch_load_metadata(classifier_ckpt)
        classifier_requirements = _classifier_export_requirements(raw_classifier)
        classifier_cfg = raw_classifier.get("model_cfg") or {}
        class_names = [str(x) for x in raw_classifier.get("class_names", ["foreground"])]
        class_ids = [int(x) for x in raw_classifier.get("class_ids", list(range(len(class_names))))]
    else:
        classifier_requirements = RoiExportRequirements()
        class_names = ["foreground"]
        class_ids = [0]

    print(f"[INFO] checkpoint: {checkpoint}")
    if classifier_enabled:
        print(f"[INFO] classifier: {classifier_ckpt}")
        print(f"[INFO] classifier model_type: {classifier_cfg.get('model_type')}")
        print(f"[INFO] classifier val macro_f1: {raw_classifier.get('val_metrics', {}).get('macro_f1')}")
    else:
        print("[INFO] classifier: disabled")
    print(f"[INFO] roi export requirements: {classifier_requirements}")

    model = _build_model(
        checkpoint=checkpoint,
        target_size=args.target_size,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk_per_image=args.topk,
        device=args.device,
        config_path=Path(args.config).expanduser().resolve(),
        backbone_weights=backbone_weights,
        rpn_pre_nms_topk_test=args.rpn_pre_nms_topk_test,
        rpn_post_nms_topk_test=args.rpn_post_nms_topk_test,
        rpn_nms_thresh=args.rpn_nms_thresh,
        trt_backbone_engine=trt_engine,
    )
    _configure_roi_feature_export(model, classifier_requirements)

    if classifier_enabled and classifier_ckpt is not None:
        if classifier_from_checkpoint is None:
            raise RuntimeError(f"classifier dependencies are unavailable: {CLASSIFIER_IMPORT_ERROR}")
        classifier, loaded = classifier_from_checkpoint(classifier_ckpt, map_location=args.device)
        classifier.to(args.device).eval()
        if args.compile_classifier:
            classifier = torch.compile(
                classifier,
                mode=str(args.classifier_compile_mode),
                dynamic=bool(args.classifier_compile_dynamic),
            )
        print(
            f"[INFO] classifier loaded: epoch={loaded.get('epoch')} "
            f"macro_f1={loaded.get('val_metrics', {}).get('macro_f1')}"
        )

    videos = _collect_videos(input_path, args.recursive)
    if not videos:
        raise RuntimeError(f"No videos found under: {input_path}")

    runs: list[dict[str, object]] = []
    for video_path in videos:
        jsonl_path = jsonl_dir / f"{video_path.stem}.jsonl"
        overlay_path = overlay_dir / f"{video_path.stem}{args.overlay_ext}" if args.write_overlay else None
        if jsonl_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {jsonl_path}")
            continue
        if overlay_path is not None and overlay_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {overlay_path}")
            continue

        result = _infer_one_video(
            model=model,
            classifier=classifier,
            classifier_requirements=classifier_requirements,
            video_path=video_path,
            jsonl_path=jsonl_path,
            overlay_path=overlay_path,
            class_names=class_names,
            class_ids=class_ids,
            target_size=args.target_size,
            device=args.device,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            score_thresh=args.score_thresh,
            max_frames=args.max_frames,
            warmup_frames=max(0, int(args.warmup_frames)),
            batch_size=max(1, int(args.batch_size)),
            io_prefetch=args.io_prefetch,
            prefetch_batches=max(1, int(args.prefetch_batches)),
            gpu_prefetch=args.gpu_prefetch,
            json_backend=args.json_backend,
            async_writer=args.async_writer,
            writer_queue_size=max(1, int(args.writer_queue_size)),
            mask_approx=args.mask_approx,
            classifier_pad_to=max(0, int(args.classifier_pad_to)),
            classifier_pad_multiple=max(0, int(args.classifier_pad_multiple)),
            overlay_fourcc=str(args.overlay_fourcc),
        )
        runs.append(result)
        print(
            f"[DONE] {video_path.name}: e2e_fps={result['e2e_fps']:.2f} "
            f"compute_fps={result['compute_fps']:.2f} "
            f"det/frame={result['detections_per_frame']:.3f}"
        )

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "jsonl_dir": str(jsonl_dir),
        "overlay_dir": str(overlay_dir) if args.write_overlay else None,
        "write_overlay": bool(args.write_overlay),
        "checkpoint": str(checkpoint),
        "classifier_enabled": bool(classifier_enabled),
        "classifier_checkpoint": str(classifier_ckpt) if classifier_ckpt is not None else None,
        "classifier_model_type": str(classifier_cfg.get("model_type", "")) if classifier_enabled else None,
        "classifier_val_macro_f1": (
            raw_classifier.get("val_metrics", {}).get("macro_f1") if classifier_enabled else None
        ),
        "class_names": class_names,
        "class_ids": class_ids,
        "roi_export_requirements": asdict(classifier_requirements),
        "target_size": _format_size_for_summary(args.target_size),
        "score_thresh": float(args.score_thresh),
        "nms_thresh": float(args.nms_thresh),
        "topk": int(args.topk),
        "rpn_pre_nms_topk_test": int(args.rpn_pre_nms_topk_test),
        "rpn_post_nms_topk_test": int(args.rpn_post_nms_topk_test),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "batch_size": int(args.batch_size),
        "max_frames": args.max_frames,
        "warmup_frames": int(args.warmup_frames),
        "json_backend": str(args.json_backend),
        "async_writer": bool(args.async_writer),
        "mask_approx": str(args.mask_approx),
        "compile_classifier": bool(args.compile_classifier),
        "classifier_pad_to": int(args.classifier_pad_to),
        "classifier_pad_multiple": int(args.classifier_pad_multiple),
        "runs": runs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
