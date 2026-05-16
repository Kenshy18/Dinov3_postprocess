#!/usr/bin/env python3
"""DINOv3 + Co-DINO video inference normalized to the shared detector JSONL contract.

This runtime intentionally keeps Co-DINO model execution in the canonical
Co-DINO working tree and owns only the integration boundary for this
postprocess repository:

video -> Co-DINO detections/masks -> optional ROI classifier -> JSONL + summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.detectors.jsonl_writer import make_jsonl_writer  # noqa: E402

BUNDLE_CHECKPOINTS = REPO_ROOT / "checkpoints"

EXTERNAL_ROOT = Path(
    os.environ.get("CODINO_EXTERNAL_ROOT", "/home/kenke/workspace/CV/unified_training_codino_eva02")
).expanduser()
EXTERNAL_CODINO_ROOT = Path(os.environ.get("CODINO_ROOT", EXTERNAL_ROOT / "codino")).expanduser()
EXTERNAL_CODINO_RUN_DIR = (
    EXTERNAL_CODINO_ROOT
    / "work_dirs/dinov3_codino_inst_0423_lrrestart_from0414ep10_b8_bb1e5_head2e5_cosine_freq4_20260514_112019"
)
EXTERNAL_TWO_STAGE_ROOT = (
    EXTERNAL_ROOT / "inference/dinov3_cascade_unified/two_stage_multiclass_20260426"
)

LOCAL_CODINO_DETECTOR_DIR = BUNDLE_CHECKPOINTS / "codino" / "detector"
LOCAL_CODINO_CLASSIFIER_DIR = BUNDLE_CHECKPOINTS / "codino" / "classifier"
LOCAL_CODINO_TRT_DIR = BUNDLE_CHECKPOINTS / "codino" / "trt"


def _prefer_existing(local: Path, fallback: Path) -> Path:
    return local if local.is_file() else fallback


DEFAULT_CODINO_RUNTIME_SCRIPT = EXTERNAL_CODINO_ROOT / "tools/infer_dinov3_codino_video_fast.py"
DEFAULT_CONFIG = _prefer_existing(
    LOCAL_CODINO_DETECTOR_DIR / "resolved_config.py",
    EXTERNAL_CODINO_RUN_DIR / "resolved_config.py",
)
DEFAULT_CHECKPOINT = _prefer_existing(
    LOCAL_CODINO_DETECTOR_DIR / "epoch_2.pth",
    EXTERNAL_CODINO_RUN_DIR / "epoch_2.pth",
)
DEFAULT_CLASSIFIER_CKPT = _prefer_existing(
    LOCAL_CODINO_CLASSIFIER_DIR / "best.pt",
    EXTERNAL_TWO_STAGE_ROOT
    / "outputs/roi_classifier_codino_maskroi_spatial_gap_full_fp16/run_20260516_041219/checkpoints/best.pt",
)
DEFAULT_TRT_BACKBONE_ENGINE = _prefer_existing(
    LOCAL_CODINO_TRT_DIR / "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",
    EXTERNAL_ROOT / "outputs/trt/codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",
)
DEFAULT_TRT_QUERY_ENCODER_ENGINE = _prefer_existing(
    LOCAL_CODINO_TRT_DIR / "codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",
    EXTERNAL_ROOT / "outputs/trt/codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",
)
DEFAULT_TRT_DECODER_ENGINE = _prefer_existing(
    LOCAL_CODINO_TRT_DIR / "codino_decoder_b2_736x1280_msda_plugin_fp16.engine",
    EXTERNAL_ROOT / "outputs/trt/codino_decoder_b2_736x1280_msda_plugin_fp16.engine",
)
DEFAULT_TRT_MASK_HEAD_ENGINE = _prefer_existing(
    LOCAL_CODINO_TRT_DIR / "codino_mask_head_core_n1_736x1280_fp16.engine",
    EXTERNAL_ROOT / "outputs/trt/codino_mask_head_core_n1_736x1280_fp16.engine",
)
DEFAULT_TRT_EXTRA_SITE_PACKAGES = (
    EXTERNAL_ROOT / "inference/eva02_cascade_experimental/venv/lib/python3.10/site-packages"
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
CLASS_COLORS = [
    (0, 220, 255),
    (80, 180, 255),
    (80, 255, 120),
    (255, 180, 80),
    (220, 120, 255),
]


def _import_codino_runtime(script_path: Path):
    path = script_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_codino_video_fast_runtime", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import Co-DINO runtime script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_target_size(value: str) -> tuple[int, int] | None:
    text = str(value).strip().lower()
    if text in {"auto", "config", "none"}:
        return None
    parts = text.replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"target size must be WxH or 'auto', got: {value}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"target size must be positive, got: {value}")
    return height, width


def _format_target_size(target_size: tuple[int, int]) -> str:
    height, width = target_size
    return f"{width}x{height}"


def _collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTS:
            return [input_path]
        raise RuntimeError(f"input file is not a supported video: {input_path}")
    if not input_path.is_dir():
        raise RuntimeError(f"input path not found: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def _read_video_meta(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


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


def _class_from_detection(
    det: np.ndarray,
    class_names: list[str],
    class_ids: list[int],
) -> tuple[int, str, int, float | None, list[float] | None]:
    if det.shape[0] >= 7 and class_names:
        class_index = int(round(float(det[5])))
        class_name = class_names[class_index] if 0 <= class_index < len(class_names) else str(class_index)
        category_id = int(class_ids[class_index]) if 0 <= class_index < len(class_ids) else class_index
        class_score = float(det[6])
        prob_count = min(len(class_names), max(0, int(det.shape[0]) - 7))
        class_probs = [float(v) for v in det[7 : 7 + prob_count]] if prob_count > 0 else None
        return class_index, class_name, category_id, class_score, class_probs
    class_name = class_names[0] if class_names else "foreground"
    category_id = int(class_ids[0]) if class_ids else 0
    return 0, class_name, category_id, None, None


def _detections_to_json(
    bbox_result: list[np.ndarray],
    segm_result: list[list[np.ndarray]] | None,
    *,
    class_names: list[str],
    class_ids: list[int],
    score_thresh: float,
    mask_approx: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detector_cls, bboxes in enumerate(bbox_result):
        if bboxes is None or len(bboxes) == 0:
            continue
        masks = segm_result[detector_cls] if segm_result is not None and detector_cls < len(segm_result) else []
        for det_idx, det in enumerate(bboxes):
            detector_score = float(det[4])
            if detector_score < score_thresh:
                continue
            class_index, class_name, category_id, class_score, class_probs = _class_from_detection(
                det,
                class_names,
                class_ids,
            )
            x1, y1, x2, y2 = [float(v) for v in det[:4]]
            item: dict[str, object] = {
                "label": class_name,
                "class_name": class_name,
                "category_id": int(category_id),
                "category_index": int(class_index),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": detector_score,
                "detector_score": detector_score,
            }
            if class_score is not None:
                item["class_score"] = class_score
            if class_probs is not None:
                item["class_probs"] = class_probs
            if det_idx < len(masks):
                mask = masks[det_idx]
                if mask is not None:
                    mask_bin = np.asarray(mask).astype(np.uint8, copy=False)
                    if mask_bin.ndim == 3:
                        mask_bin = mask_bin[:, :, 0]
                    if mask_bin.max(initial=0) > 1:
                        mask_bin = (mask_bin > 0.5).astype(np.uint8)
                    polygons = _mask_to_polygons(mask_bin, mask_approx)
                    item["segmentation"] = polygons
                    item["polygons"] = polygons
            rows.append(item)
    return rows


def _make_detector_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        video=Path("__unused__.mp4"),
        config=args.config,
        checkpoint=args.checkpoint,
        out=None,
        json_out=None,
        device=args.device,
        batch_size=args.batch_size,
        score_thr=float(args.score_thresh),
        model_score_thr=float(args.model_score_thr),
        amp=args.amp,
        tf32=bool(args.tf32),
        onnx_backbone=None,
        onnx_max_batch=4,
        trt_backbone_engine=args.trt_backbone_engine,
        trt_feature_engine=args.trt_feature_engine,
        trt_feature_names=args.trt_feature_names,
        trt_query_encoder_engine=args.trt_query_encoder_engine,
        trt_query_encoder_shapes=args.trt_query_encoder_shapes,
        trt_decoder_engine=args.trt_decoder_engine,
        trt_mask_head_engine=args.trt_mask_head_engine,
        trt_extra_site_packages=args.trt_extra_site_packages,
        preprocess=args.preprocess,
        draw_boxes=False,
        draw_masks=False,
        disable_mask_head=False,
        disable_mask_iou_head=bool(args.disable_mask_iou_head),
        eval_module=None,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        mask_alpha=0.45,
        mask_color="0,255,0",
        line_width=2,
        frame_stride=1,
        start_frame=0,
        max_frames=None,
        codec="mp4v",
        no_video=True,
        async_writer=True,
        writer_queue_size=16,
        skip_backbone_init=True,
        disable_act_checkpoint=True,
        warmup_batches=0,
        log_interval=60,
        profile_modules=False,
        compile_modules=args.compile_modules,
        compile_mode=args.compile_mode,
    )


def _configure_model(codino_video, model, args: argparse.Namespace) -> None:
    selected_engines = [
        args.trt_backbone_engine is not None,
        args.trt_feature_engine is not None,
    ]
    if sum(selected_engines) > 1:
        raise ValueError("--trt-backbone-engine and --trt-feature-engine are mutually exclusive")

    if args.trt_backbone_engine is not None:
        codino_video.install_trt_backbone(
            model,
            args.trt_backbone_engine.expanduser().resolve(),
            args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None,
        )
    if args.trt_feature_engine is not None:
        feature_names = tuple(part.strip() for part in str(args.trt_feature_names).split(",") if part.strip())
        codino_video.install_trt_feature_extractor(
            model,
            args.trt_feature_engine.expanduser().resolve(),
            feature_names,
            args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None,
        )

    codino_video.apply_runtime_model_options(model, _make_detector_args(args))

    if args.trt_query_encoder_engine is not None:
        codino_video.install_trt_query_encoder(
            model,
            args.trt_query_encoder_engine.expanduser().resolve(),
            codino_video.parse_hw_shapes(args.trt_query_encoder_shapes),
            args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None,
        )
    if args.trt_decoder_engine is not None:
        codino_video.install_trt_decoder(
            model,
            args.trt_decoder_engine.expanduser().resolve(),
            args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None,
        )
    if args.trt_mask_head_engine is not None:
        codino_video.install_trt_mask_head(
            model,
            args.trt_mask_head_engine.expanduser().resolve(),
            args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None,
        )
    codino_video.apply_compile_options(model, _make_detector_args(args))


def _infer_one_video(
    *,
    codino_video,
    model,
    pipeline,
    classifier: torch.nn.Module | None,
    video_path: Path,
    jsonl_path: Path,
    overlay_path: Path | None,
    class_names: list[str],
    class_ids: list[int],
    target_size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    video_meta = _read_video_meta(video_path)
    fps = float(video_meta["fps"]) if float(video_meta["fps"]) > 0 else 30.0
    writer = make_jsonl_writer(
        jsonl_path,
        str(args.json_backend),
        async_writer=bool(args.async_writer),
        queue_size=max(1, int(args.writer_queue_size)),
    )
    overlay_writer = None
    if overlay_path is not None:
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*str(args.overlay_fourcc)),
            fps / max(1, int(args.frame_stride)),
            (int(video_meta["width"]), int(video_meta["height"])),
            True,
        )
        if not overlay_writer.isOpened():
            raise RuntimeError(f"failed to open overlay writer: {overlay_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    if int(args.start_frame) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_frame))

    processed = 0
    detections = 0
    measured_frames = 0
    measured_time = 0.0
    batch_frames: list[np.ndarray] = []
    batch_ids: list[int] = []
    wall_start = time.perf_counter()
    frame_idx = int(args.start_frame) - 1
    use_cuda = torch.cuda.is_available() and next(model.parameters()).is_cuda

    def flush() -> None:
        nonlocal processed, detections, measured_frames, measured_time
        if not batch_frames:
            return
        valid_count = len(batch_frames)
        frames_for_infer = batch_frames
        if (
            args.trt_backbone_engine is not None
            or args.trt_feature_engine is not None
            or args.trt_query_encoder_engine is not None
            or args.trt_decoder_engine is not None
            or args.trt_mask_head_engine is not None
        ) and valid_count < int(args.batch_size):
            frames_for_infer = batch_frames + [batch_frames[-1]] * (int(args.batch_size) - valid_count)

        if use_cuda:
            torch.cuda.synchronize()
        batch_start = time.perf_counter()
        if classifier is None:
            results = codino_video.infer_batch(
                model,
                pipeline,
                frames_for_infer,
                args.amp,
                preprocess=args.preprocess,
                target_size=target_size,
            )
        else:
            if not hasattr(codino_video, "infer_batch_with_roi_classifier"):
                raise RuntimeError(
                    "Co-DINO runtime script does not expose infer_batch_with_roi_classifier; "
                    "use the updated infer_dinov3_codino_video_fast.py."
                )
            results = codino_video.infer_batch_with_roi_classifier(
                model,
                pipeline,
                frames_for_infer,
                args.amp,
                preprocess=args.preprocess,
                target_size=target_size,
                classifier=classifier,
                num_classifier_classes=len(class_names),
            )
        if use_cuda:
            torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - batch_start

        batch_measured = sum(1 for src_frame_idx in batch_ids if src_frame_idx >= int(args.warmup_frames))
        if batch_measured > 0:
            measured_time += batch_elapsed * (batch_measured / max(1, valid_count))
            measured_frames += batch_measured

        for frame, src_frame_idx, result in zip(batch_frames, batch_ids, results[:valid_count]):
            bbox_result, segm_result = codino_video.normalize_result(result)
            orig_shape = frame.shape[:2]
            bbox_result = codino_video.unletterbox_bboxes(bbox_result, orig_shape, target_size)
            segm_result = codino_video.unletterbox_segms(segm_result, orig_shape, target_size)
            det_rows = _detections_to_json(
                bbox_result,
                segm_result,
                class_names=class_names,
                class_ids=class_ids,
                score_thresh=float(args.score_thresh),
                mask_approx=str(args.mask_approx),
            )
            detections += len(det_rows)
            writer.write(
                {
                    "frame_index": int(src_frame_idx),
                    "frame_idx": int(src_frame_idx),
                    "time_sec": float(src_frame_idx / fps),
                    "width": int(video_meta["width"]),
                    "height": int(video_meta["height"]),
                    "detections": det_rows,
                    "instances": det_rows,
                }
            )
            if overlay_writer is not None:
                vis = codino_video.render_overlay(
                    frame,
                    bbox_result,
                    segm_result,
                    score_thr=float(args.score_thresh),
                    draw_boxes=True,
                    draw_masks=True,
                    mask_alpha=0.45,
                    mask_color=(0, 255, 0),
                    line_width=2,
                    classifier_class_names=class_names if classifier is not None else None,
                    draw_classifier_labels=True,
                )
                overlay_writer.write(vis)

        processed += valid_count
        batch_frames.clear()
        batch_ids.clear()

    try:
        selected = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx < int(args.start_frame):
                continue
            if (frame_idx - int(args.start_frame)) % max(1, int(args.frame_stride)) != 0:
                continue
            batch_frames.append(frame)
            batch_ids.append(frame_idx)
            selected += 1
            if args.max_frames is not None and selected >= int(args.max_frames):
                flush()
                break
            if len(batch_frames) >= int(args.batch_size):
                flush()
        flush()
    finally:
        cap.release()
        writer.close()
        if overlay_writer is not None:
            overlay_writer.release()

    if use_cuda:
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
        "warmup_frames": int(args.warmup_frames),
        "measured_frames": int(measured_frames),
        "measured_time_sec": float(measured_time),
        "compute_fps": float(measured_fps),
        "compute_ms_per_frame": float(1000.0 / measured_fps) if measured_fps > 0 else 0.0,
        "measured_fps": float(measured_fps),
        "measured_ms_per_frame": float(1000.0 / measured_fps) if measured_fps > 0 else 0.0,
        "jsonl_size_bytes": int(jsonl_path.stat().st_size if jsonl_path.is_file() else 0),
        "overlay_size_bytes": int(
            overlay_path.stat().st_size if overlay_path is not None and overlay_path.is_file() else 0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 + Co-DINO JSONL inference")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--codino-runtime-script", type=Path, default=DEFAULT_CODINO_RUNTIME_SCRIPT)
    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CKPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--target-size", type=_parse_target_size, default=(720, 1280))
    parser.add_argument("--score-thresh", type=float, default=0.30)
    parser.add_argument("--model-score-thr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preprocess", choices=("direct", "pipeline"), default="direct")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json-backend", choices=("json", "orjson"), default="orjson")
    parser.add_argument("--async-writer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--writer-queue-size", type=int, default=512)
    parser.add_argument("--mask-approx", choices=("none", "simple"), default="none")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--overlay-ext", default=".mp4")
    parser.add_argument("--overlay-fourcc", default="mp4v")
    parser.add_argument("--disable-mask-iou-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--compile-modules", default="")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--trt-backbone-engine", type=Path, default=None)
    parser.add_argument("--trt-feature-engine", type=Path, default=None)
    parser.add_argument("--trt-feature-names", default="feat0,feat1,feat2,feat3,feat4")
    parser.add_argument("--trt-query-encoder-engine", type=Path, default=None)
    parser.add_argument("--trt-query-encoder-shapes", default="184x320,92x160,46x80,23x40,12x20")
    parser.add_argument("--trt-decoder-engine", type=Path, default=None)
    parser.add_argument("--trt-mask-head-engine", type=Path, default=None)
    parser.add_argument("--trt-extra-site-packages", type=Path, default=DEFAULT_TRT_EXTRA_SITE_PACKAGES)
    return parser


def _resolve_optional_existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return resolved if resolved.is_file() else None


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    jsonl_dir = output_dir / "jsonl"
    overlay_dir = args.overlay_dir.expanduser().resolve() if args.overlay_dir else output_dir / "overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    if args.write_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    args.config = args.config.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.classifier_checkpoint = args.classifier_checkpoint.expanduser().resolve()
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.classifier and not args.classifier_checkpoint.is_file():
        raise FileNotFoundError(args.classifier_checkpoint)

    # Treat absent optional TensorRT engines as disabled, so a portable artifact
    # bundle can run in PyTorch mode before engines are rebuilt locally.
    args.trt_backbone_engine = _resolve_optional_existing(args.trt_backbone_engine)
    args.trt_feature_engine = _resolve_optional_existing(args.trt_feature_engine)
    args.trt_query_encoder_engine = _resolve_optional_existing(args.trt_query_encoder_engine)
    args.trt_decoder_engine = _resolve_optional_existing(args.trt_decoder_engine)
    args.trt_mask_head_engine = _resolve_optional_existing(args.trt_mask_head_engine)
    args.trt_extra_site_packages = (
        args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None
    )

    codino_video = _import_codino_runtime(args.codino_runtime_script)
    codino_video._prepare_imports()

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    print(f"[INFO] Co-DINO runtime script: {args.codino_runtime_script}")
    print(f"[INFO] checkpoint: {args.checkpoint}")
    print(f"[INFO] config: {args.config}")

    detector_args = _make_detector_args(args)
    model = codino_video.load_detector(detector_args)
    _configure_model(codino_video, model, args)
    pipeline = codino_video.build_test_pipeline(model)
    target_size = args.target_size or codino_video.find_letterbox_size(model)

    classifier = None
    classifier_raw: dict[str, Any] = {}
    if args.classifier:
        classifier, classifier_raw, class_names, class_ids = codino_video.load_roi_classifier(
            args.classifier_checkpoint,
            args.device,
        )
    else:
        class_names = ["foreground"]
        class_ids = [0]

    videos = _collect_videos(input_path, bool(args.recursive))
    if not videos:
        raise RuntimeError(f"no videos found under: {input_path}")

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
            codino_video=codino_video,
            model=model,
            pipeline=pipeline,
            classifier=classifier,
            video_path=video_path,
            jsonl_path=jsonl_path,
            overlay_path=overlay_path,
            class_names=class_names,
            class_ids=class_ids,
            target_size=target_size,
            args=args,
        )
        runs.append(result)
        print(
            f"[DONE] {video_path.name}: e2e_fps={result['e2e_fps']:.2f} "
            f"compute_fps={result['compute_fps']:.2f} det/frame={result['detections_per_frame']:.3f}"
        )

    summary = {
        "detector": "codino",
        "runtime": str(BASE_DIR),
        "codino_runtime_script": str(args.codino_runtime_script),
        "input": str(input_path),
        "output_dir": str(output_dir),
        "jsonl_dir": str(jsonl_dir),
        "overlay_dir": str(overlay_dir) if args.write_overlay else None,
        "write_overlay": bool(args.write_overlay),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "classifier_enabled": bool(args.classifier),
        "classifier_checkpoint": str(args.classifier_checkpoint) if args.classifier else None,
        "classifier_model_type": str((classifier_raw.get("model_cfg") or {}).get("model_type", ""))
        if args.classifier
        else None,
        "classifier_val_macro_f1": (classifier_raw.get("val_metrics") or {}).get("macro_f1")
        if args.classifier
        else None,
        "class_names": class_names,
        "class_ids": class_ids,
        "target_size": _format_target_size(target_size),
        "score_thresh": float(args.score_thresh),
        "model_score_thr": float(args.model_score_thr),
        "amp": str(args.amp),
        "tf32": bool(args.tf32),
        "batch_size": int(args.batch_size),
        "max_frames": args.max_frames,
        "warmup_frames": int(args.warmup_frames),
        "json_backend": str(args.json_backend),
        "async_writer": bool(args.async_writer),
        "mask_approx": str(args.mask_approx),
        "disable_mask_iou_head": bool(args.disable_mask_iou_head),
        "trt": {
            "backbone_engine": None if args.trt_backbone_engine is None else str(args.trt_backbone_engine),
            "feature_engine": None if args.trt_feature_engine is None else str(args.trt_feature_engine),
            "query_encoder_engine": None
            if args.trt_query_encoder_engine is None
            else str(args.trt_query_encoder_engine),
            "decoder_engine": None if args.trt_decoder_engine is None else str(args.trt_decoder_engine),
            "mask_head_engine": None if args.trt_mask_head_engine is None else str(args.trt_mask_head_engine),
        },
        "runs": runs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
