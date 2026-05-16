#!/usr/bin/env python3
"""Fast EVA02 Cascade video inference with ROI classification and JSONL output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import infer_video_folder_backbone_half_amp_compile_spatialcls_sota_rich as rich


BUNDLE_CHECKPOINTS = REPO_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = BUNDLE_CHECKPOINTS / "eva02" / "detector" / "model_final.pth"
DEFAULT_CLASSIFIER_CKPT = BUNDLE_CHECKPOINTS / "eva02" / "classifier" / "best.pt"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "eva02"
    / "eva02_det"
    / "projects"
    / "ViTDet"
    / "configs"
    / "eva2_o365_to_coco"
    / "eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1280_lrd0p8.py"
)
DEFAULT_RUNTIME_PROFILE = REPO_ROOT / "configs" / "runtime_profile.json"
DEFAULT_LOCAL_RUNTIME_PROFILE = REPO_ROOT / ".runtime" / "runtime_profile.json"


def _profile_default(section: str, key: str, default: int) -> int:
    try:
        profile_path = DEFAULT_LOCAL_RUNTIME_PROFILE if DEFAULT_LOCAL_RUNTIME_PROFILE.is_file() else DEFAULT_RUNTIME_PROFILE
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            value = profile.get("recommendations", {}).get(section, {}).get(key)
            if value is not None:
                return int(value)
    except Exception:
        pass
    return default


def _parse_target_size(value: str) -> int:
    text = str(value).strip().lower()
    if "x" in text:
        width_s, height_s = text.split("x", 1)
        width = int(width_s)
        height = int(height_s)
        if width != height:
            raise argparse.ArgumentTypeError("EVA02 runtime expects a square target size, e.g. 1280")
        return width
    return int(text)


def _collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    return rich.base._collect_videos(input_path, recursive)


def _link_or_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel_src = os.path.relpath(src, start=dst.parent)
    try:
        dst.symlink_to(rel_src)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _write_summary(args: argparse.Namespace, output_dir: Path) -> None:
    internal_jsonl_dir = output_dir / "jsonl" / "infer"
    jsonl_dir = output_dir / "jsonl"
    videos = _collect_videos(Path(args.input).expanduser().resolve(), bool(args.recursive))
    runs = []
    for video in videos:
        internal_jsonl_path = internal_jsonl_dir / f"{video.stem}.jsonl"
        jsonl_path = jsonl_dir / f"{video.stem}.jsonl"
        if internal_jsonl_path.is_file():
            jsonl_path = _link_or_copy(internal_jsonl_path, jsonl_path)
        runs.append(
            {
                "video": str(video),
                "output_jsonl": str(jsonl_path),
                "jsonl_size_bytes": int(jsonl_path.stat().st_size if jsonl_path.is_file() else 0),
            }
        )

    summary = {
        "detector": "eva02",
        "runtime": str(BASE_DIR),
        "jsonl_dir": str(jsonl_dir),
        "classifier_enabled": bool(args.classifier),
        "classifier_checkpoint": str(Path(args.classifier_checkpoint).expanduser().resolve())
        if args.classifier
        else None,
        "class_names": ["女性器", "男性器", "結合部分"] if args.classifier else ["foreground"],
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "target_size": str(args.target_size),
        "score_thresh": float(args.score_thresh),
        "nms_thresh": float(args.nms_thresh),
        "topk": int(args.topk),
        "batch_size": int(args.batch_size),
        "runs": runs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast EVA02 Cascade + ROI classifier JSONL inference")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-checkpoint", default=str(DEFAULT_CLASSIFIER_CKPT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target-size", type=_parse_target_size, default=1280)
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=_profile_default("eva02", "batch_size", 1))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json-backend", choices=["json", "orjson"], default="json")
    parser.add_argument("--async-writer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-approx", choices=["none", "simple"], default="simple")
    parser.add_argument("--write-overlay", action="store_true")
    parser.add_argument("--compile-backbone", default="max-autotune")
    parser.add_argument("--drop-block-indices", default="19,21,22")
    parser.add_argument("--pack-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-detector-postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-to-orig-mask-postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-batch-size", type=int, default=_profile_default("eva02", "classifier_batch_size", 1024))
    parser.add_argument("--measure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval-sec", type=float, default=float(os.environ.get("PIPELINE_PROGRESS_INTERVAL_SEC", "5")))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.classifier:
        raise RuntimeError("EVA02 integrated runtime currently requires the ROI classifier")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rich.INPUT_DIR = str(Path(args.input).expanduser().resolve())
    rich.OUTPUT_DIR = str(output_dir / "infer")
    rich.RECURSIVE = bool(args.recursive)
    rich.OVERWRITE = bool(args.overwrite)
    rich.CHECKPOINT = Path(args.checkpoint).expanduser().resolve()
    rich.CONFIG = Path(args.config).expanduser().resolve()
    rich.CLASSIFIER_CHECKPOINT = Path(args.classifier_checkpoint).expanduser().resolve()
    rich.TARGET_SIZE = int(args.target_size)
    rich.SCORE_THRESH = float(args.score_thresh)
    rich.NMS_THRESH = float(args.nms_thresh)
    rich.TOPK = int(args.topk)
    rich.DEVICE = str(args.device)
    rich.CLASSIFIER_DEVICE = str(args.device)
    rich.BATCH_SIZE = int(args.batch_size)
    rich.MAX_FRAMES = args.max_frames
    rich.WARMUP_FRAMES = int(args.warmup_frames)
    rich.JSON_BACKEND = str(args.json_backend)
    rich.ASYNC_WRITER = bool(args.async_writer)
    rich.MASK_APPROX = str(args.mask_approx)
    rich.SAVE_OVERLAY_VIDEO = bool(args.write_overlay)
    rich.COMPILE_BACKBONE = str(args.compile_backbone)
    rich.DROP_BLOCK_INDICES = str(args.drop_block_indices)
    rich.PACK_INPUTS = bool(args.pack_inputs)
    rich.RAW_DETECTOR_POSTPROCESS = bool(args.raw_detector_postprocess)
    rich.RAW_TO_ORIG_MASK_POSTPROCESS = bool(args.raw_to_orig_mask_postprocess)
    rich.CLASSIFIER_BATCH_SIZE = int(args.classifier_batch_size)
    rich.MEASURE = bool(args.measure)
    rich.PROGRESS_INTERVAL_SEC = float(args.progress_interval_sec)

    exit_code = rich.main()
    if exit_code == 0:
        _write_summary(args, output_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
