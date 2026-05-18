"""Command builder for optional RT-DETR Head/Face SQLite inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.detectors.common import maybe_add, require_script


def build_command(args: argparse.Namespace, video: Path, output_sqlite: Path) -> list[str]:
    repo = Path(args.rtdetr_repo).expanduser().resolve()
    script = require_script(repo / "tools" / "inference" / "video_sqlite_inf.py")

    command = [
        str(args.python),
        str(script),
        "--input",
        str(video),
        "--output",
        str(output_sqlite),
        "--classes",
        "Head",
        "Face",
        "--output-mode",
        "tracks",
        "--tracker",
        "bytetrack",
        "--conf-thr",
        str(args.head_face_conf_thr),
        "--bytetrack-high-thr",
        str(args.head_face_conf_thr),
        "--bytetrack-low-thr",
        str(args.head_face_low_thr),
        "--bytetrack-new-thr",
        str(args.head_face_new_track_thr),
        "--track-min-hits",
        str(args.head_face_track_min_hits),
        "--nms-iou-thr",
        str(args.head_face_nms_iou_thr),
        "--batch-size",
        str(args.head_face_batch_size),
        "--device",
        str(args.head_face_device),
        "--progress-interval",
        str(args.head_face_progress_interval),
    ]
    maybe_add(command, "--config", args.rtdetr_config)
    maybe_add(command, "--resume", args.rtdetr_checkpoint)
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.head_face_compile:
        command.append("--compile")
    return command
