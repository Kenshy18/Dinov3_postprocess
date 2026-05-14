"""Command builder for the EVA02 video detector runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.detectors.common import maybe_add, require_script


def build_command(args: argparse.Namespace, video: Path, output_dir: Path) -> list[str]:
    script = require_script(args.eva02_runtime / "infer_video_eva02_jsonl.py")

    command = [
        str(args.python),
        str(script),
        "--input",
        str(video),
        "--output",
        str(output_dir),
        "--classifier" if args.classifier else "--no-classifier",
        "--target-size",
        str(args.eva02_target_size),
        "--score-thresh",
        str(args.eva02_score_thresh),
        "--nms-thresh",
        str(args.eva02_nms_thresh),
        "--topk",
        str(args.eva02_topk),
        "--batch-size",
        str(args.eva02_batch_size),
        "--warmup-frames",
        str(args.eva02_warmup_frames),
        "--classifier-batch-size",
        str(args.eva02_classifier_batch_size),
        "--json-backend",
        str(args.eva02_json_backend),
        "--mask-approx",
        str(args.eva02_mask_approx),
        "--async-writer" if args.eva02_async_writer else "--no-async-writer",
        "--checkpoint",
        str(args.eva02_detector_checkpoint),
        "--classifier-checkpoint",
        str(args.eva02_classifier_checkpoint),
        "--overwrite",
    ]
    maybe_add(command, "--max-frames", args.max_frames)
    if args.recursive:
        command.append("--recursive")
    if args.write_detector_overlay:
        command.append("--write-overlay")
    return command
