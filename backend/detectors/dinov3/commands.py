"""Command builder for the DINOv3 video detector runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.detectors.common import maybe_add, require_script


def build_command(args: argparse.Namespace, video: Path, output_dir: Path) -> list[str]:
    script = require_script(args.dinov3_runtime / "infer_video_dinov3_jsonl.py")
    python = getattr(args, "dinov3_python", None) or args.python

    command = [
        str(python),
        str(script),
        "--input",
        str(video),
        "--output",
        str(output_dir),
        "--classifier" if args.classifier else "--no-classifier",
        "--target-size",
        str(args.target_size),
        "--score-thresh",
        str(args.score_thresh),
        "--nms-thresh",
        str(args.nms_thresh),
        "--topk",
        str(args.topk),
        "--rpn-pre-nms-topk-test",
        str(args.rpn_pre_nms_topk_test),
        "--rpn-post-nms-topk-test",
        str(args.rpn_post_nms_topk_test),
        "--rpn-nms-thresh",
        str(args.rpn_nms_thresh),
        "--batch-size",
        str(args.batch_size),
        "--warmup-frames",
        str(args.warmup_frames),
        "--json-backend",
        str(args.json_backend),
        "--mask-approx",
        str(args.mask_approx),
        "--async-writer" if args.async_writer else "--no-async-writer",
        "--overwrite",
    ]
    maybe_add(command, "--max-frames", args.max_frames)
    maybe_add(command, "--checkpoint", args.detector_checkpoint)
    maybe_add(command, "--classifier-checkpoint", args.classifier_checkpoint)
    maybe_add(command, "--trt-backbone-engine", args.trt_backbone_engine)
    maybe_add(command, "--backbone-weights", args.backbone_weights)
    maybe_add(command, "--progress-interval-sec", getattr(args, "progress_interval_sec", None))
    if args.write_detector_overlay:
        command.append("--write-overlay")
    if args.gpu_prefetch is not None:
        command.append("--gpu-prefetch" if args.gpu_prefetch else "--no-gpu-prefetch")
    return command
