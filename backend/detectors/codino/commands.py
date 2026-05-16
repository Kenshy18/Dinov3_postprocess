"""Command builder for the DINOv3 + Co-DINO video detector runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.detectors.common import maybe_add, require_script


def _pick(args: argparse.Namespace, codino_name: str, shared_name: str):
    value = getattr(args, codino_name, None)
    return value if value is not None else getattr(args, shared_name)


def build_command(args: argparse.Namespace, video: Path, output_dir: Path) -> list[str]:
    script = require_script(args.codino_runtime / "infer_video_codino_jsonl.py")

    command = [
        str(args.python),
        str(script),
        "--input",
        str(video),
        "--output",
        str(output_dir),
        "--classifier" if args.classifier else "--no-classifier",
        "--target-size",
        str(_pick(args, "codino_target_size", "target_size")),
        "--score-thresh",
        str(_pick(args, "codino_score_thresh", "score_thresh")),
        "--model-score-thr",
        str(args.codino_model_score_thr),
        "--batch-size",
        str(_pick(args, "codino_batch_size", "batch_size")),
        "--warmup-frames",
        str(_pick(args, "codino_warmup_frames", "warmup_frames")),
        "--json-backend",
        str(args.codino_json_backend),
        "--mask-approx",
        str(args.codino_mask_approx),
        "--amp",
        str(args.codino_amp),
        "--async-writer" if args.codino_async_writer else "--no-async-writer",
        "--tf32" if args.codino_tf32 else "--no-tf32",
        "--overwrite",
    ]
    maybe_add(command, "--max-frames", args.max_frames)
    maybe_add(command, "--codino-runtime-script", args.codino_runtime_script)
    maybe_add(command, "--config", args.codino_config)
    maybe_add(command, "--checkpoint", args.codino_checkpoint)
    maybe_add(command, "--classifier-checkpoint", args.codino_classifier_checkpoint)
    maybe_add(command, "--trt-backbone-engine", args.codino_trt_backbone_engine)
    maybe_add(command, "--trt-feature-engine", args.codino_trt_feature_engine)
    maybe_add(command, "--trt-query-encoder-engine", args.codino_trt_query_encoder_engine)
    maybe_add(command, "--trt-decoder-engine", args.codino_trt_decoder_engine)
    maybe_add(command, "--trt-mask-head-engine", args.codino_trt_mask_head_engine)
    maybe_add(command, "--trt-extra-site-packages", args.codino_trt_extra_site_packages)
    if args.recursive:
        command.append("--recursive")
    if args.write_detector_overlay:
        command.append("--write-overlay")
    if args.codino_disable_mask_iou_head:
        command.append("--disable-mask-iou-head")
    return command

