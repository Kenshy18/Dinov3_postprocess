#!/usr/bin/env python3
"""Command builders for the integrated detector/postprocess pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.detectors.codino.commands import build_command as build_codino_detector_command
from backend.detectors.dinov3.commands import build_command as build_dinov3_detector_command
from backend.detectors.eva02.commands import build_command as build_eva02_detector_command
from backend.detectors.rtdetr.commands import build_command as build_rtdetr_head_face_command
from backend.postprocess.commands import build_env as build_atosyori_env
from backend.postprocess.commands import build_run_command as build_atosyori_run_command


def build_dinov3_command(args: argparse.Namespace, video: Path, dinov3_out: Path) -> list[str]:
    return build_dinov3_detector_command(args, video, dinov3_out)


def build_eva02_command(args: argparse.Namespace, video: Path, eva02_out: Path) -> list[str]:
    return build_eva02_detector_command(args, video, eva02_out)


def build_codino_command(args: argparse.Namespace, video: Path, codino_out: Path) -> list[str]:
    return build_codino_detector_command(args, video, codino_out)


def build_detector_command(args: argparse.Namespace, video: Path, detector_out: Path) -> list[str]:
    if args.detector == "dinov3":
        return build_dinov3_command(args, video, detector_out)
    if args.detector == "eva02":
        return build_eva02_command(args, video, detector_out)
    if args.detector == "codino":
        return build_codino_command(args, video, detector_out)
    raise RuntimeError(f"unsupported detector: {args.detector}")


def build_head_face_command(args: argparse.Namespace, video: Path, output_sqlite: Path) -> list[str]:
    return build_rtdetr_head_face_command(args, video, output_sqlite)


def atosyori_env(args: argparse.Namespace) -> dict[str, str]:
    return build_atosyori_env(args.atosyori_repo)


def build_postprocess_command(
    args: argparse.Namespace,
    video: Path,
    jsonl_path: Path,
    postprocess_out: Path,
) -> list[str]:
    engine_args = [
        "--overlay-encoder",
        str(args.overlay_encoder),
        "--raw-remove-short-tracks-max-frames",
        str(args.raw_remove_short_tracks_max_frames),
        "--raw-cut-detect" if args.raw_cut_detect else "--no-raw-cut-detect",
        "--progress-interval-sec",
        str(getattr(args, "progress_interval_sec", 5.0)),
    ]
    if args.postprocess_extra_args:
        engine_args.extend(args.postprocess_extra_args)
    return build_atosyori_run_command(
        python=args.python,
        atosyori_repo=args.atosyori_repo,
        input_jsonl=jsonl_path,
        input_video=video,
        output_dir=postprocess_out,
        model_root=args.postprocess_model_root,
        intervals=args.intervals,
        default_shape_mode=args.default_shape_mode,
        class_policy_json=args.class_policy_json,
        k2_device=args.k2_device,
        polygon_predictor_device=args.polygon_predictor_device,
        render_overlays=args.render_overlays,
        force=args.force,
        engine_args=engine_args,
    )
