#!/usr/bin/env python3
"""Run DINOv3 video inference, optional ROI classification, and Atosyori postprocess.

The integration intentionally keeps the DINOv3 detector runtime and the
Atosyori postprocess repository as separate components. This script wires them
together and writes a compact per-video summary with easy-to-find SQLite and
overlay links.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = SCRIPT_DIR.parent
DEFAULT_DINOV3_RUNTIME = INTEGRATION_ROOT / "inference" / "dinov3_video_jsonl_runtime"
LOCAL_ATOSYORI_REPO = INTEGRATION_ROOT / "external" / "atosyori-pipeline-dev"
DEFAULT_ATOSYORI_REPO = Path(
    os.environ.get(
        "ATOSYORI_REPO",
        str(
            LOCAL_ATOSYORI_REPO
            if (LOCAL_ATOSYORI_REPO / "src" / "atosyori_postprocess").is_dir()
            else Path("/home/kenke/workspace/CV/atosyori-pipeline-dev")
        ),
    )
)
DEFAULT_MODEL_ROOT = INTEGRATION_ROOT / "checkpoints" / "postprocess"
DEFAULT_DETECTOR_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "detector" / "model_final.pth"
DEFAULT_DINOV3_WEIGHTS = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "dinov3"
    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_CLASSIFIER_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "classifier" / "best.pt"
DEFAULT_TRT_BACKBONE_ENGINE = (
    INTEGRATION_ROOT
    / "checkpoints"
    / "trt"
    / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"
)
DEFAULT_POLICY = INTEGRATION_ROOT / "configs" / "class_policy_default.json"


def abs_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def abs_path_preserve_symlink(path: str | Path) -> Path:
    expanded = os.path.expanduser(str(path))
    return Path(os.path.abspath(expanded))


def collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise RuntimeError(f"input file is not a supported video: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise RuntimeError(f"input path not found: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    label: str,
) -> dict[str, Any]:
    print(f"[run] {label}", flush=True)
    print("[cmd] " + " ".join(command), flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd) if cwd is not None else None, env=env, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    print(f"[done] {label}: {elapsed:.2f}s", flush=True)
    return {
        "label": label,
        "cmd": command,
        "cwd": None if cwd is None else str(cwd),
        "returncode": int(completed.returncode),
        "wall_seconds": float(elapsed),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel_src = os.path.relpath(src, start=dst.parent)
    try:
        dst.symlink_to(rel_src)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def maybe_add(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_dinov3_command(args: argparse.Namespace, video: Path, dinov3_out: Path) -> list[str]:
    script = args.dinov3_runtime / "infer_video_dinov3_jsonl.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    command = [
        str(args.python),
        str(script),
        "--input",
        str(video),
        "--output",
        str(dinov3_out),
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
    if args.write_detector_overlay:
        command.append("--write-overlay")
    if args.gpu_prefetch is not None:
        command.append("--gpu-prefetch" if args.gpu_prefetch else "--no-gpu-prefetch")
    return command


def atosyori_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    src = args.atosyori_repo / "src"
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) if not current else str(src) + os.pathsep + current
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_postprocess_command(
    args: argparse.Namespace,
    video: Path,
    jsonl_path: Path,
    postprocess_out: Path,
) -> list[str]:
    if not (args.atosyori_repo / "src" / "atosyori_postprocess").is_dir():
        raise FileNotFoundError(args.atosyori_repo / "src" / "atosyori_postprocess")

    command = [
        str(args.python),
        "-m",
        "atosyori_postprocess",
        "run",
        "--input-jsonl",
        str(jsonl_path),
        "--input-video",
        str(video),
        "--output-dir",
        str(postprocess_out),
        "--model-root",
        str(args.postprocess_model_root),
        "--intervals",
        str(args.intervals),
        "--default-shape-mode",
        str(args.default_shape_mode),
        "--k2-device",
        str(args.k2_device),
        "--polygon-predictor-device",
        str(args.polygon_predictor_device),
        "--render-overlays" if args.render_overlays else "--no-render-overlays",
    ]
    if args.class_policy_json is not None:
        command.extend(["--class-policy-json", str(args.class_policy_json)])
    if args.force:
        command.append("--force")

    engine_args = [
        "--overlay-encoder",
        str(args.overlay_encoder),
        "--raw-remove-short-tracks-max-frames",
        str(args.raw_remove_short_tracks_max_frames),
        "--raw-cut-detect" if args.raw_cut_detect else "--no-raw-cut-detect",
    ]
    if args.postprocess_extra_args:
        engine_args.extend(args.postprocess_extra_args)
    if engine_args:
        command.append("--")
        command.extend(engine_args)
    return command


def summarize_dinov3(dinov3_out: Path, video: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = dinov3_out / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    jsonl_path = dinov3_out / "jsonl" / f"{video.stem}.jsonl"
    runs = list(summary.get("runs", []))
    if runs:
        run_jsonl = Path(str(runs[0].get("output_jsonl", jsonl_path)))
        if run_jsonl.is_file():
            jsonl_path = run_jsonl
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    return jsonl_path, summary


def collect_postprocess_outputs(run_dir: Path, postprocess_out: Path, video: Path) -> dict[str, Any]:
    summary_path = postprocess_out / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)

    sqlite_links: dict[str, str] = {}
    overlay_links: dict[str, str] = {}
    interval_results = dict(summary.get("interval_results", {}))

    for label, result_obj in sorted(interval_results.items()):
        if not isinstance(result_obj, dict):
            continue
        paths = result_obj.get("paths", {})
        if not isinstance(paths, dict):
            continue

        sqlite_path = paths.get("merged_pred_sqlite")
        if sqlite_path:
            src = Path(str(sqlite_path))
            if src.is_file():
                dst = run_dir / "sqlite" / f"{video.stem}_{label}_predictions.sqlite"
                sqlite_links[label] = str(link_or_copy(src, dst))

        overlay_path = paths.get("overlay_video")
        if overlay_path:
            src = Path(str(overlay_path))
            if src.is_file():
                dst = run_dir / "overlay" / f"{video.stem}_{label}_postprocess.mp4"
                overlay_links[label] = str(link_or_copy(src, dst))

    tracked_sqlite = summary.get("tracked_sqlite")
    tracked_link = None
    if tracked_sqlite:
        src = Path(str(tracked_sqlite))
        if src.is_file():
            tracked_link = str(link_or_copy(src, run_dir / "sqlite" / f"{video.stem}_tracked.sqlite"))

    return {
        "summary": str(summary_path),
        "tracked_sqlite": tracked_sqlite,
        "tracked_sqlite_link": tracked_link,
        "prediction_sqlite_links": sqlite_links,
        "overlay_links": overlay_links,
        "interval_results": interval_results,
    }


def model_status(model_root: Path) -> dict[str, Any]:
    return {
        "model_root": str(model_root),
        "k2_checkpoint": str(model_root / "k2_v5" / "best_exact.pt"),
        "k2_checkpoint_exists": (model_root / "k2_v5" / "best_exact.pt").is_file(),
        "polygon_checkpoint": str(model_root / "polygon_point_predictor" / "best.pt"),
        "polygon_checkpoint_exists": (model_root / "polygon_point_predictor" / "best.pt").is_file(),
        "polygon_feature_stats": str(model_root / "polygon_point_predictor" / "feature_stats.npz"),
        "polygon_feature_stats_exists": (
            model_root / "polygon_point_predictor" / "feature_stats.npz"
        ).is_file(),
    }


def run_one_video(args: argparse.Namespace, video: Path, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    dinov3_out = run_dir / "dinov3"
    postprocess_out = run_dir / "postprocess"

    timings: list[dict[str, Any]] = []
    timings.append(
        run_command(
            build_dinov3_command(args, video, dinov3_out),
            cwd=args.dinov3_runtime,
            label=f"dinov3 inference: {video.name}",
        )
    )
    jsonl_path, dinov3_summary = summarize_dinov3(dinov3_out, video)

    postprocess_summary: dict[str, Any] | None = None
    if args.postprocess:
        timings.append(
            run_command(
                build_postprocess_command(args, video, jsonl_path, postprocess_out),
                cwd=args.atosyori_repo,
                env=atosyori_env(args),
                label=f"atosyori postprocess: {video.name}",
            )
        )
        postprocess_summary = collect_postprocess_outputs(run_dir, postprocess_out, video)

    summary = {
        "video": str(video),
        "run_dir": str(run_dir),
        "dinov3_runtime": str(args.dinov3_runtime),
        "atosyori_repo": str(args.atosyori_repo),
        "postprocess_model_status": model_status(args.postprocess_model_root),
        "class_policy_json": None if args.class_policy_json is None else str(args.class_policy_json),
        "artifacts": {
            "dinov3_jsonl": str(jsonl_path),
            "dinov3_summary": str(dinov3_out / "summary.json"),
            "postprocess_summary": None if postprocess_summary is None else postprocess_summary["summary"],
        },
        "dinov3": {
            "classifier_enabled": dinov3_summary.get("classifier_enabled"),
            "class_names": dinov3_summary.get("class_names"),
            "runs": dinov3_summary.get("runs", []),
        },
        "postprocess": postprocess_summary,
        "timings": timings,
        "total_wall_seconds": float(sum(float(row["wall_seconds"]) for row in timings)),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[summary] {run_dir / 'summary.json'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video -> DINOv3 JSONL/classification -> Atosyori SQLite/overlay pipeline"
    )
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output-root", type=Path, default=INTEGRATION_ROOT / "output" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dinov3-runtime", type=Path, default=DEFAULT_DINOV3_RUNTIME)
    parser.add_argument("--atosyori-repo", type=Path, default=DEFAULT_ATOSYORI_REPO)
    parser.add_argument("--postprocess-model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_DETECTOR_CHECKPOINT)
    parser.add_argument("--trt-backbone-engine", type=Path, default=DEFAULT_TRT_BACKBONE_ENGINE)
    parser.add_argument("--backbone-weights", type=Path, default=DEFAULT_DINOV3_WEIGHTS)
    parser.add_argument("--target-size", default="1280x720")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--rpn-pre-nms-topk-test", type=int, default=100)
    parser.add_argument("--rpn-post-nms-topk-test", type=int, default=40)
    parser.add_argument("--rpn-nms-thresh", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=300)
    parser.add_argument("--json-backend", choices=("json", "orjson"), default="orjson")
    parser.add_argument("--mask-approx", choices=("none", "simple"), default="none")
    parser.add_argument("--async-writer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-prefetch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-detector-overlay", action="store_true")

    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intervals", default="3,6")
    parser.add_argument("--class-policy-json", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--default-shape-mode", choices=("ellipse", "polygon"), default="ellipse")
    parser.add_argument("--render-overlays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overlay-encoder", choices=("cpu", "nvenc"), default="cpu")
    parser.add_argument("--raw-cut-detect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-remove-short-tracks-max-frames", type=int, default=10)
    parser.add_argument("--k2-device", default="auto")
    parser.add_argument("--polygon-predictor-device", default="auto")
    parser.add_argument(
        "--postprocess-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended to atosyori-postprocess run after '--'.",
    )
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input = abs_path(args.input)
    args.output_root = abs_path(args.output_root)
    args.python = abs_path_preserve_symlink(args.python)
    args.dinov3_runtime = abs_path(args.dinov3_runtime)
    args.atosyori_repo = abs_path(args.atosyori_repo)
    args.postprocess_model_root = abs_path(args.postprocess_model_root)
    for name in (
        "classifier_checkpoint",
        "detector_checkpoint",
        "trt_backbone_engine",
        "backbone_weights",
        "class_policy_json",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, abs_path(value))
    if args.postprocess_extra_args and args.postprocess_extra_args[0] == "--":
        args.postprocess_extra_args = args.postprocess_extra_args[1:]
    return args


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    videos = collect_videos(args.input, args.recursive)
    if not videos:
        raise RuntimeError(f"no videos found under: {args.input}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"
    root_run_dir = args.output_root / run_name
    all_summaries: list[dict[str, Any]] = []

    for video in videos:
        video_run_dir = root_run_dir / video.stem if len(videos) > 1 else root_run_dir
        all_summaries.append(run_one_video(args, video, video_run_dir))

    index = {
        "input": str(args.input),
        "output_root": str(args.output_root),
        "run_dir": str(root_run_dir),
        "video_count": len(videos),
        "summaries": [str(Path(str(item["run_dir"])) / "summary.json") for item in all_summaries],
    }
    write_json(root_run_dir / "index.json", index)
    print(f"[index] {root_run_dir / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
