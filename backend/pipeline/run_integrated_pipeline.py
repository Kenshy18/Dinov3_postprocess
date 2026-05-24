#!/usr/bin/env python3
"""Run detector video inference, optional ROI classification, and Atosyori postprocess.

The integration intentionally keeps detector runtimes and the
Atosyori postprocess repository as separate components. This script wires them
together and writes a compact per-video summary with easy-to-find SQLite and
overlay links.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .pipeline_commands import atosyori_env, build_detector_command, build_head_face_command, build_postprocess_command
from .pipeline_defaults import (
    CODINO_DEFAULT_AMP,
    CODINO_DEFAULT_ASYNC_WRITER,
    CODINO_DEFAULT_BATCH_SIZE,
    CODINO_DEFAULT_DISABLE_MASK_IOU_HEAD,
    CODINO_DEFAULT_JSON_BACKEND,
    CODINO_DEFAULT_MASK_APPROX,
    CODINO_DEFAULT_MODEL_SCORE_THR,
    CODINO_DEFAULT_SCORE_THRESH,
    CODINO_DEFAULT_TARGET_SIZE,
    CODINO_DEFAULT_TF32,
    CODINO_DEFAULT_TRT_FEATURE_NAMES,
    CODINO_DEFAULT_TRT_QUERY_ENCODER_SHAPES,
    CODINO_DEFAULT_WARMUP_FRAMES,
    DETECTOR_CHOICES,
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_CODINO_CHECKPOINT,
    DEFAULT_CODINO_CLASSIFIER_CHECKPOINT,
    DEFAULT_CODINO_CONFIG,
    DEFAULT_CODINO_RUNTIME,
    DEFAULT_CODINO_RUNTIME_SCRIPT,
    DEFAULT_CODINO_TRT_BACKBONE_ENGINE,
    DEFAULT_CODINO_TRT_DECODER_ENGINE,
    DEFAULT_CODINO_TRT_EXTRA_SITE_PACKAGES,
    DEFAULT_CODINO_TRT_FEATURE_ENGINE,
    DEFAULT_CODINO_TRT_MASK_HEAD_ENGINE,
    DEFAULT_CODINO_TRT_QUERY_ENCODER_ENGINE,
    DEFAULT_DETECTOR_CHECKPOINT,
    DEFAULT_DINOV3_RUNTIME,
    DEFAULT_DINOV3_WEIGHTS,
    DEFAULT_EVA02_CLASSIFIER_CHECKPOINT,
    DEFAULT_EVA02_DETECTOR_CHECKPOINT,
    DEFAULT_EVA02_RUNTIME,
    DEFAULT_MODEL_ROOT,
    DEFAULT_POLICY,
    DEFAULT_TRT_BACKBONE_ENGINE,
    DINO_DEFAULT_BATCH_SIZE,
    DINO_DEFAULT_WARMUP_FRAMES,
    EVA02_DEFAULT_ASYNC_WRITER,
    EVA02_DEFAULT_BATCH_SIZE,
    EVA02_DEFAULT_CLASSIFIER_BATCH_SIZE,
    EVA02_DEFAULT_JSON_BACKEND,
    EVA02_DEFAULT_MASK_APPROX,
    EVA02_DEFAULT_NMS_THRESH,
    EVA02_DEFAULT_SCORE_THRESH,
    EVA02_DEFAULT_TARGET_SIZE,
    EVA02_DEFAULT_TOPK,
    EVA02_DEFAULT_WARMUP_FRAMES,
    INTEGRATION_ROOT,
    LOCAL_ATOSYORI_REPO,
)
from .pipeline_outputs import collect_postprocess_outputs, model_status, summarize_detector, write_json
from .progress import ProgressReporter, limited_total
from backend.schemas.head_face_sqlite import enrich_head_face_sqlite, merge_ai_and_head_face_sqlite
from backend.schemas.mask_sqlite import jsonl_to_raw_sqlite
from backend.schemas.postprocess_coordinate_space import (
    transform_from_summary,
    write_postprocess_work_jsonl,
    write_source_space_mask_sqlite,
)


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

DEFAULT_ATOSYORI_REPO = Path(
    os.environ.get(
        "ATOSYORI_REPO",
        str(LOCAL_ATOSYORI_REPO),
    )
)


def default_rtdetr_repo() -> Path:
    env_value = os.environ.get("RTDETR_REPO")
    if env_value:
        return Path(env_value)
    for candidate in (
        INTEGRATION_ROOT / "external" / "RT-DETR" / "RT-DETRv4",
        INTEGRATION_ROOT.parent / "CV" / "RT-DETR" / "RT-DETRv4",
        INTEGRATION_ROOT.parent / "RT-DETR" / "RT-DETRv4",
    ):
        if (candidate / "tools" / "inference" / "video_sqlite_inf.py").is_file():
            return candidate
    return INTEGRATION_ROOT / "external" / "RT-DETR" / "RT-DETRv4"


DEFAULT_RTDETR_REPO = default_rtdetr_repo()
DEFAULT_RTDETR_CONFIG = DEFAULT_RTDETR_REPO / "configs" / "rtv2" / "rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml"
DEFAULT_RTDETR_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "rtdetr" / "head_face_best_stg1.pth"


def restore_postprocess_outputs_to_source_space(
    *,
    run_dir: Path,
    video: Path,
    postprocess_summary: dict[str, Any],
    coordinate_summary: dict[str, Any],
) -> dict[str, Any]:
    transform = transform_from_summary(coordinate_summary)
    sqlite_dir = run_dir / "sqlite"
    restore_summaries: dict[str, Any] = {}

    work_prediction_links = dict(postprocess_summary.get("prediction_sqlite_links") or {})
    source_prediction_links: dict[str, str] = {}

    if transform.is_identity:
        work_tracked_sqlite = postprocess_summary.get("tracked_sqlite")
        work_tracked_sqlite_link = postprocess_summary.get("tracked_sqlite_link")
        tracked_link = Path(str(work_tracked_sqlite_link)) if work_tracked_sqlite_link else None
        tracked_src = Path(str(work_tracked_sqlite)) if work_tracked_sqlite else None
        source_tracked_sqlite: str | None = None
        if tracked_link is not None and tracked_link.is_file():
            source_tracked_sqlite = str(tracked_link)
        elif tracked_src is not None and tracked_src.is_file():
            source_tracked_sqlite = str(tracked_src)

        postprocess_summary["coordinate_space"] = coordinate_summary
        postprocess_summary["work_prediction_sqlite_links"] = work_prediction_links
        postprocess_summary["source_prediction_sqlite_links"] = work_prediction_links
        postprocess_summary["prediction_sqlite_links"] = work_prediction_links
        postprocess_summary["work_tracked_sqlite"] = work_tracked_sqlite
        postprocess_summary["work_tracked_sqlite_link"] = work_tracked_sqlite_link
        if source_tracked_sqlite is not None:
            postprocess_summary["tracked_sqlite"] = source_tracked_sqlite
            postprocess_summary["tracked_sqlite_link"] = source_tracked_sqlite
        postprocess_summary["coordinate_restore_summaries"] = restore_summaries
        return postprocess_summary

    for label, value in sorted(work_prediction_links.items()):
        result = postprocess_summary.get("interval_results", {}).get(label, {})
        paths = result.get("paths", {}) if isinstance(result, dict) else {}
        work_src_raw = paths.get("merged_pred_sqlite") if isinstance(paths, dict) else None
        src = Path(str(work_src_raw or value))
        if not src.is_file():
            continue
        dst = Path(str(value))
        restore_summaries[f"prediction:{label}"] = write_source_space_mask_sqlite(src, dst, coordinate_summary)
        source_prediction_links[str(label)] = str(dst)
        work_prediction_links[str(label)] = str(src)

    work_tracked_sqlite = postprocess_summary.get("tracked_sqlite")
    work_tracked_sqlite_link = postprocess_summary.get("tracked_sqlite_link")
    tracked_src_raw = work_tracked_sqlite or work_tracked_sqlite_link
    source_tracked_sqlite: str | None = None
    if tracked_src_raw:
        tracked_src = Path(str(tracked_src_raw))
        if tracked_src.is_file():
            dst = Path(str(work_tracked_sqlite_link or sqlite_dir / f"{video.stem}_tracked.sqlite"))
            restore_summaries["tracked"] = write_source_space_mask_sqlite(tracked_src, dst, coordinate_summary)
            source_tracked_sqlite = str(dst)

    postprocess_summary["coordinate_space"] = coordinate_summary
    postprocess_summary["work_prediction_sqlite_links"] = work_prediction_links
    postprocess_summary["source_prediction_sqlite_links"] = source_prediction_links
    postprocess_summary["prediction_sqlite_links"] = source_prediction_links
    postprocess_summary["work_tracked_sqlite"] = work_tracked_sqlite
    postprocess_summary["work_tracked_sqlite_link"] = work_tracked_sqlite_link
    if source_tracked_sqlite is not None:
        postprocess_summary["tracked_sqlite"] = source_tracked_sqlite
        postprocess_summary["tracked_sqlite_link"] = source_tracked_sqlite
    postprocess_summary["coordinate_restore_summaries"] = restore_summaries
    return postprocess_summary


def runtime_profile_recommendations() -> dict[str, Any]:
    for raw in (
        os.environ.get("DINOV3_RUNTIME_PROFILE"),
        str(INTEGRATION_ROOT / ".runtime" / "runtime_profile.json"),
        str(INTEGRATION_ROOT / "configs" / "runtime_profile.json"),
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = INTEGRATION_ROOT / path
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                recs = data.get("recommendations", {})
                return recs if isinstance(recs, dict) else {}
        except Exception:
            continue
    return {}


def runtime_profile_int(section: str, key: str, default: int) -> int:
    try:
        value = runtime_profile_recommendations().get(section, {}).get(key)
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def runtime_profile_str(section: str, key: str, default: str | None = None) -> str | None:
    try:
        value = runtime_profile_recommendations().get(section, {}).get(key)
    except Exception:
        return default
    if value in (None, ""):
        return default
    return str(value)


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
    print(f"[phase-start] command: {label}", flush=True)
    print("[cmd] " + " ".join(command), flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd) if cwd is not None else None, env=env, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    print(f"[done] {label}: {elapsed:.2f}s", flush=True)
    print(f"[phase-done] command: {label} elapsed={elapsed:.2f}s", flush=True)
    return {
        "label": label,
        "cmd": command,
        "cwd": None if cwd is None else str(cwd),
        "returncode": int(completed.returncode),
        "wall_seconds": float(elapsed),
    }


def run_head_face_command(
    command: list[str],
    *,
    cwd: Path,
    label: str,
    video: Path,
    total_frames: int | None,
    progress_interval_sec: float,
) -> dict[str, Any]:
    print(f"[run] {label}", flush=True)
    print(f"[phase-start] command: {label}", flush=True)
    print("[cmd] " + " ".join(command), flush=True)
    progress = ProgressReporter(
        "head_face",
        total=total_frames,
        unit="frames",
        interval_sec=float(progress_interval_sec),
        static_fields={"video": video.name},
    )
    progress.emit(0, force=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.perf_counter()
    processed_frames = 0
    rows = 0
    last_fps: float | None = None
    pattern = re.compile(r"processed\s+(\d+)(?:/(\d+))?\s+frames.*?rows=(\d+).*?throughput=([0-9.]+)")
    final_pattern = re.compile(r"processed\s+(\d+)\s+frames\s+in\s+[0-9.]+s\s+\(([0-9.]+)\s+fps\)")
    rows_pattern = re.compile(r"wrote\s+(\d+)\s+detection rows")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = pattern.search(line)
        if match:
            processed_frames = int(match.group(1))
            rows = int(match.group(3))
            fps = float(match.group(4))
            last_fps = fps
            progress.emit(processed_frames, fps=fps, extra={"detections": rows})
            continue
        final_match = final_pattern.search(line)
        if final_match:
            processed_frames = int(final_match.group(1))
            fps = float(final_match.group(2))
            last_fps = fps
            progress.emit(processed_frames, fps=fps, extra={"detections": rows})
            continue
        rows_match = rows_pattern.search(line)
        if rows_match:
            rows = int(rows_match.group(1))
            progress.emit(processed_frames, extra={"detections": rows})
    returncode = process.wait()
    elapsed = time.perf_counter() - start
    if progress.last_current != processed_frames:
        progress.emit(processed_frames, force=True, extra={"detections": rows})
    if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")
    print(f"[done] {label}: {elapsed:.2f}s", flush=True)
    print(f"[phase-done] command: {label} elapsed={elapsed:.2f}s", flush=True)
    return {
        "label": label,
        "cmd": command,
        "cwd": str(cwd),
        "returncode": int(returncode),
        "wall_seconds": float(elapsed),
        "processed_frames": int(processed_frames),
        "detections": int(rows),
        "fps": last_fps,
        "e2e_fps": last_fps,
    }


def detector_processed_frames(summary: dict[str, Any]) -> int | None:
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return None
    total = 0
    for item in runs:
        if isinstance(item, dict):
            total += int(item.get("processed_frames") or item.get("frames") or 0)
    return total or None


def video_frame_count(video: Path) -> int | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return count if count > 0 else None
    finally:
        cap.release()


def run_one_video(args: argparse.Namespace, video: Path, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    detector_out = run_dir / args.detector
    postprocess_out = run_dir / "postprocess"
    head_face_only = bool(args.head_face_only)
    detector_runtime = {
        "dinov3": args.dinov3_runtime,
        "eva02": args.eva02_runtime,
        "codino": args.codino_runtime,
    }[args.detector]

    timings: list[dict[str, Any]] = []
    jsonl_path: Path | None = None
    detector_summary: dict[str, Any] = {
        "classifier_enabled": None,
        "class_names": [],
        "runs": [{"processed_frames": video_frame_count(video)}],
    }
    if not head_face_only:
        timings.append(
            run_command(
                build_detector_command(args, video, detector_out),
                cwd=detector_runtime,
                label=f"{args.detector} inference: {video.name}",
            )
        )
        jsonl_path, detector_summary = summarize_detector(detector_out, video)
    raw_sqlite_summary: dict[str, Any] | None = None
    raw_sqlite_path: Path | None = None
    if args.raw_sqlite and jsonl_path is not None:
        raw_sqlite_path = run_dir / "sqlite" / f"{video.stem}_raw_detections.sqlite"
        raw_progress = ProgressReporter(
            "raw_sqlite",
            total=limited_total(detector_processed_frames(detector_summary), args.max_frames),
            unit="frames",
            interval_sec=float(args.progress_interval_sec),
            static_fields={"video": video.name, "detector": args.detector},
        )
        raw_progress.emit(0, force=True)
        raw_sqlite_summary = jsonl_to_raw_sqlite(
            jsonl_path,
            raw_sqlite_path,
            detector=args.detector,
            video=video,
            progress_callback=lambda frames, masks: raw_progress.emit(frames, extra={"masks": masks}),
        )
        raw_progress.emit(int(raw_sqlite_summary["frames"]), force=True, extra={"masks": raw_sqlite_summary["masks"]})
        print(
            f"[raw-sqlite] {raw_sqlite_path} frames={raw_sqlite_summary['frames']} masks={raw_sqlite_summary['masks']}",
            flush=True,
        )

    head_face_summary: dict[str, Any] | None = None
    head_face_sqlite_path: Path | None = None
    if args.head_face_detect:
        head_face_sqlite_path = run_dir / "head_face" / "sqlite" / f"{video.stem}_head_face.sqlite"
        head_face_sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if head_face_sqlite_path.exists() and args.force:
            head_face_sqlite_path.unlink()
        if not args.rtdetr_repo.is_dir():
            raise FileNotFoundError(
                f"RT-DETR repo not found: {args.rtdetr_repo}. "
                "Set RTDETR_REPO or pass --rtdetr-repo when --head-face-detect is enabled."
            )
        if args.rtdetr_config is None or not args.rtdetr_config.is_file():
            raise FileNotFoundError(
                f"RT-DETR config not found: {args.rtdetr_config}. "
                "Run tools/setup_runtime.sh or pass --rtdetr-config."
            )
        if args.rtdetr_checkpoint is None or not args.rtdetr_checkpoint.is_file():
            raise FileNotFoundError(
                f"RT-DETR checkpoint not found: {args.rtdetr_checkpoint}. "
                "Run tools/setup_runtime.sh with RUNTIME_ARTIFACTS_URL/RUNTIME_ARTIFACTS_DIR, "
                "or pass --rtdetr-checkpoint."
            )
        timings.append(
            run_head_face_command(
                build_head_face_command(args, video, head_face_sqlite_path),
                cwd=args.rtdetr_repo,
                label=f"head/face RT-DETR inference: {video.name}",
                video=video,
                total_frames=limited_total(detector_processed_frames(detector_summary) or video_frame_count(video), args.max_frames),
                progress_interval_sec=float(args.progress_interval_sec),
            )
        )
        head_face_summary = enrich_head_face_sqlite(head_face_sqlite_path)
        print(
            f"[head-face-sqlite] {head_face_sqlite_path} "
            f"detections={head_face_summary['detections']} faces={head_face_summary['faces']} "
            f"heads={head_face_summary['heads']} tracks={head_face_summary['tracks']}",
            flush=True,
        )

    postprocess_summary: dict[str, Any] | None = None
    postprocess_work_jsonl_path: Path | None = None
    postprocess_coordinate_summary: dict[str, Any] | None = None
    postprocess_coordinate_summary_path: Path | None = None
    combined_final_summary: dict[str, Any] | None = None
    combined_final_sqlite_path: Path | None = None
    if args.postprocess and jsonl_path is not None:
        postprocess_work_jsonl_path = run_dir / "postprocess_input" / f"{video.stem}_postprocess_1920x1080.jsonl"
        postprocess_coordinate_summary = write_postprocess_work_jsonl(jsonl_path, postprocess_work_jsonl_path)
        postprocess_coordinate_summary_path = run_dir / "postprocess_input" / "coordinate_space.json"
        write_json(postprocess_coordinate_summary_path, postprocess_coordinate_summary)
        transform = transform_from_summary(postprocess_coordinate_summary)
        if not transform.is_identity:
            print(
                "[postprocess-coordinate-space] "
                f"{transform.source_width}x{transform.source_height} -> "
                f"{transform.work_width}x{transform.work_height} "
                f"scale={transform.scale:.8f} pad=({transform.pad_left:.3f},{transform.pad_top:.3f})",
                flush=True,
            )
        timings.append(
            run_command(
                build_postprocess_command(args, video, postprocess_work_jsonl_path, postprocess_out),
                cwd=args.atosyori_repo,
                env=atosyori_env(args),
                label=f"atosyori postprocess: {video.name}",
            )
        )
        postprocess_summary = collect_postprocess_outputs(run_dir, postprocess_out, video)
        postprocess_summary = restore_postprocess_outputs_to_source_space(
            run_dir=run_dir,
            video=video,
            postprocess_summary=postprocess_summary,
            coordinate_summary=postprocess_coordinate_summary,
        )
        if head_face_sqlite_path is not None and head_face_summary is not None:
            combined_final_sqlite_path = run_dir / "sqlite" / f"{video.stem}_combined_final.sqlite"
            combined_final_summary = merge_ai_and_head_face_sqlite(
                ai_sqlites={str(key): str(value) for key, value in (postprocess_summary.get("prediction_sqlite_links") or {}).items()},
                head_face_sqlite=head_face_sqlite_path,
                output_sqlite=combined_final_sqlite_path,
            )
            print(
                f"[combined-final-sqlite] {combined_final_sqlite_path} "
                f"ai_masks={combined_final_summary['ai_mask_rows']} "
                f"head_face={combined_final_summary['head_face_detections']} "
                f"studio_face_masks={combined_final_summary.get('studio_face_masks', 0)}",
                flush=True,
            )

    summary = {
        "video": str(video),
        "run_dir": str(run_dir),
        "detector": args.detector,
        "detector_runtime": str(detector_runtime),
        "dinov3_runtime": str(args.dinov3_runtime),
        "eva02_runtime": str(args.eva02_runtime),
        "codino_runtime": str(args.codino_runtime),
        "atosyori_repo": str(args.atosyori_repo),
        "postprocess_model_status": model_status(args.postprocess_model_root),
        "head_face_enabled": bool(args.head_face_detect),
        "head_face_only": head_face_only,
        "class_policy_json": None if args.class_policy_json is None else str(args.class_policy_json),
        "artifacts": {
            "detector_jsonl": None if jsonl_path is None else str(jsonl_path),
            "raw_sqlite": None if raw_sqlite_path is None else str(raw_sqlite_path),
            "head_face_sqlite": None if head_face_sqlite_path is None else str(head_face_sqlite_path),
            "combined_final_sqlite": None if combined_final_sqlite_path is None else str(combined_final_sqlite_path),
            "postprocess_work_jsonl": None if postprocess_work_jsonl_path is None else str(postprocess_work_jsonl_path),
            "postprocess_coordinate_summary": None
            if postprocess_coordinate_summary_path is None
            else str(postprocess_coordinate_summary_path),
            "detector_summary": None if jsonl_path is None else str(detector_out / "summary.json"),
            "dinov3_jsonl": str(jsonl_path) if args.detector == "dinov3" and jsonl_path is not None else None,
            "dinov3_summary": str(detector_out / "summary.json") if args.detector == "dinov3" and jsonl_path is not None else None,
            "codino_jsonl": str(jsonl_path) if args.detector == "codino" and jsonl_path is not None else None,
            "codino_summary": str(detector_out / "summary.json") if args.detector == "codino" and jsonl_path is not None else None,
            "postprocess_summary": None if postprocess_summary is None else postprocess_summary["summary"],
        },
        "raw_sqlite": raw_sqlite_summary,
        "head_face": head_face_summary,
        "combined_final_sqlite": combined_final_summary,
        "detector_summary": {
            "classifier_enabled": detector_summary.get("classifier_enabled"),
            "class_names": detector_summary.get("class_names"),
            "runs": detector_summary.get("runs", []),
        },
        "dinov3": {
            "classifier_enabled": detector_summary.get("classifier_enabled") if args.detector == "dinov3" else None,
            "class_names": detector_summary.get("class_names") if args.detector == "dinov3" else None,
            "runs": detector_summary.get("runs", []) if args.detector == "dinov3" else [],
        },
        "codino": {
            "classifier_enabled": detector_summary.get("classifier_enabled") if args.detector == "codino" else None,
            "class_names": detector_summary.get("class_names") if args.detector == "codino" else None,
            "runs": detector_summary.get("runs", []) if args.detector == "codino" else [],
            "trt": detector_summary.get("trt") if args.detector == "codino" else None,
        },
        "postprocess": postprocess_summary,
        "postprocess_coordinate_space": postprocess_coordinate_summary,
        "timings": timings,
        "total_wall_seconds": float(sum(float(row["wall_seconds"]) for row in timings)),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[summary] {run_dir / 'summary.json'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video -> detector JSONL/classification -> Atosyori SQLite/overlay pipeline"
    )
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output-root", type=Path, default=INTEGRATION_ROOT / "output" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--dinov3-python",
        type=Path,
        default=None,
        help="Python executable for DINOv3 detector inference. Defaults to --python.",
    )
    parser.add_argument(
        "--eva02-python",
        type=Path,
        default=None,
        help="Python executable for EVA02 detector inference. Defaults to --python.",
    )
    parser.add_argument("--detector", choices=DETECTOR_CHOICES, default="dinov3")
    parser.add_argument("--dinov3-runtime", type=Path, default=DEFAULT_DINOV3_RUNTIME)
    parser.add_argument("--eva02-runtime", type=Path, default=DEFAULT_EVA02_RUNTIME)
    parser.add_argument("--codino-runtime", type=Path, default=DEFAULT_CODINO_RUNTIME)
    parser.add_argument("--atosyori-repo", type=Path, default=DEFAULT_ATOSYORI_REPO)
    parser.add_argument("--rtdetr-repo", type=Path, default=DEFAULT_RTDETR_REPO)
    parser.add_argument("--postprocess-model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_DETECTOR_CHECKPOINT)
    parser.add_argument("--eva02-classifier-checkpoint", type=Path, default=DEFAULT_EVA02_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--eva02-detector-checkpoint", type=Path, default=DEFAULT_EVA02_DETECTOR_CHECKPOINT)
    parser.add_argument("--codino-runtime-script", type=Path, default=DEFAULT_CODINO_RUNTIME_SCRIPT)
    parser.add_argument("--codino-config", type=Path, default=DEFAULT_CODINO_CONFIG)
    parser.add_argument("--codino-checkpoint", type=Path, default=DEFAULT_CODINO_CHECKPOINT)
    parser.add_argument("--codino-classifier-checkpoint", type=Path, default=DEFAULT_CODINO_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--trt-backbone-engine", type=Path, default=DEFAULT_TRT_BACKBONE_ENGINE)
    parser.add_argument("--backbone-weights", type=Path, default=DEFAULT_DINOV3_WEIGHTS)
    parser.add_argument("--target-size", default="720x1280")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--rpn-pre-nms-topk-test", type=int, default=100)
    parser.add_argument("--rpn-post-nms-topk-test", type=int, default=40)
    parser.add_argument("--rpn-nms-thresh", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=DINO_DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=DINO_DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--json-backend", choices=("json", "orjson"), default="orjson")
    parser.add_argument("--mask-approx", choices=("none", "simple"), default="none")
    parser.add_argument("--async-writer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-prefetch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-detector-overlay", action="store_true")
    parser.add_argument("--raw-sqlite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval-sec", type=float, default=float(os.environ.get("PIPELINE_PROGRESS_INTERVAL_SEC", "5")))
    parser.add_argument("--head-face-detect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--head-face-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only RT-DETR Head/Face detection and skip the main AI detector/postprocess.",
    )
    parser.add_argument(
        "--rtdetr-config",
        type=Path,
        default=os.environ.get("RTDETR_CONFIG")
        or runtime_profile_str("rtdetr", "config", str(DEFAULT_RTDETR_CONFIG)),
    )
    parser.add_argument(
        "--rtdetr-checkpoint",
        type=Path,
        default=os.environ.get("RTDETR_CHECKPOINT")
        or runtime_profile_str("rtdetr", "checkpoint", str(DEFAULT_RTDETR_CHECKPOINT)),
    )
    parser.add_argument("--head-face-device", default=os.environ.get("RTDETR_DEVICE", runtime_profile_str("rtdetr", "device", "cuda:0")))
    parser.add_argument(
        "--head-face-batch-size",
        type=int,
        default=int(os.environ.get("RTDETR_BATCH_SIZE") or runtime_profile_int("rtdetr", "batch_size", 128)),
    )
    parser.add_argument("--head-face-conf-thr", type=float, default=float(os.environ.get("RTDETR_CONF_THR", "0.50")))
    parser.add_argument("--head-face-low-thr", type=float, default=float(os.environ.get("RTDETR_LOW_THR", "0.15")))
    parser.add_argument("--head-face-new-track-thr", type=float, default=float(os.environ.get("RTDETR_NEW_TRACK_THR", "0.55")))
    parser.add_argument("--head-face-track-min-hits", type=int, default=int(os.environ.get("RTDETR_TRACK_MIN_HITS", "5")))
    parser.add_argument("--head-face-nms-iou-thr", type=float, default=float(os.environ.get("RTDETR_NMS_IOU_THR", "0.55")))
    parser.add_argument(
        "--head-face-progress-interval",
        type=int,
        default=int(os.environ.get("RTDETR_PROGRESS_INTERVAL") or runtime_profile_int("rtdetr", "progress_interval", 30)),
    )
    parser.add_argument("--head-face-compile", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--eva02-target-size", type=int, default=EVA02_DEFAULT_TARGET_SIZE)
    parser.add_argument("--eva02-score-thresh", type=float, default=EVA02_DEFAULT_SCORE_THRESH)
    parser.add_argument("--eva02-nms-thresh", type=float, default=EVA02_DEFAULT_NMS_THRESH)
    parser.add_argument("--eva02-topk", type=int, default=EVA02_DEFAULT_TOPK)
    parser.add_argument("--eva02-batch-size", type=int, default=EVA02_DEFAULT_BATCH_SIZE)
    parser.add_argument("--eva02-warmup-frames", type=int, default=EVA02_DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--eva02-classifier-batch-size", type=int, default=EVA02_DEFAULT_CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--eva02-compile-backbone", default="max-autotune")
    parser.add_argument("--eva02-json-backend", choices=("json", "orjson"), default=EVA02_DEFAULT_JSON_BACKEND)
    parser.add_argument("--eva02-mask-approx", choices=("none", "simple"), default=EVA02_DEFAULT_MASK_APPROX)
    parser.add_argument("--eva02-async-writer", action=argparse.BooleanOptionalAction, default=EVA02_DEFAULT_ASYNC_WRITER)

    parser.add_argument("--codino-target-size", default=CODINO_DEFAULT_TARGET_SIZE)
    parser.add_argument("--codino-score-thresh", type=float, default=CODINO_DEFAULT_SCORE_THRESH)
    parser.add_argument("--codino-model-score-thr", type=float, default=CODINO_DEFAULT_MODEL_SCORE_THR)
    parser.add_argument("--codino-batch-size", type=int, default=CODINO_DEFAULT_BATCH_SIZE)
    parser.add_argument("--codino-warmup-frames", type=int, default=CODINO_DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--codino-json-backend", choices=("json", "orjson"), default=CODINO_DEFAULT_JSON_BACKEND)
    parser.add_argument("--codino-mask-approx", choices=("none", "simple"), default=CODINO_DEFAULT_MASK_APPROX)
    parser.add_argument("--codino-async-writer", action=argparse.BooleanOptionalAction, default=CODINO_DEFAULT_ASYNC_WRITER)
    parser.add_argument("--codino-amp", choices=("fp16", "bf16", "off"), default=CODINO_DEFAULT_AMP)
    parser.add_argument("--codino-tf32", action=argparse.BooleanOptionalAction, default=CODINO_DEFAULT_TF32)
    parser.add_argument(
        "--codino-disable-mask-iou-head",
        action=argparse.BooleanOptionalAction,
        default=CODINO_DEFAULT_DISABLE_MASK_IOU_HEAD,
    )
    parser.add_argument("--codino-trt-backbone-engine", type=Path, default=DEFAULT_CODINO_TRT_BACKBONE_ENGINE)
    parser.add_argument("--codino-trt-feature-engine", type=Path, default=DEFAULT_CODINO_TRT_FEATURE_ENGINE)
    parser.add_argument("--codino-trt-feature-names", default=CODINO_DEFAULT_TRT_FEATURE_NAMES)
    parser.add_argument("--codino-trt-query-encoder-engine", type=Path, default=DEFAULT_CODINO_TRT_QUERY_ENCODER_ENGINE)
    parser.add_argument("--codino-trt-query-encoder-shapes", default=CODINO_DEFAULT_TRT_QUERY_ENCODER_SHAPES)
    parser.add_argument("--codino-trt-decoder-engine", type=Path, default=DEFAULT_CODINO_TRT_DECODER_ENGINE)
    parser.add_argument("--codino-trt-mask-head-engine", type=Path, default=DEFAULT_CODINO_TRT_MASK_HEAD_ENGINE)
    parser.add_argument("--codino-trt-extra-site-packages", type=Path, default=DEFAULT_CODINO_TRT_EXTRA_SITE_PACKAGES)

    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intervals", default="3")
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
    if args.detector == "eva02" and not args.classifier and not args.head_face_only:
        raise RuntimeError("EVA02 runtime currently requires --classifier")
    if args.head_face_only:
        args.head_face_detect = True
        args.postprocess = False
        args.raw_sqlite = False
    args.input = abs_path(args.input)
    args.output_root = abs_path(args.output_root)
    args.python = abs_path_preserve_symlink(args.python)
    args.dinov3_python = abs_path_preserve_symlink(args.dinov3_python) if args.dinov3_python is not None else args.python
    args.eva02_python = abs_path_preserve_symlink(args.eva02_python) if args.eva02_python is not None else args.python
    args.dinov3_runtime = abs_path(args.dinov3_runtime)
    args.eva02_runtime = abs_path(args.eva02_runtime)
    args.codino_runtime = abs_path(args.codino_runtime)
    args.atosyori_repo = abs_path(args.atosyori_repo)
    args.rtdetr_repo = abs_path(args.rtdetr_repo)
    args.postprocess_model_root = abs_path(args.postprocess_model_root)
    for name in (
        "classifier_checkpoint",
        "detector_checkpoint",
        "eva02_classifier_checkpoint",
        "eva02_detector_checkpoint",
        "codino_runtime_script",
        "codino_config",
        "codino_checkpoint",
        "codino_classifier_checkpoint",
        "trt_backbone_engine",
        "backbone_weights",
        "class_policy_json",
        "codino_trt_backbone_engine",
        "codino_trt_feature_engine",
        "codino_trt_query_encoder_engine",
        "codino_trt_decoder_engine",
        "codino_trt_mask_head_engine",
        "codino_trt_extra_site_packages",
        "rtdetr_config",
        "rtdetr_checkpoint",
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
        "detector": args.detector,
        "video_count": len(videos),
        "summaries": [str(Path(str(item["run_dir"])) / "summary.json") for item in all_summaries],
    }
    write_json(root_run_dir / "index.json", index)
    print(f"[index] {root_run_dir / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
