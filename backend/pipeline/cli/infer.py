#!/usr/bin/env python3
"""Primary user-facing inference/postprocess CLI.

This command is intentionally a small orchestration layer.  The detector and
postprocess implementations stay in the existing backend modules; this file
owns the stable command surface, batch progress, and GUI-compatible output
layout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from backend.pipeline.pipeline_defaults import DETECTOR_CHOICES, DEFAULT_MODEL_ROOT, LOCAL_ATOSYORI_REPO
from backend.pipeline.progress import BATCH_PROGRESS_PREFIX, encode_progress_fields, parse_progress_line

from .flow_cli_common import ROOT, add_policy_args, command_to_text, timestamp, validate_interval, validate_recall


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
PIPELINE_SCRIPT = ROOT / "scripts" / "run_integrated_pipeline.py"
UI_JOB_SCRIPT = ROOT / "apps" / "qt_ui" / "run_ui_job.py"
POSTPROCESS_SCRIPT = ROOT / "scripts" / "run_postprocess_only.py"


def clean_run_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("._-") or "video"


def collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTS:
            raise RuntimeError(f"input file is not a supported video: {path}")
        return [path]
    if not path.is_dir():
        raise RuntimeError(f"input path not found: {path}")
    iterator = path.rglob("*") if recursive else path.iterdir()
    videos = sorted(p.resolve() for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"no videos found under: {path}")
    return videos


def expand_existing_inputs(paths: list[Path], suffixes: tuple[str, ...]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if path.is_dir():
            expanded.extend(sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix in suffixes))
        elif path.is_file():
            expanded.append(path)
        else:
            raise RuntimeError(f"input artifact not found: {path}")
    if not expanded:
        raise RuntimeError("no input artifacts found")
    return expanded


def find_video_for_artifact(artifact: Path, input_video: Path | None, total: int) -> Path | None:
    if input_video is None:
        return None
    video = input_video.expanduser().resolve()
    if video.is_file():
        if total != 1:
            raise RuntimeError("--input-video may be a file only when one postprocess input is provided")
        return video
    if not video.is_dir():
        raise RuntimeError(f"input video path not found: {video}")
    stem_candidates = [artifact.stem]
    for suffix in (".tracked", "_raw_detections"):
        if artifact.stem.endswith(suffix):
            stem_candidates.append(artifact.stem[: -len(suffix)])
    for stem in stem_candidates:
        for ext in VIDEO_EXTS:
            candidate = video / f"{stem}{ext}"
            if candidate.is_file():
                return candidate.resolve()
    for candidate in video.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTS and candidate.stem in stem_candidates:
            return candidate.resolve()
    return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_audit(audit_path: Path, event: str, **fields: object) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _extract_fps(line: str) -> str | None:
    match = re.search(r"(?:e2e_fps|fps|render_fps)=([0-9]+(?:\.[0-9]+)?)", line)
    return match.group(1) if match else None


def pre_sqlite_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pre_sqlite", getattr(args, "raw_sqlite", True)))


def pre_overlay_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pre_overlay", getattr(args, "raw_overlay", False)))


def post_sqlite_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "post_sqlite", True))


def post_overlay_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "post_overlay", getattr(args, "overlay_mode", "detailed")))


def run_streamed(
    command: list[str],
    *,
    log_path: Path,
    audit_path: Path,
    dry_run: bool,
    item_index: int,
    item_total: int,
    phase: str,
) -> tuple[int, float, str | None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    last_fps: str | None = None
    print("[cmd] " + command_to_text(command), flush=True)
    write_audit(audit_path, "command_start", phase=phase, item_index=item_index, item_total=item_total, command=command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("[cmd] " + command_to_text(command) + "\n")
        log.flush()
        if dry_run:
            log.write("[dry-run]\n")
            write_audit(audit_path, "command_dry_run", phase=phase, item_index=item_index, item_total=item_total)
            return 0, 0.0, None
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
                progress_fields = parse_progress_line(line)
                if progress_fields:
                    phase_percent = None
                    try:
                        phase_percent = float(progress_fields.get("percent", ""))
                    except ValueError:
                        phase_percent = None
                    overall = None
                    if phase_percent is not None:
                        overall = ((item_index - 1) + max(0.0, min(100.0, phase_percent)) / 100.0) / max(1, item_total) * 100.0
                    batch_fields = {
                        "item": f"{item_index}/{item_total}",
                        "phase": progress_fields.get("phase"),
                        "phase_percent": phase_percent,
                        "overall": overall,
                        "fps": progress_fields.get("fps"),
                        "eta": progress_fields.get("eta"),
                    }
                    print(f"{BATCH_PROGRESS_PREFIX} {encode_progress_fields(batch_fields)}", flush=True)
                    write_audit(
                        audit_path,
                        "phase_progress",
                        item_index=item_index,
                        item_total=item_total,
                        phase=progress_fields.get("phase"),
                        phase_percent=phase_percent,
                        overall_percent=overall,
                        fields=progress_fields,
                    )
                    continue
                fps = _extract_fps(line)
                if fps is not None:
                    last_fps = fps
                    overall = ((item_index - 1) / max(1, item_total)) * 100.0
                    print(
                        f"[progress] item={item_index}/{item_total} phase={phase} fps={fps} overall={overall:.1f}%",
                        flush=True,
                    )
            returncode = process.wait()
        except BaseException as exc:
            write_audit(
                audit_path,
                "command_exception",
                phase=phase,
                item_index=item_index,
                item_total=item_total,
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            raise
    elapsed = time.perf_counter() - start
    write_audit(
        audit_path,
        "command_done",
        phase=phase,
        item_index=item_index,
        item_total=item_total,
        returncode=returncode,
        elapsed_seconds=elapsed,
        fps=last_fps,
    )
    return returncode, elapsed, last_fps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detector inference, optional Atosyori postprocess, and overlays with stable outputs."
    )
    parser.add_argument("--mode", choices=("full", "inference", "postprocess"), default="full")
    parser.add_argument("--input", type=Path, default=None, help="Video file or directory for full/inference modes")
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[], help="Detector JSONL for postprocess mode")
    parser.add_argument(
        "--input-sqlite",
        type=Path,
        action="append",
        default=[],
        help="Tracked SQLite for postprocess mode",
    )
    parser.add_argument("--input-video", type=Path, default=None, help="Video file/dir used by postprocess overlays")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--detector", choices=DETECTOR_CHOICES, default="dinov3")
    parser.add_argument("--classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--score-thresh", type=float, default=None)

    parser.add_argument(
        "--post-overlay",
        "--overlay-mode",
        dest="post_overlay",
        choices=("none", "detailed", "simple", "both"),
        default="detailed",
        help="Postprocess overlay output mode. --overlay-mode is kept as a compatibility alias.",
    )
    parser.add_argument(
        "--pre-overlay",
        "--raw-overlay",
        dest="pre_overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the pre-postprocess raw detector overlay. --raw-overlay is a compatibility alias.",
    )
    parser.add_argument("--overlay-encoder", choices=("nvenc", "cpu"), default="nvenc")
    parser.add_argument(
        "--pre-sqlite",
        "--raw-sqlite",
        dest="pre_sqlite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write pre-postprocess raw detector SQLite. --raw-sqlite is a compatibility alias.",
    )
    parser.add_argument("--post-sqlite", action=argparse.BooleanOptionalAction, default=True)

    add_policy_args(parser)
    parser.add_argument("--raw-cut-detect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--short-track-max-frames", type=int, default=10)
    parser.add_argument("--raw-det-score-min", type=float, default=0.35)
    parser.add_argument("--k2-device", default="auto")
    parser.add_argument("--polygon-predictor-device", default="auto")
    parser.add_argument("--progress-interval-sec", type=float, default=5.0)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--atosyori-repo", type=Path, default=Path(os.environ.get("ATOSYORI_REPO", str(LOCAL_ATOSYORI_REPO))))
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output_root = args.output_root.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.atosyori_repo = args.atosyori_repo.expanduser().resolve()
    args.keyframe_interval = validate_interval(args.keyframe_interval)
    args.recall_target = validate_recall(args.recall_target)
    if args.mode in {"full", "inference"} and args.input is None:
        raise RuntimeError("--input is required for full/inference modes")
    if args.mode == "postprocess" and not args.input_jsonl and not args.input_sqlite:
        raise RuntimeError("--input-jsonl or --input-sqlite is required for postprocess mode")
    if args.input_jsonl and args.input_sqlite:
        raise RuntimeError("--input-jsonl and --input-sqlite cannot be mixed in one command")
    return args


def add_detector_options(command: list[str], args: argparse.Namespace) -> None:
    if args.detector == "eva02":
        if args.batch_size is not None:
            command.extend(["--eva02-batch-size", str(args.batch_size)])
        if args.warmup_frames is not None:
            command.extend(["--eva02-warmup-frames", str(args.warmup_frames)])
        if args.score_thresh is not None:
            command.extend(["--eva02-score-thresh", str(args.score_thresh)])
    elif args.detector == "codino":
        if args.batch_size is not None:
            command.extend(["--codino-batch-size", str(args.batch_size)])
        if args.warmup_frames is not None:
            command.extend(["--codino-warmup-frames", str(args.warmup_frames)])
        if args.score_thresh is not None:
            command.extend(["--codino-score-thresh", str(args.score_thresh)])
    else:
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.warmup_frames is not None:
            command.extend(["--warmup-frames", str(args.warmup_frames)])
        if args.score_thresh is not None:
            command.extend(["--score-thresh", str(args.score_thresh)])


def write_policy_for_run(args: argparse.Namespace, run_dir: Path) -> Path:
    from .flow_cli_common import resolve_policy_path

    return resolve_policy_path(args, run_dir / "config")


def build_pipeline_command(args: argparse.Namespace, *, video: Path, output_root: Path, run_name: str, policy_path: Path | None) -> list[str]:
    postprocess = args.mode == "full"
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--input",
        str(video),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--detector",
        str(args.detector),
        "--classifier" if args.classifier else "--no-classifier",
        "--postprocess" if postprocess else "--no-postprocess",
        "--raw-sqlite" if pre_sqlite_enabled(args) else "--no-raw-sqlite",
        "--progress-interval-sec",
        str(args.progress_interval_sec),
    ]
    if args.force:
        command.append("--force")
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    add_detector_options(command, args)

    if postprocess:
        command.extend(
            [
                "--intervals",
                str(args.keyframe_interval),
                "--default-shape-mode",
                str(args.shape_mode),
                "--class-policy-json",
                str(policy_path),
                "--no-render-overlays",
                "--overlay-encoder",
                str(args.overlay_encoder),
                "--raw-cut-detect" if args.raw_cut_detect else "--no-raw-cut-detect",
                "--raw-remove-short-tracks-max-frames",
                str(args.short_track_max_frames),
                "--k2-device",
                str(args.k2_device),
                "--polygon-predictor-device",
                str(args.polygon_predictor_device),
                "--postprocess-extra-args",
                "--embed-original-masks" if post_overlay_mode(args) in {"detailed", "both"} else "--no-embed-original-masks",
                "--raw-det-score-min",
                str(args.raw_det_score_min),
                "--dense-recall-target",
                str(args.recall_target),
                "--polygon-recall-min",
                str(args.recall_target),
            ]
        )
    return command


def build_ui_job_command(args: argparse.Namespace, *, video: Path, output_root: Path, run_name: str, policy_path: Path | None) -> list[str]:
    pipeline_command = build_pipeline_command(args, video=video, output_root=output_root, run_name=run_name, policy_path=policy_path)
    overlay_mode = post_overlay_mode(args) if args.mode == "full" else "none"
    command = [
        sys.executable,
        str(UI_JOB_SCRIPT),
        "--input",
        str(video),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--overlay-mode",
        overlay_mode,
        "--raw-overlay" if pre_overlay_enabled(args) else "--no-raw-overlay",
        "--encoder",
        str(args.overlay_encoder),
        "--raw-sqlite-output" if pre_sqlite_enabled(args) else "--no-raw-sqlite-output",
        "--post-sqlite-output" if post_sqlite_enabled(args) else "--no-post-sqlite-output",
    ]
    if args.force:
        command.append("--force")
    command.extend(["--", *pipeline_command])
    return command


def run_full_or_inference(args: argparse.Namespace) -> int:
    assert args.input is not None
    videos = collect_videos(args.input, args.recursive)
    multiple = len(videos) > 1
    base_name = clean_run_part(args.run_name or f"{args.mode}_{timestamp()}")
    batch_root = args.output_root / base_name if multiple else args.output_root
    batch_summary_path = (batch_root / "batch_summary.json") if multiple else (args.output_root / base_name / "infer_summary.json")
    items: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"[batch-start] mode={args.mode} videos={len(videos)} output_root={batch_root if multiple else args.output_root}", flush=True)

    for index, video in enumerate(videos, start=1):
        run_name = f"{index:02d}_{clean_run_part(video.stem)}" if multiple else base_name
        output_root = batch_root if multiple else args.output_root
        run_dir = output_root / run_name
        policy_path = write_policy_for_run(args, run_dir) if args.mode == "full" else None
        command = build_ui_job_command(args, video=video, output_root=output_root, run_name=run_name, policy_path=policy_path)
        percent_before = (index - 1) / len(videos) * 100.0
        print(
            f"[video-start] {index}/{len(videos)} ({percent_before:.1f}%) video={video.name} run={run_name}",
            flush=True,
        )
        audit_path = run_dir / "logs" / "infer_audit.jsonl"
        write_audit(
            audit_path,
            "video_start",
            item_index=index,
            item_total=len(videos),
            video=str(video),
            run_dir=str(run_dir),
            overall_percent=percent_before,
        )
        returncode, elapsed, fps = run_streamed(
            command,
            log_path=run_dir / "logs" / "infer_cli.log",
            audit_path=audit_path,
            dry_run=args.dry_run,
            item_index=index,
            item_total=len(videos),
            phase=args.mode,
        )
        percent_after = index / len(videos) * 100.0
        item = {
            "video": str(video),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "fps": fps,
            "status": "completed" if returncode == 0 else "failed",
        }
        write_audit(
            audit_path,
            "video_done",
            item_index=index,
            item_total=len(videos),
            returncode=returncode,
            elapsed_seconds=elapsed,
            fps=fps,
            overall_percent=percent_after,
        )
        items.append(item)
        print(
            f"[video-done] {index}/{len(videos)} ({percent_after:.1f}%) status={item['status']} "
            f"elapsed={elapsed:.2f}s run_dir={run_dir}",
            flush=True,
        )
        if returncode != 0 and not args.continue_on_error:
            break

    total_elapsed = time.perf_counter() - started
    failed = [item for item in items if item["returncode"] != 0]
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "detector": args.detector,
        "output_root": str(batch_root if multiple else args.output_root),
        "video_count": len(videos),
        "completed_count": len([item for item in items if item["returncode"] == 0]),
        "failed_count": len(failed),
        "elapsed_seconds": total_elapsed,
        "items": items,
    }
    write_json(batch_summary_path, summary)
    print(f"[batch-complete] failed={len(failed)} elapsed={total_elapsed:.2f}s summary={batch_summary_path}", flush=True)
    return 1 if failed else 0


def build_postprocess_command(
    args: argparse.Namespace,
    *,
    artifact: Path,
    input_kind: str,
    input_video: Path | None,
    output_root: Path,
    run_name: str,
) -> list[str]:
    command = [
        sys.executable,
        str(POSTPROCESS_SCRIPT),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        "--model-root",
        str(args.model_root),
        "--atosyori-repo",
        str(args.atosyori_repo),
        "--overlay" if post_overlay_mode(args) != "none" else "--no-overlay",
        "--overlay-encoder",
        str(args.overlay_encoder),
        "--shape-mode",
        str(args.shape_mode),
        "--keyframe-interval",
        str(args.keyframe_interval),
        "--recall-target",
        str(args.recall_target),
        "--raw-cut-detect" if args.raw_cut_detect else "--no-raw-cut-detect",
        "--short-track-max-frames",
        str(args.short_track_max_frames),
        "--raw-det-score-min",
        str(args.raw_det_score_min),
        "--k2-device",
        str(args.k2_device),
        "--polygon-predictor-device",
        str(args.polygon_predictor_device),
        "--progress-interval-sec",
        str(args.progress_interval_sec),
    ]
    if input_kind == "jsonl":
        command.extend(["--input-jsonl", str(artifact)])
    else:
        command.extend(["--input-sqlite", str(artifact)])
    if input_video is not None:
        command.extend(["--input-video", str(input_video)])
    if args.force:
        command.append("--force")
    if args.class_policy_json is not None:
        command.extend(["--class-policy-json", str(args.class_policy_json)])
    for override in args.class_policy or []:
        command.extend(["--class-policy", str(override)])
    return command


def run_postprocess(args: argparse.Namespace) -> int:
    input_kind = "jsonl" if args.input_jsonl else "sqlite"
    suffixes = (".jsonl",) if input_kind == "jsonl" else (".sqlite", ".db")
    artifacts = expand_existing_inputs(args.input_jsonl or args.input_sqlite, suffixes)
    multiple = len(artifacts) > 1
    base_name = clean_run_part(args.run_name or f"postprocess_{timestamp()}")
    batch_root = args.output_root / base_name if multiple else args.output_root
    batch_summary_path = (batch_root / "batch_summary.json") if multiple else (args.output_root / base_name / "postprocess_summary.json")
    items: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"[batch-start] mode=postprocess inputs={len(artifacts)} output_root={batch_root if multiple else args.output_root}", flush=True)

    for index, artifact in enumerate(artifacts, start=1):
        run_name = f"{index:02d}_{clean_run_part(artifact.stem)}" if multiple else base_name
        output_root = batch_root if multiple else args.output_root
        run_dir = output_root / run_name
        video = find_video_for_artifact(artifact, args.input_video, len(artifacts))
        command = build_postprocess_command(
            args,
            artifact=artifact,
            input_kind=input_kind,
            input_video=video,
            output_root=output_root,
            run_name=run_name,
        )
        print(f"[postprocess-start] {index}/{len(artifacts)} input={artifact.name} run={run_name}", flush=True)
        audit_path = run_dir / "logs" / "infer_audit.jsonl"
        write_audit(
            audit_path,
            "postprocess_start",
            item_index=index,
            item_total=len(artifacts),
            input=str(artifact),
            input_video=None if video is None else str(video),
            run_dir=str(run_dir),
        )
        returncode, elapsed, fps = run_streamed(
            command,
            log_path=run_dir / "logs" / "infer_cli.log",
            audit_path=audit_path,
            dry_run=args.dry_run,
            item_index=index,
            item_total=len(artifacts),
            phase="postprocess",
        )
        item = {
            "input": str(artifact),
            "input_kind": input_kind,
            "input_video": None if video is None else str(video),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "fps": fps,
            "status": "completed" if returncode == 0 else "failed",
        }
        write_audit(
            audit_path,
            "postprocess_done",
            item_index=index,
            item_total=len(artifacts),
            returncode=returncode,
            elapsed_seconds=elapsed,
            fps=fps,
        )
        items.append(item)
        print(f"[postprocess-done] {index}/{len(artifacts)} status={item['status']} elapsed={elapsed:.2f}s", flush=True)
        if returncode != 0 and not args.continue_on_error:
            break

    total_elapsed = time.perf_counter() - started
    failed = [item for item in items if item["returncode"] != 0]
    summary = {
        "schema_version": 1,
        "mode": "postprocess",
        "input_kind": input_kind,
        "output_root": str(batch_root if multiple else args.output_root),
        "input_count": len(artifacts),
        "completed_count": len([item for item in items if item["returncode"] == 0]),
        "failed_count": len(failed),
        "elapsed_seconds": total_elapsed,
        "items": items,
    }
    write_json(batch_summary_path, summary)
    print(f"[batch-complete] failed={len(failed)} elapsed={total_elapsed:.2f}s summary={batch_summary_path}", flush=True)
    return 1 if failed else 0


def main() -> int:
    args = normalize_args(build_parser().parse_args())
    if args.mode == "postprocess":
        return run_postprocess(args)
    return run_full_or_inference(args)


def cli_main() -> int:
    try:
        return main()
    except SystemExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        print("[cancelled] interrupted by user", file=sys.stderr, flush=True)
        return 130
    except BaseException as exc:
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
