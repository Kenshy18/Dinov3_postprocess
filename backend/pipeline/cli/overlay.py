#!/usr/bin/env python3
"""Primary standalone overlay CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .flow_cli_common import ROOT


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render raw or postprocessed mask overlays from video + mask artifacts."
    )
    parser.add_argument("--video", type=Path, default=None, help="Source video for direct artifact mode")
    parser.add_argument("--raw-jsonl", type=Path, default=None, help="Raw detector JSONL")
    parser.add_argument("--tracked-sqlite", type=Path, default=None, help="Raw/tracked SQLite used by detailed overlay")
    parser.add_argument("--pred-sqlite", type=Path, default=None, help="Final prediction SQLite")
    parser.add_argument("--head-face-sqlite", type=Path, default=None, help="Optional Head/Face SQLite used by simple/detailed overlays")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Completed run directory")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Directory containing run directories")
    parser.add_argument("--output", type=Path, default=None, help="Output video for direct single overlay")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("auto", "raw", "simple", "detailed"), default="auto")
    parser.add_argument("--encoder", choices=("nvenc", "cpu"), default="nvenc")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def direct_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_jsonl = resolve_path(args.raw_jsonl)
    pred_sqlite = resolve_path(args.pred_sqlite)
    tracked_sqlite = resolve_path(args.tracked_sqlite)
    head_face_sqlite = resolve_path(getattr(args, "head_face_sqlite", None))
    video = resolve_path(args.video)
    if raw_jsonl is None and pred_sqlite is None:
        return []
    if video is None:
        raise RuntimeError("--video is required when --raw-jsonl or --pred-sqlite is used")
    if not video.is_file():
        raise FileNotFoundError(video)
    if raw_jsonl is not None and pred_sqlite is not None:
        raise RuntimeError("--raw-jsonl and --pred-sqlite cannot be rendered in one direct output")

    output_dir = resolve_path(args.output_dir) or ROOT / "output" / "overlays"
    if raw_jsonl is not None:
        if not raw_jsonl.is_file():
            raise FileNotFoundError(raw_jsonl)
        output = resolve_path(args.output) or output_dir / f"{video.stem}_raw_overlay.mp4"
        return [{"kind": "raw", "video": video, "jsonl": raw_jsonl, "output": output}]

    assert pred_sqlite is not None
    if not pred_sqlite.is_file():
        raise FileNotFoundError(pred_sqlite)
    if tracked_sqlite is not None and not tracked_sqlite.is_file():
        raise FileNotFoundError(tracked_sqlite)
    if head_face_sqlite is not None and not head_face_sqlite.is_file():
        raise FileNotFoundError(head_face_sqlite)
    mode = "simple" if args.mode == "auto" else args.mode
    if mode == "raw":
        raise RuntimeError("--mode raw requires --raw-jsonl")
    output = resolve_path(args.output) or output_dir / f"{video.stem}_{mode}_overlay.mp4"
    return [
        {
            "kind": mode,
            "video": video,
            "tracked_sqlite": tracked_sqlite,
            "pred_sqlite": pred_sqlite,
            "head_face_sqlite": head_face_sqlite,
            "output": output,
        }
    ]


def run_dirs_from_batch(batch_dir: Path) -> list[Path]:
    batch_dir = batch_dir.expanduser().resolve()
    summary = batch_dir / "batch_summary.json"
    if summary.is_file():
        data = load_json(summary)
        dirs = [Path(str(item["run_dir"])).expanduser().resolve() for item in data.get("items", []) if item.get("run_dir")]
        return [path for path in dirs if path.is_dir()]
    return sorted(p.resolve() for p in batch_dir.iterdir() if p.is_dir() and ((p / "summary.json").is_file() or (p / "最終成果物.json").is_file()))


def specs_from_run_dir(run_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    specs: list[dict[str, Any]] = []
    final_summary = run_dir / "最終成果物.json"
    pipeline_summary = run_dir / "summary.json"
    output_dir = resolve_path(args.output_dir) or run_dir / "overlay_cli"
    mode = args.mode

    if final_summary.is_file():
        data = load_json(final_summary)
        processed_video = Path(str(data.get("processed_input") or ""))
        original_video = Path(str(data.get("original_input") or ""))
        video = processed_video if processed_video.is_file() else original_video
        tracked_raw = data.get("tracked_sqlite")
        tracked = Path(str(tracked_raw)) if tracked_raw else None
        head_face_raw = data.get("head_face_sqlite")
        head_face_sqlite = Path(str(head_face_raw)) if head_face_raw else None
        if mode in {"auto", "raw"}:
            jsonl_raw = data.get("detector_jsonl")
            if jsonl_raw:
                jsonl = Path(str(jsonl_raw))
                if video.is_file() and jsonl.is_file():
                    specs.append({"kind": "raw", "video": video, "jsonl": jsonl, "output": output_dir / f"{video.stem}_raw_overlay.mp4"})
        if mode in {"auto", "simple", "detailed"}:
            final_sqlite = data.get("final_sqlite") or {}
            if isinstance(final_sqlite, dict):
                modes = ("simple", "detailed") if mode == "auto" else (mode,)
                for label, sqlite_raw in sorted(final_sqlite.items()):
                    sqlite_path = Path(str(sqlite_raw))
                    if not video.is_file() or not sqlite_path.is_file():
                        continue
                    for overlay_mode in modes:
                        specs.append(
                            {
                                "kind": overlay_mode,
                                "video": video,
                                "tracked_sqlite": tracked if tracked and tracked.is_file() else None,
                                "pred_sqlite": sqlite_path,
                                "head_face_sqlite": head_face_sqlite if head_face_sqlite and head_face_sqlite.is_file() else None,
                                "output": output_dir / f"{video.stem}_{label}_{overlay_mode}.mp4",
                            }
                        )
        return specs

    if pipeline_summary.is_file():
        data = load_json(pipeline_summary)
        video = Path(str(data.get("video") or ""))
        artifacts = data.get("artifacts") or {}
        postprocess = data.get("postprocess") or {}
        if mode in {"auto", "raw"}:
            jsonl_raw = artifacts.get("detector_jsonl") or artifacts.get("dinov3_jsonl") or artifacts.get("codino_jsonl")
            if jsonl_raw:
                jsonl = Path(str(jsonl_raw))
                if video.is_file() and jsonl.is_file():
                    specs.append({"kind": "raw", "video": video, "jsonl": jsonl, "output": output_dir / f"{video.stem}_raw_overlay.mp4"})
        if mode in {"auto", "simple", "detailed"}:
            tracked_raw = postprocess.get("tracked_sqlite") or postprocess.get("tracked_sqlite_link")
            tracked = Path(str(tracked_raw)) if tracked_raw else None
            head_face_summary = data.get("head_face") or {}
            head_face_raw = artifacts.get("head_face_sqlite") or (
                head_face_summary.get("path") if isinstance(head_face_summary, dict) else None
            )
            head_face_sqlite = Path(str(head_face_raw)) if head_face_raw else None
            links = postprocess.get("prediction_sqlite_links") or {}
            modes = ("simple", "detailed") if mode == "auto" else (mode,)
            if isinstance(links, dict):
                for label, sqlite_raw in sorted(links.items()):
                    sqlite_path = Path(str(sqlite_raw))
                    if not video.is_file() or not sqlite_path.is_file():
                        continue
                    for overlay_mode in modes:
                        specs.append(
                            {
                                "kind": overlay_mode,
                                "video": video,
                                "tracked_sqlite": tracked if tracked and tracked.is_file() else None,
                                "pred_sqlite": sqlite_path,
                                "head_face_sqlite": head_face_sqlite if head_face_sqlite and head_face_sqlite.is_file() else None,
                                "output": output_dir / f"{video.stem}_{label}_{overlay_mode}.mp4",
                            }
                        )
        return specs

    raise FileNotFoundError(f"missing summary in run dir: {run_dir}")


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = direct_specs(args)
    run_dirs = [p.expanduser().resolve() for p in getattr(args, "run_dir", [])]
    batch_dir = getattr(args, "batch_dir", None)
    if batch_dir is not None:
        run_dirs.extend(run_dirs_from_batch(batch_dir))
    for run_dir in run_dirs:
        specs.extend(specs_from_run_dir(run_dir, args))
    if not specs:
        raise RuntimeError("no overlay work found; pass direct artifacts, --run-dir, or --batch-dir")
    return specs


def render_spec(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from apps.qt_ui.run_ui_job import render_raw_overlay, render_sqlite_overlay

    output = Path(spec["output"])
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists: {output} (use --force)")
    started = time.perf_counter()
    kind = str(spec["kind"])
    print(f"[overlay-start] kind={kind} video={Path(spec['video']).name} output={output}", flush=True)
    if kind == "raw":
        render_raw_overlay(
            Path(spec["video"]),
            Path(spec["jsonl"]),
            output,
            encoder=str(args.encoder),
            frame_limit=args.frame_limit,
        )
    else:
        render_sqlite_overlay(
            Path(spec["video"]),
            spec.get("tracked_sqlite"),
            Path(spec["pred_sqlite"]),
            output,
            mode=kind,
            encoder=str(args.encoder),
            frame_limit=args.frame_limit,
            head_face_sqlite=spec.get("head_face_sqlite"),
        )
    elapsed = time.perf_counter() - started
    print(f"[overlay-done] kind={kind} elapsed={elapsed:.2f}s output={output}", flush=True)
    return {"kind": kind, "output": str(output), "elapsed_seconds": elapsed, "status": "completed"}


def main() -> int:
    args = build_parser().parse_args()
    specs = build_specs(args)
    items: list[dict[str, Any]] = []
    started = time.perf_counter()
    print(f"[overlay-batch-start] count={len(specs)}", flush=True)
    for index, spec in enumerate(specs, start=1):
        print(f"[overlay-item] {index}/{len(specs)}", flush=True)
        try:
            items.append(render_spec(spec, args))
        except BaseException as exc:
            item = {"kind": spec.get("kind"), "output": str(spec.get("output")), "status": "failed", "error": repr(exc)}
            items.append(item)
            print(f"[overlay-error] {exc}", flush=True)
            if not args.continue_on_error:
                break
    failed = [item for item in items if item.get("status") != "completed"]
    elapsed = time.perf_counter() - started
    summary_dir = resolve_path(args.output_dir) or (Path(specs[0]["output"]).parent if specs else ROOT / "output" / "overlays")
    summary_path = summary_dir / f"overlay_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(summary_path, {"schema_version": 1, "count": len(specs), "failed_count": len(failed), "elapsed_seconds": elapsed, "items": items})
    print(f"[overlay-batch-complete] failed={len(failed)} elapsed={elapsed:.2f}s summary={summary_path}", flush=True)
    return 1 if failed else 0


def cli_main() -> int:
    try:
        return main()
    except SystemExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        print("[overlay-cancelled]", file=sys.stderr, flush=True)
        return 130
    except BaseException as exc:
        print(f"[overlay-fatal] {exc}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
