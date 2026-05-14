#!/usr/bin/env python3
"""Summary and output-link helpers for integrated pipeline runs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from backend.schemas.detection_jsonl import summarize_detection_jsonl


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


def summarize_detector(detector_out: Path, video: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = detector_out / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    jsonl_path = detector_out / "jsonl" / f"{video.stem}.jsonl"
    runs = list(summary.get("runs", []))
    if runs:
        run_jsonl = Path(str(runs[0].get("output_jsonl", jsonl_path)))
        if run_jsonl.is_file():
            jsonl_path = run_jsonl
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    summary["jsonl_contract"] = summarize_detection_jsonl(jsonl_path).as_dict()
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
