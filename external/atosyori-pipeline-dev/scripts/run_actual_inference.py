#!/usr/bin/env python3
"""Run the actual Atosyori inference pipeline with in-code settings.

Edit the constants in the "User settings" block below instead of passing
command-line arguments. The default policy is:

- label == "男性器": polygon keyframes, target interval 3 frames
- all other labels: ellipse keyframes, target interval 6 frames
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# =============================================================================
# User settings
# =============================================================================

INPUT_JSONL = PROJECT_ROOT / "input" / "アクセル様２月解析用白カン01.26.jsonl"
INPUT_VIDEO = PROJECT_ROOT / "input" / "アクセル様２月解析用白カン01.26.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output" / "actual_inference"
MODEL_ROOT = PROJECT_ROOT / "models"

# Set this to an existing tracked SQLite to skip JSONL/video preprocessing.
# Leave as None to preprocess INPUT_JSONL + INPUT_VIDEO.
TRACKED_SQLITE_OVERRIDE: Path | None = None

FORCE = False
RUN_EXACT_EVALUATION = True

POLYGON_LABEL = "男性器"

POLYGON_SETTINGS = {
    "target_ratio": 1.0 / 3.0,
    "max_gap": 3,
    "anchors_per_contour": 48,
    "num_workers": 8,
    "predictor_device": "cuda",
}

ELLIPSE_SETTINGS = {
    "target_ratio": 1.0 / 6.0,
    "max_gap": 6,
    "k2_device": "cuda",
    "dense_recall_target": 0.98,
}


# =============================================================================
# Implementation
# =============================================================================

from atosyori_postprocess.engine.standalone_runtime_fst import fst


LEGACY_ENGINE = PROJECT_ROOT / "src" / "atosyori_postprocess" / "legacy" / "run_standalone.py"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(name: str, output_dir: Path, required_outputs: list[Path], runner) -> dict[str, Any]:
    if FORCE and output_dir.exists():
        shutil.rmtree(output_dir)

    if not FORCE and all(path.exists() for path in required_outputs):
        print(f"[skip] {name}")
        return {"name": name, "skipped": True, "wall_seconds": 0.0}

    print(f"[run] {name}")
    start = time.perf_counter()
    code = runner()
    elapsed = time.perf_counter() - start
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}")
    print(f"[done] {name}: {elapsed:.2f}s")
    return {"name": name, "skipped": False, "wall_seconds": elapsed}


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) if not current else str(SRC_ROOT) + os.pathsep + current
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_command(command: list[str]) -> int:
    print("[cmd] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=command_env(), check=False).returncode


def package_stage_command(stage_name: str, args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "atosyori_postprocess",
        "stage",
        "--model-root",
        str(MODEL_ROOT),
        stage_name,
        "--",
        *args,
    ]


def legacy_stage_command(stage_name: str, args: list[str]) -> list[str]:
    return [sys.executable, str(LEGACY_ENGINE), stage_name, *args]


def table_count(sqlite_path: Path, table: str) -> int:
    with sqlite3.connect(sqlite_path) as conn:
        return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def split_sqlite_by_mask_label(source_sqlite: Path, output_sqlite: Path, where_sql: str) -> dict[str, Any]:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if FORCE and output_sqlite.exists():
        output_sqlite.unlink()
    if output_sqlite.exists():
        return inspect_sqlite(output_sqlite)

    def quote(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    src = sqlite3.connect(source_sqlite)
    dst = sqlite3.connect(output_sqlite)
    try:
        src_cur = src.cursor()
        dst_cur = dst.cursor()
        table_names = [str(row[0]) for row in src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if "masks" not in table_names:
            raise RuntimeError(f"input sqlite does not contain masks table: {source_sqlite}")

        mask_columns = [str(row[1]) for row in src_cur.execute("PRAGMA table_info(masks)")]
        dst_cur.execute("CREATE TABLE masks({})".format(", ".join(quote(col) for col in mask_columns)))
        mask_select = ", ".join(quote(col) for col in mask_columns)
        mask_placeholders = ", ".join("?" for _ in mask_columns)
        mask_rows = src_cur.execute(
            f"SELECT {mask_select} FROM masks WHERE {where_sql} ORDER BY frame, CAST(track_id AS INTEGER)"
        ).fetchall()
        if mask_rows:
            dst_cur.executemany(f"INSERT INTO masks({mask_select}) VALUES ({mask_placeholders})", mask_rows)

        keep_tracks = sorted(
            {str(row[mask_columns.index("track_id")]) for row in mask_rows},
            key=lambda value: int(value),
        )

        if "tracks" in table_names:
            track_columns = [str(row[1]) for row in src_cur.execute("PRAGMA table_info(tracks)")]
            dst_cur.execute("CREATE TABLE tracks({})".format(", ".join(quote(col) for col in track_columns)))
            if keep_tracks:
                track_select = ", ".join(quote(col) for col in track_columns)
                track_placeholders = ", ".join("?" for _ in track_columns)
                in_placeholders = ", ".join("?" for _ in keep_tracks)
                track_rows = src_cur.execute(
                    f"SELECT {track_select} FROM tracks WHERE track_id IN ({in_placeholders})",
                    keep_tracks,
                ).fetchall()
                if track_rows:
                    dst_cur.executemany(
                        f"INSERT INTO tracks({track_select}) VALUES ({track_placeholders})",
                        track_rows,
                    )

        if "cuts" in table_names:
            cut_columns = [str(row[1]) for row in src_cur.execute("PRAGMA table_info(cuts)")]
            dst_cur.execute("CREATE TABLE cuts({})".format(", ".join(quote(col) for col in cut_columns)))
            cut_select = ", ".join(quote(col) for col in cut_columns)
            cut_rows = src_cur.execute(f"SELECT {cut_select} FROM cuts").fetchall()
            if cut_rows:
                dst_cur.executemany(
                    f"INSERT INTO cuts({cut_select}) VALUES ({', '.join('?' for _ in cut_columns)})",
                    cut_rows,
                )

        dst.commit()
    finally:
        dst.close()
        src.close()

    return inspect_sqlite(output_sqlite)


def inspect_sqlite(sqlite_path: Path) -> dict[str, Any]:
    with sqlite3.connect(sqlite_path) as conn:
        labels = dict(
            conn.execute(
                "SELECT COALESCE(label, ''), count(*) FROM masks GROUP BY COALESCE(label, '')"
            ).fetchall()
        )
        rows, tracks, frames, frame_min, frame_max = conn.execute(
            "SELECT count(*), count(distinct track_id), count(distinct frame), min(frame), max(frame) FROM masks"
        ).fetchone()
    return {
        "sqlite": str(sqlite_path),
        "rows": int(rows),
        "tracks": int(tracks),
        "distinct_frames": int(frames),
        "frame_min": None if frame_min is None else int(frame_min),
        "frame_max": None if frame_max is None else int(frame_max),
        "label_counts": {str(key): int(value) for key, value in labels.items()},
    }


def load_prediction_rows(sqlite_path: Path) -> list[tuple[int, str, str]]:
    with sqlite3.connect(sqlite_path) as conn:
        return [
            (int(frame), str(track_id), str(polygons_json))
            for frame, track_id, polygons_json in conn.execute(
                "SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)"
            )
        ]


def load_source_labels(sqlite_path: Path) -> dict[tuple[int, str], str]:
    with sqlite3.connect(sqlite_path) as conn:
        return {
            (int(frame), str(track_id)): str(label or "")
            for frame, track_id, label in conn.execute("SELECT frame, track_id, label FROM masks")
        }


def merge_prediction_sqlites(
    *,
    polygon_sqlite: Path,
    ellipse_sqlite: Path,
    output_sqlite: Path,
    reference_sqlite: Path,
) -> Path:
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if FORCE and output_sqlite.exists():
        output_sqlite.unlink()
    if output_sqlite.exists():
        return output_sqlite

    source_labels = load_source_labels(reference_sqlite)
    rows_by_key: dict[tuple[int, str], tuple[int, str, str]] = {}
    branch_by_key: dict[tuple[int, str], str] = {}

    def source_label_accepts(branch: str, key: tuple[int, str]) -> bool:
        label = source_labels.get(key)
        if label is None:
            return True
        if branch == "polygon":
            return label == POLYGON_LABEL
        return label != POLYGON_LABEL

    def branch_priority(branch: str, key: tuple[int, str]) -> int:
        label = source_labels.get(key)
        if label == POLYGON_LABEL:
            return 3 if branch == "polygon" else 0
        if label is not None:
            return 3 if branch == "ellipse" else 0
        return 2 if branch == "polygon" else 1

    for branch, sqlite_path in (("polygon", polygon_sqlite), ("ellipse", ellipse_sqlite)):
        for frame, track_id, polygons_json in load_prediction_rows(sqlite_path):
            key = (int(frame), str(track_id))
            if not source_label_accepts(branch, key):
                continue
            previous_branch = branch_by_key.get(key)
            if previous_branch is None or branch_priority(branch, key) > branch_priority(previous_branch, key):
                rows_by_key[key] = (int(frame), str(track_id), str(polygons_json))
                branch_by_key[key] = branch

    merged_rows = list(rows_by_key.values())
    merged_rows.sort(key=lambda row: (row[0], int(row[1])))
    fst.write_sqlite(merged_rows, output_sqlite, reference_sqlite=reference_sqlite)
    return output_sqlite


def evaluate_prediction_sqlite(tracked_sqlite: Path, pred_sqlite: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if not FORCE and summary_path.exists():
        return read_json(summary_path)

    gt_rows = load_prediction_rows(tracked_sqlite)
    pred_rows = load_prediction_rows(pred_sqlite)
    pred_lookup = {(frame, track_id): polygons_json for frame, track_id, polygons_json in pred_rows}

    metric_rows: list[dict[str, Any]] = []
    total = {
        "row_count": 0.0,
        "gt_area": 0.0,
        "pred_area": 0.0,
        "intersection": 0.0,
        "union": 0.0,
        "weighted_error_total": 0.0,
    }
    missing_rows = 0

    for frame, track_id, gt_json in gt_rows:
        pred_json = pred_lookup.get((frame, track_id))
        if pred_json is None:
            pred_json = "[]"
            missing_rows += 1
        metrics = fst.compute_exact_metrics_from_polygons(
            fst.parse_polygons(gt_json),
            fst.parse_polygons(pred_json),
        )
        weighted_error = float(metrics.get("weighted_error", 0.0))
        total["row_count"] += 1.0
        total["gt_area"] += float(metrics["gt_area"])
        total["pred_area"] += float(metrics["pred_area"])
        total["intersection"] += float(metrics["intersection"])
        total["union"] += float(metrics["union"])
        total["weighted_error_total"] += weighted_error
        metric_rows.append(
            {
                "frame": int(frame),
                "track_id": str(track_id),
                "gt_area": float(metrics["gt_area"]),
                "pred_area": float(metrics["pred_area"]),
                "intersection": float(metrics["intersection"]),
                "union": float(metrics["union"]),
                "recall": float(metrics["recall"]),
                "precision": float(metrics["precision"]),
                "iou": float(metrics["iou"]),
                "weighted_error": weighted_error,
            }
        )

    metrics_csv = output_dir / "keyframe_exact_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "frame",
            "track_id",
            "gt_area",
            "pred_area",
            "intersection",
            "union",
            "recall",
            "precision",
            "iou",
            "weighted_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(metric_rows, key=lambda row: (int(row["frame"]), int(str(row["track_id"])))))

    aggregate = {
        **total,
        "global_recall": total["intersection"] / total["gt_area"] if total["gt_area"] else 1.0,
        "global_precision": total["intersection"] / total["pred_area"] if total["pred_area"] else 1.0,
        "global_iou": total["intersection"] / total["union"] if total["union"] else 1.0,
        "weighted_error_mean": total["weighted_error_total"] / total["row_count"] if total["row_count"] else 0.0,
        "missing_rows": int(missing_rows),
        "prediction_rows": len(pred_rows),
    }
    summary = {
        "tracked_sqlite": str(tracked_sqlite),
        "pred_sqlite": str(pred_sqlite),
        "metrics_csv": str(metrics_csv),
        "metrics": aggregate,
    }
    write_json(summary_path, summary)
    return summary


def ensure_preprocess(timings: list[dict[str, Any]]) -> Path:
    if TRACKED_SQLITE_OVERRIDE is not None:
        require_file(TRACKED_SQLITE_OVERRIDE)
        return TRACKED_SQLITE_OVERRIDE

    require_file(INPUT_JSONL)
    require_file(INPUT_VIDEO)
    preprocess_dir = OUTPUT_DIR / "preprocess"
    expected_sqlite = preprocess_dir / "preprocess" / f"{INPUT_JSONL.stem}.tracked.sqlite"
    summary_path = preprocess_dir / "summary.json"

    timings.append(
        run_step(
            "preprocess",
            preprocess_dir,
            [expected_sqlite, summary_path],
            lambda: run_command(
                package_stage_command(
                    "preprocess",
                    [
                        "--input-jsonl",
                        str(INPUT_JSONL),
                        "--input-video",
                        str(INPUT_VIDEO),
                        "--output-dir",
                        str(preprocess_dir),
                    ],
                )
            ),
        )
    )
    if summary_path.exists():
        return Path(read_json(summary_path)["tracked_sqlite"])
    return expected_sqlite


def run_polygon_branch(input_sqlite: Path, timings: list[dict[str, Any]]) -> Path:
    output_dir = OUTPUT_DIR / "branches" / "male_polygon_3f"
    pred_sqlite = output_dir / "pred" / "predictions.sqlite"
    timings.append(
        run_step(
            "male polygon keyframes",
            output_dir,
            [pred_sqlite, output_dir / "summary.json"],
            lambda: run_command(
                package_stage_command(
                    "polygon-keyframes",
                    [
                        "--input-sqlite",
                        str(input_sqlite),
                        "--output-dir",
                        str(output_dir),
                        "--target-ratio",
                        str(POLYGON_SETTINGS["target_ratio"]),
                        "--max-gap",
                        str(POLYGON_SETTINGS["max_gap"]),
                        "--anchors-per-contour",
                        str(POLYGON_SETTINGS["anchors_per_contour"]),
                        "--num-workers",
                        str(POLYGON_SETTINGS["num_workers"]),
                        "--evaluate-exact",
                        "--write-pred-sqlite",
                        "--predictor-device",
                        str(POLYGON_SETTINGS["predictor_device"]),
                    ],
                )
            ),
        )
    )
    return pred_sqlite


def run_ellipse_branch(input_sqlite: Path, timings: list[dict[str, Any]]) -> Path:
    branch_dir = OUTPUT_DIR / "branches" / "other_ellipse_6f"
    inference_dir = branch_dir / "inference"
    keyframes_dir = branch_dir / "keyframes"
    pred_sqlite = branch_dir / "pred" / "predictions.sqlite"

    timings.append(
        run_step(
            "other ellipse inference",
            inference_dir,
            [inference_dir / "k1_exact_k2_v5_metrics.csv", inference_dir / "summary.json"],
            lambda: run_command(
                package_stage_command(
                    "ellipse-inference",
                    [
                        "--input-sqlite",
                        str(input_sqlite),
                        "--output-dir",
                        str(inference_dir),
                        "--k2-device",
                        str(ELLIPSE_SETTINGS["k2_device"]),
                    ],
                )
            ),
        )
    )

    metrics_csv = inference_dir / "k1_exact_k2_v5_metrics.csv"
    timings.append(
        run_step(
            "other ellipse keyframes",
            keyframes_dir,
            [keyframes_dir / "interpolated_union.json", keyframes_dir / "summary.json"],
            lambda: run_command(
                package_stage_command(
                    "ellipse-keyframes",
                    [
                        "--input-metrics-csv",
                        str(metrics_csv),
                        "--output-dir",
                        str(keyframes_dir),
                        "--target-ratio",
                        str(ELLIPSE_SETTINGS["target_ratio"]),
                        "--max-gap",
                        str(ELLIPSE_SETTINGS["max_gap"]),
                        "--solver",
                        "dp",
                        "--keyframe-value-source",
                        "confidence_blend",
                        "--smooth-alpha",
                        "1.0",
                        "--value-refine",
                        "global_ls",
                        "--dense-recall-target",
                        str(ELLIPSE_SETTINGS["dense_recall_target"]),
                    ],
                )
            ),
        )
    )

    timings.append(
        run_step(
            "other ellipse union to sqlite",
            pred_sqlite.parent,
            [pred_sqlite],
            lambda: run_command(
                legacy_stage_command(
                    "__onefile_union_to_sqlite",
                    [
                        "--input-union-json",
                        str(keyframes_dir / "interpolated_union.json"),
                        "--output-sqlite",
                        str(pred_sqlite),
                        "--reference-sqlite",
                        str(input_sqlite),
                    ],
                )
            ),
        )
    )
    return pred_sqlite


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, Any]] = []

    tracked_sqlite = ensure_preprocess(timings)
    require_file(tracked_sqlite)

    split_dir = OUTPUT_DIR / "splits"
    male_split = split_dir / "male_polygon.sqlite"
    other_split = split_dir / "other_ellipse.sqlite"
    split_summary = {
        "male_polygon": split_sqlite_by_mask_label(
            tracked_sqlite,
            male_split,
            f"COALESCE(label, '') = '{POLYGON_LABEL}'",
        ),
        "other_ellipse": split_sqlite_by_mask_label(
            tracked_sqlite,
            other_split,
            f"COALESCE(label, '') <> '{POLYGON_LABEL}'",
        ),
    }
    write_json(split_dir / "summary.json", split_summary)

    male_pred_sqlite = run_polygon_branch(male_split, timings)
    other_pred_sqlite = run_ellipse_branch(other_split, timings)

    final_pred_sqlite = OUTPUT_DIR / "final" / "predictions.sqlite"
    merge_start = time.perf_counter()
    merge_prediction_sqlites(
        polygon_sqlite=male_pred_sqlite,
        ellipse_sqlite=other_pred_sqlite,
        output_sqlite=final_pred_sqlite,
        reference_sqlite=tracked_sqlite,
    )
    timings.append(
        {
            "name": "merge predictions",
            "skipped": False,
            "wall_seconds": time.perf_counter() - merge_start,
        }
    )

    final_eval_summary: dict[str, Any] | None = None
    if RUN_EXACT_EVALUATION:
        eval_start = time.perf_counter()
        final_eval_summary = evaluate_prediction_sqlite(
            tracked_sqlite,
            final_pred_sqlite,
            OUTPUT_DIR / "final" / "exact",
        )
        timings.append(
            {
                "name": "final exact evaluation",
                "skipped": False,
                "wall_seconds": time.perf_counter() - eval_start,
            }
        )

    summary = {
        "input": {
            "input_jsonl": str(INPUT_JSONL),
            "input_video": str(INPUT_VIDEO),
            "tracked_sqlite": str(tracked_sqlite),
        },
        "output_dir": str(OUTPUT_DIR),
        "model_root": str(MODEL_ROOT),
        "policy": {
            "polygon_label": POLYGON_LABEL,
            "polygon_settings": POLYGON_SETTINGS,
            "ellipse_settings": ELLIPSE_SETTINGS,
        },
        "splits": split_summary,
        "artifacts": {
            "male_polygon_pred_sqlite": str(male_pred_sqlite),
            "other_ellipse_pred_sqlite": str(other_pred_sqlite),
            "final_pred_sqlite": str(final_pred_sqlite),
            "final_exact_summary": None
            if final_eval_summary is None
            else str(OUTPUT_DIR / "final" / "exact" / "summary.json"),
        },
        "counts": {
            "tracked_rows": table_count(tracked_sqlite, "masks"),
            "final_prediction_rows": table_count(final_pred_sqlite, "masks"),
        },
        "final_exact": None if final_eval_summary is None else final_eval_summary["metrics"],
        "timings": timings,
        "total_wall_seconds": sum(float(row["wall_seconds"]) for row in timings),
    }
    write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
