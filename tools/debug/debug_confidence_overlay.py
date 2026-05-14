#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ATOSYORI_SRC = ROOT / "external" / "atosyori-pipeline-dev" / "src"
if str(ATOSYORI_SRC) not in sys.path:
    sys.path.insert(0, str(ATOSYORI_SRC))

from atosyori_postprocess.engine import standalone_runtime_fst as _fst_runtime  # noqa: E402,F401
import final_standalone_t5000 as fst  # noqa: E402


def load_legacy_run_standalone():
    path = ATOSYORI_SRC / "atosyori_postprocess" / "legacy" / "run_standalone.py"
    spec = importlib.util.spec_from_file_location("atosyori_legacy_run_standalone_debug", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_legacy_run_standalone()


CLASS_ASCII = {
    "\u5973\u6027\u5668": "female",
    "\u7537\u6027\u5668": "male",
    "\u7d50\u5408\u90e8\u5206": "junction",
}

CLASS_COLORS = {
    "female": (80, 220, 80),
    "male": (255, 120, 70),
    "junction": (255, 80, 220),
}
FALLBACK_COLOR = (0, 215, 255)
GT_OUTLINE_COLOR = (255, 255, 255)
GT_MASK_COLOR = np.array([0, 0, 255], dtype=np.float32)
TEXT_BG = (18, 18, 18)
TEXT_FG = (245, 245, 245)


@dataclass
class ConfidenceRow:
    frame: int
    raw_track_id: str
    track_id: str
    label: str
    score: float
    class_score: float | None
    detector_score: float | None
    category_id: int | None
    category_index: int | None


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def ascii_label(label: str | None, category_id: int | None = None) -> str:
    if label and label in CLASS_ASCII:
        return CLASS_ASCII[label]
    if category_id == 1:
        return "female"
    if category_id == 2:
        return "male"
    if category_id == 3:
        return "junction"
    return str(label or "unknown")


def iter_raw_detections(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            detections = obj.get("detections") or obj.get("instances") or []
            if not isinstance(detections, list):
                continue
            for det in detections:
                if isinstance(det, dict):
                    yield det


def collect_confidence_distribution(jsonl_path: Path) -> dict[str, object]:
    scores: list[float] = []
    class_scores: list[float] = []
    labels: list[str] = []
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"score": [], "class_score": []})
    for det in iter_raw_detections(jsonl_path):
        score = to_float(det.get("score"))
        class_score = to_float(det.get("class_score"))
        category_id = to_int(det.get("category_id"))
        label = ascii_label(str(det.get("class_name", det.get("label", ""))), category_id)
        if score is not None:
            scores.append(score)
            by_label[label]["score"].append(score)
            labels.append(label)
        if class_score is not None:
            class_scores.append(class_score)
            by_label[label]["class_score"].append(class_score)
    return {"scores": scores, "class_scores": class_scores, "labels": labels, "by_label": by_label}


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    qs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    return {str(q): float(np.quantile(arr, q)) for q in qs}


def plot_confidence(args: argparse.Namespace) -> None:
    dist = collect_confidence_distribution(args.jsonl)
    scores = list(dist["scores"])
    class_scores = list(dist["class_scores"])
    by_label = dist["by_label"]
    output_png: Path = args.output_png
    output_png.parent.mkdir(parents=True, exist_ok=True)

    dino_thresh = float(args.dino_score_thresh)
    post_thresh = float(args.post_score_thresh)
    score_arr = np.asarray(scores, dtype=np.float64)
    below_post = int(np.count_nonzero(score_arr < post_thresh)) if len(score_arr) else 0
    label_counts = Counter(dist["labels"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    bins_score = np.linspace(max(0.0, min(dino_thresh, np.min(score_arr) if len(score_arr) else dino_thresh)), 1.0, 80)

    ax = axes[0, 0]
    ax.hist(scores, bins=bins_score, color="#476A9F", alpha=0.85)
    ax.axvline(dino_thresh, color="#222222", linestyle="--", linewidth=1.4, label=f"DINO output >= {dino_thresh:.2f}")
    ax.axvline(post_thresh, color="#D62728", linestyle="-", linewidth=1.6, label=f"post preprocess >= {post_thresh:.2f}")
    ax.set_title("Detector score distribution")
    ax.set_xlabel("detector score")
    ax.set_ylabel("detections")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    for label, values in sorted(by_label.items()):
        vals = values["score"]
        if vals:
            ax.hist(vals, bins=bins_score, histtype="step", linewidth=1.7, label=f"{label} n={len(vals)}")
    ax.axvline(post_thresh, color="#D62728", linestyle="-", linewidth=1.3)
    ax.set_title("Detector score by class")
    ax.set_xlabel("detector score")
    ax.set_ylabel("detections")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 0]
    bins_class = np.linspace(0.0, 1.0, 80)
    ax.hist(class_scores, bins=bins_class, color="#65A665", alpha=0.85)
    ax.set_title("Classifier softmax max score")
    ax.set_xlabel("class_score")
    ax.set_ylabel("detections")

    ax = axes[1, 1]
    zoom_hi = min(0.55, max(0.55, post_thresh + 0.2))
    zoom_bins = np.linspace(dino_thresh, zoom_hi, 70)
    ax.hist(scores, bins=zoom_bins, color="#B07AA1", alpha=0.85)
    ax.axvline(dino_thresh, color="#222222", linestyle="--", linewidth=1.4)
    ax.axvline(post_thresh, color="#D62728", linestyle="-", linewidth=1.6)
    ax.set_title("Detector score near thresholds")
    ax.set_xlabel("detector score")
    ax.set_ylabel("detections")
    ax.text(
        0.03,
        0.95,
        "\n".join(
            [
                f"total JSONL detections: {len(scores):,}",
                f"score < {post_thresh:.2f}: {below_post:,} ({below_post / max(len(scores), 1):.2%})",
                "class_score cutoff: none",
                "JSONL is already thresholded at DINO output.",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )

    fig.suptitle(args.title or args.jsonl.name, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_png)
    plt.close(fig)

    summary = {
        "jsonl": str(args.jsonl),
        "output_png": str(output_png),
        "dino_score_thresh": dino_thresh,
        "post_score_thresh": post_thresh,
        "total_detections": len(scores),
        "score_below_post_thresh": below_post,
        "class_counts": dict(label_counts),
        "detector_score_quantiles": quantiles(scores),
        "class_score_quantiles": quantiles(class_scores),
        "by_class": {
            label: {
                "count": len(values["score"]),
                "score_below_post_thresh": int(np.count_nonzero(np.asarray(values["score"]) < post_thresh)),
                "detector_score_quantiles": quantiles(values["score"]),
                "class_score_quantiles": quantiles(values["class_score"]),
            }
            for label, values in sorted(by_label.items())
        },
    }
    summary_path = output_png.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_cuts(tracked_sqlite: Path) -> set[int]:
    conn = sqlite3.connect(str(tracked_sqlite))
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "cuts" not in tables:
            return set()
        return {int(row[0]) for row in conn.execute("SELECT frame FROM cuts")}
    finally:
        conn.close()


def replay_tracking_confidence(
    jsonl_path: Path,
    tracked_sqlite: Path,
    remove_short_tracks_max_frames: int,
    post_score_thresh: float,
) -> tuple[dict[tuple[int, str], ConfidenceRow], dict[str, object]]:
    tracks: dict[int, legacy.infer_RawTrack] = {}
    active_track_ids: list[int] = []
    next_tid = 1
    cut_frames = load_cuts(tracked_sqlite)
    all_rows: list[dict[str, object]] = []
    raw_total = 0
    below_post = 0
    after_score = 0
    after_nms = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src_dets = obj.get("detections") or obj.get("instances") or []
            src_dicts = [det for det in src_dets if isinstance(det, dict)] if isinstance(src_dets, list) else []
            frame_idx, detections = legacy.infer_normalize_raw_record(obj)
            for det, src in zip(detections, src_dicts):
                det["class_score"] = to_float(src.get("class_score"))
                det["detector_score"] = to_float(src.get("detector_score", src.get("score")))
                det["category_id"] = to_int(src.get("category_id"))
                det["category_index"] = to_int(src.get("category_index"))
            if frame_idx in cut_frames:
                active_track_ids.clear()
            raw_total += len(detections)
            below_post += sum(1 for det in detections if float(det.get("score") or 0.0) < post_score_thresh)
            detections = [det for det in detections if float(det.get("score") or 0.0) >= post_score_thresh]
            after_score += len(detections)
            detections = legacy.infer_raw_apply_nms(detections)
            after_nms += len(detections)
            det_features = [legacy.infer_raw_compute_features(det) for det in detections]
            if active_track_ids:
                active_track_ids = [
                    tid for tid in active_track_ids if frame_idx - tracks[tid].last_frame <= legacy.infer_RAW_MAX_GAP_FRAMES
                ]
            candidates: list[tuple[float, int, int]] = []
            for det_idx, feat in enumerate(det_features):
                for tid in active_track_ids:
                    score = legacy.infer_raw_compute_match_score(tracks[tid], feat, frame_idx)
                    if score is not None:
                        candidates.append((score, tid, det_idx))
            candidates.sort(reverse=True)
            assigned_dets: set[int] = set()
            assigned_tracks: set[int] = set()
            det_to_track: dict[int, int] = {}
            for score, tid, det_idx in candidates:
                if det_idx in assigned_dets or tid in assigned_tracks:
                    continue
                det_to_track[det_idx] = tid
                assigned_dets.add(det_idx)
                assigned_tracks.add(tid)
            for det_idx, feat in enumerate(det_features):
                if det_idx in det_to_track:
                    tracks[det_to_track[det_idx]].update(frame_idx, feat)
                    continue
                tid = next_tid
                next_tid += 1
                tracks[tid] = legacy.infer_RawTrack(
                    track_id=tid,
                    scene_id=0,
                    last_frame=frame_idx,
                    bbox=feat.bbox,
                    center=feat.center,
                    area=feat.area,
                    aspect=feat.aspect,
                    poly_area=feat.poly_area,
                    fill_ratio=feat.fill_ratio,
                )
                det_to_track[det_idx] = tid
                active_track_ids.append(tid)
            for det_idx, det in enumerate(detections):
                polygons = det.get("polygons") or []
                if not polygons:
                    continue
                raw_tid = str(det_to_track[det_idx])
                all_rows.append(
                    {
                        "frame": int(frame_idx),
                        "raw_track_id": raw_tid,
                        "label": str(det.get("class_name", "")),
                        "score": float(det.get("score") or 0.0),
                        "class_score": to_float(det.get("class_score")),
                        "detector_score": to_float(det.get("detector_score")),
                        "category_id": to_int(det.get("category_id")),
                        "category_index": to_int(det.get("category_index")),
                    }
                )
    track_counts = Counter(str(row["raw_track_id"]) for row in all_rows)
    remove_tids = {tid for tid, count in track_counts.items() if count <= int(remove_short_tracks_max_frames)}
    keep_tids = sorted((tid for tid in track_counts if tid not in remove_tids), key=lambda value: int(value))
    id_map = {old: str(new) for new, old in enumerate(keep_tids, start=1)}
    lookup: dict[tuple[int, str], ConfidenceRow] = {}
    for row in all_rows:
        raw_tid = str(row["raw_track_id"])
        if raw_tid not in id_map:
            continue
        track_id = id_map[raw_tid]
        lookup[(int(row["frame"]), track_id)] = ConfidenceRow(
            frame=int(row["frame"]),
            raw_track_id=raw_tid,
            track_id=track_id,
            label=str(row["label"]),
            score=float(row["score"]),
            class_score=row["class_score"],
            detector_score=row["detector_score"],
            category_id=row["category_id"],
            category_index=row["category_index"],
        )
    summary = {
        "raw_jsonl_detections": raw_total,
        "score_below_post_thresh": below_post,
        "after_post_score_filter": after_score,
        "after_raw_nms": after_nms,
        "rows_before_short_track_prune": len(all_rows),
        "rows_after_short_track_prune": len(lookup),
        "removed_short_tracks": len(remove_tids),
        "removed_short_track_rows": sum(track_counts[tid] for tid in remove_tids),
        "post_score_thresh": float(post_score_thresh),
        "remove_short_tracks_max_frames": int(remove_short_tracks_max_frames),
    }
    return lookup, summary


def load_mask_rows(sqlite_path: Path, include_label: bool = False):
    conn = sqlite3.connect(str(sqlite_path))
    try:
        if include_label:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(masks)")}
            label_expr = "label" if "label" in cols else "NULL AS label"
            rows = conn.execute(f"SELECT frame, track_id, polygons, {label_expr} FROM masks ORDER BY frame, CAST(track_id AS INTEGER)").fetchall()
            return [(int(frame), str(track_id), str(polygons), None if label is None else str(label)) for frame, track_id, polygons, label in rows]
        rows = conn.execute("SELECT frame, track_id, polygons FROM masks ORDER BY frame, CAST(track_id AS INTEGER)").fetchall()
        return [(int(frame), str(track_id), str(polygons)) for frame, track_id, polygons in rows]
    finally:
        conn.close()


def load_track_labels(sqlite_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        labels: dict[str, str] = {}
        if "tracks" in tables:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
            if "label" in cols:
                for track_id, label in conn.execute("SELECT track_id, label FROM tracks"):
                    if label is not None:
                        labels[str(track_id)] = str(label)
        if not labels:
            for _frame, track_id, _polygons, label in load_mask_rows(sqlite_path, include_label=True):
                if label is not None:
                    labels.setdefault(track_id, label)
        return labels
    finally:
        conn.close()


def load_metric_lookup(csv_path: Path) -> dict[tuple[int, str], dict[str, object]]:
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            track_id = str(row["track_id"])
            lookup[(frame, track_id)] = {
                "mode": str(row.get("mode", "")),
                "candidate_name": str(row.get("candidate_name", "")),
                "recall": float(row.get("recall", 0.0)),
                "precision": float(row.get("precision", 0.0)),
                "iou": float(row.get("iou", 0.0)),
                "weighted_error": int(float(row.get("weighted_error", 0.0))),
            }
    return lookup


def load_k1_cost_lookup(csv_path: Path) -> dict[tuple[int, str], int]:
    return fst.load_k1_cost_lookup(csv_path)


def open_ffmpeg_writer(output_video: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.exists():
        output_video.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def draw_label_box(img: np.ndarray, anchor: tuple[int, int], lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.46
    thickness = 1
    sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text, _color in lines]
    text_w = max(w for w, _h in sizes)
    line_h = max(h for _w, h in sizes) + 6
    text_h = line_h * len(lines) + 8
    x, y = anchor
    x = int(np.clip(x, 6, max(6, img.shape[1] - text_w - 14)))
    y = int(np.clip(y, text_h + 6, max(text_h + 6, img.shape[0] - 6)))
    cv2.rectangle(img, (x - 5, y - text_h), (x + text_w + 9, y + 4), TEXT_BG, thickness=-1)
    cv2.rectangle(img, (x - 5, y - text_h), (x + text_w + 9, y + 4), (210, 210, 210), thickness=1)
    base_y = y - text_h + 18
    for idx, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x, base_y + idx * line_h), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_header(img: np.ndarray, frame_idx: int, args: argparse.Namespace) -> None:
    lines = [
        f"F:{frame_idx}  DINO score>={args.dino_score_thresh:.2f}  post score>={args.post_score_thresh:.2f}",
        f"class_score cutoff:none  short-track<= {args.remove_short_tracks_max_frames} removed",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 26
    height = 58
    cv2.rectangle(img, (10, 10), (10 + width, 10 + height), TEXT_BG, thickness=-1)
    cv2.rectangle(img, (10, 10), (10 + width, 10 + height), (220, 220, 220), thickness=1)
    cv2.putText(img, lines[0], (22, 34), font, scale, TEXT_FG, thickness, cv2.LINE_AA)
    cv2.putText(img, lines[1], (22, 58), font, scale, (160, 220, 255), thickness, cv2.LINE_AA)


def render_debug_overlay(args: argparse.Namespace) -> None:
    conf_lookup, tracking_summary = replay_tracking_confidence(
        args.jsonl,
        args.gt_sqlite,
        int(args.remove_short_tracks_max_frames),
        float(args.post_score_thresh),
    )
    gt_rows = load_mask_rows(args.gt_sqlite, include_label=True)
    pred_rows = load_mask_rows(args.pred_sqlite, include_label=False)
    labels_by_track = load_track_labels(args.gt_sqlite)
    metric_lookup = load_metric_lookup(args.metrics_csv)
    k1_cost_lookup = load_k1_cost_lookup(args.k1_cost_csv)

    tracked_key_count = len(gt_rows)
    missing_conf = sum(1 for frame, track_id, _polygons, _label in gt_rows if (frame, track_id) not in conf_lookup)
    gt_by_frame: dict[int, list[tuple[str, str, str | None]]] = defaultdict(list)
    for frame, track_id, polygons, label in gt_rows:
        gt_by_frame[frame].append((track_id, polygons, label))
    pred_by_frame: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for frame, track_id, polygons in pred_rows:
        pred_by_frame[frame].append((track_id, polygons))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = open_ffmpeg_writer(args.output_video, width, height, fps)
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            overlay = frame.copy()
            gt_entries = gt_by_frame.get(frame_idx, [])
            pred_entries = pred_by_frame.get(frame_idx, [])
            gt_by_track = {track_id: polygons for track_id, polygons, _label in gt_entries}
            if gt_entries:
                gt_mask = np.zeros((height, width), dtype=np.uint8)
                for _track_id, polygons_json, _label in gt_entries:
                    gt_mask |= fst.rasterize_full(polygons_json, height=height, width=width)
                    fst.draw_outlines(overlay, polygons_json, GT_OUTLINE_COLOR, 1)
                fst.blend_mask(overlay, gt_mask, GT_MASK_COLOR, float(args.mask_alpha))
                for _track_id, polygons_json, _label in gt_entries:
                    fst.draw_outlines(overlay, polygons_json, GT_OUTLINE_COLOR, 1)
            for track_id, polygons_json in pred_entries:
                label = labels_by_track.get(track_id)
                conf = conf_lookup.get((frame_idx, track_id))
                if conf is not None:
                    label = conf.label or label
                class_name = ascii_label(label, conf.category_id if conf is not None else None)
                color = CLASS_COLORS.get(class_name, FALLBACK_COLOR)
                fst.draw_outlines(overlay, polygons_json, color, int(args.pred_thickness))
                anchor_json = gt_by_track.get(track_id, polygons_json)
                anchor = fst.get_annotation_anchor(anchor_json, width=width, height=height)
                metric = metric_lookup.get((frame_idx, track_id))
                k1_cost = k1_cost_lookup.get((frame_idx, track_id))
                score_text = "det:gap"
                class_score_text = "cls:-"
                if conf is not None:
                    score_text = f"det:{conf.score:.3f}"
                    if conf.class_score is not None:
                        class_score_text = f"cls:{conf.class_score:.3f}"
                metric_text = "IoU:-"
                if metric is not None:
                    metric_text = f"IoU:{float(metric['iou']):.3f} R:{float(metric['recall']):.3f}"
                cost_text = "K1:-"
                if k1_cost is not None:
                    cost_text = f"K1:{int(k1_cost)}/5000"
                lines = [
                    (f"T{track_id} class:{class_name}", color),
                    (f"{score_text} {class_score_text} thr:{args.post_score_thresh:.2f}", TEXT_FG),
                    (f"{metric_text} {cost_text}", (160, 220, 255)),
                ]
                draw_label_box(overlay, anchor, lines)
            draw_header(overlay, frame_idx, args)
            proc.stdin.write(overlay.tobytes())
            frame_idx += 1
            if frame_idx % int(args.progress_every) == 0:
                print(f"rendered {frame_idx}/{total_frames}", flush=True)
            if args.max_frames and frame_idx >= int(args.max_frames):
                break
    finally:
        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        code = proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed with code {code}: {stderr}")

    summary = {
        "video": str(args.video),
        "jsonl": str(args.jsonl),
        "gt_sqlite": str(args.gt_sqlite),
        "pred_sqlite": str(args.pred_sqlite),
        "metrics_csv": str(args.metrics_csv),
        "k1_cost_csv": str(args.k1_cost_csv),
        "output_video": str(args.output_video),
        "frames_rendered": int(frame_idx),
        "tracked_rows": int(tracked_key_count),
        "tracked_rows_missing_replayed_confidence": int(missing_conf),
        "tracking_replay": tracking_summary,
        "dino_score_thresh": float(args.dino_score_thresh),
        "post_score_thresh": float(args.post_score_thresh),
    }
    args.output_video.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO confidence diagnostics and debug overlay renderer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plot = sub.add_parser("plot-confidence")
    p_plot.add_argument("--jsonl", type=Path, required=True)
    p_plot.add_argument("--output-png", type=Path, required=True)
    p_plot.add_argument("--dino-score-thresh", type=float, default=0.30)
    p_plot.add_argument("--post-score-thresh", type=float, default=0.35)
    p_plot.add_argument("--title", default="")
    p_plot.set_defaults(func=plot_confidence)

    p_render = sub.add_parser("render-overlay")
    p_render.add_argument("--video", type=Path, required=True)
    p_render.add_argument("--jsonl", type=Path, required=True)
    p_render.add_argument("--gt-sqlite", type=Path, required=True)
    p_render.add_argument("--pred-sqlite", type=Path, required=True)
    p_render.add_argument("--metrics-csv", type=Path, required=True)
    p_render.add_argument("--k1-cost-csv", type=Path, required=True)
    p_render.add_argument("--output-video", type=Path, required=True)
    p_render.add_argument("--dino-score-thresh", type=float, default=0.30)
    p_render.add_argument("--post-score-thresh", type=float, default=0.35)
    p_render.add_argument("--remove-short-tracks-max-frames", type=int, default=10)
    p_render.add_argument("--mask-alpha", type=float, default=0.24)
    p_render.add_argument("--pred-thickness", type=int, default=3)
    p_render.add_argument("--progress-every", type=int, default=300)
    p_render.add_argument("--max-frames", type=int, default=0)
    p_render.set_defaults(func=render_debug_overlay)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
