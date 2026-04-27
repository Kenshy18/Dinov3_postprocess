from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import standalone_runtime_fst


import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from bisect import bisect_left
import cv2
import numpy as np
render_ROOT = Path(__file__).resolve().parent
render_TEACHER_ROOT = render_ROOT / 'Teacher'
if str(render_TEACHER_ROOT) not in sys.path:
    sys.path.insert(0, str(render_TEACHER_ROOT))
import final_standalone_t5000 as fst
render_PRED_FALLBACK_COLOR = (0, 215, 255)
render_GT_OUTLINE_COLOR = (255, 255, 255)
render_TEXT_BG = (18, 18, 18)
render_TEXT_FG = (245, 245, 245)
render_MODE_K1_COLOR = (0, 255, 0)
render_MODE_K2_COLOR = (255, 0, 255)

def render_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render overlay video for exact K1 + K2 V5 output.')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--gt-sqlite', type=Path, required=True)
    parser.add_argument('--pred-sqlite', type=Path, required=True)
    parser.add_argument('--metrics-csv', type=Path, required=True)
    parser.add_argument('--k1-cost-csv', type=Path, required=True, help='CSV containing the original K1 weighted_error for each frame/track row.')
    parser.add_argument('--output-video', type=Path, required=True)
    parser.add_argument('--encoder', choices=('nvenc', 'cpu'), default='nvenc')
    parser.add_argument('--mask-alpha', type=float, default=0.28)
    parser.add_argument('--gt-thickness', type=int, default=2)
    parser.add_argument('--pred-thickness', type=int, default=3)
    parser.add_argument('--k1-threshold', type=int, default=5000)
    parser.add_argument('--k1-threshold-edge', type=int, default=5000)
    parser.add_argument('--show-keyframe-progress', action='store_true')
    parser.add_argument('--progress-thickness', type=int, default=4)
    parser.add_argument('--keyframes-json', type=Path, default=None)
    return parser

def render_load_rows(sqlite_path: Path) -> list[tuple[int, str, str]]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        return [(int(frame), str(track_id), str(polygons)) for frame, track_id, polygons in conn.execute('SELECT frame, track_id, polygons FROM masks ORDER BY frame, track_id')]
    finally:
        conn.close()

def render_load_metric_data(csv_path: Path) -> tuple[dict[tuple[int, str], dict[str, object]], dict[str, list[int]]]:
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    keyframes_by_track: dict[str, list[int]] = defaultdict(list)
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row['frame'])
            track_id = str(row['track_id'])
            has_keyframe = int(float(row.get('has_keyframe', 0)))
            key = (frame, track_id)
            lookup[key] = {'mode': str(row.get('mode', '')), 'candidate_name': str(row.get('candidate_name', '')), 'recall': float(row.get('recall', 0.0)), 'precision': float(row.get('precision', 0.0)), 'iou': float(row.get('iou', 0.0)), 'weighted_error': int(float(row.get('weighted_error', 0.0))), 'has_keyframe': has_keyframe}
            if has_keyframe:
                keyframes_by_track[track_id].append(frame)
    for frames in keyframes_by_track.values():
        frames.sort()
    return (lookup, dict(keyframes_by_track))

def render_load_slot_keyframes(path: Path | None) -> dict[tuple[str, int, int], list[int]]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding='utf-8'))
    lookup: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (str(row['track_id']), int(row.get('run_id', -1)), int(row.get('slot_id', 0)))
        lookup[key].append(int(row['frame']))
    for frames in lookup.values():
        frames.sort()
    return dict(lookup)

def render_draw_track_annotation(img: np.ndarray, anchor: tuple[int, int], track_id: str, metric_row: dict[str, object] | None, k1_weighted_error: int | None, k1_threshold: int) -> None:
    if metric_row is None:
        return
    mode = str(metric_row['mode']).upper()
    score_line = f"T{track_id} {mode} IoU:{float(metric_row['iou']):.3f} R:{float(metric_row['recall']):.3f} P:{float(metric_row['precision']):.3f}"
    if k1_weighted_error is None:
        aux_line = f"K1cost:NA/{int(k1_threshold)} {str(metric_row['candidate_name'])}"
    else:
        aux_line = f"K1cost:{int(k1_weighted_error)}/{int(k1_threshold)} {str(metric_row['candidate_name'])}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (w1, h1), b1 = cv2.getTextSize(score_line, font, font_scale, thickness)
    (w2, h2), b2 = cv2.getTextSize(aux_line, font, font_scale, thickness)
    text_w = max(w1, w2)
    text_h = h1 + h2 + b1 + b2 + 16
    x, y = anchor
    x = int(np.clip(x, 6, max(6, img.shape[1] - text_w - 12)))
    y = int(np.clip(y, text_h + 4, max(text_h + 4, img.shape[0] - 6)))
    top_left = (x - 4, y - text_h)
    bottom_right = (x + text_w + 8, y + 4)
    mode_color = render_MODE_K2_COLOR if mode == 'K2' else render_MODE_K1_COLOR
    cv2.rectangle(img, top_left, bottom_right, render_TEXT_BG, thickness=-1)
    cv2.rectangle(img, top_left, bottom_right, mode_color, thickness=2)
    cv2.putText(img, score_line, (x, y - h2 - 8), font, font_scale, render_TEXT_FG, thickness, cv2.LINE_AA)
    cv2.putText(img, aux_line, (x, y - 4), font, font_scale, mode_color, thickness, cv2.LINE_AA)

def render_get_pred_color(metric_row: dict[str, object] | None) -> tuple[int, int, int]:
    if metric_row is None:
        return render_PRED_FALLBACK_COLOR
    mode = str(metric_row.get('mode', '')).upper()
    if mode == 'K2':
        return render_MODE_K2_COLOR
    if mode == 'K1':
        return render_MODE_K1_COLOR
    return render_PRED_FALLBACK_COLOR

def render_get_progress_ratio(frame_idx: int, track_id: str, keyframes_by_track: dict[str, list[int]]) -> float | None:
    frames = keyframes_by_track.get(track_id)
    if not frames:
        return None
    pos = bisect_left(frames, frame_idx)
    if pos < len(frames) and frames[pos] == frame_idx:
        return 0.0
    if pos == 0 or pos >= len(frames):
        return None
    prev_frame = frames[pos - 1]
    next_frame = frames[pos]
    span = next_frame - prev_frame
    if span <= 0:
        return None
    return float((frame_idx - prev_frame) / span)

def render_get_slot_progress_ratio(frame_idx: int, track_id: str, run_id: int, slot_id: int, slot_keyframes: dict[tuple[str, int, int], list[int]]) -> float | None:
    frames = slot_keyframes.get((track_id, run_id, slot_id))
    if not frames:
        return None
    pos = bisect_left(frames, frame_idx)
    if pos < len(frames) and frames[pos] == frame_idx:
        return 0.0
    if pos == 0 or pos >= len(frames):
        return None
    prev_frame = frames[pos - 1]
    next_frame = frames[pos]
    span = next_frame - prev_frame
    if span <= 0:
        return None
    return float((frame_idx - prev_frame) / span)

def render_fit_progress_ellipse(polygons_json: str) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    polys = fst.parse_polygons(polygons_json)
    if not polys:
        return None
    points = np.concatenate([poly.astype(np.float32) for poly in polys if len(poly) >= 2], axis=0)
    if len(points) < 5:
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        center = (x + 0.5 * w, y + 0.5 * h)
        axes = (max(1.0, 0.5 * w), max(1.0, 0.5 * h))
        return (center, axes, 0.0)
    fitted = cv2.fitEllipse(points.reshape(-1, 1, 2))
    (cx, cy), (major, minor), angle = fitted
    return ((float(cx), float(cy)), (max(1.0, float(major) * 0.55), max(1.0, float(minor) * 0.55)), float(angle))

def render_fit_progress_ellipse_from_polygon(poly: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    points = poly.astype(np.float32)
    if len(points) < 2:
        return None
    if len(points) < 5:
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        center = (x + 0.5 * w, y + 0.5 * h)
        axes = (max(1.0, 0.5 * w), max(1.0, 0.5 * h))
        return (center, axes, 0.0)
    fitted = cv2.fitEllipse(points.reshape(-1, 1, 2))
    (cx, cy), (major, minor), angle = fitted
    return ((float(cx), float(cy)), (max(1.0, float(major) * 0.55), max(1.0, float(minor) * 0.55)), float(angle))

def render_draw_keyframe_progress(img: np.ndarray, polygons_json: str, ratio: float | None, color: tuple[int, int, int], thickness: int) -> None:
    if ratio is None:
        return
    fitted = render_fit_progress_ellipse(polygons_json)
    if fitted is None:
        return
    center, axes, angle = fitted
    outer_axes = (axes[0] + 8.0, axes[1] + 8.0)
    center_int = (int(round(center[0])), int(round(center[1])))
    axes_int = (max(1, int(round(outer_axes[0]))), max(1, int(round(outer_axes[1]))))
    cv2.ellipse(img, center_int, axes_int, angle, 0, 360, (40, 40, 40), thickness + 1, cv2.LINE_AA)
    end_angle = max(0.0, min(360.0, 360.0 * ratio))
    if end_angle > 0.0:
        cv2.ellipse(img, center_int, axes_int, angle, -90, -90 + end_angle, color, thickness, cv2.LINE_AA)
    else:
        cv2.circle(img, center_int, max(2, thickness), color, thickness=-1, lineType=cv2.LINE_AA)

def render_draw_keyframe_progress_poly(img: np.ndarray, poly: np.ndarray, ratio: float | None, color: tuple[int, int, int], thickness: int) -> None:
    if ratio is None:
        return
    fitted = render_fit_progress_ellipse_from_polygon(poly)
    if fitted is None:
        return
    center, axes, angle = fitted
    outer_axes = (axes[0] + 8.0, axes[1] + 8.0)
    center_int = (int(round(center[0])), int(round(center[1])))
    axes_int = (max(1, int(round(outer_axes[0]))), max(1, int(round(outer_axes[1]))))
    cv2.ellipse(img, center_int, axes_int, angle, 0, 360, (40, 40, 40), thickness + 1, cv2.LINE_AA)
    end_angle = max(0.0, min(360.0, 360.0 * ratio))
    if end_angle > 0.0:
        cv2.ellipse(img, center_int, axes_int, angle, -90, -90 + end_angle, color, thickness, cv2.LINE_AA)
    else:
        cv2.circle(img, center_int, max(2, thickness), color, thickness=-1, lineType=cv2.LINE_AA)

def render_render_overlay_video(video_path: Path, gt_rows: list[tuple[int, str, str]], pred_rows: list[tuple[int, str, str]], metric_lookup: dict[tuple[int, str], dict[str, object]], k1_cost_lookup: dict[tuple[int, str], int], keyframes_by_track: dict[str, list[int]], slot_keyframes: dict[tuple[str, int, int], list[int]], output_video: Path, *, encoder: str, mask_alpha: float, gt_thickness: int, pred_thickness: int, k1_threshold: int, k1_threshold_edge: int, show_keyframe_progress: bool, progress_thickness: int) -> None:
    gt_by_frame: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for frame, track_id, polygons in gt_rows:
        gt_by_frame[frame].append((track_id, polygons))
    pred_by_frame: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for frame, track_id, polygons in pred_rows:
        pred_by_frame[frame].append((track_id, polygons))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.exists():
        output_video.unlink()
    proc = render_open_video_writer(str(output_video), width, height, fps, encoder=encoder)
    if proc.stdin is None:
        raise RuntimeError('Failed to open ffmpeg stdin.')
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            overlay = frame.copy()
            gt_entries = gt_by_frame.get(frame_idx, [])
            pred_entries = pred_by_frame.get(frame_idx, [])
            gt_by_track = {track_id: polygons_json for track_id, polygons_json in gt_entries}
            if gt_entries:
                gt_mask = np.zeros((height, width), dtype=np.uint8)
                for _track_id, polygons_json in gt_entries:
                    gt_mask |= fst.rasterize_full(polygons_json, height=height, width=width)
                    fst.draw_outlines(overlay, polygons_json, render_GT_OUTLINE_COLOR, gt_thickness)
                fst.blend_mask(overlay, gt_mask, fst.MASK_COLOR, mask_alpha)
                for _track_id, polygons_json in gt_entries:
                    fst.draw_outlines(overlay, polygons_json, render_GT_OUTLINE_COLOR, gt_thickness)
            for track_id, polygons_json in pred_entries:
                metric_row = metric_lookup.get((frame_idx, track_id))
                pred_color = render_get_pred_color(metric_row)
                fst.draw_outlines(overlay, polygons_json, pred_color, pred_thickness)
                if show_keyframe_progress:
                    run_id = int(metric_row.get('run_id', -1)) if metric_row is not None else -1
                    polys = fst.parse_polygons(polygons_json)
                    if slot_keyframes and polys:
                        for slot_id, poly in enumerate(polys):
                            render_draw_keyframe_progress_poly(overlay, poly, render_get_slot_progress_ratio(frame_idx, track_id, run_id, slot_id, slot_keyframes), pred_color, progress_thickness)
                    else:
                        render_draw_keyframe_progress(overlay, polygons_json, render_get_progress_ratio(frame_idx, track_id, keyframes_by_track), pred_color, progress_thickness)
                anchor_polygons = gt_by_track.get(track_id, polygons_json)
                anchor = fst.get_annotation_anchor(anchor_polygons, width=width, height=height)
                gt_track_mask = fst.rasterize_full(anchor_polygons, height=height, width=width)
                edge_touch = bool(np.any(gt_track_mask[0, :]) or np.any(gt_track_mask[-1, :]) or np.any(gt_track_mask[:, 0]) or np.any(gt_track_mask[:, -1]))
                threshold = int(k1_threshold_edge if edge_touch else k1_threshold)
                render_draw_track_annotation(overlay, anchor, track_id, metric_row, k1_cost_lookup.get((frame_idx, track_id)), threshold)
            cv2.putText(overlay, f'F:{frame_idx}', (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            proc.stdin.write(overlay.tobytes())
            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f'  rendered {frame_idx}/{total_frames}')
    finally:
        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr is not None else ''
        code = proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        if code != 0:
            raise RuntimeError(f'ffmpeg encode failed with code {code}: {stderr}')

def render_open_video_writer(output_video: str, width: int, height: int, fps: float, *, encoder: str) -> subprocess.Popen:
    if encoder == 'nvenc':
        return fst.open_nvenc_writer(output_video, width, height, fps)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{fps:.8f}', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', output_video]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

def render_main() -> None:
    args = render_build_parser().parse_args()
    gt_rows = render_load_rows(args.gt_sqlite)
    pred_rows = render_load_rows(args.pred_sqlite)
    metric_lookup, keyframes_by_track = render_load_metric_data(args.metrics_csv)
    k1_cost_lookup = fst.load_k1_cost_lookup(args.k1_cost_csv)
    slot_keyframes = render_load_slot_keyframes(args.keyframes_json)
    render_render_overlay_video(args.video, gt_rows, pred_rows, metric_lookup, k1_cost_lookup, keyframes_by_track, slot_keyframes, args.output_video, encoder=str(args.encoder), mask_alpha=float(args.mask_alpha), gt_thickness=int(args.gt_thickness), pred_thickness=int(args.pred_thickness), k1_threshold=int(args.k1_threshold), k1_threshold_edge=int(args.k1_threshold_edge), show_keyframe_progress=bool(args.show_keyframe_progress), progress_thickness=int(args.progress_thickness))
    summary = {'video': str(args.video), 'gt_sqlite': str(args.gt_sqlite), 'pred_sqlite': str(args.pred_sqlite), 'metrics_csv': str(args.metrics_csv), 'k1_cost_csv': str(args.k1_cost_csv), 'output_video': str(args.output_video), 'row_count': len(pred_rows), 'k1_threshold': int(args.k1_threshold), 'k1_threshold_edge': int(args.k1_threshold_edge)}
    summary_path = args.output_video.with_suffix('.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
