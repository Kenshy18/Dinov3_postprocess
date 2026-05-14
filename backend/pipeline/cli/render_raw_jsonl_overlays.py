#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import orjson
except Exception:  # pragma: no cover - optional speedup
    orjson = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = ROOT / "output" / "runs" / "input2_batch_20260511_003329"

LABEL_COLORS_BGR = {
    "女性器": (64, 80, 255),
    "男性器": (255, 170, 40),
    "結合部分": (80, 220, 80),
}
FALLBACK_COLOR_BGR = (0, 215, 255)
TEXT_BG_BGR = (14, 14, 14)
TEXT_FG_BGR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render clean overlay videos from DINOv3 raw JSONL masks. "
            "Only translucent masks, bounding boxes, and confidence scores are drawn."
        )
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-name", default="raw_ai_mask_confidence_overlay.mp4")
    parser.add_argument("--encoder", choices=("nvenc", "cpu"), default="nvenc")
    parser.add_argument("--mask-alpha", type=float, default=0.35)
    parser.add_argument("--box-thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--font-thickness", type=int, default=1)
    parser.add_argument("--progress-frames", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-frames", type=int, default=0, help="Debug only.")
    return parser.parse_args()


def load_json_line(line: str) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def ffprobe_stream(video_path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        cmd.append("-count_frames")
    cmd += [
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
    ]
    cmd.append(str(video_path))
    data = json.loads(run_command(cmd).stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")
    return streams[0]


def parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        den = float(den_s)
        return float(num_s) / den if den else 0.0
    return float(rate)


def choose_fps_expr(stream: dict[str, Any]) -> str:
    avg = str(stream.get("avg_frame_rate") or "")
    if avg and avg != "0/0":
        return avg
    r = str(stream.get("r_frame_rate") or "")
    if r and r != "0/0":
        return r
    raise RuntimeError("Could not determine source FPS.")


def stream_frame_count(stream: dict[str, Any]) -> int | None:
    for key in ("nb_read_frames", "nb_frames"):
        raw = stream.get(key)
        if raw not in (None, "N/A", ""):
            return int(raw)
    return None


def open_writer(
    output_video: Path,
    *,
    width: int,
    height: int,
    fps_expr: str,
    encoder: str,
) -> subprocess.Popen[bytes]:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if encoder == "nvenc":
        codec_args = [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            "23",
        ]
    else:
        codec_args = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
        ]
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
        "-framerate",
        fps_expr,
        "-i",
        "-",
        "-an",
        *codec_args,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def polygons_from_detection(det: dict[str, Any]) -> list[np.ndarray]:
    raw_polys = det.get("polygons")
    if raw_polys is None:
        raw_polys = det.get("segmentation")
    if not isinstance(raw_polys, list):
        return []

    polygons: list[np.ndarray] = []
    for poly in raw_polys:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 2 != 0:
                continue
            arr = arr.reshape(-1, 2)
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pass
        else:
            continue
        if arr.shape[0] >= 3:
            polygons.append(arr)
    return polygons


def bbox_from_detection(det: dict[str, Any], polygons: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    bbox = det.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    elif polygons:
        pts = np.concatenate(polygons, axis=0)
        x1, y1 = pts.min(axis=0).tolist()
        x2, y2 = pts.max(axis=0).tolist()
    else:
        return None
    return (
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    )


def confidence_from_detection(det: dict[str, Any]) -> float | None:
    for key in ("score", "detector_score", "class_score"):
        value = det.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def color_for_detection(det: dict[str, Any], index: int) -> tuple[int, int, int]:
    label = str(det.get("label") or det.get("class_name") or "")
    color = LABEL_COLORS_BGR.get(label)
    if color is not None:
        return color
    # Deterministic fallback palette jitter, still no label text is drawn.
    base = np.array(FALLBACK_COLOR_BGR, dtype=np.int16)
    jitter = np.array(((index * 47) % 80, (index * 29) % 80, (index * 71) % 80), dtype=np.int16)
    out = np.clip(base + jitter - 40, 0, 255)
    return int(out[0]), int(out[1]), int(out[2])


def blend_polygons(frame: np.ndarray, polygons: list[np.ndarray], color: tuple[int, int, int], alpha: float) -> None:
    if not polygons:
        return
    height, width = frame.shape[:2]
    pts = np.concatenate(polygons, axis=0)
    x0 = max(0, int(math.floor(float(pts[:, 0].min()))) - 2)
    y0 = max(0, int(math.floor(float(pts[:, 1].min()))) - 2)
    x1 = min(width - 1, int(math.ceil(float(pts[:, 0].max()))) + 2)
    y1 = min(height - 1, int(math.ceil(float(pts[:, 1].max()))) + 2)
    if x1 < x0 or y1 < y0:
        return

    roi = frame[y0 : y1 + 1, x0 : x1 + 1]
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
    shift = np.array([x0, y0], dtype=np.float32)
    shifted = [np.round(poly - shift).astype(np.int32).reshape(-1, 1, 2) for poly in polygons]
    cv2.fillPoly(mask, shifted, 1)
    idx = mask > 0
    if np.any(idx):
        color_arr = np.asarray(color, dtype=np.float32)
        roi[idx] = np.rint(roi[idx].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).clip(0, 255).astype(np.uint8)


def draw_score(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], font_scale: float, thickness: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = frame.shape[:2]
    px = int(np.clip(x, 0, max(0, width - tw - 7)))
    py = int(np.clip(y, th + baseline + 5, max(th + baseline + 5, height - 3)))
    cv2.rectangle(frame, (px, py - th - baseline - 5), (px + tw + 7, py + 3), TEXT_BG_BGR, thickness=-1)
    cv2.rectangle(frame, (px, py - th - baseline - 5), (px + tw + 7, py + 3), color, thickness=1)
    cv2.putText(frame, text, (px + 3, py - baseline - 1), font, font_scale, TEXT_FG_BGR, thickness, cv2.LINE_AA)


def draw_detection(
    frame: np.ndarray,
    det: dict[str, Any],
    *,
    index: int,
    alpha: float,
    box_thickness: int,
    font_scale: float,
    font_thickness: int,
) -> bool:
    polygons = polygons_from_detection(det)
    bbox = bbox_from_detection(det, polygons)
    score = confidence_from_detection(det)
    if bbox is None and not polygons:
        return False

    color = color_for_detection(det, index)
    blend_polygons(frame, polygons, color, alpha)

    height, width = frame.shape[:2]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        if x2 >= x1 and y2 >= y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=box_thickness, lineType=cv2.LINE_AA)
            if score is not None:
                draw_score(frame, f"{score:.2f}", x1, y1 - 4, color, font_scale, font_thickness)
    elif score is not None and polygons:
        pts = np.concatenate(polygons, axis=0)
        draw_score(frame, f"{score:.2f}", int(pts[:, 0].min()), int(pts[:, 1].min()) - 4, color, font_scale, font_thickness)
    return True


def load_run_specs(args: argparse.Namespace) -> list[dict[str, Path]]:
    if args.run_dir:
        run_dirs = [p.resolve() for p in args.run_dir]
    else:
        batch_summary = args.batch_dir / "batch_summary.json"
        if not batch_summary.exists():
            raise FileNotFoundError(f"Missing batch summary: {batch_summary}")
        data = json.loads(batch_summary.read_text(encoding="utf-8"))
        run_dirs = [Path(item["run_dir"]).resolve() for item in data.get("items", []) if item.get("status") == "completed"]

    specs: list[dict[str, Path]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        video = Path(summary["video"])
        jsonl = Path(summary["artifacts"]["dinov3_jsonl"])
        output_video = run_dir / "overlays" / args.output_name
        specs.append({"run_dir": run_dir, "summary": summary_path, "video": video, "jsonl": jsonl, "output": output_video})
    return specs


def render_one(spec: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    video_path = spec["video"]
    jsonl_path = spec["jsonl"]
    output_video = spec["output"]
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)
    if output_video.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_video} (use --force)")
    if output_video.exists():
        output_video.unlink()

    source_stream = ffprobe_stream(video_path)
    width = int(source_stream["width"])
    height = int(source_stream["height"])
    fps_expr = choose_fps_expr(source_stream)
    fps_float = parse_rate(fps_expr)
    expected_frames = stream_frame_count(source_stream)
    if expected_frames is None:
        counted = ffprobe_stream(video_path, count_frames=True)
        expected_frames = stream_frame_count(counted)
    if expected_frames is None:
        raise RuntimeError(f"Could not determine source frame count: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    proc = open_writer(output_video, width=width, height=height, fps_expr=fps_expr, encoder=args.encoder)
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")

    started = time.monotonic()
    written = 0
    detections_seen = 0
    detections_drawn = 0
    json_lines = 0
    last_print = started
    error: BaseException | None = None

    print(
        f"[overlay-start] run={spec['run_dir'].name} frames={expected_frames} "
        f"fps={fps_expr} size={width}x{height} encoder={args.encoder} output={output_video}",
        flush=True,
    )
    try:
        with jsonl_path.open("r", encoding="utf-8") as jf:
            frame_idx = 0
            while True:
                if args.limit_frames and frame_idx >= args.limit_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                line = jf.readline()
                if not line:
                    raise RuntimeError(f"JSONL ended before video at frame {frame_idx}: {jsonl_path}")
                json_lines += 1
                obj = load_json_line(line)
                json_frame = int(obj.get("frame_idx", obj.get("frame_index", frame_idx)))
                if json_frame != frame_idx:
                    raise RuntimeError(f"JSONL/video frame mismatch: video={frame_idx} jsonl={json_frame}")
                detections = obj.get("detections") or []
                detections_seen += len(detections)
                for det_idx, det in enumerate(detections):
                    if isinstance(det, dict) and draw_detection(
                        frame,
                        det,
                        index=det_idx,
                        alpha=float(args.mask_alpha),
                        box_thickness=int(args.box_thickness),
                        font_scale=float(args.font_scale),
                        font_thickness=int(args.font_thickness),
                    ):
                        detections_drawn += 1

                proc.stdin.write(frame.tobytes())
                written += 1
                frame_idx += 1

                if args.progress_frames > 0 and written % args.progress_frames == 0:
                    now = time.monotonic()
                    elapsed = now - started
                    fps = written / elapsed if elapsed > 0 else 0.0
                    remain = (expected_frames - written) / fps if fps > 0 else 0.0
                    print(
                        f"[overlay-progress] run={spec['run_dir'].name} "
                        f"{written}/{expected_frames} ({written / expected_frames * 100:.1f}%) "
                        f"fps={fps:.2f} elapsed={format_seconds(elapsed)} eta={format_seconds(remain)}",
                        flush=True,
                    )
                    last_print = now

            if not args.limit_frames:
                extra = jf.readline()
                if extra:
                    raise RuntimeError(f"JSONL has extra rows after video ended: {jsonl_path}")
    except BaseException as exc:
        error = exc
        raise
    finally:
        cap.release()
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        code = proc.wait()
        if proc.stderr is not None:
            proc.stderr.close()
        if code != 0 and error is None:
            raise RuntimeError(f"ffmpeg encode failed with code {code}: {stderr}")

    elapsed = time.monotonic() - started
    output_stream = ffprobe_stream(output_video, count_frames=True)
    output_frames = stream_frame_count(output_stream)
    output_fps_expr = choose_fps_expr(output_stream)
    output_width = int(output_stream["width"])
    output_height = int(output_stream["height"])
    if not args.limit_frames:
        if written != expected_frames:
            raise RuntimeError(f"Wrote {written} frames, expected {expected_frames}: {output_video}")
        if output_frames != expected_frames:
            raise RuntimeError(f"Encoded {output_frames} frames, expected {expected_frames}: {output_video}")
    if output_width != width or output_height != height:
        raise RuntimeError(f"Output size mismatch: {output_width}x{output_height} vs {width}x{height}")
    if abs(parse_rate(output_fps_expr) - fps_float) > 1e-5:
        raise RuntimeError(f"Output FPS mismatch: {output_fps_expr} vs {fps_expr}")

    result = {
        "run_dir": str(spec["run_dir"]),
        "video": str(video_path),
        "jsonl": str(jsonl_path),
        "output_video": str(output_video),
        "encoder": args.encoder,
        "source": {
            "width": width,
            "height": height,
            "fps": fps_expr,
            "fps_float": fps_float,
            "frames": expected_frames,
        },
        "output": {
            "width": output_width,
            "height": output_height,
            "fps": output_fps_expr,
            "frames": output_frames,
        },
        "json_lines_read": json_lines,
        "frames_written": written,
        "detections_seen": detections_seen,
        "detections_drawn": detections_drawn,
        "elapsed_seconds": elapsed,
        "render_fps": written / elapsed if elapsed > 0 else 0.0,
        "mask_alpha": float(args.mask_alpha),
        "verification": {
            "frame_count_matches": output_frames == expected_frames,
            "fps_matches": abs(parse_rate(output_fps_expr) - fps_float) <= 1e-5,
            "size_matches": output_width == width and output_height == height,
        },
    }
    summary_path = output_video.with_suffix(".json")
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[overlay-done] run={spec['run_dir'].name} elapsed={format_seconds(elapsed)} "
        f"render_fps={result['render_fps']:.2f} frames={written} detections={detections_drawn} "
        f"summary={summary_path}",
        flush=True,
    )
    return result


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    specs = load_run_specs(args)
    if not specs:
        raise RuntimeError("No runs to render.")

    batch_overlay_dir = batch_dir / "overlays"
    batch_overlay_dir.mkdir(parents=True, exist_ok=True)
    batch_summary_path = batch_overlay_dir / "raw_ai_mask_confidence_overlay_summary.json"

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        print(f"[batch-overlay-start] {idx}/{len(specs)} run={spec['run_dir'].name}", flush=True)
        result = render_one(spec, args)
        results.append(result)
        batch_summary_path.write_text(
            json.dumps(
                {
                    "status": "running" if idx < len(specs) else "completed",
                    "batch_dir": str(batch_dir),
                    "started_at_unix": started,
                    "elapsed_seconds": time.monotonic() - started,
                    "items": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[batch-overlay-done] {idx}/{len(specs)} elapsed={format_seconds(time.monotonic() - started)}",
            flush=True,
        )

    print(f"[batch-overlay-complete] elapsed={format_seconds(time.monotonic() - started)} summary={batch_summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[overlay-cancelled]", file=sys.stderr)
        raise
