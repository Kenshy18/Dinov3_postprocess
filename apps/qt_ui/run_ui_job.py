#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.run_audit import build_output_audit  # noqa: E402


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
RAW_COLOR_BGR = (0, 180, 255)
ORIGINAL_COLOR_BGR = (255, 255, 255)
POST_COLOR_BGR = (30, 230, 80)
TEXT_BG_BGR = (12, 12, 12)
TEXT_FG_BGR = (255, 255, 255)
LABEL_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
_LABEL_FONT: ImageFont.ImageFont | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one UI pipeline job and arrange user-facing outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overlay-mode", choices=("none", "detailed", "simple", "both"), default="none")
    parser.add_argument("--raw-overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder", choices=("nvenc", "cpu"), default="nvenc")
    parser.add_argument("--keep-normalized-input", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("pipeline_command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def strip_remainder(command: list[str]) -> list[str]:
    return command[1:] if command and command[0] == "--" else command


def extract_max_frames(command: list[str]) -> int | None:
    for index, part in enumerate(command):
        if part == "--max-frames" and index + 1 < len(command):
            try:
                return max(0, int(command[index + 1]))
            except ValueError:
                return None
        if part.startswith("--max-frames="):
            try:
                return max(0, int(part.split("=", 1)[1]))
            except ValueError:
                return None
    return None


def option_value(command: list[str], option: str) -> str | None:
    prefix = option + "="
    for index, part in enumerate(command):
        if part == option and index + 1 < len(command):
            return str(command[index + 1])
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None


def option_enabled(command: list[str], positive: str, negative: str) -> bool | None:
    value: bool | None = None
    for part in command:
        if part == positive:
            value = True
        elif part == negative:
            value = False
    return value


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_audit(audit_path: Path, event: str, **fields: object) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def runtime_snapshot(pipeline_command: list[str]) -> dict[str, Any]:
    tracked_env = {
        key: os.environ.get(key)
        for key in (
            "DINOV3_RUNTIME_PROFILE",
            "DINOV3_BATCH_BENCHMARK",
            "DINOV3_TRT_BACKBONE_ENGINE",
            "ATOSYORI_REPO",
            "PYTHONPATH",
        )
        if os.environ.get(key)
    }
    return {
        "python": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "environment": tracked_env,
        "command": pipeline_command,
        "settings": {
            "detector": option_value(pipeline_command, "--detector"),
            "postprocess": option_enabled(pipeline_command, "--postprocess", "--no-postprocess"),
            "shape_mode": option_value(pipeline_command, "--default-shape-mode"),
            "intervals": option_value(pipeline_command, "--intervals"),
            "class_policy_json": option_value(pipeline_command, "--class-policy-json"),
            "batch_size": option_value(pipeline_command, "--batch-size"),
            "eva02_batch_size": option_value(pipeline_command, "--eva02-batch-size"),
            "eva02_classifier_batch_size": option_value(pipeline_command, "--eva02-classifier-batch-size"),
            "codino_batch_size": option_value(pipeline_command, "--codino-batch-size"),
            "warmup_frames": option_value(pipeline_command, "--warmup-frames"),
            "eva02_warmup_frames": option_value(pipeline_command, "--eva02-warmup-frames"),
            "codino_warmup_frames": option_value(pipeline_command, "--codino-warmup-frames"),
            "max_frames": option_value(pipeline_command, "--max-frames"),
            "score_thresh": option_value(pipeline_command, "--score-thresh"),
            "eva02_score_thresh": option_value(pipeline_command, "--eva02-score-thresh"),
            "codino_score_thresh": option_value(pipeline_command, "--codino-score-thresh"),
            "trt_backbone_engine": option_value(pipeline_command, "--trt-backbone-engine"),
            "codino_trt_backbone_engine": option_value(pipeline_command, "--codino-trt-backbone-engine"),
        },
    }


def run_streamed(command: list[str], *, cwd: Path, log_path: Path, audit_path: Path, label: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.perf_counter()
    write_audit(audit_path, "command_start", label=label, command=command, cwd=str(cwd))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[cmd] {command_text(command)}\n")
        log.flush()
        try:
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
                log.write(line)
                log.flush()
            return_code = process.wait()
        except BaseException as exc:
            elapsed = time.perf_counter() - start
            write_audit(
                audit_path,
                "command_exception",
                label=label,
                command=command,
                elapsed_sec=elapsed,
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            raise
    elapsed = time.perf_counter() - start
    write_audit(audit_path, "command_done", label=label, command=command, returncode=return_code, elapsed_sec=elapsed)
    return return_code


def run_capture(command: list[str], *, audit_path: Path | None = None, label: str | None = None) -> subprocess.CompletedProcess[str]:
    start = time.perf_counter()
    if audit_path is not None:
        write_audit(audit_path, "command_start", label=label or command[0], command=command)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    elapsed = time.perf_counter() - start
    if audit_path is not None:
        write_audit(
            audit_path,
            "command_done",
            label=label or command[0],
            command=command,
            returncode=completed.returncode,
            elapsed_sec=elapsed,
            stdout_tail=completed.stdout[-4000:],
            stderr_tail=completed.stderr[-4000:],
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command_text(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    return completed


def ffprobe_media(video_path: Path, *, audit_path: Path | None = None) -> dict[str, Any]:
    data = json.loads(
        run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(video_path),
            ],
            audit_path=audit_path,
            label="ffprobe",
        ).stdout
    )
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")
    return {"stream": streams[0], "format": data.get("format") or {}}


def ffprobe_stream(video_path: Path, *, audit_path: Path | None = None) -> dict[str, Any]:
    return dict(ffprobe_media(video_path, audit_path=audit_path)["stream"])


def is_interlaced(video_path: Path, *, audit_path: Path | None = None) -> bool:
    field_order = str(ffprobe_stream(video_path, audit_path=audit_path).get("field_order") or "").lower()
    return field_order not in {"", "unknown", "progressive"}


def parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 30.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 30.0
    return float(rate)


def video_meta(video_path: Path, *, audit_path: Path | None = None) -> tuple[int, int, float]:
    stream = ffprobe_stream(video_path, audit_path=audit_path)
    fps = parse_rate(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"))
    return int(stream["width"]), int(stream["height"]), fps


def stream_rotation_degrees(stream: dict[str, Any]) -> int:
    candidates: list[object] = []
    tags = stream.get("tags")
    if isinstance(tags, dict):
        candidates.append(tags.get("rotate"))
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict):
                candidates.append(item.get("rotation"))
    for value in candidates:
        if value in (None, ""):
            continue
        try:
            degrees = int(round(float(str(value).strip())))
        except ValueError:
            continue
        return degrees % 360
    return 0


def sample_aspect_ratio(stream: dict[str, Any]) -> str:
    return str(stream.get("sample_aspect_ratio") or "").strip().lower()


def is_square_sar(stream: dict[str, Any]) -> bool:
    sar = sample_aspect_ratio(stream)
    return sar in {"", "0:1", "1:1", "1/1"}


def is_variable_frame_rate(stream: dict[str, Any]) -> bool:
    avg = parse_rate(str(stream.get("avg_frame_rate") or "0/0"))
    nominal = parse_rate(str(stream.get("r_frame_rate") or "0/0"))
    if avg <= 0 or nominal <= 0:
        return False
    return abs(avg - nominal) / max(avg, nominal) > 0.005


def output_fps_for_normalization(stream: dict[str, Any]) -> float:
    avg = parse_rate(str(stream.get("avg_frame_rate") or "0/0"))
    nominal = parse_rate(str(stream.get("r_frame_rate") or "0/0"))
    fps = avg if avg > 0 else nominal
    if fps <= 0:
        fps = 30.0
    return max(1.0, min(fps, 240.0))


def normalization_reasons(media: dict[str, Any], input_path: Path) -> list[str]:
    stream = dict(media.get("stream") or {})
    fmt = dict(media.get("format") or {})
    field_order = str(stream.get("field_order") or "").lower()
    codec = str(stream.get("codec_name") or "").lower()
    pix_fmt = str(stream.get("pix_fmt") or "").lower()
    format_name = str(fmt.get("format_name") or "").lower()
    interlaced = field_order not in {"", "unknown", "progressive"}
    stable_mp4 = input_path.suffix.lower() in {".mp4", ".m4v"} and "mp4" in format_name
    stable_codec = codec in {"h264", "avc1"}
    reasons: list[str] = []
    if interlaced:
        reasons.append(f"interlaced:{field_order}")
    if not stable_mp4:
        reasons.append(f"container:{format_name or input_path.suffix}")
    if not stable_codec:
        reasons.append(f"codec:{codec or 'unknown'}")
    if pix_fmt and pix_fmt not in {"yuv420p", "yuvj420p"}:
        reasons.append(f"pix_fmt:{pix_fmt}")
    rotation = stream_rotation_degrees(stream)
    if rotation:
        reasons.append(f"rotation:{rotation}")
    if not is_square_sar(stream):
        reasons.append(f"sar:{sample_aspect_ratio(stream)}")
    if is_variable_frame_rate(stream):
        avg = str(stream.get("avg_frame_rate") or "")
        nominal = str(stream.get("r_frame_rate") or "")
        reasons.append(f"variable_frame_rate:avg={avg}:r={nominal}")
    return reasons


def needs_normalization(media: dict[str, Any], input_path: Path) -> tuple[bool, str]:
    reasons = normalization_reasons(media, input_path)
    if reasons:
        return True, ";".join(reasons)
    return False, "stable_h264_mp4"


def split_normalization_reason(reason: str) -> list[str]:
    if reason == "stable_h264_mp4":
        return []
    return [part for part in reason.split(";") if part]


def build_normalization_filter(media: dict[str, Any], reason: str) -> str:
    stream = dict(media.get("stream") or {})
    reasons = split_normalization_reason(reason)
    filters: list[str] = []
    if any(item.startswith("interlaced:") for item in reasons):
        filters.append("yadif=mode=send_frame:parity=auto:deint=all")
    rotation = stream_rotation_degrees(stream)
    if rotation == 90:
        filters.append("transpose=1")
    elif rotation == 180:
        filters.extend(["hflip", "vflip"])
    elif rotation == 270:
        filters.append("transpose=2")
    elif rotation:
        filters.append(f"rotate={rotation}*PI/180:ow=rotw(iw):oh=roth(ih)")
    if any(item.startswith("variable_frame_rate:") for item in reasons):
        filters.append(f"fps={output_fps_for_normalization(stream):.8f}")
    if not is_square_sar(stream):
        filters.extend(["scale=ceil(iw*sar/2)*2:ceil(ih/2)*2", "setsar=1"])
    else:
        filters.append("scale=ceil(iw/2)*2:ceil(ih/2)*2")
    filters.append("format=yuv420p")
    return ",".join(filters)


def normalize_input_if_needed(input_path: Path, run_dir: Path, *, force: bool, audit_path: Path) -> tuple[Path, bool, str]:
    if input_path.suffix.lower() not in VIDEO_EXTS:
        raise RuntimeError(f"Unsupported video extension: {input_path}")

    media = ffprobe_media(input_path, audit_path=audit_path)
    normalize, reason = needs_normalization(media, input_path)
    write_audit(audit_path, "input_probe", input=input_path, media=media, normalize=normalize, reason=reason)
    if not normalize:
        return input_path, False, reason

    normalized_dir = run_dir / "sod_job_dir" / "normalized_input"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    output_path = normalized_dir / f"{input_path.stem}_normalized.mp4"
    if output_path.exists() and not force:
        write_audit(audit_path, "normalize_reuse", output=output_path, reason=reason)
        return output_path, True, reason

    print(f"[phase-start] normalize_input: {reason} {input_path}", flush=True)
    start = time.perf_counter()
    vf = build_normalization_filter(media, reason)
    input_options = ["-fflags", "+genpts"]
    if stream_rotation_degrees(dict(media.get("stream") or {})):
        input_options.extend(["-display_rotation", "0"])
    input_options.append("-noautorotate")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_options,
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        "-metadata:s:v:0",
        "rotate=0",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_capture(command, audit_path=audit_path, label="normalize_input")
    print(f"[phase-done] normalize_input: elapsed={time.perf_counter() - start:.1f}s output={output_path}", flush=True)
    output_media = ffprobe_media(output_path, audit_path=audit_path)
    write_audit(
        audit_path,
        "normalize_done",
        input=input_path,
        output=output_path,
        reason=reason,
        elapsed_sec=time.perf_counter() - start,
        output_media=output_media,
    )
    return output_path, True, reason


def replace_option_value(command: list[str], option: str, value: str) -> list[str]:
    updated = list(command)
    try:
        index = updated.index(option)
    except ValueError:
        updated.extend([option, value])
        return updated
    if index + 1 >= len(updated):
        raise RuntimeError(f"Missing value after {option}")
    updated[index + 1] = value
    return updated


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_existing(dst)
    rel_src = os.path.relpath(src, start=dst.parent)
    try:
        dst.symlink_to(rel_src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return dst


def parse_polygons(value: object) -> list[np.ndarray]:
    if value in (None, "", []):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list) or not value:
        return []

    polygons: list[np.ndarray] = []
    items = value
    if items and all(isinstance(x, (int, float)) for x in items):
        items = [items]
    for item in items:
        if not isinstance(item, list) or not item:
            continue
        arr = np.asarray(item, dtype=np.float32)
        if arr.ndim == 1 and arr.size >= 6:
            arr = arr.reshape(-1, 2)
        elif arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
            arr = arr.reshape(-1, 2)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            continue
        if len(arr) >= 3:
            polygons.append(arr)
    return polygons


def polygons_from_detection(det: dict[str, Any]) -> list[np.ndarray]:
    return parse_polygons(det.get("polygons") or det.get("segmentation"))


def detection_items(frame_record: dict[str, Any]) -> list[dict[str, Any]]:
    items = frame_record.get("detections")
    if items is None:
        items = frame_record.get("instances")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def fill_polygons(frame: np.ndarray, polygons: list[np.ndarray], color: tuple[int, int, int], alpha: float) -> None:
    if not polygons:
        return
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = [np.round(poly).astype(np.int32).reshape(-1, 1, 2) for poly in polygons]
    cv2.fillPoly(mask, pts, 1)
    idx = mask > 0
    if np.any(idx):
        color_arr = np.asarray(color, dtype=np.float32)
        frame[idx] = np.rint(frame[idx].astype(np.float32) * (1.0 - alpha) + color_arr * alpha).clip(0, 255).astype(np.uint8)


def draw_polygons(frame: np.ndarray, polygons: list[np.ndarray], color: tuple[int, int, int], thickness: int) -> None:
    for poly in polygons:
        cv2.polylines(frame, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)


def polygon_anchor(polygons: list[np.ndarray], width: int, height: int) -> tuple[int, int]:
    if not polygons:
        return 8, 24
    points = np.concatenate(polygons, axis=0)
    x = int(np.clip(points[:, 0].min(), 4, max(4, width - 120)))
    y = int(np.clip(points[:, 1].min() - 6, 24, max(24, height - 8)))
    return x, y


def draw_label(frame: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    global _LABEL_FONT
    if _LABEL_FONT is None:
        for path in LABEL_FONT_CANDIDATES:
            if path.exists():
                _LABEL_FONT = ImageFont.truetype(str(path), 16)
                break
        if _LABEL_FONT is None:
            _LABEL_FONT = ImageFont.load_default()

    bbox = _LABEL_FONT.getbbox(text)
    tw = int(bbox[2] - bbox[0])
    th = int(bbox[3] - bbox[1])
    pad_x = 4
    pad_y = 4
    x, y = anchor
    x = int(np.clip(x, 2, max(2, frame.shape[1] - tw - pad_x * 2 - 2)))
    y = int(np.clip(y, th + pad_y * 2, max(th + pad_y * 2, frame.shape[0] - 2)))
    top = y - th - pad_y * 2
    bottom = y + pad_y
    right = x + tw + pad_x * 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    bg_rgb = tuple(reversed(TEXT_BG_BGR))
    fg_rgb = tuple(reversed(TEXT_FG_BGR))
    border_rgb = tuple(reversed(color))
    draw.rectangle((x, top, right, bottom), fill=bg_rgb, outline=border_rgb, width=1)
    draw.text((x + pad_x, top + pad_y - bbox[1]), text, font=_LABEL_FONT, fill=fg_rgb)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def open_writer(output_video: Path, width: int, height: int, fps: float, encoder: str) -> subprocess.Popen:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.exists():
        output_video.unlink()
    codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20"] if encoder == "nvenc" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    command = [
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
        *codec,
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def close_writer(proc: subprocess.Popen) -> None:
    if proc.stdin is not None:
        proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
    code = proc.wait()
    if proc.stderr is not None:
        proc.stderr.close()
    if code != 0:
        raise RuntimeError(f"ffmpeg encode failed with code {code}: {stderr}")


def render_with_fallback(render_func, *, encoder: str) -> None:
    try:
        render_func(encoder)
    except Exception as exc:
        if encoder != "nvenc":
            raise
        print(f"[warn] nvenc overlay encode failed; retrying with cpu: {exc}", flush=True)
        render_func("cpu")


def render_raw_overlay(video_path: Path, jsonl_path: Path, output_video: Path, *, encoder: str, frame_limit: int | None = None) -> None:
    def _render(active_encoder: str) -> None:
        width, height, fps = video_meta(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        proc = open_writer(output_video, width, height, fps, active_encoder)
        assert proc.stdin is not None
        written = 0
        try:
            with jsonl_path.open("r", encoding="utf-8") as jf:
                for line in jf:
                    if frame_limit is not None and written >= frame_limit:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    obj = json.loads(line)
                    for det in detection_items(obj):
                        fill_polygons(frame, polygons_from_detection(det), RAW_COLOR_BGR, 0.42)
                    proc.stdin.write(frame.tobytes())
                    written += 1
                    if written % 300 == 0:
                        print(f"  rendered raw {written}", flush=True)
        finally:
            cap.release()
            close_writer(proc)

    print(f"[phase-start] raw_overlay: {output_video}", flush=True)
    render_with_fallback(_render, encoder=encoder)
    print(f"[phase-done] raw_overlay: output={output_video}", flush=True)


def sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}


def load_track_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    conn = sqlite3.connect(str(path))
    try:
        tables = {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}
        if "tracks" in tables:
            columns = sqlite_columns(conn, "tracks")
            if {"track_id", "label"}.issubset(columns):
                for track_id, label in conn.execute("select track_id, label from tracks"):
                    if label is not None:
                        labels[str(track_id)] = str(label)
        if not labels and "masks" in tables and "label" in sqlite_columns(conn, "masks"):
            for track_id, label in conn.execute("select track_id, max(label) from masks where label is not null group by track_id"):
                if label is not None:
                    labels[str(track_id)] = str(label)
    finally:
        conn.close()
    return labels


class FrameSqliteReader:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.conn: sqlite3.Connection | None = None
        self.has_label = False
        if path is not None and path.exists():
            self.conn = sqlite3.connect(str(path))
            self.conn.row_factory = sqlite3.Row
            self.has_label = "label" in sqlite_columns(self.conn, "masks")

    def rows_for_frame(self, frame: int) -> list[dict[str, Any]]:
        if self.conn is None:
            return []
        label_expr = "label" if self.has_label else "NULL as label"
        rows: list[dict[str, Any]] = []
        for row in self.conn.execute(
            f"select track_id, polygons, {label_expr} from masks where frame = ? order by track_id",
            (int(frame),),
        ):
            rows.append(
                {
                    "track_id": str(row["track_id"]),
                    "label": None if row["label"] is None else str(row["label"]),
                    "polygons": parse_polygons(row["polygons"]),
                }
            )
        return rows

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def render_sqlite_overlay(
    video_path: Path,
    tracked_sqlite: Path | None,
    pred_sqlite: Path,
    output_video: Path,
    *,
    mode: str,
    encoder: str,
    frame_limit: int | None = None,
) -> None:
    def _render(active_encoder: str) -> None:
        width, height, fps = video_meta(video_path)
        labels_by_track = {}
        if tracked_sqlite is not None and tracked_sqlite.exists():
            labels_by_track.update(load_track_labels(tracked_sqlite))
        labels_by_track.update(load_track_labels(pred_sqlite))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        proc = open_writer(output_video, width, height, fps, active_encoder)
        assert proc.stdin is not None
        raw_reader = FrameSqliteReader(tracked_sqlite)
        pred_reader = FrameSqliteReader(pred_sqlite)
        frame_idx = 0
        try:
            while True:
                if frame_limit is not None and frame_idx >= frame_limit:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                raw_entries = raw_reader.rows_for_frame(frame_idx)
                pred_entries = pred_reader.rows_for_frame(frame_idx)
                if mode == "detailed":
                    for entry in raw_entries:
                        fill_polygons(frame, entry["polygons"], (255, 255, 255), 0.22)
                        draw_polygons(frame, entry["polygons"], ORIGINAL_COLOR_BGR, 1)
                    for entry in pred_entries:
                        polygons = entry["polygons"]
                        draw_polygons(frame, polygons, POST_COLOR_BGR, 3)
                        label = labels_by_track.get(str(entry["track_id"]), entry.get("label") or "-")
                        text = f"ID:{entry['track_id']} {label}"
                        draw_label(frame, text, polygon_anchor(polygons, width, height), POST_COLOR_BGR)
                else:
                    for entry in pred_entries:
                        fill_polygons(frame, entry["polygons"], POST_COLOR_BGR, 0.45)
                proc.stdin.write(frame.tobytes())
                frame_idx += 1
                if frame_idx % 300 == 0:
                    print(f"  rendered {mode} {frame_idx}", flush=True)
        finally:
            raw_reader.close()
            pred_reader.close()
            cap.release()
            close_writer(proc)

    print(f"[phase-start] {mode}_overlay: {output_video}", flush=True)
    render_with_fallback(_render, encoder=encoder)
    print(f"[phase-done] {mode}_overlay: output={output_video}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def organize_outputs(
    *,
    run_dir: Path,
    original_input: Path,
    processed_input: Path,
    normalized: bool,
    normalization_reason: str,
    keep_normalized_input: bool,
    overlay_mode: str,
    raw_overlay: bool,
    encoder: str,
    frame_limit: int | None,
) -> None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    artifacts = summary.get("artifacts", {})
    postprocess = summary.get("postprocess") or {}

    job_dir = run_dir / "sod_job_dir"
    layout = {
        "integrated_overlay": run_dir / "統合マスクオーバーレイ",
        "detailed_overlay": run_dir / "詳細オーバーレイ",
        "raw_overlay": run_dir / "AI生成カバーオーバーレイ",
        "final_sqlite": run_dir / "最終SQLite",
        "raw_sqlite": run_dir / "推論生SQLite",
        "jsonl": run_dir / "jsonl",
        "logs": run_dir / "logs",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    detector_jsonl = Path(str(artifacts.get("detector_jsonl") or artifacts.get("dinov3_jsonl") or ""))
    if detector_jsonl.exists():
        link_or_copy(detector_jsonl, layout["jsonl"] / detector_jsonl.name)
    detector_summary = Path(str(artifacts.get("detector_summary") or artifacts.get("dinov3_summary") or ""))
    if detector_summary.exists():
        link_or_copy(detector_summary, layout["jsonl"] / f"{detector_summary.parent.name}_summary.json")

    if (run_dir / "postprocess").exists():
        link_or_copy(run_dir / "postprocess", run_dir / "postprocessed")

    tracked_sqlite_raw = postprocess.get("tracked_sqlite") or postprocess.get("tracked_sqlite_link")
    tracked_sqlite = Path(str(tracked_sqlite_raw)) if tracked_sqlite_raw else None
    if tracked_sqlite is not None and tracked_sqlite.exists():
        link_or_copy(tracked_sqlite, layout["raw_sqlite"] / tracked_sqlite.name)

    prediction_links = postprocess.get("prediction_sqlite_links") or {}
    overlay_outputs: dict[str, str] = {}
    sqlite_outputs: dict[str, str] = {}
    if isinstance(prediction_links, dict):
        for label, value in sorted(prediction_links.items()):
            pred_sqlite = Path(str(value))
            if not pred_sqlite.exists():
                continue
            final_sqlite = link_or_copy(pred_sqlite, layout["final_sqlite"] / f"{label}_predictions.sqlite")
            sqlite_outputs[str(label)] = str(final_sqlite)
            overlay_modes = ["detailed", "simple"] if overlay_mode == "both" else [overlay_mode]
            for mode in overlay_modes:
                if mode not in {"detailed", "simple"}:
                    continue
                folder = layout["detailed_overlay"] if mode == "detailed" else layout["integrated_overlay"]
                output_video = folder / f"{label}_{mode}.mp4"
                render_sqlite_overlay(
                    processed_input,
                    tracked_sqlite,
                    pred_sqlite,
                    output_video,
                    mode=mode,
                    encoder=encoder,
                    frame_limit=frame_limit,
                )
                overlay_outputs[f"{label}_{mode}"] = str(output_video)

    if raw_overlay and detector_jsonl.exists():
        output_video = layout["raw_overlay"] / f"{detector_jsonl.stem}_ai_raw_mask.mp4"
        render_raw_overlay(processed_input, detector_jsonl, output_video, encoder=encoder, frame_limit=frame_limit)
        overlay_outputs["ai_raw_mask"] = str(output_video)

    normalized_input_removed = False
    if normalized and not keep_normalized_input and processed_input.exists():
        processed_input.unlink()
        normalized_input_removed = True

    final_summary = {
        "original_input": str(original_input),
        "processed_input": str(processed_input),
        "normalized_input": bool(normalized),
        "normalization_reason": normalization_reason,
        "normalized_input_retained": bool(normalized and not normalized_input_removed),
        "normalized_input_removed": bool(normalized_input_removed),
        "run_dir": str(run_dir),
        "frame_limit": frame_limit,
        "pipeline_summary": str(summary_path),
        "detector_jsonl": str(detector_jsonl) if detector_jsonl.exists() else None,
        "tracked_sqlite": None if tracked_sqlite is None else str(tracked_sqlite),
        "final_sqlite": sqlite_outputs,
        "overlays": overlay_outputs,
        "folders": {name: str(path) for name, path in layout.items()},
    }
    output_audit = build_output_audit(
        detector_jsonl=detector_jsonl,
        tracked_sqlite=tracked_sqlite,
        sqlite_outputs=sqlite_outputs,
        overlay_outputs=overlay_outputs,
        pipeline_summary=summary,
    )
    final_summary["output_audit"] = output_audit
    (run_dir / "最終成果物.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "job_manifest.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (layout["logs"] / "job_audit_summary.json").write_text(
        json.dumps(output_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    pipeline_command = strip_remainder(list(args.pipeline_command))
    if not pipeline_command:
        raise RuntimeError("Missing pipeline command after --")
    frame_limit = extract_max_frames(pipeline_command)

    run_dir = args.output_root.expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "ui_job.log"
    audit_path = run_dir / "logs" / "audit.jsonl"

    try:
        write_audit(
            audit_path,
            "job_start",
            args={key: value for key, value in vars(args).items() if key != "pipeline_command"},
            pipeline_command=pipeline_command,
            runtime=runtime_snapshot(pipeline_command),
            cwd=str(Path.cwd()),
        )
        original_input = args.input.expanduser().resolve()
        processed_input, normalized, reason = normalize_input_if_needed(original_input, run_dir, force=args.force, audit_path=audit_path)
        pipeline_command = replace_option_value(pipeline_command, "--input", str(processed_input))

        return_code = run_streamed(pipeline_command, cwd=Path.cwd(), log_path=log_path, audit_path=audit_path, label="integrated_pipeline")
        if return_code != 0:
            write_audit(audit_path, "job_failed", stage="integrated_pipeline", returncode=return_code)
            return return_code

        organize_outputs(
            run_dir=run_dir,
            original_input=original_input,
            processed_input=processed_input,
            normalized=normalized,
            normalization_reason=reason,
            keep_normalized_input=bool(args.keep_normalized_input),
            overlay_mode=args.overlay_mode,
            raw_overlay=bool(args.raw_overlay),
            encoder=str(args.encoder),
            frame_limit=frame_limit,
        )
        write_audit(audit_path, "job_done", final_summary=run_dir / "最終成果物.json")
        print(f"[ui-job] arranged outputs: {run_dir / '最終成果物.json'}", flush=True)
        return 0
    except BaseException as exc:
        write_audit(audit_path, "job_exception", error=repr(exc), traceback=traceback.format_exc())
        print(f"[ui-job-error] {exc}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
