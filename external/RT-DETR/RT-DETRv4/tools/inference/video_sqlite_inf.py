"""
Run RT-DETR/RT-DETRv2 detection on a video and write results to SQLite.

This reuses the video inference path from video_overlay_inf.py, but skips
rendering and video encoding. The output database is intended for downstream
analysis: each row has frame index, timestamp, class, score, box coordinates,
and optionally a ByteTrack/Simple tracker id.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from video_overlay_inf import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    LABEL_NAMES,
    ByteTracker,
    ProfileMeter,
    SimpleTracker,
    build_model,
    configure_torch,
    filter_detections,
    parse_class_filter,
    resolve_existing_path,
    run_batch,
    select_precision,
    warmup_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RT-DETR video inference and write detections/tracks to SQLite."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path.")
    parser.add_argument("-o", "--output", required=True, help="Output SQLite path.")
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Model config path. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument("-d", "--device", default="cuda:0", help="Inference device.")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Frames per forward pass.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Override eval input size. Defaults to eval_spatial_size in the config.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="CUDA precision.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile(mode='reduce-overhead'). Best for long videos.",
    )
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class filter by id or name, e.g. --classes Head Face.",
    )
    parser.add_argument("--conf-thr", type=float, default=0.35)
    parser.add_argument(
        "--det-thr",
        type=float,
        default=None,
        help="Detection threshold before tracker/NMS. For ByteTrack defaults to low threshold.",
    )
    parser.add_argument("--nms-iou-thr", type=float, default=0.65)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--max-area-ratio", type=float, default=1.0)
    parser.add_argument(
        "--tracker",
        choices=("none", "bytetrack", "simple"),
        default="none",
        help="Optional tracker backend.",
    )
    parser.add_argument("--track", action="store_true", help="Alias for --tracker bytetrack.")
    parser.add_argument("--track-iou-thr", type=float, default=0.35)
    parser.add_argument("--track-max-miss", type=int, default=12)
    parser.add_argument("--track-min-hits", type=int, default=2)
    parser.add_argument("--bytetrack-high-thr", type=float, default=None)
    parser.add_argument("--bytetrack-low-thr", type=float, default=0.10)
    parser.add_argument("--bytetrack-new-thr", type=float, default=None)
    parser.add_argument("--bytetrack-match-thr", type=float, default=0.80)
    parser.add_argument("--bytetrack-buffer", type=int, default=30)
    parser.add_argument("--smooth-alpha", type=float, default=0.7)
    parser.add_argument("--score-alpha", type=float, default=0.8)
    parser.add_argument(
        "--output-mode",
        choices=("detections", "tracks", "both"),
        default="detections",
        help="Rows to store. tracks requires --tracker.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional cap on processed frames.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N frames. Use 0 to disable.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing database instead of replacing it.",
    )
    parser.add_argument(
        "--safe-sqlite",
        action="store_true",
        help="Use safer SQLite pragmas instead of faster bulk-load pragmas.",
    )
    return parser.parse_args()


def open_database(path: Path, append: bool, safe_sqlite: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not append:
        path.unlink()

    conn = sqlite3.connect(str(path))
    if safe_sqlite:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    else:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS frames (
            frame_index INTEGER PRIMARY KEY,
            timestamp_sec REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_index INTEGER NOT NULL,
            timestamp_sec REAL NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            score REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            box_width REAL NOT NULL,
            box_height REAL NOT NULL,
            track_id INTEGER,
            source TEXT NOT NULL,
            FOREIGN KEY(frame_index) REFERENCES frames(frame_index)
        );
        """
    )


def write_metadata(conn: sqlite3.Connection, values: dict):
    rows = [(str(key), json.dumps(value, ensure_ascii=False)) for key, value in values.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        rows,
    )


def create_indexes(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_index);
        CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id);
        CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_id);
        CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source);
        """
    )


def make_tracker(args: argparse.Namespace, fps: float, tracker_backend: str):
    if tracker_backend == "simple":
        return SimpleTracker(
            iou_thr=args.track_iou_thr,
            max_miss=args.track_max_miss,
            min_hits=args.track_min_hits,
            smooth_alpha=args.smooth_alpha,
            score_alpha=args.score_alpha,
        )
    if tracker_backend == "bytetrack":
        high_thr = args.conf_thr if args.bytetrack_high_thr is None else args.bytetrack_high_thr
        new_thr = args.conf_thr if args.bytetrack_new_thr is None else args.bytetrack_new_thr
        return ByteTracker(
            high_thresh=high_thr,
            low_thresh=args.bytetrack_low_thr,
            new_track_thresh=new_thr,
            match_thresh=args.bytetrack_match_thr,
            track_buffer=args.bytetrack_buffer,
            min_hits=args.track_min_hits,
            frame_rate=fps,
        )
    return None


def detection_rows(
    frame_index: int,
    timestamp_sec: float,
    labels,
    boxes,
    scores,
    source: str,
    track_ids: Sequence[int | None] | None = None,
):
    rows = []
    if track_ids is None:
        track_ids = [None] * len(scores)

    for label, box, score, track_id in zip(labels.tolist(), boxes.tolist(), scores.tolist(), track_ids):
        class_id = int(label)
        x1, y1, x2, y2 = [float(value) for value in box]
        rows.append(
            (
                frame_index,
                timestamp_sec,
                class_id,
                LABEL_NAMES.get(class_id, str(class_id)),
                float(score),
                x1,
                y1,
                x2,
                y2,
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                None if track_id is None else int(track_id),
                source,
            )
        )
    return rows


def track_rows(frame_index: int, timestamp_sec: float, tracks):
    rows = []
    for track in tracks:
        class_id = int(track.label)
        x1, y1, x2, y2 = [float(value) for value in track.box.tolist()]
        rows.append(
            (
                frame_index,
                timestamp_sec,
                class_id,
                LABEL_NAMES.get(class_id, str(class_id)),
                float(track.score),
                x1,
                y1,
                x2,
                y2,
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                int(track.track_id),
                "track",
            )
        )
    return rows


def insert_batch(conn: sqlite3.Connection, frame_rows: list[tuple], detection_rows_: list[tuple]):
    if frame_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO frames(frame_index, timestamp_sec, width, height) VALUES (?, ?, ?, ?)",
            frame_rows,
        )
    if detection_rows_:
        conn.executemany(
            """
            INSERT INTO detections(
                frame_index, timestamp_sec, class_id, class_name, score,
                x1, y1, x2, y2, box_width, box_height, track_id, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            detection_rows_,
        )


def process_video(args: argparse.Namespace):
    import cv2

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.output_mode in {"tracks", "both"} and args.tracker == "none" and not args.track:
        raise ValueError("--output-mode tracks/both requires --tracker bytetrack/simple or --track.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")

    config_path = resolve_existing_path(args.config)
    checkpoint_path = resolve_existing_path(args.resume)
    input_path = resolve_existing_path(args.input)
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    device = torch.device(args.device)
    precision_dtype, precision_name = select_precision(device, args.precision)
    autocast_dtype = precision_dtype if device.type == "cuda" and precision_name != "fp32" else None
    channels_last = not args.no_channels_last
    configure_torch(device, enable_tf32=not args.no_tf32)

    model, input_size = build_model(
        config_path,
        checkpoint_path,
        device,
        args.size,
        use_channels_last=channels_last,
        use_compile=args.compile,
    )
    warmup_model(
        model,
        input_size,
        args.batch_size,
        device,
        precision_dtype,
        autocast_dtype,
        args.warmup,
        channels_last,
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker_backend = "bytetrack" if args.track and args.tracker == "none" else args.tracker
    tracker = make_tracker(args, fps, tracker_backend)
    class_filter = parse_class_filter(args.classes)
    if tracker_backend == "bytetrack":
        det_thr = args.bytetrack_low_thr if args.det_thr is None else args.det_thr
    else:
        det_thr = args.conf_thr if args.det_thr is None else args.det_thr

    conn = open_database(output_path, args.append, args.safe_sqlite)
    init_schema(conn)
    write_metadata(
        conn,
        {
            "input": str(input_path),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "classes": args.classes,
            "class_names": LABEL_NAMES,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames_reported": total_frames,
            "input_size": input_size,
            "batch_size": args.batch_size,
            "precision": precision_name,
            "compile": args.compile,
            "conf_thr": args.conf_thr,
            "det_thr": det_thr,
            "nms_iou_thr": args.nms_iou_thr,
            "tracker": tracker_backend,
            "track_min_hits": args.track_min_hits,
            "output_mode": args.output_mode,
        },
    )
    conn.commit()

    print(
        "sqlite inference:",
        f"input={input_path}",
        f"output={output_path}",
        f"frames={total_frames or 'unknown'}",
        f"fps={fps:.3f}",
        f"size={input_size[0]}x{input_size[1]}",
        f"batch={args.batch_size}",
        f"precision={precision_name}",
        f"tracker={tracker_backend}",
        f"mode={args.output_mode}",
    )

    frame_idx = 0
    pending_frames = []
    pending_indices = []
    profile = ProfileMeter() if args.profile else None
    start = time.perf_counter()
    written_detections = 0
    read_time = 0.0
    sqlite_time = 0.0

    def flush_pending():
        nonlocal frame_idx, written_detections, sqlite_time
        if not pending_frames:
            return

        valid_count = len(pending_frames)
        batch_frames = pending_frames
        if args.compile and valid_count < args.batch_size:
            batch_frames = pending_frames + [pending_frames[-1]] * (args.batch_size - valid_count)

        labels_batch, boxes_batch, scores_batch = run_batch(
            model,
            batch_frames,
            input_size,
            device,
            precision_dtype,
            autocast_dtype,
            channels_last,
            profile=profile,
            sync_cuda=args.profile,
        )

        frame_rows = []
        det_rows = []
        for local_idx, frame_index in enumerate(pending_indices):
            timestamp_sec = frame_index / fps
            frame_rows.append((frame_index, timestamp_sec, width, height))

            t_filter = time.perf_counter()
            labels_f, boxes_f, scores_f = filter_detections(
                labels_batch[local_idx],
                boxes_batch[local_idx],
                scores_batch[local_idx],
                det_thr=det_thr,
                nms_iou_thr=args.nms_iou_thr,
                max_detections=args.max_detections,
                max_area_ratio=args.max_area_ratio,
                class_filter=class_filter,
                frame_area=float(width * height),
            )
            if profile is not None:
                profile.add("filter_nms", time.perf_counter() - t_filter)
                profile.add_detections(int(len(scores_batch[local_idx])), int(len(scores_f)))

            if args.output_mode in {"detections", "both"}:
                det_keep = scores_f >= args.conf_thr
                det_rows.extend(
                    detection_rows(
                        frame_index,
                        timestamp_sec,
                        labels_f[det_keep],
                        boxes_f[det_keep],
                        scores_f[det_keep],
                        source="detection",
                    )
                )

            if tracker is not None:
                tracks = tracker.update(labels_f, boxes_f, scores_f)
                if args.output_mode in {"tracks", "both"}:
                    det_rows.extend(track_rows(frame_index, timestamp_sec, tracks))

        t_sqlite = time.perf_counter()
        insert_batch(conn, frame_rows, det_rows)
        conn.commit()
        sqlite_time += time.perf_counter() - t_sqlite
        written_detections += len(det_rows)

        pending_frames.clear()
        pending_indices.clear()

    try:
        while True:
            t_read = time.perf_counter()
            ok, frame_bgr = cap.read()
            read_time += time.perf_counter() - t_read
            if not ok:
                break
            if args.max_frames >= 0 and frame_idx >= args.max_frames:
                break

            pending_frames.append(frame_bgr)
            pending_indices.append(frame_idx)
            frame_idx += 1

            if len(pending_frames) >= args.batch_size:
                flush_pending()

            if args.progress_interval > 0 and frame_idx > 0 and frame_idx % args.progress_interval == 0:
                elapsed = max(time.perf_counter() - start, 1e-6)
                progress = f"{frame_idx}"
                if total_frames > 0:
                    progress += f"/{total_frames}"
                print(
                    f"processed {progress} frames, "
                    f"rows={written_detections}, throughput={frame_idx / elapsed:.2f} fps"
                )

        flush_pending()
        create_indexes(conn)
        write_metadata(
            conn,
            {
                "processed_frames": frame_idx,
                "written_detection_rows": written_detections,
                "elapsed_sec": time.perf_counter() - start,
            },
        )
        conn.commit()
    finally:
        cap.release()
        conn.close()

    elapsed = max(time.perf_counter() - start, 1e-6)
    print(f"saved sqlite to: {output_path}")
    print(f"processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx / elapsed:.2f} fps)")
    print(f"wrote {written_detections} detection rows ({written_detections / max(frame_idx, 1):.2f} rows/frame)")
    print(f"read_time={read_time:.2f}s sqlite_time={sqlite_time:.2f}s")
    if profile is not None:
        profile.add("read", read_time, frame_idx)
        profile.add("sqlite", sqlite_time, frame_idx)
        profile.print_report(frame_idx, elapsed)


def main():
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
