"""
Simple video-to-SQLite inference entrypoint.

Defaults are tuned for the current VHF model and the latest requested use case:
Head/Face only, stricter ByteTrack post-processing, and SQLite track output.

Example:
    .venv/bin/python tools/inference/video_to_sqlite_simple.py \
        -i /path/to/video.mp4

The output DB is created under outputs/sqlite/ unless -o is provided.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from video_sqlite_inf import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, process_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VHF RT-DETRv2 + ByteTrack on a video and write SQLite."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output SQLite path. Defaults to outputs/sqlite/<video_stem>_head_face_tracks.sqlite.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Head", "Face"],
        help="Classes to store. Default: Head Face.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("tracks", "detections", "both"),
        default="tracks",
        help="Store tracked boxes, raw detections, or both. Default: tracks.",
    )
    parser.add_argument("--conf-thr", type=float, default=0.50, help="Draw/store threshold.")
    parser.add_argument("--new-track-thr", type=float, default=0.55, help="ByteTrack new-track threshold.")
    parser.add_argument("--low-thr", type=float, default=0.15, help="ByteTrack low threshold for existing tracks.")
    parser.add_argument("--track-min-hits", type=int, default=5, help="Minimum hits before emitting a track.")
    parser.add_argument("--nms-iou-thr", type=float, default=0.55, help="Class-wise NMS IoU threshold.")
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def default_output_path(input_path: str) -> str:
    stem = Path(input_path).stem
    return str(REPO_ROOT / "outputs" / "sqlite" / f"{stem}_head_face_tracks.sqlite")


def to_full_args(args: argparse.Namespace) -> Namespace:
    output = args.output or default_output_path(args.input)
    return Namespace(
        input=args.input,
        output=output,
        config=args.config,
        resume=args.resume,
        device=args.device,
        batch_size=args.batch_size,
        size=None,
        precision="auto",
        compile=not args.no_compile,
        no_channels_last=False,
        no_tf32=False,
        warmup=3,
        classes=args.classes,
        conf_thr=args.conf_thr,
        det_thr=None,
        nms_iou_thr=args.nms_iou_thr,
        max_detections=300,
        max_area_ratio=1.0,
        tracker="bytetrack",
        track=False,
        track_iou_thr=0.35,
        track_max_miss=12,
        track_min_hits=args.track_min_hits,
        bytetrack_high_thr=args.conf_thr,
        bytetrack_low_thr=args.low_thr,
        bytetrack_new_thr=args.new_track_thr,
        bytetrack_match_thr=0.80,
        bytetrack_buffer=30,
        smooth_alpha=0.7,
        score_alpha=0.8,
        output_mode=args.output_mode,
        max_frames=args.max_frames,
        progress_interval=args.progress_interval,
        profile=args.profile,
        append=False,
        safe_sqlite=False,
    )


def main():
    args = parse_args()
    process_video(to_full_args(args))


if __name__ == "__main__":
    main()
