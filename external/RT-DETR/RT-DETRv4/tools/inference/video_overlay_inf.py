"""
Run RT-DETR/RT-DETRv2 detection on a video and write an overlay video.

The default config/checkpoint target the 3-class VHF model trained in this
workspace:

    0: VisibleBody
    1: Head
    2: Face

The preprocessing intentionally matches the final training setup: aspect-ratio
preserving letterbox resize to eval_spatial_size, not square warping.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import batched_nms, box_iou


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

if REPO_ROOT.parent.name == "RT-DETR" and REPO_ROOT.parent.parent.name == "external":
    INTEGRATION_ROOT = REPO_ROOT.parents[2]
else:
    INTEGRATION_ROOT = REPO_ROOT

from engine.core import YAMLConfig  # noqa: E402


DEFAULT_CONFIG = "configs/rtv2/rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml"
_DEFAULT_TRAINING_CHECKPOINT = (
    REPO_ROOT / "outputs" / "pth" / "rtv2-r18-vhf-512x896-80e-bs16-20260518-010638" / "best_stg1.pth"
)
_DEFAULT_PORTABLE_CHECKPOINT = INTEGRATION_ROOT / "checkpoints" / "rtdetr" / "head_face_best_stg1.pth"
if INTEGRATION_ROOT != REPO_ROOT:
    DEFAULT_CHECKPOINT = str(_DEFAULT_PORTABLE_CHECKPOINT)
else:
    DEFAULT_CHECKPOINT = str(_DEFAULT_TRAINING_CHECKPOINT)

LABEL_NAMES = {
    0: "VisibleBody",
    1: "Head",
    2: "Face",
}

# OpenCV uses BGR.
LABEL_COLORS = {
    0: (60, 220, 80),
    1: (255, 170, 40),
    2: (40, 70, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast batched video inference with RT-DETR overlay output."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path.")
    parser.add_argument("-o", "--output", required=True, help="Output video path.")
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
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Frames per forward pass. 64 is the measured throughput default for this R18 512x896 model on the local GPU.",
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
        help="CUDA precision. auto uses fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile(mode='reduce-overhead'). Good for long videos, slower startup.",
    )
    parser.add_argument(
        "--no-channels-last",
        action="store_true",
        help="Disable channels-last tensors on CUDA.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 matmul/convolution on CUDA.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of dummy warmup iterations before reading the video.",
    )
    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.35,
        help="Score threshold for drawing detections.",
    )
    parser.add_argument(
        "--det-thr",
        type=float,
        default=None,
        help="Lower detection threshold before NMS/tracking. Defaults to conf-thr.",
    )
    parser.add_argument(
        "--nms-iou-thr",
        type=float,
        default=0.65,
        help="Class-wise NMS IoU threshold. Use 1.0 to nearly disable NMS.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help="Maximum detections drawn per frame after filtering/NMS.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=1.0,
        help="Drop boxes larger than this fraction of the frame area.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class filter by id or name, e.g. --classes VisibleBody Head Face.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Draw boxes only.",
    )
    parser.add_argument(
        "--hide-scores",
        action="store_true",
        help="Draw labels without scores.",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Skip overlay drawing. Useful for profiling model/video I/O cost.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write an output video. Useful for profiling decode/preprocess/inference/postprocess only.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print a timing breakdown for read/preprocess/inference/filter/draw/write.",
    )
    parser.add_argument(
        "--tracker",
        choices=("none", "bytetrack", "simple"),
        default="none",
        help="Tracker backend. Use bytetrack for MOT-quality temporal association.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Alias for --tracker bytetrack.",
    )
    parser.add_argument(
        "--track-iou-thr",
        type=float,
        default=0.35,
        help="Minimum IoU for the simple tracker association.",
    )
    parser.add_argument(
        "--track-max-miss",
        type=int,
        default=12,
        help="How many missed frames a simple track can survive.",
    )
    parser.add_argument(
        "--track-min-hits",
        type=int,
        default=2,
        help="Minimum matched frames before drawing a track.",
    )
    parser.add_argument(
        "--bytetrack-high-thr",
        type=float,
        default=None,
        help="ByteTrack high-confidence threshold. Defaults to --conf-thr.",
    )
    parser.add_argument(
        "--bytetrack-low-thr",
        type=float,
        default=0.10,
        help="ByteTrack low-confidence threshold used only for matching existing tracks.",
    )
    parser.add_argument(
        "--bytetrack-new-thr",
        type=float,
        default=None,
        help="Minimum score for starting a new ByteTrack track. Defaults to --conf-thr.",
    )
    parser.add_argument(
        "--bytetrack-match-thr",
        type=float,
        default=0.80,
        help="ByteTrack assignment cost threshold. Larger values keep tracks through lower IoU matches.",
    )
    parser.add_argument(
        "--bytetrack-buffer",
        type=int,
        default=30,
        help="Frames a lost ByteTrack track is kept before removal.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="EMA momentum for track box smoothing. Higher is smoother.",
    )
    parser.add_argument(
        "--score-alpha",
        type=float,
        default=0.8,
        help="EMA momentum for track score smoothing.",
    )
    parser.add_argument(
        "--video-codec",
        default="auto",
        help="Output codec. auto tries h264_nvenc, then falls back to mp4v. Examples: mp4v, h264_nvenc, libx264.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="p5",
        help="Preset used by h264_nvenc/libx264/libx265.",
    )
    parser.add_argument(
        "--video-bitrate",
        default=None,
        help="Optional output video bitrate, e.g. 8M or 12000k. Useful when encoder defaults are too low quality.",
    )
    parser.add_argument(
        "--video-maxrate",
        default=None,
        help="Optional maxrate for ffmpeg encoders, e.g. 12M.",
    )
    parser.add_argument(
        "--video-bufsize",
        default=None,
        help="Optional rate-control buffer size for ffmpeg encoders, e.g. 16M.",
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
        default=100,
        help="Print progress every N frames. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_existing_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    repo_candidate = (REPO_ROOT / path).expanduser()
    if repo_candidate.exists():
        return repo_candidate.resolve()
    integration_candidate = (INTEGRATION_ROOT / path).expanduser()
    if integration_candidate.exists():
        return integration_candidate.resolve()
    raise FileNotFoundError(f"Path does not exist: {path}")


def parse_class_filter(values: Sequence[str] | None) -> set[int] | None:
    if not values:
        return None

    name_to_id = {name.lower(): idx for idx, name in LABEL_NAMES.items()}
    class_ids: set[int] = set()
    for value in values:
        lowered = value.lower()
        if lowered in name_to_id:
            class_ids.add(name_to_id[lowered])
            continue
        try:
            class_ids.add(int(value))
        except ValueError as exc:
            valid = ", ".join([str(k) for k in LABEL_NAMES] + list(LABEL_NAMES.values()))
            raise ValueError(f"Unknown class '{value}'. Valid values: {valid}") from exc

    return class_ids


def load_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError(f"Unsupported checkpoint format: {checkpoint_path}")


class DetectionModel(nn.Module):
    def __init__(self, cfg: YAMLConfig):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(
        self,
        images: torch.Tensor,
        orig_sizes: torch.Tensor,
        letterbox_padding: torch.Tensor,
        letterbox_scale: torch.Tensor,
        input_sizes: torch.Tensor,
    ):
        outputs = self.model(images)
        return self.postprocessor(
            outputs,
            orig_sizes,
            letterbox_padding=letterbox_padding,
            letterbox_scale=letterbox_scale,
            input_sizes=input_sizes,
        )


def build_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    size_override: Sequence[int] | None,
    use_channels_last: bool,
    use_compile: bool,
) -> tuple[nn.Module, tuple[int, int]]:
    overrides = {}
    if size_override is not None:
        overrides["eval_spatial_size"] = [int(size_override[0]), int(size_override[1])]

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path), **overrides)

    # Avoid unnecessary ImageNet-pretrain loading when we immediately load the checkpoint.
    if "PResNet" in cfg.yaml_cfg:
        cfg.yaml_cfg["PResNet"]["pretrained"] = False
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    eval_size = cfg.yaml_cfg.get("eval_spatial_size")
    if not eval_size or len(eval_size) != 2:
        raise ValueError("eval_spatial_size must be defined as [height, width].")
    input_size = (int(eval_size[0]), int(eval_size[1]))

    state = load_state_dict(checkpoint_path)
    cfg.model.load_state_dict(state)

    model = DetectionModel(cfg).eval()
    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")

    return model, input_size


def select_precision(device: torch.device, precision: str) -> tuple[torch.dtype | None, str]:
    if device.type != "cuda":
        return None, "fp32"
    if precision == "auto":
        return torch.float16, "fp16"
    if precision == "fp16":
        return torch.float16, "fp16"
    if precision == "bf16":
        return torch.bfloat16, "bf16"
    return None, "fp32"


def configure_torch(device: torch.device, enable_tf32: bool):
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32


@dataclass
class Track:
    track_id: int
    label: int
    box: torch.Tensor
    score: float
    hits: int = 1
    misses: int = 0

    def update(self, label: int, box: torch.Tensor, score: float, smooth_alpha: float, score_alpha: float):
        self.label = label
        self.box = self.box * smooth_alpha + box * (1.0 - smooth_alpha)
        self.score = self.score * score_alpha + score * (1.0 - score_alpha)
        self.hits += 1
        self.misses = 0

    def mark_missed(self):
        self.misses += 1


class SimpleTracker:
    def __init__(
        self,
        iou_thr: float,
        max_miss: int,
        min_hits: int,
        smooth_alpha: float,
        score_alpha: float,
    ):
        self.iou_thr = iou_thr
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.smooth_alpha = smooth_alpha
        self.score_alpha = score_alpha
        self.tracks: List[Track] = []
        self.next_track_id = 1

    def _new_track(self, label: int, box: torch.Tensor, score: float):
        self.tracks.append(Track(self.next_track_id, label, box.clone(), score))
        self.next_track_id += 1

    def update(self, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> List[Track]:
        if len(self.tracks) == 0:
            for label, box, score in zip(labels.tolist(), boxes, scores.tolist()):
                self._new_track(int(label), box, float(score))
            return [track for track in self.tracks if track.hits >= self.min_hits]

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if len(boxes) > 0:
            track_boxes = torch.stack([track.box for track in self.tracks], dim=0)
            ious = box_iou(track_boxes, boxes)
            candidates = []
            for track_idx, track in enumerate(self.tracks):
                for det_idx, det_label in enumerate(labels.tolist()):
                    if track.label != int(det_label):
                        continue
                    iou = float(ious[track_idx, det_idx].item())
                    if iou >= self.iou_thr:
                        candidates.append((iou, track_idx, det_idx))

            candidates.sort(reverse=True)
            for _, track_idx, det_idx in candidates:
                if track_idx in matched_tracks or det_idx in matched_dets:
                    continue
                self.tracks[track_idx].update(
                    int(labels[det_idx].item()),
                    boxes[det_idx],
                    float(scores[det_idx].item()),
                    self.smooth_alpha,
                    self.score_alpha,
                )
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_tracks:
                track.mark_missed()

        for det_idx, (label, box, score) in enumerate(zip(labels.tolist(), boxes, scores.tolist())):
            if det_idx not in matched_dets:
                self._new_track(int(label), box, float(score))

        self.tracks = [track for track in self.tracks if track.misses <= self.max_miss]
        return [
            track
            for track in self.tracks
            if track.hits >= self.min_hits and track.misses <= 1
        ]


def tlbr_to_xyah(tlbr: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = tlbr.astype(np.float32)
    w = max(0.0, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    return np.asarray([x1 + w * 0.5, y1 + h * 0.5, w / h, h], dtype=np.float32)


def xyah_to_tlbr(xyah: np.ndarray) -> np.ndarray:
    x, y, aspect, h = xyah.astype(np.float32)
    w = max(0.0, float(aspect * h))
    h = max(0.0, float(h))
    return np.asarray([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dtype=np.float32)


def bbox_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def linear_assignment_matches(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))
    for row, col in zip(row_indices.tolist(), col_indices.tolist()):
        if cost_matrix[row, col] > thresh:
            continue
        matches.append((row, col))
        unmatched_rows.discard(row)
        unmatched_cols.discard(col)
    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


class ByteKalmanFilter:
    def __init__(self):
        ndim = 4
        dt = 1.0
        self.motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = dt
        self.update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean = np.r_[measurement, np.zeros_like(measurement)].astype(np.float32)
        h = max(float(measurement[3]), 1e-6)
        std = np.asarray([
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * h,
            1e-2,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * h,
            1e-5,
            10 * self.std_weight_velocity * h,
        ], dtype=np.float32)
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1e-6)
        std_pos = np.asarray([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ], dtype=np.float32)
        std_vel = np.asarray([
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ], dtype=np.float32)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        h = max(float(mean[3]), 1e-6)
        std = np.asarray([
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ], dtype=np.float32)
        innovation_cov = np.diag(np.square(std)).astype(np.float32)
        projected_mean = self.update_mat @ mean
        projected_covariance = self.update_mat @ covariance @ self.update_mat.T + innovation_cov
        return projected_mean.astype(np.float32), projected_covariance.astype(np.float32)

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_covariance = self.project(mean, covariance)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_covariance)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_covariance @ kalman_gain.T
        return new_mean.astype(np.float32), new_covariance.astype(np.float32)


class ByteTracklet:
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"

    def __init__(self, tlbr: np.ndarray, score: float, label: int):
        self._tlbr = tlbr.astype(np.float32)
        self.score = float(score)
        self.label = int(label)
        self.track_id = 0
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.state = self.TRACKED
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @property
    def tlbr(self) -> np.ndarray:
        if self.mean is None:
            return self._tlbr.copy()
        return xyah_to_tlbr(self.mean[:4])

    def activate(self, kalman: ByteKalmanFilter, frame_id: int, track_id: int):
        self.mean, self.covariance = kalman.initiate(tlbr_to_xyah(self._tlbr))
        self.track_id = track_id
        self.state = self.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1

    def predict(self, kalman: ByteKalmanFilter):
        if self.mean is None or self.covariance is None:
            return
        mean = self.mean.copy()
        if self.state != self.TRACKED:
            mean[7] = 0
        self.mean, self.covariance = kalman.predict(mean, self.covariance)

    def update(self, detection: "ByteTracklet", frame_id: int):
        if self.mean is None or self.covariance is None:
            raise RuntimeError("ByteTracklet must be activated before update.")
        self.mean, self.covariance = ByteTracker.KALMAN.update(
            self.mean,
            self.covariance,
            tlbr_to_xyah(detection.tlbr),
        )
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = self.TRACKED
        self.is_activated = True
        self.score = detection.score
        self.label = detection.label

    def re_activate(self, detection: "ByteTracklet", frame_id: int):
        self.update(detection, frame_id)

    def mark_lost(self):
        self.state = self.LOST

    def mark_removed(self):
        self.state = self.REMOVED

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            label=self.label,
            box=torch.as_tensor(self.tlbr, dtype=torch.float32),
            score=self.score,
            hits=self.tracklet_len,
            misses=0,
        )


class ByteTracker:
    KALMAN = ByteKalmanFilter()

    def __init__(
        self,
        high_thresh: float,
        low_thresh: float,
        new_track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        min_hits: int,
        frame_rate: float,
    ):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = 0.5
        self.max_time_lost = int(max(1, round(float(track_buffer) * max(frame_rate, 1.0) / 30.0)))
        self.min_hits = int(max(1, min_hits))
        self.frame_id = 0
        self.next_track_id = 1
        self.tracked_stracks: list[ByteTracklet] = []
        self.lost_stracks: list[ByteTracklet] = []
        self.removed_stracks: list[ByteTracklet] = []

    @staticmethod
    def _joint_stracks(a: list[ByteTracklet], b: list[ByteTracklet]) -> list[ByteTracklet]:
        exists = set()
        result = []
        for track in a + b:
            if track.track_id in exists:
                continue
            exists.add(track.track_id)
            result.append(track)
        return result

    @staticmethod
    def _sub_stracks(a: list[ByteTracklet], b: list[ByteTracklet]) -> list[ByteTracklet]:
        removed_ids = {track.track_id for track in b}
        return [track for track in a if track.track_id not in removed_ids]

    @staticmethod
    def _remove_duplicate_stracks(
        tracked: list[ByteTracklet],
        lost: list[ByteTracklet],
    ) -> tuple[list[ByteTracklet], list[ByteTracklet]]:
        if not tracked or not lost:
            return tracked, lost
        ious = bbox_iou_np(
            np.asarray([track.tlbr for track in tracked], dtype=np.float32),
            np.asarray([track.tlbr for track in lost], dtype=np.float32),
        )
        duplicate_pairs = np.argwhere(ious > 0.85)
        tracked_remove = set()
        lost_remove = set()
        for tracked_idx, lost_idx in duplicate_pairs:
            tracked_time = tracked[tracked_idx].frame_id - tracked[tracked_idx].start_frame
            lost_time = lost[lost_idx].frame_id - lost[lost_idx].start_frame
            if tracked_time > lost_time:
                lost_remove.add(int(lost_idx))
            else:
                tracked_remove.add(int(tracked_idx))
        tracked = [track for idx, track in enumerate(tracked) if idx not in tracked_remove]
        lost = [track for idx, track in enumerate(lost) if idx not in lost_remove]
        return tracked, lost

    @staticmethod
    def _iou_distance(
        tracks: list[ByteTracklet],
        detections: list[ByteTracklet],
        fuse_score: bool,
    ) -> np.ndarray:
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        ious = bbox_iou_np(
            np.asarray([track.tlbr for track in tracks], dtype=np.float32),
            np.asarray([det.tlbr for det in detections], dtype=np.float32),
        )
        cost = 1.0 - ious
        label_mismatch = np.asarray(
            [[track.label != det.label for det in detections] for track in tracks],
            dtype=bool,
        )
        cost[label_mismatch] = 1.0
        if fuse_score:
            det_scores = np.asarray([det.score for det in detections], dtype=np.float32)[None, :]
            cost = 1.0 - ious * det_scores
            cost[label_mismatch] = 1.0
        return cost.astype(np.float32)

    def _new_track_id(self) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def _make_detections(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[list[ByteTracklet], list[ByteTracklet]]:
        if len(scores) == 0:
            return [], []
        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        detections = [
            ByteTracklet(box, float(score), int(label))
            for label, box, score in zip(labels_np, boxes_np, scores_np)
            if score >= self.low_thresh
        ]
        high = [det for det in detections if det.score >= self.high_thresh]
        low = [det for det in detections if self.low_thresh <= det.score < self.high_thresh]
        return high, low

    def update(self, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor) -> list[Track]:
        self.frame_id += 1
        activated = []
        re_found = []
        lost = []
        removed = []

        high_dets, low_dets = self._make_detections(labels, boxes, scores)
        tracked = [track for track in self.tracked_stracks if track.state == ByteTracklet.TRACKED]
        track_pool = self._joint_stracks(tracked, self.lost_stracks)
        for track in track_pool:
            track.predict(self.KALMAN)

        cost = self._iou_distance(track_pool, high_dets, fuse_score=True)
        matches, unmatched_track_idx, unmatched_high_idx = linear_assignment_matches(cost, self.match_thresh)
        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            det = high_dets[det_idx]
            if track.state == ByteTracklet.TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                re_found.append(track)

        remaining_tracks = [
            track_pool[idx]
            for idx in unmatched_track_idx
            if track_pool[idx].state == ByteTracklet.TRACKED
        ]
        cost_low = self._iou_distance(remaining_tracks, low_dets, fuse_score=False)
        low_matches, unmatched_remaining_idx, _ = linear_assignment_matches(cost_low, self.second_match_thresh)
        for track_idx, det_idx in low_matches:
            track = remaining_tracks[track_idx]
            track.update(low_dets[det_idx], self.frame_id)
            activated.append(track)

        for idx in unmatched_remaining_idx:
            track = remaining_tracks[idx]
            if track.state != ByteTracklet.LOST:
                track.mark_lost()
                lost.append(track)

        for idx in unmatched_high_idx:
            det = high_dets[idx]
            if det.score < self.new_track_thresh:
                continue
            det.activate(self.KALMAN, self.frame_id, self._new_track_id())
            activated.append(det)

        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [track for track in self.tracked_stracks if track.state == ByteTracklet.TRACKED]
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = self._joint_stracks(self.tracked_stracks, re_found)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = self._sub_stracks(self.lost_stracks, removed)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = self._remove_duplicate_stracks(
            self.tracked_stracks,
            self.lost_stracks,
        )

        return [
            track.to_track()
            for track in self.tracked_stracks
            if track.is_activated and track.tracklet_len >= self.min_hits
        ]


class ProfileMeter:
    def __init__(self):
        self.times: dict[str, float] = {}
        self.units: dict[str, int] = {}
        self.input_detections = 0
        self.output_detections = 0

    def add(self, name: str, seconds: float, units: int = 1):
        self.times[name] = self.times.get(name, 0.0) + seconds
        self.units[name] = self.units.get(name, 0) + units

    def add_detections(self, before: int, after: int):
        self.input_detections += before
        self.output_detections += after

    def print_report(self, frames: int, total_elapsed: float):
        if frames <= 0:
            return
        print("profile summary:")
        for name in [
            "read",
            "preprocess",
            "inference",
            "filter_nms",
            "draw",
            "write",
            "writer_release",
        ]:
            seconds = self.times.get(name, 0.0)
            units = max(self.units.get(name, frames), 1)
            share = seconds / max(total_elapsed, 1e-9) * 100.0
            print(
                f"  {name:14s} {seconds:8.3f}s"
                f"  {seconds / units * 1000.0:7.3f} ms/unit"
                f"  {share:5.1f}%"
            )
        print(
            f"  detections     raw={self.input_detections / frames:.1f}/frame"
            f" drawn={self.output_detections / frames:.1f}/frame"
        )


class OpenCVWriter:
    def __init__(self, writer):
        self.writer = writer

    def write(self, frame_bgr: np.ndarray):
        self.writer.write(frame_bgr)

    def release(self):
        self.writer.release()


class FFmpegWriter:
    def __init__(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
        codec: str,
        preset: str,
        bitrate: str | None,
        maxrate: str | None,
        bufsize: str | None,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
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
            codec,
        ]
        if codec.endswith("_nvenc"):
            cmd += ["-preset", preset, "-pix_fmt", "yuv420p"]
        elif codec in {"libx264", "libx265"}:
            cmd += ["-preset", preset, "-pix_fmt", "yuv420p"]
        if bitrate:
            cmd += ["-b:v", bitrate]
        if maxrate:
            cmd += ["-maxrate", maxrate]
        if bufsize:
            cmd += ["-bufsize", bufsize]
        cmd.append(str(output_path))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr: np.ndarray):
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is not available.")
        if self.process.poll() is not None:
            raise RuntimeError(f"FFmpeg exited early with code {self.process.returncode}.")
        self.process.stdin.write(frame_bgr.tobytes())

    def release(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"FFmpeg writer failed with exit code {code}.")


def ffmpeg_has_encoder(encoder: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return encoder in result.stdout


def resolve_video_codec(codec: str) -> str:
    if codec != "auto":
        return codec
    if ffmpeg_has_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "mp4v"


def make_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec: str,
    preset: str,
    bitrate: str | None,
    maxrate: str | None,
    bufsize: str | None,
):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_codec = resolve_video_codec(codec)
    if len(resolved_codec) == 4 and resolved_codec.isascii():
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*resolved_codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return OpenCVWriter(writer), resolved_codec
        raise RuntimeError(f"Failed to open OpenCV VideoWriter codec={resolved_codec}.")

    return FFmpegWriter(output_path, fps, width, height, resolved_codec, preset, bitrate, maxrate, bufsize), resolved_codec


def letterbox_frame(frame_bgr: np.ndarray, out_h: int, out_w: int, fill: int = 0):
    import cv2

    in_h, in_w = frame_bgr.shape[:2]
    scale = min(out_w / max(1, in_w), out_h / max(1, in_h))
    new_w = max(1, min(out_w, int(round(in_w * scale))))
    new_h = max(1, min(out_h, int(round(in_h * scale))))
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2
    pad_right = out_w - new_w - pad_left
    pad_bottom = out_h - new_h - pad_top

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), fill, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, (pad_left, pad_top, pad_right, pad_bottom)


def make_tensor_batch(
    frames_bgr: Sequence[np.ndarray],
    input_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype | None,
    channels_last: bool,
):
    import cv2

    out_h, out_w = input_size
    batch_np = np.zeros((len(frames_bgr), out_h, out_w, 3), dtype=np.uint8)
    orig_sizes = []
    paddings = []
    scales = []

    for batch_idx, frame in enumerate(frames_bgr):
        h, w = frame.shape[:2]
        scale = min(out_w / max(1, w), out_h / max(1, h))
        new_w = max(1, min(out_w, int(round(w * scale))))
        new_h = max(1, min(out_h, int(round(h * scale))))
        pad_left = (out_w - new_w) // 2
        pad_top = (out_h - new_h) // 2
        pad_right = out_w - new_w - pad_left
        pad_bottom = out_h - new_h - pad_top

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        batch_np[
            batch_idx,
            pad_top:pad_top + new_h,
            pad_left:pad_left + new_w,
            :,
        ] = resized

        orig_sizes.append((w, h))
        paddings.append((pad_left, pad_top, pad_right, pad_bottom))
        scales.append(scale)

    images_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2)
    images_t = images_t.to(device=device, dtype=dtype or torch.float32, non_blocking=True).div_(255.0)
    images_t = images_t.index_select(1, torch.tensor([2, 1, 0], device=device))
    if channels_last and device.type == "cuda":
        images_t = images_t.contiguous(memory_format=torch.channels_last)
    else:
        images_t = images_t.contiguous()

    orig_sizes_t = torch.tensor(orig_sizes, dtype=torch.float32, device=device)
    paddings_t = torch.tensor(paddings, dtype=torch.float32, device=device)
    scales_t = torch.tensor(scales, dtype=torch.float32, device=device)
    input_sizes_t = torch.tensor([(out_w, out_h)] * len(frames_bgr), dtype=torch.float32, device=device)
    return images_t, orig_sizes_t, paddings_t, scales_t, input_sizes_t


def filter_detections(
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    det_thr: float,
    nms_iou_thr: float,
    max_detections: int,
    max_area_ratio: float,
    class_filter: set[int] | None,
    frame_area: float,
):
    labels = labels.detach().cpu()
    boxes = boxes.detach().cpu().float()
    scores = scores.detach().cpu().float()

    keep = scores >= det_thr
    if class_filter is not None:
        class_ids = torch.tensor(sorted(class_filter), dtype=labels.dtype)
        class_keep = (labels[:, None] == class_ids[None, :]).any(dim=1)
        keep &= class_keep

    if max_area_ratio < 1.0:
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
        areas = wh[:, 0] * wh[:, 1]
        keep &= (areas / max(frame_area, 1.0)) <= max_area_ratio

    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    if boxes.numel() == 0:
        return labels, boxes, scores

    if nms_iou_thr < 1.0:
        keep_idx = batched_nms(boxes, scores, labels, nms_iou_thr)
        labels = labels[keep_idx]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

    if max_detections > 0 and len(scores) > max_detections:
        top_idx = torch.argsort(scores, descending=True)[:max_detections]
        labels = labels[top_idx]
        boxes = boxes[top_idx]
        scores = scores[top_idx]

    return labels, boxes, scores


def caption_for(label: int, score: float, hide_labels: bool, hide_scores: bool) -> str:
    if hide_labels and hide_scores:
        return ""
    label_name = LABEL_NAMES.get(label, str(label))
    if hide_labels:
        return f"{score:.2f}"
    if hide_scores:
        return label_name
    return f"{label_name} {score:.2f}"


def draw_box(
    frame_bgr: np.ndarray,
    box: torch.Tensor,
    label: int,
    score: float,
    line_width: int,
    hide_labels: bool,
    hide_scores: bool,
    prefix: str = "",
):
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    color = LABEL_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, line_width)

    caption = caption_for(label, score, hide_labels, hide_scores)
    if prefix and caption:
        caption = f"{prefix} {caption}"
    elif prefix:
        caption = prefix
    if not caption:
        return

    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_y = max(y1, text_h + baseline + 4)
    cv2.rectangle(
        frame_bgr,
        (x1, text_y - text_h - baseline - 4),
        (min(w - 1, x1 + text_w + 6), text_y),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame_bgr,
        caption,
        (x1 + 3, text_y - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    frame_bgr: np.ndarray,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    conf_thr: float,
    line_width: int,
    hide_labels: bool,
    hide_scores: bool,
):
    for label, box, score in zip(labels.tolist(), boxes, scores.tolist()):
        if float(score) < conf_thr:
            continue
        draw_box(frame_bgr, box, int(label), float(score), line_width, hide_labels, hide_scores)


def draw_tracks(
    frame_bgr: np.ndarray,
    tracks: Iterable[Track],
    conf_thr: float,
    line_width: int,
    hide_labels: bool,
    hide_scores: bool,
):
    for track in tracks:
        if track.score < conf_thr:
            continue
        draw_box(
            frame_bgr,
            track.box,
            track.label,
            track.score,
            line_width,
            hide_labels,
            hide_scores,
            prefix=f"#{track.track_id}",
        )


def warmup_model(
    model: nn.Module,
    input_size: tuple[int, int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype | None,
    autocast_dtype: torch.dtype | None,
    iterations: int,
    channels_last: bool,
):
    if iterations <= 0:
        return
    out_h, out_w = input_size
    images = torch.zeros((batch_size, 3, out_h, out_w), dtype=dtype or torch.float32, device=device)
    if channels_last and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    orig_sizes = torch.tensor([(out_w, out_h)] * batch_size, dtype=torch.float32, device=device)
    paddings = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
    scales = torch.ones((batch_size,), dtype=torch.float32, device=device)
    input_sizes = torch.tensor([(out_w, out_h)] * batch_size, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for _ in range(iterations):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                model(images, orig_sizes, paddings, scales, input_sizes)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_batch(
    model: nn.Module,
    frames: Sequence[np.ndarray],
    input_size: tuple[int, int],
    device: torch.device,
    tensor_dtype: torch.dtype | None,
    autocast_dtype: torch.dtype | None,
    channels_last: bool,
    profile: ProfileMeter | None = None,
    sync_cuda: bool = False,
):
    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize(device)
    t_preprocess = time.perf_counter()
    images, orig_sizes, paddings, scales, input_sizes = make_tensor_batch(
        frames,
        input_size,
        device,
        tensor_dtype,
        channels_last,
    )
    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize(device)
    if profile is not None:
        profile.add("preprocess", time.perf_counter() - t_preprocess, len(frames))

    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inference = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            outputs = model(images, orig_sizes, paddings, scales, input_sizes)
    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize(device)
    if profile is not None:
        profile.add("inference", time.perf_counter() - t_inference, len(frames))
    return outputs


def process_video(args: argparse.Namespace):
    import cv2

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
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
    writer = None
    resolved_codec = "none"
    if not args.no_output:
        writer, resolved_codec = make_writer(
            output_path,
            fps,
            width,
            height,
            args.video_codec,
            args.ffmpeg_preset,
            args.video_bitrate,
            args.video_maxrate,
            args.video_bufsize,
        )

    class_filter = parse_class_filter(args.classes)
    tracker_backend = "bytetrack" if args.track and args.tracker == "none" else args.tracker
    bytetrack_high_thr = args.conf_thr if args.bytetrack_high_thr is None else args.bytetrack_high_thr
    bytetrack_new_thr = args.conf_thr if args.bytetrack_new_thr is None else args.bytetrack_new_thr
    if tracker_backend == "bytetrack":
        det_thr = args.bytetrack_low_thr if args.det_thr is None else args.det_thr
    else:
        det_thr = args.conf_thr if args.det_thr is None else args.det_thr

    tracker = None
    if tracker_backend == "simple":
        tracker = SimpleTracker(
            iou_thr=args.track_iou_thr,
            max_miss=args.track_max_miss,
            min_hits=args.track_min_hits,
            smooth_alpha=args.smooth_alpha,
            score_alpha=args.score_alpha,
        )
    elif tracker_backend == "bytetrack":
        tracker = ByteTracker(
            high_thresh=bytetrack_high_thr,
            low_thresh=args.bytetrack_low_thr,
            new_track_thresh=bytetrack_new_thr,
            match_thresh=args.bytetrack_match_thr,
            track_buffer=args.bytetrack_buffer,
            min_hits=args.track_min_hits,
            frame_rate=fps,
        )

    print(
        "video inference:",
        f"input={input_path}",
        f"output={output_path}",
        f"frames={total_frames or 'unknown'}",
        f"fps={fps:.3f}",
        f"size={input_size[0]}x{input_size[1]}",
        f"batch={args.batch_size}",
        f"precision={precision_name}",
        f"codec={resolved_codec}",
        f"tracker={tracker_backend}",
    )

    frame_idx = 0
    start = time.perf_counter()
    profile = ProfileMeter() if args.profile else None
    pending_frames: list[np.ndarray] = []

    def flush_pending():
        nonlocal frame_idx
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

        for frame_bgr, labels, boxes, scores in zip(
            pending_frames,
            labels_batch[:valid_count],
            boxes_batch[:valid_count],
            scores_batch[:valid_count],
        ):
            frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
            t_filter = time.perf_counter()
            labels_f, boxes_f, scores_f = filter_detections(
                labels,
                boxes,
                scores,
                det_thr=det_thr,
                nms_iou_thr=args.nms_iou_thr,
                max_detections=args.max_detections,
                max_area_ratio=args.max_area_ratio,
                class_filter=class_filter,
                frame_area=frame_area,
            )
            if profile is not None:
                profile.add("filter_nms", time.perf_counter() - t_filter)
                profile.add_detections(int(len(scores)), int(len(scores_f)))

            t_draw = time.perf_counter()
            if args.no_draw:
                pass
            elif tracker is not None:
                tracks = tracker.update(labels_f, boxes_f, scores_f)
                draw_tracks(frame_bgr, tracks, args.conf_thr, args.line_width, args.hide_labels, args.hide_scores)
            else:
                draw_detections(
                    frame_bgr,
                    labels_f,
                    boxes_f,
                    scores_f,
                    args.conf_thr,
                    args.line_width,
                    args.hide_labels,
                    args.hide_scores,
                )
            if profile is not None:
                profile.add("draw", time.perf_counter() - t_draw)

            if writer is not None:
                t_write = time.perf_counter()
                writer.write(frame_bgr)
                if profile is not None:
                    profile.add("write", time.perf_counter() - t_write)
            frame_idx += 1

            if args.progress_interval > 0 and frame_idx % args.progress_interval == 0:
                elapsed = max(time.perf_counter() - start, 1e-6)
                fps_now = frame_idx / elapsed
                progress = f"{frame_idx}"
                if total_frames > 0:
                    progress += f"/{total_frames}"
                print(f"processed {progress} frames, throughput={fps_now:.2f} fps")

        pending_frames.clear()

    try:
        while True:
            t_read = time.perf_counter()
            ok, frame_bgr = cap.read()
            if profile is not None:
                profile.add("read", time.perf_counter() - t_read)
            if not ok:
                break
            if args.max_frames >= 0 and frame_idx + len(pending_frames) >= args.max_frames:
                break

            pending_frames.append(frame_bgr)
            if len(pending_frames) >= args.batch_size:
                flush_pending()
        flush_pending()
    finally:
        cap.release()
        if writer is not None:
            t_release = time.perf_counter()
            writer.release()
            if profile is not None:
                profile.add("writer_release", time.perf_counter() - t_release)

    elapsed = max(time.perf_counter() - start, 1e-6)
    if writer is not None:
        print(f"saved overlay video to: {output_path}")
    print(f"processed {frame_idx} frames in {elapsed:.2f}s ({frame_idx / elapsed:.2f} fps)")
    if profile is not None:
        profile.print_report(frame_idx, elapsed)


def main():
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
