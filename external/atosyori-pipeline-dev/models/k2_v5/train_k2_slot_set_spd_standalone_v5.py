from __future__ import annotations

import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TEACHER_ROOT = ROOT / "Teacher"
if str(TEACHER_ROOT) not in sys.path:
    sys.path.insert(0, str(TEACHER_ROOT))

import final_standalone_t5000 as fst


JOB_CONFIG = {
    # Edit these values directly for long runs.
    "train_jsonl": ROOT / "ellipse_distill_jsonl_200k/k2_model_rows.jsonl",
    "output_dir": ROOT / "Checkpoint/V5",
    "manifest_cache": ROOT / "Checkpoint/k2_slot_set_spd_manifest.pkl",
    "init_checkpoint": ROOT / "Checkpoint/V5/best_exact.pt",
    "load_optimizer_state": False,
    "image_size": 192,
    "batch_size": 24,
    "epochs": 10,
    "lr": 2.0e-5,
    "weight_decay": 1.0e-4,
    "num_workers": 4,
    "device": "cuda",
    "val_ratio": 0.02,
    "max_rows": 0,
    "seed": 42,
    "step_log_interval": 300,
    "exact_eval_every": 1,
    "base_width": 32,
    "slot_dim": 256,
    "decoder_layers": 3,
    "num_heads": 8,
    "render_sharpness": 28.0,
    "geometry_loss_weight": 2.0,
    "center_loss_weight": 2.0,
    "shape_loss_weight": 1.5,
    "extent_loss_weight": 0.75,
    "edge_geometry_boost": 2.5,
    "teacher_extent_weight": 2.5,
    "teacher_boundary_weight": 2.0,
    "teacher_directional_weight": 1.5,
    "teacher_boundary_sigma": 0.12,
    "directional_strip_fracs": [0.02, 0.04, 0.08],
    "union_dice_weight": 1.0,
    "union_bce_weight": 0.5,
    "fn_weight": 3.0,
    "boundary_weight": 1.0,
    "boundary_band_width": 0.06,
    "touch_boundary_weight": 0.5,
    "touch_boundary_band_width": 0.10,
    "touch_fn_weight": 2.0,
    "multi_scale_weight": 0.75,
    "sdf_weight": 0.5,
    "overlap_weight": 0.05,
    "grad_clip_norm": 1.0,
    "touch_strip_width_px": 2,
    "touch_strip_weight": 0.1,
    "worst_recall_quantile": 0.15,
    "worst_recall_boost": 0.5,
    "worst_iou_threshold": 0.90,
    "worst_recall_threshold": 0.97,
    "worst_precision_threshold": 0.95,
    "enable_wandb": True,
    "wandb_project": "k2-slot-set-spd",
    "wandb_entity": None,
    "wandb_run_name": "train_k2_slot_set_spd_standalone_v5",
    "wandb_mode": "online",
    "wandb_tags": ["k2", "slot_set_spd", "v5", "teacher_geometry"],
}


ELLIPSE_CENTER_MIN = -0.75
ELLIPSE_CENTER_MAX = 1.75
LOG_AXIS_MIN = -9.0
LOG_AXIS_MAX = 0.75

_COORD_GRID_CACHE: dict[int, np.ndarray] = {}
_BORDER_GRID_CACHE: dict[int, np.ndarray] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def flatten_for_wandb(obj: dict[str, object], prefix: str = "") -> dict[str, float | int | str]:
    flat: dict[str, float | int | str] = {}
    for key, value in obj.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_for_wandb(value, name))
        elif isinstance(value, (int, float, str)):
            flat[name] = value
    return flat


def build_summary_plot(history: list[dict[str, object]], output_path: Path) -> Path | None:
    if not history:
        return None
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(4, 2, figsize=(18, 18))

    ax = axes[0, 0]
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("global_iou", np.nan)) for row in history], label="model")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("global_iou", np.nan)) for row in history], label="teacher")
    ax.set_title("Global IoU")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("global_recall", np.nan)) for row in history], label="model")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("global_recall", np.nan)) for row in history], label="teacher")
    ax.set_title("Global Recall")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("worst_iou_count", np.nan)) for row in history], label="worst_iou_count")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("worst_recall_count", np.nan)) for row in history], label="worst_recall_count")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("worst_precision_count", np.nan)) for row in history], label="worst_precision_count")
    ax.set_title("Worst Counts")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("iou_p05", np.nan)) for row in history], label="iou_p05(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("iou_p05", np.nan)) for row in history], label="iou_p05(teacher)")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("recall_p05", np.nan)) for row in history], label="recall_p05(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("recall_p05", np.nan)) for row in history], label="recall_p05(teacher)")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("precision_p05", np.nan)) for row in history], label="precision_p05(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("precision_p05", np.nan)) for row in history], label="precision_p05(teacher)")
    ax.set_title("Worst-Case Distribution (p05)")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("iou_std", np.nan)) for row in history], label="iou_std(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("iou_std", np.nan)) for row in history], label="iou_std(teacher)")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("recall_std", np.nan)) for row in history], label="recall_std(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("recall_std", np.nan)) for row in history], label="recall_std(teacher)")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("precision_std", np.nan)) for row in history], label="precision_std(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("precision_std", np.nan)) for row in history], label="precision_std(teacher)")
    ax.set_title("Distribution Spread (std)")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("strip2_mean", np.nan)) for row in history], label="strip2_mean(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("strip2_mean", np.nan)) for row in history], label="strip2_mean(teacher)")
    ax.plot(epochs, [float(row.get("exact", {}).get("model", {}).get("strip2_p05", np.nan)) for row in history], label="strip2_p05(model)")
    ax.plot(epochs, [float(row.get("exact", {}).get("teacher", {}).get("strip2_p05", np.nan)) for row in history], label="strip2_p05(teacher)")
    ax.set_title("Strip2 Metrics")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 0]
    ax.plot(epochs, [float(row.get("train_loss", np.nan)) for row in history], label="train_loss")
    ax.plot(epochs, [float(row.get("val_loss", np.nan)) for row in history], label="val_loss")
    ax.plot(epochs, [float(row.get("train_geometry_loss", np.nan)) for row in history], label="train_geometry_loss")
    ax.plot(epochs, [float(row.get("train_union_bce", np.nan)) for row in history], label="train_union_bce")
    ax.plot(epochs, [float(row.get("train_strip2_loss", np.nan)) for row in history], label="train_strip2_loss")
    ax.plot(epochs, [float(row.get("train_touch_boundary_loss", np.nan)) for row in history], label="train_touch_boundary_loss")
    ax.plot(epochs, [float(row.get("train_teacher_extent_loss", np.nan)) for row in history], label="train_teacher_extent_loss")
    ax.plot(epochs, [float(row.get("train_teacher_boundary_loss", np.nan)) for row in history], label="train_teacher_boundary_loss")
    ax.set_title("Main Losses")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 1]
    ax.plot(epochs, [float(row.get("train_mask_iou", np.nan)) for row in history], label="train_mask_iou")
    ax.plot(epochs, [float(row.get("val_mask_iou", np.nan)) for row in history], label="val_mask_iou")
    ax.plot(epochs, [float(row.get("train_soft_recall", np.nan)) for row in history], label="train_soft_recall")
    ax.plot(epochs, [float(row.get("val_soft_recall", np.nan)) for row in history], label="val_soft_recall")
    ax.plot(epochs, [float(row.get("train_worst_recall_frac", np.nan)) for row in history], label="train_worst_recall_frac")
    ax.set_title("Training Dynamics")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_jsonl_offsets(jsonl_path: Path) -> np.ndarray:
    cache_path = jsonl_path.with_suffix(jsonl_path.suffix + ".offsets.npy")
    if cache_path.exists() and cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
        return np.load(cache_path)
    offsets: list[int] = []
    offset = 0
    with jsonl_path.open("rb") as f:
        for line in f:
            offsets.append(offset)
            offset += len(line)
    arr = np.asarray(offsets, dtype=np.int64)
    np.save(cache_path, arr)
    return arr


def parse_teacher_line(path: Path, offset: int) -> dict[str, object]:
    with path.open("rb") as f:
        f.seek(int(offset))
        return json.loads(f.readline().decode("utf-8"))


def deterministic_split(key: str, val_ratio: float) -> str:
    digest = int.from_bytes(__import__("hashlib").blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")
    value = digest / float(2**64 - 1)
    return "val" if value < val_ratio else "train"


def square_pad_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    height, width = mask.shape
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_left = (side - width) // 2
    padded = np.zeros((side, side), dtype=np.uint8)
    padded[pad_top:pad_top + height, pad_left:pad_left + width] = mask
    return padded, (pad_left, pad_top), side


def build_signed_distance_channel(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8, copy=False)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    signed = inside - outside
    scale = float(max(mask.shape))
    return (signed / max(scale, 1.0)).astype(np.float32)


def build_edge_channel(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return edge.astype(np.float32)


def get_coord_grid(image_size: int) -> np.ndarray:
    cached = _COORD_GRID_CACHE.get(image_size)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx = np.repeat(grid[None, :], image_size, axis=0)
    yy = np.repeat(grid[:, None], image_size, axis=1)
    stacked = np.stack([xx, yy], axis=0)
    _COORD_GRID_CACHE[image_size] = stacked
    return stacked


def get_border_distance_grid(image_size: int) -> np.ndarray:
    cached = _BORDER_GRID_CACHE.get(image_size)
    if cached is not None:
        return cached
    grid = np.linspace(0.0, 1.0, image_size, dtype=np.float32)
    xx = np.repeat(grid[None, :], image_size, axis=0)
    yy = np.repeat(grid[:, None], image_size, axis=1)
    border = np.minimum.reduce([xx, 1.0 - xx, yy, 1.0 - yy]).astype(np.float32)
    maxv = float(border.max()) if border.size else 1.0
    if maxv > 0.0:
        border /= maxv
    border = border[None, ...]
    _BORDER_GRID_CACHE[image_size] = border
    return border


def build_touch_flag_planes(image_size: int, touch_flags: np.ndarray) -> np.ndarray:
    planes = np.broadcast_to(touch_flags.astype(np.float32)[:, None, None], (4, image_size, image_size))
    return np.asarray(planes, dtype=np.float32)


def edge_touch_vector_from_row(row: dict[str, object], gt_mask: np.ndarray | None = None) -> np.ndarray:
    meta = row.get("source_metadata")
    if isinstance(meta, dict):
        sides = meta.get("edge_sides")
        if isinstance(sides, dict):
            return np.asarray(
                [
                    float(bool(sides.get("left", False))),
                    float(bool(sides.get("right", False))),
                    float(bool(sides.get("top", False))),
                    float(bool(sides.get("bottom", False))),
                ],
                dtype=np.float32,
            )
    if gt_mask is None:
        return np.zeros((4,), dtype=np.float32)
    touches = fst.detect_edge_touches(gt_mask.astype(np.uint8))
    return np.asarray(
        [
            float(bool(touches.get("left", False))),
            float(bool(touches.get("right", False))),
            float(bool(touches.get("top", False))),
            float(bool(touches.get("bottom", False))),
        ],
        dtype=np.float32,
    )


def build_input_image(
    mask_resized: np.ndarray,
    signed_resized: np.ndarray,
    edge_resized: np.ndarray,
    touch_flags: np.ndarray,
    image_size: int,
) -> np.ndarray:
    coords = get_coord_grid(image_size)
    border = get_border_distance_grid(image_size)
    touch_planes = build_touch_flag_planes(image_size, touch_flags)
    return np.concatenate(
        [
            mask_resized[None, ...].astype(np.float32, copy=False),
            signed_resized[None, ...].astype(np.float32, copy=False),
            edge_resized[None, ...].astype(np.float32, copy=False),
            coords.astype(np.float32, copy=False),
            border.astype(np.float32, copy=False),
            touch_planes.astype(np.float32, copy=False),
        ],
        axis=0,
    )


def ellipse_to_normalized_state(ellipse: tuple[float, float, float, float, float], square_size: int) -> np.ndarray:
    cx, cy, a, b, angle = fst.normalize_ellipse(ellipse)
    scale = float(max(square_size, 1))
    theta = math.radians(angle)
    return np.asarray(
        [
            cx / scale,
            cy / scale,
            math.log(max(a / scale, 1e-4)),
            math.log(max(b / scale, 1e-4)),
            math.cos(2.0 * theta),
            math.sin(2.0 * theta),
        ],
        dtype=np.float32,
    )


def absolute_to_local_square_ellipses_from_payload(
    absolute_ellipses: list[tuple[float, float, float, float, float]],
    payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]],
) -> tuple[list[tuple[float, float, float, float, float]], int]:
    (height, width), origin, _ = payload
    pad_side = max(height, width)
    pad_left = (pad_side - width) // 2
    pad_top = (pad_side - height) // 2
    local = fst.shift_ellipses_to_local(absolute_ellipses, origin)
    padded = [(cx + float(pad_left), cy + float(pad_top), a, b, angle) for cx, cy, a, b, angle in local]
    return padded, int(pad_side)


def load_target_abs_ellipses(row: dict[str, object]) -> list[tuple[float, float, float, float, float]]:
    if "target_ellipse_params" in row:
        return fst.deserialize_ellipses(row["target_ellipse_params"])
    k2 = row.get("k2", {})
    if isinstance(k2, dict):
        best_direction = str(row.get("best_direction") or k2.get("best_direction", "forward"))
        solution = k2.get(best_direction)
        if isinstance(solution, dict) and "ellipses" in solution:
            return fst.deserialize_ellipses(solution["ellipses"])
    raise KeyError("Teacher row does not contain target K2 ellipse params.")


def states_to_abs_ellipses_from_payload(
    states: np.ndarray,
    payload: tuple[tuple[int, int], tuple[int, int], list[np.ndarray]],
) -> list[tuple[float, float, float, float, float]]:
    if states.ndim == 1:
        states = states.reshape(2, 6)
    (height, width), origin, _ = payload
    side = max(height, width)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    absolute: list[tuple[float, float, float, float, float]] = []
    for state in states:
        cx_n, cy_n, loga, logb, cos2, sin2 = [float(v) for v in state]
        cx_n = min(max(cx_n, ELLIPSE_CENTER_MIN), ELLIPSE_CENTER_MAX)
        cy_n = min(max(cy_n, ELLIPSE_CENTER_MIN), ELLIPSE_CENTER_MAX)
        loga = min(max(loga, LOG_AXIS_MIN), LOG_AXIS_MAX)
        logb = min(max(logb, LOG_AXIS_MIN), LOG_AXIS_MAX)
        norm = math.hypot(cos2, sin2)
        if not math.isfinite(norm) or norm < 1e-6:
            cos2, sin2 = 1.0, 0.0
        else:
            cos2 /= norm
            sin2 /= norm
        cx = cx_n * side - pad_left
        cy = cy_n * side - pad_top
        a = math.exp(loga) * side
        b = math.exp(logb) * side
        angle = math.degrees(0.5 * math.atan2(sin2, cos2))
        absolute.append(fst.normalize_ellipse((cx + origin[0], cy + origin[1], a, b, angle)))
    return absolute


@lru_cache(maxsize=4096)
def load_source_context(
    jsonl_path_str: str,
    offset: int,
) -> tuple[dict[str, object], tuple[tuple[int, int], tuple[int, int], list[np.ndarray]], np.ndarray, list[np.ndarray]]:
    row = parse_teacher_line(Path(jsonl_path_str), int(offset))
    payload = fst.prepare_local_raster_payload(str(row["polygons"]))
    gt_mask, _origin = fst.rasterize_local_mask_from_payload(payload)
    gt_square, _pad, _side = square_pad_mask(gt_mask)
    gt_polygons = fst.parse_polygons(str(row["polygons"]))
    return row, payload, gt_square.astype(np.uint8, copy=False), gt_polygons


def build_manifest_cache_path(base_path: Path, *, jsonl_path: Path, image_size: int, val_ratio: float, max_rows: int) -> Path:
    stem = {
        "jsonl": str(Path(jsonl_path).resolve()),
        "image_size": int(image_size),
        "val_ratio": float(val_ratio),
        "max_rows": int(max_rows),
        "version": 1,
    }
    digest = __import__("hashlib").sha1(json.dumps(stem, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_{digest}{base_path.suffix}")
    return base_path.with_name(f"{base_path.name}_{digest}.pkl")


def build_or_load_manifest(
    *,
    jsonl_path: Path,
    manifest_cache: Path,
    image_size: int,
    val_ratio: float,
    max_rows: int,
) -> tuple[dict[str, object], Path, bool]:
    cache_path = build_manifest_cache_path(
        manifest_cache,
        jsonl_path=jsonl_path,
        image_size=image_size,
        val_ratio=val_ratio,
        max_rows=max_rows,
    )
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f), cache_path, True

    offsets = load_jsonl_offsets(jsonl_path)
    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    for offset in offsets.tolist():
        row = parse_teacher_line(jsonl_path, int(offset))
        row_mode = row.get("target_mode") or row.get("best_local_mode")
        if row_mode != "K2":
            continue
        split = deterministic_split(f"{row['frame']}::{row['track_id']}", float(val_ratio))
        if max_rows > 0 and counts[split] >= max_rows:
            if all(counts[name] >= max_rows for name in ("train", "val")):
                break
            continue
        _row, payload, gt_square, gt_polys = load_source_context(str(jsonl_path), int(offset))
        target_abs = load_target_abs_ellipses(row)
        target_local, side = absolute_to_local_square_ellipses_from_payload(target_abs, payload)
        target_states = np.stack([ellipse_to_normalized_state(ellipse, side) for ellipse in target_local], axis=0).astype(np.float32)
        gt_metrics = fst.compute_exact_metrics_from_polygons(gt_polys, fst.ellipses_to_polygon_arrays(target_abs))
        gt_metrics["weighted_error"] = float(fst.compute_weighted_error(gt_metrics))
        splits[split].append(
            {
                "offset": int(offset),
                "frame": int(row["frame"]),
                "track_id": str(row["track_id"]),
                "touch_flags": edge_touch_vector_from_row(row, gt_mask=gt_square).astype(np.float32),
                "target_states": target_states,
                "teacher_exact_metrics": gt_metrics,
            }
        )
        counts[split] += 1

    manifest = {
        "jsonl_path": str(jsonl_path),
        "image_size": int(image_size),
        "splits": splits,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)
    return manifest, cache_path, False


@dataclass
class DirectRow:
    offset: int
    frame: int
    track_id: str
    touch_flags: np.ndarray
    target_states: np.ndarray
    teacher_exact_metrics: dict[str, float]


class K2SlotSetDataset(Dataset):
    def __init__(self, manifest: dict[str, object], split: str) -> None:
        self.image_size = int(manifest["image_size"])
        self.jsonl_path = str(manifest["jsonl_path"])
        self.rows = [
            DirectRow(
                offset=int(item["offset"]),
                frame=int(item["frame"]),
                track_id=str(item["track_id"]),
                touch_flags=np.asarray(item["touch_flags"], dtype=np.float32),
                target_states=np.asarray(item["target_states"], dtype=np.float32),
                teacher_exact_metrics=dict(item["teacher_exact_metrics"]),
            )
            for item in manifest["splits"][split]
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        _src_row, _payload, gt_square, _gt_polys = load_source_context(self.jsonl_path, int(row.offset))
        padded_mask = gt_square.astype(np.uint8, copy=False)
        signed = build_signed_distance_channel(padded_mask)
        edge = build_edge_channel(padded_mask)
        mask_resized = cv2.resize(padded_mask.astype(np.float32), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        signed_resized = cv2.resize(signed, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        edge_resized = cv2.resize(edge, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        target_mask_resized = cv2.resize(padded_mask.astype(np.float32), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        input_image = build_input_image(mask_resized, signed_resized, edge_resized, row.touch_flags, self.image_size)
        signed_resized = signed_resized.astype(np.float32, copy=False)
        return {
            "input_image": torch.from_numpy(input_image.astype(np.float32, copy=False)),
            "target_states": torch.from_numpy(row.target_states.astype(np.float32, copy=True)),
            "target_mask": torch.from_numpy(target_mask_resized.astype(np.float32, copy=False)),
            "target_signed": torch.from_numpy(signed_resized),
            "touch_flags": torch.from_numpy(row.touch_flags.astype(np.float32, copy=True)),
            "offset": torch.tensor(int(row.offset), dtype=torch.int64),
        }


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, hidden, 3, groups=hidden),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SlotDecoderBlock(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_q3 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q1(queries)
        c = self.norm_ctx(context)
        queries = queries + self.cross_attn(q, c, c, need_weights=False)[0]
        q = self.norm_q2(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        queries = queries + self.mlp(self.norm_q3(queries))
        return queries


def render_soft_slots_from_spd(
    centers: torch.Tensor,
    chol_params: torch.Tensor,
    image_size: int,
    sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n_slots, _ = centers.shape
    grid = torch.linspace(0.0, 1.0, image_size, device=centers.device, dtype=centers.dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xx = xx.view(1, 1, image_size, image_size)
    yy = yy.view(1, 1, image_size, image_size)

    l11 = torch.nn.functional.softplus(chol_params[..., 0]).view(batch, n_slots, 1, 1) + 1e-4
    l21 = chol_params[..., 1].view(batch, n_slots, 1, 1)
    l22 = torch.nn.functional.softplus(chol_params[..., 2]).view(batch, n_slots, 1, 1) + 1e-4

    a11 = l11 * l11
    a12 = l11 * l21
    a22 = l21 * l21 + l22 * l22

    dx = xx - centers[..., 0].view(batch, n_slots, 1, 1)
    dy = yy - centers[..., 1].view(batch, n_slots, 1, 1)
    quad = a11 * dx * dx + 2.0 * a12 * dx * dy + a22 * dy * dy
    q = 1.0 - quad
    slot_masks = torch.sigmoid(q * sharpness)
    union = 1.0 - torch.prod(1.0 - slot_masks, dim=1)
    return slot_masks, union


def spd_to_normalized_states(centers: torch.Tensor, chol_params: torch.Tensor) -> torch.Tensor:
    l11 = torch.nn.functional.softplus(chol_params[..., 0]) + 1e-4
    l21 = chol_params[..., 1]
    l22 = torch.nn.functional.softplus(chol_params[..., 2]) + 1e-4
    a11 = l11 * l11
    a12 = l11 * l21
    a22 = l21 * l21 + l22 * l22

    trace = a11 + a22
    disc = torch.sqrt(((a11 - a22) ** 2 + 4.0 * a12 * a12).clamp_min(1e-10))
    lam_min = ((trace - disc) * 0.5).clamp_min(1e-8)
    lam_max = ((trace + disc) * 0.5).clamp_min(1e-8)
    major = torch.rsqrt(lam_min).clamp_min(1e-4)
    minor = torch.rsqrt(lam_max).clamp_min(1e-4)

    det = (a11 * a22 - a12 * a12).clamp_min(1e-10)
    cov_xx = a22 / det
    cov_xy = -a12 / det
    cov_yy = a11 / det
    denom = torch.sqrt((cov_xx - cov_yy) ** 2 + (2.0 * cov_xy) ** 2).clamp_min(1e-8)
    cos2 = (cov_xx - cov_yy) / denom
    sin2 = (2.0 * cov_xy) / denom

    return torch.stack(
        [
            centers[..., 0],
            centers[..., 1],
            torch.log(major),
            torch.log(minor),
            cos2,
            sin2,
        ],
        dim=-1,
    )


class K2SlotSetSPDNet(nn.Module):
    def __init__(self, in_channels: int, base_width: int, slot_dim: int, decoder_layers: int, num_heads: int, sharpness: float) -> None:
        super().__init__()
        c1 = int(base_width)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 12
        self.sharpness = float(sharpness)

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, 3, stride=1),
            ResidualBlock(c1),
            ResidualBlock(c1),
        )
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, 3, stride=2), ResidualBlock(c2), ResidualBlock(c2))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, stride=2), ResidualBlock(c3), ResidualBlock(c3))
        self.stage4 = nn.Sequential(ConvBNAct(c3, c4, 3, stride=2), ResidualBlock(c4), ResidualBlock(c4), ResidualBlock(c4))
        self.stage5 = nn.Sequential(ConvBNAct(c4, c5, 3, stride=2), ResidualBlock(c5), ResidualBlock(c5), ResidualBlock(c5))

        self.lat5 = nn.Conv2d(c5, slot_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, slot_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, slot_dim, kernel_size=1)
        self.fpn4 = ConvBNAct(slot_dim, slot_dim, 3)
        self.fpn3 = ConvBNAct(slot_dim, slot_dim, 3)
        self.context_proj = ConvBNAct(slot_dim, slot_dim, 3)

        self.slot_queries = nn.Parameter(torch.randn(2, slot_dim) * 0.02)
        self.decoder = nn.ModuleList([SlotDecoderBlock(slot_dim, num_heads=num_heads, mlp_ratio=4) for _ in range(decoder_layers)])
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(slot_dim, slot_dim, 1), nn.SiLU(inplace=True))
        self.slot_refine = nn.Sequential(nn.Linear(slot_dim * 2, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim))
        self.center_head = nn.Linear(slot_dim, 2)
        self.chol_head = nn.Linear(slot_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        p5 = self.lat5(s5)
        p4 = self.fpn4(self.lat4(s4) + torch.nn.functional.interpolate(p5, size=s4.shape[-2:], mode="bilinear", align_corners=False))
        p3 = self.fpn3(self.lat3(s3) + torch.nn.functional.interpolate(p4, size=s3.shape[-2:], mode="bilinear", align_corners=False))
        context_map = self.context_proj(p4)
        context_tokens = context_map.flatten(2).transpose(1, 2)

        queries = self.slot_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        global_feat = self.global_pool(torch.nn.functional.interpolate(p3, size=context_map.shape[-2:], mode="bilinear", align_corners=False))
        global_feat = global_feat.flatten(1).unsqueeze(1).expand(-1, 2, -1)
        queries = queries + self.slot_refine(torch.cat([queries, global_feat], dim=-1))
        for block in self.decoder:
            queries = block(queries, context_tokens)

        centers = self.center_head(queries)
        chol_params = self.chol_head(queries)
        slot_masks, union_mask = render_soft_slots_from_spd(centers, chol_params, image_size=x.shape[-1], sharpness=self.sharpness)
        states = spd_to_normalized_states(centers, chol_params)
        return {
            "states": states.reshape(x.shape[0], 12),
            "centers": centers,
            "chol_params": chol_params,
            "slot_masks": slot_masks,
            "union_mask": union_mask,
        }


def pair_cost(pred_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([2.0, 2.0, 1.5, 1.5, 0.5, 0.5], device=pred_states.device, dtype=pred_states.dtype)
    diff = torch.nn.functional.smooth_l1_loss(pred_states, target_states, reduction="none")
    return (diff * weights.view(1, 1, 6)).mean(dim=(1, 2))


def states_to_geometry_tensors(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centers = states[..., :2]
    log_major = states[..., 2]
    log_minor = states[..., 3]
    cos2 = states[..., 4].clamp(-1.0, 1.0)
    sin2 = states[..., 5].clamp(-1.0, 1.0)

    cos_theta = torch.sqrt(((1.0 + cos2) * 0.5).clamp_min(1e-8))
    sin_theta = torch.sign(sin2) * torch.sqrt(((1.0 - cos2) * 0.5).clamp_min(1e-8))
    major2 = torch.exp(2.0 * log_major).clamp_min(1e-8)
    minor2 = torch.exp(2.0 * log_minor).clamp_min(1e-8)

    c2 = cos_theta * cos_theta
    s2 = sin_theta * sin_theta
    cs = cos_theta * sin_theta
    cov_xx = c2 * major2 + s2 * minor2
    cov_xy = cs * (major2 - minor2)
    cov_yy = s2 * major2 + c2 * minor2
    cov = torch.stack([cov_xx, cov_xy, cov_yy], dim=-1)
    return centers, cov


def pair_geometry_cost(
    pred_centers: torch.Tensor,
    pred_chol_params: torch.Tensor,
    target_states: torch.Tensor,
    *,
    center_weight: float,
    shape_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_states = spd_to_normalized_states(pred_centers, pred_chol_params)
    pred_centers_geom, pred_a = states_to_geometry_tensors(pred_states)
    target_centers, target_a = states_to_geometry_tensors(target_states)

    center_loss = torch.nn.functional.smooth_l1_loss(pred_centers_geom, target_centers, reduction="none").mean(dim=(1, 2))
    shape_loss = torch.nn.functional.smooth_l1_loss(pred_a, target_a, reduction="none").mean(dim=(1, 2))
    total = float(center_weight) * center_loss + float(shape_weight) * shape_loss
    return total, center_loss, shape_loss


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    intersection = (pred * target).sum(dim=(1, 2))
    denom = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def weighted_union_bce(pred_union: torch.Tensor, target_mask: torch.Tensor, fn_weight: float) -> torch.Tensor:
    pred = pred_union.clamp(1e-5, 1.0 - 1e-5)
    weight = 1.0 + float(fn_weight) * target_mask
    bce = -(target_mask * torch.log(pred) + (1.0 - target_mask) * torch.log(1.0 - pred))
    return (bce * weight).mean()


def boundary_band_loss(pred_union: torch.Tensor, target_mask: torch.Tensor, target_signed: torch.Tensor, band_width: float) -> torch.Tensor:
    band = (target_signed.abs() <= float(band_width)).to(pred_union.dtype)
    if float(band.sum()) <= 0.0:
        return pred_union.new_tensor(0.0)
    return ((pred_union - target_mask).abs() * band).sum() / band.sum().clamp_min(1.0)


def build_touch_side_mask(shape: tuple[int, int], touch_flags: torch.Tensor, band_frac: float) -> torch.Tensor:
    batch = touch_flags.shape[0]
    height, width = int(shape[0]), int(shape[1])
    yy = torch.linspace(0.0, 1.0, height, device=touch_flags.device, dtype=touch_flags.dtype).view(1, height, 1)
    xx = torch.linspace(0.0, 1.0, width, device=touch_flags.device, dtype=touch_flags.dtype).view(1, 1, width)
    band = max(float(band_frac), 1e-4)
    side_mask = torch.zeros((batch, height, width), device=touch_flags.device, dtype=touch_flags.dtype)
    side_mask = side_mask + touch_flags[:, 0].view(batch, 1, 1) * (xx <= band).to(touch_flags.dtype)
    side_mask = side_mask + touch_flags[:, 1].view(batch, 1, 1) * (xx >= 1.0 - band).to(touch_flags.dtype)
    side_mask = side_mask + touch_flags[:, 2].view(batch, 1, 1) * (yy <= band).to(touch_flags.dtype)
    side_mask = side_mask + touch_flags[:, 3].view(batch, 1, 1) * (yy >= 1.0 - band).to(touch_flags.dtype)
    return (side_mask > 0.0).to(touch_flags.dtype)


def touch_side_boundary_loss(
    pred_union: torch.Tensor,
    target_mask: torch.Tensor,
    target_signed: torch.Tensor,
    touch_flags: torch.Tensor,
    *,
    boundary_band_width: float,
    touch_band_width: float,
) -> torch.Tensor:
    boundary_band = (target_signed.abs() <= float(boundary_band_width)).to(pred_union.dtype)
    touch_side = build_touch_side_mask(target_mask.shape[-2:], touch_flags, band_frac=float(touch_band_width))
    weight = boundary_band * touch_side
    if float(weight.sum()) <= 0.0:
        return pred_union.new_tensor(0.0)
    return ((pred_union - target_mask).abs() * weight).sum() / weight.sum().clamp_min(1.0)


def touch_side_weighted_bce(
    pred_union: torch.Tensor,
    target_mask: torch.Tensor,
    touch_flags: torch.Tensor,
    *,
    fn_weight: float,
    touch_fn_weight: float,
    touch_band_width: float,
) -> torch.Tensor:
    pred = pred_union.clamp(1e-5, 1.0 - 1e-5)
    touch_side = build_touch_side_mask(target_mask.shape[-2:], touch_flags, band_frac=float(touch_band_width))
    weight = 1.0 + float(fn_weight) * target_mask + float(touch_fn_weight) * (target_mask * touch_side)
    bce = -(target_mask * torch.log(pred) + (1.0 - target_mask) * torch.log(1.0 - pred))
    return (bce * weight).mean()


def multi_scale_union_loss(pred_union: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    pred = pred_union.unsqueeze(1)
    target = target_mask.unsqueeze(1)
    for scale in (1, 2, 4):
        if scale == 1:
            p = pred[:, 0]
            t = target[:, 0]
        else:
            p = torch.nn.functional.avg_pool2d(pred, kernel_size=scale, stride=scale)[:, 0]
            t = torch.nn.functional.avg_pool2d(target, kernel_size=scale, stride=scale)[:, 0]
        losses.append(soft_dice_loss(p, t))
    return torch.stack(losses).mean()


def sdf_narrow_band_loss(pred_union: torch.Tensor, target_signed: torch.Tensor, band_width: float) -> torch.Tensor:
    band = (target_signed.abs() <= float(band_width)).to(pred_union.dtype)
    target_soft = (target_signed <= 0.0).to(pred_union.dtype)
    if float(band.sum()) <= 0.0:
        return pred_union.new_tensor(0.0)
    return ((pred_union - target_soft).abs() * band).sum() / band.sum().clamp_min(1.0)


def build_touch_strip_targets(target_mask: torch.Tensor, touch_flags: torch.Tensor, strip_width_px: int) -> torch.Tensor:
    batch, height, width = target_mask.shape
    strip = torch.zeros_like(target_mask)
    px = max(int(strip_width_px), 1)
    if px > width:
        px = width
    if px > height:
        px = height
    if px > 0:
        strip[:, :, :px] += touch_flags[:, 0].view(batch, 1, 1)
        strip[:, :, width - px:] += touch_flags[:, 1].view(batch, 1, 1)
        strip[:, :px, :] += touch_flags[:, 2].view(batch, 1, 1)
        strip[:, height - px:, :] += touch_flags[:, 3].view(batch, 1, 1)
    strip = (strip > 0.0).to(target_mask.dtype)
    return strip * target_mask


def touch_strip_recall_loss(
    pred_union: torch.Tensor,
    target_mask: torch.Tensor,
    touch_flags: torch.Tensor,
    strip_width_px: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_strip = build_touch_strip_targets(target_mask, touch_flags, strip_width_px=strip_width_px)
    strip_area = target_strip.sum(dim=(1, 2))
    coverage = (pred_union * target_strip).sum(dim=(1, 2)) / strip_area.clamp_min(1.0)
    valid = (strip_area > 0.0).to(pred_union.dtype)
    loss_per_sample = (1.0 - coverage) * valid
    return loss_per_sample.mean(), coverage


def touch_strip_coverages_from_masks(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    touch_flags: np.ndarray,
    strip_width_px: int,
) -> list[float]:
    height, width = gt_mask.shape
    px = max(int(strip_width_px), 1)
    px_w = min(px, width)
    px_h = min(px, height)
    coverages: list[float] = []
    for side_name, touched in zip(["left", "right", "top", "bottom"], touch_flags):
        if float(touched) <= 0.5:
            continue
        strip = np.zeros_like(gt_mask, dtype=np.uint8)
        if side_name == "left":
            strip[:, :px_w] = 1
        elif side_name == "right":
            strip[:, width - px_w:] = 1
        elif side_name == "top":
            strip[:px_h, :] = 1
        elif side_name == "bottom":
            strip[height - px_h:, :] = 1
        target = (gt_mask.astype(bool) & strip.astype(bool))
        denom = int(target.sum())
        if denom <= 0:
            continue
        coverages.append(float((pred_mask.astype(bool) & target).sum() / denom))
    return coverages


def soft_side_extents_from_mask(pred_union: torch.Tensor, touch_flags: torch.Tensor, power: float = 8.0) -> torch.Tensor:
    batch, height, width = pred_union.shape
    yy = torch.linspace(0.0, 1.0, height, device=pred_union.device, dtype=pred_union.dtype).view(1, height, 1)
    xx = torch.linspace(0.0, 1.0, width, device=pred_union.device, dtype=pred_union.dtype).view(1, 1, width)
    weight = pred_union.clamp_min(0.0).pow(float(power))
    norm = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    right_extent = (weight * xx).sum(dim=(1, 2)) / norm.view(batch)
    left_extent = (weight * (1.0 - xx)).sum(dim=(1, 2)) / norm.view(batch)
    top_extent = (weight * (1.0 - yy)).sum(dim=(1, 2)) / norm.view(batch)
    bottom_extent = (weight * yy).sum(dim=(1, 2)) / norm.view(batch)
    extents = torch.stack([left_extent, right_extent, top_extent, bottom_extent], dim=-1)
    return extents * touch_flags


def render_soft_union_from_states(target_states: torch.Tensor, image_size: int, sharpness: float) -> torch.Tensor:
    batch, n_slots, _ = target_states.shape
    centers, cov = states_to_geometry_tensors(target_states)
    cxx = cov[..., 0].clamp_min(1e-8)
    cxy = cov[..., 1]
    cyy = cov[..., 2].clamp_min(1e-8)
    det = (cxx * cyy - cxy * cxy).clamp_min(1e-10)
    a11 = (cyy / det).view(batch, n_slots, 1, 1)
    a12 = (-cxy / det).view(batch, n_slots, 1, 1)
    a22 = (cxx / det).view(batch, n_slots, 1, 1)

    grid = torch.linspace(0.0, 1.0, image_size, device=target_states.device, dtype=target_states.dtype)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xx = xx.view(1, 1, image_size, image_size)
    yy = yy.view(1, 1, image_size, image_size)
    dx = xx - centers[..., 0].view(batch, n_slots, 1, 1)
    dy = yy - centers[..., 1].view(batch, n_slots, 1, 1)
    quad = a11 * dx * dx + 2.0 * a12 * dx * dy + a22 * dy * dy
    slot_masks = torch.sigmoid((1.0 - quad) * float(sharpness))
    return 1.0 - torch.prod(1.0 - slot_masks, dim=1)


def teacher_boundary_distance_loss(pred_union: torch.Tensor, teacher_union: torch.Tensor, sigma: float) -> torch.Tensor:
    boundary_weight = torch.exp(-((teacher_union - 0.5) / max(float(sigma), 1e-4)) ** 2)
    if float(boundary_weight.sum()) <= 0.0:
        return pred_union.new_tensor(0.0)
    return ((pred_union - teacher_union).abs() * boundary_weight).sum() / boundary_weight.sum().clamp_min(1.0)


def directional_strip_mass_loss(
    pred_union: torch.Tensor,
    teacher_union: torch.Tensor,
    touch_flags: torch.Tensor,
    strip_fracs: list[float],
) -> torch.Tensor:
    if not strip_fracs:
        return pred_union.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    for frac in strip_fracs:
        side_mask = build_touch_side_mask(pred_union.shape[-2:], touch_flags, band_frac=float(frac))
        valid = (side_mask.sum(dim=(1, 2)) > 0.0).to(pred_union.dtype)
        denom = side_mask.sum(dim=(1, 2)).clamp_min(1.0)
        pred_mass = (pred_union * side_mask).sum(dim=(1, 2)) / denom
        teacher_mass = (teacher_union * side_mask).sum(dim=(1, 2)) / denom
        loss = (torch.nn.functional.smooth_l1_loss(pred_mass, teacher_mass, reduction="none") * valid).sum() / valid.sum().clamp_min(1.0)
        losses.append(loss)
    return torch.stack(losses).mean()


def slot_set_spd_loss(
    pred_output: dict[str, torch.Tensor],
    target_states: torch.Tensor,
    target_mask: torch.Tensor,
    target_signed: torch.Tensor,
    touch_flags: torch.Tensor,
    *,
    geometry_loss_weight: float,
    center_loss_weight: float,
    shape_loss_weight: float,
    extent_loss_weight: float,
    edge_geometry_boost: float,
    teacher_extent_weight: float,
    teacher_boundary_weight: float,
    teacher_directional_weight: float,
    teacher_boundary_sigma: float,
    directional_strip_fracs: list[float],
    union_dice_weight: float,
    union_bce_weight: float,
    fn_weight: float,
    boundary_weight: float,
    touch_boundary_weight: float,
    touch_boundary_band_width: float,
    touch_fn_weight: float,
    multi_scale_weight: float,
    sdf_weight: float,
    overlap_weight: float,
    boundary_band_width: float,
    touch_strip_width_px: int,
    touch_strip_weight: float,
    worst_recall_quantile: float,
    worst_recall_boost: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_states = pred_output["states"].view(target_states.shape[0], 2, 6)
    pred_centers = pred_output["centers"]
    pred_chol_params = pred_output["chol_params"]
    slot_masks = pred_output["slot_masks"]
    pred_union = pred_output["union_mask"]
    teacher_union = render_soft_union_from_states(target_states, image_size=pred_union.shape[-1], sharpness=float(JOB_CONFIG["render_sharpness"]))

    swapped_target = target_states[:, [1, 0], :]
    direct_states = pair_cost(pred_states, target_states)
    swapped_states = pair_cost(pred_states, swapped_target)
    direct_geom, direct_center, direct_shape = pair_geometry_cost(
        pred_centers, pred_chol_params, target_states, center_weight=center_loss_weight, shape_weight=shape_loss_weight
    )
    swapped_geom, swapped_center, swapped_shape = pair_geometry_cost(
        pred_centers, pred_chol_params, swapped_target, center_weight=center_loss_weight, shape_weight=shape_loss_weight
    )
    direct_total = direct_geom + 0.25 * direct_states
    swapped_total = swapped_geom + 0.25 * swapped_states
    choose_direct = (direct_total <= swapped_total).to(pred_union.dtype)
    geometry_loss_per_sample = choose_direct * direct_geom + (1.0 - choose_direct) * swapped_geom
    center_loss_per_sample = choose_direct * direct_center + (1.0 - choose_direct) * swapped_center
    shape_loss_per_sample = choose_direct * direct_shape + (1.0 - choose_direct) * swapped_shape
    state_loss_per_sample = choose_direct * direct_states + (1.0 - choose_direct) * swapped_states

    edge_mask = (touch_flags.sum(dim=1) > 0.0).to(pred_union.dtype)
    geometry_sample_weight = 1.0 + edge_mask * max(float(edge_geometry_boost) - 1.0, 0.0)
    geometry_loss = (geometry_loss_per_sample * geometry_sample_weight).sum() / geometry_sample_weight.sum().clamp_min(1.0)
    center_loss = (center_loss_per_sample * geometry_sample_weight).sum() / geometry_sample_weight.sum().clamp_min(1.0)
    shape_loss = (shape_loss_per_sample * geometry_sample_weight).sum() / geometry_sample_weight.sum().clamp_min(1.0)
    slot_loss = (state_loss_per_sample * geometry_sample_weight).sum() / geometry_sample_weight.sum().clamp_min(1.0)

    union_dice = soft_dice_loss(pred_union, target_mask)
    union_bce = weighted_union_bce(pred_union, target_mask, fn_weight=fn_weight)
    touch_fn_bce = touch_side_weighted_bce(
        pred_union,
        target_mask,
        touch_flags,
        fn_weight=fn_weight,
        touch_fn_weight=touch_fn_weight,
        touch_band_width=touch_boundary_band_width,
    )
    boundary_loss = boundary_band_loss(pred_union, target_mask, target_signed, band_width=boundary_band_width)
    touch_boundary = touch_side_boundary_loss(
        pred_union,
        target_mask,
        target_signed,
        touch_flags,
        boundary_band_width=boundary_band_width,
        touch_band_width=touch_boundary_band_width,
    )
    ms_loss = multi_scale_union_loss(pred_union, target_mask)
    sdf_loss = sdf_narrow_band_loss(pred_union, target_signed, band_width=boundary_band_width)
    overlap_loss = (slot_masks[:, 0] * slot_masks[:, 1]).mean()
    strip_loss, strip_coverage = touch_strip_recall_loss(
        pred_union,
        target_mask,
        touch_flags,
        strip_width_px=touch_strip_width_px,
    )
    target_extents = soft_side_extents_from_mask(target_mask, touch_flags)
    pred_extents = soft_side_extents_from_mask(pred_union, touch_flags)
    extent_valid = (touch_flags.sum(dim=1) > 0.0).to(pred_union.dtype)
    extent_loss = (torch.nn.functional.smooth_l1_loss(pred_extents, target_extents, reduction="none").mean(dim=1) * extent_valid).sum()
    extent_loss = extent_loss / extent_valid.sum().clamp_min(1.0)
    teacher_extents = soft_side_extents_from_mask(teacher_union, touch_flags)
    teacher_extent_loss = (torch.nn.functional.smooth_l1_loss(pred_extents, teacher_extents, reduction="none").mean(dim=1) * extent_valid).sum()
    teacher_extent_loss = teacher_extent_loss / extent_valid.sum().clamp_min(1.0)
    teacher_boundary_loss = teacher_boundary_distance_loss(pred_union, teacher_union, sigma=teacher_boundary_sigma)
    teacher_directional_loss = directional_strip_mass_loss(pred_union, teacher_union, touch_flags, strip_fracs=directional_strip_fracs)

    loss = (
        float(geometry_loss_weight) * geometry_loss
        + 0.25 * slot_loss
        + float(union_dice_weight) * union_dice
        + float(union_bce_weight) * union_bce
        + 0.5 * touch_fn_bce
        + float(boundary_weight) * boundary_loss
        + float(touch_boundary_weight) * touch_boundary
        + float(multi_scale_weight) * ms_loss
        + float(sdf_weight) * sdf_loss
        + float(overlap_weight) * overlap_loss
        + float(extent_loss_weight) * extent_loss
        + float(teacher_extent_weight) * teacher_extent_loss
        + float(teacher_boundary_weight) * teacher_boundary_loss
        + float(teacher_directional_weight) * teacher_directional_loss
        + float(touch_strip_weight) * strip_loss
    )
    soft_recall_per_sample = (pred_union * target_mask).sum(dim=(1, 2)) / target_mask.sum(dim=(1, 2)).clamp_min(1.0)
    recall_threshold = torch.quantile(soft_recall_per_sample.detach(), min(max(float(worst_recall_quantile), 0.0), 1.0))
    recall_worst_mask = (soft_recall_per_sample.detach() <= recall_threshold).to(pred_union.dtype)
    worst_recall_loss = ((1.0 - soft_recall_per_sample) * recall_worst_mask).mean()
    loss = loss + float(worst_recall_boost) * worst_recall_loss
    with torch.no_grad():
        iou = (((pred_union > 0.5) & (target_mask > 0.5)).sum(dim=(1, 2)).float() /
               (((pred_union > 0.5) | (target_mask > 0.5)).sum(dim=(1, 2)).float().clamp_min(1.0))).mean()
    return loss, {
        "mask_iou": float(iou.detach().cpu()),
        "geometry_loss": float(geometry_loss.detach().cpu()),
        "center_loss": float(center_loss.detach().cpu()),
        "shape_loss": float(shape_loss.detach().cpu()),
        "slot_loss": float(slot_loss.detach().cpu()),
        "union_dice": float(union_dice.detach().cpu()),
        "union_bce": float(union_bce.detach().cpu()),
        "touch_fn_bce": float(touch_fn_bce.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "touch_boundary_loss": float(touch_boundary.detach().cpu()),
        "ms_loss": float(ms_loss.detach().cpu()),
        "sdf_loss": float(sdf_loss.detach().cpu()),
        "overlap_loss": float(overlap_loss.detach().cpu()),
        "extent_loss": float(extent_loss.detach().cpu()),
        "teacher_extent_loss": float(teacher_extent_loss.detach().cpu()),
        "teacher_boundary_loss": float(teacher_boundary_loss.detach().cpu()),
        "teacher_directional_loss": float(teacher_directional_loss.detach().cpu()),
        "strip2_loss": float(strip_loss.detach().cpu()),
        "strip2_coverage": float(strip_coverage.mean().detach().cpu()),
        "worst_recall_frac": float(recall_worst_mask.mean().detach().cpu()),
        "soft_recall": float(soft_recall_per_sample.mean().detach().cpu()),
        "worst_recall_loss": float(worst_recall_loss.detach().cpu()),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    epoch_index: int,
    total_epochs: int,
    split_name: str,
    global_step_start: int,
    training_started_at: float,
    step_log_interval: int,
    step_log_path: Path | None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    total_iou = 0.0
    total_count = 0
    skipped_batches = 0
    global_step = int(global_step_start)
    stat_sums: dict[str, float] = {}
    stat_window_sums: dict[str, float] = {}
    window_count = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        image = batch["input_image"].to(device, non_blocking=True)
        target_states = batch["target_states"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        target_signed = batch["target_signed"].to(device, non_blocking=True)
        touch_flags = batch["touch_flags"].to(device, non_blocking=True)
        if train_mode:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        pred = model(image)
        loss, stats = slot_set_spd_loss(
            pred,
            target_states,
            target_mask,
            target_signed,
            touch_flags,
            geometry_loss_weight=float(JOB_CONFIG["geometry_loss_weight"]),
            center_loss_weight=float(JOB_CONFIG["center_loss_weight"]),
            shape_loss_weight=float(JOB_CONFIG["shape_loss_weight"]),
            extent_loss_weight=float(JOB_CONFIG["extent_loss_weight"]),
            edge_geometry_boost=float(JOB_CONFIG["edge_geometry_boost"]),
            teacher_extent_weight=float(JOB_CONFIG["teacher_extent_weight"]),
            teacher_boundary_weight=float(JOB_CONFIG["teacher_boundary_weight"]),
            teacher_directional_weight=float(JOB_CONFIG["teacher_directional_weight"]),
            teacher_boundary_sigma=float(JOB_CONFIG["teacher_boundary_sigma"]),
            directional_strip_fracs=list(JOB_CONFIG["directional_strip_fracs"]),
            union_dice_weight=float(JOB_CONFIG["union_dice_weight"]),
            union_bce_weight=float(JOB_CONFIG["union_bce_weight"]),
            fn_weight=float(JOB_CONFIG["fn_weight"]),
            boundary_weight=float(JOB_CONFIG["boundary_weight"]),
            touch_boundary_weight=float(JOB_CONFIG["touch_boundary_weight"]),
            touch_boundary_band_width=float(JOB_CONFIG["touch_boundary_band_width"]),
            touch_fn_weight=float(JOB_CONFIG["touch_fn_weight"]),
            multi_scale_weight=float(JOB_CONFIG["multi_scale_weight"]),
            sdf_weight=float(JOB_CONFIG["sdf_weight"]),
            overlap_weight=float(JOB_CONFIG["overlap_weight"]),
            boundary_band_width=float(JOB_CONFIG["boundary_band_width"]),
            touch_strip_width_px=int(JOB_CONFIG["touch_strip_width_px"]),
            touch_strip_weight=float(JOB_CONFIG["touch_strip_weight"]),
            worst_recall_quantile=float(JOB_CONFIG["worst_recall_quantile"]),
            worst_recall_boost=float(JOB_CONFIG["worst_recall_boost"]),
        )
        if train_mode:
            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            if float(JOB_CONFIG["grad_clip_norm"]) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(JOB_CONFIG["grad_clip_norm"]))
            optimizer.step()
            global_step += 1

        batch_size = int(image.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_iou += float(stats["mask_iou"]) * batch_size
        total_count += batch_size
        stat_sums["loss"] = stat_sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        for key, value in stats.items():
            stat_sums[key] = stat_sums.get(key, 0.0) + float(value) * batch_size
            stat_window_sums[key] = stat_window_sums.get(key, 0.0) + float(value) * batch_size
        stat_window_sums["loss"] = stat_window_sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        window_count += batch_size

        if train_mode and int(step_log_interval) > 0 and global_step % int(step_log_interval) == 0:
            elapsed = float(time.time() - training_started_at)
            progress = {
                "type": "step",
                "split": split_name,
                "epoch": int(epoch_index),
                "total_epochs": int(total_epochs),
                "batch_index": int(batch_idx),
                "num_batches": int(num_batches),
                "global_step": int(global_step),
                "window_samples": int(window_count),
                "elapsed_from_start_sec": elapsed,
                "elapsed_from_start_hms": format_duration(elapsed),
                "lr": float(optimizer.param_groups[0]["lr"]) if optimizer is not None else None,
                "loss": stat_window_sums["loss"] / max(window_count, 1),
            }
            for key, value in stat_window_sums.items():
                if key == "loss":
                    continue
                progress[key] = value / max(window_count, 1)
            line = json.dumps(progress, ensure_ascii=False)
            print(line, flush=True)
            if step_log_path is not None:
                with step_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            stat_window_sums = {}
            window_count = 0

    result = {
        "loss": total_loss / max(total_count, 1),
        "mask_iou": total_iou / max(total_count, 1),
        "skipped_batches": float(skipped_batches),
        "global_step_end": float(global_step),
    }
    for key, value in stat_sums.items():
        if key == "loss":
            continue
        result[key] = value / max(total_count, 1)
    return result


def main() -> None:
    out_dir = Path(JOB_CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(JOB_CONFIG["seed"]))

    device = torch.device(str(JOB_CONFIG["device"]))
    manifest_started = time.time()
    manifest, manifest_cache_path, manifest_loaded_from_cache = build_or_load_manifest(
        jsonl_path=Path(JOB_CONFIG["train_jsonl"]),
        manifest_cache=Path(JOB_CONFIG["manifest_cache"]),
        image_size=int(JOB_CONFIG["image_size"]),
        val_ratio=float(JOB_CONFIG["val_ratio"]),
        max_rows=int(JOB_CONFIG["max_rows"]),
    )
    manifest_build_sec = float(time.time() - manifest_started)

    train_ds = K2SlotSetDataset(manifest, "train")
    val_ds = K2SlotSetDataset(manifest, "val")
    offset_to_row = {row.offset: row for row in val_ds.rows}

    train_loader = DataLoader(
        train_ds,
        batch_size=int(JOB_CONFIG["batch_size"]),
        shuffle=True,
        num_workers=int(JOB_CONFIG["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(JOB_CONFIG["num_workers"]) > 0,
        worker_init_fn=seed_worker if int(JOB_CONFIG["num_workers"]) > 0 else None,
        generator=torch.Generator().manual_seed(int(JOB_CONFIG["seed"])),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(JOB_CONFIG["batch_size"]),
        shuffle=False,
        num_workers=int(JOB_CONFIG["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(JOB_CONFIG["num_workers"]) > 0,
        worker_init_fn=seed_worker if int(JOB_CONFIG["num_workers"]) > 0 else None,
        generator=torch.Generator().manual_seed(int(JOB_CONFIG["seed"]) + 1),
    )

    model_build_started = time.time()
    model = K2SlotSetSPDNet(
        in_channels=10,
        base_width=int(JOB_CONFIG["base_width"]),
        slot_dim=int(JOB_CONFIG["slot_dim"]),
        decoder_layers=int(JOB_CONFIG["decoder_layers"]),
        num_heads=int(JOB_CONFIG["num_heads"]),
        sharpness=float(JOB_CONFIG["render_sharpness"]),
    ).to(device)
    model_build_sec = float(time.time() - model_build_started)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(JOB_CONFIG["lr"]), weight_decay=float(JOB_CONFIG["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(JOB_CONFIG["epochs"]), 1))
    checkpoint_loaded = False
    checkpoint_epoch = None
    init_checkpoint = JOB_CONFIG.get("init_checkpoint")
    if init_checkpoint:
        ckpt_path = Path(init_checkpoint)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            checkpoint_loaded = True
            checkpoint_epoch = checkpoint.get("epoch")
            if bool(JOB_CONFIG.get("load_optimizer_state", False)):
                opt_state = checkpoint.get("optimizer")
                sched_state = checkpoint.get("scheduler")
                if opt_state is not None:
                    optimizer.load_state_dict(opt_state)
                if sched_state is not None:
                    scheduler.load_state_dict(sched_state)
        else:
            raise FileNotFoundError(f"init_checkpoint not found: {ckpt_path}")

    run_config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in JOB_CONFIG.items()},
        "manifest_cache_resolved": str(manifest_cache_path),
        "manifest_loaded_from_cache": bool(manifest_loaded_from_cache),
        "manifest_build_sec": float(manifest_build_sec),
        "device_resolved": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "model_build_sec": float(model_build_sec),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_loaded": bool(checkpoint_loaded),
        "checkpoint_epoch": checkpoint_epoch,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_config, ensure_ascii=False), flush=True)

    wandb_run = None
    if bool(JOB_CONFIG.get("enable_wandb", False)):
        try:
            wandb_run = wandb.init(
                project=str(JOB_CONFIG["wandb_project"]),
                entity=JOB_CONFIG.get("wandb_entity"),
                name=str(JOB_CONFIG["wandb_run_name"]),
                mode=str(JOB_CONFIG.get("wandb_mode", "online")),
                tags=list(JOB_CONFIG.get("wandb_tags", [])),
                dir=str(out_dir),
                config=run_config,
                reinit="finish_previous",
            )
        except Exception as exc:
            print(json.dumps({"type": "wandb_init_failed", "error": str(exc)}, ensure_ascii=False), flush=True)
            wandb_run = None

    history: list[dict[str, object]] = []
    best_exact = -1.0
    best_epoch = -1
    training_started_at = time.time()
    global_step = 0
    step_log_path = out_dir / "step_metrics.jsonl"
    if step_log_path.exists():
        step_log_path.unlink()

    for epoch in range(1, int(JOB_CONFIG["epochs"]) + 1):
        started = time.time()
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch_index=int(epoch),
            total_epochs=int(JOB_CONFIG["epochs"]),
            split_name="train",
            global_step_start=global_step,
            training_started_at=training_started_at,
            step_log_interval=int(JOB_CONFIG["step_log_interval"]),
            step_log_path=step_log_path,
        )
        global_step = int(train_stats["global_step_end"])

        with torch.no_grad():
            val_stats = run_epoch(
                model,
                val_loader,
                None,
                device,
                epoch_index=int(epoch),
                total_epochs=int(JOB_CONFIG["epochs"]),
                split_name="val",
                global_step_start=global_step,
                training_started_at=training_started_at,
                step_log_interval=0,
                step_log_path=None,
            )

        exact = None
        if int(JOB_CONFIG["exact_eval_every"]) > 0 and (epoch % int(JOB_CONFIG["exact_eval_every"]) == 0 or epoch == int(JOB_CONFIG["epochs"])):
            model.eval()
            model_rows: list[dict[str, float]] = []
            teacher_rows: list[dict[str, float]] = []
            model_strip_coverages: list[float] = []
            teacher_strip_coverages: list[float] = []
            for batch in val_loader:
                image = batch["input_image"].to(device, non_blocking=True)
                offsets = batch["offset"].tolist()
                pred_output = model(image)
                pred_states = pred_output["states"].view(image.shape[0], 2, 6).detach().cpu().numpy()
                for local_idx, offset in enumerate(offsets):
                    row = offset_to_row[int(offset)]
                    _src_row, payload, _gt_square, gt_polys = load_source_context(val_ds.jsonl_path, int(offset))
                    pred_abs = states_to_abs_ellipses_from_payload(pred_states[local_idx], payload)
                    teacher_abs = load_target_abs_ellipses(_src_row)
                    pred_polys = fst.ellipses_to_polygon_arrays(pred_abs)
                    metrics = fst.compute_exact_metrics_from_polygons(gt_polys, pred_polys)
                    metrics["weighted_error"] = float(fst.compute_weighted_error(metrics))
                    model_rows.append(metrics)
                    teacher_rows.append(dict(row.teacher_exact_metrics))

                    local_teacher, side = absolute_to_local_square_ellipses_from_payload(teacher_abs, payload)
                    local_student, side2 = absolute_to_local_square_ellipses_from_payload(pred_abs, payload)
                    if side == side2:
                        teacher_mask = fst.render_ellipses((side, side), local_teacher, [1.0] * len(local_teacher)).astype(np.uint8)
                        student_mask = fst.render_ellipses((side, side), local_student, [1.0] * len(local_student)).astype(np.uint8)
                        gt_square = np.asarray(_gt_square, dtype=np.uint8)
                        model_strip_coverages.extend(
                            touch_strip_coverages_from_masks(
                                gt_square,
                                student_mask,
                                np.asarray(row.touch_flags, dtype=np.float32),
                                strip_width_px=int(JOB_CONFIG["touch_strip_width_px"]),
                            )
                        )
                        teacher_strip_coverages.extend(
                            touch_strip_coverages_from_masks(
                                gt_square,
                                teacher_mask,
                                np.asarray(row.touch_flags, dtype=np.float32),
                                strip_width_px=int(JOB_CONFIG["touch_strip_width_px"]),
                            )
                        )

            def summarize(rows: list[dict[str, float]], strip_coverages: list[float]) -> dict[str, float]:
                ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
                recalls = np.asarray([float(row["recall"]) for row in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
                precisions = np.asarray([float(row["precision"]) for row in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
                weighted_errors = np.asarray([float(row["weighted_error"]) for row in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
                total_inter = sum(float(row["intersection"]) for row in rows)
                total_union = sum(float(row["union"]) for row in rows)
                total_gt = sum(float(row["gt_area"]) for row in rows)
                total_pred = sum(float(row["pred_area"]) for row in rows)
                summary = {
                    "row_count": int(len(rows)),
                    "global_recall": total_inter / max(total_gt, 1.0),
                    "global_precision": total_inter / max(total_pred, 1.0),
                    "global_iou": total_inter / max(total_union, 1.0),
                    "mean_iou": float(np.mean(ious)) if ious.size else 0.0,
                    "weighted_error_mean": float(np.mean(weighted_errors)) if weighted_errors.size else 0.0,
                    "iou_std": float(np.std(ious)) if ious.size else 0.0,
                    "recall_std": float(np.std(recalls)) if recalls.size else 0.0,
                    "precision_std": float(np.std(precisions)) if precisions.size else 0.0,
                    "iou_p05": float(np.quantile(ious, 0.05)) if ious.size else 0.0,
                    "recall_p05": float(np.quantile(recalls, 0.05)) if recalls.size else 0.0,
                    "precision_p05": float(np.quantile(precisions, 0.05)) if precisions.size else 0.0,
                    "weighted_error_p95": float(np.quantile(weighted_errors, 0.95)) if weighted_errors.size else 0.0,
                    "worst_iou_count": int(sum(float(row["iou"]) < float(JOB_CONFIG["worst_iou_threshold"]) for row in rows)),
                    "worst_recall_count": int(sum(float(row["recall"]) < float(JOB_CONFIG["worst_recall_threshold"]) for row in rows)),
                    "worst_precision_count": int(sum(float(row["precision"]) < float(JOB_CONFIG["worst_precision_threshold"]) for row in rows)),
                }
                if strip_coverages:
                    summary["strip2_mean"] = float(np.mean(strip_coverages))
                    summary["strip2_p05"] = float(np.quantile(np.asarray(strip_coverages, dtype=np.float64), 0.05))
                    summary["strip2_count"] = int(len(strip_coverages))
                else:
                    summary["strip2_mean"] = 0.0
                    summary["strip2_p05"] = 0.0
                    summary["strip2_count"] = 0
                return summary

            exact = {
                "model": summarize(model_rows, model_strip_coverages),
                "teacher": summarize(teacher_rows, teacher_strip_coverages),
            }
            current_iou = float(exact["model"]["global_iou"])
            if current_iou > best_exact:
                best_exact = current_iou
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "config": run_config,
                    },
                    out_dir / "best_exact.pt",
                )

        scheduler.step()
        epoch_elapsed = float(time.time() - started)
        elapsed_from_start = float(time.time() - training_started_at)
        avg_epoch_sec = elapsed_from_start / float(epoch)
        eta_sec = avg_epoch_sec * float(max(int(JOB_CONFIG["epochs"]) - epoch, 0))
        row = {
            "epoch": int(epoch),
            "epoch_elapsed_sec": epoch_elapsed,
            "epoch_elapsed_hms": format_duration(epoch_elapsed),
            "elapsed_from_start_sec": elapsed_from_start,
            "elapsed_from_start_hms": format_duration(elapsed_from_start),
            "avg_epoch_sec": avg_epoch_sec,
            "avg_epoch_hms": format_duration(avg_epoch_sec),
            "eta_sec": eta_sec,
            "eta_hms": format_duration(eta_sec),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "global_step": int(global_step),
            "train_loss": float(train_stats["loss"]),
            "train_mask_iou": float(train_stats["mask_iou"]),
            "train_skipped_batches": int(train_stats["skipped_batches"]),
            "train_geometry_loss": float(train_stats.get("geometry_loss", 0.0)),
            "train_center_loss": float(train_stats.get("center_loss", 0.0)),
            "train_shape_loss": float(train_stats.get("shape_loss", 0.0)),
            "train_slot_loss": float(train_stats.get("slot_loss", 0.0)),
            "train_union_dice": float(train_stats.get("union_dice", 0.0)),
            "train_union_bce": float(train_stats.get("union_bce", 0.0)),
            "train_touch_fn_bce": float(train_stats.get("touch_fn_bce", 0.0)),
            "train_boundary_loss": float(train_stats.get("boundary_loss", 0.0)),
            "train_touch_boundary_loss": float(train_stats.get("touch_boundary_loss", 0.0)),
            "train_ms_loss": float(train_stats.get("ms_loss", 0.0)),
            "train_sdf_loss": float(train_stats.get("sdf_loss", 0.0)),
            "train_overlap_loss": float(train_stats.get("overlap_loss", 0.0)),
            "train_extent_loss": float(train_stats.get("extent_loss", 0.0)),
            "train_teacher_extent_loss": float(train_stats.get("teacher_extent_loss", 0.0)),
            "train_teacher_boundary_loss": float(train_stats.get("teacher_boundary_loss", 0.0)),
            "train_teacher_directional_loss": float(train_stats.get("teacher_directional_loss", 0.0)),
            "train_strip2_loss": float(train_stats.get("strip2_loss", 0.0)),
            "train_strip2_coverage": float(train_stats.get("strip2_coverage", 0.0)),
            "train_soft_recall": float(train_stats.get("soft_recall", 0.0)),
            "train_worst_recall_frac": float(train_stats.get("worst_recall_frac", 0.0)),
            "train_worst_recall_loss": float(train_stats.get("worst_recall_loss", 0.0)),
            "val_loss": float(val_stats["loss"]),
            "val_mask_iou": float(val_stats["mask_iou"]),
            "val_skipped_batches": int(val_stats["skipped_batches"]),
            "val_geometry_loss": float(val_stats.get("geometry_loss", 0.0)),
            "val_center_loss": float(val_stats.get("center_loss", 0.0)),
            "val_shape_loss": float(val_stats.get("shape_loss", 0.0)),
            "val_slot_loss": float(val_stats.get("slot_loss", 0.0)),
            "val_union_dice": float(val_stats.get("union_dice", 0.0)),
            "val_union_bce": float(val_stats.get("union_bce", 0.0)),
            "val_touch_fn_bce": float(val_stats.get("touch_fn_bce", 0.0)),
            "val_boundary_loss": float(val_stats.get("boundary_loss", 0.0)),
            "val_touch_boundary_loss": float(val_stats.get("touch_boundary_loss", 0.0)),
            "val_ms_loss": float(val_stats.get("ms_loss", 0.0)),
            "val_sdf_loss": float(val_stats.get("sdf_loss", 0.0)),
            "val_overlap_loss": float(val_stats.get("overlap_loss", 0.0)),
            "val_extent_loss": float(val_stats.get("extent_loss", 0.0)),
            "val_teacher_extent_loss": float(val_stats.get("teacher_extent_loss", 0.0)),
            "val_teacher_boundary_loss": float(val_stats.get("teacher_boundary_loss", 0.0)),
            "val_teacher_directional_loss": float(val_stats.get("teacher_directional_loss", 0.0)),
            "val_strip2_loss": float(val_stats.get("strip2_loss", 0.0)),
            "val_strip2_coverage": float(val_stats.get("strip2_coverage", 0.0)),
            "val_soft_recall": float(val_stats.get("soft_recall", 0.0)),
            "val_worst_recall_frac": float(val_stats.get("worst_recall_frac", 0.0)),
            "val_worst_recall_loss": float(val_stats.get("worst_recall_loss", 0.0)),
        }
        if exact is not None:
            row["exact"] = exact
            row["best_exact_global_iou_so_far"] = float(best_exact)
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"type": "epoch", **row}, ensure_ascii=False), flush=True)
        if wandb_run is not None:
            wandb.log(flatten_for_wandb(row), step=int(epoch))

    summary_plot_path = build_summary_plot(history, out_dir / "training_summary.png")
    final_summary = {
        "best_epoch": int(best_epoch),
        "best_exact_global_iou": float(best_exact),
        "teacher_exact_global_iou": float(history[-1]["exact"]["teacher"]["global_iou"]) if history and "exact" in history[-1] else None,
        "history_path": str(out_dir / "history.json"),
        "checkpoint_path": str(out_dir / "best_exact.pt"),
        "summary_plot_path": str(summary_plot_path) if summary_plot_path is not None else None,
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False), flush=True)
    if wandb_run is not None:
        for key, value in final_summary.items():
            if isinstance(value, (int, float, str)):
                wandb_run.summary[key] = value
        if summary_plot_path is not None and summary_plot_path.exists():
            wandb.log({"summary_plot": wandb.Image(str(summary_plot_path))}, step=max(len(history), 1))
        wandb.finish()


if __name__ == "__main__":
    main()
