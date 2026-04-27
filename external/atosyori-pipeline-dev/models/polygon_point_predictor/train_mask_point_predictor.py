from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_NAMES = [
    "area",
    "perimeter",
    "bbox_w",
    "bbox_h",
    "area_ratio",
    "compactness",
    "aspect_ratio",
    "extent",
    "solidity",
    "components",
    "holes",
    "eccentricity",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight mask-to-point-count predictor.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--label-min", type=int, default=4)
    parser.add_argument("--label-max", type=int, default=64)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--feature-hidden-dim", type=int, default=32)
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--feature-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stratified_split(
    rows: list[dict[str, object]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["teacher_output_points"])].append(row)
    rng = random.Random(seed)
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for label, group in sorted(grouped.items()):
        rng.shuffle(group)
        if len(group) == 1:
            train_rows.extend(group)
            continue
        val_count = max(1, int(round(len(group) * float(val_ratio))))
        val_count = min(val_count, len(group) - 1)
        val_rows.extend(group[:val_count])
        train_rows.extend(group[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def compute_feature_stats(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for row in rows:
        desc = row["descriptors"]
        feature_row = [
            math.log1p(float(desc["area"])),
            math.log1p(float(desc["perimeter"])),
            math.log1p(float(desc["bbox_w"])),
            math.log1p(float(desc["bbox_h"])),
            float(desc["area_ratio"]),
            float(desc["compactness"]),
            math.log1p(float(desc["aspect_ratio"])),
            float(desc["extent"]),
            float(desc["solidity"]),
            float(desc["components"]),
            float(desc["holes"]),
            float(desc["eccentricity"]),
        ]
        values.append(feature_row)
    array = np.asarray(values, dtype=np.float32)
    means = array.mean(axis=0)
    stds = array.std(axis=0)
    stds = np.clip(stds, 1.0e-6, None)
    return means, stds


def build_feature_vector(row: dict[str, object], means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    desc = row["descriptors"]
    values = np.asarray(
        [
            math.log1p(float(desc["area"])),
            math.log1p(float(desc["perimeter"])),
            math.log1p(float(desc["bbox_w"])),
            math.log1p(float(desc["bbox_h"])),
            float(desc["area_ratio"]),
            float(desc["compactness"]),
            math.log1p(float(desc["aspect_ratio"])),
            float(desc["extent"]),
            float(desc["solidity"]),
            float(desc["components"]),
            float(desc["holes"]),
            float(desc["eccentricity"]),
        ],
        dtype=np.float32,
    )
    return (values - means) / stds


def resize_mask_with_padding(mask: np.ndarray, image_size: int) -> np.ndarray:
    height, width = mask.shape[:2]
    scale = float(image_size) / float(max(height, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    offset_y = (image_size - new_h) // 2
    offset_x = (image_size - new_w) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


class MaskPointDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        *,
        dataset_root: Path,
        image_size: int,
        label_min: int,
        label_max: int,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        enable_feature_branch: bool,
        augment: bool,
    ) -> None:
        self.rows = rows
        self.dataset_root = dataset_root
        self.image_size = int(image_size)
        self.label_min = int(label_min)
        self.label_max = int(label_max)
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.enable_feature_branch = bool(enable_feature_branch)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        mask_path = self.dataset_root / str(row["mask_path"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = (mask > 0).astype(np.uint8) * 255
        mask = resize_mask_with_padding(mask, self.image_size)
        if self.augment:
            if random.random() < 0.5:
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                mask = np.ascontiguousarray(mask[::-1, :])
            rotations = random.randint(0, 3)
            if rotations:
                mask = np.ascontiguousarray(np.rot90(mask, k=rotations))
        image = mask.astype(np.float32) / 255.0
        image = image[None, :, :]

        label_value = int(row["teacher_output_points"])
        label_value = max(self.label_min, min(self.label_max, label_value))
        label_index = label_value - self.label_min

        if self.enable_feature_branch:
            features = build_feature_vector(row, self.feature_means, self.feature_stds)
        else:
            features = np.zeros((len(FEATURE_NAMES),), dtype=np.float32)

        return {
            "image": torch.from_numpy(image),
            "features": torch.from_numpy(features),
            "label": torch.tensor(label_index, dtype=torch.int64),
            "label_value": torch.tensor(label_value, dtype=torch.int64),
        }


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyMaskPointNet(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        use_feature_branch: bool,
        width_mult: float = 1.0,
        feature_hidden_dim: int = 32,
        head_hidden_dim: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.use_feature_branch = bool(use_feature_branch)
        def ch(value: int) -> int:
            return max(8, int(round(float(value) * float(width_mult))))

        stem_ch = ch(16)
        c1 = ch(24)
        c2 = ch(32)
        c3 = ch(48)
        c4 = ch(64)

        self.stem = ConvBNAct(1, stem_ch, stride=2)
        self.encoder = nn.Sequential(
            ConvBNAct(stem_ch, c1, stride=2),
            ConvBNAct(c1, c2, stride=2),
            ConvBNAct(c2, c3, stride=2),
            ConvBNAct(c3, c4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_head = nn.Sequential(
            nn.Linear(c4, head_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )
        if self.use_feature_branch:
            self.feature_head = nn.Sequential(
                nn.Linear(feature_dim, feature_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(feature_hidden_dim, feature_hidden_dim),
                nn.SiLU(inplace=True),
            )
            fusion_dim = head_hidden_dim + feature_hidden_dim
        else:
            self.feature_head = None
            fusion_dim = head_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        x = self.image_head(x)
        if self.use_feature_branch:
            f = self.feature_head(features)
            x = torch.cat([x, f], dim=1)
        return self.classifier(x)


def compute_class_weights(rows: list[dict[str, object]], label_min: int, label_max: int) -> torch.Tensor:
    counts = Counter(int(row["teacher_output_points"]) for row in rows)
    weights = []
    for label in range(label_min, label_max + 1):
        count = counts.get(label, 0)
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(1.0 / math.sqrt(float(count)))
    arr = np.asarray(weights, dtype=np.float32)
    positive = arr[arr > 0]
    if positive.size > 0:
        arr[arr > 0] /= float(positive.mean())
    return torch.from_numpy(arr)


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_min: int,
) -> dict[str, float]:
    pred_indices = logits.argmax(dim=1)
    pred_values = pred_indices + int(label_min)
    true_values = labels + int(label_min)
    abs_err = (pred_values - true_values).abs().float()
    under = (pred_values < true_values).float().mean()
    within1 = (abs_err <= 1.0).float().mean()
    exact = (pred_values == true_values).float().mean()
    return {
        "mae": float(abs_err.mean().item()),
        "rmse": float(torch.sqrt((abs_err**2).mean()).item()),
        "within1": float(within1.item()),
        "exact_acc": float(exact.item()),
        "underpredict_rate": float(under.item()),
    }


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    label_min: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            logits = model(image, features)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = evaluate_logits(logits, labels, label_min=label_min)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def write_split_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_history_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))

    all_rows = load_rows(args.input_jsonl)
    train_rows, val_rows = stratified_split(all_rows, val_ratio=float(args.val_ratio), seed=int(args.seed))
    feature_means, feature_stds = compute_feature_stats(train_rows)

    write_split_jsonl(output_dir / "train_split.jsonl", train_rows)
    write_split_jsonl(output_dir / "val_split.jsonl", val_rows)
    np.savez(output_dir / "feature_stats.npz", means=feature_means, stds=feature_stds, feature_names=np.asarray(FEATURE_NAMES))

    dataset_root = args.input_jsonl.parent
    train_ds = MaskPointDataset(
        train_rows,
        dataset_root=dataset_root,
        image_size=int(args.image_size),
        label_min=int(args.label_min),
        label_max=int(args.label_max),
        feature_means=feature_means,
        feature_stds=feature_stds,
        enable_feature_branch=bool(args.feature_branch),
        augment=bool(args.augment),
    )
    val_ds = MaskPointDataset(
        val_rows,
        dataset_root=dataset_root,
        image_size=int(args.image_size),
        label_min=int(args.label_min),
        label_max=int(args.label_max),
        feature_means=feature_means,
        feature_stds=feature_stds,
        enable_feature_branch=bool(args.feature_branch),
        augment=False,
    )

    pin_memory = torch.cuda.is_available() and str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=seed_worker if int(args.num_workers) > 0 else None,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=seed_worker if int(args.num_workers) > 0 else None,
        generator=torch.Generator().manual_seed(int(args.seed) + 1),
    )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(str(args.device))
    model = TinyMaskPointNet(
        feature_dim=len(FEATURE_NAMES),
        num_classes=int(args.label_max) - int(args.label_min) + 1,
        use_feature_branch=bool(args.feature_branch),
        width_mult=float(args.width_mult),
        feature_hidden_dim=int(args.feature_hidden_dim),
        head_hidden_dim=int(args.head_hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    class_weights = compute_class_weights(train_rows, int(args.label_min), int(args.label_max)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))

    run_config = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "train_count": int(len(train_ds)),
        "val_count": int(len(val_ds)),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "device": str(device),
        "seed": int(args.seed),
        "label_min": int(args.label_min),
        "label_max": int(args.label_max),
        "width_mult": float(args.width_mult),
        "feature_hidden_dim": int(args.feature_hidden_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "feature_branch": bool(args.feature_branch),
        "augment": bool(args.augment),
        "param_count": int(sum(p.numel() for p in model.parameters())),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    history: list[dict[str, object]] = []
    best_val_mae = float("inf")
    best_epoch = -1
    started_at = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_t0 = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            label_min=int(args.label_min),
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            criterion=criterion,
            label_min=int(args.label_min),
        )
        scheduler.step()
        epoch_row = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_mae": float(train_metrics["mae"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_within1": float(train_metrics["within1"]),
            "train_exact_acc": float(train_metrics["exact_acc"]),
            "train_underpredict_rate": float(train_metrics["underpredict_rate"]),
            "val_loss": float(val_metrics["loss"]),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_within1": float(val_metrics["within1"]),
            "val_exact_acc": float(val_metrics["exact_acc"]),
            "val_underpredict_rate": float(val_metrics["underpredict_rate"]),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_seconds": float(time.perf_counter() - epoch_t0),
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        if float(val_metrics["mae"]) < best_val_mae:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": int(epoch),
                    "val_metrics": val_metrics,
                    "run_config": run_config,
                    "feature_means": feature_means,
                    "feature_stds": feature_stds,
                    "feature_names": FEATURE_NAMES,
                },
                output_dir / "best.pt",
            )

    write_history_csv(output_dir / "history.csv", history)
    best_row = next((row for row in history if int(row["epoch"]) == int(best_epoch)), None)
    summary = {
        "run_config": run_config,
        "label_distribution_train": dict(sorted(Counter(int(r["teacher_output_points"]) for r in train_rows).items())),
        "label_distribution_val": dict(sorted(Counter(int(r["teacher_output_points"]) for r in val_rows).items())),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "best_row": best_row,
        "total_seconds": float(time.perf_counter() - started_at),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
