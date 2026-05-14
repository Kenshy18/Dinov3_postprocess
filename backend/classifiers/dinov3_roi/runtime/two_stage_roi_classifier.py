#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for ROI-feature based two-stage multiclass classification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

META_DIM = 5  # score, bbox_area_ratio, mask_area_ratio, meta4, meta5 (set-dependent)
ROI_FEATURE_SOURCES = (
    "box_head",
    "pooler_gap",
    "pooler_flatten",
    "box_head_plus_pooler_gap",
)
META_FEATURE_SETS = (
    "legacy_iou",
    "geo_v2",
)


def normalize_meta_feature_set(name: str) -> str:
    s = str(name).strip().lower()
    aliases = {
        "legacy": "legacy_iou",
        "iou": "legacy_iou",
        "legacy_iou": "legacy_iou",
        "geo": "geo_v2",
        "geometry": "geo_v2",
        "geo_v2": "geo_v2",
    }
    out = aliases.get(s, s)
    if out not in META_FEATURE_SETS:
        raise ValueError(f"unsupported meta feature set: {name}. choices={list(META_FEATURE_SETS)}")
    return out


def meta_feature_names_for_set(name: str) -> List[str]:
    s = normalize_meta_feature_set(name)
    if s == "legacy_iou":
        return ["score", "bbox_area_ratio", "mask_area_ratio", "mask_iou", "bbox_iou"]
    if s == "geo_v2":
        return ["score", "bbox_area_ratio", "mask_area_ratio", "log_aspect_ratio", "mask_over_bbox"]
    raise ValueError(f"unsupported meta feature set: {name}")


def _torch_load_compat(path: Path, map_location: str = "cpu"):
    """Torch load compatible with both pre/post PyTorch 2.6 defaults."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


@dataclass
class RoiFeaturePack:
    image_ids: torch.Tensor
    scores: torch.Tensor
    roi_feat: torch.Tensor
    bbox_area_ratio: torch.Tensor
    mask_area_ratio: torch.Tensor
    labels: torch.Tensor
    mask_iou: torch.Tensor
    bbox_iou: torch.Tensor
    file_names: List[str]
    source: List[str]
    class_ids: List[int]
    class_names: List[str]
    feature_source: str = "box_head"
    meta_feature_set: str = "legacy_iou"
    meta_feature_names: List[str] = field(default_factory=lambda: meta_feature_names_for_set("legacy_iou"))
    box_head_feat: Optional[torch.Tensor] = None
    box_pooler_feat: Optional[torch.Tensor] = None
    box_pooler_feat_expanded: Optional[torch.Tensor] = None
    mask_pooler_feat: Optional[torch.Tensor] = None
    box_pooler_gap: Optional[torch.Tensor] = None
    box_pooler_gap_expanded: Optional[torch.Tensor] = None
    mask_pooler_gap: Optional[torch.Tensor] = None

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_classes(self) -> int:
        return int(len(self.class_names))

    @property
    def feat_dim(self) -> int:
        if self.roi_feat.ndim != 2:
            raise ValueError(f"roi_feat must be [N, D], got {tuple(self.roi_feat.shape)}")
        return int(self.roi_feat.shape[1])


class RoiFeatureDataset(Dataset):
    def __init__(
        self,
        pack: RoiFeaturePack,
        include_meta: bool = True,
        max_samples: Optional[int] = None,
        include_box_head_feat: bool = False,
        include_box_pooler_feat: bool = False,
        include_box_pooler_feat_expanded: bool = False,
        include_mask_pooler_feat: bool = False,
        include_box_pooler_gap: bool = False,
        include_box_pooler_gap_expanded: bool = False,
        include_mask_pooler_gap: bool = False,
    ) -> None:
        self.pack = pack
        self.include_meta = bool(include_meta)
        self.include_box_head_feat = bool(include_box_head_feat)
        self.include_box_pooler_feat = bool(include_box_pooler_feat)
        self.include_box_pooler_feat_expanded = bool(include_box_pooler_feat_expanded)
        self.include_mask_pooler_feat = bool(include_mask_pooler_feat)
        self.include_box_pooler_gap = bool(include_box_pooler_gap)
        self.include_box_pooler_gap_expanded = bool(include_box_pooler_gap_expanded)
        self.include_mask_pooler_gap = bool(include_mask_pooler_gap)

        n = self.pack.num_samples
        if max_samples is not None:
            n = min(n, int(max_samples))
        self.indices = list(range(n))

        if self.include_box_head_feat and self.pack.box_head_feat is None:
            raise ValueError("include_box_head_feat=True but pack has no box_head_feat")
        if self.include_box_pooler_feat and self.pack.box_pooler_feat is None:
            raise ValueError("include_box_pooler_feat=True but pack has no box_pooler_feat")
        if self.include_box_pooler_feat_expanded and self.pack.box_pooler_feat_expanded is None:
            raise ValueError("include_box_pooler_feat_expanded=True but pack has no box_pooler_feat_expanded")
        if self.include_mask_pooler_feat and self.pack.mask_pooler_feat is None:
            raise ValueError("include_mask_pooler_feat=True but pack has no mask_pooler_feat")
        if self.include_box_pooler_gap and self.pack.box_pooler_gap is None:
            raise ValueError("include_box_pooler_gap=True but pack has no box_pooler_gap")
        if self.include_box_pooler_gap_expanded and self.pack.box_pooler_gap_expanded is None:
            raise ValueError("include_box_pooler_gap_expanded=True but pack has no box_pooler_gap_expanded")
        if self.include_mask_pooler_gap and self.pack.mask_pooler_gap is None:
            raise ValueError("include_mask_pooler_gap=True but pack has no mask_pooler_gap")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        feat = self.pack.roi_feat[idx].float()
        label = torch.tensor(int(self.pack.labels[idx]), dtype=torch.long)

        if self.include_meta:
            meta = torch.tensor(
                [
                    float(self.pack.scores[idx]),
                    float(self.pack.bbox_area_ratio[idx]),
                    float(self.pack.mask_area_ratio[idx]),
                    float(self.pack.mask_iou[idx]),
                    float(self.pack.bbox_iou[idx]),
                ],
                dtype=torch.float32,
            )
        else:
            meta = torch.zeros((META_DIM,), dtype=torch.float32)

        out = {
            "feature": feat,
            "meta": meta,
            "label": label,
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if self.include_box_head_feat:
            out["box_head_feat"] = self.pack.box_head_feat[idx].float()
        if self.include_box_pooler_feat:
            out["box_pooler_feat"] = self.pack.box_pooler_feat[idx].float()
        if self.include_box_pooler_feat_expanded:
            out["box_pooler_feat_expanded"] = self.pack.box_pooler_feat_expanded[idx].float()
        if self.include_mask_pooler_feat:
            out["mask_pooler_feat"] = self.pack.mask_pooler_feat[idx].float()
        if self.include_box_pooler_gap:
            out["box_pooler_gap"] = self.pack.box_pooler_gap[idx].float()
        if self.include_box_pooler_gap_expanded:
            out["box_pooler_gap_expanded"] = self.pack.box_pooler_gap_expanded[idx].float()
        if self.include_mask_pooler_gap:
            out["mask_pooler_gap"] = self.pack.mask_pooler_gap[idx].float()
        return out


class RoiFeatureClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)

        head_in = int(self.input_dim + (META_DIM if self.use_meta else 0))
        hidden_dim = max(16, int(hidden_dim))
        num_layers = max(1, int(num_layers))

        layers: List[nn.Module] = []
        current_dim = head_in
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=float(dropout)))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*layers)

    def forward(self, roi_feat: torch.Tensor, meta: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if roi_feat.ndim > 2:
            roi_feat = torch.flatten(roi_feat, start_dim=1)
        if roi_feat.ndim != 2:
            raise ValueError(f"roi_feat must be rank-2 after flatten, got shape={tuple(roi_feat.shape)}")
        if roi_feat.shape[1] != self.input_dim:
            raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")

        feat = roi_feat
        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            feat = torch.cat([feat, meta], dim=1)
        return self.head(feat)


class RoiSpatialConvClassifier(nn.Module):
    """ROIAlign flattened feature -> conv stack -> flatten -> MLP classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        conv_channels: Sequence[int] = (96, 64),
        conv_kernels: Sequence[int] = (1, 3),
        conv_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)

        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiSpatialConvClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        conv_channels = [int(v) for v in conv_channels]
        if any(v <= 0 for v in conv_channels):
            raise ValueError(f"conv_channels must be positive, got {conv_channels}")

        conv_kernels = [int(v) for v in conv_kernels]
        if any(v <= 0 for v in conv_kernels):
            raise ValueError(f"conv_kernels must be positive, got {conv_kernels}")

        if len(conv_kernels) == 1 and len(conv_channels) > 1:
            conv_kernels = conv_kernels * len(conv_channels)
        if len(conv_kernels) != len(conv_channels):
            raise ValueError(
                f"conv_kernels length must be 1 or equal to conv_channels length, "
                f"got kernels={conv_kernels}, channels={conv_channels}"
            )

        conv_layers: List[nn.Module] = []
        in_ch = self.pooler_channels
        for out_ch, k in zip(conv_channels, conv_kernels):
            pad = int(k // 2)
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=pad, bias=False))
            conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.SiLU(inplace=True))
            if conv_dropout > 0:
                conv_layers.append(nn.Dropout2d(p=float(conv_dropout)))
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        conv_flat_dim = int(in_ch * self.pooler_size * self.pooler_size)
        head_in = int(conv_flat_dim + (META_DIM if self.use_meta else 0))
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))

        head_layers: List[nn.Module] = []
        current_dim = head_in
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(self, roi_feat: torch.Tensor, meta: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if roi_feat.shape[1] != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            x = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            c, h, w = int(roi_feat.shape[1]), int(roi_feat.shape[2]), int(roi_feat.shape[3])
            if c != self.pooler_channels or h != self.pooler_size or w != self.pooler_size:
                raise ValueError(
                    f"roi_feat shape mismatch: expected [N,{self.pooler_channels},{self.pooler_size},{self.pooler_size}], "
                    f"got {tuple(roi_feat.shape)}"
                )
            x = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        x = self.conv(x)
        feat = torch.flatten(x, start_dim=1)

        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            feat = torch.cat([feat, meta], dim=1)
        return self.head(feat)


class RoiSpatialGapClassifier(nn.Module):
    """ROIAlign feature -> 1x1 conv -> depthwise conv -> 1x1 conv -> GAP -> FC."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        stem_channels: int = 64,
        mid_channels: int = 64,
        dw_kernel: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)

        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiSpatialGapClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        stem_channels = int(stem_channels)
        mid_channels = int(mid_channels)
        dw_kernel = int(dw_kernel)
        if stem_channels <= 0 or mid_channels <= 0:
            raise ValueError(
                f"stem_channels and mid_channels must be positive: stem={stem_channels}, mid={mid_channels}"
            )
        if dw_kernel <= 0:
            raise ValueError(f"dw_kernel must be positive, got {dw_kernel}")
        pad = int(dw_kernel // 2)

        self.stem = nn.Sequential(
            nn.Conv2d(self.pooler_channels, stem_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=dw_kernel,
                stride=1,
                padding=pad,
                groups=stem_channels,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(stem_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        head_in = int(mid_channels + (META_DIM if self.use_meta else 0))
        self.head = nn.Linear(head_in, int(num_classes))

    def forward(self, roi_feat: torch.Tensor, meta: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if roi_feat.shape[1] != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            x = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            c, h, w = int(roi_feat.shape[1]), int(roi_feat.shape[2]), int(roi_feat.shape[3])
            if c != self.pooler_channels or h != self.pooler_size or w != self.pooler_size:
                raise ValueError(
                    f"roi_feat shape mismatch: expected [N,{self.pooler_channels},{self.pooler_size},{self.pooler_size}], "
                    f"got {tuple(roi_feat.shape)}"
                )
            x = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        x = self.stem(x)
        x = self.depthwise(x)
        x = self.proj(x)
        feat = self.pool(x).flatten(start_dim=1)
        feat = self.dropout(feat)

        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            feat = torch.cat([feat, meta], dim=1)
        return self.head(feat)


class RoiRichGapFusionClassifier(nn.Module):
    """Fuse roi_flat + optional rich GAP features + meta through projected-branch MLP."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        use_roi_feat: bool = True,
        use_box_head_feat: bool = True,
        use_box_pooler_gap_expanded: bool = True,
        use_mask_pooler_gap: bool = True,
        roi_proj_dim: int = 512,
        box_head_proj_dim: int = 256,
        expanded_gap_proj_dim: int = 128,
        mask_gap_proj_dim: int = 128,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.use_roi_feat = bool(use_roi_feat)
        self.use_box_head_feat = bool(use_box_head_feat)
        self.use_box_pooler_gap_expanded = bool(use_box_pooler_gap_expanded)
        self.use_mask_pooler_gap = bool(use_mask_pooler_gap)
        self.input_dim = int(input_dim)

        if not (
            self.use_roi_feat
            or self.use_box_head_feat
            or self.use_box_pooler_gap_expanded
            or self.use_mask_pooler_gap
        ):
            raise ValueError("at least one rich feature branch must be enabled")

        if self.use_roi_feat:
            self.roi_proj = nn.Sequential(
                nn.Linear(self.input_dim, int(roi_proj_dim)),
                nn.ReLU(inplace=True),
            )
        else:
            self.roi_proj = None

        if self.use_box_head_feat:
            self.box_head_proj = nn.Sequential(
                nn.Linear(1024, int(box_head_proj_dim)),
                nn.ReLU(inplace=True),
            )
        else:
            self.box_head_proj = None

        if self.use_box_pooler_gap_expanded:
            self.exp_gap_proj = nn.Sequential(
                nn.Linear(256, int(expanded_gap_proj_dim)),
                nn.ReLU(inplace=True),
            )
        else:
            self.exp_gap_proj = None

        if self.use_mask_pooler_gap:
            self.mask_gap_proj = nn.Sequential(
                nn.Linear(256, int(mask_gap_proj_dim)),
                nn.ReLU(inplace=True),
            )
        else:
            self.mask_gap_proj = None

        fused_dim = 0
        if self.use_roi_feat:
            fused_dim += int(roi_proj_dim)
        if self.use_box_head_feat:
            fused_dim += int(box_head_proj_dim)
        if self.use_box_pooler_gap_expanded:
            fused_dim += int(expanded_gap_proj_dim)
        if self.use_mask_pooler_gap:
            fused_dim += int(mask_gap_proj_dim)
        if self.use_meta:
            fused_dim += META_DIM

        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = int(fused_dim)
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        box_pooler_gap_expanded: Optional[torch.Tensor] = None,
        mask_pooler_gap: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        chunks: List[torch.Tensor] = []

        if self.use_roi_feat:
            if roi_feat.ndim > 2:
                roi_feat = torch.flatten(roi_feat, start_dim=1)
            if roi_feat.ndim != 2 or int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(
                    f"roi_feat must be [N,{self.input_dim}] for RoiRichGapFusionClassifier, got {tuple(roi_feat.shape)}"
                )
            chunks.append(self.roi_proj(roi_feat))

        if self.use_box_head_feat:
            if box_head_feat is None:
                raise ValueError("box_head_feat is required")
            if box_head_feat.ndim != 2 or int(box_head_feat.shape[1]) != 1024:
                raise ValueError(f"box_head_feat must be [N,1024], got {tuple(box_head_feat.shape)}")
            chunks.append(self.box_head_proj(box_head_feat))

        if self.use_box_pooler_gap_expanded:
            if box_pooler_gap_expanded is None:
                raise ValueError("box_pooler_gap_expanded is required")
            if box_pooler_gap_expanded.ndim != 2 or int(box_pooler_gap_expanded.shape[1]) != 256:
                raise ValueError(
                    f"box_pooler_gap_expanded must be [N,256], got {tuple(box_pooler_gap_expanded.shape)}"
                )
            chunks.append(self.exp_gap_proj(box_pooler_gap_expanded))

        if self.use_mask_pooler_gap:
            if mask_pooler_gap is None:
                raise ValueError("mask_pooler_gap is required")
            if mask_pooler_gap.ndim != 2 or int(mask_pooler_gap.shape[1]) != 256:
                raise ValueError(f"mask_pooler_gap must be [N,256], got {tuple(mask_pooler_gap.shape)}")
            chunks.append(self.mask_gap_proj(mask_pooler_gap))

        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            chunks.append(meta)

        if not chunks:
            raise RuntimeError("no input chunks to fuse")
        feat = torch.cat(chunks, dim=1)
        return self.head(feat)


class RoiRichSpatialFusionClassifier(nn.Module):
    """Fuse local/expanded/mask spatial features + box_head + meta with light conv branches."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        mask_pooler_size: int = 14,
        local_branch_dim: int = 96,
        expanded_branch_dim: int = 96,
        mask_branch_dim: int = 96,
        box_head_proj_dim: int = 256,
        dw_kernel: int = 3,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        self.mask_pooler_size = int(mask_pooler_size)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiRichSpatialFusionClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        k = int(dw_kernel)
        if k <= 0:
            raise ValueError(f"dw_kernel must be >0, got {k}")
        pad = k // 2

        def _branch(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(self.pooler_channels, int(out_dim), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(out_dim)),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    int(out_dim),
                    int(out_dim),
                    kernel_size=k,
                    stride=1,
                    padding=pad,
                    groups=int(out_dim),
                    bias=False,
                ),
                nn.BatchNorm2d(int(out_dim)),
                nn.SiLU(inplace=True),
                nn.Conv2d(int(out_dim), int(out_dim), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(int(out_dim)),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        self.local_branch = _branch(int(local_branch_dim))
        self.expanded_branch = _branch(int(expanded_branch_dim))
        self.mask_branch = nn.Sequential(
            nn.Conv2d(self.pooler_channels, int(mask_branch_dim), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(mask_branch_dim)),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                int(mask_branch_dim),
                int(mask_branch_dim),
                kernel_size=k,
                stride=1,
                padding=pad,
                groups=int(mask_branch_dim),
                bias=False,
            ),
            nn.BatchNorm2d(int(mask_branch_dim)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(mask_branch_dim), int(mask_branch_dim), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(mask_branch_dim)),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.box_head_proj = nn.Sequential(
            nn.Linear(1024, int(box_head_proj_dim)),
            nn.ReLU(inplace=True),
        )

        fused_dim = int(local_branch_dim + expanded_branch_dim + mask_branch_dim + box_head_proj_dim)
        if self.use_meta:
            fused_dim += META_DIM
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        box_pooler_feat_expanded: Optional[torch.Tensor] = None,
        mask_pooler_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            local = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            local = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        if box_pooler_feat_expanded is None:
            raise ValueError("box_pooler_feat_expanded is required")
        if mask_pooler_feat is None:
            raise ValueError("mask_pooler_feat is required")
        if box_head_feat is None:
            raise ValueError("box_head_feat is required")
        if box_pooler_feat_expanded.ndim != 4:
            raise ValueError(f"box_pooler_feat_expanded must be rank-4, got {tuple(box_pooler_feat_expanded.shape)}")
        if mask_pooler_feat.ndim != 4:
            raise ValueError(f"mask_pooler_feat must be rank-4, got {tuple(mask_pooler_feat.shape)}")
        if box_head_feat.ndim != 2 or int(box_head_feat.shape[1]) != 1024:
            raise ValueError(f"box_head_feat must be [N,1024], got {tuple(box_head_feat.shape)}")

        local_vec = self.local_branch(local).flatten(start_dim=1)
        expanded_vec = self.expanded_branch(box_pooler_feat_expanded).flatten(start_dim=1)
        mask_vec = self.mask_branch(mask_pooler_feat).flatten(start_dim=1)
        box_vec = self.box_head_proj(box_head_feat)

        chunks = [local_vec, expanded_vec, mask_vec, box_vec]
        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            chunks.append(meta)
        fused = torch.cat(chunks, dim=1)
        return self.head(fused)


def _build_dwconv_gap_branch(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    out_ch = int(out_ch)
    k = int(k)
    pad = int(k // 2)
    return nn.Sequential(
        nn.Conv2d(int(in_ch), out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=1, padding=pad, groups=out_ch, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )


class RoiRichSpatialGatedFusionClassifier(nn.Module):
    """Dynamic branch weighting (local/expanded/mask) before fusion head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        mask_pooler_size: int = 14,
        local_branch_dim: int = 128,
        expanded_branch_dim: int = 128,
        mask_branch_dim: int = 128,
        box_head_proj_dim: int = 320,
        dw_kernel: int = 3,
        gate_hidden_dim: int = 192,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        self.mask_pooler_size = int(mask_pooler_size)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiRichSpatialGatedFusionClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        k = int(dw_kernel)
        if k <= 0:
            raise ValueError(f"dw_kernel must be >0, got {k}")

        self.local_branch = _build_dwconv_gap_branch(self.pooler_channels, int(local_branch_dim), k=k)
        self.expanded_branch = _build_dwconv_gap_branch(self.pooler_channels, int(expanded_branch_dim), k=k)
        self.mask_branch = _build_dwconv_gap_branch(self.pooler_channels, int(mask_branch_dim), k=k)
        self.box_head_proj = nn.Sequential(
            nn.Linear(1024, int(box_head_proj_dim)),
            nn.ReLU(inplace=True),
        )

        gate_in = int(box_head_proj_dim + (META_DIM if self.use_meta else 0))
        gate_hidden_dim = max(16, int(gate_hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 3),
        )

        fused_dim = int(local_branch_dim + expanded_branch_dim + mask_branch_dim + box_head_proj_dim)
        if self.use_meta:
            fused_dim += META_DIM
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        box_pooler_feat_expanded: Optional[torch.Tensor] = None,
        mask_pooler_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            local = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            local = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        if box_pooler_feat_expanded is None or box_pooler_feat_expanded.ndim != 4:
            raise ValueError("box_pooler_feat_expanded (rank-4) is required")
        if mask_pooler_feat is None or mask_pooler_feat.ndim != 4:
            raise ValueError("mask_pooler_feat (rank-4) is required")
        if box_head_feat is None or box_head_feat.ndim != 2 or int(box_head_feat.shape[1]) != 1024:
            raise ValueError("box_head_feat [N,1024] is required")

        local_vec = self.local_branch(local).flatten(start_dim=1)
        expanded_vec = self.expanded_branch(box_pooler_feat_expanded).flatten(start_dim=1)
        mask_vec = self.mask_branch(mask_pooler_feat).flatten(start_dim=1)
        box_vec = self.box_head_proj(box_head_feat)

        gate_in = box_vec if not self.use_meta else torch.cat([box_vec, meta], dim=1)
        gate_w = torch.softmax(self.gate(gate_in), dim=1)
        local_vec = local_vec * gate_w[:, 0:1]
        expanded_vec = expanded_vec * gate_w[:, 1:2]
        mask_vec = mask_vec * gate_w[:, 2:3]

        chunks = [local_vec, expanded_vec, mask_vec, box_vec]
        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            chunks.append(meta)
        fused = torch.cat(chunks, dim=1)
        return self.head(fused)


class RoiRichSpatialAttnFusionClassifier(nn.Module):
    """Cross-branch token attention on local/expanded/mask/box descriptors."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        mask_pooler_size: int = 14,
        local_branch_dim: int = 128,
        expanded_branch_dim: int = 128,
        mask_branch_dim: int = 128,
        box_head_proj_dim: int = 320,
        dw_kernel: int = 3,
        attn_token_dim: int = 192,
        attn_num_heads: int = 4,
        attn_dropout: float = 0.0,
        token_pool_type: str = "mean",
        branch_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        self.mask_pooler_size = int(mask_pooler_size)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiRichSpatialAttnFusionClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        k = int(dw_kernel)
        if k <= 0:
            raise ValueError(f"dw_kernel must be >0, got {k}")
        token_dim = int(attn_token_dim)
        if token_dim <= 0:
            raise ValueError(f"attn_token_dim must be >0, got {token_dim}")
        num_heads = int(attn_num_heads)
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError(f"attn_num_heads must divide attn_token_dim: heads={num_heads}, dim={token_dim}")
        self.token_pool_type = str(token_pool_type).strip().lower()
        if self.token_pool_type not in {"mean", "attn"}:
            raise ValueError(f"token_pool_type must be one of ['mean','attn'], got {token_pool_type}")
        self.branch_dropout = float(branch_dropout)
        if not (0.0 <= self.branch_dropout < 1.0):
            raise ValueError(f"branch_dropout must be in [0,1), got {branch_dropout}")

        self.local_branch = _build_dwconv_gap_branch(self.pooler_channels, int(local_branch_dim), k=k)
        self.expanded_branch = _build_dwconv_gap_branch(self.pooler_channels, int(expanded_branch_dim), k=k)
        self.mask_branch = _build_dwconv_gap_branch(self.pooler_channels, int(mask_branch_dim), k=k)
        self.box_head_proj = nn.Sequential(
            nn.Linear(1024, int(box_head_proj_dim)),
            nn.ReLU(inplace=True),
        )

        self.local_token = nn.Linear(int(local_branch_dim), token_dim)
        self.expanded_token = nn.Linear(int(expanded_branch_dim), token_dim)
        self.mask_token = nn.Linear(int(mask_branch_dim), token_dim)
        self.box_token = nn.Linear(int(box_head_proj_dim), token_dim)

        self.pre_ln = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ffn_ln = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.token_pool = None
        if self.token_pool_type == "attn":
            self.token_pool = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 1, bias=False),
            )

        fused_dim = token_dim
        if self.use_meta:
            fused_dim += META_DIM
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        box_pooler_feat_expanded: Optional[torch.Tensor] = None,
        mask_pooler_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            local = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            local = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        if box_pooler_feat_expanded is None or box_pooler_feat_expanded.ndim != 4:
            raise ValueError("box_pooler_feat_expanded (rank-4) is required")
        if mask_pooler_feat is None or mask_pooler_feat.ndim != 4:
            raise ValueError("mask_pooler_feat (rank-4) is required")
        if box_head_feat is None or box_head_feat.ndim != 2 or int(box_head_feat.shape[1]) != 1024:
            raise ValueError("box_head_feat [N,1024] is required")

        local_vec = self.local_branch(local).flatten(start_dim=1)
        expanded_vec = self.expanded_branch(box_pooler_feat_expanded).flatten(start_dim=1)
        mask_vec = self.mask_branch(mask_pooler_feat).flatten(start_dim=1)
        box_vec = self.box_head_proj(box_head_feat)

        tokens = torch.stack(
            [
                self.local_token(local_vec),
                self.expanded_token(expanded_vec),
                self.mask_token(mask_vec),
                self.box_token(box_vec),
            ],
            dim=1,
        )
        if self.training and self.branch_dropout > 0.0:
            keep_prob = 1.0 - self.branch_dropout
            keep = (torch.rand(tokens.shape[:2], device=tokens.device) < keep_prob).float()
            drop_all = keep.sum(dim=1) <= 0.0
            if bool(drop_all.any()):
                keep[drop_all, 0] = 1.0
            tokens = tokens * (keep.unsqueeze(-1) / keep_prob)

        attn_in = self.pre_ln(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_ln(tokens))
        if self.token_pool_type == "attn":
            assert self.token_pool is not None
            pool_scores = self.token_pool(tokens).squeeze(-1)
            pool_weights = torch.softmax(pool_scores, dim=1)
            pooled = torch.sum(tokens * pool_weights.unsqueeze(-1), dim=1)
        else:
            pooled = tokens.mean(dim=1)

        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            pooled = torch.cat([pooled, meta], dim=1)
        return self.head(pooled)


class RoiRichSpatialAttnNoExpandedFusionClassifier(nn.Module):
    """Attention fusion without the extra expanded box ROIAlign branch."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        mask_pooler_size: int = 14,
        local_branch_dim: int = 128,
        mask_branch_dim: int = 128,
        box_head_proj_dim: int = 320,
        dw_kernel: int = 3,
        attn_token_dim: int = 192,
        attn_num_heads: int = 4,
        attn_dropout: float = 0.0,
        token_pool_type: str = "mean",
        branch_dropout: float = 0.0,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        self.mask_pooler_size = int(mask_pooler_size)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiRichSpatialAttnNoExpandedFusionClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        k = int(dw_kernel)
        if k <= 0:
            raise ValueError(f"dw_kernel must be >0, got {k}")
        token_dim = int(attn_token_dim)
        if token_dim <= 0:
            raise ValueError(f"attn_token_dim must be >0, got {token_dim}")
        num_heads = int(attn_num_heads)
        if num_heads <= 0 or token_dim % num_heads != 0:
            raise ValueError(f"attn_num_heads must divide attn_token_dim: heads={num_heads}, dim={token_dim}")
        self.token_pool_type = str(token_pool_type).strip().lower()
        if self.token_pool_type not in {"mean", "attn"}:
            raise ValueError(f"token_pool_type must be one of ['mean','attn'], got {token_pool_type}")
        self.branch_dropout = float(branch_dropout)
        if not (0.0 <= self.branch_dropout < 1.0):
            raise ValueError(f"branch_dropout must be in [0,1), got {branch_dropout}")

        self.local_branch = _build_dwconv_gap_branch(self.pooler_channels, int(local_branch_dim), k=k)
        self.mask_branch = _build_dwconv_gap_branch(self.pooler_channels, int(mask_branch_dim), k=k)
        self.box_head_proj = nn.Sequential(
            nn.Linear(1024, int(box_head_proj_dim)),
            nn.ReLU(inplace=True),
        )

        self.local_token = nn.Linear(int(local_branch_dim), token_dim)
        self.mask_token = nn.Linear(int(mask_branch_dim), token_dim)
        self.box_token = nn.Linear(int(box_head_proj_dim), token_dim)

        self.pre_ln = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.ffn_ln = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.token_pool = None
        if self.token_pool_type == "attn":
            self.token_pool = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 1, bias=False),
            )

        fused_dim = token_dim
        if self.use_meta:
            fused_dim += META_DIM
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        mask_pooler_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            local = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            local = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        if mask_pooler_feat is None or mask_pooler_feat.ndim != 4:
            raise ValueError("mask_pooler_feat (rank-4) is required")
        if box_head_feat is None or box_head_feat.ndim != 2 or int(box_head_feat.shape[1]) != 1024:
            raise ValueError("box_head_feat [N,1024] is required")

        local_vec = self.local_branch(local).flatten(start_dim=1)
        mask_vec = self.mask_branch(mask_pooler_feat).flatten(start_dim=1)
        box_vec = self.box_head_proj(box_head_feat)

        tokens = torch.stack(
            [
                self.local_token(local_vec),
                self.mask_token(mask_vec),
                self.box_token(box_vec),
            ],
            dim=1,
        )
        if self.training and self.branch_dropout > 0.0:
            keep_prob = 1.0 - self.branch_dropout
            keep = (torch.rand(tokens.shape[:2], device=tokens.device) < keep_prob).float()
            drop_all = keep.sum(dim=1) <= 0.0
            if bool(drop_all.any()):
                keep[drop_all, 0] = 1.0
            tokens = tokens * (keep.unsqueeze(-1) / keep_prob)

        attn_in = self.pre_ln(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_ln(tokens))
        if self.token_pool_type == "attn":
            assert self.token_pool is not None
            pool_scores = self.token_pool(tokens).squeeze(-1)
            pool_weights = torch.softmax(pool_scores, dim=1)
            pooled = torch.sum(tokens * pool_weights.unsqueeze(-1), dim=1)
        else:
            pooled = tokens.mean(dim=1)

        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            pooled = torch.cat([pooled, meta], dim=1)
        return self.head(pooled)


class RoiRichSpatialMaskGuidedFusionClassifier(nn.Module):
    """Use mask feature to spatially guide local/expanded pooling before fusion."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        use_meta: bool = True,
        pooler_channels: int = 256,
        pooler_size: int = 7,
        mask_pooler_size: int = 14,
        local_branch_dim: int = 128,
        expanded_branch_dim: int = 128,
        mask_branch_dim: int = 128,
        box_head_proj_dim: int = 320,
        dw_kernel: int = 3,
        mask_guided_alpha: float = 1.0,
        head_hidden_dim: int = 512,
        head_num_layers: int = 2,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_meta = bool(use_meta)
        self.input_dim = int(input_dim)
        self.pooler_channels = int(pooler_channels)
        self.pooler_size = int(pooler_size)
        self.mask_pooler_size = int(mask_pooler_size)
        self.mask_guided_alpha = float(mask_guided_alpha)
        expected_input_dim = self.pooler_channels * self.pooler_size * self.pooler_size
        if self.input_dim != expected_input_dim:
            raise ValueError(
                f"input_dim mismatch for RoiRichSpatialMaskGuidedFusionClassifier: "
                f"expected {expected_input_dim} (=C*H*W), got {self.input_dim}"
            )

        k = int(dw_kernel)
        if k <= 0:
            raise ValueError(f"dw_kernel must be >0, got {k}")

        self.mask_guidance_conv = nn.Conv2d(self.pooler_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_branch = _build_dwconv_gap_branch(self.pooler_channels, int(local_branch_dim), k=k)
        self.expanded_branch = _build_dwconv_gap_branch(self.pooler_channels, int(expanded_branch_dim), k=k)
        self.mask_branch = _build_dwconv_gap_branch(self.pooler_channels, int(mask_branch_dim), k=k)
        self.box_head_proj = nn.Sequential(
            nn.Linear(1024, int(box_head_proj_dim)),
            nn.ReLU(inplace=True),
        )

        fused_dim = int(local_branch_dim + expanded_branch_dim + mask_branch_dim + box_head_proj_dim)
        if self.use_meta:
            fused_dim += META_DIM
        head_hidden_dim = max(16, int(head_hidden_dim))
        head_num_layers = max(1, int(head_num_layers))
        head_layers: List[nn.Module] = []
        current_dim = fused_dim
        for _ in range(head_num_layers - 1):
            head_layers.append(nn.Linear(current_dim, head_hidden_dim))
            head_layers.append(nn.ReLU(inplace=True))
            head_layers.append(nn.Dropout(p=float(head_dropout)))
            current_dim = head_hidden_dim
        head_layers.append(nn.Linear(current_dim, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        roi_feat: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        box_head_feat: Optional[torch.Tensor] = None,
        box_pooler_feat_expanded: Optional[torch.Tensor] = None,
        mask_pooler_feat: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if roi_feat.ndim == 2:
            if int(roi_feat.shape[1]) != self.input_dim:
                raise ValueError(f"roi_feat dim mismatch: expected {self.input_dim}, got {int(roi_feat.shape[1])}")
            local = roi_feat.contiguous().view(-1, self.pooler_channels, self.pooler_size, self.pooler_size)
        elif roi_feat.ndim == 4:
            local = roi_feat
        else:
            raise ValueError(f"roi_feat must be rank-2 or rank-4, got shape={tuple(roi_feat.shape)}")

        if box_pooler_feat_expanded is None or box_pooler_feat_expanded.ndim != 4:
            raise ValueError("box_pooler_feat_expanded (rank-4) is required")
        if mask_pooler_feat is None or mask_pooler_feat.ndim != 4:
            raise ValueError("mask_pooler_feat (rank-4) is required")
        if box_head_feat is None or box_head_feat.ndim != 2 or int(box_head_feat.shape[1]) != 1024:
            raise ValueError("box_head_feat [N,1024] is required")

        guide14 = torch.sigmoid(self.mask_guidance_conv(mask_pooler_feat))
        guide7 = torch.nn.functional.interpolate(
            guide14, size=(self.pooler_size, self.pooler_size), mode="bilinear", align_corners=False
        )
        local_guided = local * (1.0 + self.mask_guided_alpha * guide7)
        expanded_guided = box_pooler_feat_expanded * (1.0 + self.mask_guided_alpha * guide7)

        local_vec = self.local_branch(local_guided).flatten(start_dim=1)
        expanded_vec = self.expanded_branch(expanded_guided).flatten(start_dim=1)
        mask_vec = self.mask_branch(mask_pooler_feat).flatten(start_dim=1)
        box_vec = self.box_head_proj(box_head_feat)

        chunks = [local_vec, expanded_vec, mask_vec, box_vec]
        if self.use_meta:
            if meta is None:
                raise ValueError("meta tensor is required when use_meta=True")
            chunks.append(meta)
        fused = torch.cat(chunks, dim=1)
        return self.head(fused)


def _to_tensor_2d(x: torch.Tensor | np.ndarray | List[float], name: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim > 2:
        t = torch.flatten(t, start_dim=1)
    if t.ndim != 2:
        raise ValueError(f"{name} must become [N, D], got shape={tuple(t.shape)}")
    return t.float()


def _to_tensor_4d(x: torch.Tensor | np.ndarray, name: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x)
    if t.ndim != 4:
        raise ValueError(f"{name} must be rank-4 [N,C,H,W], got {tuple(t.shape)}")
    return t


def normalize_roi_feature_source(source: str) -> str:
    s = str(source).strip().lower()
    s = s.replace("+", "_plus_")
    aliases = {
        "boxhead": "box_head",
        "box_head": "box_head",
        "pooler": "pooler_gap",
        "roi_align_gap": "pooler_gap",
        "pooler_gap": "pooler_gap",
        "roi_align_flatten": "pooler_flatten",
        "pooler_flatten": "pooler_flatten",
        "box_head_plus_pooler_gap": "box_head_plus_pooler_gap",
        "boxhead_plus_pooler_gap": "box_head_plus_pooler_gap",
    }
    out = aliases.get(s, s)
    if out not in ROI_FEATURE_SOURCES:
        raise ValueError(f"unsupported roi feature source: {source}. choices={list(ROI_FEATURE_SOURCES)}")
    return out


def roi_feature_source_needs(source: str) -> Tuple[bool, bool]:
    src = normalize_roi_feature_source(source)
    need_box_head = src in ("box_head", "box_head_plus_pooler_gap")
    need_pooler = src in ("pooler_gap", "pooler_flatten", "box_head_plus_pooler_gap")
    return need_box_head, need_pooler


def load_roi_feature_pack(pt_path: Path) -> RoiFeaturePack:
    p = Path(pt_path)
    if not p.is_file():
        raise FileNotFoundError(f"ROI feature pack not found: {p}")

    raw = _torch_load_compat(p, map_location="cpu")
    required = [
        "image_id",
        "score",
        "roi_feat",
        "bbox_area_ratio",
        "mask_area_ratio",
        "label_idx",
        "mask_iou",
        "bbox_iou",
        "file_name",
        "source",
        "index_to_class_id",
        "index_to_class_name",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"missing keys in ROI feature pack {p}: {missing}")

    roi_feat = _to_tensor_2d(raw["roi_feat"], name="roi_feat")
    meta_feature_set = normalize_meta_feature_set(raw.get("meta_feature_set", "legacy_iou"))
    default_meta_names = meta_feature_names_for_set(meta_feature_set)
    raw_meta_names = raw.get("meta_feature_names")
    if isinstance(raw_meta_names, (list, tuple)) and len(raw_meta_names) == META_DIM:
        meta_feature_names = [str(x) for x in raw_meta_names]
    else:
        meta_feature_names = list(default_meta_names)

    box_head_feat = _to_tensor_2d(raw["box_head_feat"], name="box_head_feat") if "box_head_feat" in raw else None
    box_pooler_feat = _to_tensor_4d(raw["box_pooler_feat"], name="box_pooler_feat") if "box_pooler_feat" in raw else None
    box_pooler_feat_expanded = (
        _to_tensor_4d(raw["box_pooler_feat_expanded"], name="box_pooler_feat_expanded")
        if "box_pooler_feat_expanded" in raw
        else None
    )
    mask_pooler_feat = (
        _to_tensor_4d(raw["mask_pooler_feat"], name="mask_pooler_feat")
        if "mask_pooler_feat" in raw
        else None
    )
    box_pooler_gap = _to_tensor_2d(raw["box_pooler_gap"], name="box_pooler_gap") if "box_pooler_gap" in raw else None
    box_pooler_gap_expanded = (
        _to_tensor_2d(raw["box_pooler_gap_expanded"], name="box_pooler_gap_expanded")
        if "box_pooler_gap_expanded" in raw
        else None
    )
    mask_pooler_gap = _to_tensor_2d(raw["mask_pooler_gap"], name="mask_pooler_gap") if "mask_pooler_gap" in raw else None

    pack = RoiFeaturePack(
        image_ids=torch.as_tensor(raw["image_id"]).clone().cpu().long(),
        scores=torch.as_tensor(raw["score"]).clone().cpu().float(),
        roi_feat=roi_feat,
        bbox_area_ratio=torch.as_tensor(raw["bbox_area_ratio"]).clone().cpu().float(),
        mask_area_ratio=torch.as_tensor(raw["mask_area_ratio"]).clone().cpu().float(),
        labels=torch.as_tensor(raw["label_idx"]).clone().cpu().long(),
        mask_iou=torch.as_tensor(raw["mask_iou"]).clone().cpu().float(),
        bbox_iou=torch.as_tensor(raw["bbox_iou"]).clone().cpu().float(),
        file_names=[str(x) for x in raw["file_name"]],
        source=[str(x) for x in raw["source"]],
        class_ids=[int(x) for x in raw["index_to_class_id"]],
        class_names=[str(x) for x in raw["index_to_class_name"]],
        feature_source=normalize_roi_feature_source(raw.get("roi_feat_source", "box_head")),
        meta_feature_set=meta_feature_set,
        meta_feature_names=meta_feature_names,
        box_head_feat=box_head_feat,
        box_pooler_feat=box_pooler_feat,
        box_pooler_feat_expanded=box_pooler_feat_expanded,
        mask_pooler_feat=mask_pooler_feat,
        box_pooler_gap=box_pooler_gap,
        box_pooler_gap_expanded=box_pooler_gap_expanded,
        mask_pooler_gap=mask_pooler_gap,
    )

    n = pack.num_samples
    if pack.roi_feat.shape[0] != n:
        raise ValueError(f"roi_feat row mismatch: expected {n}, got {int(pack.roi_feat.shape[0])}")
    if not (
        len(pack.file_names) == n
        and len(pack.source) == n
        and pack.scores.shape[0] == n
        and pack.bbox_area_ratio.shape[0] == n
        and pack.mask_area_ratio.shape[0] == n
        and pack.mask_iou.shape[0] == n
        and pack.bbox_iou.shape[0] == n
    ):
        raise ValueError("inconsistent lengths among fields in ROI feature pack")

    opt_2d = {
        "box_head_feat": pack.box_head_feat,
        "box_pooler_gap": pack.box_pooler_gap,
        "box_pooler_gap_expanded": pack.box_pooler_gap_expanded,
        "mask_pooler_gap": pack.mask_pooler_gap,
    }
    for name, feat in opt_2d.items():
        if feat is not None and int(feat.shape[0]) != n:
            raise ValueError(f"{name} row mismatch: expected {n}, got {int(feat.shape[0])}")

    opt_4d = {
        "box_pooler_feat": pack.box_pooler_feat,
        "box_pooler_feat_expanded": pack.box_pooler_feat_expanded,
        "mask_pooler_feat": pack.mask_pooler_feat,
    }
    for name, feat in opt_4d.items():
        if feat is not None and int(feat.shape[0]) != n:
            raise ValueError(f"{name} row mismatch: expected {n}, got {int(feat.shape[0])}")

    return pack


def _ensure_rank2_feature(feat: torch.Tensor, name: str) -> torch.Tensor:
    if feat.ndim > 2:
        feat = torch.flatten(feat, start_dim=1)
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
    if feat.ndim != 2:
        raise ValueError(f"{name} must be rank-2 after flatten, got {tuple(feat.shape)}")
    return feat


def extract_box_head_features_from_instances(instances) -> torch.Tensor:
    if not instances.has("pred_box_features"):
        raise KeyError(
            "instances has no 'pred_box_features'. "
            "Patch CascadeROIHeads to attach final-stage ROI features."
        )
    feat = instances.pred_box_features
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)
    return _ensure_rank2_feature(feat, "pred_box_features")


def extract_box_pooler_features_from_instances(instances, mode: str = "gap") -> torch.Tensor:
    if not instances.has("pred_box_pooler_features"):
        raise KeyError(
            "instances has no 'pred_box_pooler_features'. "
            "Enable roi_heads.return_box_pooler_features=True."
        )
    feat = instances.pred_box_pooler_features
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "raw":
        if feat.ndim != 4:
            raise ValueError(f"pred_box_pooler_features expected rank-4 for raw, got {tuple(feat.shape)}")
        return feat
    if mode_norm == "gap":
        if feat.ndim != 4:
            raise ValueError(f"pred_box_pooler_features expected rank-4 for gap, got {tuple(feat.shape)}")
        feat = feat.mean(dim=(2, 3))
        return _ensure_rank2_feature(feat, "pred_box_pooler_features(gap)")
    if mode_norm == "flatten":
        return _ensure_rank2_feature(feat, "pred_box_pooler_features(flatten)")
    raise ValueError(f"unsupported pooler mode: {mode}")


def extract_box_pooler_features_expanded_from_instances(instances, mode: str = "raw") -> torch.Tensor:
    if not instances.has("pred_box_pooler_features_expanded"):
        raise KeyError(
            "instances has no 'pred_box_pooler_features_expanded'. "
            "Enable roi_heads.return_box_pooler_features_expanded=True."
        )
    feat = instances.pred_box_pooler_features_expanded
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "raw":
        if feat.ndim != 4:
            raise ValueError(
                f"pred_box_pooler_features_expanded expected rank-4 for raw, got {tuple(feat.shape)}"
            )
        return feat
    if mode_norm == "gap":
        if feat.ndim != 4:
            raise ValueError(
                f"pred_box_pooler_features_expanded expected rank-4 for gap, got {tuple(feat.shape)}"
            )
        feat = feat.mean(dim=(2, 3))
        return _ensure_rank2_feature(feat, "pred_box_pooler_features_expanded(gap)")
    if mode_norm == "flatten":
        return _ensure_rank2_feature(feat, "pred_box_pooler_features_expanded(flatten)")
    raise ValueError(f"unsupported pooler mode: {mode}")


def extract_mask_pooler_features_from_instances(instances, mode: str = "raw") -> torch.Tensor:
    if not instances.has("pred_mask_pooler_features"):
        raise KeyError(
            "instances has no 'pred_mask_pooler_features'. "
            "Enable roi_heads.return_mask_pooler_features=True."
        )
    feat = instances.pred_mask_pooler_features
    if not isinstance(feat, torch.Tensor):
        feat = torch.as_tensor(feat)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "raw":
        if feat.ndim != 4:
            raise ValueError(f"pred_mask_pooler_features expected rank-4 for raw, got {tuple(feat.shape)}")
        return feat
    if mode_norm == "gap":
        if feat.ndim != 4:
            raise ValueError(f"pred_mask_pooler_features expected rank-4 for gap, got {tuple(feat.shape)}")
        feat = feat.mean(dim=(2, 3))
        return _ensure_rank2_feature(feat, "pred_mask_pooler_features(gap)")
    if mode_norm == "flatten":
        return _ensure_rank2_feature(feat, "pred_mask_pooler_features(flatten)")
    raise ValueError(f"unsupported pooler mode: {mode}")


def build_roi_feature_tensor_from_instances(instances, feature_source: str) -> torch.Tensor:
    src = normalize_roi_feature_source(feature_source)
    if src == "box_head":
        return extract_box_head_features_from_instances(instances)
    if src == "pooler_gap":
        return extract_box_pooler_features_from_instances(instances, mode="gap")
    if src == "pooler_flatten":
        return extract_box_pooler_features_from_instances(instances, mode="flatten")
    if src == "box_head_plus_pooler_gap":
        box_head = extract_box_head_features_from_instances(instances)
        pooler_gap = extract_box_pooler_features_from_instances(instances, mode="gap")
        return torch.cat([box_head, pooler_gap], dim=1)
    raise ValueError(f"unsupported feature source: {feature_source}")


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=int(num_classes)).float()
    inv = counts.sum() / torch.clamp(counts, min=1.0)
    weights = inv / inv.mean().clamp_min(1e-6)
    return weights


def confusion_and_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[str, object]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)

    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = tp / np.maximum(pred_count, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)

    macro_f1 = float(f1.mean())
    acc = float(tp.sum() / max(int(cm.sum()), 1))

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _coerce_int_list(value, default: List[int]) -> List[int]:
    if value is None:
        return [int(v) for v in default]
    if isinstance(value, (list, tuple)):
        out = [int(v) for v in value]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [x.strip() for x in text.split(",") if x.strip()]
        out = [int(v) for v in items] if items else [int(v) for v in default]
    if any(v <= 0 for v in out):
        raise ValueError(f"integer list contains non-positive values: {out}")
    return out


def classifier_from_checkpoint(ckpt_path: Path, map_location: str = "cpu") -> Tuple[nn.Module, Dict]:
    ckpt = _torch_load_compat(Path(ckpt_path), map_location=map_location)
    cfg = ckpt.get("model_cfg") or {}
    num_classes = int(cfg.get("num_classes", 0))
    if num_classes <= 0:
        class_names = ckpt.get("class_names") or []
        num_classes = int(len(class_names))
    input_dim = int(cfg.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError(f"invalid input_dim in checkpoint: {input_dim}")

    model_type = str(cfg.get("model_type", "mlp")).strip().lower()
    if model_type in ("mlp", "roi_mlp"):
        model = RoiFeatureClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            hidden_dim=int(cfg.get("hidden_dim", 512)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
    elif model_type in ("spatial_conv", "roi_spatial_conv"):
        conv_channels = _coerce_int_list(cfg.get("conv_channels", [96, 64]), default=[96, 64])
        conv_kernels = _coerce_int_list(cfg.get("conv_kernels", [1, 3]), default=[1, 3])
        model = RoiSpatialConvClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            conv_channels=conv_channels,
            conv_kernels=conv_kernels,
            conv_dropout=float(cfg.get("conv_dropout", 0.0)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    elif model_type in ("spatial_gap", "roi_spatial_gap", "dwconv_gap", "roi_dwconv_gap"):
        model = RoiSpatialGapClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            stem_channels=int(cfg.get("gap_stem_channels", 64)),
            mid_channels=int(cfg.get("gap_mid_channels", 64)),
            dw_kernel=int(cfg.get("gap_dw_kernel", 3)),
            dropout=float(cfg.get("gap_dropout", cfg.get("dropout", 0.0))),
        )
    elif model_type in ("rich_gap_fusion", "roi_rich_gap_fusion"):
        model = RoiRichGapFusionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            use_roi_feat=bool(cfg.get("use_roi_feat", True)),
            use_box_head_feat=bool(cfg.get("use_box_head_feat", True)),
            use_box_pooler_gap_expanded=bool(cfg.get("use_box_pooler_gap_expanded", True)),
            use_mask_pooler_gap=bool(cfg.get("use_mask_pooler_gap", True)),
            roi_proj_dim=int(cfg.get("roi_proj_dim", 512)),
            box_head_proj_dim=int(cfg.get("box_head_proj_dim", 256)),
            expanded_gap_proj_dim=int(cfg.get("expanded_gap_proj_dim", 128)),
            mask_gap_proj_dim=int(cfg.get("mask_gap_proj_dim", 128)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    elif model_type in ("rich_spatial_fusion", "roi_rich_spatial_fusion"):
        model = RoiRichSpatialFusionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            mask_pooler_size=int(cfg.get("mask_pooler_size", 14)),
            local_branch_dim=int(cfg.get("local_branch_dim", 96)),
            expanded_branch_dim=int(cfg.get("expanded_branch_dim", 96)),
            mask_branch_dim=int(cfg.get("mask_branch_dim", 96)),
            box_head_proj_dim=int(cfg.get("box_head_proj_dim", 256)),
            dw_kernel=int(cfg.get("fusion_dw_kernel", 3)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    elif model_type in ("rich_spatial_gated_fusion", "roi_rich_spatial_gated_fusion"):
        model = RoiRichSpatialGatedFusionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            mask_pooler_size=int(cfg.get("mask_pooler_size", 14)),
            local_branch_dim=int(cfg.get("local_branch_dim", 128)),
            expanded_branch_dim=int(cfg.get("expanded_branch_dim", 128)),
            mask_branch_dim=int(cfg.get("mask_branch_dim", 128)),
            box_head_proj_dim=int(cfg.get("box_head_proj_dim", 320)),
            dw_kernel=int(cfg.get("fusion_dw_kernel", 3)),
            gate_hidden_dim=int(cfg.get("gate_hidden_dim", 192)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    elif model_type in ("rich_spatial_attn_fusion", "roi_rich_spatial_attn_fusion"):
        model = RoiRichSpatialAttnFusionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            mask_pooler_size=int(cfg.get("mask_pooler_size", 14)),
            local_branch_dim=int(cfg.get("local_branch_dim", 128)),
            expanded_branch_dim=int(cfg.get("expanded_branch_dim", 128)),
            mask_branch_dim=int(cfg.get("mask_branch_dim", 128)),
            box_head_proj_dim=int(cfg.get("box_head_proj_dim", 320)),
            dw_kernel=int(cfg.get("fusion_dw_kernel", 3)),
            attn_token_dim=int(cfg.get("attn_token_dim", 192)),
            attn_num_heads=int(cfg.get("attn_num_heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.0)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    elif model_type in (
        "rich_spatial_attn_no_expanded_fusion",
        "roi_rich_spatial_attn_no_expanded_fusion",
    ):
        model = RoiRichSpatialAttnNoExpandedFusionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            mask_pooler_size=int(cfg.get("mask_pooler_size", 14)),
            local_branch_dim=int(cfg.get("local_branch_dim", 128)),
            mask_branch_dim=int(cfg.get("mask_branch_dim", 128)),
            box_head_proj_dim=int(cfg.get("box_head_proj_dim", 320)),
            dw_kernel=int(cfg.get("fusion_dw_kernel", 3)),
            attn_token_dim=int(cfg.get("attn_token_dim", 192)),
            attn_num_heads=int(cfg.get("attn_num_heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.0)),
            token_pool_type=str(cfg.get("attn_pool_type", "mean")),
            branch_dropout=float(cfg.get("attn_branch_dropout", 0.0)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    elif model_type in ("rich_spatial_mask_guided_fusion", "roi_rich_spatial_mask_guided_fusion"):
        model = RoiRichSpatialMaskGuidedFusionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            use_meta=bool(cfg.get("use_meta", True)),
            pooler_channels=int(cfg.get("pooler_channels", 256)),
            pooler_size=int(cfg.get("pooler_size", 7)),
            mask_pooler_size=int(cfg.get("mask_pooler_size", 14)),
            local_branch_dim=int(cfg.get("local_branch_dim", 128)),
            expanded_branch_dim=int(cfg.get("expanded_branch_dim", 128)),
            mask_branch_dim=int(cfg.get("mask_branch_dim", 128)),
            box_head_proj_dim=int(cfg.get("box_head_proj_dim", 320)),
            dw_kernel=int(cfg.get("fusion_dw_kernel", 3)),
            mask_guided_alpha=float(cfg.get("mask_guided_alpha", 1.0)),
            head_hidden_dim=int(cfg.get("head_hidden_dim", cfg.get("hidden_dim", 512))),
            head_num_layers=int(cfg.get("head_num_layers", cfg.get("num_layers", 2))),
            head_dropout=float(cfg.get("head_dropout", cfg.get("dropout", 0.2))),
        )
    else:
        raise ValueError(f"unsupported model_type in checkpoint: {model_type}")

    state_dict = ckpt.get("model_state")
    if state_dict is None:
        raise KeyError(f"checkpoint missing model_state: {ckpt_path}")
    model.load_state_dict(state_dict, strict=True)
    return model, ckpt
