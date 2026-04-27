#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-class image/video inference for DINOv3 ViT-L/16 Cascade Mask R-CNN."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import queue
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DINOV3_PACKAGE_ROOT = REPO_ROOT / "dinov3"
EVA02_DET_PATH = Path(os.environ.get("EVA02_DET_PATH", REPO_ROOT / "eva02" / "eva02_det"))
DEFAULT_CONFIG = (
    EVA02_DET_PATH
    / "projects/ViTDet/configs/eva2_o365_to_coco/"
    "eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1280_lrd0p8.py"
)
DEFAULT_RUN_PTH_DIR = REPO_ROOT / "checkpoints" / "detector"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_INFER_STREAMS: dict[str, torch.cuda.Stream] = {}


@dataclass
class _DeferredPrediction:
    image_bgr: np.ndarray
    instances: Instances
    lb: dict[str, float | int]
    orig_h: int
    orig_w: int
    target_size: int | tuple[int, int]


def _prepare_import_paths() -> Path:
    if not EVA02_DET_PATH.is_dir():
        raise FileNotFoundError(f"EVA02_DET_PATH not found: {EVA02_DET_PATH}")
    for path in (str(SCRIPTS_DIR), str(REPO_ROOT), str(DINOV3_PACKAGE_ROOT), str(EVA02_DET_PATH)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return Path.cwd()


_ORIGINAL_CWD = _prepare_import_paths()

os.environ.setdefault("DINOV3_USE_XFORMERS", "1")
os.environ.setdefault("EVA02_XATTN", "1")

# Importing the training module registers ATSSRPN and applies its safe checkpoint
# loading compatibility patch. It also chdirs to eva02_det; restore the caller cwd.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from train_dinov3_cascade_unified import ATSSRPN  # noqa: F401

os.chdir(_ORIGINAL_CWD)

from configs import paths as unified_paths
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import Instances


class _ChannelsLastPyramidWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for stage in getattr(self.backbone, "stages", []):
            stage.to(memory_format=torch.channels_last)

    @property
    def size_divisibility(self) -> int:
        return int(getattr(self.backbone, "size_divisibility", 0))

    @property
    def padding_constraints(self) -> dict[str, int]:
        return getattr(self.backbone, "padding_constraints", {})

    def output_shape(self):
        return self.backbone.output_shape()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bottom_up_features = self.backbone.net(x)
        features = bottom_up_features[self.backbone.in_feature]
        if features.is_cuda and features.ndim == 4:
            features = features.contiguous(memory_format=torch.channels_last)

        results = [stage(features) for stage in self.backbone.stages]
        if self.backbone.top_block is not None:
            if self.backbone.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.backbone.top_block.in_feature]
            else:
                top_block_in_feature = results[self.backbone._out_features.index(self.backbone.top_block.in_feature)]
            results.extend(self.backbone.top_block(top_block_in_feature))
        return {f: res for f, res in zip(self.backbone._out_features, results)}


class _CudaGraphBackboneWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, warmup_iters: int = 3) -> None:
        super().__init__()
        self.backbone = backbone
        self.warmup_iters = max(0, int(warmup_iters))
        self._static_input: torch.Tensor | None = None
        self._static_outputs: dict[str, torch.Tensor] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._signature: tuple[tuple[int, ...], torch.dtype, str] | None = None
        self._capture_failed = False

    @property
    def size_divisibility(self) -> int:
        return int(getattr(self.backbone, "size_divisibility", 0))

    @property
    def padding_constraints(self) -> dict[str, int]:
        return getattr(self.backbone, "padding_constraints", {})

    def output_shape(self):
        return self.backbone.output_shape()

    def _make_signature(self, x: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype, str]:
        return tuple(int(v) for v in x.shape), x.dtype, str(x.device)

    def _capture(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        signature = self._make_signature(x)
        self._static_input = torch.empty_like(x)
        self._static_input.copy_(x)

        for _ in range(self.warmup_iters):
            self._static_outputs = self.backbone(self._static_input)
        torch.cuda.synchronize(device=x.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._static_outputs = self.backbone(self._static_input)
        torch.cuda.synchronize(device=x.device)

        self._graph = graph
        self._signature = signature
        print(f"[INFO] Captured CUDA Graph for backbone: shape={signature[0]}, dtype={signature[1]}")
        return self._static_outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if not x.is_cuda or self._capture_failed:
            return self.backbone(x)

        signature = self._make_signature(x)
        if self._graph is None or self._signature != signature:
            try:
                return self._capture(x)
            except Exception as exc:
                self._capture_failed = True
                print(f"[WARN] CUDA Graph backbone capture failed; falling back to eager backbone: {exc}")
                return self.backbone(x)

        assert self._static_input is not None
        assert self._static_outputs is not None
        self._static_input.copy_(x)
        self._graph.replay()
        return self._static_outputs


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True", "yes", "YES")


def _compile_module(module: torch.nn.Module, label: str) -> torch.nn.Module:
    if not hasattr(torch, "compile"):
        print(f"[WARN] torch.compile unavailable; skipped {label}")
        return module

    try:
        torch_dynamo = importlib.import_module("torch._dynamo")

        if _env_flag("EVA_TORCH_COMPILE_SUPPRESS_ERRORS", "1"):
            torch_dynamo.config.suppress_errors = True
    except Exception:
        pass

    mode = os.environ.get("EVA_TORCH_COMPILE_MODE", "reduce-overhead")
    fullgraph = _env_flag("EVA_TORCH_COMPILE_FULLGRAPH", "0")
    dynamic_value = os.environ.get("EVA_TORCH_COMPILE_DYNAMIC", "").strip().lower()
    dynamic: bool | None
    if dynamic_value in ("1", "true", "yes"):
        dynamic = True
    elif dynamic_value in ("0", "false", "no"):
        dynamic = False
    else:
        dynamic = None

    kwargs: dict[str, object] = {"mode": mode, "fullgraph": fullgraph}
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    try:
        compiled = torch.compile(module, **kwargs)
        print(f"[INFO] torch.compile enabled for {label}: mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
        return compiled
    except Exception as exc:
        print(f"[WARN] torch.compile setup failed for {label}: {exc}")
        return module


def _replace_pyramid_stage(backbone: torch.nn.Module, index: int, compiled: torch.nn.Module) -> None:
    old_stage = backbone.stages[index]
    backbone.stages[index] = compiled
    for module_name, module in list(backbone._modules.items()):
        if module is old_stage:
            setattr(backbone, module_name, compiled)
            return


def _apply_torch_compile_optimizations(model: torch.nn.Module) -> None:
    requested = {
        part.strip().lower()
        for part in os.environ.get("EVA_TORCH_COMPILE_MODULES", "").replace(";", ",").split(",")
        if part.strip()
    }
    if not requested:
        return

    compile_all = "all" in requested

    if compile_all or "pyramid" in requested or "fpn" in requested:
        backbone = getattr(model, "backbone", None)
        pyramid = getattr(backbone, "backbone", backbone)
        stages = getattr(pyramid, "stages", None)
        if stages:
            for idx, stage in enumerate(list(stages)):
                _replace_pyramid_stage(pyramid, idx, _compile_module(stage, f"SimpleFeaturePyramid.stage{idx}"))
        else:
            print("[WARN] SimpleFeaturePyramid stages not found; skipped pyramid compile")

    if compile_all or "rpn_head" in requested or "rpn" in requested:
        proposal_generator = getattr(model, "proposal_generator", None)
        rpn_head = getattr(proposal_generator, "rpn_head", None)
        if isinstance(rpn_head, torch.nn.Module):
            proposal_generator.rpn_head = _compile_module(rpn_head, "proposal_generator.rpn_head")
        else:
            print("[WARN] proposal_generator.rpn_head not found; skipped RPN head compile")

    if compile_all or "roi_box" in requested or "box_head" in requested:
        roi_heads = getattr(model, "roi_heads", None)
        box_heads = getattr(roi_heads, "box_head", None)
        if isinstance(box_heads, torch.nn.ModuleList):
            for idx, head in enumerate(list(box_heads)):
                box_heads[idx] = _compile_module(head, f"roi_heads.box_head[{idx}]")
        elif isinstance(box_heads, torch.nn.Module):
            roi_heads.box_head = _compile_module(box_heads, "roi_heads.box_head")
        else:
            print("[WARN] roi_heads.box_head not found; skipped ROI box head compile")

    if compile_all or "roi_predictor" in requested or "box_predictor" in requested:
        roi_heads = getattr(model, "roi_heads", None)
        box_predictors = getattr(roi_heads, "box_predictor", None)
        if isinstance(box_predictors, torch.nn.ModuleList):
            for idx, predictor in enumerate(list(box_predictors)):
                box_predictors[idx] = _compile_module(predictor, f"roi_heads.box_predictor[{idx}]")
        elif isinstance(box_predictors, torch.nn.Module):
            roi_heads.box_predictor = _compile_module(box_predictors, "roi_heads.box_predictor")
        else:
            print("[WARN] roi_heads.box_predictor not found; skipped ROI box predictor compile")


def _unpack_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    return int(size[0]), int(size[1])


def _parse_target_size(value: str) -> int | tuple[int, int]:
    text = str(value).lower().replace(",", "x")
    if "x" not in text:
        return int(text)
    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target size must be INT or HxW, e.g. 1280x720")
    return int(parts[0]), int(parts[1])


def _format_size(size: int | tuple[int, int]) -> str:
    h, w = _unpack_size(size)
    return f"{h}x{w}"


def _resolve_checkpoint(path_value: str | os.PathLike[str] | None) -> Path:
    if path_value:
        candidate = Path(path_value).expanduser()
    else:
        candidate = DEFAULT_RUN_PTH_DIR / "last_checkpoint"

    if candidate.is_dir():
        candidate = candidate / "last_checkpoint"

    if candidate.name == "last_checkpoint":
        if not candidate.is_file():
            raise FileNotFoundError(f"last_checkpoint not found: {candidate}")
        checkpoint_name = candidate.read_text(encoding="utf-8").strip()
        if not checkpoint_name:
            raise RuntimeError(f"Empty last_checkpoint file: {candidate}")
        resolved = Path(checkpoint_name)
        if not resolved.is_absolute():
            resolved = candidate.parent / resolved
        candidate = resolved

    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def _collect_paths(input_path: Path, exts: set[str], recursive: bool, kind: str) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            return [input_path]
        raise RuntimeError(f"Input file is not a {kind}: {input_path}")
    if not input_path.is_dir():
        raise RuntimeError(f"Input path not found: {input_path}")
    iterator: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in exts)


def _patch_test_thresholds(
    cfg,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    rpn_pre_nms_topk_test: int,
    rpn_post_nms_topk_test: int,
    rpn_nms_thresh: float,
) -> None:
    if hasattr(cfg.model.roi_heads, "box_predictors"):
        for predictor in cfg.model.roi_heads.box_predictors:
            predictor.test_score_thresh = score_thresh
            predictor.test_nms_thresh = nms_thresh
            predictor.test_topk_per_image = topk_per_image
    if hasattr(cfg.model, "proposal_generator"):
        cfg.model.proposal_generator.pre_nms_topk = (20_000, int(rpn_pre_nms_topk_test))
        cfg.model.proposal_generator.post_nms_topk = (2_000, int(rpn_post_nms_topk_test))
        cfg.model.proposal_generator.nms_thresh = float(rpn_nms_thresh)


def _build_model(
    checkpoint: Path,
    target_size: int | tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    device: str,
    config_path: Path,
    backbone_weights: str,
    rpn_pre_nms_topk_test: int = 10_000,
    rpn_post_nms_topk_test: int = 1_000,
    rpn_nms_thresh: float = 0.9,
    onnx_backbone_path: Path | None = None,
    onnx_max_batch: int = 64,
    onnx_dtype: str = "fp16",
    onnx_iobind: bool = True,
    onnx_disable_trt: bool = False,
    trt_backbone_engine: Path | None = None,
    trt_full_backbone_engine: Path | None = None,
    safe_full_half: bool = False,
) -> torch.nn.Module:
    from detectron2.modeling import DINOv3Backbone

    cfg = LazyConfig.load(str(config_path))
    target_h, target_w = _unpack_size(target_size)

    cfg.model.roi_heads.num_classes = 1
    cfg.model.backbone.square_pad = target_h if target_h == target_w else 0
    cfg.model.backbone.net = L(DINOv3Backbone)(
        img_size=max(target_h, target_w),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        layers_to_use=1,
        out_feature="last_feat",
        weights=backbone_weights,
        pretrained=True,
    )
    cfg.model.backbone.in_feature = "last_feat"

    original_pg = cfg.model.proposal_generator
    cfg.model.proposal_generator = L(ATSSRPN)(
        topk=20,
        center_radius=2.0,
        in_features=original_pg.in_features,
        head=original_pg.head,
        anchor_generator=original_pg.anchor_generator,
        anchor_matcher=None,
        box2box_transform=original_pg.box2box_transform,
        batch_size_per_image=256,
        positive_fraction=0.7,
        pre_nms_topk=(20_000, int(rpn_pre_nms_topk_test)),
        post_nms_topk=(2_000, int(rpn_post_nms_topk_test)),
        nms_thresh=float(rpn_nms_thresh),
        min_box_size=0,
        anchor_boundary_thresh=getattr(original_pg, "anchor_boundary_thresh", -1),
        loss_weight=getattr(original_pg, "loss_weight", 1.0),
        box_reg_loss_type=getattr(original_pg, "box_reg_loss_type", "smooth_l1"),
        smooth_l1_beta=getattr(original_pg, "smooth_l1_beta", 0.0),
    )

    cfg.model.roi_heads.batch_size_per_image = 512
    cfg.model.roi_heads.positive_fraction = 0.7
    cfg.model.roi_heads.proposal_append_gt = True
    if hasattr(cfg.model.roi_heads, "proposal_matchers"):
        for idx, threshold in enumerate([0.45, 0.55, 0.65]):
            if idx < len(cfg.model.roi_heads.proposal_matchers):
                cfg.model.roi_heads.proposal_matchers[idx].thresholds = [threshold]

    _patch_test_thresholds(
        cfg,
        score_thresh,
        nms_thresh,
        topk_per_image,
        rpn_pre_nms_topk_test,
        rpn_post_nms_topk_test,
        rpn_nms_thresh,
    )

    model = instantiate(cfg.model).to(device).eval()
    DetectionCheckpointer(model).load(str(checkpoint))

    cascade_num_stages = os.environ.get("EVA_CASCADE_NUM_STAGES")
    if cascade_num_stages:
        if not hasattr(model, "roi_heads") or not hasattr(model.roi_heads, "num_cascade_stages"):
            raise RuntimeError("model.roi_heads.num_cascade_stages not found")
        requested_stages = int(cascade_num_stages)
        original_stages = int(model.roi_heads.num_cascade_stages)
        if requested_stages < 1 or requested_stages > original_stages:
            raise ValueError(f"EVA_CASCADE_NUM_STAGES must be in [1, {original_stages}], got {requested_stages}")
        model.roi_heads.num_cascade_stages = requested_stages
        print(f"[INFO] Using {requested_stages}/{original_stages} Cascade box stages")

    fixed_num_proposals = os.environ.get("EVA_ROI_FIXED_NUM_PROPOSALS")
    if fixed_num_proposals:
        if not hasattr(model, "roi_heads") or not hasattr(model.roi_heads, "fixed_num_proposals"):
            raise RuntimeError("model.roi_heads.fixed_num_proposals not found")
        fixed_num_proposals_int = int(fixed_num_proposals)
        if fixed_num_proposals_int < 1:
            raise ValueError(f"EVA_ROI_FIXED_NUM_PROPOSALS must be >= 1, got {fixed_num_proposals_int}")
        model.roi_heads.fixed_num_proposals = fixed_num_proposals_int
        print(f"[INFO] Using fixed ROI proposals per image: {fixed_num_proposals_int}")

    if safe_full_half:
        if onnx_backbone_path is not None:
            raise ValueError("--safe-full-half cannot be combined with --onnx-backbone")
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            raise ValueError("--safe-full-half requires a CUDA device")
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
            raise RuntimeError("model.backbone.net not found; cannot apply safe full-half")
        from safe_fp16_backbone import apply_safe_fp16_islands

        model.backbone.net = apply_safe_fp16_islands(model.backbone.net)
        torch.cuda.empty_cache()

    if trt_full_backbone_engine is not None:
        if onnx_backbone_path is not None or trt_backbone_engine is not None:
            raise ValueError("--trt-full-backbone-engine cannot be combined with ONNX/TRT net backbone")
        resolved_engine = Path(trt_full_backbone_engine).expanduser().resolve()
        if not resolved_engine.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {resolved_engine}")
        from trt_backbone import TensorRTFeatureDictBackboneAdapter

        old_backbone = model.backbone
        model.backbone = TensorRTFeatureDictBackboneAdapter(
            str(resolved_engine),
            size_divisibility=getattr(old_backbone, "size_divisibility", 0),
            padding_constraints=getattr(old_backbone, "padding_constraints", None),
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] Using native TensorRT full backbone: {resolved_engine}")

    if trt_backbone_engine is not None:
        if onnx_backbone_path is not None:
            raise ValueError("--trt-backbone-engine cannot be combined with --onnx-backbone")
        resolved_engine = Path(trt_backbone_engine).expanduser().resolve()
        if not resolved_engine.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {resolved_engine}")
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
            raise RuntimeError("model.backbone.net not found; cannot replace with TensorRT backbone")
        from trt_backbone import TensorRTBackboneAdapter

        model.backbone.net = TensorRTBackboneAdapter(str(resolved_engine))
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] Using native TensorRT backbone: {resolved_engine}")

    if onnx_backbone_path is not None:
        resolved_onnx = Path(onnx_backbone_path).expanduser().resolve()
        if not resolved_onnx.is_file():
            raise FileNotFoundError(f"ONNX backbone not found: {resolved_onnx}")
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
            raise RuntimeError("model.backbone.net not found; cannot replace with ONNX backbone")

        os.environ.setdefault("EVA_REQUIRE_CUDA_EP", "1")
        os.environ.setdefault("EVA_DISABLE_CPU_EP", "1")
        os.environ["EVA_ONNXRT_IOBIND"] = "1" if onnx_iobind else "0"
        os.environ["EVA_DISABLE_TRT_EP"] = "1" if onnx_disable_trt else "0"

        from onnx_backbone import ORTBackboneAdapter

        expect_dtype = torch.float16 if onnx_dtype == "fp16" else torch.float32
        model.backbone.net = ORTBackboneAdapter(
            str(resolved_onnx),
            max_batch=max(1, int(onnx_max_batch)),
            expect_dtype=expect_dtype,
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] Using ONNX Runtime backbone: {resolved_onnx}")
    if os.environ.get("EVA_PYRAMID_CHANNELS_LAST", "0") in ("1", "true", "True"):
        if hasattr(model, "backbone") and hasattr(model.backbone, "stages") and hasattr(model.backbone, "net"):
            model.backbone = _ChannelsLastPyramidWrapper(model.backbone)
            print("[INFO] Using channels_last SimpleFeaturePyramid")
    if _env_flag("EVA_BACKBONE_CUDAGRAPH", "0"):
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            raise ValueError("EVA_BACKBONE_CUDAGRAPH requires CUDA")
        warmup_iters = int(os.environ.get("EVA_BACKBONE_CUDAGRAPH_WARMUP", "3"))
        model.backbone = _CudaGraphBackboneWrapper(model.backbone, warmup_iters=warmup_iters)
        print(f"[INFO] Using CUDA Graph backbone wrapper: warmup_iters={warmup_iters}")
    _apply_torch_compile_optimizations(model)
    return model


class LetterboxTransform:
    def __init__(self, src_shape: tuple[int, int], target_size: int | tuple[int, int], pad_value: int = 128) -> None:
        h, w = src_shape
        target_h, target_w = _unpack_size(target_size)
        self.scale = min(target_h / h, target_w / w)
        self.new_h = int(h * self.scale)
        self.new_w = int(w * self.scale)
        pad_h = target_h - self.new_h
        pad_w = target_w - self.new_w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left
        self.target_h = target_h
        self.target_w = target_w
        self.pad_value = pad_value

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        border_value = (self.pad_value,) * img.shape[2] if img.ndim == 3 else self.pad_value
        return cv2.copyMakeBorder(
            resized,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )

    def params(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "new_h": self.new_h,
            "new_w": self.new_w,
            "pad_top": self.pad_top,
            "pad_left": self.pad_left,
        }


def _unletterbox_instances(
    instances: Instances,
    lb: dict[str, float | int],
    orig_h: int,
    orig_w: int,
    target_size: int | tuple[int, int],
) -> Instances:
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.detach().clone()
        boxes[:, [0, 2]] -= float(lb["pad_left"])
        boxes[:, [1, 3]] -= float(lb["pad_top"])
        boxes /= float(lb["scale"])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h - 1)
        instances.pred_boxes.tensor = boxes

    if instances.has("pred_masks") and len(instances.pred_masks) > 0:
        masks = instances.pred_masks
        target_h, target_w = _unpack_size(target_size)
        if masks.ndim == 3 and masks.shape[1] == target_h and masks.shape[2] == target_w:
            y0 = int(lb["pad_top"])
            x0 = int(lb["pad_left"])
            cropped = masks[:, y0 : y0 + int(lb["new_h"]), x0 : x0 + int(lb["new_w"])]
            if cropped.shape[1] > 0 and cropped.shape[2] > 0:
                resized = [
                    cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    for mask in cropped.cpu().numpy().astype(np.uint8)
                ]
                instances.pred_masks = torch.from_numpy(np.stack(resized, axis=0) > 0)
            else:
                instances.pred_masks = torch.zeros((len(masks), orig_h, orig_w), dtype=torch.bool)
    return instances


def _prep_image(img_bgr: np.ndarray | None) -> np.ndarray:
    if img_bgr is None:
        raise RuntimeError("Failed to read image")
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[2] == 4:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    return img_bgr


def _torch_amp_dtype(value: str) -> torch.dtype:
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {value}")


def _autocast_context(device: str, amp: bool, amp_dtype: str = "fp16"):
    if not amp or not device.startswith("cuda"):
        return contextlib.nullcontext()
    dtype = _torch_amp_dtype(amp_dtype)
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def _non_default_infer_stream(device: str) -> torch.cuda.Stream | None:
    if os.environ.get("EVA_INFER_NON_DEFAULT_STREAM", "0") not in ("1", "true", "True"):
        return None
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None
    device_obj = torch.device(device)
    key = str(device_obj)
    stream = _INFER_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device_obj)
        _INFER_STREAMS[key] = stream
    return stream


def _prepare_model_input_cpu(
    image_bgr: np.ndarray,
    target_size: int | tuple[int, int],
    *,
    pin_memory: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    image_bgr = _prep_image(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    target_h, target_w = _unpack_size(target_size)

    tfm = LetterboxTransform(src_shape=(orig_h, orig_w), target_size=target_size, pad_value=128)
    letterboxed = tfm.apply_image(image_rgb)
    x = torch.from_numpy(np.ascontiguousarray(letterboxed.transpose(2, 0, 1))).float()
    if pin_memory:
        x = x.pin_memory()
    return image_bgr, {
        "tensor_cpu": x,
        "height": target_h,
        "width": target_w,
        "lb": tfm.params(),
        "orig_h": orig_h,
        "orig_w": orig_w,
    }


def _materialize_inputs_on_device(
    prepared: list[tuple[np.ndarray, dict[str, object]]],
    device: str,
) -> list[dict[str, object]]:
    inputs: list[dict[str, object]] = []
    for _, meta in prepared:
        x = meta["tensor_cpu"].to(device, non_blocking=True)  # type: ignore[union-attr]
        inputs.append({"image": x, "height": int(meta["height"]), "width": int(meta["width"])})
    return inputs


@dataclass
class _PreparedBatch:
    items: list[tuple[np.ndarray, dict[str, object]]]


@dataclass
class _DeviceBatch:
    items: list[tuple[np.ndarray, dict[str, object]]]
    inputs: list[dict[str, object]]


class _AsyncBatchProducer:
    def __init__(
        self,
        *,
        video_path: Path,
        target_size: int | tuple[int, int],
        batch_size: int,
        max_frames: int | None,
        pin_memory: bool,
        prefetch_batches: int,
    ) -> None:
        self.video_path = video_path
        self.target_size = target_size
        self.batch_size = max(1, batch_size)
        self.max_frames = max_frames
        self.pin_memory = pin_memory
        self.queue: queue.Queue[_PreparedBatch | BaseException | None] = queue.Queue(maxsize=max(1, prefetch_batches))
        self.thread = threading.Thread(target=self._worker, name=f"prefetch:{video_path.name}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _worker(self) -> None:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.queue.put(RuntimeError(f"Failed to open video: {self.video_path}"))
            return

        frame_count = 0
        try:
            while True:
                batch: list[tuple[np.ndarray, dict[str, object]]] = []
                while len(batch) < self.batch_size:
                    if self.max_frames is not None and frame_count >= self.max_frames:
                        break
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    batch.append(
                        _prepare_model_input_cpu(
                            frame_bgr,
                            self.target_size,
                            pin_memory=self.pin_memory,
                        )
                    )
                    frame_count += 1

                if not batch:
                    break
                self.queue.put(_PreparedBatch(batch))
                if self.max_frames is not None and frame_count >= self.max_frames:
                    break
        except BaseException as exc:
            self.queue.put(exc)
        finally:
            cap.release()
            self.queue.put(None)

    def get(self) -> _PreparedBatch | None:
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.thread.join(timeout=1.0)


class _CudaBatchPrefetcher:
    def __init__(self, producer: _AsyncBatchProducer, device: str, enabled: bool) -> None:
        self.producer = producer
        self.device = device
        self.enabled = enabled and device.startswith("cuda") and torch.cuda.is_available()
        self.transfer_stream = torch.cuda.Stream(device=device) if self.enabled else None
        self.next_batch: _DeviceBatch | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._preload_blocking()

    def _schedule_to_device(self, prepared_batch: _PreparedBatch) -> _DeviceBatch:
        if self.transfer_stream is None:
            inputs = _materialize_inputs_on_device(prepared_batch.items, self.device)
            return _DeviceBatch(items=prepared_batch.items, inputs=inputs)

        with torch.cuda.stream(self.transfer_stream):
            inputs = _materialize_inputs_on_device(prepared_batch.items, self.device)
        return _DeviceBatch(items=prepared_batch.items, inputs=inputs)

    def _preload_blocking(self) -> None:
        prepared_batch = self.producer.get()
        if prepared_batch is None:
            self.next_batch = None
            return
        self.next_batch = self._schedule_to_device(prepared_batch)

    def _launch_preload(self) -> None:
        self._prefetch_thread = threading.Thread(
            target=self._preload_blocking,
            name="cuda_prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def next(self) -> _DeviceBatch | None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

        batch = self.next_batch
        if batch is None:
            return None
        if self.transfer_stream is not None:
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_stream(self.transfer_stream)
            for input_dict in batch.inputs:
                image = input_dict["image"]
                if isinstance(image, torch.Tensor):
                    image.record_stream(current_stream)
        self.next_batch = None
        self._launch_preload()
        return batch

    def close(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=1.0)
        self.producer.close()


class _AsyncVideoRenderer:
    def __init__(
        self,
        *,
        writer: cv2.VideoWriter,
        class_name: str,
        score_thresh: float,
        queue_batches: int,
    ) -> None:
        self.writer = writer
        self.class_name = class_name
        self.score_thresh = score_thresh
        self.queue: queue.Queue[list[tuple[np.ndarray, Instances] | _DeferredPrediction] | BaseException | None] = queue.Queue(
            maxsize=max(1, int(queue_batches))
        )
        self.thread = threading.Thread(target=self._worker, name="async_render_write", daemon=True)
        self.frames_written = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()

    def submit(self, predictions: list[tuple[np.ndarray, Instances] | _DeferredPrediction]) -> None:
        if self.error is not None:
            raise self.error
        self.queue.put(predictions)

    def _worker(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                for prediction in item:
                    self.writer.write(_draw_prediction(prediction, self.class_name, self.score_thresh))
                    self.frames_written += 1
        except BaseException as exc:
            self.error = exc
        finally:
            self.writer.release()

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise self.error


def _run_model_on_inputs(
    model: torch.nn.Module,
    prepared: list[tuple[np.ndarray, dict[str, object]]],
    inputs: list[dict[str, object]],
    target_size: int | tuple[int, int],
    device: str,
    amp: bool,
    amp_dtype: str = "fp16",
) -> list[tuple[np.ndarray, Instances]]:
    infer_stream = _non_default_infer_stream(device)
    if infer_stream is not None:
        current_stream = torch.cuda.current_stream(device=torch.device(device))
        infer_stream.wait_stream(current_stream)
        for item in inputs:
            image = item.get("image")
            if isinstance(image, torch.Tensor) and image.is_cuda:
                image.record_stream(infer_stream)
        with torch.inference_mode(), torch.cuda.stream(infer_stream), _autocast_context(device, amp, amp_dtype):
            outputs = model(inputs)
        current_stream.wait_stream(infer_stream)
    else:
        with torch.inference_mode(), _autocast_context(device, amp, amp_dtype):
            outputs = model(inputs)

    results: list[tuple[np.ndarray, Instances]] = []
    defer_postprocess = os.environ.get("EVA_DEFER_POSTPROCESS_TO_RENDER", "0") in ("1", "true", "True")
    for (image_bgr, meta), output in zip(prepared, outputs):
        instances = output["instances"]
        if defer_postprocess:
            results.append(
                _DeferredPrediction(
                    image_bgr=image_bgr,
                    instances=instances,
                    lb=meta["lb"],  # type: ignore[arg-type]
                    orig_h=int(meta["orig_h"]),
                    orig_w=int(meta["orig_w"]),
                    target_size=target_size,
                )
            )
            continue
        instances = _unletterbox_instances(
            instances,
            meta["lb"],  # type: ignore[arg-type]
            int(meta["orig_h"]),
            int(meta["orig_w"]),
            target_size,
        )
        results.append((image_bgr, instances.to("cpu")))
    return results


def _finalize_prediction(
    prediction: tuple[np.ndarray, Instances] | _DeferredPrediction,
) -> tuple[np.ndarray, Instances]:
    if isinstance(prediction, _DeferredPrediction):
        instances = _unletterbox_instances(
            prediction.instances,
            prediction.lb,
            prediction.orig_h,
            prediction.orig_w,
            prediction.target_size,
        )
        return prediction.image_bgr, instances.to("cpu")
    return prediction


def _draw_prediction(
    prediction: tuple[np.ndarray, Instances] | _DeferredPrediction,
    class_name: str,
    score_thresh: float,
) -> np.ndarray:
    image_bgr, instances = _finalize_prediction(prediction)
    return _draw(image_bgr, instances, class_name, score_thresh)


def _predict_prepared_batch(
    model: torch.nn.Module,
    prepared: list[tuple[np.ndarray, dict[str, object]]],
    target_size: int | tuple[int, int],
    device: str,
    amp: bool,
    amp_dtype: str = "fp16",
) -> list[tuple[np.ndarray, Instances]]:
    inputs = _materialize_inputs_on_device(prepared, device)
    return _run_model_on_inputs(model, prepared, inputs, target_size, device, amp, amp_dtype)


def _predict_batch(
    model: torch.nn.Module,
    images_bgr: list[np.ndarray],
    target_size: int | tuple[int, int],
    device: str,
    amp: bool,
    amp_dtype: str = "fp16",
) -> list[tuple[np.ndarray, Instances]]:
    prepared = [
        _prepare_model_input_cpu(
            image_bgr,
            target_size,
            pin_memory=device.startswith("cuda") and torch.cuda.is_available(),
        )
        for image_bgr in images_bgr
    ]
    return _predict_prepared_batch(model, prepared, target_size, device, amp, amp_dtype)


def _predict_one(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    target_size: int | tuple[int, int],
    device: str,
    amp: bool,
    amp_dtype: str = "fp16",
) -> tuple[np.ndarray, Instances]:
    return _finalize_prediction(_predict_batch(model, [image_bgr], target_size, device, amp, amp_dtype)[0])


def _draw(image_bgr: np.ndarray, instances: Instances, class_name: str, score_thresh: float) -> np.ndarray:
    if instances.has("scores") and score_thresh > 0:
        instances = instances[instances.scores >= score_thresh]

    img = image_bgr.copy()
    color = (0, 255, 0)
    color_array = np.asarray(color, dtype=np.float32)
    alpha = 0.4
    boxes = instances.pred_boxes.tensor.float().cpu().numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.float().cpu().numpy() if instances.has("scores") else None
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None

    for idx in range(len(instances)):
        box_xyxy: tuple[int, int, int, int] | None = None
        if len(boxes) > idx:
            x1, y1, x2, y2 = boxes[idx].astype(int)
            box_xyxy = (x1, y1, x2, y2)

        if masks is not None:
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask = mask.astype(bool, copy=False)
            if box_xyxy is not None:
                x1, y1, x2, y2 = box_xyxy
                h, w = mask.shape[:2]
                rx1 = max(0, min(w, x1))
                ry1 = max(0, min(h, y1))
                rx2 = max(0, min(w, x2 + 1))
                ry2 = max(0, min(h, y2 + 1))
                if rx2 > rx1 and ry2 > ry1:
                    mask_roi = mask[ry1:ry2, rx1:rx2]
                    if mask_roi.any():
                        img_roi = img[ry1:ry2, rx1:rx2]
                        masked_pixels = img_roi[mask_roi].astype(np.float32, copy=False)
                        blended = masked_pixels * (1 - alpha) + color_array * alpha
                        img_roi[mask_roi] = np.rint(blended).clip(0, 255).astype(np.uint8)
            elif mask.any():
                masked_pixels = img[mask].astype(np.float32, copy=False)
                blended = masked_pixels * (1 - alpha) + color_array * alpha
                img[mask] = np.rint(blended).clip(0, 255).astype(np.uint8)

        if box_xyxy is not None:
            x1, y1, x2, y2 = box_xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {scores[idx]:.2f}" if scores is not None else class_name
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return img


def _infer_video(
    model: torch.nn.Module,
    video_path: Path,
    output_path: Path,
    class_name: str,
    target_size: int | tuple[int, int],
    device: str,
    amp: bool,
    amp_dtype: str,
    score_thresh: float,
    fourcc: str,
    max_frames: int | None,
    batch_size: int,
    io_prefetch: bool,
    prefetch_batches: int,
    gpu_prefetch: bool,
    async_render: bool = False,
    render_queue_batches: int = 2,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")

    renderer: _AsyncVideoRenderer | None = None
    if async_render:
        renderer = _AsyncVideoRenderer(
            writer=writer,
            class_name=class_name,
            score_thresh=score_thresh,
            queue_batches=render_queue_batches,
        )
        renderer.start()

    frame_idx = 0
    if io_prefetch and device.startswith("cuda") and torch.cuda.is_available():
        cap.release()
        producer = _AsyncBatchProducer(
            video_path=video_path,
            target_size=target_size,
            batch_size=batch_size,
            max_frames=max_frames,
            pin_memory=True,
            prefetch_batches=prefetch_batches,
        )
        producer.start()
        prefetcher = _CudaBatchPrefetcher(producer, device, enabled=gpu_prefetch)
        try:
            while True:
                device_batch = prefetcher.next()
                if device_batch is None:
                    break
                predictions = _run_model_on_inputs(
                    model,
                    device_batch.items,
                    device_batch.inputs,
                    target_size,
                    device,
                    amp,
                    amp_dtype,
                )
                if renderer is not None:
                    renderer.submit(predictions)
                    frame_idx += len(predictions)
                    if frame_idx % 50 == 0:
                        print(f"[INFO] {video_path.name}: submitted {frame_idx} frames")
                else:
                    for prediction in predictions:
                        writer.write(_draw_prediction(prediction, class_name, score_thresh))
                        frame_idx += 1
                        if frame_idx % 50 == 0:
                            print(f"[INFO] {video_path.name}: processed {frame_idx} frames")
        finally:
            prefetcher.close()
            if renderer is not None:
                renderer.close()
            else:
                writer.release()
    else:
        try:
            while True:
                batch_frames: list[np.ndarray] = []
                while len(batch_frames) < batch_size:
                    if max_frames is not None and frame_idx + len(batch_frames) >= max_frames:
                        break
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    batch_frames.append(frame_bgr)

                if not batch_frames:
                    break

                predictions = _predict_batch(model, batch_frames, target_size, device, amp, amp_dtype)
                if renderer is not None:
                    renderer.submit(predictions)
                    frame_idx += len(predictions)
                    if frame_idx % 50 == 0:
                        print(f"[INFO] {video_path.name}: submitted {frame_idx} frames")
                else:
                    for prediction in predictions:
                        writer.write(_draw_prediction(prediction, class_name, score_thresh))
                        frame_idx += 1
                        if frame_idx % 50 == 0:
                            print(f"[INFO] {video_path.name}: processed {frame_idx} frames")
        finally:
            cap.release()
            if renderer is not None:
                renderer.close()
            else:
                writer.release()

    print(f"[DONE] wrote {frame_idx} frames -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-class DINOv3 Cascade Mask R-CNN inference")
    parser.add_argument("--input", required=True, help="Input image/video file or directory")
    parser.add_argument("--output", required=True, help="Output directory for overlays/videos")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path, checkpoint directory, or last_checkpoint file. "
            f"Default: {DEFAULT_RUN_PTH_DIR / 'last_checkpoint'}"
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Detectron2 LazyConfig path")
    parser.add_argument("--backbone-weights", default=unified_paths.DINOv3_WEIGHTS, help="DINOv3 pretrained weights")
    parser.add_argument("--class-name", default="foreground", help="Class name to display")
    parser.add_argument(
        "--target-size",
        type=_parse_target_size,
        default=(1280, 720),
        help="Letterbox size as INT or HxW. Default matches training: 1280x720",
    )
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold for drawing")
    parser.add_argument("--nms-thresh", type=float, default=0.4, help="Cascade test NMS threshold")
    parser.add_argument("--topk", type=int, default=200, help="Top-K detections per image")
    parser.add_argument("--rpn-pre-nms-topk-test", type=int, default=10_000)
    parser.add_argument("--rpn-post-nms-topk-test", type=int, default=1_000)
    parser.add_argument("--rpn-nms-thresh", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use autocast on CUDA")
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Autocast dtype used with --amp",
    )
    parser.add_argument(
        "--safe-full-half",
        action="store_true",
        help="Convert the DINOv3 backbone to fp16 while keeping LayerNorm/LayerScale/residual islands in fp32",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--mode", choices=["auto", "image", "video"], default="auto")
    parser.add_argument("--fourcc", default="mp4v", help="FourCC for video output")
    parser.add_argument("--output-ext", default=".mp4", help="Output video extension")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames for debugging")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of frames to infer together")
    parser.add_argument(
        "--io-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use background decode/preprocess prefetch on CUDA",
    )
    parser.add_argument("--prefetch-batches", type=int, default=1, help="How many prepared batches to queue ahead")
    parser.add_argument(
        "--gpu-prefetch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a dedicated CUDA stream to pre-copy the next batch to GPU",
    )
    parser.add_argument(
        "--async-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render overlays and write video in a background thread",
    )
    parser.add_argument("--render-queue-batches", type=int, default=2, help="Async render queue size in batches")
    parser.add_argument("--onnx-backbone", default=None, help="Use an ONNX Runtime backbone (.onnx)")
    parser.add_argument("--trt-backbone-engine", default=None, help="Use a native TensorRT backbone engine (.engine)")
    parser.add_argument("--trt-full-backbone-engine", default=None, help="Use a native TensorRT full backbone engine (.engine)")
    parser.add_argument("--onnx-dtype", choices=["fp16", "fp32"], default="fp16", help="ONNX input dtype")
    parser.add_argument("--onnx-max-batch", type=int, default=64, help="Max batch used for TensorRT EP profile")
    parser.add_argument(
        "--onnx-iobind",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ONNX Runtime IO binding for CUDA tensors",
    )
    parser.add_argument("--onnx-disable-trt", action="store_true", help="Disable TensorRT EP and use CUDA EP")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    if args.safe_full_half and not args.amp:
        raise ValueError("--safe-full-half requires --amp")
    if args.safe_full_half and args.amp_dtype != "fp16":
        raise ValueError("--safe-full-half is an fp16-only mode; use --amp-dtype fp16")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print(f"[INFO] checkpoint: {checkpoint}")
    print(f"[INFO] target_size: {_format_size(args.target_size)}")
    model = _build_model(
        checkpoint=checkpoint,
        target_size=args.target_size,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk_per_image=args.topk,
        rpn_pre_nms_topk_test=args.rpn_pre_nms_topk_test,
        rpn_post_nms_topk_test=args.rpn_post_nms_topk_test,
        rpn_nms_thresh=args.rpn_nms_thresh,
        device=args.device,
        config_path=Path(args.config).expanduser().resolve(),
        backbone_weights=args.backbone_weights,
        onnx_backbone_path=Path(args.onnx_backbone).expanduser().resolve() if args.onnx_backbone else None,
        onnx_max_batch=args.onnx_max_batch,
        onnx_dtype=args.onnx_dtype,
        onnx_iobind=args.onnx_iobind,
        onnx_disable_trt=args.onnx_disable_trt,
        trt_backbone_engine=Path(args.trt_backbone_engine).expanduser().resolve() if args.trt_backbone_engine else None,
        trt_full_backbone_engine=Path(args.trt_full_backbone_engine).expanduser().resolve() if args.trt_full_backbone_engine else None,
        safe_full_half=args.safe_full_half,
    )

    mode = args.mode
    if mode == "auto":
        if input_path.is_file():
            mode = "video" if input_path.suffix.lower() in VIDEO_EXTS else "image"
        else:
            mode = "video" if _collect_paths(input_path, VIDEO_EXTS, args.recursive, "video") else "image"

    if mode == "video":
        video_paths = _collect_paths(input_path, VIDEO_EXTS, args.recursive, "video")
        if not video_paths:
            raise RuntimeError(f"No videos found under: {input_path}")
        processed = 0
        for video_path in video_paths:
            out_path = output_dir / f"{video_path.stem}{args.output_ext}"
            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] exists: {out_path}")
                continue
            _infer_video(
                model=model,
                video_path=video_path,
                output_path=out_path,
                class_name=args.class_name,
                target_size=args.target_size,
                device=args.device,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                score_thresh=args.score_thresh,
                fourcc=args.fourcc,
                max_frames=args.max_frames,
                batch_size=max(1, args.batch_size),
                io_prefetch=args.io_prefetch,
                prefetch_batches=max(1, args.prefetch_batches),
                gpu_prefetch=args.gpu_prefetch,
                async_render=args.async_render,
                render_queue_batches=max(1, args.render_queue_batches),
            )
            processed += 1
        print(f"[DONE] processed {processed} videos -> {output_dir}")
        return 0

    image_paths = _collect_paths(input_path, IMAGE_EXTS, args.recursive, "image")
    if not image_paths:
        raise RuntimeError(f"No images found under: {input_path}")

    processed = 0
    for image_path in image_paths:
        out_path = output_dir / image_path.name
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out_path}")
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        image_bgr, instances = _predict_one(
            model,
            image_bgr,
            args.target_size,
            args.device,
            args.amp,
            args.amp_dtype,
        )
        cv2.imwrite(str(out_path), _draw(image_bgr, instances, args.class_name, args.score_thresh))
        processed += 1

    print(f"[DONE] wrote {processed} overlays to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
