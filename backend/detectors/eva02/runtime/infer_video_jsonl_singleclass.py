#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-class video inference for EVA-02 Cascade Mask R-CNN.
- Letterbox to target size (default 1280)
- Output JSONL with boxes + masks polygons per frame
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Avoid xFormers import failures; prefer SDPA
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("EVA2_PREFER_SDPA", "1")

# Provide a minimal xformers.ops fallback if xformers is not installed.
# This enables EVA-02 xattn path to run using PyTorch SDPA.
try:
    import xformers.ops as xops  # noqa: F401
except Exception:
    import types
    import torch.nn.functional as F

    def _memory_efficient_attention(q, k, v):
        # q,k,v: [B, N, H, C] -> SDPA expects [B, H, N, C]
        q_ = q.permute(0, 2, 1, 3)
        k_ = k.permute(0, 2, 1, 3)
        v_ = v.permute(0, 2, 1, 3)
        out = F.scaled_dot_product_attention(q_, k_, v_)
        return out.permute(0, 2, 1, 3)

    xformers_mod = types.ModuleType("xformers")
    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.memory_efficient_attention = _memory_efficient_attention
    xformers_mod.ops = ops_mod
    sys.modules["xformers"] = xformers_mod
    sys.modules["xformers.ops"] = ops_mod

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[3]
EVA02_DET_CANDIDATES = [
    BASE_DIR / "eva02_det",
    REPO_ROOT / "eva02" / "eva02_det",
]

# Prefer bundled EVA-02 detectron2 if available.
for _eva02_det in EVA02_DET_CANDIDATES:
    if _eva02_det.is_dir():
        sys.path.insert(0, str(_eva02_det))
        break

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from train_eva02_clean_final_minimal import (  # noqa: E402
    LetterboxTransform,
    compute_letterbox_params,
    unletterbox_instances,
)
from detectron2.config import LazyConfig, instantiate  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.layers.mask_ops import _do_paste_mask, paste_masks_in_image  # noqa: E402
from vit_pruning_codex import drop_blocks, parse_block_indices  # noqa: E402
from backend.detectors.jsonl_writer import AsyncJsonlWriter, JsonlWriter  # noqa: E402

try:
    import orjson  # type: ignore
except Exception:
    orjson = None

ORJSON_OPTS = getattr(orjson, "OPT_SERIALIZE_NUMPY", 0) if orjson is not None else 0


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MASK_RETR_MODE = "ccomp"  # ccomp|external
RPN_PRE_NMS_MULTIPLIER = 10


def _copy_tensor_to_cpu(
    tensor: torch.Tensor,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    src = tensor.detach()
    if src.device.type == "cpu":
        return src.contiguous()
    if pin_memory:
        dst = torch.empty_like(src, device="cpu", pin_memory=True)
        dst.copy_(src, non_blocking=non_blocking)
        return dst
    return src.to(device="cpu", non_blocking=False)


def _array_like_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            value = value.to("cpu")
        return value.numpy()
    return np.asarray(value)


def _payload_masks_to_array_like(value, mask_format: Optional[str]):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            value = value.to("cpu")
        return value.numpy()
    if isinstance(value, list):
        if mask_format == "cropped_full":
            return value
        try:
            return np.stack(value, axis=0)
        except Exception:
            return value
    return np.asarray(value)


def _patch_torch_load_for_checkpoint() -> None:
    """Allow loading legacy checkpoints with PyTorch>=2.6 defaults."""
    try:
        def _no_weights_only(_pickle_module=None):
            return False
        torch.serialization._default_to_weights_only = _no_weights_only
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata
        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
    except Exception:
        pass


def _patch_vit_attention_backend(prefer_sdpa: bool) -> None:
    if not prefer_sdpa:
        return
    try:
        import detectron2.modeling.backbone.vit as vit_mod
        import torch.nn.functional as F

        class _Xops:
            @staticmethod
            def memory_efficient_attention(q, k, v):
                q_ = q.permute(0, 2, 1, 3)
                k_ = k.permute(0, 2, 1, 3)
                v_ = v.permute(0, 2, 1, 3)
                out = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0, is_causal=False)
                return out.permute(0, 2, 1, 3)

        vit_mod.xops = _Xops
        print("[INFO] forced ViT attention backend: PyTorch SDPA")
    except Exception as exc:
        print(f"[WARN] failed to force SDPA attention backend: {exc}")


def _count_checkpoint_wrapped_blocks(model: torch.nn.Module) -> int:
    try:
        blocks = getattr(getattr(model, "backbone", None), "net", None)
        blocks = getattr(blocks, "blocks", [])
    except Exception:
        return -1
    wrapped = 0
    for block in blocks:
        cls = type(block)
        if "checkpoint" in cls.__name__.lower() or "checkpoint" in cls.__module__.lower():
            wrapped += 1
    return wrapped


def _collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTS:
            return [input_path]
        raise RuntimeError(f"Input file is not a video: {input_path}")
    if not input_path.is_dir():
        raise RuntimeError(f"Input path not found: {input_path}")
    if recursive:
        paths = [p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    else:
        paths = [p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    return sorted(paths)


def _build_model(
    config_path: Path,
    checkpoint: Path,
    num_classes: int,
    target_size: int,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    device: str,
    compile_backbone_mode: str,
    model_half: bool,
    backbone_half: bool,
    onnx_backbone_path: Optional[Path],
    onnx_max_batch: int,
    onnx_dtype: str,
    onnx_iobind: bool,
    onnx_disable_trt: bool,
    clean_sys_path: bool,
    compile_heads_mode: str,
    inductor_disable_cudagraphs: bool,
    disable_act_checkpoint: bool,
    prefer_sdpa: bool,
    drop_block_indices: Optional[list[int]],
    channels_last: bool = False,
    rpn_pre_nms_multiplier: int = 10,
) -> torch.nn.Module:
    cfg_path = str(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    import io
    import contextlib

    with contextlib.redirect_stderr(io.StringIO()):
        cfg = LazyConfig.load(cfg_path)

    if inductor_disable_cudagraphs:
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_skip_dynamic_graphs = True
            print("[INFO] torch._inductor cudagraphs disabled")
        except Exception as exc:
            print(f"[WARN] failed to disable inductor cudagraphs: {exc}")

    # Fix input size for ViT
    try:
        if hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "net") and hasattr(cfg.model.backbone.net, "img_size"):
            cfg.model.backbone.net.img_size = target_size
        if hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "square_pad"):
            cfg.model.backbone.square_pad = target_size
        if (
            disable_act_checkpoint
            and hasattr(cfg.model, "backbone")
            and hasattr(cfg.model.backbone, "net")
            and hasattr(cfg.model.backbone.net, "use_act_checkpoint")
        ):
            cfg.model.backbone.net.use_act_checkpoint = False
            print("[INFO] disabled backbone activation checkpointing in config")
    except Exception:
        pass

    cfg.model.roi_heads.num_classes = num_classes
    if hasattr(cfg.model.roi_heads, "box_predictors"):
        for predictor in cfg.model.roi_heads.box_predictors:
            predictor.test_score_thresh = score_thresh
            predictor.test_nms_thresh = nms_thresh
            predictor.test_topk_per_image = topk_per_image
    if hasattr(cfg.model, "proposal_generator"):
        pre_nms_topk = int(max(int(topk_per_image), int(topk_per_image) * int(rpn_pre_nms_multiplier)))
        cfg.model.proposal_generator.pre_nms_topk = (pre_nms_topk, pre_nms_topk)
        cfg.model.proposal_generator.post_nms_topk = (topk_per_image, topk_per_image)

    _patch_vit_attention_backend(prefer_sdpa=prefer_sdpa)
    model = instantiate(cfg.model).to(device).eval()
    wrapped_blocks = _count_checkpoint_wrapped_blocks(model)
    if wrapped_blocks >= 0:
        print(f"[INFO] backbone checkpoint_wrapped_blocks={wrapped_blocks}")

    checkpointer = DetectionCheckpointer(model)
    _patch_torch_load_for_checkpoint()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpointer.load(str(checkpoint))

    if drop_block_indices:
        model, applied_drop_indices = drop_blocks(model, drop_block_indices)
        print(
            f"[INFO] structurally dropped ViT blocks: indices={applied_drop_indices} "
            f"remaining={len(model.backbone.net.blocks)}"
        )

    if model_half and device.startswith("cuda"):
        model = model.half()
        print("[INFO] model.half() enabled")
    elif backbone_half and hasattr(model, "backbone") and hasattr(model.backbone, "net"):
        try:
            model.backbone.net = model.backbone.net.half()
            print("[INFO] backbone.net.half() enabled")
        except Exception as exc:
            print(f"[WARN] backbone_half failed: {exc}")

    use_channels_last = bool(channels_last and device.startswith("cuda"))
    os.environ["DETECTRON2_CHANNELS_LAST"] = "1" if use_channels_last else "0"
    if use_channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("[INFO] model channels_last enabled")
        except Exception as exc:
            os.environ["DETECTRON2_CHANNELS_LAST"] = "0"
            print(f"[WARN] channels_last failed: {exc}")

    if onnx_backbone_path:
        if clean_sys_path:
            sys.path = [p for p in sys.path if "eva02_trt" not in p]
        if not onnx_backbone_path.exists():
            raise FileNotFoundError(f"ONNX backbone not found: {onnx_backbone_path}")
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
            raise RuntimeError("model.backbone.net not found; cannot replace with ONNX backbone")
        # Configure ORT providers via env vars consumed by onnx_backbone.py
        os.environ.setdefault("EVA_REQUIRE_CUDA_EP", "1")
        os.environ.setdefault("EVA_DISABLE_CPU_EP", "1")
        os.environ["EVA_ONNXRT_IOBIND"] = "1" if onnx_iobind else "0"
        os.environ["EVA_DISABLE_TRT_EP"] = "1" if onnx_disable_trt else "0"

        from onnx_backbone import ORTBackboneAdapter

        expect_dtype = torch.float16 if onnx_dtype == "fp16" else torch.float32
        model.backbone.net = ORTBackboneAdapter(
            str(onnx_backbone_path),
            max_batch=onnx_max_batch,
            expect_dtype=expect_dtype,
        )
        print(f"[INFO] Using ONNX Runtime backbone: {onnx_backbone_path}")
    elif compile_backbone_mode != "none":
        if not hasattr(torch, "compile"):
            print("[WARN] torch.compile unavailable; skip compile_backbone")
        elif not hasattr(model, "backbone") or not hasattr(model.backbone, "net"):
            print("[WARN] model.backbone.net not found; skip compile_backbone")
        else:
            try:
                model.backbone.net = torch.compile(model.backbone.net, mode=compile_backbone_mode)
                print(f"[INFO] compiled backbone.net with mode={compile_backbone_mode}")
            except Exception as exc:
                print(f"[WARN] compile_backbone failed ({compile_backbone_mode}): {exc}")

    compile_proposal_generator = False
    compile_roi_heads = False
    resolved_compile_heads_mode = str(compile_heads_mode)
    if resolved_compile_heads_mode == "none":
        pass
    elif resolved_compile_heads_mode.startswith("proposal-only:"):
        compile_proposal_generator = True
        resolved_compile_heads_mode = resolved_compile_heads_mode.split(":", 1)[1]
    elif resolved_compile_heads_mode.startswith("roi-only:"):
        compile_roi_heads = True
        resolved_compile_heads_mode = resolved_compile_heads_mode.split(":", 1)[1]
    else:
        compile_proposal_generator = True
        compile_roi_heads = True

    if compile_proposal_generator or compile_roi_heads:
        if compile_roi_heads and hasattr(model, "roi_heads"):
            try:
                model.roi_heads.fixed_num_proposals = int(topk_per_image)
                print(f"[INFO] fixed roi_heads proposal count={int(topk_per_image)} for compile_heads")
            except Exception as exc:
                print(f"[WARN] failed to set fixed roi_heads proposal count: {exc}")
            try:
                if hasattr(model.roi_heads, "box_pooler"):
                    model.roi_heads.box_pooler.fixed_boxes_per_image = int(topk_per_image)
                    print(f"[INFO] fixed roi_heads.box_pooler boxes/image={int(topk_per_image)}")
            except Exception as exc:
                print(f"[WARN] failed to set fixed roi_heads box_pooler boxes/image: {exc}")
        if not hasattr(torch, "compile"):
            print("[WARN] torch.compile unavailable; skip compile_heads")
        else:
            try:
                if compile_proposal_generator and hasattr(model, "proposal_generator"):
                    model.proposal_generator = torch.compile(
                        model.proposal_generator, mode=resolved_compile_heads_mode
                    )
                    print(f"[INFO] compiled proposal_generator with mode={resolved_compile_heads_mode}")
                if compile_roi_heads and hasattr(model, "roi_heads"):
                    model.roi_heads = torch.compile(model.roi_heads, mode=resolved_compile_heads_mode)
                    print(f"[INFO] compiled roi_heads with mode={resolved_compile_heads_mode}")
            except Exception as exc:
                print(f"[WARN] compile_heads failed ({compile_heads_mode}): {exc}")

    return model


def _prep_image(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise RuntimeError("Failed to read image")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    return img_bgr


def _prepare_batch_inputs(
    frames_bgr: list[np.ndarray],
    target_size: int,
    device: str,
    gpu_preprocess_float: bool,
    model_half: bool,
    pin_inputs: bool = False,
    pack_inputs: bool = True,
) -> tuple[list[dict], list[tuple[int, int, dict]]]:
    inputs: list[dict] = []
    metas: list[tuple[int, int, dict]] = []
    device_is_cuda = device.startswith("cuda")

    if pack_inputs and frames_bgr:
        batch_hwc = np.empty((len(frames_bgr), target_size, target_size, 3), dtype=np.uint8)
        for idx, frame_bgr in enumerate(frames_bgr):
            image_bgr = _prep_image(frame_bgr)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            tfm = LetterboxTransform(src_shape=(h, w), target_size=target_size, pad_value=128)
            batch_hwc[idx] = tfm.apply_image(image_rgb)
            metas.append((h, w, compute_letterbox_params(h, w, target_size)))

        batch = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).contiguous()
        if device_is_cuda and pin_inputs:
            batch = batch.pin_memory()

        if device_is_cuda and gpu_preprocess_float:
            batch = batch.to(device, non_blocking=True, dtype=torch.uint8).float()
        else:
            batch = batch.float().to(device, non_blocking=True)
        if model_half and device_is_cuda:
            batch = batch.half()

        for idx in range(batch.shape[0]):
            inputs.append({"image": batch[idx], "height": target_size, "width": target_size})
        return inputs, metas

    for frame_bgr in frames_bgr:
        image_bgr = _prep_image(frame_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        tfm = LetterboxTransform(src_shape=(h, w), target_size=target_size, pad_value=128)
        lb = tfm.apply_image(image_rgb)
        x = torch.from_numpy(lb.transpose(2, 0, 1)).contiguous()
        if device_is_cuda and pin_inputs:
            x = x.pin_memory()

        if device_is_cuda and gpu_preprocess_float:
            x = x.to(device, non_blocking=True, dtype=torch.uint8).float()
        else:
            x = x.float().to(device, non_blocking=True)
        if model_half and device_is_cuda:
            x = x.half()

        inputs.append({"image": x, "height": target_size, "width": target_size})
        metas.append((h, w, compute_letterbox_params(h, w, target_size)))

    return inputs, metas


def _run_model(
    model: torch.nn.Module,
    inputs: list[dict],
    device: str,
    amp: bool,
    model_half: bool,
    postprocess: bool = True,
):
    with torch.inference_mode():
        # Mark each batched invocation as a fresh cudagraph step so repeated
        # compiled runs do not reuse overwritten graph outputs across batches.
        if device.startswith("cuda"):
            try:
                mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
                if mark_step is not None:
                    mark_step()
            except Exception:
                pass
        if model_half and device.startswith("cuda"):
            if postprocess:
                return model(inputs)
            return [{"instances": x} for x in model.inference(inputs, do_postprocess=False)]
        if amp and device.startswith("cuda") and hasattr(torch, "amp"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if postprocess:
                    return model(inputs)
                return [{"instances": x} for x in model.inference(inputs, do_postprocess=False)]
        if postprocess:
            return model(inputs)
        return [{"instances": x} for x in model.inference(inputs, do_postprocess=False)]


def _predict_one(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    target_size: int,
    device: str,
    amp: bool,
    model_half: bool = False,
    gpu_preprocess_float: bool = False,
    pin_inputs: bool = False,
    pack_inputs: bool = True,
):
    inputs, metas = _prepare_batch_inputs(
        frames_bgr=[image_bgr],
        target_size=target_size,
        device=device,
        gpu_preprocess_float=gpu_preprocess_float,
        model_half=model_half,
        pin_inputs=pin_inputs,
        pack_inputs=pack_inputs,
    )
    outputs = _run_model(model, inputs, device, amp, model_half)
    instances = outputs[0]["instances"]
    h, w, lb_meta = metas[0]
    instances = unletterbox_instances(instances, lb_meta, h, w, target_size)
    return image_bgr, instances.to("cpu")


def _mask_to_polygons(mask: np.ndarray, approx_mode: str) -> list[list[float]]:
    # Mirrors detectron2 GenericMask.mask_to_polygons without matplotlib dependency.
    mask = np.ascontiguousarray(mask.astype("uint8"))
    chain_mode = cv2.CHAIN_APPROX_SIMPLE if approx_mode == "simple" else cv2.CHAIN_APPROX_NONE
    retr_mode = cv2.RETR_EXTERNAL if str(MASK_RETR_MODE).lower() == "external" else cv2.RETR_CCOMP
    res = cv2.findContours(mask, retr_mode, chain_mode)
    hierarchy = res[-1]
    if hierarchy is None:
        return []
    contours = res[-2]
    polys = []
    for c in contours:
        c = c.flatten()
        if len(c) >= 6:
            c = c.astype(np.float32) + 0.5
            polys.append(c.tolist())
    return polys


def _offset_polygons(polys: list[list[float]], x_off: float, y_off: float) -> list[list[float]]:
    if not polys or (x_off == 0.0 and y_off == 0.0):
        return polys
    out: list[list[float]] = []
    for poly in polys:
        shifted = poly.copy()
        for idx in range(0, len(shifted), 2):
            shifted[idx] += x_off
            shifted[idx + 1] += y_off
        out.append(shifted)
    return out


def _mask_to_polygons_in_box(
    mask: np.ndarray,
    approx_mode: str,
    bbox_xyxy: Optional[np.ndarray],
) -> list[list[float]]:
    if bbox_xyxy is None:
        return _mask_to_polygons(mask, approx_mode)

    h, w = mask.shape[:2]
    x1 = max(0, min(w, int(np.floor(float(bbox_xyxy[0])))))
    y1 = max(0, min(h, int(np.floor(float(bbox_xyxy[1])))))
    x2 = max(0, min(w, int(np.ceil(float(bbox_xyxy[2])))))
    y2 = max(0, min(h, int(np.ceil(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return _mask_to_polygons(mask, approx_mode)

    cropped = mask[y1:y2, x1:x2]
    polys = _mask_to_polygons(cropped, approx_mode)
    return _offset_polygons(polys, float(x1), float(y1))


def _paste_roi_masks_to_union_crop(
    masks: np.ndarray,
    boxes: np.ndarray,
    height: int,
    width: int,
    threshold: float = 0.5,
) -> tuple[Optional[np.ndarray], tuple[int, int]]:
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None, (0, 0)

    masks_t = torch.from_numpy(np.ascontiguousarray(masks)).float()
    boxes_t = torch.from_numpy(np.ascontiguousarray(boxes)).float()
    pasted, spatial_inds = _do_paste_mask(masks_t[:, None, :, :], boxes_t, int(height), int(width), skip_empty=True)
    pasted = (pasted >= float(threshold)).to(dtype=torch.uint8).cpu().numpy()
    y_off = int(spatial_inds[0].start) if spatial_inds else 0
    x_off = int(spatial_inds[1].start) if spatial_inds else 0
    return pasted, (x_off, y_off)


def _materialize_full_masks(instances, height: int, width: int, threshold: float = 0.5):
    if not instances.has("pred_masks") or len(instances) == 0:
        return instances
    masks = instances.pred_masks
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim != 3:
        return instances
    if int(masks.shape[-2]) == int(height) and int(masks.shape[-1]) == int(width):
        return instances
    if not instances.has("pred_boxes"):
        return instances

    out = instances.to("cpu")
    masks_t = out.pred_masks.float()
    if masks_t.ndim == 4:
        masks_t = masks_t[:, 0]
    boxes_t = out.pred_boxes.tensor.float()
    out.pred_masks = paste_masks_in_image(masks_t, boxes_t, (int(height), int(width)), threshold=float(threshold))
    return out


def _crop_full_mask_to_box(mask, bbox_xyxy: Optional[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
    mask_h, mask_w = mask.shape[-2:]
    if bbox_xyxy is None:
        if isinstance(mask, torch.Tensor):
            mask_np = (mask > 0.5).to(dtype=torch.uint8).cpu().numpy()
        else:
            mask_np = (np.asarray(mask) > 0.5).astype(np.uint8, copy=False)
        return mask_np, (0, 0)

    x1 = max(0, min(mask_w, int(np.floor(float(bbox_xyxy[0])))))
    y1 = max(0, min(mask_h, int(np.floor(float(bbox_xyxy[1])))))
    x2 = max(0, min(mask_w, int(np.ceil(float(bbox_xyxy[2])))))
    y2 = max(0, min(mask_h, int(np.ceil(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=np.uint8), (x1, y1)

    if isinstance(mask, torch.Tensor):
        cropped = (mask[y1:y2, x1:x2] > 0.5).to(dtype=torch.uint8).cpu().numpy()
    else:
        cropped = (np.asarray(mask[y1:y2, x1:x2]) > 0.5).astype(np.uint8, copy=False)
    return cropped, (x1, y1)


def _build_serialization_payload(instances, height: int, width: int) -> dict:
    payload: dict = {
        "_kind": "compact_instances",
        "height": int(height),
        "width": int(width),
        "boxes": None,
        "scores": None,
        "masks": None,
        "mask_format": None,
        "mask_offsets": None,
    }

    num_inst = len(instances)
    if num_inst == 0:
        return payload

    boxes = instances.pred_boxes.tensor.detach().cpu().numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.detach().cpu().numpy() if instances.has("scores") else None
    payload["boxes"] = boxes
    payload["scores"] = scores

    if not instances.has("pred_masks"):
        return payload

    masks = instances.pred_masks
    if isinstance(masks, torch.Tensor) and masks.ndim == 4:
        masks = masks[:, 0]
    elif not isinstance(masks, torch.Tensor):
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
    if getattr(masks, "ndim", 0) != 3:
        return payload

    mask_h = int(masks.shape[-2])
    mask_w = int(masks.shape[-1])
    if boxes is not None and mask_h == int(height) and mask_w == int(width):
        cropped_masks: list[np.ndarray] = []
        mask_offsets: list[tuple[int, int]] = []
        for idx in range(num_inst):
            cropped_mask, offset_xy = _crop_full_mask_to_box(masks[idx], boxes[idx])
            cropped_masks.append(cropped_mask)
            mask_offsets.append(offset_xy)
        payload["masks"] = cropped_masks
        payload["mask_offsets"] = mask_offsets
        payload["mask_format"] = "cropped_full"
        return payload

    if isinstance(masks, torch.Tensor):
        roi_masks = [(masks[idx] > 0.5).to(dtype=torch.uint8).cpu().numpy() for idx in range(num_inst)]
    else:
        roi_masks = [(np.asarray(masks[idx]) > 0.5).astype(np.uint8, copy=False) for idx in range(num_inst)]
    payload["masks"] = roi_masks
    payload["mask_format"] = "roi"
    return payload


def _build_tensor_serialization_payload(
    instances,
    height: int,
    width: int,
    *,
    pin_memory: bool = False,
    non_blocking: bool = False,
) -> dict:
    payload: dict = {
        "_kind": "compact_instances",
        "height": int(height),
        "width": int(width),
        "boxes": None,
        "scores": None,
        "masks": None,
        "mask_format": None,
        "mask_offsets": None,
    }

    num_inst = len(instances)
    if num_inst == 0:
        return payload

    if instances.has("pred_boxes"):
        payload["boxes"] = _copy_tensor_to_cpu(
            instances.pred_boxes.tensor,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
    if instances.has("scores"):
        payload["scores"] = _copy_tensor_to_cpu(
            instances.scores,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
    if not instances.has("pred_masks"):
        return payload

    masks = instances.pred_masks
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.ndim == 3:
            payload["masks"] = _copy_tensor_to_cpu(
                masks,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            mask_h = int(masks.shape[-2])
            mask_w = int(masks.shape[-1])
            payload["mask_format"] = "full" if mask_h == int(height) and mask_w == int(width) else "roi"
        return payload

    masks_np = np.asarray(masks)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    if masks_np.ndim == 3:
        payload["masks"] = masks_np
        mask_h = int(masks_np.shape[-2])
        mask_w = int(masks_np.shape[-1])
        payload["mask_format"] = "full" if mask_h == int(height) and mask_w == int(width) else "roi"
    return payload


def _build_device_serialization_payload(
    instances,
    height: int,
    width: int,
) -> dict:
    payload: dict = {
        "_kind": "compact_instances",
        "height": int(height),
        "width": int(width),
        "boxes": None,
        "scores": None,
        "masks": None,
        "mask_format": None,
        "mask_offsets": None,
    }

    num_inst = len(instances)
    if num_inst == 0:
        return payload

    if instances.has("pred_boxes"):
        payload["boxes"] = instances.pred_boxes.tensor.detach().clone()
    if instances.has("scores"):
        payload["scores"] = instances.scores.detach().clone()
    if not instances.has("pred_masks"):
        return payload

    masks = instances.pred_masks
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.ndim == 3:
            payload["masks"] = masks.detach().clone()
            mask_h = int(masks.shape[-2])
            mask_w = int(masks.shape[-1])
            payload["mask_format"] = "full" if mask_h == int(height) and mask_w == int(width) else "roi"
        return payload

    masks_np = np.asarray(masks)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    if masks_np.ndim == 3:
        payload["masks"] = masks_np.copy()
        mask_h = int(masks_np.shape[-2])
        mask_w = int(masks_np.shape[-1])
        payload["mask_format"] = "full" if mask_h == int(height) and mask_w == int(width) else "roi"
    return payload


def _serialization_item_len(item) -> int:
    if isinstance(item, dict) and item.get("_kind") == "compact_instances":
        boxes = item.get("boxes")
        return 0 if boxes is None else int(len(boxes))
    return len(item)


def _serialization_payload_to_json(
    payload: dict,
    class_name: str,
    score_thresh: float,
    mask_approx: str,
) -> list[dict]:
    boxes = _array_like_to_numpy(payload.get("boxes"))
    scores = _array_like_to_numpy(payload.get("scores"))
    mask_format = payload.get("mask_format")
    masks = _payload_masks_to_array_like(payload.get("masks"), mask_format)
    mask_offsets = payload.get("mask_offsets")
    union_crop_offset = payload.get("union_crop_offset")

    num_inst = 0 if boxes is None else int(len(boxes))
    if num_inst == 0:
        return []

    union_crop_masks = None
    if mask_format == "roi" and masks is not None and boxes is not None:
        pasted_masks, pasted_offset = _paste_roi_masks_to_union_crop(
            masks=masks if isinstance(masks, np.ndarray) else np.asarray(masks),
            boxes=np.asarray(boxes),
            height=int(payload["height"]),
            width=int(payload["width"]),
        )
        if pasted_masks is not None and len(pasted_masks) == num_inst:
            union_crop_masks = pasted_masks
            union_crop_offset = pasted_offset

    results: list[dict] = []
    for i in range(num_inst):
        score = float(scores[i]) if scores is not None else None
        if score is not None and score_thresh > 0 and score < score_thresh:
            continue

        item = {"class_name": class_name, "category_id": 0}
        box_xyxy = boxes[i]
        x1, y1, x2, y2 = box_xyxy.tolist()
        item["bbox_xyxy"] = [float(x1), float(y1), float(x2), float(y2)]
        item["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if score is not None:
            item["score"] = score

        if masks is not None:
            mask = masks[i]
            if mask_format == "cropped_full":
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                polygons = _offset_polygons(
                    _mask_to_polygons(np.ascontiguousarray(mask.astype(np.uint8, copy=False)), mask_approx),
                    float(mask_offsets[i][0]),
                    float(mask_offsets[i][1]),
                )
            elif mask_format == "full" and box_xyxy is not None:
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                mask_bin = mask.astype(np.uint8, copy=False)
                if mask_bin.max() > 1:
                    mask_bin = (mask_bin > 0.5).astype(np.uint8)
                polygons = _mask_to_polygons_in_box(mask_bin, mask_approx, box_xyxy)
            elif mask_format == "roi" and union_crop_masks is not None and union_crop_offset is not None:
                polygons = _offset_polygons(
                    _mask_to_polygons(np.ascontiguousarray(union_crop_masks[i].astype(np.uint8, copy=False)), mask_approx),
                    float(union_crop_offset[0]),
                    float(union_crop_offset[1]),
                )
            elif box_xyxy is not None:
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                polygons = _roi_mask_to_polygons_with_box(
                    mask=np.ascontiguousarray(mask.astype(np.uint8, copy=False)),
                    bbox_xyxy=box_xyxy,
                    approx_mode=mask_approx,
                    image_h=int(payload["height"]),
                    image_w=int(payload["width"]),
                )
            else:
                polygons = []
            item["segmentation"] = polygons

        results.append(item)

    return results


def _serialization_item_to_json(
    item,
    class_name: str,
    score_thresh: float,
    height: int,
    width: int,
    mask_approx: str,
) -> list[dict]:
    if isinstance(item, dict) and item.get("_kind") == "compact_instances":
        return _serialization_payload_to_json(
            payload=item,
            class_name=class_name,
            score_thresh=score_thresh,
            mask_approx=mask_approx,
        )
    return _instances_to_json(
        instances=item,
        class_name=class_name,
        score_thresh=score_thresh,
        height=height,
        width=width,
        mask_approx=mask_approx,
    )


def _roi_mask_to_polygons_with_box(
    mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    approx_mode: str,
    image_h: int,
    image_w: int,
) -> list[list[float]]:
    x1 = max(0, min(int(image_w), int(np.floor(float(bbox_xyxy[0])))))
    y1 = max(0, min(int(image_h), int(np.floor(float(bbox_xyxy[1])))))
    x2 = max(0, min(int(image_w), int(np.ceil(float(bbox_xyxy[2])))))
    y2 = max(0, min(int(image_h), int(np.ceil(float(bbox_xyxy[3])))))
    if x2 <= x1 or y2 <= y1:
        return []

    mask_bin = (mask > 0.5).astype(np.uint8)
    resized = cv2.resize(mask_bin, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    polys = _mask_to_polygons(resized, approx_mode)
    return _offset_polygons(polys, float(x1), float(y1))


def _instances_to_json(
    instances,
    class_name: str,
    score_thresh: float,
    height: int,
    width: int,
    mask_approx: str,
) -> list[dict]:
    if hasattr(instances, "scores") and score_thresh > 0:
        keep = instances.scores >= score_thresh
        instances = instances[keep]

    num_inst = len(instances)
    if num_inst == 0:
        return []

    boxes = instances.pred_boxes.tensor.cpu().numpy() if instances.has("pred_boxes") else None
    scores = instances.scores.cpu().numpy() if instances.has("scores") else None
    masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
    raw_roi_masks = False
    union_crop_masks = None
    union_crop_offset = None
    if masks is not None and boxes is not None:
        mask_h = int(masks.shape[-2]) if masks.ndim >= 3 else -1
        mask_w = int(masks.shape[-1]) if masks.ndim >= 3 else -1
        raw_roi_masks = mask_h != int(height) or mask_w != int(width)
        if raw_roi_masks:
            pasted_masks, pasted_offset = _paste_roi_masks_to_union_crop(
                masks=masks,
                boxes=boxes,
                height=int(height),
                width=int(width),
            )
            if pasted_masks is not None and len(pasted_masks) == num_inst:
                union_crop_masks = pasted_masks
                union_crop_offset = pasted_offset

    results = []
    for i in range(num_inst):
        item = {"class_name": class_name, "category_id": 0}
        box_xyxy = None
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i].tolist()
            box_xyxy = boxes[i]
            item["bbox_xyxy"] = [float(x1), float(y1), float(x2), float(y2)]
            item["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        if scores is not None:
            item["score"] = float(scores[i])
        if masks is not None:
            if raw_roi_masks and union_crop_masks is not None and union_crop_offset is not None:
                polygons = _offset_polygons(
                    _mask_to_polygons(np.ascontiguousarray(union_crop_masks[i].astype(np.uint8, copy=False)), mask_approx),
                    float(union_crop_offset[0]),
                    float(union_crop_offset[1]),
                )
            elif raw_roi_masks and box_xyxy is not None:
                mask = masks[i]
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                polygons = _roi_mask_to_polygons_with_box(
                    mask=mask,
                    bbox_xyxy=box_xyxy,
                    approx_mode=mask_approx,
                    image_h=int(height),
                    image_w=int(width),
                )
            else:
                mask = masks[i]
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                mask_bin = mask.astype(np.uint8)
                if mask_bin.max() > 1:
                    mask_bin = (mask_bin > 0.5).astype(np.uint8)
                polygons = _mask_to_polygons_in_box(mask_bin, mask_approx, box_xyxy)
            item["segmentation"] = polygons
        results.append(item)

    return results


class _JsonlWriter(JsonlWriter):
    def __init__(self, output_path: Path, backend: str):
        super().__init__(output_path=output_path, backend=backend, orjson_option=ORJSON_OPTS)


class _AsyncJsonlWriter(AsyncJsonlWriter):
    def __init__(self, output_path: Path, backend: str, max_queue: int = 512):
        super().__init__(output_path=output_path, backend=backend, max_queue=max_queue, orjson_option=ORJSON_OPTS)


class _FramePrefetcher:
    def __init__(self, cap: cv2.VideoCapture, max_queue: int, max_frames: int | None):
        self._cap = cap
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._max_frames = max_frames
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        idx = 0
        try:
            while True:
                if self._max_frames is not None and idx >= self._max_frames:
                    break
                ok, frame = self._cap.read()
                if not ok:
                    break
                self._queue.put((idx, frame))
                idx += 1
        finally:
            self._queue.put(self._stop)

    def get(self) -> object:
        return self._queue.get()

    def close(self) -> None:
        self._thread.join()


def _infer_video_jsonl(
    model: torch.nn.Module,
    video_path: Path,
    output_path: Path,
    class_name: str,
    target_size: int,
    device: str,
    amp: bool,
    score_thresh: float,
    max_frames: int | None,
    measure: bool,
    warmup_frames: int,
    batch_size: int,
    json_backend: str,
    gpu_preprocess_float: bool,
    model_half: bool,
    flush_every: int,
    mask_approx: str,
    async_writer: bool,
    prefetch: bool,
    prefetch_queue: int,
) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >=1, got {batch_size}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if async_writer:
        writer = _AsyncJsonlWriter(output_path=output_path, backend=json_backend)
    else:
        writer = _JsonlWriter(output_path=output_path, backend=json_backend)

    total_time = 0.0
    measured_frames = 0
    processed_frames = 0
    frames_bgr: list[np.ndarray] = []
    frame_ids: list[int] = []
    prefetcher = _FramePrefetcher(cap, max_queue=prefetch_queue, max_frames=max_frames) if prefetch else None

    try:
        while True:
            end_of_stream = False
            if prefetcher is not None:
                item = prefetcher.get()
                if item is prefetcher._stop:
                    end_of_stream = True
                else:
                    frame_id, frame_bgr = item  # type: ignore[misc]
                    frames_bgr.append(frame_bgr)
                    frame_ids.append(frame_id)
                    processed_frames += 1
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    end_of_stream = True
                else:
                    frames_bgr.append(frame_bgr)
                    frame_ids.append(processed_frames)
                    processed_frames += 1
                    if max_frames is not None and processed_frames >= max_frames:
                        end_of_stream = True

            if len(frames_bgr) < batch_size and not end_of_stream:
                continue
            if not frames_bgr:
                break

            if measure and device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter() if measure else None

            inputs, metas = _prepare_batch_inputs(
                frames_bgr=frames_bgr,
                target_size=target_size,
                device=device,
                gpu_preprocess_float=gpu_preprocess_float,
                model_half=model_half,
            )
            outputs = _run_model(model, inputs, device, amp, model_half)

            for out_item, frame_id, (h_src, w_src, lb_meta) in zip(outputs, frame_ids, metas):
                instances = out_item["instances"]
                instances = unletterbox_instances(instances, lb_meta, h_src, w_src, target_size).to("cpu")
                inst_list = _instances_to_json(instances, class_name, score_thresh, height, width, mask_approx)

                record = {
                    "frame_idx": frame_id,
                    "time_sec": float(frame_id / fps),
                    "width": width,
                    "height": height,
                    "instances": inst_list,
                }
                writer.write(record)

            if measure and device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            if measure and t0 is not None:
                t1 = time.perf_counter()
                batch_measured = sum(1 for fid in frame_ids if fid >= warmup_frames)
                if batch_measured > 0:
                    # Warmup boundary can fall inside a batch; scale batch time proportionally.
                    total_time += (t1 - t0) * (batch_measured / len(frame_ids))
                    measured_frames += batch_measured

            if processed_frames % flush_every == 0:
                writer.flush()
                print(f"[INFO] {video_path.name}: processed {processed_frames} frames")

            frames_bgr.clear()
            frame_ids.clear()
            if end_of_stream:
                break
    finally:
        cap.release()
        if prefetcher is not None:
            prefetcher.close()
        writer.flush()
        writer.close()

    print(f"[DONE] wrote {processed_frames} frames -> {output_path}")
    if measure and measured_frames > 0:
        fps_e2e = measured_frames / total_time if total_time > 0 else 0.0
        print(f"[MEASURE] e2e_fps={fps_e2e:.4f} over {measured_frames} frames (warmup={warmup_frames})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-class video inference for EVA-02 Cascade Mask R-CNN (JSONL output)")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output", required=True, help="Output directory for JSONL")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "checkpoints/eva02/detector/model_final.pth"),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--config",
        default=str(
            REPO_ROOT
            / "eva02/eva02_det/projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1280_lrd0p8.py"
        ),
        help="Detectron2 LazyConfig path",
    )
    parser.add_argument("--class-name", default="foreground", help="Class name to display")
    parser.add_argument("--target-size", type=int, default=1280, help="Letterbox size")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold for output")
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="NMS threshold")
    parser.add_argument("--topk", type=int, default=200, help="Top-K per image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use AMP on CUDA")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (debug)")
    parser.add_argument("--measure", action="store_true", help="Measure E2E FPS with CUDA sync")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Warmup frames excluded from measurement")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for batched inference")
    parser.add_argument(
        "--compile-backbone",
        choices=["none", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"],
        default="none",
        help="Apply torch.compile to model.backbone.net",
    )
    parser.add_argument(
        "--compile-heads",
        choices=["none", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"],
        default="none",
        help="Apply torch.compile to ROI heads / proposal generator",
    )
    parser.add_argument("--onnx-backbone", default=None, help="Use ONNX Runtime backbone (.onnx)")
    parser.add_argument("--onnx-dtype", choices=["fp16", "fp32"], default="fp16", help="ONNX input dtype")
    parser.add_argument("--onnx-max-batch", type=int, default=12, help="Max batch for ONNX backbone")
    parser.add_argument("--onnx-iobind", action="store_true", help="Enable ORT IO binding")
    parser.add_argument("--onnx-disable-trt", action="store_true", help="Disable TensorRT EP for ORT")
    parser.add_argument("--clean-sys-path", action="store_true", help="Drop conda env paths to prefer venv packages")
    parser.add_argument("--model-half", action="store_true", help="Run model with half-precision weights")
    parser.add_argument("--backbone-half", action="store_true", help="Cast only backbone.net weights to half")
    parser.add_argument("--inductor-disable-cudagraphs", action="store_true", help="Disable torch._inductor cudagraphs")
    parser.add_argument("--disable-act-checkpoint", action="store_true", help="Disable ViT activation checkpoint wrappers at model build time")
    parser.add_argument("--prefer-sdpa", action="store_true", help="Force ViT attention to use PyTorch SDPA instead of xFormers")
    parser.add_argument("--drop-block-indices", default="", help="Comma-separated ViT block indices to remove structurally")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last for Detectron2 image batches and 4D weights")
    parser.add_argument(
        "--gpu-preprocess-float",
        action="store_true",
        help="Send uint8 letterboxed tensor to GPU first, then cast to float on GPU",
    )
    parser.add_argument(
        "--json-backend",
        choices=["json", "orjson"],
        default="json",
        help="JSON serializer backend for output",
    )
    parser.add_argument("--flush-every", type=int, default=50, help="Flush JSONL every N processed frames")
    parser.add_argument(
        "--mask-approx",
        choices=["none", "simple"],
        default="none",
        help="Mask polygon contour approximation (simple reduces points)",
    )
    parser.add_argument("--async-writer", action="store_true", help="Write JSONL in a background thread")
    parser.add_argument("--prefetch", action="store_true", help="Prefetch frames in a background thread")
    parser.add_argument("--prefetch-queue", type=int, default=64, help="Prefetch queue size")

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(
        config_path=Path(args.config),
        checkpoint=Path(args.checkpoint),
        num_classes=1,
        target_size=args.target_size,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        topk_per_image=args.topk,
        device=args.device,
        compile_backbone_mode=args.compile_backbone,
        model_half=args.model_half,
        backbone_half=args.backbone_half,
        onnx_backbone_path=Path(args.onnx_backbone).expanduser().resolve() if args.onnx_backbone else None,
        onnx_max_batch=args.onnx_max_batch,
        onnx_dtype=args.onnx_dtype,
        onnx_iobind=args.onnx_iobind,
        onnx_disable_trt=args.onnx_disable_trt,
        clean_sys_path=args.clean_sys_path,
        compile_heads_mode=args.compile_heads,
        inductor_disable_cudagraphs=args.inductor_disable_cudagraphs,
        disable_act_checkpoint=args.disable_act_checkpoint,
        prefer_sdpa=args.prefer_sdpa,
        drop_block_indices=parse_block_indices(args.drop_block_indices),
        channels_last=args.channels_last,
    )

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    video_paths = _collect_videos(input_path, args.recursive)
    if not video_paths:
        raise RuntimeError(f"No videos found under: {input_path}")

    processed = 0
    for vid_path in video_paths:
        out_name = vid_path.stem + ".jsonl"
        out_path = output_dir / out_name
        if out_path.exists() and not args.overwrite:
            continue

        _infer_video_jsonl(
            model=model,
            video_path=vid_path,
            output_path=out_path,
            class_name=args.class_name,
            target_size=args.target_size,
            device=args.device,
            amp=args.amp,
            score_thresh=args.score_thresh,
            max_frames=args.max_frames,
            measure=args.measure,
            warmup_frames=args.warmup_frames,
            batch_size=args.batch_size,
            json_backend=args.json_backend,
            gpu_preprocess_float=args.gpu_preprocess_float,
            model_half=args.model_half,
            flush_every=max(1, args.flush_every),
            mask_approx=args.mask_approx,
            async_writer=args.async_writer,
            prefetch=args.prefetch,
            prefetch_queue=max(1, args.prefetch_queue),
        )
        processed += 1

    print(f"[DONE] processed {processed} videos -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
