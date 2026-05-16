#!/usr/bin/env python3
"""Fast video inference for DINOv3 ViT-L + Co-DINO instance segmentation.

The default config/checkpoint point at the current 0423 LR-restart epoch_2 run.
The script keeps the exact 720x1280 letterbox test pipeline, batches frames,
uses torch inference_mode/autocast, and draws overlays with OpenCV instead of
MMDetection's slower matplotlib-style visualizer.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
CODINO_ROOT = ROOT / "external" / "codino"
DINOV3_ROOT = ROOT / "dinov3"
DEFAULT_RUN_DIR = ROOT / "checkpoints" / "codino" / "detector"
DEFAULT_CONFIG = DEFAULT_RUN_DIR / "resolved_config.py"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "epoch_2.pth"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "codino_video_inference"
CASCADE_INFER_ROOT = ROOT / "backend" / "detectors" / "dinov3" / "runtime"
DEFAULT_ROI_CLASSIFIER_CKPT = ROOT / "checkpoints" / "codino" / "classifier" / "best.pt"


def _default_trt_site_packages() -> Path | None:
    raw = os.environ.get("TENSORRT_SITE_PACKAGES")
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    home = Path.home()
    for env_name in ("eva02_trt", "trt_env"):
        for pyver in ("python3.10", "python3.11", "python3.12", "python3.8"):
            path = home / "miniconda3" / "envs" / env_name / "lib" / pyver / "site-packages"
            if (path / "tensorrt").exists():
                return path
    return None


DEFAULT_TRT_SITE_PACKAGES = _default_trt_site_packages()
CLASSIFIER_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 220, 255),
    (80, 180, 255),
    (80, 255, 120),
    (255, 180, 80),
    (220, 120, 255),
)


def _prepare_imports() -> None:
    for path in (ROOT, DINOV3_ROOT, CODINO_ROOT, CASCADE_INFER_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # Register Co-DINO project heads and local DINOv3 backbone.
    import projects.models  # noqa: F401
    import mmdet.models.backbones.dinov3_vit  # noqa: F401

    # Register local unified letterbox transforms by path so an editable install
    # cannot accidentally resolve another working tree.
    aug_path = CODINO_ROOT / "mmdet/datasets/pipelines/unified_letterbox_aug.py"
    spec = importlib.util.spec_from_file_location("unified_letterbox_aug_local", aug_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load pipeline transforms: {aug_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Compatibility for newer torch where mmcv may pass an int device id.
    import mmcv.parallel._functions as mmcv_parallel_functions

    orig_get_stream = mmcv_parallel_functions._get_stream

    def torch210_compatible_get_stream(device):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return orig_get_stream(device)

    mmcv_parallel_functions._get_stream = torch210_compatible_get_stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=None, help="Output overlay video path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional bbox prediction JSON path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument(
        "--model-score-thr",
        type=float,
        default=None,
        help="Filter detections before mask prediction. Defaults to min(--score-thr, 0.05); use 0 for raw model output.",
    )
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--onnx-backbone",
        type=Path,
        default=None,
        help="Optional ONNX Runtime DINOv3 backbone. It must match the padded input size.",
    )
    parser.add_argument("--onnx-max-batch", type=int, default=4)
    parser.add_argument(
        "--trt-backbone-engine",
        type=Path,
        default=None,
        help="Optional native TensorRT DINOv3 backbone engine. Fixed-batch engines are supported.",
    )
    parser.add_argument(
        "--trt-feature-engine",
        type=Path,
        default=None,
        help="Optional native TensorRT backbone+neck engine returning Co-DINO multi-level features.",
    )
    parser.add_argument("--trt-feature-names", default="feat0,feat1,feat2,feat3,feat4")
    parser.add_argument(
        "--trt-query-encoder-engine",
        type=Path,
        default=None,
        help="Optional native TensorRT Co-DINO transformer encoder engine. Fixed batch/shape only.",
    )
    parser.add_argument(
        "--trt-query-encoder-shapes",
        default="184x320,92x160,46x80,23x40,12x20",
        help="Feature HW shapes baked into --trt-query-encoder-engine.",
    )
    parser.add_argument(
        "--trt-decoder-engine",
        type=Path,
        default=None,
        help="Optional native TensorRT Co-DINO transformer decoder engine. Fixed batch/shape only.",
    )
    parser.add_argument(
        "--trt-mask-head-engine",
        type=Path,
        default=None,
        help="Optional native TensorRT Co-DINO instance mask head engine. Fixed RoI count/shape only.",
    )
    parser.add_argument(
        "--trt-extra-site-packages",
        type=Path,
        default=DEFAULT_TRT_SITE_PACKAGES,
        help="Site-packages containing TensorRT Python bindings/libs, appended only when --trt-backbone-engine is used.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("direct", "pipeline"),
        default="direct",
        help="direct skips MMDetection Compose/collate for fixed-size video batches.",
    )
    parser.add_argument("--draw-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-classifier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--roi-classifier-checkpoint", type=Path, default=DEFAULT_ROI_CLASSIFIER_CKPT)
    parser.add_argument("--draw-classifier-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-mask-head", action="store_true", help="BBox-only upper-bound benchmark; disables mask prediction.")
    parser.add_argument("--disable-mask-iou-head", action="store_true", help="Skip mask IoU scoring head when masks are enabled.")
    parser.add_argument("--eval-module", choices=("detr", "one-stage", "two-stage"), default=None)
    parser.add_argument("--encoder-layers", type=int, default=None, help="Speed/accuracy trade-off: keep only the first N Co-DINO encoder layers.")
    parser.add_argument("--decoder-layers", type=int, default=None, help="Speed/accuracy trade-off: keep only the first N Co-DINO decoder layers.")
    parser.add_argument("--mask-alpha", type=float, default=0.45)
    parser.add_argument("--mask-color", default="0,255,0", help="BGR mask color, e.g. 0,255,0.")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--frame-stride", type=int, default=1, help="1 keeps every frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc. mp4v is widely available.")
    parser.add_argument("--no-video", action="store_true", help="Run inference but do not write overlay video.")
    parser.add_argument("--async-writer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--writer-queue-size", type=int, default=16)
    parser.add_argument("--skip-backbone-init", action=argparse.BooleanOptionalAction, default=True,
                        help="Avoid loading DINOv3 pretrain before loading the full detector checkpoint.")
    parser.add_argument(
        "--disable-act-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable training-only activation checkpoint wrappers in the inference config.",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=60, help="Written-frame log interval.")
    parser.add_argument("--profile-modules", action="store_true", help="Print coarse CUDA-synchronized module timings.")
    parser.add_argument(
        "--compile-modules",
        default="",
        help="Comma-separated experimental torch.compile targets: decoder,mask_head,mask_iou_head,mask_roi_extractor.",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    return parser.parse_args()


def parse_bgr_color(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError(f"--mask-color must be three BGR values in 0..255, got: {text}")
    return parts[0], parts[1], parts[2]


def _torch_load_metadata(path: Path) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_roi_classifier(checkpoint: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any], list[str], list[int]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    from backend.classifiers.dinov3_roi.runtime.two_stage_roi_classifier import classifier_from_checkpoint

    raw = _torch_load_metadata(checkpoint)
    classifier, loaded = classifier_from_checkpoint(checkpoint, map_location=device)
    classifier.to(device).eval()
    class_names = [str(x) for x in raw.get("class_names", ["class_0"])]
    class_ids = [int(x) for x in raw.get("class_ids", list(range(1, len(class_names) + 1)))]
    print(f"[classifier] checkpoint={checkpoint}")
    print(
        f"[classifier] model_type={(raw.get('model_cfg') or {}).get('model_type')} "
        f"epoch={loaded.get('epoch')} val_macro_f1={(raw.get('val_metrics') or {}).get('macro_f1')}"
    )
    print(f"[classifier] classes={class_names}")
    return classifier, raw, class_names, class_ids


def load_detector(args: argparse.Namespace):
    from mmcv import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(str(args.config))
    if args.skip_backbone_init and "backbone" in cfg.model:
        cfg.model.backbone.pretrained = False
        cfg.model.backbone.weights = None
    if "fp16" in cfg:
        # We use PyTorch autocast explicitly in this script.
        cfg.pop("fp16")
    if args.disable_act_checkpoint:
        disable_activation_checkpoint_cfg(cfg.model)
    model_score_thr = min(args.score_thr, 0.05) if args.model_score_thr is None else args.model_score_thr
    if model_score_thr > 0:
        set_model_score_thr_cfg(cfg.model, float(model_score_thr))
    cfg.load_from = None
    cfg.resume_from = None

    model = init_detector(cfg, str(args.checkpoint), device=args.device)
    model.eval()
    model.CLASSES = ("foreground",)
    return model


def disable_activation_checkpoint_cfg(node: Any) -> None:
    """Remove training-time checkpoint wrappers before modules are built."""
    if isinstance(node, (list, tuple)):
        for item in node:
            disable_activation_checkpoint_cfg(item)
        return
    if not isinstance(node, dict):
        return
    for key in list(node.keys()):
        value = node[key]
        if key == "with_cp":
            node[key] = False if isinstance(value, bool) else -1
        elif key in {"use_checkpoint", "use_act_checkpoint"}:
            node[key] = False
        else:
            disable_activation_checkpoint_cfg(value)


def set_model_score_thr_cfg(model_cfg: Any, score_thr: float) -> None:
    test_cfg = model_cfg.get("test_cfg", None) if isinstance(model_cfg, dict) else None
    if test_cfg is None:
        return
    cfg_items = test_cfg if isinstance(test_cfg, list) else [test_cfg]
    for item in cfg_items:
        if not isinstance(item, dict):
            continue
        # Co-DINO's query-head test cfg may not define score_thr, but get_bboxes
        # reads it. Setting it here avoids mask inference for invisible boxes.
        if "max_per_img" in item or "nms" in item or "mask_thr_binary" in item:
            item["score_thr"] = float(score_thr)
        rcnn_cfg = item.get("rcnn", None)
        if isinstance(rcnn_cfg, dict):
            rcnn_cfg["score_thr"] = float(score_thr)


def _onnx_input_torch_dtype(path: Path) -> torch.dtype:
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path))
    if not model.graph.input:
        return torch.float32
    elem_type = model.graph.input[0].type.tensor_type.elem_type
    if elem_type == TensorProto.FLOAT16:
        return torch.float16
    if elem_type == TensorProto.FLOAT:
        return torch.float32
    raise TypeError(f"Unsupported ONNX input dtype: {elem_type}")


class CoDINOORTBackboneAdapter(torch.nn.Module):
    """Adapt the shared ORT backbone wrapper to Co-DINO's list feature API."""

    def __init__(self, onnx_path: Path, *, max_batch: int, input_dtype: torch.dtype, device_id: int | None) -> None:
        super().__init__()
        cascade_path = str(CASCADE_INFER_ROOT)
        if cascade_path not in sys.path:
            sys.path.insert(0, cascade_path)
        from onnx_backbone import ORTBackboneAdapter

        self.ort = ORTBackboneAdapter(
            str(onnx_path),
            max_batch=max_batch,
            expect_dtype=input_dtype,
            provider_device_id=device_id,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.ort(x)["last_feat"]]


class CoDINOTRTBackboneAdapter(torch.nn.Module):
    """Adapt the shared native TensorRT backbone wrapper to Co-DINO."""

    def __init__(self, engine_path: Path, *, extra_site_packages: Path | None) -> None:
        super().__init__()
        _prepare_tensorrt_runtime(extra_site_packages)
        cascade_path = str(CASCADE_INFER_ROOT)
        if cascade_path not in sys.path:
            sys.path.insert(0, cascade_path)
        from trt_backbone import TensorRTBackboneAdapter

        self.trt = TensorRTBackboneAdapter(str(engine_path))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.trt(x)["last_feat"]]


class CoDINOTRTFeatureListAdapter(torch.nn.Module):
    """Native TensorRT full feature extractor that replaces backbone+neck."""

    def __init__(self, engine_path: Path, *, feature_names: tuple[str, ...], extra_site_packages: Path | None) -> None:
        super().__init__()
        _prepare_tensorrt_runtime(extra_site_packages)
        cascade_path = str(CASCADE_INFER_ROOT)
        if cascade_path not in sys.path:
            sys.path.insert(0, cascade_path)
        from trt_backbone import TensorRTFeatureDictBackboneAdapter

        self.feature_names = feature_names
        self.trt = TensorRTFeatureDictBackboneAdapter(str(engine_path), feature_names=feature_names)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.trt(x)
        return [outputs[name] for name in self.trt.feature_names]


def _trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.DataType.FLOAT:
        return torch.float32
    if dtype == trt.DataType.HALF:
        return torch.float16
    if dtype == trt.DataType.BF16:
        return torch.bfloat16
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def parse_hw_shapes(text: str) -> tuple[tuple[int, int], ...]:
    shapes: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid HW shape '{item}', expected HxW.")
        h_text, w_text = item.split("x", 1)
        h, w = int(h_text), int(w_text)
        if h < 1 or w < 1:
            raise ValueError(f"Invalid HW shape '{item}'.")
        shapes.append((h, w))
    if not shapes:
        raise ValueError("--trt-query-encoder-shapes must define at least one shape.")
    return tuple(shapes)


class CoDINOTRTQueryEncoderAdapter(torch.nn.Module):
    """Replace Co-DINO's transformer encoder with a fixed-shape TensorRT engine."""

    def __init__(
        self,
        engine_path: Path,
        *,
        feature_shapes: tuple[tuple[int, int], ...],
        extra_site_packages: Path | None,
    ) -> None:
        super().__init__()
        _prepare_tensorrt_runtime(extra_site_packages)
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")

        self.feature_shapes = tuple(feature_shapes)
        self.engine_path = str(engine_path.expanduser().resolve())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(self.engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT query encoder execution context")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = tuple(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        outputs = tuple(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        self.uses_query_input = len(self.input_names) == 1
        if not self.uses_query_input and len(self.input_names) != len(self.feature_shapes):
            raise RuntimeError(
                f"Expected one flattened query input or {len(self.feature_shapes)} feature inputs, got {self.input_names}"
            )
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one query encoder output, got {outputs}")
        self.output_name = outputs[0]
        self.input_dtypes = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name)) for name in self.input_names
        }
        self.output_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self.output_name))
        self._output_tensor: torch.Tensor | None = None
        self._output_shape: tuple[int, ...] | None = None
        self._last_inputs: list[torch.Tensor] = []
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_QUERY_ENCODER_DEDICATED_STREAM", "0") in ("1", "true", "True"):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_output(self, device: torch.device) -> torch.Tensor:
        out_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        if self._output_tensor is None or self._output_shape != out_shape or self._output_tensor.dtype != self.output_dtype:
            self._output_tensor = torch.empty(out_shape, device=device, dtype=self.output_dtype)
            self._output_shape = out_shape
        return self._output_tensor

    def forward(self, query: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not query.is_cuda:
            raise RuntimeError("TensorRT query encoder requires CUDA tensors")
        total_tokens, batch_size, channels = query.shape
        expected_tokens = sum(h * w for h, w in self.feature_shapes)
        if total_tokens != expected_tokens:
            raise RuntimeError(f"Query token count mismatch: expected {expected_tokens}, got {total_tokens}")

        inputs: list[torch.Tensor] = []
        if self.uses_query_input:
            name = self.input_names[0]
            feat = query.contiguous()
            target_dtype = self.input_dtypes[name]
            if feat.dtype != target_dtype:
                feat = feat.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(feat.shape))
            except Exception:
                pass
            inputs.append(feat)
        else:
            by_channel = query.permute(1, 2, 0).contiguous()
            start = 0
            for name, (height, width) in zip(self.input_names, self.feature_shapes):
                end = start + height * width
                feat = by_channel[:, :, start:end].reshape(batch_size, channels, height, width).contiguous()
                target_dtype = self.input_dtypes[name]
                if feat.dtype != target_dtype:
                    feat = feat.to(target_dtype)
                try:
                    self.context.set_input_shape(name, tuple(feat.shape))
                except Exception:
                    pass
                inputs.append(feat)
                start = end

        output = self._ensure_output(query.device)
        for name, feat in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(feat.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output.data_ptr()))

        current_stream = torch.cuda.current_stream(device=query.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for feat in inputs:
                feat.record_stream(self.trt_stream)
            output.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for feat in inputs:
                feat.record_stream(current_stream)
            output.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT query encoder execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        if tuple(output.shape) == tuple(query.shape):
            return output
        if output.ndim == 3 and output.shape[0] == batch_size and output.shape[1] == total_tokens:
            return output.permute(1, 0, 2).contiguous()
        raise RuntimeError(f"Unexpected TensorRT query encoder output shape: {tuple(output.shape)}")


class CoDINOTRTDecoderAdapter(torch.nn.Module):
    """Replace Co-DINO's fixed-shape transformer decoder with TensorRT."""

    def __init__(self, engine_path: Path, *, extra_site_packages: Path | None) -> None:
        super().__init__()
        _prepare_tensorrt_runtime(extra_site_packages)
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")
        self.engine_path = str(engine_path.expanduser().resolve())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(self.engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT decoder engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT decoder execution context")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = tuple(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names = tuple(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        expected_inputs = ("query", "memory", "reference_points")
        expected_outputs = ("inter_states", "inter_references")
        if self.input_names != expected_inputs or self.output_names != expected_outputs:
            raise RuntimeError(
                f"Unexpected decoder engine IO: inputs={self.input_names}, outputs={self.output_names}"
            )
        self.input_dtypes = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name)) for name in self.input_names
        }
        self.output_dtypes = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name)) for name in self.output_names
        }
        self.num_layers = int(self.engine.get_tensor_shape("inter_states")[0])
        self._output_tensors: dict[str, torch.Tensor] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._last_inputs: list[torch.Tensor] = []
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_DECODER_DEDICATED_STREAM", "0") in ("1", "true", "True"):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_outputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            out_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            out_dtype = self.output_dtypes[name]
            tensor = self._output_tensors.get(name)
            if tensor is None or self._output_shapes.get(name) != out_shape or tensor.dtype != out_dtype:
                tensor = torch.empty(out_shape, device=device, dtype=out_dtype)
                self._output_tensors[name] = tensor
                self._output_shapes[name] = out_shape
            outputs[name] = tensor
        return outputs

    def forward(self, query: torch.Tensor, *args, reference_points=None, **kwargs):
        memory = kwargs.get("value", None)
        if memory is None:
            raise RuntimeError("TensorRT decoder requires value=memory")
        if reference_points is None:
            raise RuntimeError("TensorRT decoder requires reference_points")
        if not query.is_cuda or not memory.is_cuda or not reference_points.is_cuda:
            raise RuntimeError("TensorRT decoder requires CUDA tensors")

        raw_inputs = {
            "query": query.contiguous(),
            "memory": memory.contiguous(),
            "reference_points": reference_points.contiguous(),
        }
        inputs: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = raw_inputs[name]
            target_dtype = self.input_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                pass
            inputs.append(tensor)

        outputs = self._ensure_outputs(query.device)
        for name, tensor in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        current_stream = torch.cuda.current_stream(device=query.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for tensor in inputs:
                tensor.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for tensor in inputs:
                tensor.record_stream(current_stream)
            for tensor in outputs.values():
                tensor.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT decoder execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return outputs["inter_states"], outputs["inter_references"]


def _prepare_tensorrt_runtime(extra_site_packages: Path | None) -> None:
    if extra_site_packages is not None and extra_site_packages.exists():
        extra_site = str(extra_site_packages)
        if extra_site not in sys.path:
            sys.path.append(extra_site)
        trt_libs = extra_site_packages / "tensorrt_libs"
        plugin = trt_libs / "libnvinfer_plugin.so.10"
        if plugin.exists():
            ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
            vc_plugin = trt_libs / "libnvinfer_vc_plugin.so.10"
            if not vc_plugin.exists():
                shim_dir = Path("/tmp/trt_vc_plugin_shim")
                shim_dir.mkdir(parents=True, exist_ok=True)
                shim = shim_dir / "libnvinfer_vc_plugin.so.10"
                if not shim.exists():
                    shim.symlink_to(plugin)
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{shim_dir}:{trt_libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                )
                ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)


class CoDINOTRTMaskHeadAdapter(torch.nn.Module):
    """Replace Co-DINO's fixed-shape SimpleRefineMaskHead with TensorRT."""

    def __init__(
        self,
        original_mask_head: torch.nn.Module,
        engine_path: Path,
        *,
        extra_site_packages: Path | None,
    ) -> None:
        super().__init__()
        _prepare_tensorrt_runtime(extra_site_packages)
        import tensorrt as trt

        trt.init_libnvinfer_plugins(None, "")
        self.original_mask_head = original_mask_head
        self.stage_num_classes = original_mask_head.stage_num_classes
        self.pre_upsample_last_stage = original_mask_head.pre_upsample_last_stage
        if any(int(num_classes) != 1 for num_classes in self.stage_num_classes):
            raise RuntimeError("TensorRT mask head adapter currently supports class-agnostic/1-class heads only.")

        self.engine_path = str(engine_path.expanduser().resolve())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(self.engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT mask head engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT mask head execution context")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = tuple(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names = tuple(
            name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        expected_inputs = ("instance_feats", "semantic_feat", "rois")
        expected_core_inputs = ("instance_feats", "sem0", "sem1", "sem2")
        expected_outputs = ("stage0", "stage1", "stage2", "stage3")
        if self.input_names == expected_inputs:
            self.mode = "full"
        elif self.input_names == expected_core_inputs:
            self.mode = "core"
        else:
            raise RuntimeError(
                f"Unexpected mask head engine IO: inputs={self.input_names}, outputs={self.output_names}"
            )
        if self.output_names != expected_outputs:
            raise RuntimeError(f"Unexpected mask head engine outputs: {self.output_names}")

        self.input_dtypes = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name)) for name in self.input_names
        }
        self.output_dtypes = {
            name: _trt_dtype_to_torch(self.engine.get_tensor_dtype(name)) for name in self.output_names
        }
        self.engine_num_rois = int(self.engine.get_tensor_shape("instance_feats")[0])
        self.engine_semantic_batch = (
            int(self.engine.get_tensor_shape("semantic_feat")[0]) if self.mode == "full" else 0
        )
        self._output_tensors: dict[str, torch.Tensor] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._last_inputs: list[torch.Tensor] = []
        self.trt_stream: torch.cuda.Stream | None = None
        if os.environ.get("CODINO_TRT_MASK_HEAD_DEDICATED_STREAM", "0") in ("1", "true", "True"):
            self.trt_stream = torch.cuda.Stream()

    def _ensure_outputs(self, device: torch.device) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name in self.output_names:
            out_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            out_dtype = self.output_dtypes[name]
            tensor = self._output_tensors.get(name)
            if tensor is None or self._output_shapes.get(name) != out_shape or tensor.dtype != out_dtype:
                tensor = torch.empty(out_shape, device=device, dtype=out_dtype)
                self._output_tensors[name] = tensor
                self._output_shapes[name] = out_shape
            outputs[name] = tensor
        return outputs

    def _pad_inputs(
        self,
        instance_feats: torch.Tensor,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        num_rois = int(rois.shape[0])
        if num_rois > self.engine_num_rois:
            raise RuntimeError("Internal error: _pad_inputs received too many RoIs for the TensorRT mask engine.")

        padded_instance_feats = instance_feats.contiguous()
        padded_rois = rois.contiguous()
        if num_rois < self.engine_num_rois:
            feat_pad = torch.zeros(
                (self.engine_num_rois - num_rois, *instance_feats.shape[1:]),
                device=instance_feats.device,
                dtype=instance_feats.dtype,
            )
            roi_pad = torch.zeros(
                (self.engine_num_rois - num_rois, rois.shape[1]),
                device=rois.device,
                dtype=rois.dtype,
            )
            padded_instance_feats = torch.cat((padded_instance_feats, feat_pad), dim=0).contiguous()
            padded_rois = torch.cat((padded_rois, roi_pad), dim=0).contiguous()

        semantic_batch = int(semantic_feat.shape[0])
        if semantic_batch > self.engine_semantic_batch:
            raise RuntimeError(
                f"TensorRT mask head engine expects semantic batch <= {self.engine_semantic_batch}, "
                f"got {semantic_batch}."
            )
        padded_semantic_feat = semantic_feat.contiguous()
        if semantic_batch < self.engine_semantic_batch:
            sem_pad = torch.zeros(
                (self.engine_semantic_batch - semantic_batch, *semantic_feat.shape[1:]),
                device=semantic_feat.device,
                dtype=semantic_feat.dtype,
            )
            padded_semantic_feat = torch.cat((padded_semantic_feat, sem_pad), dim=0).contiguous()
        return padded_instance_feats, padded_semantic_feat, padded_rois, num_rois

    def _execute_fixed(
        self,
        instance_feats: torch.Tensor,
        semantic_feat: torch.Tensor,
        rois: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        instance_feats, semantic_feat, rois, num_rois = self._pad_inputs(instance_feats, semantic_feat, rois)
        raw_inputs = {
            "instance_feats": instance_feats,
            "semantic_feat": semantic_feat,
            "rois": rois,
        }
        inputs: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = raw_inputs[name]
            target_dtype = self.input_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                pass
            inputs.append(tensor.contiguous())

        outputs = self._ensure_outputs(instance_feats.device)
        for name, tensor in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        current_stream = torch.cuda.current_stream(device=instance_feats.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for tensor in inputs:
                tensor.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for tensor in inputs:
                tensor.record_stream(current_stream)
            for tensor in outputs.values():
                tensor.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT mask head execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return tuple(outputs[name][:num_rois] for name in self.output_names)

    def _compute_core_semantic_rois(self, semantic_feat: torch.Tensor, rois: torch.Tensor) -> tuple[torch.Tensor, ...]:
        for conv in self.original_mask_head.semantic_convs:
            semantic_feat = conv(semantic_feat)
        roi_feats = []
        rois_fp32 = rois.float() if rois.dtype != torch.float32 else rois
        for stage in self.original_mask_head.stages:
            transformed = stage.relu(stage.semantic_transform_in(semantic_feat))
            transformed_fp32 = transformed.float() if transformed.dtype != torch.float32 else transformed
            roi_feats.append(stage.semantic_roi_extractor([transformed_fp32], rois_fp32))
        return tuple(roi_feats)

    def _pad_core_inputs(
        self,
        instance_feats: torch.Tensor,
        semantic_rois: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
        num_rois = int(instance_feats.shape[0])
        if num_rois > self.engine_num_rois:
            raise RuntimeError("Internal error: _pad_core_inputs received too many RoIs for the TensorRT mask engine.")
        padded_instance_feats = instance_feats.contiguous()
        padded_semantic_rois = [feat.contiguous() for feat in semantic_rois]
        if num_rois < self.engine_num_rois:
            feat_pad = torch.zeros(
                (self.engine_num_rois - num_rois, *instance_feats.shape[1:]),
                device=instance_feats.device,
                dtype=instance_feats.dtype,
            )
            padded_instance_feats = torch.cat((padded_instance_feats, feat_pad), dim=0).contiguous()
            padded = []
            for feat in padded_semantic_rois:
                sem_pad = torch.zeros(
                    (self.engine_num_rois - num_rois, *feat.shape[1:]),
                    device=feat.device,
                    dtype=feat.dtype,
                )
                padded.append(torch.cat((feat, sem_pad), dim=0).contiguous())
            padded_semantic_rois = padded
        return padded_instance_feats, tuple(padded_semantic_rois), num_rois

    def _execute_core_fixed(
        self,
        instance_feats: torch.Tensor,
        semantic_rois: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        instance_feats, semantic_rois, num_rois = self._pad_core_inputs(instance_feats, semantic_rois)
        raw_inputs = {
            "instance_feats": instance_feats,
            "sem0": semantic_rois[0],
            "sem1": semantic_rois[1],
            "sem2": semantic_rois[2],
        }
        inputs: list[torch.Tensor] = []
        for name in self.input_names:
            tensor = raw_inputs[name]
            target_dtype = self.input_dtypes[name]
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            try:
                self.context.set_input_shape(name, tuple(tensor.shape))
            except Exception:
                pass
            inputs.append(tensor.contiguous())

        outputs = self._ensure_outputs(instance_feats.device)
        for name, tensor in zip(self.input_names, inputs):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        current_stream = torch.cuda.current_stream(device=instance_feats.device)
        if self.trt_stream is not None:
            self.trt_stream.wait_stream(current_stream)
            for tensor in inputs:
                tensor.record_stream(self.trt_stream)
            for tensor in outputs.values():
                tensor.record_stream(self.trt_stream)
            stream = self.trt_stream.cuda_stream
        else:
            for tensor in inputs:
                tensor.record_stream(current_stream)
            for tensor in outputs.values():
                tensor.record_stream(current_stream)
            stream = current_stream.cuda_stream
        self._last_inputs = inputs
        ok = self.context.execute_async_v3(stream_handle=int(stream))
        if not ok:
            raise RuntimeError("TensorRT mask head execute_async_v3 failed")
        if self.trt_stream is not None:
            current_stream.wait_stream(self.trt_stream)
        return tuple(outputs[name][:num_rois] for name in self.output_names)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        if not instance_feats.is_cuda or not semantic_feat.is_cuda or not rois.is_cuda:
            raise RuntimeError("TensorRT mask head requires CUDA tensors")
        num_rois = int(rois.shape[0])
        if num_rois == 0:
            return self.original_mask_head(instance_feats, semantic_feat, rois, roi_labels)
        if self.mode == "core":
            semantic_rois = self._compute_core_semantic_rois(semantic_feat, rois)
            if num_rois <= self.engine_num_rois:
                return list(self._execute_core_fixed(instance_feats, semantic_rois)), []

            stage_chunks = [[] for _ in self.output_names]
            for start in range(0, num_rois, self.engine_num_rois):
                end = min(start + self.engine_num_rois, num_rois)
                chunk_outputs = self._execute_core_fixed(
                    instance_feats[start:end],
                    tuple(feat[start:end] for feat in semantic_rois),
                )
                for idx, tensor in enumerate(chunk_outputs):
                    stage_chunks[idx].append(tensor.clone())
            return [torch.cat(chunks, dim=0) for chunks in stage_chunks], []

        if num_rois <= self.engine_num_rois:
            return list(self._execute_fixed(instance_feats, semantic_feat, rois)), []

        stage_chunks = [[] for _ in self.output_names]
        for start in range(0, num_rois, self.engine_num_rois):
            end = min(start + self.engine_num_rois, num_rois)
            chunk_outputs = self._execute_fixed(
                instance_feats[start:end],
                semantic_feat,
                rois[start:end],
            )
            for idx, tensor in enumerate(chunk_outputs):
                stage_chunks[idx].append(tensor.clone())
        return [torch.cat(chunks, dim=0) for chunks in stage_chunks], []

    def get_seg_masks(self, *args, **kwargs):
        return self.original_mask_head.get_seg_masks(*args, **kwargs)


def install_onnx_backbone(model: torch.nn.Module, onnx_path: Path, max_batch: int) -> None:
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    device = next(model.parameters()).device
    device_id = None
    if device.type == "cuda":
        device_id = torch.cuda.current_device() if device.index is None else int(device.index)
    input_dtype = _onnx_input_torch_dtype(onnx_path)
    model.backbone = CoDINOORTBackboneAdapter(
        onnx_path,
        max_batch=max_batch,
        input_dtype=input_dtype,
        device_id=device_id,
    )
    model.backbone.eval()
    print(f"[model] using ONNX Runtime backbone={onnx_path} input_dtype={input_dtype} max_batch={max_batch}")


def install_trt_backbone(model: torch.nn.Module, engine_path: Path, extra_site_packages: Path | None) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    model.backbone = CoDINOTRTBackboneAdapter(engine_path, extra_site_packages=extra_site_packages)
    model.backbone.eval()
    print(f"[model] using native TensorRT backbone={engine_path}")


def install_trt_feature_extractor(
    model: torch.nn.Module,
    engine_path: Path,
    feature_names: tuple[str, ...],
    extra_site_packages: Path | None,
) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    model.backbone = CoDINOTRTFeatureListAdapter(
        engine_path,
        feature_names=feature_names,
        extra_site_packages=extra_site_packages,
    )
    model.neck = None
    model.backbone.eval()
    print(f"[model] using native TensorRT feature extractor={engine_path} features={feature_names}")


def install_trt_query_encoder(
    model: torch.nn.Module,
    engine_path: Path,
    feature_shapes: tuple[tuple[int, int], ...],
    extra_site_packages: Path | None,
) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    transformer = getattr(getattr(model, "query_head", None), "transformer", None)
    if transformer is None:
        raise RuntimeError("Model has no query_head.transformer to patch.")
    transformer.encoder = CoDINOTRTQueryEncoderAdapter(
        engine_path,
        feature_shapes=feature_shapes,
        extra_site_packages=extra_site_packages,
    )
    transformer.encoder.eval()
    print(f"[model] using native TensorRT query encoder={engine_path} shapes={feature_shapes}")


def install_trt_decoder(
    model: torch.nn.Module,
    engine_path: Path,
    extra_site_packages: Path | None,
) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    transformer = getattr(getattr(model, "query_head", None), "transformer", None)
    if transformer is None:
        raise RuntimeError("Model has no query_head.transformer to patch.")
    transformer.decoder = CoDINOTRTDecoderAdapter(
        engine_path,
        extra_site_packages=extra_site_packages,
    )
    transformer.decoder.eval()
    print(f"[model] using native TensorRT decoder={engine_path}")


def install_trt_mask_head(
    model: torch.nn.Module,
    engine_path: Path,
    extra_site_packages: Path | None,
) -> None:
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    model.mask_head = CoDINOTRTMaskHeadAdapter(
        model.mask_head,
        engine_path,
        extra_site_packages=extra_site_packages,
    )
    model.mask_head.eval()
    print(f"[model] using native TensorRT mask head={engine_path}")


def apply_compile_options(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if not args.compile_modules:
        return
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")
    targets = {part.strip() for part in args.compile_modules.split(",") if part.strip()}
    valid = {"decoder", "mask_head", "mask_iou_head", "mask_roi_extractor"}
    unknown = targets - valid
    if unknown:
        raise ValueError(f"Unknown --compile-modules entries: {sorted(unknown)}")

    def compile_attr(parent: torch.nn.Module, name: str) -> None:
        if not hasattr(parent, name):
            print(f"[compile] skipped missing {name}")
            return
        module = getattr(parent, name)
        compiled = torch.compile(module, mode=args.compile_mode, fullgraph=False, dynamic=False)
        setattr(parent, name, compiled)
        print(f"[compile] {name} mode={args.compile_mode}")

    transformer = getattr(getattr(model, "query_head", None), "transformer", None)
    if "decoder" in targets:
        if transformer is None or not hasattr(transformer, "decoder"):
            print("[compile] skipped missing transformer.decoder")
        else:
            transformer.decoder = torch.compile(
                transformer.decoder,
                mode=args.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            print(f"[compile] transformer.decoder mode={args.compile_mode}")
    for name in ("mask_head", "mask_iou_head", "mask_roi_extractor"):
        if name in targets:
            compile_attr(model, name)


def apply_runtime_model_options(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if args.eval_module is not None and hasattr(model, "eval_module"):
        model.eval_module = args.eval_module
        print(f"[model] eval_module={args.eval_module}")
    if args.disable_mask_iou_head and hasattr(model, "mask_iou_head"):
        delattr(model, "mask_iou_head")
        print("[model] disabled mask_iou_head")
    if args.disable_mask_head:
        for name in ("mask_head", "mask_roi_extractor", "mask_iou_head"):
            if hasattr(model, name):
                delattr(model, name)
        print("[model] disabled mask head path")
    transformer = getattr(getattr(model, "query_head", None), "transformer", None)
    if transformer is not None and args.encoder_layers is not None:
        truncate_transformer_layers(transformer.encoder, args.encoder_layers, "encoder")
    if transformer is not None and args.decoder_layers is not None:
        truncate_transformer_layers(transformer.decoder, args.decoder_layers, "decoder")


def truncate_transformer_layers(module: torch.nn.Module, keep: int, name: str) -> None:
    layers = getattr(module, "layers", None)
    if layers is None:
        raise RuntimeError(f"{name} has no layers attribute")
    original = len(layers)
    if keep < 1 or keep > original:
        raise ValueError(f"--{name}-layers must be in 1..{original}, got {keep}")
    if keep == original:
        return
    module.layers = torch.nn.ModuleList(list(layers[:keep]))
    if hasattr(module, "num_layers"):
        module.num_layers = keep
    print(f"[model] truncated {name} layers: {original} -> {keep}")


def build_test_pipeline(model):
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose

    cfg = model.cfg.copy()
    cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return Compose(cfg.data.test.pipeline)


def find_letterbox_size(model) -> tuple[int, int]:
    cfg = model.cfg
    if hasattr(cfg, "letterbox_size"):
        return int(cfg.letterbox_size[0]), int(cfg.letterbox_size[1])
    pipeline = cfg.data.test.pipeline
    for step in pipeline:
        for transform in step.get("transforms", []):
            if transform.get("type") == "UnifiedLetterboxResize":
                h, w = transform["target_size"]
                return int(h), int(w)
    raise RuntimeError("Could not find UnifiedLetterboxResize target_size in cfg.data.test.pipeline.")


def prepare_batch(model, pipeline, frames: list[np.ndarray]):
    from mmcv.ops import RoIPool
    from mmcv.parallel import collate, scatter

    datas = []
    for frame in frames:
        datas.append(pipeline(dict(img=frame)))

    data = collate(datas, samples_per_gpu=len(frames))
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]

    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        for module in model.modules():
            assert not isinstance(module, RoIPool), "CPU inference with RoIPool is not supported."
    return data


def _ceil_to_multiple(value: int, divisor: int) -> int:
    return int((value + divisor - 1) // divisor * divisor)


def prepare_batch_direct(
    model,
    frames: list[np.ndarray],
    target_size: tuple[int, int],
) -> dict[str, list[Any]]:
    """Build MMDetection test inputs directly for the fixed video pipeline.

    Equivalent to:
      LoadImageFromWebcam -> UnifiedLetterboxResize(720x1280,pad=128)
      -> Normalize(to_rgb=True) -> Pad(size_divisor=32) -> ImageToTensor.
    """
    target_h, target_w = target_size
    pad_h = _ceil_to_multiple(target_h, 32)
    pad_w = _ceil_to_multiple(target_w, 32)
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    batch = np.zeros((len(frames), pad_h, pad_w, 3), dtype=np.float32)
    metas: list[dict[str, Any]] = []

    for idx, frame in enumerate(frames):
        orig_h, orig_w = frame.shape[:2]
        scale, new_h, new_w, pad_top, pad_left = letterbox_params(orig_h, orig_w, target_h, target_w)
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # MMDetection Normalize receives BGR input and converts to RGB when to_rgb=True.
        rgb = canvas[:, :, [2, 1, 0]].astype(np.float32)
        batch[idx, :target_h, :target_w, :] = (rgb - mean) / std
        scale_factor = np.asarray([scale, scale, scale, scale], dtype=np.float32)
        metas.append({
            "filename": None,
            "ori_filename": None,
            "ori_shape": (orig_h, orig_w, 3),
            "img_shape": (target_h, target_w, 3),
            "pad_shape": (pad_h, pad_w, 3),
            "scale_factor": scale_factor,
            "flip": False,
            "flip_direction": None,
            "img_norm_cfg": dict(
                mean=np.asarray([123.675, 116.28, 103.53], dtype=np.float32),
                std=np.asarray([58.395, 57.12, 57.375], dtype=np.float32),
                to_rgb=True,
            ),
        })

    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        tensor = tensor.to(device, non_blocking=True)
    return {"img": [tensor], "img_metas": [metas]}


def infer_batch(
    model,
    pipeline,
    frames: list[np.ndarray],
    amp: str,
    *,
    preprocess: str,
    target_size: tuple[int, int],
):
    if preprocess == "direct":
        data = prepare_batch_direct(model, frames, target_size)
    else:
        data = prepare_batch(model, pipeline, frames)
    use_cuda = next(model.parameters()).is_cuda
    with torch.inference_mode():
        if amp == "off" or not use_cuda:
            return model(return_loss=False, rescale=True, **data)
        if amp == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("--amp bf16 requested, but this GPU does not support bf16.")
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            return model(return_loss=False, rescale=True, **data)


def _slice_features_for_image(x, img_idx: int, batch_size: int):
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor) and x[0].shape[0] == batch_size:
        return [
            feat[img_idx : img_idx + 1]
            if isinstance(feat, torch.Tensor) and feat.shape[0] == batch_size
            else feat
            for feat in x
        ]
    return x


def _scale_factor_tensor(img_meta: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    scale_factor = img_meta["scale_factor"]
    if isinstance(scale_factor, torch.Tensor):
        return scale_factor.to(device=device, dtype=dtype)
    return torch.as_tensor(scale_factor, device=device, dtype=dtype)


def _paste_bboxes_for_masks(det_bboxes: torch.Tensor, img_meta: dict[str, Any], scale_factor: torch.Tensor) -> torch.Tensor:
    scaled = det_bboxes[:, :4] * scale_factor
    paste_bboxes = scaled
    img_shape = img_meta.get("img_shape", img_meta.get("pad_shape", img_meta["ori_shape"]))
    input_h, input_w = int(img_shape[0]), int(img_shape[1])
    orig_h, orig_w = int(img_meta["ori_shape"][0]), int(img_meta["ori_shape"][1])
    scale_x = float(scale_factor[0].detach().cpu().item())
    scale_y = float(scale_factor[1].detach().cpu().item())
    new_w = int(orig_w * scale_x)
    new_h = int(orig_h * scale_y)
    pad_left = max((input_w - new_w) // 2, 0)
    pad_top = max((input_h - new_h) // 2, 0)
    if pad_left or pad_top:
        paste_bboxes = scaled.clone()
        paste_bboxes[:, 0::2] -= pad_left
        paste_bboxes[:, 1::2] -= pad_top
        paste_bboxes[:, 0::2].clamp_(min=0, max=max(new_w, 1))
        paste_bboxes[:, 1::2].clamp_(min=0, max=max(new_h, 1))
    return paste_bboxes


def _refine_stage_instance_preds(stage_instance_preds: list[torch.Tensor]) -> torch.Tensor:
    from mmdet.models.losses.cross_entropy_loss import generate_block_target

    stage_instance_preds = list(stage_instance_preds[1:])
    for idx in range(len(stage_instance_preds) - 1):
        instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
        non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
        non_boundary_mask = (
            F.interpolate(
                non_boundary_mask.float(),
                stage_instance_preds[idx + 1].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            >= 0.5
        )
        pre_pred = F.interpolate(
            stage_instance_preds[idx],
            stage_instance_preds[idx + 1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        if pre_pred.dtype != stage_instance_preds[idx + 1].dtype:
            pre_pred = pre_pred.to(stage_instance_preds[idx + 1].dtype)
        stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
    return stage_instance_preds[-1]


def _unletterbox_boxes_array_for_meta(
    boxes: np.ndarray,
    img_meta: dict[str, Any],
    target_size: tuple[int, int],
) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    orig_h, orig_w = int(img_meta["ori_shape"][0]), int(img_meta["ori_shape"][1])
    target_h, target_w = target_size
    scale, _, _, pad_top, pad_left = letterbox_params(orig_h, orig_w, target_h, target_w)
    out = boxes.astype(np.float32, copy=True)
    dx = pad_left / scale if scale > 0 else 0.0
    dy = pad_top / scale if scale > 0 else 0.0
    out[:, [0, 2]] -= dx
    out[:, [1, 3]] -= dy
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, max(orig_w - 1, 0))
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, max(orig_h - 1, 0))
    return out


def _build_classifier_meta(
    det_bboxes: torch.Tensor,
    masks: list[Any],
    img_meta: dict[str, Any],
    target_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    n = int(det_bboxes.shape[0])
    if n == 0:
        return torch.empty((0, 5), device=device, dtype=torch.float32)
    scores = det_bboxes[:, 4].detach().float().cpu().numpy()
    raw_boxes = det_bboxes[:, :4].detach().float().cpu().numpy()
    boxes = _unletterbox_boxes_array_for_meta(raw_boxes, img_meta, target_size)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    bbox_area = widths * heights
    img_area = float(max(1, int(img_meta["ori_shape"][0]) * int(img_meta["ori_shape"][1])))
    mask_area = np.zeros((n,), dtype=np.float32)
    for idx, mask in enumerate(masks[:n]):
        mask_area[idx] = float(np.asarray(mask).astype(bool).sum())
    bbox_area_safe = np.maximum(bbox_area, 1e-6)
    meta = np.stack(
        [
            scores.astype(np.float32),
            (bbox_area / img_area).astype(np.float32),
            (mask_area / img_area).astype(np.float32),
            np.log((widths + 1e-6) / (heights + 1e-6)).astype(np.float32),
            (mask_area / bbox_area_safe).astype(np.float32),
        ],
        axis=1,
    )
    return torch.from_numpy(meta).to(device=device, dtype=torch.float32)


def infer_batch_with_roi_classifier(
    model,
    pipeline,
    frames: list[np.ndarray],
    amp: str,
    *,
    preprocess: str,
    target_size: tuple[int, int],
    classifier: torch.nn.Module,
    num_classifier_classes: int,
):
    from mmdet.core import bbox2result, bbox2roi

    if not hasattr(model, "mask_head"):
        raise RuntimeError("--roi-classifier requires the Co-DINO mask head.")
    if preprocess == "direct":
        data = prepare_batch_direct(model, frames, target_size)
    else:
        data = prepare_batch(model, pipeline, frames)

    img = data["img"][0]
    img_metas = data["img_metas"][0]
    batch_input_shape = tuple(img[0].size()[-2:])
    for img_meta in img_metas:
        img_meta["batch_input_shape"] = batch_input_shape
    if hasattr(model, "with_attn_mask") and not model.with_attn_mask:
        for img_meta in img_metas:
            input_img_h, input_img_w = img_meta["batch_input_shape"]
            img_meta["img_shape"] = [input_img_h, input_img_w, 3]

    use_cuda = next(model.parameters()).is_cuda
    if amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("--amp bf16 requested, but this GPU does not support bf16.")
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    autocast_enabled = amp != "off" and use_cuda

    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=autocast_enabled):
            x = model.extract_feat(img, img_metas)
            results_list, x = model.query_head.simple_test(
                x,
                img_metas,
                rescale=True,
                return_encoder_output=True,
            )

            detector_classes = int(getattr(model.query_head, "num_classes", 1))
            outputs = []
            for img_idx, ((det_bboxes_i, det_labels_i), img_meta) in enumerate(zip(results_list, img_metas)):
                if det_bboxes_i.ndim == 1:
                    det_bboxes_i = det_bboxes_i.unsqueeze(0)
                if det_labels_i.ndim == 0:
                    det_labels_i = det_labels_i.unsqueeze(0)

                n = int(det_bboxes_i.shape[0])
                segm_result = [[] for _ in range(model.mask_head.stage_num_classes[0])]
                if n == 0:
                    outputs.append((bbox2result(det_bboxes_i, det_labels_i, detector_classes), segm_result))
                    continue

                x_i = _slice_features_for_image(x, img_idx, len(img_metas))
                scale_factor = _scale_factor_tensor(img_meta, det_bboxes_i.device, det_bboxes_i.dtype)
                scaled_bboxes = det_bboxes_i[:, :4] * scale_factor
                paste_bboxes = _paste_bboxes_for_masks(det_bboxes_i, img_meta, scale_factor)
                mask_rois = bbox2roi([scaled_bboxes])
                feat_dtype = x_i[0].dtype if isinstance(x_i, (list, tuple)) and x_i else mask_rois.dtype
                if mask_rois.dtype != feat_dtype:
                    mask_rois = mask_rois.to(dtype=feat_dtype)

                extra_cols = det_bboxes_i.new_zeros((n, 2 + int(num_classifier_classes)))
                interval = 150
                for start in range(0, n, interval):
                    end = min(start + interval, n)
                    mask_results = model._mask_forward(x_i, mask_rois[start:end], det_labels_i[start:end])
                    stage_instance_preds = mask_results["stage_instance_preds"]
                    instance_pred = _refine_stage_instance_preds(stage_instance_preds)
                    chunk_masks = model.mask_head.get_seg_masks(
                        instance_pred,
                        paste_bboxes[start:end],
                        det_labels_i[start:end],
                        model.rcnn_test_cfg,
                        img_meta["ori_shape"],
                        scale_factor,
                        True,
                    )

                    for cls_id, segm in zip(det_labels_i[start:end], chunk_masks):
                        segm_result[int(cls_id)].append(segm)

                    mask_feats = mask_results["mask_feats"]
                    meta = _build_classifier_meta(
                        det_bboxes_i[start:end],
                        list(chunk_masks),
                        img_meta,
                        target_size,
                        det_bboxes_i.device,
                    )
                    logits = classifier(mask_feats, meta)
                    probs = torch.softmax(logits.float(), dim=1)
                    cls_scores, cls_ids = torch.max(probs, dim=1)
                    extra_cols[start:end, 0] = cls_ids.to(dtype=extra_cols.dtype)
                    extra_cols[start:end, 1] = cls_scores.to(dtype=extra_cols.dtype)
                    extra_cols[start:end, 2 : 2 + int(num_classifier_classes)] = probs.to(dtype=extra_cols.dtype)

                det_bboxes_out = torch.cat([det_bboxes_i, extra_cols], dim=1)
                outputs.append((bbox2result(det_bboxes_out, det_labels_i, detector_classes), segm_result))
            return outputs


def letterbox_params(orig_h: int, orig_w: int, target_h: int, target_w: int) -> tuple[float, int, int, int, int]:
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    return scale, new_h, new_w, pad_top, pad_left


def normalize_result(result: Any):
    if isinstance(result, dict) and "ins_results" in result:
        result = result["ins_results"]
    if isinstance(result, tuple) and len(result) == 2:
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple) and len(segm_result) == 2:
            segm_result = segm_result[0]
        return bbox_result, segm_result
    return result, None


def unletterbox_bboxes(
    bbox_result: list[np.ndarray],
    orig_shape: tuple[int, int],
    target_size: tuple[int, int],
) -> list[np.ndarray]:
    orig_h, orig_w = orig_shape
    target_h, target_w = target_size
    scale, _, _, pad_top, pad_left = letterbox_params(orig_h, orig_w, target_h, target_w)
    dx = pad_left / scale if scale > 0 else 0.0
    dy = pad_top / scale if scale > 0 else 0.0

    fixed = []
    for bboxes in bbox_result:
        arr = bboxes.copy()
        if arr.size:
            arr[:, [0, 2]] -= dx
            arr[:, [1, 3]] -= dy
            arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0, orig_w - 1)
            arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0, orig_h - 1)
        fixed.append(arr)
    return fixed


def unletterbox_mask(mask: Any, orig_shape: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    from pycocotools import mask as mask_utils

    orig_h, orig_w = orig_shape
    target_h, target_w = target_size
    if isinstance(mask, dict) and "counts" in mask:
        mask = mask_utils.decode(mask)
    mask_u8 = np.asarray(mask).astype(np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    if mask_u8.shape[:2] == (orig_h, orig_w):
        return mask_u8.astype(bool)
    if mask_u8.shape[:2] == (target_h, target_w):
        _, new_h, new_w, pad_top, pad_left = letterbox_params(orig_h, orig_w, target_h, target_w)
        mask_u8 = mask_u8[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    if mask_u8.shape[:2] != (orig_h, orig_w):
        mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_u8.astype(bool)


def unletterbox_segms(
    segm_result: list[list[Any]] | None,
    orig_shape: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[np.ndarray]] | None:
    if segm_result is None:
        return None
    return [[unletterbox_mask(mask, orig_shape, target_size) for mask in cls_masks] for cls_masks in segm_result]


def render_overlay(
    frame: np.ndarray,
    bbox_result: list[np.ndarray],
    segm_result: list[list[np.ndarray]] | None,
    *,
    score_thr: float,
    draw_boxes: bool,
    draw_masks: bool,
    mask_alpha: float,
    mask_color: tuple[int, int, int],
    line_width: int,
    classifier_class_names: list[str] | None = None,
    draw_classifier_labels: bool = True,
) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    has_classifier = bool(classifier_class_names)

    def det_color(det: np.ndarray) -> tuple[int, int, int]:
        if has_classifier and det.shape[0] >= 7:
            cls_idx = int(round(float(det[5])))
            return CLASSIFIER_COLORS[cls_idx % len(CLASSIFIER_COLORS)]
        return mask_color

    if draw_masks and segm_result is not None:
        union_mask = None
        for cls_id in range(min(len(bbox_result), len(segm_result))):
            bboxes = bbox_result[cls_id]
            masks = segm_result[cls_id]
            if bboxes is None or len(bboxes) == 0 or masks is None:
                continue
            keep = np.nonzero(bboxes[:, 4] >= score_thr)[0]
            for det_idx in keep.tolist():
                if det_idx >= len(masks):
                    continue
                mask = masks[det_idx]
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                if has_classifier:
                    color_arr = np.asarray(det_color(bboxes[det_idx]), dtype=np.float32)
                    region = out[mask].astype(np.float32)
                    out[mask] = (region * (1.0 - mask_alpha) + color_arr * mask_alpha).astype(np.uint8)
                else:
                    union_mask = mask if union_mask is None else (union_mask | mask)
        if union_mask is not None:
            color_arr = np.asarray(mask_color, dtype=np.float32)
            region = out[union_mask].astype(np.float32)
            out[union_mask] = (region * (1.0 - mask_alpha) + color_arr * mask_alpha).astype(np.uint8)

    if draw_boxes:
        for cls_id, bboxes in enumerate(bbox_result):
            if bboxes is None or len(bboxes) == 0:
                continue
            keep = bboxes[:, 4] >= score_thr
            for det in bboxes[keep]:
                x1, y1, x2, y2 = [int(round(float(v))) for v in det[:4]]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                color = det_color(det)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)
                label = f"{float(det[4]):.2f}"
                if draw_classifier_labels and has_classifier and det.shape[0] >= 7:
                    cls_idx = int(round(float(det[5])))
                    cls_name = classifier_class_names[cls_idx] if 0 <= cls_idx < len(classifier_class_names) else str(cls_idx)
                    label = f"{cls_name} {float(det[6]):.2f}/{float(det[4]):.2f}"
                cv2.putText(
                    out,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    max(1, line_width),
                    cv2.LINE_AA,
                )

    return out


def serialize_bboxes(
    frame_idx: int,
    bbox_result: list[np.ndarray],
    score_thr: float,
    *,
    classifier_class_names: list[str] | None = None,
    classifier_class_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    records = []
    for cls_id, bboxes in enumerate(bbox_result):
        if bboxes is None or len(bboxes) == 0:
            continue
        for det in bboxes[bboxes[:, 4] >= score_thr]:
            record = {
                "frame": int(frame_idx),
                "category_id": int(cls_id + 1),
                "bbox_xyxy": [round(float(v), 3) for v in det[:4]],
                "score": round(float(det[4]), 6),
            }
            if classifier_class_names is not None and det.shape[0] >= 7:
                multi_idx = int(round(float(det[5])))
                record["multiclass_label_idx"] = multi_idx
                if 0 <= multi_idx < len(classifier_class_names):
                    record["multiclass_label"] = classifier_class_names[multi_idx]
                if classifier_class_ids is not None and 0 <= multi_idx < len(classifier_class_ids):
                    record["multiclass_category_id"] = int(classifier_class_ids[multi_idx])
                record["multiclass_score"] = round(float(det[6]), 6)
                prob_count = min(len(classifier_class_names), max(0, det.shape[0] - 7))
                if prob_count > 0:
                    record["multiclass_probs"] = [
                        round(float(v), 6) for v in det[7 : 7 + prob_count]
                    ]
            records.append(record)
    return records


def make_output_path(args: argparse.Namespace) -> Path:
    if args.out is not None:
        return args.out
    stem = args.video.stem
    ckpt_stem = args.checkpoint.stem
    return DEFAULT_OUTPUT_DIR / f"{stem}_{ckpt_stem}_codino_overlay.mp4"


def make_writer(path: Path, fps: float, size: tuple[int, int], codec: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output writer: {path}")
    return writer


class AsyncVideoWriter:
    def __init__(self, writer, max_queue: int = 16) -> None:
        self.writer = writer
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, max_queue))
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="video-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                try:
                    if frame is None:
                        return
                    self.writer.write(frame)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc

    def write(self, frame: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError("Async video writer failed") from self.error
        self.queue.put(frame)

    def release(self) -> None:
        self.queue.put(None)
        self.thread.join()
        self.writer.release()
        if self.error is not None:
            raise RuntimeError("Async video writer failed") from self.error


class ModuleForwardProfiler:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.records: dict[str, list[float]] = {}
        self._originals: list[tuple[torch.nn.Module, Any]] = []

    def _wrap(self, name: str, module: torch.nn.Module | None) -> None:
        if module is None or not hasattr(module, "forward"):
            return
        original = module.forward
        records = self.records.setdefault(name, [])

        def timed_forward(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            records.append(time.perf_counter() - t0)
            return out

        module.forward = timed_forward
        self._originals.append((module, original))

    def install(self) -> None:
        self._wrap("backbone", getattr(self.model, "backbone", None))
        self._wrap("neck", getattr(self.model, "neck", None))
        self._wrap("query_head", getattr(self.model, "query_head", None))
        query_head = getattr(self.model, "query_head", None)
        transformer = getattr(query_head, "transformer", None)
        self._wrap("positional_encoding", getattr(query_head, "positional_encoding", None))
        self._wrap("transformer", transformer)
        self._wrap("transformer_encoder", getattr(transformer, "encoder", None))
        self._wrap("transformer_decoder", getattr(transformer, "decoder", None))
        self._wrap("mask_roi_extractor", getattr(self.model, "mask_roi_extractor", None))
        self._wrap("mask_head", getattr(self.model, "mask_head", None))
        self._wrap("mask_iou_head", getattr(self.model, "mask_iou_head", None))

    def print_summary(self, frames: int) -> None:
        for name, values in self.records.items():
            if not values:
                continue
            total_ms = sum(values) * 1000.0
            avg_ms = total_ms / len(values)
            per_frame = total_ms / max(1, frames)
            print(
                f"[profile] {name} calls={len(values)} "
                f"total_ms={total_ms:.1f} ms/frame={per_frame:.1f} avg_call_ms={avg_ms:.1f}"
            )


def assert_cuda_runtime_compatible(device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    torch_device = torch.device(device)
    capability = torch.cuda.get_device_capability(torch_device)
    required = f"sm_{capability[0]}{capability[1]}"
    supported = set(torch.cuda.get_arch_list())
    if supported and required not in supported:
        raise RuntimeError(
            f"current PyTorch build does not support this GPU capability ({required}). "
            f"supported={sorted(supported)}. Use a Co-DINO runtime Python with a PyTorch/CUDA build "
            "that supports the installed GPU."
        )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    assert_cuda_runtime_compatible(args.device)
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.roi_classifier and args.disable_mask_head:
        raise ValueError("--roi-classifier requires masks; remove --disable-mask-head.")
    if args.roi_classifier and args.eval_module not in (None, "detr"):
        raise ValueError("--roi-classifier was trained on the Co-DINO query-head path; use default eval module or --eval-module detr.")

    _prepare_imports()

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    if args.tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    mask_color = parse_bgr_color(args.mask_color)
    model = load_detector(args)
    selected_engines = [args.onnx_backbone is not None, args.trt_backbone_engine is not None, args.trt_feature_engine is not None]
    if sum(selected_engines) > 1:
        raise ValueError("--onnx-backbone, --trt-backbone-engine, and --trt-feature-engine are mutually exclusive.")
    if args.onnx_backbone is not None:
        install_onnx_backbone(model, args.onnx_backbone.expanduser().resolve(), args.onnx_max_batch)
    if args.trt_backbone_engine is not None:
        extra_site_packages = args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None
        install_trt_backbone(model, args.trt_backbone_engine.expanduser().resolve(), extra_site_packages)
    if args.trt_feature_engine is not None:
        extra_site_packages = args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None
        feature_names = tuple(part.strip() for part in args.trt_feature_names.split(",") if part.strip())
        install_trt_feature_extractor(
            model,
            args.trt_feature_engine.expanduser().resolve(),
            feature_names,
            extra_site_packages,
        )
    apply_runtime_model_options(model, args)
    if args.trt_query_encoder_engine is not None:
        extra_site_packages = args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None
        install_trt_query_encoder(
            model,
            args.trt_query_encoder_engine.expanduser().resolve(),
            parse_hw_shapes(args.trt_query_encoder_shapes),
            extra_site_packages,
        )
    if args.trt_decoder_engine is not None:
        extra_site_packages = args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None
        install_trt_decoder(
            model,
            args.trt_decoder_engine.expanduser().resolve(),
            extra_site_packages,
        )
    if args.trt_mask_head_engine is not None:
        extra_site_packages = args.trt_extra_site_packages.expanduser().resolve() if args.trt_extra_site_packages else None
        install_trt_mask_head(
            model,
            args.trt_mask_head_engine.expanduser().resolve(),
            extra_site_packages,
        )
    apply_compile_options(model, args)
    roi_classifier = None
    roi_classifier_raw: dict[str, Any] = {}
    roi_classifier_class_names: list[str] | None = None
    roi_classifier_class_ids: list[int] | None = None
    if args.roi_classifier:
        roi_classifier_ckpt = args.roi_classifier_checkpoint.expanduser().resolve()
        roi_classifier, roi_classifier_raw, roi_classifier_class_names, roi_classifier_class_ids = load_roi_classifier(
            roi_classifier_ckpt,
            args.device,
        )
    module_profiler = None
    if args.profile_modules:
        module_profiler = ModuleForwardProfiler(model)
        module_profiler.install()
    pipeline = build_test_pipeline(model)
    target_size = find_letterbox_size(model)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    output_path = make_output_path(args)
    writer = None
    if not args.no_video:
        writer = make_writer(output_path, fps / args.frame_stride, (width, height), args.codec)
        if args.async_writer:
            writer = AsyncVideoWriter(writer, max_queue=args.writer_queue_size)

    json_records: list[dict[str, Any]] = []
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[model] config={args.config}")
    print(f"[model] checkpoint={args.checkpoint}")
    print(f"[video] in={args.video} size={width}x{height} fps={fps:.3f} frames={total_frames or 'unknown'}")
    print(
        f"[cfg] device={args.device} batch={args.batch_size} amp={args.amp} "
        f"tf32={args.tf32} preprocess={args.preprocess} "
        f"score_thr={args.score_thr} model_score_thr={min(args.score_thr, 0.05) if args.model_score_thr is None else args.model_score_thr} "
        f"target={target_size[1]}x{target_size[0]}"
    )
    if roi_classifier is not None:
        print(f"[cfg] roi_classifier=true labels={roi_classifier_class_names}")
    if writer is not None:
        print(f"[video] out={output_path}")
    if args.json_out is not None:
        print(f"[json] out={args.json_out}")

    batch_frames: list[np.ndarray] = []
    batch_ids: list[int] = []
    written = 0
    seen = 0
    batches = 0
    infer_s = 0.0
    draw_s = 0.0
    write_s = 0.0
    recent = deque(maxlen=max(1, args.log_interval))
    wall_start = time.perf_counter()
    frame_idx = args.start_frame - 1

    def flush() -> None:
        nonlocal batches, infer_s, draw_s, write_s, written
        if not batch_frames:
            return
        valid_count = len(batch_frames)
        frames_for_infer = batch_frames
        ids_for_infer = batch_ids
        if (
            args.onnx_backbone is not None
            or args.trt_backbone_engine is not None
            or args.trt_feature_engine is not None
            or args.trt_query_encoder_engine is not None
        ) and valid_count < args.batch_size:
            pad_count = args.batch_size - valid_count
            frames_for_infer = batch_frames + [batch_frames[-1]] * pad_count
            ids_for_infer = batch_ids + [batch_ids[-1]] * pad_count
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if roi_classifier is None:
            results = infer_batch(
                model,
                pipeline,
                frames_for_infer,
                args.amp,
                preprocess=args.preprocess,
                target_size=target_size,
            )
        else:
            results = infer_batch_with_roi_classifier(
                model,
                pipeline,
                frames_for_infer,
                args.amp,
                preprocess=args.preprocess,
                target_size=target_size,
                classifier=roi_classifier,
                num_classifier_classes=len(roi_classifier_class_names or []),
            )
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        infer_s += t1 - t0
        batches += 1

        for frame, src_frame_idx, result in zip(batch_frames, batch_ids, results[:valid_count]):
            bbox_result, segm_result = normalize_result(result)
            orig_shape = frame.shape[:2]
            bbox_result = unletterbox_bboxes(bbox_result, orig_shape, target_size)
            if writer is not None and args.draw_masks:
                segm_result = unletterbox_segms(segm_result, orig_shape, target_size)
            else:
                segm_result = None

            if args.json_out is not None:
                json_records.extend(
                    serialize_bboxes(
                        src_frame_idx,
                        bbox_result,
                        args.score_thr,
                        classifier_class_names=roi_classifier_class_names,
                        classifier_class_ids=roi_classifier_class_ids,
                    )
                )

            vis = frame
            if writer is not None:
                td0 = time.perf_counter()
                vis = render_overlay(
                    frame,
                    bbox_result,
                    segm_result,
                    score_thr=args.score_thr,
                    draw_boxes=args.draw_boxes,
                    draw_masks=args.draw_masks,
                    mask_alpha=args.mask_alpha,
                    mask_color=mask_color,
                    line_width=args.line_width,
                    classifier_class_names=roi_classifier_class_names,
                    draw_classifier_labels=bool(args.draw_classifier_labels),
                )
                td1 = time.perf_counter()
                draw_s += td1 - td0
                tw0 = time.perf_counter()
                writer.write(vis)
                tw1 = time.perf_counter()
                write_s += tw1 - tw0

            written += 1
            recent.append(time.perf_counter())
            if args.log_interval > 0 and written % args.log_interval == 0:
                elapsed = time.perf_counter() - wall_start
                fps_total = written / elapsed if elapsed > 0 else 0.0
                if len(recent) >= 2:
                    fps_recent = (len(recent) - 1) / (recent[-1] - recent[0])
                else:
                    fps_recent = 0.0
                print(
                    f"[perf] frame={src_frame_idx} written={written} "
                    f"fps(avg={fps_total:.2f},recent={fps_recent:.2f}) "
                    f"ms/frame(infer={infer_s / written * 1000:.1f},draw={draw_s / max(1, written) * 1000:.1f},write={write_s / max(1, written) * 1000:.1f})"
                )

        batch_frames.clear()
        batch_ids.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx < args.start_frame:
                continue
            if (frame_idx - args.start_frame) % args.frame_stride != 0:
                continue
            batch_frames.append(frame)
            batch_ids.append(frame_idx)
            seen += 1
            if args.max_frames is not None and seen >= args.max_frames:
                flush()
                break
            if len(batch_frames) >= args.batch_size:
                flush()
                if args.warmup_batches > 0 and batches == args.warmup_batches:
                    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                        torch.cuda.reset_peak_memory_stats()
        flush()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if args.json_out is not None:
        payload = {
            "video": str(args.video),
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "score_thr": args.score_thr,
            "frame_stride": args.frame_stride,
            "roi_classifier": {
                "enabled": bool(roi_classifier is not None),
                "checkpoint": str(args.roi_classifier_checkpoint.expanduser().resolve()) if args.roi_classifier else None,
                "model_type": str((roi_classifier_raw.get("model_cfg") or {}).get("model_type", "")) if args.roi_classifier else None,
                "val_macro_f1": (roi_classifier_raw.get("val_metrics") or {}).get("macro_f1") if args.roi_classifier else None,
                "class_names": roi_classifier_class_names if roi_classifier_class_names is not None else None,
            },
            "predictions": json_records,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - wall_start
    fps_total = written / elapsed if elapsed > 0 else 0.0
    peak_gb = None
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_text = f" peak_alloc={peak_gb:.2f}GB" if peak_gb is not None else ""
    target = output_path if writer is not None else "(no video)"
    print(
        f"[done] frames={written} elapsed={elapsed:.2f}s fps={fps_total:.2f} "
        f"ms/frame(infer={infer_s / max(1, written) * 1000:.1f},draw={draw_s / max(1, written) * 1000:.1f},write={write_s / max(1, written) * 1000:.1f})"
        f"{peak_text} -> {target}"
    )
    if module_profiler is not None:
        module_profiler.print_summary(written)


if __name__ == "__main__":
    main()
