import os
import time
import ctypes.util
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


def _collect_site_paths() -> list[str]:
    paths: list[str] = []
    try:
        import site as _site, sys as _sys, os as _os
        cand = [p for p in _sys.path if p and "site-packages" in p]
        try:
            cand += list(_site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = _site.getusersitepackages()
            if user_site:
                cand.append(user_site)
        except Exception:
            pass
        for p in cand:
            rp = _os.path.realpath(p)
            if rp not in paths and _os.path.isdir(rp):
                paths.append(rp)
    except Exception:
        pass
    return paths


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") in ("1", "true", "True")


def _extract_onnx_input_meta(onnx_path: str) -> Optional[tuple[str, list[Optional[int]]]]:
    try:
        import onnx
        model = onnx.load(onnx_path)
        if not model.graph.input:
            return None
        inp = model.graph.input[0]
        dims: list[Optional[int]] = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        return inp.name, dims
    except Exception:
        return None


def _append_runtime_libs_to_env():
    """Add venv vendor lib paths (TensorRT/cuDNN/cuBLAS/cudart) to LD_LIBRARY_PATH."""
    try:
        import os as _os
        cand = []
        for site in _collect_site_paths():
            cand.extend(
                [
                    f"{site}/tensorrt_libs",
                    f"{site}/nvidia/cudnn/lib",
                    f"{site}/nvidia/cublas/lib",
                    f"{site}/nvidia/cuda_runtime/lib",
                ]
            )
        trt_lib_dir = _os.environ.get("TENSORRT_LIB_DIR")
        if trt_lib_dir:
            cand.append(trt_lib_dir)
        cur = _os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cand if _os.path.isdir(p)] + ([cur] if cur else [])
        _os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    except Exception:
        pass


def _preload_vendor_libs():
    """dlopen vendor libs to help ORT find deps even if LD_LIBRARY_PATH is flaky."""
    try:
        import os as _os, glob as _glob, ctypes as _ctypes
        for site in _collect_site_paths():
            for p in (
                _os.path.join(site, 'tensorrt_libs', 'libnvinfer.so.10'),
                _os.path.join(site, 'tensorrt_libs', 'libnvinfer_plugin.so.10'),
                _os.path.join(site, 'tensorrt_libs', 'libnvonnxparser.so.10'),
            ):
                if _os.path.isfile(p):
                    try:
                        _ctypes.CDLL(p)
                    except Exception:
                        pass
            for pat in (
                _os.path.join(site, 'nvidia', 'cudnn', 'lib', 'libcudnn*.so.9'),
                _os.path.join(site, 'nvidia', 'cublas', 'lib', 'libcublas*.so.12'),
                _os.path.join(site, 'nvidia', 'cuda_runtime', 'lib', 'libcudart*.so*'),
            ):
                for p in _glob.glob(pat):
                    try:
                        _ctypes.CDLL(p)
                    except Exception:
                        pass
    except Exception:
        pass


def _has_tensorrt_lib() -> bool:
    """Check whether TensorRT runtime libs are discoverable."""
    try:
        if ctypes.util.find_library("nvinfer"):
            return True
    except Exception:
        pass
    try:
        import os as _os
        trt_lib_dir = _os.environ.get("TENSORRT_LIB_DIR")
        if trt_lib_dir:
            for name in ("libnvinfer.so.10", "libnvinfer.so"):
                if _os.path.isfile(_os.path.join(trt_lib_dir, name)):
                    return True
        for site in _collect_site_paths():
            for name in ("libnvinfer.so.10", "libnvinfer.so"):
                if _os.path.isfile(_os.path.join(site, "tensorrt_libs", name)):
                    return True
    except Exception:
        pass
    return False


def _get_providers(
    fp16: bool = True,
    input_profile: Optional[dict[str, list[tuple[int, int, int, int]]]] = None,
    user_compute_stream: str | None = None,
) -> list:
    providers = []
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
        ws_env = os.environ.get('ORT_TRT_MAX_WORKSPACE_BYTES')
        try:
            ws_bytes = int(ws_env) if ws_env else (8 << 30)
        except Exception:
            ws_bytes = (8 << 30)
        disable_trt = _env_flag('EVA_DISABLE_TRT_EP')
        require_cuda = _env_flag('EVA_REQUIRE_CUDA_EP')
        disable_cpu = _env_flag('EVA_DISABLE_CPU_EP')
        trt_opts: Dict[str, Any] = {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": os.getenv("ORT_TRT_CACHE", "./output/trt/ort_trt_cache"),
            "trt_max_workspace_size": str(ws_bytes),
            "trt_timing_cache_enable": "True",
            "trt_fp16_enable": "True" if fp16 else "False",
        }
        if _env_flag("EVA_TRT_FORCE_FP16"):
            trt_opts["trt_fp16_enable"] = "True"
        if _env_flag("EVA_TRT_DISABLE_FP16"):
            trt_opts["trt_fp16_enable"] = "False"
        cache_prefix = os.environ.get("EVA_TRT_ENGINE_CACHE_PREFIX")
        if cache_prefix:
            trt_opts["trt_engine_cache_prefix"] = cache_prefix
        if _env_flag("EVA_TRT_CUDA_GRAPH"):
            trt_opts["trt_cuda_graph_enable"] = "True"
        if _env_flag("EVA_TRT_DETAILED_LOG"):
            trt_opts["trt_detailed_build_log"] = "True"
        if _env_flag("EVA_TRT_DUMP_SUBGRAPHS"):
            trt_opts["trt_dump_subgraphs"] = "True"
        if _env_flag("EVA_TRT_LAYER_NORM_FP32"):
            trt_opts["trt_layer_norm_fp32_fallback"] = "True"
        if _env_flag("EVA_TRT_CONTEXT_MEMORY_SHARING"):
            trt_opts["trt_context_memory_sharing_enable"] = "True"
        if _env_flag("EVA_TRT_BUILD_HEURISTICS"):
            trt_opts["trt_build_heuristics_enable"] = "True"
        builder_level = os.environ.get("EVA_TRT_BUILDER_OPT_LEVEL")
        if builder_level:
            trt_opts["trt_builder_optimization_level"] = builder_level
        aux_streams = os.environ.get("EVA_TRT_AUX_STREAMS")
        if aux_streams:
            trt_opts["trt_auxiliary_streams"] = aux_streams
        tactic_sources = os.environ.get("EVA_TRT_TACTIC_SOURCES")
        if tactic_sources:
            trt_opts["trt_tactic_sources"] = tactic_sources
        excluded_ops = os.environ.get("EVA_TRT_OP_TYPES_TO_EXCLUDE")
        if excluded_ops:
            trt_opts["trt_op_types_to_exclude"] = excluded_ops
        min_subgraph = os.environ.get("EVA_TRT_MIN_SUBGRAPH_SIZE")
        if min_subgraph:
            trt_opts["trt_min_subgraph_size"] = min_subgraph
        if user_compute_stream is None and torch.cuda.is_available() and not _env_flag("EVA_ORT_DISABLE_TORCH_STREAM"):
            try:
                user_compute_stream = str(torch.cuda.current_stream().cuda_stream)
            except Exception:
                user_compute_stream = None
        if user_compute_stream is not None:
            trt_opts["user_compute_stream"] = user_compute_stream
        if input_profile:
            min_shapes = []
            opt_shapes = []
            max_shapes = []
            for name, shapes in input_profile.items():
                if not isinstance(shapes, list) or len(shapes) != 3:
                    continue
                min_shape, opt_shape, max_shape = shapes
                min_shapes.append(f"{name}:" + "x".join([str(x) for x in min_shape]))
                opt_shapes.append(f"{name}:" + "x".join([str(x) for x in opt_shape]))
                max_shapes.append(f"{name}:" + "x".join([str(x) for x in max_shape]))
            if min_shapes and opt_shapes and max_shapes:
                trt_opts["trt_profile_min_shapes"] = ",".join(min_shapes)
                trt_opts["trt_profile_opt_shapes"] = ",".join(opt_shapes)
                trt_opts["trt_profile_max_shapes"] = ",".join(max_shapes)
        cuda_opts = {
            "cudnn_conv_use_max_workspace": True,
            "arena_extend_strategy": "kNextPowerOfTwo",
        }
        if user_compute_stream is not None:
            cuda_opts["user_compute_stream"] = user_compute_stream
        trt_ok = ("TensorrtExecutionProvider" in avail) and _has_tensorrt_lib()
        if ("TensorrtExecutionProvider" in avail) and not trt_ok and not disable_trt:
            print("[ORT] TensorRT EP available but libnvinfer not found; skipping TRT EP.")
        if trt_ok and not disable_trt:
            providers.append(("TensorrtExecutionProvider", trt_opts))
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", cuda_opts))
        elif require_cuda and not trt_ok:
            raise RuntimeError("CUDAExecutionProvider not available; refusing to run on CPU.")
        if not disable_cpu:
            providers.append("CPUExecutionProvider")
    except Exception:
        providers = ["CPUExecutionProvider"]
    return providers


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    raise TypeError(f"Unsupported dtype: {dtype}")


def _parse_onnx_type(type_str: str) -> tuple[np.dtype, torch.dtype]:
    t = type_str.lower()
    if "float16" in t:
        return np.float16, torch.float16
    if "float" in t:
        return np.float32, torch.float32
    return np.float32, torch.float32


class ORTBackboneAdapter(nn.Module):
    def __init__(self, onnx_path: str, max_batch: int = 4, expect_dtype: Optional[torch.dtype] = torch.float16, provider_device_id: int | None = None):
        super().__init__()
        import onnxruntime as ort
        self.expect_dtype = expect_dtype
        self.force_fp32_output = os.environ.get("EVA_ONNX_FORCE_FP32_OUTPUT", "0") in ("1", "true", "True")
        _append_runtime_libs_to_env()
        _preload_vendor_libs()
        input_profile = None
        self.ort_stream: torch.cuda.Stream | None = None
        user_compute_stream = None
        if _env_flag("EVA_ORT_USE_DEDICATED_STREAM") and torch.cuda.is_available():
            self.ort_stream = torch.cuda.Stream()
            user_compute_stream = str(self.ort_stream.cuda_stream)
        elif torch.cuda.is_available() and not _env_flag("EVA_ORT_DISABLE_TORCH_STREAM"):
            try:
                user_compute_stream = str(torch.cuda.current_stream().cuda_stream)
            except Exception:
                user_compute_stream = None
        self.using_user_compute_stream = user_compute_stream is not None
        if os.environ.get("EVA_TRT_PROFILE_DISABLE", "0") not in ("1", "true", "True"):
            meta = _extract_onnx_input_meta(onnx_path)
            if meta:
                input_name, shape = meta
                if len(shape) >= 4:
                    c, h, w = shape[1], shape[2], shape[3]
                    if c and h and w:
                        min_b = max(1, _get_env_int("EVA_TRT_PROFILE_MIN_BATCH", 1))
                        opt_b = max(min_b, _get_env_int("EVA_TRT_PROFILE_OPT_BATCH", max_batch))
                        max_b = max(opt_b, _get_env_int("EVA_TRT_PROFILE_MAX_BATCH", max_batch))
                        input_profile = {
                            input_name: [
                                (min_b, int(c), int(h), int(w)),
                                (opt_b, int(c), int(h), int(w)),
                                (max_b, int(c), int(h), int(w)),
                            ]
                        }
        providers = _get_providers(
            fp16=(expect_dtype == torch.float16),
            input_profile=input_profile,
            user_compute_stream=user_compute_stream,
        )
        # ensure cache dir
        try:
            for p in providers:
                if isinstance(p, tuple) and p[0] == 'TensorrtExecutionProvider':
                    cache_dir = p[1].get('trt_engine_cache_path')
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        ins = self.session.get_inputs(); outs = self.session.get_outputs()
        self.input_name = ins[0].name if ins else 'input'
        self.output_name = outs[0].name if outs else 'last_feat'
        self.use_iobinding = os.environ.get("EVA_ONNXRT_IOBIND", "1") in ("1", "true", "True")
        self.device_id = provider_device_id if provider_device_id is not None else (torch.cuda.current_device() if torch.cuda.is_available() else 0)
        self.input_np_dtype = _torch_dtype_to_numpy(expect_dtype or torch.float32)
        self.output_np_dtype, self.output_torch_dtype = _parse_onnx_type(outs[0].type if outs else "tensor(float)")
        self.input_shape_template = ins[0].shape if ins else None
        self.output_shape_template = outs[0].shape if outs else None
        self.output_tensor: Optional[torch.Tensor] = None
        self.providers_in_use: list[str] = []
        self.sync_after_iobinding = _env_flag("EVA_ONNXRT_SYNC_AFTER_IOBIND")
        try:
            prov = self.session.get_providers()
            self.providers_in_use = list(prov)
            if "TensorrtExecutionProvider" in self.providers_in_use and os.environ.get("EVA_ONNXRT_SYNC_AFTER_IOBIND") is None:
                # TensorRT EP can return from IO binding before downstream PyTorch
                # kernels observe the freshly-written output tensor on this setup.
                # Passing PyTorch's stream to ORT fixes the ordering without a
                # device-wide synchronize.
                self.sync_after_iobinding = not self.using_user_compute_stream
            print(f"[ORT] Providers in use: {prov}")
            require_cuda = os.environ.get('EVA_REQUIRE_CUDA_EP', '0') in ('1', 'true', 'True')
            if require_cuda:
                has_gpu_ep = any(p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") for p in prov)
                if not has_gpu_ep:
                    raise RuntimeError("CUDA/TRT EP not active; aborting to avoid CPU inference.")
        except Exception:
            pass

    def _resolve_output_shape(self, batch: int) -> Optional[list]:
        if not self.output_shape_template:
            return None
        shape: list = []
        for i, dim in enumerate(self.output_shape_template):
            if isinstance(dim, int):
                shape.append(dim)
            else:
                if i == 0:
                    shape.append(batch)
                else:
                    shape.append(None)
        if all(dim is not None for dim in shape):
            return shape

        # DINOv3 ViT-L/16 backbone export often leaves non-batch output dims
        # symbolic even at a fixed input resolution. Infer them from the fixed
        # ONNX input shape so CUDA IO binding can be used from the first batch.
        if len(shape) == 4 and self.input_shape_template and len(self.input_shape_template) == 4:
            try:
                in_h = self.input_shape_template[2]
                in_w = self.input_shape_template[3]
                if isinstance(in_h, int) and isinstance(in_w, int):
                    out_channels = _get_env_int("DINOV3_ONNX_OUT_CHANNELS", 1024)
                    patch_stride = _get_env_int("DINOV3_ONNX_PATCH_STRIDE", 16)
                    return [batch, out_channels, int(in_h) // patch_stride, int(in_w) // patch_stride]
            except Exception:
                pass
        return None

    def _forward_numpy(self, x: torch.Tensor) -> dict:
        if self.expect_dtype is not None and x.dtype != self.expect_dtype:
            x = x.to(self.expect_dtype)
        x_np = x.detach().cpu().numpy()
        # 初回実行で空出力/失敗が出る稀ケースに対して軽リトライ
        outs = []
        for _ in range(3):
            try:
                outs = self.session.run([self.output_name], {self.input_name: x_np})
                if outs:
                    break
                outs = self.session.run(None, {self.input_name: x_np})
                if outs:
                    break
            except Exception:
                time.sleep(0.05)
        if not outs:
            # 最後の手段: 全出力で再試行
            outs = self.session.run(None, {self.input_name: x_np})
        y_np = outs[0]
        y = torch.from_numpy(y_np).to(x.device, non_blocking=True)
        if self.force_fp32_output and y.dtype != torch.float32:
            y = y.to(torch.float32)
        return {"last_feat": y}

    def _forward_iobinding(self, x: torch.Tensor) -> dict:
        if self.expect_dtype is not None and x.dtype != self.expect_dtype:
            x = x.to(self.expect_dtype)
        x = x.contiguous()
        if (self.output_tensor is None) or (self.output_tensor.shape[0] != x.shape[0]):
            shape = self._resolve_output_shape(x.shape[0])
            if shape is None:
                # Fallback once to infer shape
                y = self._forward_numpy(x)
                self.output_tensor = y["last_feat"].to(self.output_torch_dtype)
                return {"last_feat": y["last_feat"]}
            self.output_tensor = torch.empty(shape, device=x.device, dtype=self.output_torch_dtype)
        io_binding = self.session.io_binding()
        io_binding.bind_input(self.input_name, "cuda", self.device_id, self.input_np_dtype, tuple(x.shape), x.data_ptr())
        io_binding.bind_output(self.output_name, "cuda", self.device_id, self.output_np_dtype, tuple(self.output_tensor.shape), self.output_tensor.data_ptr())
        current_stream = None
        if self.ort_stream is not None:
            current_stream = torch.cuda.current_stream(device=x.device)
            self.ort_stream.wait_stream(current_stream)
        self.session.run_with_iobinding(io_binding)
        # Ensure ORT has finished writing outputs before downstream PyTorch uses them.
        try:
            io_binding.synchronize_outputs()
        except Exception:
            pass
        if self.sync_after_iobinding and x.is_cuda:
            torch.cuda.synchronize(device=x.device)
        elif current_stream is not None and self.ort_stream is not None:
            current_stream.wait_stream(self.ort_stream)
        y = self.output_tensor
        if self.force_fp32_output and self.output_torch_dtype != torch.float32:
            y = y.to(torch.float32)
        return {"last_feat": y}

    def forward(self, x: torch.Tensor) -> dict:
        if self.use_iobinding and x.is_cuda:
            return self._forward_iobinding(x)
        return self._forward_numpy(x)
