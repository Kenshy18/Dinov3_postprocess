#!/usr/bin/env python3
"""Detect local GPU/runtime capabilities and write conservative runtime defaults."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / ".runtime" / "runtime_profile.json"


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def codino_engine_env_is_set() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "CODINO_TRT_BACKBONE_ENGINE",
            "CODINO_TRT_QUERY_ENCODER_ENGINE",
            "CODINO_TRT_DECODER_ENGINE",
            "CODINO_TRT_MASK_HEAD_ENGINE",
        )
    )


DEFAULT_TRT_ENGINE = env_path(
    "DINOV3_TRT_BACKBONE_ENGINE",
    ROOT / "checkpoints" / "trt" / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine",
)
DEFAULT_CODINO_TRT_BACKBONE = env_path(
    "CODINO_TRT_BACKBONE_ENGINE",
    ROOT / "checkpoints/codino/trt/codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine",
)
DEFAULT_CODINO_TRT_QUERY_ENCODER = env_path(
    "CODINO_TRT_QUERY_ENCODER_ENGINE",
    ROOT / "checkpoints/codino/trt/codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine",
)
DEFAULT_CODINO_TRT_DECODER = env_path(
    "CODINO_TRT_DECODER_ENGINE",
    ROOT / "checkpoints/codino/trt/codino_decoder_b2_736x1280_msda_plugin_fp16.engine",
)
DEFAULT_CODINO_TRT_MASK_HEAD = env_path(
    "CODINO_TRT_MASK_HEAD_ENGINE",
    ROOT / "checkpoints/codino/trt/codino_mask_head_core_n1_736x1280_fp16.engine",
)
DEFAULT_CODINO_TRT_MANIFEST = env_path(
    "CODINO_TRT_MANIFEST",
    ROOT / "checkpoints/codino/trt/codino_trt_manifest.json",
)
DEFAULT_BENCHMARK = env_path("DINOV3_BATCH_BENCHMARK", ROOT / ".runtime" / "runtime_benchmark.json")
DEFAULT_RTDETR_REPO = env_path("RTDETR_REPO", ROOT / "external/RT-DETR/RT-DETRv4")
DEFAULT_RTDETR_CONFIG = env_path(
    "RTDETR_CONFIG",
    DEFAULT_RTDETR_REPO / "configs" / "rtv2" / "rtv2_r18vd_72e_crowdhuman_citypersons_vhf.yml",
)
DEFAULT_RTDETR_CHECKPOINT = env_path(
    "RTDETR_CHECKPOINT",
    ROOT / "checkpoints" / "rtdetr" / "head_face_best_stg1.pth",
)


def _codino_default_paths_for_batch(batch: int) -> dict[str, str]:
    trt_dir = ROOT / "checkpoints/codino/trt"
    return {
        "trt_backbone_engine": str(
            trt_dir / f"codino_dinov3_vitl_backbone_736x1280_fp32_b{batch}_fixed_bf16.engine"
        ),
        "trt_query_encoder_engine": str(
            trt_dir / f"codino_query_encoder_b{batch}_736x1280_msda_plugin_sbc_fp16.engine"
        ),
        "trt_decoder_engine": str(trt_dir / f"codino_decoder_b{batch}_736x1280_msda_plugin_fp16.engine"),
        "trt_mask_head_engine": str(trt_dir / "codino_mask_head_core_n1_736x1280_fp16.engine"),
    }


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def nvidia_smi_info() -> dict[str, Any] | None:
    text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text:
        return None
    first = text.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        total_mib = int(float(parts[1]))
        free_mib = int(float(parts[2]))
    except ValueError:
        total_mib = free_mib = None  # type: ignore[assignment]
    return {
        "name": parts[0],
        "memory_total_mib": total_mib,
        "memory_free_mib": free_mib,
        "driver_version": parts[3],
    }


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception as exc:
        info["import_error"] = repr(exc)
        return info

    info.update(
        {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["memory_total_mib"] = int(props.total_memory // (1024 * 1024))
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            info["memory_free_mib"] = int(free_bytes // (1024 * 1024))
            info["memory_total_runtime_mib"] = int(total_bytes // (1024 * 1024))
        except Exception:
            pass
        try:
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            info["bf16_supported"] = None
    return info


def tensorrt_info() -> dict[str, Any]:
    try:
        import tensorrt as trt
    except Exception as exc:
        return {"available": False, "import_error": repr(exc)}
    return {"available": True, "version": getattr(trt, "__version__", None)}


def tensorrt_site_packages() -> Path | None:
    raw = os.environ.get("TENSORRT_SITE_PACKAGES")
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    return None


def choose_vram_mib(torch_data: dict[str, Any], smi_data: dict[str, Any] | None) -> int | None:
    for source in (torch_data, smi_data or {}):
        value = source.get("memory_total_mib")
        if isinstance(value, int) and value > 0:
            return value
    return None


def load_codino_trt_manifest(path: Path = DEFAULT_CODINO_TRT_MANIFEST) -> dict[str, Any] | None:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def codino_trt_paths(batch: int | None = None) -> dict[str, str]:
    paths = {
        "trt_backbone_engine": str(DEFAULT_CODINO_TRT_BACKBONE),
        "trt_query_encoder_engine": str(DEFAULT_CODINO_TRT_QUERY_ENCODER),
        "trt_decoder_engine": str(DEFAULT_CODINO_TRT_DECODER),
        "trt_mask_head_engine": str(DEFAULT_CODINO_TRT_MASK_HEAD),
    }
    if batch is not None and batch > 0 and not codino_engine_env_is_set():
        paths.update(_codino_default_paths_for_batch(batch))
    manifest = load_codino_trt_manifest()
    engines = (
        manifest.get("engines", {})
        if batch is None and not codino_engine_env_is_set() and isinstance(manifest, dict)
        else {}
    )
    if isinstance(engines, dict):
        mapping = {
            "backbone": "trt_backbone_engine",
            "query_encoder": "trt_query_encoder_engine",
            "decoder": "trt_decoder_engine",
            "mask_head": "trt_mask_head_engine",
        }
        for src_key, dst_key in mapping.items():
            value = engines.get(src_key)
            if value:
                paths[dst_key] = str(Path(str(value)).expanduser())
    return paths


def codino_manifest_batch() -> int | None:
    manifest = load_codino_trt_manifest()
    if not isinstance(manifest, dict):
        return None
    try:
        batch = int(manifest.get("batch_size"))
    except Exception:
        return None
    return batch if batch > 0 else None


def recommendations(
    total_mib: int | None,
    *,
    tensorrt_available: bool,
    engine_exists: bool,
    benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if total_mib is None:
        total_gib = 0.0
    else:
        total_gib = total_mib / 1024.0

    if total_gib <= 0:
        dinov3_batch = 1
        eva02_batch = 1
        codino_batch = 1
        rtdetr_batch = 16
        classifier_batch = 512
    elif total_gib < 10:
        dinov3_batch = 2
        eva02_batch = 1
        codino_batch = 1
        rtdetr_batch = 16
        classifier_batch = 512
    elif total_gib < 16:
        dinov3_batch = 4
        eva02_batch = 1
        codino_batch = 1
        rtdetr_batch = 32
        classifier_batch = 1024
    elif total_gib < 24:
        dinov3_batch = 6
        eva02_batch = 2
        codino_batch = 1
        rtdetr_batch = 64
        classifier_batch = 1536
    elif total_gib < 40:
        dinov3_batch = 8
        eva02_batch = 4
        codino_batch = 2
        rtdetr_batch = 128
        classifier_batch = 2048
    else:
        dinov3_batch = 8
        eva02_batch = 8
        codino_batch = 2
        rtdetr_batch = 128
        classifier_batch = 4096

    notes = []
    trt_site = tensorrt_site_packages()
    if not tensorrt_available:
        notes.append("TensorRT import failed; DINOv3 will be much slower unless dependencies are fixed.")
        dinov3_batch = min(dinov3_batch, 2)
    if not engine_exists:
        notes.append("DINOv3 TensorRT engine is missing; run setup with REBUILD_TRT=1 or allow auto rebuild.")
    benchmark_codino_batch: int | None = None
    selected = benchmark.get("selected") if isinstance(benchmark, dict) else None
    selected_codino = selected.get("codino") if isinstance(selected, dict) else None
    if isinstance(selected_codino, dict):
        try:
            benchmark_codino_batch = int(selected_codino.get("batch_size"))
        except Exception:
            benchmark_codino_batch = None
    env_codino_batch = env_int("CODINO_TRT_BATCH_SIZE")
    selected_codino_batch = benchmark_codino_batch or env_codino_batch or codino_manifest_batch() or codino_batch
    selected_rtdetr_batch = env_int("RTDETR_BATCH_SIZE") or rtdetr_batch
    trt_paths = codino_trt_paths(selected_codino_batch)
    codino_trt_engines = tuple(Path(value) for value in trt_paths.values())
    if not all(path.is_file() for path in codino_trt_engines):
        notes.append("One or more Co-DINO TensorRT engines are missing under checkpoints/codino/trt.")
    if total_gib and total_gib < 16:
        notes.append("VRAM is below 16 GiB; EVA02 batch is intentionally kept at 1 to avoid shared-memory fallback.")

    return {
        "total_vram_gib": round(total_gib, 2) if total_gib else None,
        "dinov3": {
            "batch_size": dinov3_batch,
            "warmup_frames": 0,
            "trt_backbone_engine": str(DEFAULT_TRT_ENGINE),
        },
        "eva02": {
            "batch_size": eva02_batch,
            "warmup_frames": 0,
            "classifier_batch_size": classifier_batch,
        },
        "codino": {
            "batch_size": selected_codino_batch,
            "warmup_frames": 0,
            "target_size": "1280x720",
            "score_thresh": 0.3,
            "model_score_thr": 0.05,
            "amp": "fp16",
            "tf32": True,
            "json_backend": "orjson",
            "mask_approx": "none",
            "async_writer": True,
            "disable_mask_iou_head": True,
            "runtime_script": str(ROOT / "backend/detectors/codino/runtime/codino_video_fast_runtime.py"),
            "config": str(ROOT / "checkpoints/codino/detector/resolved_config.py"),
            "checkpoint": str(ROOT / "checkpoints/codino/detector/epoch_2.pth"),
            "classifier_checkpoint": str(ROOT / "checkpoints/codino/classifier/best.pt"),
            **trt_paths,
            "trt_extra_site_packages": None if trt_site is None else str(trt_site),
            "trt_query_encoder_shapes": "184x320,92x160,46x80,23x40,12x20",
        },
        "rtdetr": {
            "batch_size": selected_rtdetr_batch,
            "device": os.environ.get("RTDETR_DEVICE", "cuda:0"),
            "repo": str(DEFAULT_RTDETR_REPO),
            "config": str(DEFAULT_RTDETR_CONFIG),
            "checkpoint": str(DEFAULT_RTDETR_CHECKPOINT),
            "progress_interval": int(os.environ.get("RTDETR_PROGRESS_INTERVAL", "30")),
            "conf_thr": float(os.environ.get("RTDETR_CONF_THR", "0.50")),
            "low_thr": float(os.environ.get("RTDETR_LOW_THR", "0.15")),
            "new_track_thr": float(os.environ.get("RTDETR_NEW_TRACK_THR", "0.55")),
            "track_min_hits": int(os.environ.get("RTDETR_TRACK_MIN_HITS", "5")),
            "nms_iou_thr": float(os.environ.get("RTDETR_NMS_IOU_THR", "0.55")),
        },
        "postprocess": {
            "k2_device": "auto",
            "polygon_predictor_device": "auto",
            "overlay_encoder": "nvenc",
        },
        "notes": notes,
    }


def load_benchmark(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None


def apply_benchmark_recommendations(recs: dict[str, Any], benchmark: dict[str, Any] | None) -> None:
    if not benchmark:
        return
    selected = benchmark.get("selected")
    if not isinstance(selected, dict):
        return
    for detector, section in (("dinov3", "dinov3"), ("eva02", "eva02"), ("codino", "codino"), ("rtdetr", "rtdetr")):
        item = selected.get(detector)
        if not isinstance(item, dict):
            continue
        try:
            batch_size = int(item.get("batch_size"))
        except Exception:
            continue
        if batch_size <= 0:
            continue
        recs.setdefault(section, {})["batch_size"] = batch_size
        if detector == "codino":
            recs.setdefault(section, {}).update(_codino_default_paths_for_batch(batch_size))
        recs.setdefault("notes", []).append(
            f"{detector} batch-size was selected by setup benchmark: batch={batch_size}, "
            f"metric_fps={float(item.get('metric_fps') or 0):.4f}."
        )


def build_profile(benchmark_path: Path = DEFAULT_BENCHMARK) -> dict[str, Any]:
    torch_data = torch_info()
    smi_data = nvidia_smi_info()
    trt_data = tensorrt_info()
    engine_exists = DEFAULT_TRT_ENGINE.is_file()
    total_mib = choose_vram_mib(torch_data, smi_data)
    benchmark = load_benchmark(benchmark_path)
    recs = recommendations(
        total_mib,
        tensorrt_available=bool(trt_data.get("available")),
        engine_exists=engine_exists,
        benchmark=benchmark,
    )
    apply_benchmark_recommendations(recs, benchmark)
    profile = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "python": sys.executable,
        "runtime": {
            "python": sys.executable,
            "venv": sys.prefix,
            "dinov3_runtime_profile": str(DEFAULT_PROFILE),
        },
        "platform": platform.platform(),
        "gpu": {
            "torch": torch_data,
            "nvidia_smi": smi_data,
        },
        "dependencies": {
            "tensorrt": trt_data,
        },
        "tensorrt_engine": {
            "path": str(DEFAULT_TRT_ENGINE),
            "exists": engine_exists,
            "size_bytes": DEFAULT_TRT_ENGINE.stat().st_size if engine_exists else 0,
        },
        "batch_benchmark": {
            "path": str(benchmark_path),
            "exists": bool(benchmark),
            "selected": benchmark.get("selected", {}) if isinstance(benchmark, dict) else {},
        },
        "recommendations": recs,
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Write local runtime tuning profile for GUI/integrated pipeline")
    parser.add_argument("--output", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--print", dest="print_only", action="store_true", help="Print profile without writing")
    args = parser.parse_args()

    profile = build_profile(args.benchmark.expanduser().resolve())
    output = args.output.expanduser().resolve()
    profile.setdefault("runtime", {})["dinov3_runtime_profile"] = str(output)
    text = json.dumps(profile, ensure_ascii=False, indent=2)
    if args.print_only:
        print(text)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    rec = profile["recommendations"]
    print(f"[PROFILE] wrote {output}")
    print(
        "[PROFILE] recommended batches: "
        f"dinov3={rec['dinov3']['batch_size']} "
        f"eva02={rec['eva02']['batch_size']} "
        f"codino={rec['codino']['batch_size']} "
        f"rtdetr={rec['rtdetr']['batch_size']} "
        f"eva02_classifier={rec['eva02']['classifier_batch_size']}"
    )
    for note in rec.get("notes", []):
        print(f"[PROFILE] note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
