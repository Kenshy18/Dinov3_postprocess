#!/usr/bin/env python3
"""Benchmark detector batch sizes during local runtime setup.

The script creates a temporary synthetic video, runs candidate batch sizes in
separate processes, records throughput/VRAM/error details, and removes the
temporary video/output tree by default. The resulting JSON is local machine
state and is intended to feed tools/setup/configure_runtime_profile.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".runtime" / "runtime_benchmark.json"
DEFAULT_WORK_DIR = ROOT / "output" / "runtime_batch_benchmark"
DEFAULT_TRT_ENGINE = ROOT / "checkpoints" / "trt" / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"
DEFAULT_CODINO_TRT_DIR = ROOT / "checkpoints" / "codino" / "trt"
DEFAULT_CODINO_TRT_BACKBONE = DEFAULT_CODINO_TRT_DIR / "codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine"
DEFAULT_CODINO_TRT_QUERY_ENCODER = DEFAULT_CODINO_TRT_DIR / "codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine"
DEFAULT_CODINO_TRT_DECODER = DEFAULT_CODINO_TRT_DIR / "codino_decoder_b2_736x1280_msda_plugin_fp16.engine"
DEFAULT_CODINO_TRT_MASK_HEAD = DEFAULT_CODINO_TRT_DIR / "codino_mask_head_core_n1_736x1280_fp16.engine"


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def nvidia_smi() -> dict[str, Any]:
    text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not text:
        return {"available": False}
    parts = [part.strip() for part in text.splitlines()[0].split(",")]
    if len(parts) < 5:
        return {"available": False, "raw": text}
    out: dict[str, Any] = {
        "available": True,
        "name": parts[0],
        "driver_version": parts[4],
    }
    for key, value in (("memory_total_mib", parts[1]), ("memory_used_mib", parts[2]), ("memory_free_mib", parts[3])):
        try:
            out[key] = int(float(value))
        except ValueError:
            out[key] = None
    return out


def current_gpu_used_mib() -> int | None:
    info = nvidia_smi()
    value = info.get("memory_used_mib")
    return int(value) if isinstance(value, int) else None


def torch_gpu_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "import_error": repr(exc)}
    info: dict[str, Any] = {
        "available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
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
    return info


def choose_total_vram_mib(torch_info: dict[str, Any], smi_info: dict[str, Any]) -> int | None:
    for source in (torch_info, smi_info):
        value = source.get("memory_total_mib")
        if isinstance(value, int) and value > 0:
            return value
    return None


def parse_candidates(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for item in re.split(r"[,:\s]+", raw.strip()):
        if not item:
            continue
        value = int(item)
        if value > 0 and value not in out:
            out.append(value)
    return out


def auto_candidates(detector: str, total_mib: int | None) -> list[int]:
    if detector == "codino":
        try:
            return [max(1, int(os.environ.get("CODINO_TRT_BATCH_SIZE", "2")))]
        except ValueError:
            return [2]
    total_gib = (total_mib or 0) / 1024.0
    if detector == "eva02":
        if total_gib <= 0:
            return [1]
        if total_gib < 10:
            return [1, 2]
        if total_gib < 16:
            return [1, 2, 3, 4]
        if total_gib < 24:
            return [1, 2, 3, 4, 6]
        return [1, 2, 4, 6, 8]
    if total_gib <= 0:
        return [1]
    if total_gib < 10:
        return [1, 2, 4]
    if total_gib < 16:
        return [2, 4, 6, 8]
    return [4, 6, 8]


def create_dummy_video(path: Path, *, width: int, height: int, frames: int, fps: float) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create dummy video: {path}")
    x_grad = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y_grad = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    try:
        for index in range(frames):
            frame = np.empty((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (x_grad + index * 3) % 255
            frame[:, :, 1] = (y_grad + index * 5) % 255
            frame[:, :, 2] = ((x_grad // 2) + (y_grad // 2) + index * 7) % 255
            cx = int((index * 31) % width)
            cy = int(height * 0.5 + np.sin(index / 6.0) * height * 0.22)
            cv2.ellipse(frame, (cx, cy), (max(24, width // 12), max(18, height // 10)), index * 9, 0, 360, (245, 245, 245), -1)
            cv2.rectangle(
                frame,
                (max(0, width - 220 - index * 4 % width), max(0, height // 5)),
                (min(width - 1, width - 40), min(height - 1, height // 5 + 140)),
                (40, 180, 250),
                -1,
            )
            writer.write(frame)
    finally:
        writer.release()


def count_jsonl_lines(output_dir: Path) -> int:
    total = 0
    seen: set[Path] = set()
    for path in output_dir.rglob("*.jsonl"):
        try:
            real_path = path.resolve()
            if real_path in seen:
                continue
            seen.add(real_path)
            with path.open("rb") as handle:
                total += sum(1 for _ in handle)
        except OSError:
            pass
    return total


def parse_measured_fps(text: str, summary_path: Path | None) -> float | None:
    if summary_path is not None and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runs = summary.get("runs") or []
            if runs and isinstance(runs[0], dict):
                for key in ("measured_fps", "compute_fps", "e2e_fps"):
                    value = runs[0].get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        return float(value)
        except Exception:
            pass
    matches = re.findall(r"\[MEASURE\]\s+e2e_fps=([0-9.]+)", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    matches = re.findall(r"fps_total=([0-9.]+)", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def classify_failure(text: str, returncode: int, timed_out: bool) -> str | None:
    lower = text.lower()
    if timed_out:
        return "timeout"
    if returncode == 0:
        return None
    if "out of memory" in lower or "cuda error: out of memory" in lower or "cudnn_status_alloc_failed" in lower:
        return "cuda_oom"
    if "failed to deserialize" in lower or "tensorrt" in lower and "error" in lower:
        return "runtime_engine_error"
    return "process_failed"


def command_for_candidate(
    *,
    detector: str,
    python: Path,
    input_video: Path,
    output_dir: Path,
    batch: int,
    frames: int,
    engine: Path,
    classifier_batch_size: int,
    eva02_compile_backbone: str,
) -> list[str]:
    if detector == "eva02":
        return [
            str(python),
            str(ROOT / "backend" / "detectors" / "eva02" / "runtime" / "infer_video_eva02_jsonl.py"),
            "--input",
            str(input_video),
            "--output",
            str(output_dir),
            "--classifier",
            "--target-size",
            "1280",
            "--batch-size",
            str(batch),
            "--warmup-frames",
            "0",
            "--classifier-batch-size",
            str(classifier_batch_size),
            "--max-frames",
            str(frames),
            "--json-backend",
            "orjson",
            "--async-writer",
            "--mask-approx",
            "simple",
            "--compile-backbone",
            str(eva02_compile_backbone),
            "--overwrite",
        ]
    if detector == "codino":
        return [
            str(python),
            str(ROOT / "backend" / "detectors" / "codino" / "runtime" / "infer_video_codino_jsonl.py"),
            "--input",
            str(input_video),
            "--output",
            str(output_dir),
            "--classifier",
            "--target-size",
            "1280x720",
            "--batch-size",
            str(batch),
            "--warmup-frames",
            "0",
            "--max-frames",
            str(frames),
            "--score-thresh",
            "0.30",
            "--model-score-thr",
            "0.05",
            "--amp",
            "fp16",
            "--json-backend",
            "orjson",
            "--mask-approx",
            "none",
            "--async-writer",
            "--disable-mask-iou-head",
            "--trt-backbone-engine",
            str(Path(os.environ.get("CODINO_TRT_BACKBONE_ENGINE", DEFAULT_CODINO_TRT_BACKBONE))),
            "--trt-query-encoder-engine",
            str(Path(os.environ.get("CODINO_TRT_QUERY_ENCODER_ENGINE", DEFAULT_CODINO_TRT_QUERY_ENCODER))),
            "--trt-decoder-engine",
            str(Path(os.environ.get("CODINO_TRT_DECODER_ENGINE", DEFAULT_CODINO_TRT_DECODER))),
            "--trt-mask-head-engine",
            str(Path(os.environ.get("CODINO_TRT_MASK_HEAD_ENGINE", DEFAULT_CODINO_TRT_MASK_HEAD))),
            "--overwrite",
        ]
    return [
        str(python),
        str(ROOT / "backend" / "detectors" / "dinov3" / "runtime" / "infer_video_dinov3_jsonl.py"),
        "--input",
        str(input_video),
        "--output",
        str(output_dir),
        "--classifier",
        "--checkpoint",
        str(ROOT / "checkpoints" / "detector" / "model_final.pth"),
        "--backbone-weights",
        str(ROOT / "checkpoints" / "dinov3" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        "--classifier-checkpoint",
        str(ROOT / "checkpoints" / "classifier" / "best.pt"),
        "--trt-backbone-engine",
        str(engine),
        "--target-size",
        "1280x720",
        "--batch-size",
        str(batch),
        "--warmup-frames",
        "0",
        "--max-frames",
        str(frames),
        "--json-backend",
        "orjson",
        "--mask-approx",
        "simple",
        "--async-writer",
        "--overwrite",
    ]


def run_candidate(command: list[str], *, output_dir: Path, timeout_sec: int, poll_sec: float) -> dict[str, Any]:
    start_used = current_gpu_used_mib()
    peak_used = start_used
    start = time.perf_counter()
    lines: list[str] = []
    timed_out = False
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            for key, _ in selector.select(timeout=poll_sec):
                line = key.fileobj.readline()
                if line:
                    lines.append(line.rstrip())
            used = current_gpu_used_mib()
            if used is not None:
                peak_used = used if peak_used is None else max(peak_used, used)
            if process.poll() is not None:
                rest = process.stdout.read()
                if rest:
                    lines.extend(part.rstrip() for part in rest.splitlines())
                break
            if time.perf_counter() - start > timeout_sec:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        process.kill()
                break
    finally:
        selector.close()
    elapsed = time.perf_counter() - start
    returncode = int(process.returncode if process.returncode is not None else -9)
    text = "\n".join(lines)
    summary_path = output_dir / "summary.json"
    jsonl_lines = count_jsonl_lines(output_dir)
    measured_fps = parse_measured_fps(text, summary_path)
    wall_fps = jsonl_lines / elapsed if elapsed > 0 and jsonl_lines > 0 else 0.0
    metric_fps = float(measured_fps if measured_fps and measured_fps > 0 else wall_fps)
    failure = classify_failure(text, returncode, timed_out)
    success = bool(returncode == 0 and jsonl_lines > 0 and not timed_out)
    return {
        "command": command,
        "returncode": returncode,
        "success": success,
        "failure": failure,
        "elapsed_sec": elapsed,
        "jsonl_lines": jsonl_lines,
        "measured_fps": measured_fps,
        "wall_fps": wall_fps,
        "metric_fps": metric_fps,
        "gpu_memory_used_start_mib": start_used,
        "gpu_memory_used_peak_mib": peak_used,
        "stdout_tail": "\n".join(lines[-120:]),
    }


def benchmark_detector(
    *,
    detector: str,
    candidates: list[int],
    python: Path,
    input_video: Path,
    temp_dir: Path,
    frames: int,
    engine: Path,
    classifier_batch_size: int,
    timeout_sec: int,
    poll_sec: float,
    eva02_compile_backbone: str,
) -> dict[str, Any]:
    results = []
    selected: dict[str, Any] | None = None
    for batch in candidates:
        output_dir = temp_dir / detector / f"batch_{batch}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        command = command_for_candidate(
            detector=detector,
            python=python,
            input_video=input_video,
            output_dir=output_dir,
            batch=batch,
            frames=frames,
            engine=engine,
            classifier_batch_size=classifier_batch_size,
            eva02_compile_backbone=eva02_compile_backbone,
        )
        print(f"[BENCH] {detector} batch={batch}", flush=True)
        result = run_candidate(command, output_dir=output_dir, timeout_sec=timeout_sec, poll_sec=poll_sec)
        result["batch_size"] = batch
        results.append(result)
        if result["success"]:
            if selected is None or float(result["metric_fps"]) > float(selected["metric_fps"]):
                selected = {
                    "batch_size": batch,
                    "metric_fps": float(result["metric_fps"]),
                    "measured_fps": result.get("measured_fps"),
                    "wall_fps": float(result["wall_fps"]),
                    "gpu_memory_used_peak_mib": result.get("gpu_memory_used_peak_mib"),
                    "source": "benchmark",
                }
            print(
                f"[BENCH] ok {detector} batch={batch} metric_fps={float(result['metric_fps']):.4f} "
                f"peak_mib={result.get('gpu_memory_used_peak_mib')}",
                flush=True,
            )
        else:
            print(f"[BENCH] fail {detector} batch={batch} reason={result.get('failure')}", flush=True)
            if result.get("failure") in {"cuda_oom", "timeout"}:
                print(f"[BENCH] stop higher {detector} batches after {result.get('failure')}", flush=True)
                break
    return {
        "candidates": candidates,
        "results": results,
        "selected": selected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local detector batch sizes using a temporary dummy video")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--detectors", default=os.environ.get("BATCH_BENCHMARK_DETECTORS", "eva02"))
    parser.add_argument("--frames", type=int, default=int(os.environ.get("BATCH_BENCHMARK_FRAMES", "24")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("BATCH_BENCHMARK_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("BATCH_BENCHMARK_HEIGHT", "720")))
    parser.add_argument("--fps", type=float, default=float(os.environ.get("BATCH_BENCHMARK_FPS", "30")))
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("BATCH_BENCHMARK_TIMEOUT_SEC", "360")))
    parser.add_argument("--poll-sec", type=float, default=0.25)
    parser.add_argument("--eva02-candidates", default=os.environ.get("EVA02_BATCH_CANDIDATES", ""))
    parser.add_argument("--dinov3-candidates", default=os.environ.get("DINOV3_BATCH_CANDIDATES", ""))
    parser.add_argument("--codino-candidates", default=os.environ.get("CODINO_BATCH_CANDIDATES", ""))
    parser.add_argument("--classifier-batch-size", type=int, default=int(os.environ.get("EVA02_CLASSIFIER_BATCH_SIZE", "1024")))
    parser.add_argument("--engine", type=Path, default=Path(os.environ.get("DINOV3_TRT_BACKBONE_ENGINE", DEFAULT_TRT_ENGINE)))
    parser.add_argument("--eva02-compile-backbone", default=os.environ.get("EVA02_BENCHMARK_COMPILE_BACKBONE", "none"))
    parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    python = args.python.expanduser()
    if not python.is_absolute():
        python = ROOT / python
    output = args.output.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    engine = args.engine.expanduser().resolve()
    detectors = [item.strip().lower() for item in args.detectors.split(",") if item.strip()]
    detectors = [item for item in detectors if item in {"eva02", "dinov3", "codino"}]
    if not detectors:
        detectors = ["eva02"]

    torch_info = torch_gpu_info()
    smi_info = nvidia_smi()
    total_mib = choose_total_vram_mib(torch_info, smi_info)
    temp_dir = work_dir / ("tmp_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    dummy_video = temp_dir / "input" / "runtime_batch_probe.mp4"
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": str(python),
        "settings": {
            "detectors": detectors,
            "frames": int(args.frames),
            "width": int(args.width),
            "height": int(args.height),
            "fps": float(args.fps),
            "timeout_sec": int(args.timeout_sec),
            "cleanup": bool(args.cleanup),
            "eva02_compile_backbone": str(args.eva02_compile_backbone),
        },
        "gpu": {
            "torch": torch_info,
            "nvidia_smi": smi_info,
            "total_vram_mib": total_mib,
        },
        "dummy_video": {
            "path": str(dummy_video),
            "created": False,
            "deleted": False,
        },
        "detectors": {},
        "selected": {},
    }

    try:
        create_dummy_video(dummy_video, width=int(args.width), height=int(args.height), frames=int(args.frames), fps=float(args.fps))
        result["dummy_video"]["created"] = True
        for detector in detectors:
            explicit_raw = {
                "eva02": args.eva02_candidates,
                "dinov3": args.dinov3_candidates,
                "codino": args.codino_candidates,
            }[detector]
            explicit = parse_candidates(explicit_raw)
            candidates = explicit or auto_candidates(detector, total_mib)
            detector_result = benchmark_detector(
                detector=detector,
                candidates=candidates,
                python=python,
                input_video=dummy_video,
                temp_dir=temp_dir,
                frames=int(args.frames),
                engine=engine,
                classifier_batch_size=int(args.classifier_batch_size),
                timeout_sec=int(args.timeout_sec),
                poll_sec=float(args.poll_sec),
                eva02_compile_backbone=str(args.eva02_compile_backbone),
            )
            result["detectors"][detector] = detector_result
            if detector_result.get("selected"):
                result["selected"][detector] = detector_result["selected"]
    finally:
        if args.cleanup and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            result["dummy_video"]["deleted"] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[BENCH] wrote {output}")
    for detector, selected in result.get("selected", {}).items():
        print(
            f"[BENCH] selected {detector} batch={selected.get('batch_size')} "
            f"metric_fps={float(selected.get('metric_fps') or 0):.4f}"
        )
    if not result.get("selected"):
        print("[BENCH] no successful benchmark candidate; profile will keep conservative defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
