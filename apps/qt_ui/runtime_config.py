from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_STATE_DIR = ROOT / ".runtime"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_RAW_REMOVE_SHORT_TRACKS_MAX_FRAMES = 10
DEFAULT_OVERLAY_ENCODER = "nvenc"
GUI_RUNTIME_ENV = Path(os.environ.get("GUI_RUNTIME_ENV", RUNTIME_STATE_DIR / "gui_runtime.env"))
LEGACY_GUI_RUNTIME_ENV = ROOT / "configs" / "gui_runtime.env"
DEFAULT_RUNTIME_PROFILE = RUNTIME_STATE_DIR / "runtime_profile.json"
LEGACY_RUNTIME_PROFILE = ROOT / "configs" / "runtime_profile.json"
DEFAULT_BATCH_BENCHMARK = RUNTIME_STATE_DIR / "runtime_benchmark.json"
LEGACY_BATCH_BENCHMARK = ROOT / "configs" / "runtime_benchmark.json"
FALLBACK_TRT_ENGINE = ROOT / "checkpoints" / "trt" / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"


def ensure_repo_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _valid_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def load_gui_runtime_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = GUI_RUNTIME_ENV if GUI_RUNTIME_ENV.is_file() else LEGACY_GUI_RUNTIME_ENV
    if not env_path.is_file():
        return values
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError:
                tokens = [line]
            if not tokens:
                continue
            key, value = tokens[0].split("=", 1)
            values[key] = value
    except Exception:
        return {}
    return values


def runtime_profile_path() -> Path:
    runtime_env = load_gui_runtime_env()
    raw = os.environ.get("DINOV3_RUNTIME_PROFILE") or runtime_env.get("DINOV3_RUNTIME_PROFILE")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return DEFAULT_RUNTIME_PROFILE if DEFAULT_RUNTIME_PROFILE.is_file() else LEGACY_RUNTIME_PROFILE


def load_runtime_profile() -> dict:
    try:
        path = runtime_profile_path()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def default_python() -> Path:
    runtime_env = load_gui_runtime_env()
    profile = load_runtime_profile()
    candidates = [
        os.environ.get("GUI_RUNTIME_PYTHON"),
        runtime_env.get("GUI_RUNTIME_PYTHON"),
        profile.get("runtime", {}).get("python") if isinstance(profile.get("runtime"), dict) else None,
        profile.get("python"),
        str(ROOT / ".venv_integrated" / "bin" / "python"),
        sys.executable,
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if _valid_executable(path):
            return path
    integrated = ROOT / ".venv_integrated" / "bin" / "python"
    return integrated if integrated.is_file() else Path(sys.executable)


def profile_recommendations() -> dict:
    recommendations = load_runtime_profile().get("recommendations", {})
    return recommendations if isinstance(recommendations, dict) else {}


def profile_int(section: str, key: str) -> int | None:
    try:
        value = profile_recommendations().get(section, {}).get(key)
        return int(value) if value is not None else None
    except Exception:
        return None


def profile_path(section: str, key: str) -> Path | None:
    try:
        value = profile_recommendations().get(section, {}).get(key)
        if not value:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else ROOT / path
    except Exception:
        return None


def selected_trt_engine() -> Path:
    runtime_env = load_gui_runtime_env()
    raw = os.environ.get("DINOV3_TRT_BACKBONE_ENGINE") or runtime_env.get("DINOV3_TRT_BACKBONE_ENGINE")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return profile_path("dinov3", "trt_backbone_engine") or FALLBACK_TRT_ENGINE


def runtime_summary_text() -> str:
    profile = load_runtime_profile()
    rec = profile.get("recommendations", {})
    rec = rec if isinstance(rec, dict) else {}
    benchmark = profile.get("batch_benchmark", {})
    benchmark_exists = bool(benchmark.get("exists")) if isinstance(benchmark, dict) else False
    dinov3 = rec.get("dinov3", {}) if isinstance(rec.get("dinov3"), dict) else {}
    eva02 = rec.get("eva02", {}) if isinstance(rec.get("eva02"), dict) else {}
    codino = rec.get("codino", {}) if isinstance(rec.get("codino"), dict) else {}
    engine = selected_trt_engine()
    return (
        f"python={default_python()} | "
        f"profile={runtime_profile_path()} | "
        f"benchmark={'ok' if benchmark_exists else 'none'} | "
        f"DINOv3 batch={dinov3.get('batch_size', '既定')} | "
        f"EVA02 batch={eva02.get('batch_size', '既定')} | "
        f"Co-DINO batch={codino.get('batch_size', '既定')} | "
        f"EVA02 cls={eva02.get('classifier_batch_size', '既定')} | "
        f"engine={'ok' if engine.is_file() else 'missing'}"
    )


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def clean_run_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "video"


def as_command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)
