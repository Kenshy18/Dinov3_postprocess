"""Small TensorRT setup helpers for Co-DINO export/build scripts."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def default_tensorrt_site_packages() -> Path | None:
    raw = os.environ.get("TENSORRT_SITE_PACKAGES")
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    for raw_path in sys.path:
        path = Path(raw_path)
        if (path / "tensorrt").exists():
            return path
    home = Path.home()
    for env_name in ("eva02_trt", "trt_env"):
        for pyver in ("python3.10", "python3.11", "python3.12", "python3.8"):
            path = home / "miniconda3" / "envs" / env_name / "lib" / pyver / "site-packages"
            if (path / "tensorrt").exists():
                return path
    return None


def prepare_tensorrt(extra_site_packages: Path | None) -> None:
    if extra_site_packages is not None and extra_site_packages.exists():
        extra_site = str(extra_site_packages)
        if extra_site not in sys.path:
            sys.path.append(extra_site)
        libs = extra_site_packages / "tensorrt_libs"
        if libs.exists():
            os.environ["LD_LIBRARY_PATH"] = f"{libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            for name in ("libnvinfer.so.10", "libnvonnxparser.so.10", "libnvinfer_plugin.so.10"):
                path = libs / name
                if path.exists():
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            plugin = libs / "libnvinfer_plugin.so.10"
            vc_plugin = libs / "libnvinfer_vc_plugin.so.10"
            if plugin.exists() and not vc_plugin.exists():
                shim_dir = Path("/tmp/trt_vc_plugin_shim")
                shim_dir.mkdir(parents=True, exist_ok=True)
                shim = shim_dir / "libnvinfer_vc_plugin.so.10"
                if not shim.exists():
                    shim.symlink_to(plugin)
                os.environ["LD_LIBRARY_PATH"] = f"{shim_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)
