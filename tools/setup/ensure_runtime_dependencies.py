#!/usr/bin/env python3
"""Install/check fragile GPU runtime dependencies with explicit diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import site
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path


DEFAULT_ADA_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
DEFAULT_BLACKWELL_TORCH_INDEX = "https://download.pytorch.org/whl/nightly/cu129"
DEFAULT_ADA_TORCH_SPEC = "torch==2.1.2"
DEFAULT_ADA_TORCHVISION_SPEC = "torchvision==0.16.2"
DEFAULT_MMCV_FULL_VERSION = "1.7.2"
DEFAULT_ADA_MMCV_FIND_LINKS = "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
OPENMMLAB_COMPAT_VERSION_PINS = [
    ("numpy", "1.26.4"),
    ("opencv-python", "4.8.1.78"),
    ("setuptools", "60.2.0"),
    ("requests", "2.28.2"),
    ("charset-normalizer", "3.3.2"),
    ("idna", "3.4"),
    ("urllib3", "1.26.13"),
    ("certifi", "2022.12.7"),
    ("filelock", "3.14.0"),
]
OPENMMLAB_COMPAT_PACKAGES = [f"{name}=={version}" for name, version in OPENMMLAB_COMPAT_VERSION_PINS]
TORCH_LOCAL_CLEAN_PATTERNS = [
    "torch",
    "torch-*.dist-info",
    "torch.libs",
    "torchvision",
    "torchvision-*.dist-info",
    "torchvision.libs",
    "functorch",
    "torchgen",
    "triton",
    "triton-*.dist-info",
]
OPENMMLAB_COMPAT_LOCAL_CLEAN_PATTERNS = [
    "numpy",
    "numpy-*.dist-info",
    "numpy.libs",
    "cv2",
    "opencv_python-*.dist-info",
    "opencv_python.libs",
    "requests",
    "requests-*.dist-info",
    "charset_normalizer",
    "charset_normalizer-*.dist-info",
    "idna",
    "idna-*.dist-info",
    "urllib3",
    "urllib3-*.dist-info",
    "certifi",
    "certifi-*.dist-info",
    "filelock",
    "filelock-*.dist-info",
]


def log(message: str) -> None:
    print(f"[SETUP] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def fail(message: str, code: int = 2) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def run(command: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    log(shlex.join(command))
    completed = subprocess.run(command, env=env, text=True)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed


def local_site_package_dirs() -> list[Path]:
    """Return only site-packages owned by this Python prefix."""
    prefix = Path(sys.prefix).resolve()
    dirs: list[Path] = []
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    for raw in candidates:
        path = Path(raw).resolve()
        if not path.is_dir():
            continue
        try:
            inside_prefix = path.is_relative_to(prefix)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            inside_prefix = str(path).startswith(str(prefix))
        if inside_prefix and path not in dirs:
            dirs.append(path)
    return dirs


def remove_local_package_artifacts(patterns: list[str]) -> None:
    for site_dir in local_site_package_dirs():
        for pattern in patterns:
            for path in site_dir.glob(pattern):
                resolved = path.resolve()
                try:
                    inside_site = resolved.is_relative_to(site_dir)
                except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
                    inside_site = str(resolved).startswith(str(site_dir))
                if not inside_site:
                    continue
                log(f"removing stale package artifact: {path}")
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)


def normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def local_dist_info_paths(distribution_name: str) -> list[Path]:
    normalized = normalized_distribution_name(distribution_name)
    paths: list[Path] = []
    for site_dir in local_site_package_dirs():
        paths.extend(site_dir.glob(f"{normalized}-*.dist-info"))
    return paths


def openmmlab_compat_packages_ok() -> bool:
    for name, expected_version in OPENMMLAB_COMPAT_VERSION_PINS:
        try:
            actual_version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return False
        if actual_version != expected_version:
            return False
        if name != "setuptools" and len(local_dist_info_paths(name)) > 1:
            return False
    return True


def install_openmmlab_compat_packages() -> None:
    if env_bool("SKIP_OPENMMLAB_COMPAT_PINS", "0"):
        log("OpenMMLab compatibility pins skipped by SKIP_OPENMMLAB_COMPAT_PINS")
        return
    if openmmlab_compat_packages_ok():
        log("OpenMMLab compatibility pins already satisfied")
        return
    remove_local_package_artifacts(OPENMMLAB_COMPAT_LOCAL_CLEAN_PATTERNS)
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *OPENMMLAB_COMPAT_PACKAGES])
    if not openmmlab_compat_packages_ok():
        fail("OpenMMLab compatibility package pins did not settle after install.")


def env_bool(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def env_disabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"0", "false", "no", "off"}


@dataclass
class TorchInfo:
    import_ok: bool = False
    version: str | None = None
    cuda_version: str | None = None
    cuda_available: bool = False
    cuda_usable: bool = False
    gpu_name: str | None = None
    capability: str | None = None
    error: str | None = None


def probe_torch() -> TorchInfo:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host env
        return TorchInfo(error=repr(exc))
    info = TorchInfo(
        import_ok=True,
        version=getattr(torch, "__version__", None),
        cuda_version=getattr(torch.version, "cuda", None),
        cuda_available=bool(torch.cuda.is_available()),
    )
    if info.cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            info.gpu_name = props.name
            info.capability = f"{props.major}.{props.minor}"
        except Exception as exc:  # pragma: no cover - depends on host env
            info.error = repr(exc)
        try:
            tensor = torch.ones(1, device="cuda")
            result = (tensor + 1).sum()
            torch.cuda.synchronize()
            info.cuda_usable = bool(float(result.item()) == 2.0)
        except Exception as exc:  # pragma: no cover - depends on host env
            info.cuda_usable = False
            info.error = repr(exc)
    return info


def probe_torch_subprocess() -> TorchInfo:
    code = r"""
import json
try:
    import torch
    info = {
        "import_ok": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_usable": False,
        "gpu_name": None,
        "capability": None,
        "error": None,
    }
    if info["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["capability"] = f"{props.major}.{props.minor}"
        except Exception as exc:
            info["error"] = repr(exc)
        try:
            tensor = torch.ones(1, device="cuda")
            result = (tensor + 1).sum()
            torch.cuda.synchronize()
            info["cuda_usable"] = bool(float(result.item()) == 2.0)
        except Exception as exc:
            info["cuda_usable"] = False
            info["error"] = repr(exc)
except Exception as exc:
    info = {"import_ok": False, "error": repr(exc)}
print(json.dumps(info))
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return TorchInfo(error=detail or f"probe failed with exit code {completed.returncode}")
    try:
        data = json.loads((completed.stdout or "").strip().splitlines()[-1])
    except Exception as exc:
        return TorchInfo(error=f"could not parse torch probe output: {exc!r}")
    return TorchInfo(**data)


def nvidia_smi_info() -> dict[str, str]:
    if shutil.which("nvidia-smi") is None:
        return {}
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return {}
    first = (completed.stdout or "").splitlines()[0:1]
    if not first:
        return {}
    parts = [part.strip() for part in first[0].split(",")]
    if len(parts) < 4:
        return {}
    return {
        "gpu_name": parts[0],
        "capability": parts[1],
        "driver": parts[2],
        "memory_mib": parts[3],
    }


def capability_major(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?", value)
    return int(match.group(1)) if match else None


def diagnose() -> None:
    torch_info = probe_torch()
    smi = nvidia_smi_info()
    log(f"python={sys.version.split()[0]} executable={sys.executable}")
    log(f"platform={platform.platform()}")
    if shutil.which("nvcc"):
        completed = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        last_line = (completed.stdout or completed.stderr).strip().splitlines()[-1:]
        log(f"nvcc={last_line[0] if last_line else 'available'}")
    else:
        log("nvcc=missing")
    if shutil.which("gcc"):
        completed = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        first_line = (completed.stdout or completed.stderr).strip().splitlines()[0:1]
        log(f"gcc={first_line[0] if first_line else 'available'}")
    else:
        log("gcc=missing")
    if smi:
        log(
            "nvidia-smi="
            f"gpu={smi.get('gpu_name')} compute={smi.get('capability')} "
            f"driver={smi.get('driver')} memory_mib={smi.get('memory_mib')}"
        )
    else:
        log("nvidia-smi=unavailable")
    if torch_info.import_ok:
        log(
            "torch="
            f"{torch_info.version} torch_cuda={torch_info.cuda_version} "
            f"cuda_available={torch_info.cuda_available} cuda_usable={torch_info.cuda_usable} "
            f"gpu={torch_info.gpu_name or '-'} compute={torch_info.capability or '-'}"
        )
        if torch_info.cuda_available and not torch_info.cuda_usable:
            warn(f"torch CUDA is visible but not usable: {torch_info.error}")
    else:
        warn(f"torch import failed: {torch_info.error}")


def install_torch(*, force: bool = False) -> None:
    mode = os.environ.get("INSTALL_TORCH", "auto").strip().lower()
    torch_info = probe_torch_subprocess()
    require_cuda = env_bool("REQUIRE_TORCH_CUDA", "1")
    if not force and torch_info.import_ok and (torch_info.cuda_usable or not require_cuda):
        log(
            "torch already usable: "
            f"version={torch_info.version} cuda={torch_info.cuda_version} "
            f"gpu={torch_info.gpu_name or '-'} compute={torch_info.capability or '-'}"
        )
        return
    if mode in {"0", "false", "no", "off", "skip"}:
        fail("torch/torchvision are not usable and INSTALL_TORCH is disabled.")

    smi = nvidia_smi_info()
    major = capability_major(torch_info.capability) or capability_major(smi.get("capability"))
    blackwell = major is not None and major >= 12
    torch_spec = os.environ.get("TORCH_PIP_SPEC", "auto").strip()
    vision_spec = os.environ.get("TORCHVISION_PIP_SPEC", "auto").strip()
    index_url = os.environ.get("TORCH_INDEX_URL", "auto").strip()
    extra_args = shlex.split(os.environ.get("TORCH_PIP_EXTRA_ARGS", ""))

    if torch_spec == "auto":
        if blackwell:
            package_args = ["--pre", "torch", "torchvision"]
        else:
            package_args = [DEFAULT_ADA_TORCH_SPEC, DEFAULT_ADA_TORCHVISION_SPEC]
    else:
        package_args = shlex.split(torch_spec)
        if vision_spec and vision_spec != "auto":
            package_args.extend(shlex.split(vision_spec))
        elif vision_spec == "auto" and not any(part.startswith("torchvision") for part in package_args):
            package_args.append("torchvision")

    if index_url == "auto":
        index_url = DEFAULT_BLACKWELL_TORCH_INDEX if blackwell else DEFAULT_ADA_TORCH_INDEX

    log(
        "installing torch runtime: "
        f"profile={'blackwell-nightly' if blackwell else 'stable-cu121'} "
        f"index={index_url} packages={' '.join(package_args)}"
    )
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "wheel"])
    if force or (torch_info.import_ok and require_cuda and not torch_info.cuda_usable):
        remove_local_package_artifacts(TORCH_LOCAL_CLEAN_PATTERNS)
    command = [sys.executable, "-m", "pip", "install", *package_args, "--index-url", index_url, *extra_args]
    run(command)
    verified = probe_torch_subprocess()
    if not verified.import_ok or (require_cuda and not verified.cuda_usable):
        fail(
            "torch install completed but CUDA torch is not usable. "
            f"torch_error={verified.error!r} cuda_available={verified.cuda_available} cuda_usable={verified.cuda_usable}. "
            "Set TORCH_PIP_SPEC/TORCH_INDEX_URL or pass REFERENCE_VENV to a known-good runtime."
        )
    log(f"torch ready: version={verified.version} cuda={verified.cuda_version} gpu={verified.gpu_name or '-'}")


def mmcv_ops_ok() -> bool:
    code = """
import mmcv
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
print(getattr(mmcv, "__version__", "unknown"))
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        warn(f"mmcv ops import failed: {' | '.join(detail)}")
        return False
    log(f"mmcv ops import ok: version={(completed.stdout or '').strip().splitlines()[-1]}")
    return True


def torch_tag_candidates(torch_version: str | None) -> list[str]:
    override = os.environ.get("MMCV_TORCH_TAG", "").strip()
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    match = re.match(r"(\d+)\.(\d+)", torch_version or "")
    if not match:
        return []
    major, minor = match.groups()
    return [f"torch{major}.{minor}", f"torch{major}.{minor}.0"]


def cuda_tag(torch_cuda: str | None) -> str:
    override = os.environ.get("MMCV_CUDA_TAG", "").strip()
    if override:
        return override
    if not torch_cuda:
        return "cpu"
    parts = torch_cuda.split(".")
    if len(parts) < 2:
        return f"cu{parts[0]}"
    return f"cu{parts[0]}{parts[1]}"


def verify_python_for_mmcv() -> None:
    if sys.version_info >= (3, 12):
        fail(
            "Python 3.12+ is not a good default for Co-DINO/MMCV 1.x. "
            "Use Python 3.10 or 3.11 via BASE_PYTHON=/path/to/python3.10."
        )


def pip_install_mmcv_from_links(version: str, links: list[str]) -> bool:
    for link in links:
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--only-binary=:all:",
                    f"mmcv-full=={version}",
                    "-f",
                    link,
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            warn(f"mmcv-full wheel install failed from {link}")
            continue
        if mmcv_ops_ok():
            return True
    return False


def should_repin_torch_for_mmcv(torch_info: TorchInfo) -> bool:
    value = os.environ.get("REPIN_TORCH_FOR_MMCV", "auto").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    smi = nvidia_smi_info()
    major = capability_major(torch_info.capability) or capability_major(smi.get("capability"))
    blackwell = major is not None and major >= 12
    if blackwell and value != "1":
        return False
    if torch_info.version and torch_info.version.startswith("2.1.") and torch_info.cuda_version == "12.1":
        return False
    return value in {"1", "true", "yes", "on"} or not blackwell


def repin_torch_for_mmcv() -> TorchInfo:
    log(
        "repinning torch for portable Co-DINO/MMCV wheels: "
        f"{DEFAULT_ADA_TORCH_SPEC} {DEFAULT_ADA_TORCHVISION_SPEC} index={DEFAULT_ADA_TORCH_INDEX}"
    )
    old_env = {
        "TORCH_PIP_SPEC": os.environ.get("TORCH_PIP_SPEC"),
        "TORCHVISION_PIP_SPEC": os.environ.get("TORCHVISION_PIP_SPEC"),
        "TORCH_INDEX_URL": os.environ.get("TORCH_INDEX_URL"),
    }
    os.environ["TORCH_PIP_SPEC"] = DEFAULT_ADA_TORCH_SPEC
    os.environ["TORCHVISION_PIP_SPEC"] = DEFAULT_ADA_TORCHVISION_SPEC
    os.environ["TORCH_INDEX_URL"] = DEFAULT_ADA_TORCH_INDEX
    try:
        install_torch(force=True)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return probe_torch_subprocess()


def pip_install_direct_wheel(wheel: str) -> bool:
    try:
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", wheel], check=True)
    except subprocess.CalledProcessError:
        warn(f"direct mmcv wheel install failed: {wheel}")
        return False
    return mmcv_ops_ok()


def source_build_allowed() -> bool:
    value = os.environ.get("ALLOW_MMCV_SOURCE_BUILD", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return shutil.which("nvcc") is not None


def install_mmcv_source(version: str) -> bool:
    if shutil.which("nvcc") is None:
        warn("nvcc is missing; mmcv-full source build cannot compile CUDA ops.")
        return False
    env = dict(os.environ)
    env.setdefault("MMCV_WITH_OPS", "1")
    env.setdefault("FORCE_CUDA", "1")
    env.setdefault("MAX_JOBS", str(max(1, min(os.cpu_count() or 1, 8))))
    try:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-binary=mmcv-full",
                f"mmcv-full=={version}",
            ],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        warn("mmcv-full source build failed")
        return False
    return mmcv_ops_ok()


def install_mmcv() -> None:
    verify_python_for_mmcv()
    install_openmmlab_compat_packages()
    if mmcv_ops_ok():
        return
    version = os.environ.get("MMCV_FULL_VERSION", DEFAULT_MMCV_FULL_VERSION).strip() or DEFAULT_MMCV_FULL_VERSION
    torch_info = probe_torch_subprocess()
    if not torch_info.import_ok:
        fail("torch must import before installing mmcv-full. Run the torch setup step first.")

    run([sys.executable, "-m", "pip", "uninstall", "-y", "mmcv", "mmcv-full"], check=False)

    direct_wheel = (
        os.environ.get("MMCV_FULL_WHEEL")
        or os.environ.get("MMCV_FULL_WHEEL_PATH")
        or os.environ.get("MMCV_WHEEL_PATH")
        or ""
    ).strip()
    if direct_wheel and pip_install_direct_wheel(direct_wheel):
        return

    find_links: list[str] = []
    if os.environ.get("MMCV_FIND_LINKS"):
        find_links.extend(shlex.split(os.environ["MMCV_FIND_LINKS"]))

    ctag = cuda_tag(torch_info.cuda_version)
    for tag in torch_tag_candidates(torch_info.version):
        find_links.append(f"https://download.openmmlab.com/mmcv/dist/{ctag}/{tag}/index.html")
    if ctag == "cu121" and torch_info.version and torch_info.version.startswith("2.1."):
        find_links.append(DEFAULT_ADA_MMCV_FIND_LINKS)
    find_links = list(dict.fromkeys(find_links))

    if pip_install_mmcv_from_links(version, find_links):
        install_openmmlab_compat_packages()
        return

    if should_repin_torch_for_mmcv(torch_info):
        torch_info = repin_torch_for_mmcv()
        ctag = cuda_tag(torch_info.cuda_version)
        repinned_links = []
        for tag in torch_tag_candidates(torch_info.version):
            repinned_links.append(f"https://download.openmmlab.com/mmcv/dist/{ctag}/{tag}/index.html")
        if ctag == "cu121" and torch_info.version and torch_info.version.startswith("2.1."):
            repinned_links.append(DEFAULT_ADA_MMCV_FIND_LINKS)
        repinned_links = list(dict.fromkeys(repinned_links))
        if pip_install_mmcv_from_links(version, repinned_links):
            install_openmmlab_compat_packages()
            return
        find_links.extend(link for link in repinned_links if link not in find_links)

    if source_build_allowed() and install_mmcv_source(version):
        install_openmmlab_compat_packages()
        return

    fail(
        "Could not install mmcv-full with CUDA ops.\n"
        f"  python={sys.version.split()[0]}\n"
        f"  torch={torch_info.version} torch_cuda={torch_info.cuda_version}\n"
        f"  attempted_links={find_links or '-'}\n"
        "Fix options:\n"
        "  - Use Python 3.10/3.11 and a PyTorch/CUDA combo with an OpenMMLab wheel.\n"
        "  - Set MMCV_FULL_WHEEL=/path/to/mmcv_full-1.7.2-...whl.\n"
        f"  - Set MMCV_FIND_LINKS={DEFAULT_ADA_MMCV_FIND_LINKS}.\n"
        "  - Install nvcc and set ALLOW_MMCV_SOURCE_BUILD=1 to build from source.\n"
        "  - Pass REFERENCE_VENV=/path/to/known-good-venv."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "torch", "mmcv", "all"))
    args = parser.parse_args()
    if args.command in {"diagnose", "all"}:
        diagnose()
    if args.command in {"torch", "all"}:
        install_torch()
        install_openmmlab_compat_packages()
    if args.command in {"mmcv", "all"}:
        install_mmcv()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
