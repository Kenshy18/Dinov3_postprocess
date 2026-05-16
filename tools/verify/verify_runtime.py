#!/usr/bin/env python3
"""Run cleanup-safe verification checks for the integrated runtime.

The default suite avoids detector inference so it is cheap enough to run after
documentation/tooling refactors. Use ``--detector-smoke`` when a change touches
pipeline wiring or runtime defaults.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv_integrated" / "bin" / "python"
SAMPLE_VIDEO = ROOT / "input" / "3月以降解析白カン動画-0210.mp4"
ATOSYORI_SRC = ROOT / "external" / "atosyori-pipeline-dev" / "src"
ATOSYORI_REPO = ROOT / "external" / "atosyori-pipeline-dev"


def runtime_python() -> Path:
    return PYTHON if PYTHON.is_file() else Path(sys.executable)


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("[verify] " + " ".join(command), flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"verification command failed with exit code {completed.returncode}: {command}")
    print(f"[verify-done] {elapsed:.2f}s", flush=True)


def atosyori_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ATOSYORI_SRC) if not current else str(ATOSYORI_SRC) + os.pathsep + current
    return env


def run_quick_checks(args: argparse.Namespace) -> None:
    py = str(runtime_python())
    artifact_cmd = [py, "tools/artifacts/check_artifacts.py"]
    if args.require_trt:
        artifact_cmd.append("--require-trt")
    run(artifact_cmd)
    run(
        [
            py,
            "-m",
            "compileall",
            "-q",
            "apps",
            "backend",
            "scripts",
            "training",
            "tools",
            "configs",
            "tests",
            "external/atosyori-pipeline-dev/src",
            "external/atosyori-pipeline-dev/tests",
        ]
    )
    run(
        [py, "-m", "atosyori_postprocess", "doctor", "--model-root", "checkpoints/postprocess"],
        env=atosyori_env(),
    )
    run([py, "-m", "atosyori_postprocess", "engine-check"], env=atosyori_env())
    run([py, "-m", "unittest", "discover", "-s", "tests"])
    run([py, "-m", "unittest", "discover", "-s", "tests"], cwd=ATOSYORI_REPO)
    if args.atosyori_smoke:
        run(
            [
                py,
                "-m",
                "atosyori_postprocess",
                "smoke",
                "--work-dir",
                str(args.work_dir / "atosyori_smoke"),
                "--force",
            ],
            env=atosyori_env(),
        )


def run_detector_smoke(args: argparse.Namespace) -> None:
    if not SAMPLE_VIDEO.is_file():
        raise FileNotFoundError(SAMPLE_VIDEO)
    py = str(runtime_python())
    command = [
        py,
        "scripts/run_integrated_pipeline.py",
        "--input",
        str(SAMPLE_VIDEO),
        "--output-root",
        str(args.work_dir),
        "--run-name",
        f"{args.detector}{args.frames}",
        "--detector",
        args.detector,
        "--max-frames",
        str(args.frames),
        "--no-postprocess",
        "--classifier",
        "--force",
    ]
    if args.detector == "dinov3":
        command.extend(["--warmup-frames", "0", "--no-async-writer"])
    elif args.detector == "eva02":
        command.extend(["--eva02-warmup-frames", "0"])
    else:
        command.extend(["--codino-warmup-frames", "0"])
    run(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run integrated runtime verification checks")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/dinov3_postprocess_verify"))
    parser.add_argument("--no-atosyori-smoke", dest="atosyori_smoke", action="store_false", default=True)
    parser.add_argument("--require-trt", action="store_true", help="Fail quick checks if local TensorRT engines are missing")
    parser.add_argument("--detector-smoke", action="store_true", help="Run GPU detector smoke after cheap checks")
    parser.add_argument("--detector", choices=("dinov3", "eva02", "codino"), default="dinov3")
    parser.add_argument("--frames", type=int, default=64, help="Frames for detector smoke")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_quick_checks(args)
    if args.detector_smoke:
        run_detector_smoke(args)
    print(f"[verify-ok] work_dir={args.work_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
