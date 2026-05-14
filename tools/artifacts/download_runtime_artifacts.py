#!/usr/bin/env python3
"""Download and place gitignored runtime artifacts.

This script intentionally supports a small set of layouts used by this bundle:

- unified runtime artifact folder prepared from artifacts_to_upload/runtime_artifacts/
- current shared Google Drive runtime folder, which groups detector artifacts by
  backbone under checkpoints/dinov3/ and checkpoints/Eva02/
- postprocess Google Drive folder from Atosyori
- DINOv3 runtime Google Drive folder prepared from artifacts_to_upload/

It can also place artifacts from an already downloaded local directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    from .runtime_artifacts import DINOV3_MAPPINGS, POSTPROCESS_MAPPINGS, ROOT, RUNTIME_MAPPINGS, SourceSpec
except ImportError:  # pragma: no cover - direct script execution
    from runtime_artifacts import DINOV3_MAPPINGS, POSTPROCESS_MAPPINGS, ROOT, RUNTIME_MAPPINGS, SourceSpec


DEFAULT_POSTPROCESS_URL = (
    "https://drive.google.com/drive/folders/"
    "10-Zc2ShIkJn7T1JIcvUgiEgdSoIANJcT?usp=sharing"
)


def is_drive_url(url: str) -> bool:
    return "drive.google.com" in url or "docs.google.com" in url


def is_drive_folder_url(url: str) -> bool:
    return "drive.google.com/drive/folders/" in url


def run(command: list[str]) -> None:
    print("[cmd] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def ensure_gdown() -> None:
    try:
        __import__("gdown")
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def download_to_dir(url: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if is_drive_folder_url(url):
        ensure_gdown()
        run([sys.executable, "-m", "gdown", "--folder", url, "-O", str(dest)])
        return dest

    if is_drive_url(url):
        ensure_gdown()
        output = dest / "downloaded_artifact"
        run([sys.executable, "-m", "gdown", url, "-O", str(output)])
        return output

    filename = Path(url.split("?")[0]).name or "downloaded_artifact"
    output = dest / filename
    print(f"[download] {url} -> {output}", flush=True)
    urllib.request.urlretrieve(url, output)
    return output


def path_has_suffix(path: Path, suffix: str) -> bool:
    parts = path.parts
    suffix_parts = Path(suffix).parts
    return len(parts) >= len(suffix_parts) and parts[-len(suffix_parts) :] == suffix_parts


def source_candidates(expected_rel: SourceSpec) -> tuple[str, ...]:
    if isinstance(expected_rel, str):
        return (expected_rel,)
    return expected_rel


def format_source_spec(expected_rel: SourceSpec) -> str:
    candidates = source_candidates(expected_rel)
    if len(candidates) == 1:
        return candidates[0]
    return " or ".join(candidates)


def find_source(root: Path, expected_rel: SourceSpec) -> Path:
    candidates = source_candidates(expected_rel)
    for rel in candidates:
        exact = root / rel
        if exact.is_file():
            return exact

    matches: list[Path] = []
    for rel in candidates:
        matches.extend(p for p in root.rglob(Path(rel).name) if p.is_file() and path_has_suffix(p, rel))
    if matches:
        return sorted(set(matches), key=lambda p: (len(p.parts), str(p)))[0]

    filenames = {Path(rel).name for rel in candidates}
    name_matches = [p for filename in filenames for p in root.rglob(filename) if p.is_file()]
    if len(name_matches) == 1:
        return name_matches[0]

    if name_matches:
        options = "\n".join(f"  - {p}" for p in sorted(name_matches)[:20])
        raise FileNotFoundError(f"ambiguous artifact for {format_source_spec(expected_rel)}; candidates:\n{options}")
    raise FileNotFoundError(f"artifact not found under {root}: {format_source_spec(expected_rel)}")


def place_mappings(source_root: Path, mappings: tuple[tuple[SourceSpec, str], ...], *, overwrite: bool) -> None:
    for source_rel, dest_rel in mappings:
        src = find_source(source_root, source_rel)
        dst = ROOT / dest_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                print(f"[skip] exists: {dst}", flush=True)
                continue
            dst.unlink()
        shutil.copy2(src, dst)
        print(f"[place] {src} -> {dst}", flush=True)


def all_destinations_exist(mappings: tuple[tuple[SourceSpec, str], ...]) -> bool:
    return all((ROOT / dest_rel).is_file() for _, dest_rel in mappings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/place integrated runtime artifacts")
    parser.add_argument("--runtime-artifacts-url", default=os.environ.get("RUNTIME_ARTIFACTS_URL", ""))
    parser.add_argument("--runtime-artifacts-dir", type=Path, default=None)
    parser.add_argument("--dinov3-artifacts-url", default=os.environ.get("DINOV3_ARTIFACTS_URL", ""))
    parser.add_argument("--dinov3-artifacts-dir", type=Path, default=None)
    parser.add_argument(
        "--postprocess-url",
        default=os.environ.get("POSTPROCESS_ARTIFACTS_URL", DEFAULT_POSTPROCESS_URL),
    )
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-dinov3", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "output" / "download_cache" / "artifacts")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.download_dir = args.download_dir.expanduser().resolve()

    runtime_source: Path | None = None
    if not args.skip_runtime:
        if all_destinations_exist(RUNTIME_MAPPINGS) and not args.overwrite:
            print("[skip] unified runtime artifacts already exist", flush=True)
            return 0
        if args.runtime_artifacts_dir is not None:
            runtime_source = args.runtime_artifacts_dir.expanduser().resolve()
        elif args.runtime_artifacts_url:
            runtime_dir = args.download_dir / "runtime_artifacts"
            downloaded = download_to_dir(args.runtime_artifacts_url, runtime_dir)
            runtime_source = downloaded if downloaded.is_dir() else downloaded.parent

        if runtime_source is not None:
            place_mappings(runtime_source, RUNTIME_MAPPINGS, overwrite=args.overwrite)
            return 0

    if not args.skip_postprocess:
        if all_destinations_exist(POSTPROCESS_MAPPINGS) and not args.overwrite:
            print("[skip] postprocess artifacts already exist", flush=True)
        elif not args.postprocess_url:
            print("[skip] postprocess artifacts URL is empty", flush=True)
        else:
            postprocess_dir = args.download_dir / "postprocess"
            downloaded = download_to_dir(args.postprocess_url, postprocess_dir)
            source_root = downloaded if downloaded.is_dir() else downloaded.parent
            place_mappings(source_root, POSTPROCESS_MAPPINGS, overwrite=args.overwrite)

    if not args.skip_dinov3:
        source_root: Path | None = None
        if all_destinations_exist(DINOV3_MAPPINGS) and not args.overwrite:
            print("[skip] DINOv3 artifacts already exist", flush=True)
            source_root = None
        elif args.dinov3_artifacts_dir is not None:
            source_root = args.dinov3_artifacts_dir.expanduser().resolve()
        elif args.dinov3_artifacts_url:
            dinov3_dir = args.download_dir / "dinov3_runtime"
            downloaded = download_to_dir(args.dinov3_artifacts_url, dinov3_dir)
            source_root = downloaded if downloaded.is_dir() else downloaded.parent

        if source_root is not None:
            place_mappings(source_root, DINOV3_MAPPINGS, overwrite=args.overwrite)
        elif not all_destinations_exist(DINOV3_MAPPINGS):
            print(
                "[skip] DINOv3 artifacts URL is empty. "
                "Prefer RUNTIME_ARTIFACTS_URL after uploading artifacts_to_upload/runtime_artifacts, "
                "or set DINOV3_ARTIFACTS_URL for legacy split-source setup.",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
