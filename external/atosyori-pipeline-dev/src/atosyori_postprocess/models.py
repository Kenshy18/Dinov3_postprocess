"""Model checkpoint layout helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

from .settings import resolve_models


def copy_from_legacy_root(source_root: Path, *, model_root: Path | None = None, replace: bool = False) -> None:
    layout = resolve_models(model_root)
    layout.root.mkdir(parents=True, exist_ok=True)

    pairs = {
        layout.k2_dir: source_root / "K2" / "Distiled",
        layout.polygon_point_predictor_dir: source_root / "Point_predictor" / "Distiled",
    }

    for destination, source in pairs.items():
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.exists() or destination.is_symlink():
            if not replace:
                print(f"exists: {destination}")
                continue
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.copytree(source, destination)
        print(f"copied: {source} -> {destination}")


def model_status(model_root: Path | None = None) -> dict[str, bool | str]:
    layout = resolve_models(model_root)
    return {
        "root": str(layout.root),
        "k2_dir": str(layout.k2_dir),
        "polygon_point_predictor_dir": str(layout.polygon_point_predictor_dir),
        "k2_checkpoint": (layout.k2_dir / "best_exact.pt").is_file(),
        "polygon_checkpoint": (layout.polygon_point_predictor_dir / "best.pt").is_file(),
    }
