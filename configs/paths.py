import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TYPED_DIRS = {
    "pth",
    "json",
    "tensorboard",
    "visualizations",
    "meta",
    "onnx",
    "legacy_empty",
}


def _env_path(key: str, default: str) -> str:
    val = os.environ.get(key)
    return val if val else default


DATA_ROOT = _env_path(
    "UNIFIED_DATA_ROOT",
    str(REPO_ROOT / "datasets" / "images"),
)
ANN_ALL_JSON = _env_path(
    "UNIFIED_ANN_ALL_JSON",
    str(REPO_ROOT / "datasets" / "annotations" / "annotations_all.json"),
)
ANN_ROOT = Path(
    _env_path("UNIFIED_ANN_ROOT", str(REPO_ROOT / "datasets" / "annotations"))
)
TRAIN_JSON = _env_path(
    "UNIFIED_TRAIN_JSON",
    str(ANN_ROOT / "annotations_train.json"),
)
VAL_JSON = _env_path(
    "UNIFIED_VAL_JSON",
    str(ANN_ROOT / "annotations_val.json"),
)
TEST_JSON = _env_path("UNIFIED_TEST_JSON", VAL_JSON)

DINOv3_WEIGHTS = _env_path(
    "UNIFIED_DINOV3_WEIGHTS",
    str(REPO_ROOT / "checkpoints" / "dinov3" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
)
EVA02_WEIGHTS = _env_path(
    "UNIFIED_EVA02_WEIGHTS",
    str(REPO_ROOT / "checkpoints" / "eva02" / "detector" / "model_final.pth"),
)

OUTPUT_ROOT = Path(_env_path("UNIFIED_OUTPUT_ROOT", str(REPO_ROOT / "output" / "training_outputs")))

SPLIT_RATIO = float(os.environ.get("UNIFIED_SPLIT_RATIO", "0.93"))
SPLIT_SEED = int(os.environ.get("UNIFIED_SPLIT_SEED", "42"))


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    run_rel: Path
    logical_dir: Path
    pth_dir: Path
    json_dir: Path
    tensorboard_dir: Path
    visualizations_dir: Path
    meta_dir: Path
    onnx_dir: Path


def build_output_layout(path: str | Path) -> OutputLayout:
    logical_dir = Path(path).expanduser()
    parts = logical_dir.parts
    anchor_idx = -1
    for idx, part in enumerate(parts):
        if part == "outputs":
            anchor_idx = idx
    if anchor_idx >= 0:
        root = Path(*parts[: anchor_idx + 1])
        run_rel = Path(*parts[anchor_idx + 1 :]) if anchor_idx + 1 < len(parts) else Path(logical_dir.name)
        if run_rel.parts and run_rel.parts[0] in OUTPUT_TYPED_DIRS:
            run_rel = Path(*run_rel.parts[1:]) if len(run_rel.parts) > 1 else Path(logical_dir.name)
    else:
        root = logical_dir.parent
        run_rel = Path(logical_dir.name)
    if run_rel == Path():
        run_rel = Path(logical_dir.name)
    return OutputLayout(
        root=root,
        run_rel=run_rel,
        logical_dir=logical_dir,
        pth_dir=root / "pth" / run_rel,
        json_dir=root / "json" / run_rel,
        tensorboard_dir=root / "tensorboard" / run_rel,
        visualizations_dir=root / "visualizations" / run_rel,
        meta_dir=root / "meta" / run_rel,
        onnx_dir=root / "onnx" / run_rel,
    )


def output_layout_for_run(run_name: str | Path) -> OutputLayout:
    return build_output_layout(OUTPUT_ROOT / Path(run_name))


def resolve_preferred_path(preferred: Path, legacy: Path | None = None) -> Path:
    preferred = preferred.expanduser().resolve()
    if preferred.exists():
        return preferred
    if legacy is not None:
        legacy = legacy.expanduser().resolve()
        if legacy.exists():
            return legacy
    return preferred


def resolve_output_checkpoint(run_name: str | Path, filename: str) -> Path:
    layout = output_layout_for_run(run_name)
    return resolve_preferred_path(layout.pth_dir / filename, OUTPUT_ROOT / Path(run_name) / filename)


def resolve_output_meta_artifact(run_name: str | Path, *parts: str) -> Path:
    layout = output_layout_for_run(run_name)
    rel = Path(*parts)
    return resolve_preferred_path(layout.meta_dir / rel, OUTPUT_ROOT / Path(run_name) / rel)


def resolve_output_json_artifact(run_name: str | Path, *parts: str) -> Path:
    layout = output_layout_for_run(run_name)
    rel = Path(*parts)
    return resolve_preferred_path(layout.json_dir / rel, OUTPUT_ROOT / Path(run_name) / rel)


def resolve_output_visualization_artifact(run_name: str | Path, *parts: str) -> Path:
    layout = output_layout_for_run(run_name)
    rel = Path(*parts)
    return resolve_preferred_path(layout.visualizations_dir / rel, OUTPUT_ROOT / Path(run_name) / rel)


def resolve_output_onnx_artifact(run_name: str | Path, *parts: str) -> Path:
    layout = output_layout_for_run(run_name)
    rel = Path(*parts)
    return resolve_preferred_path(layout.onnx_dir / rel, OUTPUT_ROOT / Path(run_name) / rel)
