#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$TOOLS_DIR/.." && pwd)"

ARTIFACT_ENV_FILE="${ARTIFACT_ENV_FILE:-$ROOT_DIR/configs/artifact_sources.env}"
if [[ -f "$ARTIFACT_ENV_FILE" ]]; then
  echo "[SETUP] loading artifact env: $ARTIFACT_ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ARTIFACT_ENV_FILE"
  set +a
fi

DINO_RUNTIME_DIR="${DINO_RUNTIME_DIR:-$(cd "$ROOT_DIR/inference/dinov3_video_jsonl_runtime" && pwd)}"
if [[ -z "${ATOSYORI_REPO:-}" ]]; then
  if [[ -d "$ROOT_DIR/external/atosyori-pipeline-dev/src/atosyori_postprocess" ]]; then
    ATOSYORI_REPO="$ROOT_DIR/external/atosyori-pipeline-dev"
  else
    ATOSYORI_REPO="/home/kenke/workspace/CV/atosyori-pipeline-dev"
  fi
fi
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv_integrated}"
RUN_DINO_SMOKE="${RUN_DINO_SMOKE:-1}"
RUN_IMPORT_CHECK="${RUN_IMPORT_CHECK:-1}"
POSTPROCESS_MODEL_ROOT="${POSTPROCESS_MODEL_ROOT:-$ROOT_DIR/checkpoints/postprocess}"
BUILD_DETECTRON2="${BUILD_DETECTRON2:-auto}"
BASE_PYTHON="${BASE_PYTHON:-}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-1}"
ARTIFACT_OVERWRITE="${ARTIFACT_OVERWRITE:-0}"
ARTIFACT_DOWNLOAD_DIR="${ARTIFACT_DOWNLOAD_DIR:-$ROOT_DIR/output/download_cache/artifacts}"
POSTPROCESS_ARTIFACTS_URL="${POSTPROCESS_ARTIFACTS_URL:-https://drive.google.com/drive/folders/10-Zc2ShIkJn7T1JIcvUgiEgdSoIANJcT?usp=sharing}"
RUNTIME_ARTIFACTS_URL="${RUNTIME_ARTIFACTS_URL:-https://drive.google.com/drive/folders/1cj9gPOt4MIRu6cFW80vfTZHK9oLQB6qn?usp=sharing}"
RUNTIME_ARTIFACTS_DIR="${RUNTIME_ARTIFACTS_DIR:-}"
DINOV3_ARTIFACTS_URL="${DINOV3_ARTIFACTS_URL:-}"
DINOV3_ARTIFACTS_DIR="${DINOV3_ARTIFACTS_DIR:-}"
REBUILD_TRT="${REBUILD_TRT:-auto}"

if [[ -z "$BASE_PYTHON" ]]; then
  if [[ -x /home/kenke/miniconda3/envs/eva02_trt/bin/python ]]; then
    BASE_PYTHON=/home/kenke/miniconda3/envs/eva02_trt/bin/python
  elif command -v python3.10 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3.10)"
  else
    BASE_PYTHON="$(command -v python3)"
  fi
fi

echo "[SETUP] integration root: $ROOT_DIR"
echo "[SETUP] DINO runtime:      $DINO_RUNTIME_DIR"
echo "[SETUP] Atosyori repo:     $ATOSYORI_REPO"
echo "[SETUP] env:              $ENV_DIR"
echo "[SETUP] base python:      $BASE_PYTHON"

if [[ ! -d "$ENV_DIR" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"

if [[ "$DOWNLOAD_ARTIFACTS" == "1" ]]; then
  "$PY" -m pip install -q gdown
  download_cmd=(
    "$PY"
    "$ROOT_DIR/tools/download_runtime_artifacts.py"
    "--postprocess-url"
    "$POSTPROCESS_ARTIFACTS_URL"
    "--download-dir"
    "$ARTIFACT_DOWNLOAD_DIR"
  )
  if [[ -n "$RUNTIME_ARTIFACTS_URL" ]]; then
    download_cmd+=("--runtime-artifacts-url" "$RUNTIME_ARTIFACTS_URL")
  elif [[ -n "$RUNTIME_ARTIFACTS_DIR" ]]; then
    download_cmd+=("--runtime-artifacts-dir" "$RUNTIME_ARTIFACTS_DIR")
  elif [[ -n "$DINOV3_ARTIFACTS_URL" ]]; then
    download_cmd+=("--dinov3-artifacts-url" "$DINOV3_ARTIFACTS_URL")
  elif [[ -n "$DINOV3_ARTIFACTS_DIR" ]]; then
    download_cmd+=("--dinov3-artifacts-dir" "$DINOV3_ARTIFACTS_DIR")
  fi
  if [[ "$ARTIFACT_OVERWRITE" == "1" ]]; then
    download_cmd+=("--overwrite")
  fi
  "${download_cmd[@]}"
else
  echo "[SETUP] artifact download disabled"
fi

for required in \
  "$ROOT_DIR/checkpoints/detector/model_final.pth" \
  "$ROOT_DIR/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
  "$ROOT_DIR/checkpoints/classifier/best.pt"; do
  if [[ ! -f "$required" ]]; then
    cat >&2 <<EOF
[ERROR] required DINOv3 artifact is missing:
  $required

Set RUNTIME_ARTIFACTS_URL to the Google Drive folder uploaded from:
  artifacts_to_upload/runtime_artifacts

Or place those files manually and rerun this setup script.
EOF
    exit 2
  fi
done

RUN_SMOKE="$RUN_DINO_SMOKE" \
  ENV_DIR="$ENV_DIR" \
  BASE_PYTHON="$BASE_PYTHON" \
  REBUILD_TRT="$REBUILD_TRT" \
  "$DINO_RUNTIME_DIR/tools/setup_fast_runtime_env.sh"

if [[ "$BUILD_DETECTRON2" == "1" ]] || { [[ "$BUILD_DETECTRON2" == "auto" ]] && ! ls "$ROOT_DIR/eva02/eva02_det/detectron2"/_C*.so >/dev/null 2>&1; }; then
  echo "[SETUP] building/installing bundled Detectron2/EVA02 extension"
  "$PY" -m pip install -q -e "$ROOT_DIR/eva02/eva02_det"
else
  echo "[SETUP] bundled Detectron2 extension exists"
fi

"$PY" -m pip install -q -e "$ATOSYORI_REPO"
"$PY" "$ROOT_DIR/tools/check_artifacts.py" --allow-missing

if [[ "$RUN_IMPORT_CHECK" == "1" ]]; then
  "$PY" - <<'PY'
import importlib
import sys

for name in ("torch", "cv2", "orjson", "atosyori_postprocess"):
    importlib.import_module(name)
print(f"[CHECK] imports ok: {sys.executable}")
PY
  "$PY" -m atosyori_postprocess doctor --model-root "$POSTPROCESS_MODEL_ROOT" || true
fi

cat <<EOF

[DONE] Integrated runtime environment is ready.

Run:
  $PY $ROOT_DIR/scripts/run_integrated_pipeline.py \\
    --input INPUT.mp4 \\
    --output-root $ROOT_DIR/output/runs \\
    --classifier \\
    --render-overlays \\
    --force

Postprocess checkpoints are expected under:
  $POSTPROCESS_MODEL_ROOT
EOF
