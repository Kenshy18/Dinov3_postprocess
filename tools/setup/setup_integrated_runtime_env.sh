#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$TOOLS_DIR/../.." && pwd)"

ARTIFACT_ENV_FILE="${ARTIFACT_ENV_FILE:-$ROOT_DIR/configs/artifact_sources.env}"
if [[ -f "$ARTIFACT_ENV_FILE" ]]; then
  echo "[SETUP] loading artifact env: $ARTIFACT_ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ARTIFACT_ENV_FILE"
  set +a
fi

DINO_RUNTIME_DIR="${DINO_RUNTIME_DIR:-$(cd "$ROOT_DIR/backend/detectors/dinov3/runtime" && pwd)}"
if [[ -z "${ATOSYORI_REPO:-}" ]]; then
  ATOSYORI_REPO="$ROOT_DIR/external/atosyori-pipeline-dev"
fi
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv_integrated}"
RUN_DINO_SMOKE="${RUN_DINO_SMOKE:-1}"
RUN_IMPORT_CHECK="${RUN_IMPORT_CHECK:-1}"
POSTPROCESS_MODEL_ROOT="${POSTPROCESS_MODEL_ROOT:-$ROOT_DIR/checkpoints/postprocess}"
BUILD_DETECTRON2="${BUILD_DETECTRON2:-auto}"
SETUP_CODINO_DEPS="${SETUP_CODINO_DEPS:-1}"
BASE_PYTHON="${BASE_PYTHON:-}"
DOWNLOAD_ARTIFACTS="${DOWNLOAD_ARTIFACTS:-1}"
ARTIFACT_OVERWRITE="${ARTIFACT_OVERWRITE:-0}"
ARTIFACT_DOWNLOAD_DIR="${ARTIFACT_DOWNLOAD_DIR:-$ROOT_DIR/output/download_cache/artifacts}"
POSTPROCESS_ARTIFACTS_URL="${POSTPROCESS_ARTIFACTS_URL:-}"
RUNTIME_ARTIFACTS_URL="${RUNTIME_ARTIFACTS_URL:-}"
RUNTIME_ARTIFACTS_DIR="${RUNTIME_ARTIFACTS_DIR:-}"
DINOV3_ARTIFACTS_URL="${DINOV3_ARTIFACTS_URL:-}"
DINOV3_ARTIFACTS_DIR="${DINOV3_ARTIFACTS_DIR:-}"
REBUILD_TRT="${REBUILD_TRT:-auto}"
ENGINE_PATH="${ENGINE_PATH:-$ROOT_DIR/checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine}"
REBUILD_CODINO_TRT="${REBUILD_CODINO_TRT:-auto}"
CODINO_TRT_BATCH_SIZE="${CODINO_TRT_BATCH_SIZE:-auto}"
CODINO_TRT_BACKBONE_PRECISION="${CODINO_TRT_BACKBONE_PRECISION:-bf16}"
CODINO_TRT_QUERY_PRECISION="${CODINO_TRT_QUERY_PRECISION:-fp16}"
CODINO_TRT_DECODER_PRECISION="${CODINO_TRT_DECODER_PRECISION:-fp16}"
CODINO_TRT_MASK_PRECISION="${CODINO_TRT_MASK_PRECISION:-fp16}"
DINOV3_RUNTIME_PROFILE="${DINOV3_RUNTIME_PROFILE:-$ROOT_DIR/.runtime/runtime_profile.json}"
GUI_RUNTIME_ENV="${GUI_RUNTIME_ENV:-$ROOT_DIR/.runtime/gui_runtime.env}"
RUN_BATCH_BENCHMARK="${RUN_BATCH_BENCHMARK:-1}"
BATCH_BENCHMARK_OUTPUT="${BATCH_BENCHMARK_OUTPUT:-$ROOT_DIR/.runtime/runtime_benchmark.json}"
BATCH_BENCHMARK_DETECTORS="${BATCH_BENCHMARK_DETECTORS:-eva02,dinov3,codino}"
BATCH_BENCHMARK_FRAMES="${BATCH_BENCHMARK_FRAMES:-24}"
BATCH_BENCHMARK_TIMEOUT_SEC="${BATCH_BENCHMARK_TIMEOUT_SEC:-360}"

case "$ENGINE_PATH" in
  /*) ;;
  *) ENGINE_PATH="$ROOT_DIR/$ENGINE_PATH" ;;
esac
case "$DINOV3_RUNTIME_PROFILE" in
  /*) ;;
  *) DINOV3_RUNTIME_PROFILE="$ROOT_DIR/$DINOV3_RUNTIME_PROFILE" ;;
esac
case "$BATCH_BENCHMARK_OUTPUT" in
  /*) ;;
  *) BATCH_BENCHMARK_OUTPUT="$ROOT_DIR/$BATCH_BENCHMARK_OUTPUT" ;;
esac

python_works() {
  local candidate="$1"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

if [[ -z "${REFERENCE_VENV:-}" ]]; then
  for candidate in \
    "$ROOT_DIR/.venv_integrated" \
    "$ROOT_DIR/../eva02_cascade_experimental/venv"; do
    if [[ "$candidate" != "$ENV_DIR" && -x "$candidate/bin/python" ]]; then
      REFERENCE_VENV="$candidate"
      break
    fi
  done
fi

if [[ -z "$BASE_PYTHON" ]]; then
  if [[ -n "${REFERENCE_VENV:-}" && -x "$REFERENCE_VENV/bin/python" ]]; then
    BASE_PYTHON="$REFERENCE_VENV/bin/python"
  else
    for candidate in \
      "$ROOT_DIR/.venv_integrated/bin/python" \
      "$(command -v python3.10 2>/dev/null || true)" \
      "$(command -v python3 2>/dev/null || true)" \
      /usr/bin/python3; do
      if python_works "$candidate"; then
        BASE_PYTHON="$candidate"
        break
      fi
    done
  fi
fi

if ! python_works "$BASE_PYTHON"; then
  echo "[ERROR] no working Python >= 3.10 found. Set BASE_PYTHON=/path/to/python." >&2
  exit 2
fi

echo "[SETUP] integration root: $ROOT_DIR"
echo "[SETUP] DINO runtime:      $DINO_RUNTIME_DIR"
echo "[SETUP] Atosyori repo:     $ATOSYORI_REPO"
echo "[SETUP] env:              $ENV_DIR"
echo "[SETUP] base python:      $BASE_PYTHON"
if [[ -n "${REFERENCE_VENV:-}" ]]; then
  echo "[SETUP] reference venv:   $REFERENCE_VENV"
fi

if [[ ! -d "$ENV_DIR" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"

if [[ "$DOWNLOAD_ARTIFACTS" == "1" ]]; then
  "$PY" -m pip install -q gdown
  download_cmd=(
    "$PY"
    "$ROOT_DIR/tools/artifacts/download_runtime_artifacts.py"
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

if ! "$PY" "$ROOT_DIR/tools/artifacts/check_artifacts.py"; then
  cat >&2 <<EOF
[ERROR] required runtime artifacts are missing.

For a normal clone, set RUNTIME_ARTIFACTS_URL in configs/artifact_sources.env
to the shared Drive folder and rerun this setup script.

For local artifact storage, set RUNTIME_ARTIFACTS_DIR=/path/to/runtime_artifacts.
EOF
  exit 2
fi

RUN_SMOKE="$RUN_DINO_SMOKE" \
  ENV_DIR="$ENV_DIR" \
  BASE_PYTHON="$BASE_PYTHON" \
  REFERENCE_VENV="${REFERENCE_VENV:-}" \
  ENGINE_PATH="$ENGINE_PATH" \
  REBUILD_TRT="$REBUILD_TRT" \
  "$DINO_RUNTIME_DIR/tools/setup_fast_runtime_env.sh"

if [[ "$SETUP_CODINO_DEPS" == "1" ]]; then
  echo "[SETUP] installing Co-DINO/MMCV runtime dependencies"
  "$PY" -m pip install -q \
    cython \
    numpy \
    openmim \
    onnx \
    terminaltables \
    yapf \
    scipy \
    fairscale \
    timm \
    fvcore \
    tensorboard \
    einops
  if ! "$PY" - <<'PY'
import importlib
import sys

try:
    import mmcv  # noqa: F401
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction  # noqa: F401
except Exception as exc:
    print(repr(exc), file=sys.stderr)
    raise SystemExit(1)
PY
  then
    "$PY" -m mim install "mmcv-full>=1.7.0,<1.8.0"
  fi
  "$PY" -m pip install -q -e "$ROOT_DIR/external/codino" --no-deps
fi

if [[ "$BUILD_DETECTRON2" == "1" ]] || { [[ "$BUILD_DETECTRON2" == "auto" ]] && ! ls "$ROOT_DIR/eva02/eva02_det/detectron2"/_C*.so >/dev/null 2>&1; }; then
  echo "[SETUP] building/installing bundled Detectron2/EVA02 extension"
  "$PY" -m pip install -q -e "$ROOT_DIR/eva02/eva02_det"
else
  echo "[SETUP] bundled Detectron2 extension exists"
fi

"$PY" -m pip install -q -e "$ATOSYORI_REPO"
if [[ -f "$ROOT_DIR/apps/qt_ui/requirements.txt" ]]; then
  "$PY" -m pip install -q -r "$ROOT_DIR/apps/qt_ui/requirements.txt"
elif [[ -f "$ROOT_DIR/UI/requirements.txt" ]]; then
  "$PY" -m pip install -q -r "$ROOT_DIR/UI/requirements.txt"
fi

if [[ "$CODINO_TRT_BATCH_SIZE" == "auto" ]]; then
  CODINO_TRT_BATCH_SIZE="$("$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    print(1)
else:
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(2 if total_gib >= 20 else 1)
PY
)"
fi

CODINO_TRT_DIR="$ROOT_DIR/checkpoints/codino/trt"
CODINO_TRT_BACKBONE_ENGINE="$CODINO_TRT_DIR/codino_dinov3_vitl_backbone_736x1280_fp32_b${CODINO_TRT_BATCH_SIZE}_fixed_${CODINO_TRT_BACKBONE_PRECISION}.engine"
CODINO_TRT_QUERY_ENCODER_ENGINE="$CODINO_TRT_DIR/codino_query_encoder_b${CODINO_TRT_BATCH_SIZE}_736x1280_msda_plugin_sbc_${CODINO_TRT_QUERY_PRECISION}.engine"
CODINO_TRT_DECODER_ENGINE="$CODINO_TRT_DIR/codino_decoder_b${CODINO_TRT_BATCH_SIZE}_736x1280_msda_plugin_${CODINO_TRT_DECODER_PRECISION}.engine"
CODINO_TRT_MASK_HEAD_ENGINE="$CODINO_TRT_DIR/codino_mask_head_core_n1_736x1280_${CODINO_TRT_MASK_PRECISION}.engine"
CODINO_TRT_MANIFEST="$CODINO_TRT_DIR/codino_trt_manifest.json"

codino_trt_ready() {
  [[ -f "$CODINO_TRT_BACKBONE_ENGINE" ]] \
    && [[ -f "$CODINO_TRT_QUERY_ENCODER_ENGINE" ]] \
    && [[ -f "$CODINO_TRT_DECODER_ENGINE" ]] \
    && [[ -f "$CODINO_TRT_MASK_HEAD_ENGINE" ]]
}

if [[ "$REBUILD_CODINO_TRT" == "1" ]] || { [[ "$REBUILD_CODINO_TRT" == "auto" ]] && ! codino_trt_ready; }; then
  echo "[SETUP] rebuilding Co-DINO TensorRT engines for local GPU batch=$CODINO_TRT_BATCH_SIZE"
  PYTHON="$PY" \
    CODINO_TRT_BATCH_SIZE="$CODINO_TRT_BATCH_SIZE" \
    BACKBONE_PRECISION="$CODINO_TRT_BACKBONE_PRECISION" \
    QUERY_PRECISION="$CODINO_TRT_QUERY_PRECISION" \
    DECODER_PRECISION="$CODINO_TRT_DECODER_PRECISION" \
    MASK_PRECISION="$CODINO_TRT_MASK_PRECISION" \
    "$ROOT_DIR/backend/detectors/codino/tools/rebuild_codino_trt_engines.sh"
else
  echo "[SETUP] Co-DINO TensorRT engines exist for batch=$CODINO_TRT_BATCH_SIZE"
fi

if [[ "$RUN_BATCH_BENCHMARK" == "1" ]]; then
  echo "[SETUP] benchmarking runtime batch sizes with a temporary dummy video"
  if ! BATCH_BENCHMARK_DETECTORS="$BATCH_BENCHMARK_DETECTORS" \
    BATCH_BENCHMARK_FRAMES="$BATCH_BENCHMARK_FRAMES" \
    BATCH_BENCHMARK_TIMEOUT_SEC="$BATCH_BENCHMARK_TIMEOUT_SEC" \
    DINOV3_TRT_BACKBONE_ENGINE="$ENGINE_PATH" \
    CODINO_TRT_BATCH_SIZE="$CODINO_TRT_BATCH_SIZE" \
    CODINO_TRT_BACKBONE_ENGINE="$CODINO_TRT_BACKBONE_ENGINE" \
    CODINO_TRT_QUERY_ENCODER_ENGINE="$CODINO_TRT_QUERY_ENCODER_ENGINE" \
    CODINO_TRT_DECODER_ENGINE="$CODINO_TRT_DECODER_ENGINE" \
    CODINO_TRT_MASK_HEAD_ENGINE="$CODINO_TRT_MASK_HEAD_ENGINE" \
    "$PY" "$ROOT_DIR/tools/setup/benchmark_runtime_batches.py" \
      --python "$PY" \
      --output "$BATCH_BENCHMARK_OUTPUT" \
      --engine "$ENGINE_PATH"; then
    echo "[WARN] batch benchmark failed; keeping conservative runtime-profile defaults" >&2
  fi
else
  echo "[SETUP] batch benchmark disabled"
fi
DINOV3_TRT_BACKBONE_ENGINE="$ENGINE_PATH" \
CODINO_TRT_BACKBONE_ENGINE="$CODINO_TRT_BACKBONE_ENGINE" \
CODINO_TRT_QUERY_ENCODER_ENGINE="$CODINO_TRT_QUERY_ENCODER_ENGINE" \
CODINO_TRT_DECODER_ENGINE="$CODINO_TRT_DECODER_ENGINE" \
CODINO_TRT_MASK_HEAD_ENGINE="$CODINO_TRT_MASK_HEAD_ENGINE" \
CODINO_TRT_MANIFEST="$CODINO_TRT_MANIFEST" \
DINOV3_BATCH_BENCHMARK="$BATCH_BENCHMARK_OUTPUT" \
  "$PY" "$ROOT_DIR/tools/setup/configure_runtime_profile.py" \
    --output "$DINOV3_RUNTIME_PROFILE" \
    --benchmark "$BATCH_BENCHMARK_OUTPUT"

mkdir -p "$(dirname "$GUI_RUNTIME_ENV")"
{
  echo "# Generated by tools/setup_integrated_runtime_env.sh. Do not commit."
  printf 'GUI_RUNTIME_ROOT=%q\n' "$ROOT_DIR"
  printf 'GUI_RUNTIME_ENV_DIR=%q\n' "$ENV_DIR"
  printf 'GUI_RUNTIME_PYTHON=%q\n' "$PY"
  printf 'DINOV3_RUNTIME_PROFILE=%q\n' "$DINOV3_RUNTIME_PROFILE"
  printf 'DINOV3_BATCH_BENCHMARK=%q\n' "$BATCH_BENCHMARK_OUTPUT"
  printf 'DINOV3_TRT_BACKBONE_ENGINE=%q\n' "$ENGINE_PATH"
  printf 'CODINO_TRT_BATCH_SIZE=%q\n' "$CODINO_TRT_BATCH_SIZE"
  printf 'CODINO_TRT_BACKBONE_ENGINE=%q\n' "$CODINO_TRT_BACKBONE_ENGINE"
  printf 'CODINO_TRT_QUERY_ENCODER_ENGINE=%q\n' "$CODINO_TRT_QUERY_ENCODER_ENGINE"
  printf 'CODINO_TRT_DECODER_ENGINE=%q\n' "$CODINO_TRT_DECODER_ENGINE"
  printf 'CODINO_TRT_MASK_HEAD_ENGINE=%q\n' "$CODINO_TRT_MASK_HEAD_ENGINE"
  printf 'ATOSYORI_REPO=%q\n' "$ATOSYORI_REPO"
  printf 'DINO_RUNTIME_DIR=%q\n' "$DINO_RUNTIME_DIR"
  printf 'QT_UI_DIR=%q\n' "$ROOT_DIR/apps/qt_ui"
} > "$GUI_RUNTIME_ENV"
echo "[SETUP] wrote GUI runtime env: $GUI_RUNTIME_ENV"
DINOV3_TRT_BACKBONE_ENGINE="$ENGINE_PATH" \
CODINO_TRT_BACKBONE_ENGINE="$CODINO_TRT_BACKBONE_ENGINE" \
CODINO_TRT_QUERY_ENCODER_ENGINE="$CODINO_TRT_QUERY_ENCODER_ENGINE" \
CODINO_TRT_DECODER_ENGINE="$CODINO_TRT_DECODER_ENGINE" \
CODINO_TRT_MASK_HEAD_ENGINE="$CODINO_TRT_MASK_HEAD_ENGINE" \
  "$PY" "$ROOT_DIR/tools/artifacts/check_artifacts.py" --require-trt

if [[ "$RUN_IMPORT_CHECK" == "1" ]]; then
  "$PY" - <<'PY'
import importlib
import sys

for name in ("torch", "cv2", "orjson", "atosyori_postprocess", "mmcv", "mmdet", "tensorrt"):
    importlib.import_module(name)
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
if builder is None:
    raise SystemExit("[ERROR] TensorRT builder initialization failed")
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
