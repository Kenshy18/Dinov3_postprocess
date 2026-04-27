#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUNTIME_DIR/../.." && pwd)"

ENV_DIR="${ENV_DIR:-$RUNTIME_DIR/.venv_fast}"
BASE_PYTHON="${BASE_PYTHON:-}"
if [[ -z "${REFERENCE_VENV:-}" ]]; then
  for candidate in \
    "$REPO_ROOT/../eva02_cascade_experimental/venv" \
    "/home/kenke/workspace/CV/unified_training_codino_eva02/inference/eva02_cascade_experimental/venv"; do
    if [[ -x "$candidate/bin/python" ]]; then
      REFERENCE_VENV="$candidate"
      break
    fi
  done
fi
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_FRAMES="${SMOKE_FRAMES:-8}"
SMOKE_INPUT="${SMOKE_INPUT:-$REPO_ROOT/input/アクセル様２月解析用白カン01.26.mp4}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-$RUNTIME_DIR/output_runs/setup_smoke}"
REBUILD_TRT="${REBUILD_TRT:-auto}"
ENGINE_PATH="${ENGINE_PATH:-$REPO_ROOT/checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine}"
TRT_PRECISION="${TRT_PRECISION:-bf16}"
TRT_FALLBACK_FP16="${TRT_FALLBACK_FP16:-1}"

if [[ -z "$BASE_PYTHON" ]]; then
  if [[ -n "${REFERENCE_VENV:-}" && -x "$REFERENCE_VENV/bin/python" ]]; then
    BASE_PYTHON="$REFERENCE_VENV/bin/python"
  elif [[ -x /home/kenke/miniconda3/envs/eva02_trt/bin/python ]]; then
    BASE_PYTHON=/home/kenke/miniconda3/envs/eva02_trt/bin/python
  elif command -v python3.10 >/dev/null 2>&1; then
    BASE_PYTHON="$(command -v python3.10)"
  else
    BASE_PYTHON="$(command -v python3)"
  fi
fi

echo "[SETUP] runtime: $RUNTIME_DIR"
echo "[SETUP] env:     $ENV_DIR"
echo "[SETUP] python:  $BASE_PYTHON"

if [[ ! -d "$ENV_DIR" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
SITE_DIR="$("$PY" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

if [[ -n "${REFERENCE_VENV:-}" && -d "$REFERENCE_VENV/lib/python3.10/site-packages" ]]; then
  REF_SITE="$(cd "$REFERENCE_VENV/lib/python3.10/site-packages" && pwd)"
  echo "$REF_SITE" > "$SITE_DIR/_dinov3_reference_runtime.pth"
  echo "[SETUP] reference site-packages: $REF_SITE"
else
  echo "[SETUP] reference venv not found; using BASE_PYTHON/system packages only"
fi

"$PY" -m pip install -q \
  orjson \
  einops \
  submitit \
  ftfy \
  regex \
  matplotlib \
  pycocotools \
  fvcore \
  iopath \
  omegaconf \
  hydra-core \
  timm \
  fairscale \
  opencv-python \
  ninja

"$PY" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "torchvision",
    "cv2",
    "tensorrt",
    "orjson",
    "pycocotools",
    "fvcore",
    "iopath",
    "omegaconf",
    "hydra",
    "timm",
    "fairscale",
    "einops",
]

missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc!r}")

if missing:
    print("[ERROR] missing runtime dependencies:")
    for item in missing:
        print("  -", item)
    raise SystemExit(2)

import torch
print(f"[CHECK] python={sys.version.split()[0]}")
print(f"[CHECK] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is not available; fast TensorRT inference requires CUDA")
PY

if [[ "$REBUILD_TRT" == "1" || ( "$REBUILD_TRT" == "auto" && ! -f "$ENGINE_PATH" ) ]]; then
  echo "[SETUP] TensorRT engine missing or rebuild requested; rebuilding: $ENGINE_PATH"
  echo "[SETUP] TensorRT precision: $TRT_PRECISION"
  if ! PYTHON="$PY" TRT_PRECISION="$TRT_PRECISION" "$TOOLS_DIR/rebuild_default_trt_backbone.sh"; then
    if [[ "$TRT_PRECISION" == "bf16" && "$TRT_FALLBACK_FP16" == "1" ]]; then
      echo "[WARN] BF16 TensorRT build failed; retrying with FP16 at the same engine path"
      rm -f "$ENGINE_PATH" "${ENGINE_PATH%.engine}.json"
      PYTHON="$PY" TRT_PRECISION=fp16 "$TOOLS_DIR/rebuild_default_trt_backbone.sh"
    else
      exit 1
    fi
  fi
else
  echo "[SETUP] TensorRT engine exists: $ENGINE_PATH"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  if [[ ! -f "$SMOKE_INPUT" ]]; then
    echo "[SETUP] smoke input not found; skipped: $SMOKE_INPUT"
  else
    rm -rf "$SMOKE_OUTPUT"
    "$PY" "$RUNTIME_DIR/infer_video_dinov3_jsonl.py" \
      --input "$SMOKE_INPUT" \
      --output "$SMOKE_OUTPUT" \
      --classifier \
      --checkpoint "$REPO_ROOT/checkpoints/detector/model_final.pth" \
      --backbone-weights "$REPO_ROOT/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
      --classifier-checkpoint "$REPO_ROOT/checkpoints/classifier/best.pt" \
      --trt-backbone-engine "$ENGINE_PATH" \
      --max-frames "$SMOKE_FRAMES" \
      --warmup-frames 0 \
      --batch-size 8 \
      --mask-approx simple \
      --json-backend orjson \
      --async-writer \
      --overwrite

    "$PY" - "$SMOKE_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
run = summary["runs"][0]
jsonl = Path(run["output_jsonl"])
lines = sum(1 for _ in jsonl.open("rb"))
print(f"[SMOKE] processed_frames={run['processed_frames']} jsonl_lines={lines} measured_fps={run['measured_fps']:.2f}")
if lines != int(run["processed_frames"]):
    raise SystemExit("[ERROR] JSONL line count does not match processed frame count")
PY
  fi
fi

cat <<EOF

[DONE] Fast DINOv3 runtime environment is ready.

Use:
  $PY $RUNTIME_DIR/infer_video_dinov3_jsonl.py --input INPUT.mp4 --output OUTPUT_DIR --classifier --batch-size 8 --mask-approx none --json-backend orjson --async-writer --overwrite

Disable classifier:
  --no-classifier

Enable overlay:
  --write-overlay
EOF
