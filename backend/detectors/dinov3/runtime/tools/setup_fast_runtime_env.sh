#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RUNTIME_DIR/../../../.." && pwd)"

ENV_DIR="${ENV_DIR:-$RUNTIME_DIR/.venv_fast}"
BASE_PYTHON="${BASE_PYTHON:-}"
if [[ -z "${REFERENCE_VENV:-}" ]]; then
  for candidate in \
    "$REPO_ROOT/.venv_integrated" \
    "$REPO_ROOT/../eva02_cascade_experimental/venv"; do
    if [[ "$candidate" != "$ENV_DIR" && -x "$candidate/bin/python" ]]; then
      REFERENCE_VENV="$candidate"
      break
    fi
  done
fi
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_FRAMES="${SMOKE_FRAMES:-8}"
SMOKE_INPUT="${SMOKE_INPUT:-$REPO_ROOT/input/アクセル様２月解析用白カン01.26.mp4}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-$RUNTIME_DIR/output_runs/setup_smoke}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-}"
REBUILD_TRT="${REBUILD_TRT:-auto}"
ENGINE_PATH="${ENGINE_PATH:-$REPO_ROOT/checkpoints/trt/dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine}"
TRT_PRECISION="${TRT_PRECISION:-bf16}"
TRT_FALLBACK_FP16="${TRT_FALLBACK_FP16:-1}"
TENSORRT_PIP_SPEC="${TENSORRT_PIP_SPEC:-tensorrt==10.13.0.35}"

python_works() {
  local candidate="$1"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info < (3, 12) else 1)' >/dev/null 2>&1
}

if [[ -z "$BASE_PYTHON" ]]; then
  if [[ -n "${REFERENCE_VENV:-}" && -x "$REFERENCE_VENV/bin/python" ]]; then
    BASE_PYTHON="$REFERENCE_VENV/bin/python"
  else
    for candidate in \
      "$REPO_ROOT/.venv_integrated/bin/python" \
      "$(command -v python3.10 2>/dev/null || true)" \
      "$(command -v python3.11 2>/dev/null || true)" \
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
  echo "[ERROR] no working Python 3.10/3.11 found. Set BASE_PYTHON=/path/to/python3.10." >&2
  exit 2
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

if [[ -n "${REFERENCE_VENV:-}" && -x "$REFERENCE_VENV/bin/python" ]]; then
  REF_SITES="$("$REFERENCE_VENV/bin/python" - <<'PY'
import sys
from pathlib import Path

seen = []
for raw in sys.path:
    path = Path(raw)
    if "site-packages" not in raw or not path.is_dir():
        continue
    text = str(path)
    if text not in seen:
        seen.append(text)
print("\n".join(seen))
PY
)"
  if [[ -n "$REF_SITES" ]]; then
    : > "$SITE_DIR/_dinov3_reference_runtime.pth"
    while IFS= read -r ref_site; do
      if [[ -n "$ref_site" && -d "$ref_site" && "$ref_site" != "$SITE_DIR" ]]; then
        echo "$ref_site" >> "$SITE_DIR/_dinov3_reference_runtime.pth"
      fi
    done <<< "$REF_SITES"
    echo "[SETUP] reference site-packages:"
    sed 's/^/[SETUP]   /' "$SITE_DIR/_dinov3_reference_runtime.pth"
  fi
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
  ninja \
  "$TENSORRT_PIP_SPEC"

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
import tensorrt as trt
print(f"[CHECK] python={sys.version.split()[0]}")
print(f"[CHECK] torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] CUDA is not available; fast TensorRT inference requires CUDA")
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
if builder is None:
    raise SystemExit("[ERROR] TensorRT builder initialization failed")
print(f"[CHECK] tensorrt={trt.__version__} builder_ok=True")
PY

if [[ -z "$SMOKE_BATCH_SIZE" ]]; then
  SMOKE_BATCH_SIZE="$("$PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    print(1)
else:
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gib < 10:
        print(2)
    elif total_gib < 16:
        print(4)
    elif total_gib < 24:
        print(6)
    else:
        print(8)
PY
)"
fi
echo "[SETUP] DINOv3 smoke batch size: $SMOKE_BATCH_SIZE"

rebuild_trt_engine() {
  local reason="$1"
  echo "[SETUP] TensorRT engine rebuild: $reason"
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
}

validate_trt_engine() {
  [[ -f "$ENGINE_PATH" ]] || return 1
  "$PY" - "$ENGINE_PATH" <<'PY'
import sys
from pathlib import Path

import tensorrt as trt

engine_path = Path(sys.argv[1])
logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise SystemExit(1)
print(f"[SETUP] TensorRT engine deserialize ok: {engine_path}")
PY
}

if [[ "$REBUILD_TRT" == "1" ]]; then
  rebuild_trt_engine "missing or requested: $ENGINE_PATH"
elif [[ "$REBUILD_TRT" == "auto" ]]; then
  if ! validate_trt_engine; then
    rm -f "$ENGINE_PATH" "${ENGINE_PATH%.engine}.json"
    rebuild_trt_engine "missing or incompatible: $ENGINE_PATH"
  fi
else
  echo "[SETUP] TensorRT engine exists: $ENGINE_PATH"
fi

run_dino_smoke() {
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
    --batch-size "$SMOKE_BATCH_SIZE" \
    --mask-approx simple \
    --json-backend orjson \
    --async-writer \
    --overwrite
}

validate_dino_smoke() {
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
}

if [[ "$RUN_SMOKE" == "1" ]]; then
  if [[ ! -f "$SMOKE_INPUT" ]]; then
    echo "[SETUP] smoke input not found; skipped: $SMOKE_INPUT"
  else
    if ! run_dino_smoke; then
      if [[ "$REBUILD_TRT" == "auto" ]]; then
        echo "[WARN] DINOv3 TensorRT smoke failed; rebuilding engine once and retrying"
        rm -f "$ENGINE_PATH" "${ENGINE_PATH%.engine}.json"
        rebuild_trt_engine "smoke failed, likely incompatible serialized engine"
        run_dino_smoke
      else
        exit 1
      fi
    fi
    validate_dino_smoke
  fi
fi

cat <<EOF

[DONE] Fast DINOv3 runtime environment is ready.

Use:
  $PY $RUNTIME_DIR/infer_video_dinov3_jsonl.py --input INPUT.mp4 --output OUTPUT_DIR --classifier --batch-size $SMOKE_BATCH_SIZE --mask-approx none --json-backend orjson --async-writer --overwrite

Disable classifier:
  --no-classifier

Enable overlay:
  --write-overlay
EOF
