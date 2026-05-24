#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
BUNDLE_ROOT="$(cd "$RUNTIME_DIR/../../../.." && pwd)"
cd "$RUNTIME_DIR"

PYTHON="${PYTHON:-$BUNDLE_ROOT/.venv_integrated/bin/python}"
ONNX_PATH="${ONNX_PATH:-$BUNDLE_ROOT/output/onnx/dinov3_backbone_fp32_720x1280_dynamic.onnx}"
ENGINE_PATH="${ENGINE_PATH:-$BUNDLE_ROOT/checkpoints/trt/dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine}"
CHECKPOINT="${CHECKPOINT:-$BUNDLE_ROOT/checkpoints/detector/model_final.pth}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-$BUNDLE_ROOT/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
TRT_PRECISION="${TRT_PRECISION:-bf16}"
TRT_FORCE_LAYER_PRECISION="${TRT_FORCE_LAYER_PRECISION:-1}"

mkdir -p "$(dirname "$ONNX_PATH")" "$(dirname "$ENGINE_PATH")"

"$PYTHON" "$TOOLS_DIR/export_dinov3_backbone_onnx.py" \
  --checkpoint "$CHECKPOINT" \
  --backbone-weights "$BACKBONE_WEIGHTS" \
  --output "$ONNX_PATH" \
  --target-size 720x1280 \
  --batch-size 8 \
  --no-fp16 \
  --no-verify-ort \
  --no-verify-trt

build_args=(
  "$TOOLS_DIR/build_trt_backbone_engine.py"
  --onnx "$ONNX_PATH" \
  --engine "$ENGINE_PATH" \
  --precision "$TRT_PRECISION" \
  --min-shape 1x3x720x1280 \
  --opt-shape 8x3x720x1280 \
  --max-shape 8x3x720x1280 \
  --workspace-gb 8
)

if [[ "$TRT_FORCE_LAYER_PRECISION" == "1" ]]; then
  build_args+=(--force-layer-precision)
fi

"$PYTHON" "${build_args[@]}"
