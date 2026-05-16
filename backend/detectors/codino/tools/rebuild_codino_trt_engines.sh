#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
BUNDLE_ROOT="$(cd "$TOOLS_DIR/../../../.." && pwd)"

PYTHON="${PYTHON:-$BUNDLE_ROOT/.venv_integrated/bin/python}"
CONFIG="${CONFIG:-$BUNDLE_ROOT/checkpoints/codino/detector/resolved_config.py}"
CHECKPOINT="${CHECKPOINT:-$BUNDLE_ROOT/checkpoints/codino/detector/epoch_2.pth}"
ONNX_DIR="${ONNX_DIR:-$BUNDLE_ROOT/output/onnx}"
ENGINE_DIR="${ENGINE_DIR:-$BUNDLE_ROOT/checkpoints/codino/trt}"
CODINO_TRT_BATCH_SIZE="${CODINO_TRT_BATCH_SIZE:-2}"
INPUT_HEIGHT="${INPUT_HEIGHT:-736}"
INPUT_WIDTH="${INPUT_WIDTH:-1280}"
IMG_HEIGHT="${IMG_HEIGHT:-720}"
IMG_WIDTH="${IMG_WIDTH:-1280}"
FEATURE_SHAPES="${FEATURE_SHAPES:-184x320,92x160,46x80,23x40,12x20}"
BACKBONE_PRECISION="${BACKBONE_PRECISION:-bf16}"
QUERY_PRECISION="${QUERY_PRECISION:-fp16}"
DECODER_PRECISION="${DECODER_PRECISION:-fp16}"
MASK_PRECISION="${MASK_PRECISION:-fp16}"
WORKSPACE_GB="${WORKSPACE_GB:-12}"
BUILD_BACKBONE="${BUILD_BACKBONE:-1}"
BUILD_QUERY_ENCODER="${BUILD_QUERY_ENCODER:-1}"
BUILD_DECODER="${BUILD_DECODER:-1}"
BUILD_MASK_HEAD="${BUILD_MASK_HEAD:-1}"
FORCE_LAYER_PRECISION="${FORCE_LAYER_PRECISION:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] Python not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Co-DINO config missing: $CONFIG" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Co-DINO checkpoint missing: $CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$ONNX_DIR" "$ENGINE_DIR"

BATCH="$CODINO_TRT_BATCH_SIZE"
BACKBONE_ONNX="$ONNX_DIR/codino_dinov3_vitl_backbone_${INPUT_HEIGHT}x${INPUT_WIDTH}_fp32_b${BATCH}_fixed.onnx"
BACKBONE_ENGINE="$ENGINE_DIR/codino_dinov3_vitl_backbone_${INPUT_HEIGHT}x${INPUT_WIDTH}_fp32_b${BATCH}_fixed_${BACKBONE_PRECISION}.engine"
QUERY_ONNX="$ONNX_DIR/codino_query_encoder_b${BATCH}_${INPUT_HEIGHT}x${INPUT_WIDTH}_msda_trt_plugin_sbc.onnx"
QUERY_ENGINE="$ENGINE_DIR/codino_query_encoder_b${BATCH}_${INPUT_HEIGHT}x${INPUT_WIDTH}_msda_plugin_sbc_${QUERY_PRECISION}.engine"
DECODER_ONNX="$ONNX_DIR/codino_decoder_b${BATCH}_${INPUT_HEIGHT}x${INPUT_WIDTH}_msda_trt_plugin.onnx"
DECODER_ENGINE="$ENGINE_DIR/codino_decoder_b${BATCH}_${INPUT_HEIGHT}x${INPUT_WIDTH}_msda_plugin_${DECODER_PRECISION}.engine"
MASK_ONNX="$ONNX_DIR/codino_mask_head_core_n1_${INPUT_HEIGHT}x${INPUT_WIDTH}.onnx"
MASK_ENGINE="$ENGINE_DIR/codino_mask_head_core_n1_${INPUT_HEIGHT}x${INPUT_WIDTH}_${MASK_PRECISION}.engine"

echo "[CODINO-TRT] root:       $BUNDLE_ROOT"
echo "[CODINO-TRT] python:     $PYTHON"
echo "[CODINO-TRT] batch:      $BATCH"
echo "[CODINO-TRT] config:     $CONFIG"
echo "[CODINO-TRT] checkpoint: $CHECKPOINT"

if [[ "$BUILD_BACKBONE" == "1" ]]; then
  "$PYTHON" "$TOOLS_DIR/export_codino_dinov3_backbone_onnx.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$BACKBONE_ONNX" \
    --height "$INPUT_HEIGHT" \
    --width "$INPUT_WIDTH" \
    --batch-size "$BATCH" \
    --fixed-batch

  backbone_build_args=(
    "$BUNDLE_ROOT/backend/detectors/dinov3/runtime/tools/build_trt_backbone_engine.py"
    --onnx "$BACKBONE_ONNX"
    --engine "$BACKBONE_ENGINE"
    --precision "$BACKBONE_PRECISION"
    --min-shape "${BATCH}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
    --opt-shape "${BATCH}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
    --max-shape "${BATCH}x3x${INPUT_HEIGHT}x${INPUT_WIDTH}"
    --workspace-gb "$WORKSPACE_GB"
  )
  if [[ "$FORCE_LAYER_PRECISION" == "1" ]]; then
    backbone_build_args+=(--force-layer-precision)
  fi
  "$PYTHON" "${backbone_build_args[@]}"
fi

if [[ "$BUILD_QUERY_ENCODER" == "1" ]]; then
  "$PYTHON" "$TOOLS_DIR/export_codino_query_encoder_trt.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --onnx "$QUERY_ONNX" \
    --engine "$QUERY_ENGINE" \
    --batch-size "$BATCH" \
    --input-height "$INPUT_HEIGHT" \
    --input-width "$INPUT_WIDTH" \
    --img-height "$IMG_HEIGHT" \
    --img-width "$IMG_WIDTH" \
    --feature-shapes "$FEATURE_SHAPES" \
    --precision "$QUERY_PRECISION"
fi

if [[ "$BUILD_DECODER" == "1" ]]; then
  "$PYTHON" "$TOOLS_DIR/export_codino_decoder_trt.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --onnx "$DECODER_ONNX" \
    --engine "$DECODER_ENGINE" \
    --batch-size "$BATCH" \
    --input-height "$INPUT_HEIGHT" \
    --input-width "$INPUT_WIDTH" \
    --img-height "$IMG_HEIGHT" \
    --img-width "$IMG_WIDTH" \
    --feature-shapes "$FEATURE_SHAPES" \
    --precision "$DECODER_PRECISION"
fi

if [[ "$BUILD_MASK_HEAD" == "1" ]]; then
  "$PYTHON" "$TOOLS_DIR/export_codino_mask_head_trt.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --mode core \
    --onnx "$MASK_ONNX" \
    --engine "$MASK_ENGINE" \
    --batch-size "$BATCH" \
    --num-rois 1 \
    --precision "$MASK_PRECISION"
fi

"$PYTHON" - "$ENGINE_DIR/codino_trt_manifest.json" "$BATCH" "$BACKBONE_ENGINE" "$QUERY_ENGINE" "$DECODER_ENGINE" "$MASK_ENGINE" <<'PY'
import json
import sys
from pathlib import Path

manifest = {
    "batch_size": int(sys.argv[2]),
    "engines": {
        "backbone": sys.argv[3],
        "query_encoder": sys.argv[4],
        "decoder": sys.argv[5],
        "mask_head": sys.argv[6],
    },
}
path = Path(sys.argv[1])
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[CODINO-TRT] wrote {path}")
PY

cat <<EOF

[DONE] Co-DINO TensorRT engines are ready.
  backbone:      $BACKBONE_ENGINE
  query encoder: $QUERY_ENGINE
  decoder:       $DECODER_ENGINE
  mask head:     $MASK_ENGINE
EOF
