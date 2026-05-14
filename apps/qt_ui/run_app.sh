#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNTIME_ENV="${ROOT_DIR}/configs/gui_runtime.env"
if [[ -f "${RUNTIME_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV}"
fi

export DINOV3_RUNTIME_PROFILE="${DINOV3_RUNTIME_PROFILE:-${ROOT_DIR}/configs/runtime_profile.json}"
export DINOV3_BATCH_BENCHMARK="${DINOV3_BATCH_BENCHMARK:-${ROOT_DIR}/configs/runtime_benchmark.json}"
export DINOV3_TRT_BACKBONE_ENGINE="${DINOV3_TRT_BACKBONE_ENGINE:-${ROOT_DIR}/checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine}"

PYTHON="${GUI_RUNTIME_PYTHON:-${ROOT_DIR}/.venv_integrated/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" "${SCRIPT_DIR}/app.py" "$@"
