#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNTIME_ENV="${GUI_RUNTIME_ENV:-${ROOT_DIR}/.runtime/gui_runtime.env}"
LEGACY_RUNTIME_ENV="${ROOT_DIR}/configs/gui_runtime.env"
if [[ -f "${RUNTIME_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV}"
elif [[ -f "${LEGACY_RUNTIME_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${LEGACY_RUNTIME_ENV}"
fi

LOCAL_XCB_LIBS="${ROOT_DIR}/.runtime/xcb_libs/usr/lib/x86_64-linux-gnu"
if [[ -d "${LOCAL_XCB_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${LOCAL_XCB_LIBS}:${LD_LIBRARY_PATH:-}"
fi

RUNTIME_PROFILE_DEFAULT="${ROOT_DIR}/.runtime/runtime_profile.json"
if [[ ! -f "${RUNTIME_PROFILE_DEFAULT}" && -f "${ROOT_DIR}/configs/runtime_profile.json" ]]; then
  RUNTIME_PROFILE_DEFAULT="${ROOT_DIR}/configs/runtime_profile.json"
fi
BATCH_BENCHMARK_DEFAULT="${ROOT_DIR}/.runtime/runtime_benchmark.json"
if [[ ! -f "${BATCH_BENCHMARK_DEFAULT}" && -f "${ROOT_DIR}/configs/runtime_benchmark.json" ]]; then
  BATCH_BENCHMARK_DEFAULT="${ROOT_DIR}/configs/runtime_benchmark.json"
fi

export DINOV3_RUNTIME_PROFILE="${DINOV3_RUNTIME_PROFILE:-${RUNTIME_PROFILE_DEFAULT}}"
export DINOV3_BATCH_BENCHMARK="${DINOV3_BATCH_BENCHMARK:-${BATCH_BENCHMARK_DEFAULT}}"
export DINOV3_TRT_BACKBONE_ENGINE="${DINOV3_TRT_BACKBONE_ENGINE:-${ROOT_DIR}/checkpoints/trt/dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine}"

PYTHON="${GUI_RUNTIME_PYTHON:-${ROOT_DIR}/.venv_integrated/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" "${SCRIPT_DIR}/app.py" "$@"
