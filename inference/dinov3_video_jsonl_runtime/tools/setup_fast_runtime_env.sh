#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
exec "$ROOT_DIR/backend/detectors/dinov3/runtime/tools/setup_fast_runtime_env.sh" "$@"
