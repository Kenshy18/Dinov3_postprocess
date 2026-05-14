#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
exec "$ROOT_DIR/backend/detectors/dinov3/runtime/tools/rebuild_default_trt_backbone.sh" "$@"
