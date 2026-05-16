#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/input" "$ROOT_DIR/output"
exec "$ROOT_DIR/tools/setup/setup_gui_runtime.sh" "$@"
