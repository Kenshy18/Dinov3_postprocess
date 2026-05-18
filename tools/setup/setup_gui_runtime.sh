#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

cat <<'EOF'
[SETUP] Preparing GUI runtime:
  - create/update .venv_integrated
  - install detector/postprocess/UI dependencies
  - configure repo-local RT-DETR Head/Face runtime
  - download/check artifacts when configured
  - build/reuse local DINOv3 TensorRT engine
  - build/reuse local Co-DINO fixed-batch TensorRT engines
  - benchmark local batch sizes with a temporary dummy video
    for DINOv3, EVA02, Co-DINO, and RT-DETR
  - write .runtime/runtime_profile.json and .runtime/gui_runtime.env
EOF

"$ROOT_DIR/tools/setup/setup_integrated_runtime_env.sh"

cat <<EOF

[DONE] GUI runtime is ready.

Recommended launch:
  $ROOT_DIR/apps/qt_ui/run_app.sh

Compatibility launch:
  $ROOT_DIR/UI/run_app.sh

Runtime profile:
  $ROOT_DIR/.runtime/runtime_profile.json
EOF
