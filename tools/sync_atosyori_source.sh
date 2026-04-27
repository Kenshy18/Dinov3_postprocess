#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$TOOLS_DIR/.." && pwd)"
SOURCE="${ATOSYORI_REPO:-/home/kenke/workspace/CV/atosyori-pipeline-dev}"
DEST="${DEST:-$ROOT_DIR/external/atosyori-pipeline-dev}"

if [[ ! -d "$SOURCE/src/atosyori_postprocess" ]]; then
  echo "[ERROR] Atosyori source not found: $SOURCE" >&2
  exit 2
fi

mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.venv*/' \
  --exclude 'input/*' \
  --exclude 'output/*' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  --exclude '*.sqlite' \
  --exclude '*.jsonl' \
  --exclude '*.mp4' \
  "$SOURCE/" "$DEST/"

echo "[DONE] synced Atosyori source:"
echo "  $SOURCE -> $DEST"
