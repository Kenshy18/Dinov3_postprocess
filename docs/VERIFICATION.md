# Verification

Use these checks after cleanup or refactors. The goal is to prove that wiring,
artifacts, and postprocess behavior still work without relying on memory of a
previous run.

## Quick Check

This avoids detector inference and is the default check for docs, tooling, and
wrapper refactors.

```bash
.venv_integrated/bin/python tools/verify_runtime.py
```

It runs:

- artifact placement check
- Python compile check for project-owned runtime code
- Atosyori doctor with `checkpoints/postprocess`
- Atosyori engine import check
- Atosyori unit tests
- deterministic Atosyori smoke under `/tmp/dinov3_postprocess_verify`

To require local TensorRT engines in the artifact check:

```bash
.venv_integrated/bin/python tools/verify_runtime.py --require-trt
```

## Detector Smoke

Run this when changing command construction, runtime defaults, detector
selection, artifact layout, or pipeline summary/linking code.

```bash
.venv_integrated/bin/python tools/verify_runtime.py --detector-smoke --detector dinov3 --frames 64
```

For EVA02 wiring:

```bash
.venv_integrated/bin/python tools/verify_runtime.py --detector-smoke --detector eva02 --frames 1
```

For Co-DINO wiring:

```bash
.venv_integrated/bin/python tools/verify_runtime.py --detector-smoke --detector codino --frames 8
```

## Golden Output Compare

For behavior-preserving refactors, capture a before and after run in separate
directories and compare the stable artifacts.

```bash
.venv_integrated/bin/python scripts/run_integrated_pipeline.py \
  --input "input/3月以降解析白カン動画-0210.mp4" \
  --output-root /tmp/dpp_verify_before \
  --run-name dino64 \
  --max-frames 64 \
  --warmup-frames 0 \
  --no-postprocess \
  --classifier \
  --no-async-writer \
  --force
```

Repeat with `/tmp/dpp_verify_after`, then compare the detector JSONL when the
detector path is deterministic on the target machine:

```bash
sha256sum \
  /tmp/dpp_verify_before/dino64/dinov3/jsonl/3月以降解析白カン動画-0210.jsonl \
  /tmp/dpp_verify_after/dino64/dinov3/jsonl/3月以降解析白カン動画-0210.jsonl
```

If the detector JSONL hash differs, first compare the exact command captured in
`summary.json` and the stable run counters (`processed_frames`, `detections`,
`jsonl_size_bytes`). GPU/TensorRT mask contours can vary slightly between runs
even when the command is unchanged. For refactors outside the detector runtime,
the strongest deterministic check is to run postprocess from the same fixed
JSONL before and after the change, then compare SQLite content.

Overlay MP4 byte hashes are not a reliable signal because encoders can write
nondeterministic metadata. For overlay changes, verify existence, frame count,
duration, and visual content instead.
