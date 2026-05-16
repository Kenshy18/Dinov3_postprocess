# Flow

This runtime is a self-contained bundle for:

```text
video
  -> DINOv3 or EVA02 + Cascade Mask R-CNN
  -> optional ROI classifier
  -> JSONL
  -> Atosyori raw preprocessing / tracking
  -> ellipse or polygon postprocess by class policy
  -> predictions SQLite
  -> optional overlay video
```

## 1. Detector Inference

Primary entrypoints:

```text
tools/setup_runtime.sh
scripts/infer.py
scripts/overlay.py
```

`backend/pipeline/cli/infer.py` owns the recommended end-to-end wrapper. `scripts/infer.py` is the compatibility command users should call. It keeps the common production choices visible: detector, mode (`full`, `inference`, `postprocess`), overlay mode, default shape mode, keyframe interval, recall target, and compact per-class overrides.

`backend/pipeline/cli/overlay.py` owns the standalone overlay wrapper. It accepts direct video + raw JSONL/final SQLite inputs, or completed run directories.

`run_full_flow.py`, `run_postprocess_only.py`, and `infer_video_postprocess.py` are older wrappers. `backend/pipeline/run_integrated_pipeline.py` owns the detailed low-level integration logic; `scripts/run_integrated_pipeline.py` is kept as a compatibility wrapper.

Example end-to-end run:

```bash
.venv_integrated/bin/python scripts/infer.py \
  --input input/short/0210_first30s.mp4 \
  --run-name sample_full_flow \
  --detector dinov3 \
  --shape-mode ellipse \
  --keyframe-interval 3 \
  --recall-target 0.96 \
  --class-policy female:polygon:5:0.97 \
  --post-overlay detailed \
  --pre-overlay \
  --force
```

Example postprocess-only run:

```bash
.venv_integrated/bin/python scripts/infer.py \
  --mode postprocess \
  --input-jsonl output/runs/sample_full_flow/dinov3/jsonl/0210_first30s.jsonl \
  --input-video input/short/0210_first30s.mp4 \
  --run-name sample_postprocess_only \
  --shape-mode ellipse \
  --keyframe-interval 3 \
  --recall-target 0.96 \
  --post-overlay none \
  --force
```

Per-class overrides use:

```text
--class-policy CLASS:MODE[:INTERVAL[:RECALL]]
```

`CLASS` can be `female`, `male`, `junction`, or an exact label such as `女性器`. `junction` updates both `結合部分` and `結合`. The wrapper writes the generated JSON to `<run>/config/class_policy.generated.json`.

Inference outputs are split into two explicit phases:

```text
pre-postprocess:  --pre-sqlite / --pre-overlay
postprocess:      --post-sqlite / --post-overlay {none,detailed,simple,both}
```

`--raw-sqlite`, `--raw-overlay`, and `--overlay-mode` are compatibility aliases.

The primary wrappers stream child process progress to the terminal and also write logs:

```text
<run>/logs/infer_cli.log
<run>/logs/ui_job.log
```

Use `--detector dinov3`, `--detector eva02`, or `--detector codino`. The default remains `dinov3`.

It calls one of:

```text
backend/detectors/dinov3/runtime/infer_video_dinov3_jsonl.py
backend/detectors/eva02/runtime/infer_video_eva02_jsonl.py
backend/detectors/codino/runtime/infer_video_codino_jsonl.py
```

The DINOv3 runtime uses bundled code under:

```text
training/dinov3/train_dinov3_cascade_unified.py
configs/paths.py
dinov3/
eva02/eva02_det/
```

`scripts/train_dinov3_cascade_unified.py` is kept as a compatibility wrapper
for existing commands and runtime imports.

Default acceleration:

- TensorRT BF16 DINOv3 backbone
- Co-DINO local TensorRT backbone/query encoder/decoder/mask head when `--detector codino`
- Cascade box head reduced to 1 stage
- RPN test top-k `100/40`
- BF16 autocast
- batch size from `.runtime/runtime_profile.json` when setup has been run
- async JSONL writer

Output:

```text
<run>/<detector>/jsonl/<video_stem>.jsonl
<run>/<detector>/summary.json
```

## 2. ROI classification

Classification is enabled by default and uses:

```text
checkpoints/classifier/best.pt
checkpoints/eva02/classifier/best.pt
```

The classifier adds class fields to each detection in the JSONL:

- `class_name`
- `label`
- `category_id`
- `category_index`
- `class_score`

Disable it with:

```bash
--no-classifier
```

## 3. Atosyori postprocess

The bundled postprocess source lives at:

```text
external/atosyori-pipeline-dev/
```

The integration calls:

```bash
python -m atosyori_postprocess run \
  --input-jsonl <dinov3.jsonl> \
  --input-video <video.mp4> \
  --output-dir <run>/postprocess \
  --class-policy-json configs/class_policy_default.json
```

The postprocess first converts JSONL + video into tracked SQLite:

```text
<run>/postprocess/preprocess/preprocess/<video_stem>.tracked.sqlite
```

Then it groups tracks by class label and applies each class policy.

The current default postprocess profile, including endpoint extrapolation and K1/K2 routing defaults, is documented in:

```text
docs/POSTPROCESS_SETTINGS.md
configs/class_policy_all_ellipse_int3_recall096.json
```

## 4. Class policy

Default:

```text
configs/class_policy_default.json
```

Current default policy:

- `男性器`: ellipse, target interval 3, recall 0.96
- `女性器`: ellipse, target interval 3, recall 0.96
- `結合部分`: ellipse, target interval 3, recall 0.96
- fallback: ellipse, target interval 3, recall 0.96

The older `configs/class_policy_ellipse_only.json` is also ellipse-only and follows the same interval/recall defaults.

```text
configs/class_policy_ellipse_only.json
```

For a stable explicit name for the same all-ellipse profile, use:

```text
configs/class_policy_all_ellipse_int3_recall096.json
```

## 5. Final artifacts

`scripts/infer.py` uses the same user-facing layout as the GUI:

```text
<run>/最終成果物.json
<run>/最終SQLite/<label>_predictions.sqlite
<run>/推論生SQLite/<video_stem>_raw_detections.sqlite
<run>/推論生SQLite/<video_stem>.tracked.sqlite
<run>/詳細オーバーレイ/<label>_detailed.mp4
<run>/統合マスクオーバーレイ/<label>_simple.mp4
<run>/AI生成カバーオーバーレイ/<video_stem>_ai_raw_mask.mp4
<run>/jsonl/<video_stem>.jsonl
<run>/logs/infer_cli.log
<run>/logs/infer_audit.jsonl
<run>/logs/ui_job.log
<run>/summary.json
```

The low-level integration still keeps the raw detector/postprocess tree under:

```text
<run>/<detector>/
<run>/postprocess/
```

Raw detector SQLite uses the common `raw_mask_sqlite_v1` schema:

```text
metadata(key, value)
frames(frame, time_sec, width, height)
masks(frame, mask_id, detection_index, label, class_name, category_id,
      score, detector_score, class_score, bbox_xyxy, polygons, source_json)
```

Final postprocess SQLite keeps the Atosyori prediction contract and must expose
`masks(frame, track_id, polygons)` at minimum.
