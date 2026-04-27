# Flow

This runtime is a self-contained bundle for:

```text
video
  -> DINOv3 + Cascade Mask R-CNN
  -> optional ROI classifier
  -> JSONL
  -> Atosyori raw preprocessing / tracking
  -> ellipse or polygon postprocess by class policy
  -> predictions SQLite
  -> optional overlay video
```

## 1. DINOv3 inference

Entrypoint:

```text
scripts/infer_video_postprocess.py
scripts/run_integrated_pipeline.py
```

`infer_video_postprocess.py` is the short production-style wrapper. It always runs postprocess and enables overlay only when `--overlay` is specified. `run_integrated_pipeline.py` exposes the detailed detector/postprocess options.

It calls:

```text
inference/dinov3_video_jsonl_runtime/infer_video_dinov3_jsonl.py
```

The DINOv3 runtime uses bundled code under:

```text
scripts/train_dinov3_cascade_unified.py
configs/paths.py
dinov3/
eva02/eva02_det/
```

Default acceleration:

- TensorRT BF16 DINOv3 backbone
- Cascade box head reduced to 1 stage
- RPN test top-k `100/40`
- BF16 autocast
- batch size `8`
- async JSONL writer

Output:

```text
<run>/dinov3/jsonl/<video_stem>.jsonl
<run>/dinov3/summary.json
```

## 2. ROI classification

Classification is enabled by default and uses:

```text
checkpoints/classifier/best.pt
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

## 4. Class policy

Default:

```text
configs/class_policy_default.json
```

Current policy:

- `男性器`: polygon, target interval 3
- `女性器`: ellipse, target interval 6
- `結合部分`: ellipse, target interval 6
- fallback: ellipse, target interval 6

For environments without the polygon predictor checkpoint, use:

```text
configs/class_policy_ellipse_only.json
```

## 5. Final artifacts

The integration symlinks important outputs into stable locations:

```text
<run>/sqlite/<video_stem>_tracked.sqlite
<run>/sqlite/<video_stem>_int_<N>_predictions.sqlite
<run>/overlay/<video_stem>_int_<N>_postprocess.mp4
<run>/summary.json
```

The full raw postprocess tree remains under:

```text
<run>/postprocess/
```
