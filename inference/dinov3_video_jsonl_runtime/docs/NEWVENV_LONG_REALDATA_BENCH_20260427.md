# New venv long real-video inference check

Date: 2026-04-27

## Environment

- venv:
  - `/tmp/dinov3_runtime_setup_fast_test`
- setup entry:
  - `tools/setup_fast_runtime_env.sh`
- Python:
  - `3.10.18`
- PyTorch:
  - `2.11.0.dev20260204+cu129`
- CUDA:
  - `12.9`
- TensorRT:
  - `10.13.2.6`

## Input

- video:
  - `../dinov3_cascade_unified/input/アクセル様２月解析用白カン01.26.mp4`
- source frames:
  - `18000`
- source FPS:
  - `29.9700`
- source size:
  - `1920x1080`

## Command

```bash
/tmp/dinov3_runtime_setup_fast_test/bin/python \
  inference/dinov3_video_jsonl_runtime/infer_video_dinov3_jsonl.py \
  --input inference/dinov3_cascade_unified/input/アクセル様２月解析用白カン01.26.mp4 \
  --output inference/dinov3_video_jsonl_runtime/output_runs/newvenv_long_realdata_20260427_full \
  --classifier \
  --warmup-frames 300 \
  --batch-size 8 \
  --mask-approx none \
  --json-backend orjson \
  --async-writer \
  --overwrite
```

## Speed

- processed frames:
  - `18000`
- wall FPS:
  - `44.05`
- measured FPS:
  - `44.61`
- wall ms/frame:
  - `22.70`
- measured ms/frame:
  - `22.42`

## Output

The local `output_runs/newvenv_long_realdata_20260427_full/` directory was generated during validation and later removed from the GitHub-ready runtime cleanup. The metrics below are preserved from that run.

- JSONL:
  - `output_runs/newvenv_long_realdata_20260427_full/jsonl/アクセル様２月解析用白カン01.26.jsonl`
- summary:
  - `output_runs/newvenv_long_realdata_20260427_full/summary.json`
- classification validation:
  - `output_runs/newvenv_long_realdata_20260427_full/classification_validation.json`
- JSONL lines:
  - `18000`
- JSONL size:
  - `547,937,513 bytes`

## Classification Validation

- classifier enabled:
  - `true`
- classifier model:
  - `rich_spatial_attn_no_expanded_fusion`
- classifier val macro F1:
  - `0.9805877566561362`
- detections:
  - `36774`
- frames with detections:
  - `16692`
- detections/frame:
  - `2.043`
- missing required detection fields:
  - none

Class counts:

| class | category_id | count |
|---|---:|---:|
| 女性器 | 1 | 26840 |
| 男性器 | 2 | 5736 |
| 結合部分 | 3 | 4198 |

Class score stats:

| class | n | min | mean | max |
|---|---:|---:|---:|---:|
| 女性器 | 26840 | 0.3403 | 0.9606 | 0.9982 |
| 男性器 | 5736 | 0.3643 | 0.9155 | 0.9987 |
| 結合部分 | 4198 | 0.3644 | 0.9474 | 0.9971 |

## Conclusion

The new venv setup successfully ran the full 18000-frame real video with TensorRT BF16 DINOv3 backbone and the optional classifier enabled. JSONL line count matched the input frame count, all detections had classification fields, and all three expected classes appeared in the output.
