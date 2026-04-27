# DINOv3 video JSONL runtime

動画を入力に、DINOv3 + Cascade Mask R-CNN の推論を行い、JSONLを出力する実行用ディレクトリです。
分類モジュールはオプションです。overlay mp4もオプションです。

## Main script

- `infer_video_dinov3_jsonl.py`

## Included runtime files

- `infer_video_dinov3_jsonl.py`
  - 動画入力、JSONL出力、optional classifier、optional overlayの実行入口。
- `infer_images_singleclass.py`
  - DINOv3 Cascade model build / preprocess / batch loader / TensorRT backbone接続に必要。
- `trt_backbone.py`
  - native TensorRT engineをPyTorch moduleとして呼ぶadapter。
- `onnx_backbone.py`
  - TensorRT runtime/vendor lib path setup helper。
- `tools/export_dinov3_backbone_onnx.py`
  - コピー先GPU/環境でTensorRT engineを作り直すためのONNX export入口。
- `tools/build_trt_backbone_engine.py`
  - ONNXからnative TensorRT engineを作る入口。
- `tools/rebuild_default_trt_backbone.sh`
  - default設定のONNX export + TensorRT BF16 engine rebuildコマンド。
- `tools/setup_fast_runtime_env.sh`
  - 新しいvenvを作成し、高速推論に必要な依存確認、必要ならTRT engine再作成、smoke testまで行うセットアップ入口。
- `PORTABILITY.md`
  - 他PCへ持ち出す際の注意点。
- `two_stage_multiclass_20260426/scripts/two_stage_roi_classifier.py`
  - optional classifierのモデル定義とcheckpoint loader。
- `output/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine`
  - TensorRT BF16 DINOv3 backbone engineへのsymlink。
- `two_stage_multiclass_20260426/outputs/.../checkpoints/best.pt`
  - no-expanded rich attention classifier checkpointへのsymlink。
- `checkpoints/detector/last_checkpoint`
  - detector checkpointのdefault resolver用。
- `checkpoints/detector/model_final.pth`
  - DINOv3 Cascade detector checkpointへのsymlink。
- `checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
  - DINOv3 pretrained weightsへのsymlink。

The detector checkpoint, DINOv3 pretrained weights, TensorRT engine, and classifier checkpoint are all reachable from this runtime directory. Large files are symlinked instead of copied.

## Usage

初回セットアップとsmoke test:

```bash
inference/dinov3_video_jsonl_runtime/tools/setup_fast_runtime_env.sh
```

別PCでは `BASE_PYTHON=/path/to/python3.10` を指定できます。

```bash
BASE_PYTHON=/path/to/python3.10 inference/dinov3_video_jsonl_runtime/tools/setup_fast_runtime_env.sh
```

Classifier enabled is the default:

```bash
inference/dinov3_video_jsonl_runtime/.venv_fast/bin/python \
  inference/dinov3_video_jsonl_runtime/infer_video_dinov3_jsonl.py \
  --input inference/dinov3_cascade_unified/input/アクセル様２月解析用白カン01.26.mp4 \
  --output /tmp/dinov3_jsonl_out \
  --batch-size 8 \
  --mask-approx none \
  --json-backend orjson \
  --async-writer \
  --overwrite
```

Classifier disabled:

```bash
inference/dinov3_video_jsonl_runtime/.venv_fast/bin/python \
  inference/dinov3_video_jsonl_runtime/infer_video_dinov3_jsonl.py \
  --input inference/dinov3_cascade_unified/input/アクセル様２月解析用白カン01.26.mp4 \
  --output /tmp/dinov3_jsonl_out_no_classifier \
  --no-classifier \
  --batch-size 8 \
  --mask-approx none \
  --json-backend orjson \
  --async-writer \
  --overwrite
```

Overlay output:

```bash
--write-overlay
```

## Outputs

For each input video:

- `jsonl/<video_stem>.jsonl`
- `overlay/<video_stem>.mp4` only when `--write-overlay` is set
- `summary.json`

## Default acceleration

- TensorRT BF16 DINOv3 backbone
- Cascade box head reduced to 1 stage
- RPN test top-k `pre=100`, `post=40`
- BF16 autocast
- batch size default `8`
- async JSONL writer

## Portability

他PCへコピーする場合は [PORTABILITY.md](PORTABILITY.md) を参照してください。
TensorRT engineはGPU/CUDA/TensorRT依存が強いため、コピー先で `tools/rebuild_default_trt_backbone.sh` により再作成することを推奨します。
