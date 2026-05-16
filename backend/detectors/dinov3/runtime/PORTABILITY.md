# Portability notes

このruntimeディレクトリだけを別PCへコピーしても、そのまま完全には動きません。

理由:

- `.engine` はTensorRT/CUDA/GPU世代に強く依存するため、別PCでは再作成が安全。
- 現在の大きなファイルはDrive artifactです。clone後にセットアップで `checkpoints/` へ取得してください。
- 推論コードはrepo側の `eva02/eva02_det`, `dinov3`, `training/dinov3/train_dinov3_cascade_unified.py`, `scripts/train_dinov3_cascade_unified.py`（互換入口）, `configs` に依存します。
- Python環境には PyTorch, Detectron2系コード, TensorRT, OpenCV, orjson などが必要です。

## Minimum files needed

このruntime内で参照されるモデル類:

- detector checkpoint:
  - `checkpoints/detector/model_final.pth`
- DINOv3 pretrained weights:
  - `checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- classifier checkpoint:
  - `checkpoints/classifier/best.pt`
- TensorRT engine:
  - `checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine`

他PCではrepo全体をcloneし、Drive artifactを取得してください。

例:

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
tools/setup_integrated_runtime_env.sh
```

ただし、TensorRT engineはコピー先で再作成することを推奨します。

## Rebuild TensorRT engine

まず通常は以下のセットアップ入口を使ってください。新しいvenv作成、依存確認、必要ならTensorRT engine再作成、短いsmoke testまで行います。

```bash
cd backend/detectors/dinov3/runtime
./tools/setup_fast_runtime_env.sh
```

Pythonを明示する場合:

```bash
BASE_PYTHON=/path/to/python3.10 ./tools/setup_fast_runtime_env.sh
```

`tools/rebuild_default_trt_backbone.sh` は次を行います。

1. DINOv3 backboneをONNX export
2. ONNXからTensorRT BF16 engineを作成

実行:

```bash
cd backend/detectors/dinov3/runtime
PYTHON=../eva02_cascade_experimental/venv/bin/python ./tools/rebuild_default_trt_backbone.sh
```

作成されるもの:

- `output/onnx/dinov3_backbone_fp32_1280x720_dynamic.onnx`
- `output/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine`

## Practical recommendation

別PCで安定運用するなら、次の形が現実的です。

1. repo全体の必要コードを同じ相対配置で置く。
2. Python/CUDA/TensorRT環境を合わせる。
3. セットアップでDriveからcheckpointとengineを取得する。
4. コピー先GPUで `tools/rebuild_default_trt_backbone.sh` を実行してengineを再作成する。
5. `infer_video_dinov3_jsonl.py` で短い動画smoke testを行う。
