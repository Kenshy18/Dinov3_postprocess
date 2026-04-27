# Runtime tools

初回セットアップ、ONNX export、TensorRT engine作成に関する入口をここに集約しています。

## Recommended setup

```bash
inference/dinov3_video_jsonl_runtime/tools/setup_fast_runtime_env.sh
```

このスクリプトは新しいvenv作成、依存確認、必要時のTensorRT engine作成、短いsmoke testまで実行します。

## TensorRT rebuild only

```bash
cd inference/dinov3_video_jsonl_runtime
PYTHON=../eva02_cascade_experimental/venv/bin/python ./tools/rebuild_default_trt_backbone.sh
```

生成物は `output/onnx/` と `output/trt/` に出力され、Gitでは追跡しません。
