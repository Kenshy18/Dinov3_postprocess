# Generated model artifacts

ONNXやTensorRT engineなど、初期セットアップで生成される重いファイルを置く場所です。

- `onnx/`: DINOv3 backbone ONNX export
- `trt/`: TensorRT engine

これらの生成物はGitに含めません。別PCでは `tools/rebuild_default_trt_backbone.sh` で再作成してください。
