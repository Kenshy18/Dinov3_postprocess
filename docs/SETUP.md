# Setup

## 1. Clone this directory/repository

The bundle contains the required source code for DINOv3/EVA02 inference and Atosyori postprocess.

Key source directories:

```text
inference/dinov3_video_jsonl_runtime/
scripts/
configs/
dinov3/
eva02/eva02_det/
external/atosyori-pipeline-dev/
```

## 2. Configure artifact download

The integrated runtime artifact Google Drive folder URL is already set in the setup script. Only copy the example env file when you need to override it.

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
```

Edit:

```bash
RUNTIME_ARTIFACTS_URL="https://drive.google.com/drive/folders/..."
```

This single folder should contain the detection, classification, K2, and polygon predictor artifacts.

If artifacts are already available locally, set:

```bash
RUNTIME_ARTIFACTS_DIR="/path/to/runtime_artifacts"
```

Expected integrated artifact folder layout:

```text
checkpoints/detector/model_final.pth
checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
checkpoints/classifier/best.pt
checkpoints/eva02/detector/model_final.pth
checkpoints/eva02/classifier/best.pt
checkpoints/postprocess/k2_v5/best_exact.pt
checkpoints/postprocess/k2_v5/run_config.json
checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
checkpoints/postprocess/polygon_point_predictor/best.pt
checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
checkpoints/postprocess/polygon_point_predictor/run_config.json
checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

You can check the current placement with:

```bash
python tools/check_artifacts.py
```

## 3. Create runtime environment

```bash
tools/setup_integrated_runtime_env.sh
```

The setup script performs:

- artifact download and placement into `checkpoints/`
- venv creation
- dependency installation
- bundled Detectron2/EVA02 extension build when needed
- TensorRT engine build when missing or when `REBUILD_TRT=1`
- optional smoke/import checks

Useful options:

```bash
ENV_DIR=/path/to/venv tools/setup_integrated_runtime_env.sh
BASE_PYTHON=/path/to/python3.10 tools/setup_integrated_runtime_env.sh
REFERENCE_VENV=/path/to/known-good-venv tools/setup_integrated_runtime_env.sh
RUN_DINO_SMOKE=0 tools/setup_integrated_runtime_env.sh
BUILD_DETECTRON2=1 tools/setup_integrated_runtime_env.sh
TRT_PRECISION=fp16 tools/setup_integrated_runtime_env.sh
```

`BUILD_DETECTRON2=auto` is the default. It builds the bundled Detectron2/EVA02 extension only when `eva02/eva02_det/detectron2/_C*.so` is missing.

For Blackwell GPUs, use a PyTorch/CUDA build that supports the GPU architecture. On this machine the known-good runtime is the existing EVA02 inference venv, which can be passed through `REFERENCE_VENV`.

TensorRT engines are not treated as portable artifacts. Rebuild them on each target PC. The default build uses BF16; if BF16 engine creation fails, setup retries with FP16 at the same default engine path. Set `TRT_FALLBACK_FP16=0` to make BF16 failure hard-fail.

GPU notes:

- RTX 5090 is Blackwell-class and needs an `sm_120` capable PyTorch/CUDA/TensorRT stack.
- RTX 4090 is Ada-class and should work when the TensorRT engine is rebuilt locally.
- Do not copy a `.engine` file between different GPU/driver/TensorRT combinations.

## 4. Run

Recommended production-style entrypoint:

```bash
.venv_integrated/bin/python scripts/infer_video_postprocess.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --overlay \
  --force
```

Detailed entrypoint:

```bash
.venv_integrated/bin/python scripts/run_integrated_pipeline.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --classifier \
  --render-overlays \
  --force
```

If polygon predictor artifacts are not available:

```bash
.venv_integrated/bin/python scripts/run_integrated_pipeline.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --class-policy-json configs/class_policy_ellipse_only.json \
  --force
```
