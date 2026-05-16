# Setup

## 1. Clone this directory/repository

The bundle contains the required source code for DINOv3/EVA02/Co-DINO inference
and Atosyori postprocess.

Key source directories:

```text
backend/detectors/dinov3/runtime/
backend/detectors/codino/runtime/
scripts/
configs/
dinov3/
eva02/eva02_det/
external/codino/
external/atosyori-pipeline-dev/
```

## 2. Configure Drive artifacts

Runtime checkpoints, classifiers, and postprocess models are stored outside Git.
TensorRT engines are device-specific and are created during setup. Upload the
portable runtime artifact folder to Google Drive, then set the shared folder URL.

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
```

Edit:

```bash
RUNTIME_ARTIFACTS_URL="https://drive.google.com/drive/folders/..."
```

Check the current placement after setup/download with:

```bash
python tools/check_artifacts.py
```

This single folder should contain the detection, classification, K2, and polygon predictor artifacts.

If artifacts are already available locally, set:

```bash
RUNTIME_ARTIFACTS_DIR="/path/to/runtime_artifacts"
```

Expected shared Drive artifact folder layout:

```text
checkpoints/dinov3/detector/model_final.pth
checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
checkpoints/dinov3/classifier/best.pt
checkpoints/Eva02/detector/model_final.pth
checkpoints/Eva02/classifier/best.pt
checkpoints/codino/detector/epoch_2.pth
checkpoints/codino/classifier/best.pt
checkpoints/postprocess/k2_v5/best_exact.pt
checkpoints/postprocess/k2_v5/run_config.json
checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
checkpoints/postprocess/polygon_point_predictor/best.pt
checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
checkpoints/postprocess/polygon_point_predictor/run_config.json
checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

The downloader places those grouped Drive paths into the local runtime layout
under `checkpoints/detector`, `checkpoints/classifier`, `checkpoints/eva02`,
and `checkpoints/codino`.

## 3. Create runtime environment

```bash
tools/setup_runtime.sh
```

The setup script performs:

- local `input/` and `output/` directory creation
- artifact verification under `checkpoints/`
- optional artifact download when `DOWNLOAD_ARTIFACTS=1`
- venv creation
- dependency installation
- UI dependency installation
- bundled Detectron2/EVA02 extension build when needed
- DINOv3 TensorRT engine build when missing or when `REBUILD_TRT=1`
- Co-DINO TensorRT engine build when missing or when `REBUILD_CODINO_TRT=1`
- local GPU/VRAM profiling into `.runtime/runtime_profile.json`
- optional smoke/import checks

Compatibility setup wrappers remain available:

```bash
tools/setup_gui_runtime.sh
tools/setup_integrated_runtime_env.sh
```

The generated runtime profile is used by the integrated pipeline defaults. Setup also creates a temporary dummy video, measures DINOv3, EVA02, and Co-DINO candidate batch sizes sequentially, writes the result to `.runtime/runtime_benchmark.json`, and deletes the temporary video/output tree afterward. The selected GUI Python/venv, runtime profile path, benchmark result path, and TensorRT engine paths are written to `.runtime/gui_runtime.env`, which `apps/qt_ui/run_app.sh` sources before launching the application. `UI/run_app.sh` remains a compatibility wrapper. `.runtime/runtime_profile.json`, `.runtime/runtime_benchmark.json`, and `.runtime/gui_runtime.env` are local-machine state and are gitignored; keep the tracked guideline in `configs/runtime_profile.example.json` up to date instead. Legacy `configs/runtime_profile.json` is still read as a fallback during migration. If benchmarking cannot select a value, setup falls back to conservative defaults that keep EVA02 batch-size low on GPUs below 16 GiB VRAM to avoid CUDA unified/shared-memory fallback.

Useful options:

```bash
ENV_DIR=/path/to/venv tools/setup_runtime.sh
BASE_PYTHON=/path/to/python3.10 tools/setup_runtime.sh
REFERENCE_VENV=/path/to/known-good-venv tools/setup_runtime.sh
RUN_DINO_SMOKE=0 tools/setup_runtime.sh
BUILD_DETECTRON2=1 tools/setup_runtime.sh
TRT_PRECISION=fp16 tools/setup_runtime.sh
TENSORRT_PIP_SPEC=tensorrt==10.13.0.35 tools/setup_runtime.sh
REBUILD_CODINO_TRT=1 tools/setup_runtime.sh
CODINO_TRT_BATCH_SIZE=1 tools/setup_runtime.sh
```

`BUILD_DETECTRON2=auto` is the default. It builds the bundled Detectron2/EVA02 extension only when `eva02/eva02_det/detectron2/_C*.so` is missing.

For Blackwell GPUs, use a PyTorch/CUDA build that supports the GPU architecture. On this machine the known-good runtime is the existing EVA02 inference venv, which can be passed through `REFERENCE_VENV`.

TensorRT engines are not portable across GPU/driver/TensorRT combinations. The
DINOv3 backbone engine is dynamic up to batch 8. Co-DINO creates fixed-batch
engines for the DINOv3 backbone, query encoder, decoder, and mask head core.
Setup benchmarks DINOv3, EVA02, and Co-DINO after dependencies and engines are
ready; EVA02 is benchmarked even though it does not create a TensorRT engine.
Co-DINO query encoder/decoder ONNX export maps mmcv deformable
attention to NVIDIA `MultiscaleDeformableAttnPlugin_TRT`, so TensorRT Python
bindings and plugin libraries must import correctly in the setup venv.
On the current Blackwell workstation with driver 573/CUDA 12.8, the default
`TENSORRT_PIP_SPEC` is pinned to `tensorrt==10.13.0.35`; newer CUDA 13-oriented
TensorRT wheels can import but fail during builder initialization.

GPU notes:

- RTX 5090 is Blackwell-class and needs an `sm_120` capable PyTorch/CUDA/TensorRT stack.
- RTX 4090 is Ada-class and should work when the TensorRT engine is rebuilt locally.
- Do not copy a `.engine` file between different GPU/driver/TensorRT combinations.

## 4. Run

Recommended production-style entrypoint:

```bash
.venv_integrated/bin/python scripts/infer.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --force
```

Inference-only:

```bash
.venv_integrated/bin/python scripts/infer.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --mode inference \
  --force
```

Co-DINO detector:

```bash
.venv_integrated/bin/python scripts/infer.py \
  --detector codino \
  --input input/sample.mp4 \
  --output-root output/runs \
  --force
```

If polygon predictor artifacts are not available:

```bash
.venv_integrated/bin/python scripts/infer.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --class-policy-json configs/class_policy_ellipse_only.json \
  --force
```
