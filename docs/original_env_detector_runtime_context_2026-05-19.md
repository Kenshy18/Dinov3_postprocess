# Original Detector Runtime Integration Context

Date: 2026-05-19

## Goal

Run the existing `Dinov3_postprocess` pipeline with Co-DINO disabled while keeping
the rest of the backend behavior compatible with the GUI:

- DINOv3 inference
- EVA02 inference
- face/head detection
- raw JSONL to SQLite conversion
- postprocess
- overlay generation
- progress/log output
- GUI command compatibility

The practical requirement was to reuse the known-good detector virtual
environments from the original backend repositories:

- DINOv3: `/home/accel/SOD_Dino_backend_original/.venv_dinov3/bin/python`
- EVA02: `/home/accel/SOD_Eva_backend_original/.venv/bin/python`
- EVA02 detectron2 package: `/home/accel/SOD_Eva_backend_original/eva02_det`

## Main Problem

The integrated `.venv_integrated` environment used newer/nightly PyTorch and
torchvision CUDA builds. On the target RTX 5090 / `sm_120` environment,
torchvision CUDA ops such as `roi_align` and `nms` were not reliable:

- Some paths failed with `no kernel image is available for execution on the device`.
- Fallback paths were much slower and could fall into TorchInductor/Triton
  compilation.
- TorchInductor/Triton compilation also required a C compiler via `CC`.

The original DINOv3 and EVA02 environments had working CUDA op behavior and
met the expected detector throughput range:

- DINOv3: over 20 FPS, observed around 30 FPS on the validation video.
- EVA02: over 12 FPS, observed around 14 FPS on the validation video.

## Implemented Backend Changes

### Detector-specific Python Interpreters

`backend/pipeline/run_integrated_pipeline.py` now accepts:

- `--dinov3-python`
- `--eva02-python`
- `--eva02-compile-backbone`

The DINOv3 and EVA02 command builders use these interpreter overrides while the
rest of the pipeline continues to use the shared runtime Python.

Files:

- `backend/detectors/dinov3/commands.py`
- `backend/detectors/eva02/commands.py`
- `backend/pipeline/run_integrated_pipeline.py`

### EVA02 Detectron2 Path Override

`backend/detectors/eva02/runtime/infer_video_jsonl_singleclass.py` now prefers
`EVA02_DET_PATH` when it is provided. This lets the pipeline use the original
EVA02 detectron2 tree instead of the bundled local copy.

The DINOv3 runtime already depends on EVA02/Detectron2 config/model code, so
exporting `EVA02_DET_PATH` is also important for DINOv3.

### DINOv3 TensorRT Engine Compatibility

`backend/detectors/dinov3/runtime/trt_backbone.py` was adjusted to parse ONNX
through `parse_from_file`, which avoids parser issues seen in the local runtime.

### GUI Command Compatibility

`apps/qt_ui/app.py` was adjusted so the GUI:

- hides Co-DINO from the backend selector
- passes `--dinov3-python` when `DINOV3_DETECTOR_PYTHON` is configured
- passes `--eva02-python` when `EVA02_DETECTOR_PYTHON` is configured
- passes `--eva02-compile-backbone`
- forwards detector-related environment variables to child processes

`apps/qt_ui/run_app.sh` also adds the local XCB library directory when present.

### Runtime Setup

`tools/setup/setup_integrated_runtime_env.sh` now writes runtime environment
entries needed by the GUI/runtime, including PATH and LD_LIBRARY_PATH setup for
the micromamba runtime.

For local operation, `.runtime/gui_runtime.env` should include values like:

```bash
DINOV3_DETECTOR_PYTHON=/home/accel/SOD_Dino_backend_original/.venv_dinov3/bin/python
EVA02_DETECTOR_PYTHON=/home/accel/SOD_Eva_backend_original/.venv/bin/python
EVA02_DET_PATH=/home/accel/SOD_Eva_backend_original/eva02_det
EVA02_COMPILE_BACKBONE=none
DINOV3_TRT_BACKBONE_ENGINE=/home/accel/SOD_Dino_backend_original/output/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
CC=/home/accel/0519/Dinov3_postprocess/.runtime/mamba_py311/bin/x86_64-conda-linux-gnu-gcc
CXX=/home/accel/0519/Dinov3_postprocess/.runtime/mamba_py311/bin/x86_64-conda-linux-gnu-g++
```

Note: `.runtime/gui_runtime.env` is intentionally runtime-local and is not
committed.

## Windows Frontend Issue

The Windows frontend repository at
`C:\Inference_front\Dinov3_postprocess_Windows_UI` is separate from this repo.

The Windows GUI initially failed because it sourced `.runtime/gui_runtime.env`
without exporting the variables. That meant variables such as `EVA02_DET_PATH`
were visible to the shell but not inherited by child Python processes. DINOv3
then imported the bundled local EVA02 detectron2 tree and entered a fallback
ROIAlign path that tried to invoke TorchInductor/Triton, eventually failing
with:

```text
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
```

The Windows frontend fix was to source the runtime env with:

```bash
set -a
source .runtime/gui_runtime.env
set +a
```

and to export `CC` / `CXX` if the local micromamba compilers exist.

That Windows frontend change lives in the separate Windows UI repository and is
not part of this backend repository unless mirrored separately.

## Validation Performed

### Original Environment Detector Runs

Using original detector virtual environments:

- DINOv3 on `input/01_h264_aac_progressive.mp4`
  - 600 frames
  - observed around 30 FPS end-to-end detector throughput
- EVA02 on the same input
  - 600 frames
  - observed around 14 FPS detector throughput

### Pipeline/GUI Compatibility Smoke

The integrated pipeline path was tested with DINOv3 and the original detector
environment through the GUI-style runner:

- `apps/qt_ui/run_ui_job.py`
- `scripts/run_integrated_pipeline.py`
- raw SQLite conversion
- postprocess
- output manifest generation

The smoke test completed successfully and confirmed that the original
`/home/accel/SOD_Eva_backend_original/eva02_det` path was used.

## Remaining Caveats

- Co-DINO code paths still exist in the backend, but the GUI selector no longer
  exposes Co-DINO.
- The local bundled detectron2 fallback patches are diagnostic/workaround code
  for the integrated environment. The preferred fast path is still to use the
  original DINOv3/EVA02 environments.
- For reliable Windows GUI execution, the Windows frontend repository must also
  contain the exported-env fix described above.
