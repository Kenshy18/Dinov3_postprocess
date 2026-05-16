# DINOv3 + Co-DINO Detector Boundary

Backend-facing role:

- backend adapter: `backend/detectors/codino/commands.py`
- runtime implementation: `backend/detectors/codino/runtime/infer_video_codino_jsonl.py`
- fast Co-DINO execution backend: `backend/detectors/codino/runtime/codino_video_fast_runtime.py`
- Co-DINO source: `external/codino/`
- detector config/checkpoint: `checkpoints/codino/detector/`
- classifier checkpoint: `checkpoints/codino/classifier/best.pt`
- TensorRT engines: `checkpoints/codino/trt/`
- runtime profile / model registry: `backend/pipeline/model_registry.py`

Contract:

- Input: video file or directory.
- Output: detector JSONL plus `summary.json`.
- Class labels must match the shared postprocess JSONL schema.
- The Co-DINO runtime normalizes detections/masks to this repository's detector
  JSONL contract and resolves model artifacts from `checkpoints/codino`.
- Higher layers should call this detector through `backend.pipeline`.
- Do not add UI or postprocess behavior here.

Runtime notes:

- Use a Python environment with Co-DINO-compatible `mmcv` (`>=1.3.17, <=1.7.2`)
  and a PyTorch/CUDA build that supports the installed GPU.
- TensorRT Python bindings should come from the active runtime environment.
  `TENSORRT_SITE_PACKAGES` is only an explicit override for non-standard setups.
- The runtime checks CUDA architecture compatibility before model load so an
  unsupported PyTorch build fails before TensorRT can segfault.
