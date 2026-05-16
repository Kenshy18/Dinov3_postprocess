# DINOv3 + Co-DINO Detector Boundary

Backend-facing role:

- backend adapter: `backend/detectors/codino/commands.py`
- runtime implementation: `backend/detectors/codino/runtime/infer_video_codino_jsonl.py`
- detector config/checkpoint: `checkpoints/codino/detector/`
- classifier checkpoint: `checkpoints/codino/classifier/best.pt`
- TensorRT engines: `checkpoints/codino/trt/`

Contract:

- Input: video file or directory.
- Output: detector JSONL plus `summary.json`.
- Class labels must match the shared postprocess JSONL schema.
- The Co-DINO runtime bridges to the canonical Co-DINO inference script and
  normalizes its detections/masks to this repository's detector JSONL contract.
- Higher layers should call this detector through `backend.pipeline`.
- Do not add UI or postprocess behavior here.

