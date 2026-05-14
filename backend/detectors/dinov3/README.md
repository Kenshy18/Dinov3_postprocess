# DINOv3 Detector Boundary

Backend-facing role:

- backend adapter: `backend/detectors/dinov3/commands.py`
- runtime implementation: `backend/detectors/dinov3/runtime/infer_video_dinov3_jsonl.py`
- detector checkpoint: `checkpoints/detector/model_final.pth`
- backbone weights: `checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
- classifier checkpoint: `checkpoints/classifier/best.pt`
- classifier implementation: `backend/classifiers/dinov3_roi/`
- TensorRT engine: `checkpoints/trt/*.engine`

Contract:

- Input: video file or directory.
- Output: detector JSONL plus `summary.json`.
- Class labels must match the shared postprocess JSONL schema.
- ROI classifier implementation changes belong in `backend/classifiers`, not
  in this detector runtime tree.
- Higher layers should call this detector through `backend.pipeline`.
- `backend.pipeline` calls this adapter, and this adapter invokes the runtime
  script in this detector boundary.
- Do not add UI or postprocess behavior here.
