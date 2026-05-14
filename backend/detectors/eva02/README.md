# EVA02 Detector Boundary

Backend-facing role:

- backend adapter: `backend/detectors/eva02/commands.py`
- runtime implementation: `backend/detectors/eva02/runtime/infer_video_eva02_jsonl.py`
- detector checkpoint: `checkpoints/eva02/detector/model_final.pth`
- classifier checkpoint: `checkpoints/eva02/classifier/best.pt`
- classifier implementation: `backend/classifiers/eva02_roi/`

Contract:

- Input: video file or directory.
- Output: detector JSONL plus `summary.json`.
- EVA02 currently requires the ROI classifier in the integrated runtime.
- ROI classifier implementation changes belong in `backend/classifiers`, not
  in this detector runtime tree.
- Higher layers should call this detector through `backend.pipeline`.
- `backend.pipeline` calls this adapter, and this adapter invokes the runtime
  script in this detector boundary.
- Do not add UI or postprocess behavior here.
