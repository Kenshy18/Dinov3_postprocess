# Postprocess Boundary

Backend-facing role:

- Engine: `atosyori_postprocess` from `external/atosyori-pipeline-dev/src`.
- Backend adapter: `backend/postprocess/commands.py`.
- Model root: `checkpoints/postprocess`.
- Callers: `backend.pipeline.pipeline_commands` and
  `backend.pipeline.cli.run_postprocess_only`.

Contract:

- Input: shared detector JSONL and the original video.
- Output: SQLite, overlays, audit/log files, and postprocess summary.
- This layer should not care whether the JSONL came from DINOv3 or EVA02.
- Do not add detector or UI behavior here.
