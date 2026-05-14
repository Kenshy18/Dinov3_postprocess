# Classifier Runtimes

This package owns ROI-classifier implementation code. Detector runtimes may
call these modules, but classifier source should not live under
`backend/detectors/*/runtime`.

- `dinov3_roi/`: DINOv3 ROI classifier utilities and model loader.
- `eva02_roi/`: EVA02 ROI classifier utilities and model loader.

Checkpoint files remain under `checkpoints/` and are intentionally ignored by
git.

