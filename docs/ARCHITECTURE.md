# Architecture

This repository is organized around a few explicit boundaries. Keep new work
inside the boundary that owns the behavior.

For day-to-day maintenance and verification policy, see
`docs/MAINTENANCE.md`.

## Boundaries

```text
apps/qt_ui
  Qt frontend. Builds job commands, shows status, and organizes user-facing
  outputs. It should not import detector internals.

backend/pipeline
  Orchestrates one job: collect videos, run the selected detector, normalize
  detector outputs, run postprocess, and write summaries.

backend/pipeline/cli
  User-facing CLI implementations. The short files under scripts/ are stable
  compatibility wrappers around these modules.

backend/detectors/dinov3
  DINOv3 detector boundary. Owns backend-facing command construction plus the
  DINOv3 runtime implementation under runtime/.

backend/detectors/eva02
  EVA02 detector boundary. Owns backend-facing command construction plus the
  EVA02 runtime implementation under runtime/.

backend/postprocess
  Backend-facing Atosyori adapter. Owns postprocess command construction and
  environment setup. The engine source remains under external/.

inference/dinov3_video_jsonl_runtime
  Compatibility wrappers for old DINOv3 runtime paths.

inference/eva02_video_jsonl_runtime
  Compatibility wrappers for old EVA02 runtime paths.

training/dinov3
  DINOv3 classifier/detector training implementation. Runtime code may import
  shared registration symbols through the compatibility wrapper, but training
  business logic belongs here.

external/atosyori-pipeline-dev
  Postprocess engine source. The backend calls it through its CLI/module entry.

tools
  Stable compatibility commands for operational tooling.

tools/setup
  Setup scripts, local runtime profiling, and batch benchmark implementation.

tools/artifacts
  Artifact layout definitions, artifact download, and artifact checks.

tools/verify
  Repository/runtime verification implementation.

tools/debug
  Debug-only investigation utilities.

tools/maintenance
  Local cleanup and long-term repository maintenance utilities.

configs
  Tracked shared configuration examples plus ignored local runtime state.

checkpoints
  Ignored model artifacts and TensorRT engines.
```

## Compatibility Entrypoints

The old paths remain as wrappers so existing shortcuts and tests keep working:

```text
scripts/run_integrated_pipeline.py -> backend.pipeline.run_integrated_pipeline
scripts/pipeline_*.py              -> backend.pipeline.pipeline_*
scripts/run_full_flow.py           -> backend.pipeline.cli.run_full_flow
scripts/run_postprocess_only.py    -> backend.pipeline.cli.run_postprocess_only
scripts/infer_video_postprocess.py -> backend.pipeline.cli.infer_video_postprocess
scripts/flow_cli_common.py         -> backend.pipeline.cli.flow_cli_common
scripts/render_raw_jsonl_overlays.py -> backend.pipeline.cli.render_raw_jsonl_overlays
scripts/train_dinov3_cascade_unified.py -> training.dinov3.train_dinov3_cascade_unified
inference/dinov3_video_jsonl_runtime/* -> backend.detectors.dinov3.runtime.*
inference/eva02_video_jsonl_runtime/* -> backend.detectors.eva02.runtime.*
UI/run_app.sh                      -> apps/qt_ui/run_app.sh
UI/app.py                          -> apps.qt_ui.app
UI/run_ui_job.py                   -> apps.qt_ui.run_ui_job
tools/setup_gui_runtime.sh         -> tools/setup/setup_gui_runtime.sh
tools/setup_integrated_runtime_env.sh -> tools/setup/setup_integrated_runtime_env.sh
tools/check_artifacts.py           -> tools/artifacts/check_artifacts.py
tools/download_runtime_artifacts.py -> tools/artifacts/download_runtime_artifacts.py
tools/runtime_artifacts.py         -> tools/artifacts/runtime_artifacts.py
tools/configure_runtime_profile.py -> tools/setup/configure_runtime_profile.py
tools/benchmark_runtime_batches.py -> tools/setup/benchmark_runtime_batches.py
tools/sync_atosyori_source.sh      -> tools/setup/sync_atosyori_source.sh
tools/verify_runtime.py            -> tools/verify/verify_runtime.py
tools/debug_confidence_overlay.py  -> tools/debug/debug_confidence_overlay.py
```

Prefer the new implementation paths when editing code. Prefer the compatibility
paths when documenting user commands, because they are stable and short.

## Data Contract

Both detector runtimes must emit the same detector JSONL shape and a
`summary.json`. The postprocess layer should only depend on that shared JSONL
contract, not on DINOv3/EVA02-specific internals.

## Local Runtime State

Setup writes local-machine state under `configs/`:

```text
configs/gui_runtime.env
configs/runtime_profile.json
configs/runtime_benchmark.json
```

These files are gitignored because they depend on the selected Python
environment, GPU, driver, TensorRT version, and measured batch speeds.

## Change Rules

- UI changes belong in `apps/qt_ui`.
- Pipeline orchestration changes belong in `backend/pipeline`.
- Backend-facing detector command/contract changes belong in
  `backend/detectors/<detector>`.
- Detector-specific behavior belongs in the corresponding
  `backend/detectors/<detector>/runtime`.
- Backend-facing postprocess command/contract changes belong in
  `backend/postprocess`.
- Training behavior belongs in `training/...`.
- Postprocess behavior belongs in `external/atosyori-pipeline-dev` or in a
  thin backend adapter, not in the UI.
- Setup and diagnostics belong in `tools`.
- Keep compatibility wrappers thin. They should import and call the new owner,
  not contain business logic.
