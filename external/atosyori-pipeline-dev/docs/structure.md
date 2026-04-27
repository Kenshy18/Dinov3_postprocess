# Structure

This directory is organized for development and operations.

```text
src/atosyori_postprocess/
  cli.py                 CLI entry point
  pipeline.py            full-pipeline argument normalization
  settings.py            paths and model resolution
  models.py              local checkpoint copying
  doctor.py              dependency/model checks
  devdata.py             tiny synthetic SQLite generation
  smoke.py               runnable local smoke checks
  stages/
    preprocess.py        raw JSONL/video -> tracked SQLite
    ellipse_inference.py K1/K2 ellipse inference
    ellipse_keyframes.py ellipse keyframe optimization
    polygon_keyframes.py polygon keyframe optimization
    evaluation.py        exact evaluation, gap fill, SQLite export helpers
    render.py            overlay rendering
  engine/
    standalone_runtime_fst.py
    standalone_runtime_k2v5.py
    ellipse_inference.py
    optimize_keyframes_*.py
    polygon_v22.py
    polygon_runtime.py
    render_overlay.py
  legacy/
    run_standalone.py    validated original engine
```

`engine/` is generated from `legacy/run_standalone.py` by
`tools/extract_legacy_modules.py`. The stage modules call these extracted
modules directly, while the full pipeline still uses the validated legacy
orchestrator until orchestration is refactored safely.

The raw extracted `polygon_v22.py` is available for refactoring, while
`polygon_runtime.py` uses the production-patched legacy polygon runner. Real
data currently goes through `polygon_runtime.py` so the validated contour
safety patches remain active.

The fastest operational check is:

```bash
python -m atosyori_postprocess smoke --work-dir output/smoke --force
```
