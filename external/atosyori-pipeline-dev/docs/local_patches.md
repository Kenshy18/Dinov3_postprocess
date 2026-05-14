# Local Integration Patches

This vendored Atosyori tree is called through `backend/postprocess`. Most
integration changes should live in that adapter. The engine is changed only
when the postprocess behavior or audit output itself needs to change.

Current local behavior relied on by this repository:

- The CLI uses `src/atosyori_postprocess/legacy/run_standalone.py` as the
  validated engine entrypoint.
- Raw detector JSONL can be provided directly with `--input-jsonl` plus the
  source video.
- Raw preprocessing writes `raw_tracked_masks` and `raw_tracks` audit tables so
  removed short tracks, relabeling, bbox, label, and score provenance can be
  inspected after failures.
- The raw preprocessing summary reports `raw_tracked_rows`, `raw_tracks`,
  `raw_removed_rows`, `raw_det_score_min`, cut-detection settings, and timing.
- The engine accepts the shared detector JSONL contract documented in
  `docs/ARCHITECTURE.md`: DINOv3-style `frame_index`/`detections` and
  EVA02-style `frame_idx`/`instances` both normalize into the same postprocess
  path.

When this vendored source is refreshed from upstream, verify that these items
still exist before trusting GUI or batch-run audit logs.
