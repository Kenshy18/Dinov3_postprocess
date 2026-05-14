# Postprocess Engine Policy

`external/atosyori-pipeline-dev` is part of this integrated product's
postprocess engine and is managed as this repository's postprocess engine. It
is not treated as an untouchable third-party blob.

## Ownership

- Backend orchestration and command construction belong in `backend/postprocess`
  and `backend/pipeline`.
- Postprocess algorithm behavior, raw tracking, SQLite schema, overlay
  rendering, and postprocess audit behavior belong in
  `external/atosyori-pipeline-dev`.
- Cross-stage detector JSONL contracts belong in `backend/schemas`.

## Change Rules

Change the engine directly when the behavior itself must change, for example:

- raw JSONL normalization
- tracking or short-track pruning
- raw audit table schema
- SQLite output schema
- ellipse/polygon postprocess behavior
- overlay rendering
- postprocess summary/audit fields

Keep adapter-only changes in `backend/postprocess` when the engine behavior does
not change.

## Source Refreshes

If this engine is refreshed from another Atosyori source tree:

1. Run `tools/sync_atosyori_source.sh` or a reviewed equivalent.
2. Preserve local integration requirements documented in
   `external/atosyori-pipeline-dev/docs/local_patches.md`.
3. Run `tools/verify_runtime.py --no-atosyori-smoke`.
4. For behavior changes, run at least one short detector/postprocess smoke and
   inspect `tools/diagnose_run.py <run_dir>`.
5. Commit the refresh separately from local behavior edits when possible.

## Audit Requirements

Any engine change that affects raw JSONL or SQLite outputs must preserve or
update:

- `raw_tracked_masks`
- `raw_tracks`
- `raw_det_score_min`
- `raw_tracked_rows`
- `raw_removed_rows`
- `summary.json`

When these fields intentionally change, update tests and docs in the same
commit.
