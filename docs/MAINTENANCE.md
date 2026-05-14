# Maintenance

This project should stay easy to operate on one machine and easy to move to a
new machine. Prefer clear boundaries over clever shortcuts.

## Stable Entrypoints

User-facing commands should stay short and stable:

```text
apps/qt_ui/run_app.sh
UI/run_app.sh
scripts/infer_video_postprocess.py
scripts/run_full_flow.py
scripts/run_postprocess_only.py
scripts/run_integrated_pipeline.py
tools/setup_gui_runtime.sh
tools/verify_runtime.py
```

Implementation should live behind those entrypoints:

```text
apps/qt_ui
backend/pipeline
backend/detectors/<detector>
backend/postprocess
tools/setup
tools/artifacts
tools/verify
training
```

## Change Checklist

Before finishing a structural change:

1. Update `docs/ARCHITECTURE.md` if ownership changed.
2. Add or update a test under `tests/` if a boundary should not regress.
3. Keep compatibility wrappers thin.
4. Run:

```bash
.venv_integrated/bin/python tools/verify_runtime.py --no-atosyori-smoke
```

Run detector smoke tests when command construction, runtime defaults, artifact
paths, TensorRT setup, or detector output schemas change.

## Cleanup

Use a dry-run first:

```bash
.venv_integrated/bin/python tools/clean_generated.py --all-local
```

Apply only after checking the printed targets:

```bash
.venv_integrated/bin/python tools/clean_generated.py --all-local --apply
```

`output/.gitkeep`, `output/README.md`, `.runtime/.gitkeep`, and
`.runtime/README.md` are preserved.

## Local State

These files are local-machine state and should not be committed:

```text
.runtime/gui_runtime.env
.runtime/runtime_profile.json
.runtime/runtime_benchmark.json
.venv_integrated/
output/
```

Checkpoints and engines belong under `checkpoints/`; they are also local
artifacts and should be restored by setup or artifact download.

## Vendor Policy

`external/atosyori-pipeline-dev` is a vendored engine source tree. Prefer
wrapping it from `backend/postprocess` for command construction and environment
setup. Only change the vendored engine when the behavior itself must change,
such as raw tracking audit tables, SQLite schema, or overlay rendering.

When vendor behavior changes, add a note to the summary or docs explaining why
the change cannot live in the adapter.

Current local vendor patch notes live in
`external/atosyori-pipeline-dev/docs/local_patches.md`. Keep that document in
sync with any raw JSONL, tracking audit, SQLite schema, or overlay behavior
change.

## Compatibility Policy

Compatibility wrappers under `scripts/`, `tools/`, `UI/`, and `inference/`
exist to keep old shortcuts and previous automation working. They should:

- import or exec the new owner
- contain no business logic
- be covered by a light `--help` or compile check

Delete compatibility wrappers only after confirming that no user shortcut,
automation, or documentation still depends on them.

## Cleanup Inventory

Before proposing deletion, generate an objective inventory:

```bash
.venv_integrated/bin/python tools/inventory_cleanup_candidates.py
```

Review results can be saved under ignored local state:

```bash
.venv_integrated/bin/python tools/inventory_cleanup_candidates.py \
  --output .runtime/cleanup_inventory.md
```

This reports compatibility wrappers, local generated state, and large source
files. It does not delete anything and does not approve deletion by itself.
