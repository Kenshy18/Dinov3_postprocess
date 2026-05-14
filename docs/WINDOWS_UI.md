# Windows UI Placement

The Qt UI can be launched from the tracked application root `apps/qt_ui`. The
old `UI/` directory remains a compatibility entrypoint.

## Recommended Layout

Keep the repository root as the single source of truth:

```text
Dinov3_postprocess/
  apps/qt_ui/
  backend/
  checkpoints/
  .runtime/
```

Do not copy only the `UI/` folder by itself. The UI imports backend modules,
reads `.runtime/`, and launches `scripts/run_integrated_pipeline.py`.

## PowerShell Launch

From PowerShell:

```powershell
cd C:\path\to\Dinov3_postprocess
.\apps\qt_ui\run_app.ps1
```

The compatibility path also works:

```powershell
.\UI\run_app.ps1
```

The launcher reads `.runtime/gui_runtime.env` when it exists and falls back to
legacy `configs/gui_runtime.env` during migration. It then sets:

```text
DINOV3_RUNTIME_PROFILE
DINOV3_BATCH_BENCHMARK
DINOV3_TRT_BACKBONE_ENGINE
```

If the backend and GPU runtime are in WSL/Linux, keep running the setup and
pipeline there. In that case, launch the UI from the same environment or set
`GUI_RUNTIME_PYTHON` to the Python executable that can import PyQt5 and the
backend dependencies.

## What to Check

After a GUI run, inspect:

```text
output/runs/<run_name>/logs/ui_job.log
output/runs/<run_name>/logs/audit.jsonl
output/runs/<run_name>/logs/job_audit_summary.json
output/runs/<run_name>/最終成果物.json
```

`job_audit_summary.json` records the detector JSONL contract, SQLite table
counts, overlay file sizes, and warnings such as missing masks or missing raw
audit tables.

