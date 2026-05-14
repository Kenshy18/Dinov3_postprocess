# Run Audit

Use the run audit when a GUI or CLI run finishes but the result quality is
unclear:

```bash
.venv_integrated/bin/python tools/diagnose_run.py output/runs/<run_name>
```

JSON output is also available:

```bash
.venv_integrated/bin/python tools/diagnose_run.py output/runs/<run_name> \
  --format json \
  --output output/runs/<run_name>/logs/run_audit.json
```

The audit checks:

- detector JSONL frame/detection/mask counts
- final SQLite and tracked SQLite table/row counts
- raw tracking audit tables
- overlay file existence and size
- warnings such as missing masks or missing postprocess outputs

The GUI writes the same audit into:

```text
output/runs/<run_name>/logs/job_audit_summary.json
output/runs/<run_name>/最終成果物.json
```
