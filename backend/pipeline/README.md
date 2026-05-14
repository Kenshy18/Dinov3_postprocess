# Backend Pipeline

Owns the integration layer:

```text
video(s)
  -> selected detector runtime
  -> shared detector JSONL
  -> Atosyori postprocess
  -> run summary and artifact index
```

Primary module:

```bash
.venv_integrated/bin/python -m backend.pipeline.run_integrated_pipeline --help
```

Compatibility command:

```bash
.venv_integrated/bin/python scripts/run_integrated_pipeline.py --help
```

Edit files here when changing orchestration, argument defaults, or output
summary collection.

Detector command construction belongs in `backend/detectors/<detector>`.
Postprocess command construction belongs in `backend/postprocess`.

User-facing CLI implementations live in:

```text
backend/pipeline/cli
```

The matching files under `scripts/` are thin compatibility wrappers.
