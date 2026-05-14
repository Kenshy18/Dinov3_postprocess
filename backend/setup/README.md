# Setup Boundary

Stable compatibility commands live in `tools/`:

- `tools/setup_gui_runtime.sh`
- `tools/setup_integrated_runtime_env.sh`
- `tools/benchmark_runtime_batches.py`
- `tools/configure_runtime_profile.py`
- `tools/check_artifacts.py`
- `tools/download_runtime_artifacts.py`
- `tools/verify_runtime.py`

Implementation lives under:

```text
tools/setup
tools/artifacts
tools/verify
tools/debug
```

Generated local state:

- `.runtime/gui_runtime.env`
- `.runtime/runtime_profile.json`
- `.runtime/runtime_benchmark.json`

These files are intentionally gitignored because they depend on the local
Python environment, GPU, driver, TensorRT version, and measured batch speed.
