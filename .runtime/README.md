# Local Runtime State

This directory stores machine-specific generated runtime state:

```text
gui_runtime.env
runtime_profile.json
runtime_benchmark.json
```

These files depend on the local Python environment, GPU, driver, TensorRT
version, and measured batch sizes. They are intentionally ignored by git.

Shared examples and policy JSON files belong in `configs/`.
