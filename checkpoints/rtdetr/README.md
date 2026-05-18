# RT-DETR Head/Face Checkpoints

Place the portable Head/Face detector checkpoint here via
`tools/artifacts/download_runtime_artifacts.py`.

Expected file:

```text
checkpoints/rtdetr/head_face_best_stg1.pth
```

The checkpoint is downloaded by the integrated setup script when
`RUNTIME_ARTIFACTS_URL` or `RUNTIME_ARTIFACTS_DIR` points to the shared runtime
artifact folder. Model config and RT-DETR source code are tracked under
`external/RT-DETR/RT-DETRv4`.
