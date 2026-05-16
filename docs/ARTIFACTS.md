# Artifacts

Large runtime artifacts are intentionally kept outside Git. Upload them to
Google Drive or another artifact store, then let the setup script restore the
same layout under `checkpoints/`. TensorRT engines are local-device artifacts
and are rebuilt by setup instead of being required from Drive.

Expected local runtime layout after download/placement:

```text
checkpoints/
  detector/
    model_final.pth
  dinov3/
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  classifier/
    best.pt
  eva02/
    detector/
      model_final.pth
    classifier/
      best.pt
  codino/
    detector/
      resolved_config.py
      epoch_2.pth
    classifier/
      best.pt
  postprocess/
    k2_v5/
      best_exact.pt
      run_config.json
      train_k2_slot_set_spd_standalone_v5.py
    polygon_point_predictor/
      best.pt
      feature_stats.npz
      run_config.json
      train_mask_point_predictor.py
```

Check placement:

```bash
python tools/check_artifacts.py
```

Require locally built TensorRT engines too:

```bash
python tools/check_artifacts.py --require-trt
```

## Download Sources

Configure URLs in:

```text
configs/artifact_sources.env
```

Use this template:

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
```

Canonical shared Drive upload layout:

```text
artifacts_to_upload/runtime_artifacts/
  checkpoints/dinov3/detector/model_final.pth
  checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  checkpoints/dinov3/classifier/best.pt
  checkpoints/Eva02/detector/model_final.pth
  checkpoints/Eva02/classifier/best.pt
  checkpoints/codino/detector/resolved_config.py
  checkpoints/codino/detector/epoch_2.pth
  checkpoints/codino/classifier/best.pt
  checkpoints/postprocess/k2_v5/best_exact.pt
  checkpoints/postprocess/k2_v5/run_config.json
  checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
  checkpoints/postprocess/polygon_point_predictor/best.pt
  checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
  checkpoints/postprocess/polygon_point_predictor/run_config.json
  checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

Upload that folder to Google Drive, then set `RUNTIME_ARTIFACTS_URL`.

`tools/download_runtime_artifacts.py` also accepts the older flat local aliases
such as `checkpoints/detector/model_final.pth`,
`checkpoints/classifier/best.pt`, and lowercase `checkpoints/eva02/...`.

Manual artifact download/placement:

```bash
.venv_integrated/bin/python tools/download_runtime_artifacts.py \
  --runtime-artifacts-url "$RUNTIME_ARTIFACTS_URL"
```

Notes:

- TensorRT engines are device/CUDA/TensorRT dependent and are not required in the Drive artifact folder.
- The setup script rebuilds DINOv3 and Co-DINO TensorRT engines automatically when they are missing.
- `run_config.json` and model definition files are small and tracked.
- Runtime checkpoint bodies, `.engine`, `.pt`, `.pth`, and `.npz` files under
  the documented `checkpoints/` paths are ignored by Git and should come from
  Drive or a local artifact directory. Videos, JSONL, SQLite, overlays, work
  dirs, and ad-hoc experiment outputs are ignored.
