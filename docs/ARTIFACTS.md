# Artifacts

Large runtime artifacts are not tracked in Git. The first setup script can restore them from Google Drive and then build the TensorRT engine locally.

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
    trt/
      codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine
      codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine
      codino_decoder_b2_736x1280_msda_plugin_fp16.engine
      codino_mask_head_core_n1_736x1280_fp16.engine
  trt/
    dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
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

## Download sources

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
  checkpoints/codino/trt/*.engine
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

- TensorRT engines are device/CUDA/TensorRT dependent. Rebuild them on a new machine if loading fails.
- The setup script rebuilds the TensorRT engine automatically when it is missing.
- `run_config.json` and model definition files are small and tracked.
- Checkpoint bodies, `.engine`, `.pt`, `.pth`, `.npz`, videos, JSONL, SQLite, and overlays are ignored.
