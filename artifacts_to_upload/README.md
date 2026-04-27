# Upload artifacts

`runtime_artifacts/` contains all runtime artifacts that are not tracked by Git but are required for the end-to-end pipeline.

Upload the whole `runtime_artifacts/` folder to Google Drive and share the folder URL. Then set:

```bash
RUNTIME_ARTIFACTS_URL="https://drive.google.com/drive/folders/..."
```

Expected layout after download:

```text
checkpoints/detector/model_final.pth
checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
checkpoints/classifier/best.pt
checkpoints/postprocess/k2_v5/best_exact.pt
checkpoints/postprocess/k2_v5/run_config.json
checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
checkpoints/postprocess/polygon_point_predictor/best.pt
checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
checkpoints/postprocess/polygon_point_predictor/run_config.json
checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

TensorRT `.engine` is intentionally not required here because it should be rebuilt on the target machine during first setup.

On this workstation the large files are hardlinked to the existing local artifacts to avoid an extra 6GB+ copy. They are regular file entries, not symlinks, so uploading the folder will upload the file contents.
