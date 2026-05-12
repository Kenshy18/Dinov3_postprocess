# Integrated runtime artifacts

Upload this folder as-is to Google Drive.

This is the preferred artifact bundle for the end-to-end repository. It contains every model file needed after `git clone` for:

- DINOv3 + Cascade Mask R-CNN detection
- ROI class classification
- Atosyori K2 inference
- polygon point-count / point predictor model

Required layout:

```text
checkpoints/dinov3/detector/model_final.pth
checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
checkpoints/dinov3/classifier/best.pt
checkpoints/Eva02/detector/model_final.pth
checkpoints/Eva02/classifier/best.pt
checkpoints/postprocess/k2_v5/best_exact.pt
checkpoints/postprocess/k2_v5/run_config.json
checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
checkpoints/postprocess/polygon_point_predictor/best.pt
checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
checkpoints/postprocess/polygon_point_predictor/run_config.json
checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

TensorRT engine is not included because it is tied to the target GPU / CUDA / TensorRT versions. The first setup script builds it locally.
