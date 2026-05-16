# checkpoints

Runtime checkpoint and engine files are placed here.

These files are intentionally not tracked in Git. Keep them in Google Drive or another artifact store. The download script places them into this local layout before running.

```text
checkpoints/
  detector/
    model_final.pth
  dinov3/
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  classifier/
    best.pt
  eva02/
    detector/model_final.pth
    classifier/best.pt
  codino/
    detector/resolved_config.py
    detector/epoch_2.pth
    classifier/best.pt
    trt/*.engine
  trt/
    dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
  postprocess/
    k2_v5/best_exact.pt
    polygon_point_predictor/best.pt
    polygon_point_predictor/feature_stats.npz
```

TensorRT engine is GPU/CUDA/TensorRT dependent. If it is missing or incompatible, rebuild it with the tools under `inference/dinov3_video_jsonl_runtime/tools/`.

The shared Drive upload currently groups detector artifacts by backbone:

```text
checkpoints/dinov3/detector/model_final.pth
checkpoints/dinov3/classifier/best.pt
checkpoints/Eva02/detector/model_final.pth
checkpoints/Eva02/classifier/best.pt
checkpoints/codino/detector/resolved_config.py
checkpoints/codino/detector/epoch_2.pth
checkpoints/codino/classifier/best.pt
checkpoints/codino/trt/*.engine
```
