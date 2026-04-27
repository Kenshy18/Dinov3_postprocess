# checkpoints

Runtime checkpoint and engine files are placed here.

These files are intentionally not tracked in Git. Keep them in Google Drive or another artifact store and place them into this layout before running.

```text
checkpoints/
  detector/
    model_final.pth
  dinov3/
    dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  classifier/
    best.pt
  trt/
    dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
  postprocess/
    k2_v5/best_exact.pt
    polygon_point_predictor/best.pt
    polygon_point_predictor/feature_stats.npz
```

TensorRT engine is GPU/CUDA/TensorRT dependent. If it is missing or incompatible, rebuild it with the tools under `inference/dinov3_video_jsonl_runtime/tools/`.
