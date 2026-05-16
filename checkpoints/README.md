# checkpoints

Runtime checkpoint and local TensorRT engine files are placed here.

Runtime checkpoint bodies are not tracked in Git. Restore portable checkpoints
from the shared Drive folder or another artifact store before inference.

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

TensorRT engines are GPU/CUDA/TensorRT dependent and should be created locally.
If an engine is missing or incompatible on a different machine, rebuild it with
the setup tools.

The shared Drive upload groups detector artifacts by backbone:

```text
checkpoints/dinov3/detector/model_final.pth
checkpoints/dinov3/classifier/best.pt
checkpoints/Eva02/detector/model_final.pth
checkpoints/Eva02/classifier/best.pt
checkpoints/codino/detector/resolved_config.py
checkpoints/codino/detector/epoch_2.pth
checkpoints/codino/classifier/best.pt
```
