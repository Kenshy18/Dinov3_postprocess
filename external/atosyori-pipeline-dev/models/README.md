# models

This directory is where local model files are placed.

Model checkpoints are not tracked in Git. Download or copy them from:

https://drive.google.com/drive/folders/10-Zc2ShIkJn7T1JIcvUgiEgdSoIANJcT?usp=sharing

Expected layout:

```text
models/
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

Refresh from the original standalone directory:

```bash
python -m atosyori_postprocess copy-models --source-root .. --replace
```
