# postprocess checkpoints

Atosyori後処理用checkpointを置くディレクトリです。

期待配置:

```text
checkpoints/postprocess/
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

実体ファイルはGit管理しません。別の場所を使う場合は `--postprocess-model-root` を指定してください。
