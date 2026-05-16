# EVA02 video JSONL runtime

This runtime wraps the repo-local optimized EVA02 + Cascade Mask R-CNN
inference path.

Default artifacts:

```text
checkpoints/eva02/detector/model_final.pth
checkpoints/eva02/classifier/best.pt
```

Default speed-oriented settings are `batch_size=20`, `topk=80`,
`compile_backbone=max-autotune`, `drop_block_indices=19,21,22`,
`pack_inputs=true`, `raw_detector_postprocess=true`,
`raw_to_orig_mask_postprocess=true`, and `mask_approx=simple`.

The main integration script calls:

```bash
python backend/detectors/eva02/runtime/infer_video_eva02_jsonl.py \
  --input input/sample.mp4 \
  --output output/runs/manual/eva02
```
