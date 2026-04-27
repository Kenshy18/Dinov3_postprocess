# Actual Input Validation

Validated with:

```text
input/アクセル様２月解析用白カン01.26.jsonl
input/アクセル様２月解析用白カン01.26.mp4
```

Input characteristics:

```text
JSONL frames: 18,000
Video: 1920x1080, 30000/1001 fps, 600.6 sec, 18,000 frames
```

## Commands

Preprocess:

```bash
python -m atosyori_postprocess stage preprocess -- \
  --input-jsonl input/アクセル様２月解析用白カン01.26.jsonl \
  --input-video input/アクセル様２月解析用白カン01.26.mp4 \
  --output-dir output/actual_split/preprocess
```

Polygon keyframes:

```bash
python -m atosyori_postprocess stage polygon-keyframes -- \
  --input-sqlite output/actual_split/preprocess/preprocess/アクセル様２月解析用白カン01.26.tracked.sqlite \
  --output-dir output/actual_split/polygon_keyframes \
  --target-ratio 0.1666667 \
  --anchors-per-contour 48 \
  --num-workers 8 \
  --evaluate-exact \
  --write-pred-sqlite
```

## Results

Preprocess summary:

```text
rows_before_prune: 33,740
rows_after_prune: 31,680
removed_short_tracks: 619
removed_rows: 2,060
tracks_after_prune: 398
cuts_detected: 68
scenes: 69
cut_detection_method: ffmpeg_candidates_opencv_verify
elapsed_sec: 37.89
```

Polygon keyframe summary:

```text
source_rows: 31,680
source_tracks: 398
gapfill_inserted_frames: 3,045
effective_stream_count: 483
output_rows: 34,725
global_recall: 0.9719669732206513
global_iou: 0.8404203000211178
predictions_sqlite: output/actual_split/polygon_keyframes/pred/predictions.sqlite
```

Output artifacts:

```text
output/actual_split/preprocess/preprocess/アクセル様２月解析用白カン01.26.tracked.sqlite
output/actual_split/polygon_keyframes/pred/predictions.sqlite
output/actual_split/polygon_keyframes/exact/keyframe_exact_metrics.csv
output/actual_split/polygon_keyframes/summary.json
```

## Notes

The extracted `preprocess` stage caught missing imports during validation
(`subprocess`, `orjson`). The extractor was fixed and the stage then completed
successfully.

The raw extracted `polygon_v22.py` is useful for refactoring, but real-world
input still requires the production contour safety patches from the validated
standalone. Therefore `stage polygon-keyframes` currently routes through
`engine/polygon_runtime.py`, which uses the patched production runner.
