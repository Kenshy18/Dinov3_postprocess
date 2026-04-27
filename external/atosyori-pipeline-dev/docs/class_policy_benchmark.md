# Class Policy Benchmark

Measured on `input/アクセル様２月解析用白カン01.26.mp4` using the preprocessed tracked SQLite from `output/actual_split/preprocess`.

The source data has mixed labels within the same `track_id`, so this benchmark splits by `masks.label` rows:

- `男性器`: polygon approximation, target keyframe interval 3 frames, `target-ratio=1/3`, `max-gap=3`
- `女性器` and `結合部分`: ellipse approximation, target keyframe interval 6 frames, `target-ratio=1/6`, `max-gap=6`

## Inputs

| split | rows | tracks | distinct frames |
| --- | ---: | ---: | ---: |
| 男性器 / polygon | 7,587 | 336 | 3,356 |
| その他 / ellipse | 24,093 | 381 | 13,840 |
| total | 31,680 | - | - |

## Timing

The preprocessing stage was reused. Its previous measured time was 37.89s.

| stage | wall seconds |
| --- | ---: |
| 男性器 polygon 3F | 42.12 |
| その他 ellipse inference | 15.80 |
| その他 ellipse keyframes 6F | 8.93 |
| その他 ellipse exact evaluation | 6.10 |
| postprocess total, excluding preprocessing | 72.95 |
| total including reused preprocessing time | 110.84 |

Postprocess throughput was 434.27 mask rows/s, about 8.23x realtime for the 600.6s video. Including preprocessing time, throughput was 285.83 mask rows/s, about 5.42x realtime.

## Accuracy

| scope | recall | precision | IoU |
| --- | ---: | ---: | ---: |
| 男性器 / polygon 3F | 0.9755 | 0.9849 | 0.9611 |
| 女性器 / ellipse 6F | 0.9832 | 0.7496 | 0.7402 |
| 結合部分 / ellipse 6F | 0.9829 | 0.7456 | 0.7360 |
| combined | 0.9795 | 0.8431 | 0.8285 |

## Keyframe Rate

The requested intervals were used as targets. The optimizers can still emit more keyframes or inflate segments to satisfy recall constraints.

| split | target | actual global keyframe rate |
| --- | ---: | ---: |
| 男性器 / polygon | 0.3333 | 0.3029 |
| その他 / ellipse | 0.1667 | 0.2739 |

The ellipse branch's dense recall stage raised recall after keyframe interpolation to 0.9812, but this increased the actual keyframe rate and reduced precision because many ellipse segments were inflated.

Full machine-readable results are in `output/class_policy_benchmark/benchmark_summary.json`.
