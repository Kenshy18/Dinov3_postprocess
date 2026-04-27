# Standalone Equivalence Check

Compared the split package commands against the original single-file script:

```bash
../run_standalone.py
```

The original script and the preserved legacy copy are byte-identical:

```text
904def03463765acde115a4e5336c90a32c5a62798fac78e973782ec90b20fda  ../run_standalone.py
904def03463765acde115a4e5336c90a32c5a62798fac78e973782ec90b20fda  src/atosyori_postprocess/legacy/run_standalone.py
```

## Result

Behavioral outputs matched. Timing fields and output-path strings differ where expected.

Machine-readable details:

```text
output/standalone_compare/comparison_summary.json
```

## Checks

| area | comparison | result |
| --- | --- | --- |
| preprocess | `masks`, `tracks`, `cuts` SQLite row hashes | equal |
| ellipse inference | metrics CSV sha256 | equal |
| ellipse inference | predictions SQLite row hash | equal |
| ellipse 6F keyframes | `interpolated_union.json` sha256 | equal |
| ellipse 6F keyframes | `final_keyframes.json` sha256 | equal |
| ellipse 6F keyframes | `stream_segments.csv` sha256 | equal |
| ellipse 6F exact evaluation | `keyframe_exact_metrics.csv` sha256 | equal |
| polygon 3F keyframes | `interpolated_union.json` sha256 | equal |
| polygon 3F keyframes | `final_keyframes.json` sha256 | equal |
| polygon 3F keyframes | `stream_segments.csv`, excluding timing columns | equal |
| polygon 3F exact evaluation | `keyframe_exact_metrics.csv` sha256 | equal |
| polygon 3F predictions | predictions SQLite row hash | equal |

The polygon `stream_segments.csv` file differs only in per-run timing columns such as `build_eval_contexts_seconds` and `solve_dp_seconds`. All non-timing columns are equal.

## Scope

The comparison used the real input-derived SQLite data already present in this workspace:

- full preprocess from JSONL + MP4
- `その他` ellipse branch with 6-frame target keyframes
- `男性器` polygon branch with 3-frame target keyframes

The full monolithic pipeline's class policy groups by track label, while the benchmark request required row-label behavior because labels are mixed within the same `track_id`. For that reason, this equivalence check compares the underlying stages with the same concrete split inputs and arguments.
