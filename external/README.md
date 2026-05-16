# external

Repo-local copies of external runtime code live here. Large generated files,
checkpoints, TensorRT engines, videos, JSONL, and SQLite outputs stay ignored.

`external/codino` contains the Co-DINO source needed by
`backend/detectors/codino/runtime/codino_video_fast_runtime.py`. Co-DINO model
weights, configs, classifier checkpoints, and TRT engines are resolved from
`checkpoints/codino`.

Atosyori後処理repoをこの統合ディレクトリ内へ軽量コピーしたい場合の置き場でもあります。

```bash
inference/dinov3_integrated_postprocess_runtime/tools/sync_atosyori_source.sh
```

実行時の既定値は `external/atosyori-pipeline-dev` です。別の作業コピーを明示的に使う場合だけ
`ATOSYORI_REPO=/path/to/atosyori-pipeline-dev` を設定します。
