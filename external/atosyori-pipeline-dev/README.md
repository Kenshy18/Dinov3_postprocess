# atosyori-pipeline-dev

Atosyori後処理を開発・保守運用しやすくするための作業ディレクトリです。

元の巨大なスタンドアロンスクリプトは `src/atosyori_postprocess/legacy/run_standalone.py` に保持し、外側を機能単位のモジュールに分けています。

## セットアップ

```bash
python -m pip install -e .
python -m atosyori_postprocess copy-models --source-root .. --replace
python -m atosyori_postprocess doctor
```

## モデルファイル

checkpoint本体はGit管理せず、以下のGoogle Driveに置いています。

https://drive.google.com/drive/folders/10-Zc2ShIkJn7T1JIcvUgiEgdSoIANJcT?usp=sharing

取得したモデルは `models/` に配置してください。期待する配置は [models/README.md](/home/kenke/Atosyori0331/Atosyori0425/FINAL_ATOSYORI/atosyori-pipeline-dev/models/README.md) に記載しています。

## フル実行

```bash
python -m atosyori_postprocess run \
  --input-sqlite input/sample.sqlite \
  --output-dir output/sample
```

JSONL + 動画から始める場合:

```bash
python -m atosyori_postprocess run \
  --input-jsonl input/sample.jsonl \
  --input-video input/sample.mp4 \
  --output-dir output/sample
```

polygonをデフォルトにする場合:

```bash
python -m atosyori_postprocess run \
  --input-sqlite input/sample.sqlite \
  --output-dir output/polygon \
  --default-shape-mode polygon
```

## Stage単位で実行

各機能を個別に呼べます。現在は `engine/` に抽出したモジュールの `main()` を直接呼びます。
ellipse/polygon stageでは `models/` のモデルパスを自動で補います。

```bash
python -m atosyori_postprocess stage preprocess -- --help
python -m atosyori_postprocess stage ellipse-inference -- --help
python -m atosyori_postprocess stage polygon-keyframes -- --help
python -m atosyori_postprocess stage render -- --help
```

モデルルートを明示したい場合:

```bash
python -m atosyori_postprocess stage --model-root /path/to/models polygon-keyframes -- --help
```

## 固定設定の実推論スクリプト

コマンドライン引数ではなく、Pythonコード内の定数で入力・モデル・出力・クラス別設定を管理する実行スクリプトです。

```bash
python scripts/run_actual_inference.py
```

設定は [scripts/run_actual_inference.py](/home/kenke/Atosyori0331/Atosyori0425/FINAL_ATOSYORI/atosyori-pipeline-dev/scripts/run_actual_inference.py) の `User settings` ブロックを編集してください。デフォルトでは `男性器` をpolygon 3フレーム間隔、その他をellipse 6フレーム間隔で処理し、`output/actual_inference/final/predictions.sqlite` を出力します。

## Engine抽出モジュール

`legacy/run_standalone.py` から機能別に抽出したコードは `src/atosyori_postprocess/engine/` にあります。

```bash
python -m atosyori_postprocess engine-check
python tools/extract_legacy_modules.py
```

`extract_legacy_modules.py` はlegacyからengineを再生成するための機械的な開発ツールです。

## 実行確認用の合成データ

実データがなくても、合成SQLiteでpolygon stageを実行確認できます。

```bash
python -m atosyori_postprocess make-sample --output output/dev_sample.sqlite --frames 8
python -m atosyori_postprocess smoke --work-dir output/smoke --force
```

`smoke` は小さな矩形トラックを作り、polygon keyframe最適化、exact評価、predictions.sqlite出力まで確認します。

## 構造

詳細は `docs/structure.md` を見てください。

大事な方針:

- チェックポイントは `models/` に実体コピーとして置く
- 入力と出力は `input/`, `output/` に置く
- 既存エンジンの挙動は保ったまま、周辺を機能別に分ける
- 将来アルゴリズム本体を分割するときは `stages/` の境界に沿って進める

## 開発確認

```bash
make test
make engine-check
make smoke
```
