# DINOv3 integrated postprocess runtime

動画を入力し、DINOv3 + Cascade Mask R-CNN の高速推論、任意のROI分類、Atosyori後処理をまとめて実行する自己完結型ディレクトリです。

## Components

- Integration entrypoint: `scripts/run_integrated_pipeline.py`
- DINOv3 runtime: `inference/dinov3_video_jsonl_runtime/`
- DINOv3/EVA02 source dependencies: `scripts/`, `configs/`, `dinov3/`, `eva02/eva02_det/`
- Atosyori postprocess source: `external/atosyori-pipeline-dev/`
- Runtime artifacts: `checkpoints/`

このディレクトリを単体でcloneし、checkpoint/engineを `checkpoints/` に配置すれば、動画入力からJSONL、後処理SQLite、overlayまで一気通貫で実行できます。

Flowの詳細は [docs/FLOW.md](docs/FLOW.md)、artifact配置は [docs/ARTIFACTS.md](docs/ARTIFACTS.md)、セットアップ手順は [docs/SETUP.md](docs/SETUP.md) を参照してください。

## Setup

初回セットアップは、checkpoint等のダウンロード、配置、仮想環境作成、依存関係インストール、TensorRT engine作成までまとめて行います。

標準のartifact Drive URLはセットアップスクリプトに設定済みです。別URLを使う場合だけ `configs/artifact_sources.env.example` をコピーして編集してください。

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
# configs/artifact_sources.env の RUNTIME_ARTIFACTS_URL を編集
```

```bash
tools/setup_integrated_runtime_env.sh
```

別のAtosyori repoを使う場合:

```bash
ATOSYORI_REPO=/path/to/atosyori-pipeline-dev tools/setup_integrated_runtime_env.sh
```

Atosyori repoもこの統合ディレクトリ内に持たせたい場合:

```bash
tools/sync_atosyori_source.sh
```

同期後は `external/atosyori-pipeline-dev` が自動で優先されます。

## Checkpoints

推論から後処理まで実行するには以下を配置してください。

```text
checkpoints/
  detector/model_final.pth
  dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  classifier/best.pt
  trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
  postprocess/k2_v5/best_exact.pt
  postprocess/polygon_point_predictor/best.pt
  postprocess/polygon_point_predictor/feature_stats.npz
```

配置確認:

```bash
python tools/check_artifacts.py
```

標準policyは `男性器` にpolygon branchを使うため、`polygon_point_predictor/best.pt` と `feature_stats.npz` が無い場合は標準policyのフル後処理は完走しません。モデル未配置で接続確認だけ行う場合は `configs/class_policy_ellipse_only.json` を使ってください。

## Run

通常の動画推論から後処理までは、簡易入口を使います。

```bash
.venv_integrated/bin/python \
  scripts/infer_video_postprocess.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --overlay \
  --force
```

詳細な全オプションを触る場合は統合入口を直接使います。

```bash
.venv_integrated/bin/python \
  scripts/run_integrated_pipeline.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --classifier \
  --render-overlays \
  --force
```

後処理モデルがまだない状態でDINOv3 JSONLだけ確認する場合:

```bash
.venv_integrated/bin/python \
  scripts/run_integrated_pipeline.py \
  --input input/sample.mp4 \
  --output-root /tmp/dinov3_integrated_smoke \
  --max-frames 8 \
  --warmup-frames 0 \
  --no-postprocess \
  --force
```

## Outputs

1動画ごとに以下を作ります。

```text
output/runs/<run_name>/
  dinov3/
    jsonl/<video_stem>.jsonl
    summary.json
  postprocess/
    summary.json
    preprocess/<video_stem>.tracked.sqlite
    keyframes/int_*/merged/predictions.sqlite
  sqlite/
    <video_stem>_tracked.sqlite
    <video_stem>_int_*_predictions.sqlite
  overlay/
    <video_stem>_int_*_postprocess.mp4
  summary.json
```

`sqlite/` と `overlay/` は後処理成果物へのsymlinkです。symlinkが作れない環境ではコピーします。

## Default acceleration

DINOv3側は既存の高速化済みruntimeを使います。

- TensorRT BF16 DINOv3 backbone
- Cascade 1 stage
- RPN pre/post NMS top-k `100/40`
- BF16 autocast
- batch size `8`
- async JSONL writer

## Class policy

標準設定は `configs/class_policy_default.json` です。

- `男性器`: polygon, 3フレーム相当
- `女性器`: ellipse, 6フレーム相当
- `結合部分`: ellipse, 6フレーム相当
- その他: ellipse, 6フレーム相当

全クラスをellipseで動かす検証用設定は `configs/class_policy_ellipse_only.json` です。

## Upload artifacts

検出から後処理まで一気通貫で必要なcheckpoint/model artifactは `artifacts_to_upload/runtime_artifacts/` にまとめています。

```text
artifacts_to_upload/runtime_artifacts/
  checkpoints/detector/model_final.pth
  checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  checkpoints/classifier/best.pt
  checkpoints/postprocess/k2_v5/best_exact.pt
  checkpoints/postprocess/k2_v5/run_config.json
  checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
  checkpoints/postprocess/polygon_point_predictor/best.pt
  checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
  checkpoints/postprocess/polygon_point_predictor/run_config.json
  checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

このフォルダをGoogle Driveへアップロードし、共有URLを `configs/artifact_sources.env` の `RUNTIME_ARTIFACTS_URL` に設定してください。TensorRT engineはPC依存なのでアップロード対象から外し、初回セットアップで作成します。
