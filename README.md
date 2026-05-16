# Detector integrated postprocess runtime

動画を入力し、DINOv3 + Cascade Mask R-CNN、EVA02 + Cascade Mask R-CNN、DINOv3 + Co-DINO の高速推論、任意のROI分類、Atosyori後処理をまとめて実行する自己完結型ディレクトリです。

## Components

- Backend pipeline: `backend/pipeline/`
- Backend detector adapters: `backend/detectors/`
- ROI classifier runtimes: `backend/classifiers/`
- Shared detector/postprocess schemas: `backend/schemas/`
- Backend postprocess adapter: `backend/postprocess/`
- User-facing backend CLI implementations: `backend/pipeline/cli/`
- Integration entrypoint: `scripts/run_integrated_pipeline.py`（互換ラッパー）
- Qt UI: `apps/qt_ui/`（`UI/` は互換入口）
- DINOv3 runtime: `backend/detectors/dinov3/runtime/`
- EVA02 runtime: `backend/detectors/eva02/runtime/`
- Co-DINO runtime: `backend/detectors/codino/runtime/`
- DINOv3/EVA02/Co-DINO source dependencies: `configs/`, `dinov3/`, `eva02/eva02_det/`, `external/codino/`
- Training implementations: `training/`（`scripts/train_*.py` は互換入口）
- Atosyori postprocess source: `external/atosyori-pipeline-dev/`
- Setup/verification commands: `tools/`（互換入口）
- Setup/artifact/verify/debug implementations: `tools/setup/`, `tools/artifacts/`, `tools/verify/`, `tools/debug/`
- Runtime artifacts: `checkpoints/`

このディレクトリを単体でcloneし、Drive artifactを `checkpoints/` に取得すれば、3つの検出器で動画入力からJSONL、後処理SQLite、overlayまで一気通貫で実行できます。

責務境界は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)、長期保守方針は [docs/MAINTENANCE.md](docs/MAINTENANCE.md)、Flowの詳細は [docs/FLOW.md](docs/FLOW.md)、後処理の現行設定は [docs/POSTPROCESS_SETTINGS.md](docs/POSTPROCESS_SETTINGS.md)、後処理エンジン管理方針は [docs/POSTPROCESS_ENGINE_POLICY.md](docs/POSTPROCESS_ENGINE_POLICY.md)、artifact配置は [docs/ARTIFACTS.md](docs/ARTIFACTS.md)、セットアップ手順は [docs/SETUP.md](docs/SETUP.md)、Windows側UI配置は [docs/WINDOWS_UI.md](docs/WINDOWS_UI.md)、ラン成果物診断は [docs/RUN_AUDIT.md](docs/RUN_AUDIT.md)、整理・変更後の検証は [docs/VERIFICATION.md](docs/VERIFICATION.md) を参照してください。

## Setup

初回セットアップは、Drive artifact取得、仮想環境作成、依存関係インストール、artifact確認、必要に応じたTensorRT engine再作成までまとめて行います。

Runtime artifact Drive:

```text
TODO: upload checkpoints/runtime_artifacts to Google Drive and paste the shared folder URL here.
```

Drive URLを設定する場合は `configs/artifact_sources.env.example` をコピーして編集してください。

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
# configs/artifact_sources.env の RUNTIME_ARTIFACTS_URL に上記Drive URLを設定
```

```bash
tools/setup_integrated_runtime_env.sh
```

GUIを使うPCでは、同じ処理を分かりやすい名前で呼ぶ以下の入口も使えます。UI依存関係のインストール、DINOv3 TensorRT engineの作成/再利用、GPU/VRAMに応じた安全寄りのbatch設定生成まで行います。

```bash
tools/setup_gui_runtime.sh
```

生成された推奨設定は `.runtime/runtime_profile.json` に保存され、セットアップで選ばれたPython/venvやTensorRT engineは `.runtime/gui_runtime.env` に保存されます。セットアップ時には一時的なダミー動画でbatch-size候補を順番に測定し、結果を `.runtime/runtime_benchmark.json` に保存してから一時動画と出力を削除します。これらはPC/GPUごとのローカル設定なのでgitignore対象です。GUI起動時と `scripts/run_integrated_pipeline.py` の既定値はこの設定を参照します。目安値とスキーマは `configs/runtime_profile.example.json` に記載しています。特にEVA02はVRAM不足時に共有メモリへ落ちると極端に遅くなるため、測定できない場合の既定batch-sizeは安全寄りにしています。

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

Drive artifact取得後、推論から後処理まで実行するには以下が配置されます。

```text
checkpoints/
  detector/model_final.pth
  dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  classifier/best.pt
  eva02/detector/model_final.pth
  eva02/classifier/best.pt
  codino/detector/resolved_config.py
  codino/detector/epoch_2.pth
  codino/classifier/best.pt
  codino/trt/codino_dinov3_vitl_backbone_736x1280_fp32_b2_fixed_bf16.engine
  codino/trt/codino_query_encoder_b2_736x1280_msda_plugin_sbc_fp16.engine
  codino/trt/codino_decoder_b2_736x1280_msda_plugin_fp16.engine
  codino/trt/codino_mask_head_core_n1_736x1280_fp16.engine
  trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
  postprocess/k2_v5/best_exact.pt
  postprocess/polygon_point_predictor/best.pt
  postprocess/polygon_point_predictor/feature_stats.npz
```

配置確認:

```bash
python tools/check_artifacts.py
```

整理や設定変更のあとにまとめて確認する場合:

```bash
.venv_integrated/bin/python tools/verify_runtime.py
```

標準policyは全クラスellipseのため、標準後処理だけならpolygon predictor artifactは不要です。polygon branchを使う独自policyを指定する場合は `polygon_point_predictor/best.pt` と `feature_stats.npz` を配置してください。

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

EVA02 を使う場合:

```bash
.venv_integrated/bin/python \
  scripts/infer_video_postprocess.py \
  --detector eva02 \
  --input input/sample.mp4 \
  --output-root output/runs \
  --force
```

Co-DINO を使う場合:

```bash
.venv_integrated/bin/python \
  scripts/infer_video_postprocess.py \
  --detector codino \
  --input input/sample.mp4 \
  --output-root output/runs \
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
  <detector>/
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

## FPS metrics

DINOv3推論summaryには互換性のため古い名前も残していますが、速度比較では以下を使い分けます。

- `e2e_fps` / `wall_fps`: DINOv3推論スクリプト内のE2E速度。decode/preprocess待ち、warmup frame、JSONL writer flushを含みます。実運用の動画処理速度はこちらを基準にします。
- `compute_fps` / `measured_fps`: warmup後、prefetch済みbatchが取得できた後から測るGPU推論寄りの速度。decode/preprocess待ちを含まないため、短い動画やI/O律速ではE2Eより大きく出ます。

## Default acceleration

DINOv3 Cascade と Co-DINO は高速化済みruntimeを使います。

- TensorRT BF16 DINOv3 backbone
- Cascade 1 stage
- RPN pre/post NMS top-k `100/40`
- BF16 autocast
- batch size `8`
- async JSONL writer
- Co-DINOはDINOv3 backbone、query encoder、decoder、mask headのTensorRT engineを利用可能

## Class policy

標準設定は `configs/class_policy_default.json` です。

- `男性器`: ellipse, 3フレーム相当, recall `0.96`
- `女性器`: ellipse, 3フレーム相当, recall `0.96`
- `結合部分`: ellipse, 3フレーム相当, recall `0.96`
- その他: ellipse, 3フレーム相当, recall `0.96`

同じ内容を明示名で参照したい場合は `configs/class_policy_all_ellipse_int3_recall096.json` も使えます。

現行の後処理デフォルトは [docs/POSTPROCESS_SETTINGS.md](docs/POSTPROCESS_SETTINGS.md) に明記しています。主な条件は全クラスellipse、3フレーム間隔、recall `0.96`、K1N sequence routing、元マスク同梱、エッジ条件なしの10フレーム線形fit/5フレームendpoint外挿です。

## Upload artifacts

Driveや外部artifact storeへ配布する場合、検出から後処理まで一気通貫で必要なcheckpoint/model artifactは `artifacts_to_upload/runtime_artifacts/` にまとめられます。

```text
artifacts_to_upload/runtime_artifacts/
  checkpoints/detector/model_final.pth
  checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
  checkpoints/classifier/best.pt
  checkpoints/eva02/detector/model_final.pth
  checkpoints/eva02/classifier/best.pt
  checkpoints/trt/dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine
  checkpoints/codino/detector/resolved_config.py
  checkpoints/codino/detector/epoch_2.pth
  checkpoints/codino/classifier/best.pt
  checkpoints/codino/trt/*.engine
  checkpoints/postprocess/k2_v5/best_exact.pt
  checkpoints/postprocess/k2_v5/run_config.json
  checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
  checkpoints/postprocess/polygon_point_predictor/best.pt
  checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
  checkpoints/postprocess/polygon_point_predictor/run_config.json
  checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

このフォルダをGoogle Driveへアップロードし、共有URLを `configs/artifact_sources.env` の `RUNTIME_ARTIFACTS_URL` に設定してください。TensorRT engineはGPU/driver/TensorRT依存なので、別PCで読み込みに失敗する場合は初回セットアップで再作成します。

## GPU portability

TensorRT engineはGPU、driver、CUDA、TensorRT versionに依存するため、別PCでは `tools/setup_integrated_runtime_env.sh` で再作成してください。checkpointとONNX exportは持ち回れます。

- RTX 5090などのBlackwell系: CUDA/PyTorch/TensorRTが `sm_120` に対応している必要があります。この検証機では RTX PRO 6000 Blackwell + PyTorch CUDA 12.9 + TensorRT 10.13 で確認済みです。
- RTX 4090などのAda系: TensorRT engineをそのPCで再作成する前提で対応想定です。
- 既定はTensorRT BF16です。BF16 buildが失敗した場合、セットアップは同じengine pathにFP16で自動fallbackします。明示する場合は `TRT_PRECISION=fp16 tools/setup_integrated_runtime_env.sh` を使えます。
