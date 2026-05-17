# Detector integrated postprocess runtime

動画を入力し、DINOv3 + Cascade Mask R-CNN、EVA02 + Cascade Mask R-CNN、DINOv3 + Co-DINO の高速推論、任意のROI分類、Atosyori後処理をまとめて実行する自己完結型ディレクトリです。

## Components

- Backend pipeline: `backend/pipeline/`
- Backend detector adapters: `backend/detectors/`
- ROI classifier runtimes: `backend/classifiers/`
- Shared detector/postprocess schemas: `backend/schemas/`
- Backend postprocess adapter: `backend/postprocess/`
- User-facing backend CLI implementations: `backend/pipeline/cli/`
- Primary CLI entrypoints: `tools/setup_runtime.sh`, `scripts/infer.py`, `scripts/overlay.py`
- Low-level integration entrypoint: `scripts/run_integrated_pipeline.py`（互換ラッパー）
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
https://drive.google.com/drive/folders/1cj9gPOt4MIRu6cFW80vfTZHK9oLQB6qn
```

Drive URLを設定する場合は `configs/artifact_sources.env.example` をコピーして編集してください。

```bash
cp configs/artifact_sources.env.example configs/artifact_sources.env
# configs/artifact_sources.env の RUNTIME_ARTIFACTS_URL に上記Drive URLを設定
```

```bash
tools/setup_runtime.sh
```

`tools/setup_runtime.sh` は正規セットアップ入口です。UI依存関係のインストール、DINOv3/Co-DINO TensorRT engineの作成/再利用、GPU/VRAMに応じた3モデル共通のbatch探索まで行います。互換入口として `tools/setup_integrated_runtime_env.sh` と `tools/setup_gui_runtime.sh` も残しています。

```bash
tools/setup_gui_runtime.sh
```

生成された推奨設定は `.runtime/runtime_profile.json` に保存され、セットアップで選ばれたPython/venvやTensorRT engineは `.runtime/gui_runtime.env` に保存されます。セットアップ時にはDINOv3、EVA02、Co-DINOのbatch-size候補を本番推論に近い設定で順番に測定し、結果を `.runtime/runtime_benchmark.json` に保存してから一時動画と出力を削除します。既定では `input/` 配下の最初の動画を240フレームだけ使い、動画が無い場合は一時動画へフォールバックします。`BATCH_BENCHMARK_INPUT=/path/to/sample.mp4` を指定すると任意の実動画サンプルで探索できます。これらはPC/GPUごとのローカル設定なのでgitignore対象です。GUI起動時と `scripts/run_integrated_pipeline.py` の既定値はこの設定を参照します。目安値とスキーマは `configs/runtime_profile.example.json` に記載しています。特にEVA02はTensorRT engineを作らない構成でも実測batch探索を行いますが、VRAM不足時に共有メモリへ落ちると極端に遅くなるため、測定できない場合の既定batch-sizeは安全寄りにしています。Co-DINOはDeformable Attentionを含むquery encoder/decoder/mask headを候補batchごとにローカルTensorRT engineとして作成し、その後の実測で選ばれたbatch-sizeをprofileへ反映します。

別のAtosyori repoを使う場合:

```bash
ATOSYORI_REPO=/path/to/atosyori-pipeline-dev tools/setup_runtime.sh
```

Atosyori repoもこの統合ディレクトリ内に持たせたい場合:

```bash
tools/sync_atosyori_source.sh
```

同期後は `external/atosyori-pipeline-dev` が自動で優先されます。

## Checkpoints

Drive artifact取得後、推論から後処理まで実行するには以下が配置されます。
`codino/detector/resolved_config.py` は小さいPython設定ファイルのためGitで管理します。

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
  postprocess/k2_v5/best_exact.pt
  postprocess/polygon_point_predictor/best.pt
  postprocess/polygon_point_predictor/feature_stats.npz
```

TensorRT engineはセットアップ時にローカルGPU向けに作成されます。

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

通常の動画推論から後処理までは、正規入口 `scripts/infer.py` を使います。既定ではGUIと同じ成果物配置を作り、詳細overlayも生成します。

```bash
.venv_integrated/bin/python \
  scripts/infer.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --force
```

EVA02 を使う場合:

```bash
.venv_integrated/bin/python \
  scripts/infer.py \
  --detector eva02 \
  --input input/sample.mp4 \
  --output-root output/runs \
  --force
```

Co-DINO を使う場合:

```bash
.venv_integrated/bin/python \
  scripts/infer.py \
  --detector codino \
  --input input/sample.mp4 \
  --output-root output/runs \
  --force
```

推論のみ、後処理のみも同じ入口を使います。

```bash
.venv_integrated/bin/python \
  scripts/infer.py \
  --input input/sample.mp4 \
  --output-root output/runs \
  --mode inference \
  --force
```

```bash
.venv_integrated/bin/python \
  scripts/infer.py \
  --mode postprocess \
  --input-jsonl output/runs/sample/jsonl/sample.jsonl \
  --input-video input/sample.mp4 \
  --output-root output/runs \
  --force
```

overlayのみ再生成したい場合は `scripts/overlay.py` を使います。

```bash
.venv_integrated/bin/python \
  scripts/overlay.py \
  --video input/sample.mp4 \
  --pred-sqlite output/runs/sample/最終SQLite/女性器_predictions.sqlite \
  --tracked-sqlite output/runs/sample/推論生SQLite/sample.tracked.sqlite \
  --mode detailed \
  --force
```

NVENC overlay encodeは既定で `h264_nvenc -preset p5 -cq 23` を使います。
画質優先なら `OVERLAY_NVENC_CQ=20`、環境別の追加検証では
`OVERLAY_NVENC_PRESET` と `OVERLAY_FFMPEG_EXTRA_ARGS` でffmpeg引数を上書きできます。

長時間処理では `--progress-interval-sec` 間隔で `[phase-progress]` が出ます。
GUIは同じ行を読んでフェーズ進捗、FPS、ETA、経過時間を更新します。

`scripts/run_full_flow.py`、`scripts/run_postprocess_only.py`、`scripts/infer_video_postprocess.py`、`scripts/run_integrated_pipeline.py`、`scripts/render_raw_jsonl_overlays.py` は互換・低レベル入口です。通常運用では上記3入口を使ってください。

`scripts/infer.py` の出力は後処理前と後処理後の2フェーズで選べます。

```bash
# 後処理前: raw detector SQLite + raw overlay
--pre-sqlite --pre-overlay

# 後処理後: final SQLite + postprocess overlay
--post-sqlite --post-overlay detailed
```

互換のため `--raw-sqlite`、`--raw-overlay`、`--overlay-mode` も残していますが、新規の運用では `pre/post` 名を使ってください。

## Outputs

`scripts/infer.py` の既定出力はGUIと同じ配置です。1動画ごとに以下を作ります。

```text
output/runs/<run_name>/
  最終成果物.json
  sod_job_dir/job_manifest.json
  最終SQLite/
    <label>_predictions.sqlite
  推論生SQLite/
    <video_stem>_raw_detections.sqlite
    <video_stem>.tracked.sqlite
  詳細オーバーレイ/
    <label>_detailed.mp4
  統合マスクオーバーレイ/
    <label>_simple.mp4
  AI生成カバーオーバーレイ/
    <video_stem>_ai_raw_mask.mp4
  jsonl/
    <video_stem>.jsonl
  logs/
    infer_cli.log
    infer_audit.jsonl
    ui_job.log
    audit.jsonl
    job_audit_summary.json
  <detector>/
    jsonl/<video_stem>.jsonl
    summary.json
  postprocess/
    summary.json
    preprocess/<video_stem>.tracked.sqlite
    keyframes/int_*/merged/predictions.sqlite
  summary.json
```

overlayは `--pre-overlay` と `--post-overlay` で選択できます。内部互換の `<detector>/jsonl`、`postprocess/`、`summary.json` も残します。
`*_raw_detections.sqlite` は検出直後の共通raw schemaで、`metadata`、`frames`、`masks` tableを持ちます。後処理後の `*_predictions.sqlite` は最終mask schemaで、少なくとも `masks(frame, track_id, polygons)` を持ちます。

`input/` と `output/` はローカル作業用です。フォルダ内の動画、SQLite、overlay、推論結果は `.gitignore` 対象で、Gitにはpushしません。

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
  checkpoints/codino/detector/epoch_2.pth
  checkpoints/codino/classifier/best.pt
  checkpoints/postprocess/k2_v5/best_exact.pt
  checkpoints/postprocess/k2_v5/run_config.json
  checkpoints/postprocess/k2_v5/train_k2_slot_set_spd_standalone_v5.py
  checkpoints/postprocess/polygon_point_predictor/best.pt
  checkpoints/postprocess/polygon_point_predictor/feature_stats.npz
  checkpoints/postprocess/polygon_point_predictor/run_config.json
  checkpoints/postprocess/polygon_point_predictor/train_mask_point_predictor.py
```

このフォルダをGoogle Driveへアップロードし、共有URLを `configs/artifact_sources.env` の `RUNTIME_ARTIFACTS_URL` に設定してください。TensorRT engineはGPU/driver/TensorRT依存なのでDrive配布の必須artifactには含めず、初回セットアップで再作成します。

## GPU portability

TensorRT engineはGPU、driver、CUDA、TensorRT versionに依存するため、別PCでは `tools/setup_runtime.sh` で再作成してください。checkpointは持ち回れます。

- RTX 5090などのBlackwell系: CUDA/PyTorch/TensorRTが `sm_120` に対応している必要があります。この検証機では RTX PRO 6000 Blackwell + PyTorch CUDA 12.9 + TensorRT 10.13 で確認済みです。
- RTX 4090などのAda系: TensorRT engineをそのPCで再作成する前提で対応想定です。
- DINOv3 backboneの既定はTensorRT BF16です。BF16 buildが失敗した場合、セットアップは同じengine pathにFP16で自動fallbackします。明示する場合は `TRT_PRECISION=fp16 tools/setup_runtime.sh` を使えます。
- Co-DINOは `backend/detectors/codino/tools/rebuild_codino_trt_engines.sh` でbackbone、query encoder、decoder、mask head coreを作成します。query encoder/decoderは `MultiscaleDeformableAttnPlugin_TRT` を使います。
- 現行の既定は `TENSORRT_PIP_SPEC=tensorrt==10.13.0.35` です。driver/CUDAに合わないTensorRT wheelはimportできてもbuilder初期化で失敗します。
- Co-DINOの `mmcv-full==1.7.2` はPyTorch/CUDA/Python ABIに依存します。セットアップは既存import確認、既知wheel、OpenMMLab wheel-only install、Ada以前で必要な場合のPyTorch 2.1.2/CUDA 12.1への寄せ直し、明示許可されたsource buildの順に試し、失敗時は検出環境と回避方法を表示します。Blackwell系ではCUDA 12.9対応PyTorchまたは `REFERENCE_VENV` を使います。
