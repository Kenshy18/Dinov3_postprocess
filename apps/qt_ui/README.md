# SOD推論システム Qt UI

`backend.pipeline` の互換入口 `scripts/run_integrated_pipeline.py` をQtから実行するフロントエンドです。
DINOv3/EVA02/Co-DINOの切り替え、任意のRT-DETR顔・頭検出、入力動画キュー、結果保存先、オーバーレイ、後処理パラメータを画面から指定できます。
インターレース入力、AVI、ProRes/MOV、回転メタ、非正方ピクセル、VFR、10bit/4:2:2系pix_fmtなどは実行前にH.264/MP4へ自動正規化し、変換後のプログレッシブ動画を推論に使います。

## 起動

```bash
cd Dinov3_postprocess
tools/setup_gui_runtime.sh
apps/qt_ui/run_app.sh
```

`UI/run_app.sh` は既存ショートカット用の互換入口です。
Windows/PowerShell から起動する場合は `apps/qt_ui/run_app.ps1` を使います。配置方針は `docs/WINDOWS_UI.md` を参照してください。

`tools/setup_gui_runtime.sh` はUI依存関係、推論依存関係、DINOv3 TensorRT engine、repo-local RT-DETR Head/Face runtime、Drive artifactから取得するRT-DETR Head/Face重み、batch-size実測をまとめて準備します。batch-size測定では一時的なダミー動画を作成し、DINOv3/EVA02/Co-DINO/RT-DETRの候補を順番に実行して `.runtime/runtime_benchmark.json` に結果を残した後、ダミー動画と一時出力を削除します。生成される `.runtime/runtime_profile.json`、`.runtime/runtime_benchmark.json`、`.runtime/gui_runtime.env` はPC依存のためgitignore対象です。`apps/qt_ui/run_app.sh` とGUI内の実行コマンドは、この設定からPython/venv、batch-size、TensorRT engine、RT-DETR config/checkpointを選択します。目安値は `configs/runtime_profile.example.json` にあります。

## 実行内容

キュー内の動画を1件ずつ `scripts/run_integrated_pipeline.py` に渡します。
フォルダを追加した場合は、実行開始時に対応動画ファイルへ展開して1本ずつ処理します。
ログは画面左下へそのまま流し、出力は既定で `output/runs/<run_name>/` に作成します。

主な対応項目:

- 検出エンジン: `DINOv3` / `EVA02` / `Co-DINO` / `顔・頭のみ（AIなし）`
- 追加検出: 通常のAI検出にRT-DETR顔・頭検出を追加
- Head/Face専用: `顔・頭のみ（AIなし）` を選ぶとAI検出と後処理をスキップしてHead/Face検出のみ実行し、bboxとFace楕円マスクのオーバーレイを必ず生成
- 入力: 動画ファイル複数選択、またはフォルダ
- 後処理: on/off、楕円/ポリゴン、キーフレーム間隔、recall、confidence
- overlay: 詳細後処理overlay、簡易後処理overlay、推論JSONL overlay、顔・頭検出overlay（複数同時選択可）
- 詳細: `max-frames`、`batch-size`、`warmup-frames`、`score-thresh`

固定値:

- 短命トラック削除: `10`
- 後処理overlay encoder: `nvenc`（GPU）
- 後処理進捗ログ間隔: `5` 秒

GUIは入力動画キュー、結果保存先、実行Python、Run名Prefix、検出エンジン、オーバーレイ選択、自動後処理、クラス別のマスクタイプ/キーフレーム間隔/recall/confidence、詳細設定を `.runtime/qt_ui_settings.json` に保存し、次回起動時に復元します。このファイルはPCごとのローカル設定なのでgitignore対象です。

## 出力フォルダ

1ジョブごとに `output/runs/<run_name>/` 以下へ、ユーザー向け成果物だけを見やすい名前のフォルダに作ります。成功時は検出器JSONL、`postprocess/`、内部 `sqlite/`、`config/`、`summary.json` などのデバッグ成果物を削除します。

- `最終成果物.json`: 主要成果物の一覧
- `統合マスクオーバーレイ/`: 簡易オーバーレイ。後処理後マスクとFace楕円マスク
- `詳細オーバーレイ/`: 元マスク、後処理後輪郭、Head/Face bbox、Face楕円マスク、ID/クラス表示
- `AI生成カバーオーバーレイ/`: AI生出力マスクのみ
- `顔頭生出力オーバーレイ/`: Head/Face bboxとFace楕円マスクのみ。Head/Face専用では必ず生成
- `最終SQLite/`: `AI後処理最終.sqlite`、`AI後処理_顔頭統合最終.sqlite`
- `推論生SQLite/`: 推論JSONLから作った生raw/tracked SQLite、Head/FaceのみSQLite
- `logs/`: UIジョブログ
- `logs/audit.jsonl`: 入力ffprobe、正規化理由、実行コマンド、終了コード、エラーtraceback
- `logs/pipeline_summary.json`: 削除前のpipeline summary
- `logs/job_audit_summary.json`: JSONL契約、SQLite件数、overlayサイズ、警告

失敗時は原因調査のため内部成果物を残します。成功時にも内部成果物が必要な場合は `apps/qt_ui/run_ui_job.py --keep-debug-outputs` を使います。

GUI以外で同じ成果物診断を行う場合は `tools/diagnose_run.py <run_dir>` を使います。
