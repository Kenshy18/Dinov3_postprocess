# SOD推論システム Qt UI

`backend.pipeline` の互換入口 `scripts/run_integrated_pipeline.py` をQtから実行するフロントエンドです。
DINOv3/EVA02の切り替え、入力動画キュー、結果保存先、オーバーレイ、後処理パラメータを画面から指定できます。
インターレース入力、AVI、ProRes/MOV、回転メタ、非正方ピクセル、VFR、10bit/4:2:2系pix_fmtなどは実行前にH.264/MP4へ自動正規化し、変換後のプログレッシブ動画を推論に使います。

## 起動

```bash
cd /home/kenke/Dinov3_postprocess
tools/setup_gui_runtime.sh
apps/qt_ui/run_app.sh
```

`UI/run_app.sh` は既存ショートカット用の互換入口です。

`tools/setup_gui_runtime.sh` はUI依存関係、推論依存関係、DINOv3 TensorRT engine、batch-size実測をまとめて準備します。batch-size測定では一時的なダミー動画を作成し、候補を順番に実行して `configs/runtime_benchmark.json` に結果を残した後、ダミー動画と一時出力を削除します。生成される `configs/runtime_profile.json`、`configs/runtime_benchmark.json`、`configs/gui_runtime.env` はPC依存のためgitignore対象です。`apps/qt_ui/run_app.sh` とGUI内の実行コマンドは、この設定からPython/venv、batch-size、TensorRT engineを選択します。目安値は `configs/runtime_profile.example.json` にあります。

## 実行内容

キュー内の動画を1件ずつ `scripts/run_integrated_pipeline.py` に渡します。
フォルダを追加した場合は、実行開始時に対応動画ファイルへ展開して1本ずつ処理します。
ログは画面左下へそのまま流し、出力は既定で `output/runs/<run_name>/` に作成します。

主な対応項目:

- Backend: `DINOv3` / `EVA02`
- 入力: 動画ファイル複数選択、またはフォルダ
- 後処理: on/off、楕円/ポリゴン、キーフレーム間隔、recall、confidence
- overlay: 詳細後処理overlay、簡易後処理overlay、推論JSONL overlay（複数同時選択可）
- 詳細: `max-frames`、`batch-size`、`warmup-frames`、`score-thresh`

固定値:

- 短命トラック削除: `10`
- 後処理overlay encoder: `nvenc`（GPU）
- 後処理進捗ログ間隔: `5` 秒

## 出力フォルダ

1ジョブごとに `output/runs/<run_name>/` 以下へ、内部成果物に加えて見やすい名前のフォルダを作ります。

- `最終成果物.json`: 主要成果物の一覧
- `sod_job_dir/`: UIジョブ情報、インターレース正規化動画
- `統合マスクオーバーレイ/`: 簡易オーバーレイ。後処理後マスクのみ
- `詳細オーバーレイ/`: 元マスク、後処理後輪郭、ID/クラス表示
- `AI生成カバーオーバーレイ/`: AI生出力マスクのみ
- `最終SQLite/`: 後処理後SQLite
- `推論生SQLite/`: 推論JSONLから作った生トラックSQLite
- `jsonl/`: 推論JSONLと推論summary
- `logs/`: UIジョブログ
- `logs/audit.jsonl`: 入力ffprobe、正規化理由、実行コマンド、終了コード、エラーtraceback
- `postprocessed/`: 後処理内部成果物
