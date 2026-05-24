from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from apps.qt_ui.run_ui_job import organize_outputs, runtime_snapshot
from backend.pipeline.run_audit import audit_run_dir, audit_to_markdown, build_output_audit


class UiJobAuditTests(unittest.TestCase):
    def _write_mask_sqlite(self, path: Path, *, frame: int = 0, track_id: str = "1", label: str = "女性器") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("create table masks(frame integer, track_id text, polygons text, label text)")
            conn.execute("create table tracks(track_id text, label text)")
            conn.execute("create table raw_tracked_masks(frame integer, raw_track_id text)")
            conn.execute("create table raw_tracks(raw_track_id text)")
            conn.execute("insert into masks values(?, ?, '[]', ?)", (frame, track_id, label))
            conn.execute("insert into tracks values(?, ?)", (track_id, label))
            conn.execute("insert into raw_tracked_masks values(0, 'r1')")
            conn.execute("insert into raw_tracks values('r1')")
            conn.commit()
        finally:
            conn.close()

    def _write_head_face_sqlite(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                """
                create table head_face_detections(
                    frame integer,
                    track_id integer,
                    class_name text,
                    score real,
                    x1 real,
                    y1 real,
                    x2 real,
                    y2 real
                )
                """
            )
            conn.execute("insert into head_face_detections values(0, 1, 'Face', 0.9, 1, 2, 11, 22)")
            conn.commit()
        finally:
            conn.close()

    def test_runtime_snapshot_captures_effective_pipeline_settings(self) -> None:
        snapshot = runtime_snapshot(
            [
                "python",
                "scripts/run_integrated_pipeline.py",
                "--input",
                "input.mp4",
                "--detector",
                "eva02",
                "--postprocess",
                "--default-shape-mode",
                "polygon",
                "--intervals",
                "5",
                "--eva02-batch-size",
                "2",
            ]
        )

        self.assertEqual(snapshot["settings"]["detector"], "eva02")
        self.assertEqual(snapshot["settings"]["postprocess"], True)
        self.assertEqual(snapshot["settings"]["shape_mode"], "polygon")
        self.assertEqual(snapshot["settings"]["eva02_batch_size"], "2")

    def test_output_audit_counts_jsonl_sqlite_and_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "detector.jsonl"
            jsonl.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "detections": [
                            {
                                "class_name": "女性器",
                                "score": 0.9,
                                "bbox_xyxy": [1, 2, 11, 22],
                                "polygons": [[1, 2, 11, 2, 11, 22, 1, 22]],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            tracked = root / "tracked.sqlite"
            conn = sqlite3.connect(str(tracked))
            try:
                conn.execute("create table masks(frame integer, track_id text, polygons text, label text)")
                conn.execute("create table tracks(track_id text, label text)")
                conn.execute("create table raw_tracked_masks(frame integer, raw_track_id text)")
                conn.execute("create table raw_tracks(raw_track_id text)")
                conn.execute(
                    "insert into masks values(0, '1', ?, '女性器')",
                    (json.dumps([[1, 2, 11, 2, 11, 22, 1, 22]]),),
                )
                conn.execute("insert into tracks values('1', '女性器')")
                conn.execute("insert into raw_tracked_masks values(0, 'r1')")
                conn.execute("insert into raw_tracks values('r1')")
                conn.commit()
            finally:
                conn.close()

            overlay = root / "overlay.mp4"
            overlay.write_bytes(b"not-empty")

            audit = build_output_audit(
                detector_jsonl=jsonl,
                tracked_sqlite=tracked,
                sqlite_outputs={"女性器": str(tracked)},
                overlay_outputs={"女性器_simple": str(overlay)},
                pipeline_summary={"postprocess": {"summary": "ok"}},
            )

            self.assertEqual(audit["detector_contract"]["detections"], 1)
            self.assertEqual(audit["tracked_sqlite"]["row_counts"]["raw_tracks"], 1)
            self.assertEqual(audit["final_sqlite"]["女性器"]["row_counts"]["masks"], 1)
            self.assertEqual(audit["overlays"]["女性器_simple"]["size_bytes"], len(b"not-empty"))
            self.assertEqual(audit["warnings"], [])

    def test_run_dir_audit_reads_pipeline_summary_and_final_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            jsonl_dir = run_dir / "dinov3" / "jsonl"
            jsonl_dir.mkdir(parents=True)
            jsonl = jsonl_dir / "video.jsonl"
            jsonl.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "detections": [
                            {
                                "class_name": "女性器",
                                "score": 0.9,
                                "bbox_xyxy": [1, 2, 11, 22],
                                "polygons": [[1, 2, 11, 2, 11, 22, 1, 22]],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            sqlite_path = run_dir / "pred.sqlite"
            conn = sqlite3.connect(str(sqlite_path))
            try:
                conn.execute("create table masks(frame integer, track_id text, polygons text, label text)")
                conn.execute("insert into masks values(0, '1', '[]', '女性器')")
                conn.commit()
            finally:
                conn.close()
            overlay = run_dir / "overlay.mp4"
            overlay.write_bytes(b"x")
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "detector": "dinov3",
                        "video": "video.mp4",
                        "artifacts": {"detector_jsonl": str(jsonl)},
                        "postprocess": {"prediction_sqlite_links": {"女性器": str(sqlite_path)}},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (run_dir / "最終成果物.json").write_text(
                json.dumps(
                    {"final_sqlite": {"女性器": str(sqlite_path)}, "overlays": {"simple": str(overlay)}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            audit = audit_run_dir(run_dir)
            markdown = audit_to_markdown(audit)

            self.assertEqual(audit["output_audit"]["detector_contract"]["detections"], 1)
            self.assertIn("Run Audit", markdown)

    def test_organize_outputs_copies_user_sqlite_and_removes_debug_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            detector_jsonl = run_dir / "dinov3" / "jsonl" / "video.jsonl"
            detector_jsonl.parent.mkdir(parents=True)
            detector_jsonl.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "detections": [
                            {
                                "class_name": "女性器",
                                "score": 0.9,
                                "bbox_xyxy": [1, 2, 11, 22],
                                "polygons": [[1, 2, 11, 2, 11, 22, 1, 22]],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            raw_sqlite = run_dir / "sqlite" / "video_raw_detections.sqlite"
            tracked_sqlite = run_dir / "postprocess" / "preprocess" / "preprocess" / "video.tracked.sqlite"
            pred_sqlite = run_dir / "postprocess" / "keyframes" / "int_3" / "merged" / "predictions.sqlite"
            pred_sqlite_11 = run_dir / "postprocess" / "keyframes" / "int_11" / "merged" / "predictions.sqlite"
            combined_sqlite = run_dir / "sqlite" / "video_combined_final.sqlite"
            for path in (raw_sqlite, tracked_sqlite, pred_sqlite, combined_sqlite):
                self._write_mask_sqlite(path)
            self._write_mask_sqlite(pred_sqlite_11, frame=1, track_id="2", label="男性器")
            head_face_sqlite = run_dir / "head_face" / "sqlite" / "video_head_face.sqlite"
            self._write_head_face_sqlite(head_face_sqlite)
            (run_dir / "postprocess" / "artifacts").mkdir(parents=True)
            (run_dir / "postprocess" / "artifacts" / "debug.csv").write_text("x\n", encoding="utf-8")
            (run_dir / "postprocess_input").mkdir()
            (run_dir / "postprocess_input" / "video_postprocess_1920x1080.jsonl").write_text("{}\n", encoding="utf-8")
            (run_dir / "config").mkdir()
            (run_dir / "config" / "class_policy.ui.json").write_text("{}", encoding="utf-8")
            (run_dir / "sod_job_dir").mkdir()
            (run_dir / "index.json").write_text("{}", encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "detector": "dinov3",
                        "video": "video.mp4",
                        "head_face_enabled": True,
                        "head_face_only": False,
                        "artifacts": {
                            "detector_jsonl": str(detector_jsonl),
                            "raw_sqlite": str(raw_sqlite),
                            "head_face_sqlite": str(head_face_sqlite),
                            "combined_final_sqlite": str(combined_sqlite),
                        },
                        "raw_sqlite": {"path": str(raw_sqlite)},
                        "postprocess": {
                            "tracked_sqlite": str(tracked_sqlite),
                            "prediction_sqlite_links": {"int_3": str(pred_sqlite), "int_11": str(pred_sqlite_11)},
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            organize_outputs(
                run_dir=run_dir,
                original_input=root / "video.mp4",
                processed_input=root / "video.mp4",
                normalized=False,
                normalization_reason="none",
                keep_normalized_input=False,
                overlay_mode="none",
                raw_overlay=False,
                head_face_overlay=False,
                encoder="cpu",
                frame_limit=None,
            )

            raw_output = run_dir / "推論生SQLite" / raw_sqlite.name
            tracked_output = run_dir / "推論生SQLite" / tracked_sqlite.name
            pred_output = run_dir / "最終SQLite" / "AI後処理最終.sqlite"
            combined_output = run_dir / "最終SQLite" / "AI後処理_顔頭統合最終.sqlite"
            for path in (raw_output, tracked_output, pred_output, combined_output):
                self.assertTrue(path.is_file(), path)
                self.assertFalse(path.is_symlink(), path)
            self.assertFalse((run_dir / "最終SQLite" / "AI後処理最終_int_3.sqlite").exists())
            self.assertFalse((run_dir / "最終SQLite" / "AI後処理最終_int_11.sqlite").exists())
            conn = sqlite3.connect(str(pred_output))
            try:
                self.assertEqual(conn.execute("select count(*) from masks").fetchone()[0], 2)
                self.assertEqual(conn.execute("select count(*) from tracks").fetchone()[0], 2)
            finally:
                conn.close()

            for name in (
                "dinov3",
                "head_face",
                "postprocess",
                "postprocess_input",
                "sqlite",
                "config",
                "sod_job_dir",
                "summary.json",
                "index.json",
            ):
                self.assertFalse((run_dir / name).exists(), name)
            self.assertFalse((run_dir / "AI生成カバーオーバーレイ").exists())
            self.assertFalse((run_dir / "詳細オーバーレイ").exists())
            self.assertFalse((run_dir / "統合マスクオーバーレイ").exists())

            final_summary = json.loads((run_dir / "最終成果物.json").read_text(encoding="utf-8"))
            self.assertFalse(final_summary["debug_outputs_retained"])
            self.assertIn("postprocess", final_summary["removed_debug_outputs"])
            self.assertIn("postprocess_input", final_summary["removed_debug_outputs"])
            self.assertEqual(final_summary["raw_detector_sqlite"], str(raw_output))
            self.assertEqual(final_summary["tracked_sqlite"], str(tracked_output))
            self.assertEqual(final_summary["combined_final_sqlite"], str(combined_output))
            self.assertIsNone(final_summary["head_face_sqlite"])
            self.assertTrue((run_dir / "logs" / "pipeline_summary.json").is_file())
            self.assertTrue((run_dir / "logs" / "job_audit_summary.json").is_file())
            self.assertFalse(final_summary["output_audit"]["detector_jsonl"]["exists"])
            self.assertTrue(final_summary["output_audit"]["detector_jsonl"]["deleted_after_contract_audit"])
            self.assertFalse(final_summary["output_audit"]["head_face_sqlite"]["exists"])
            self.assertTrue(final_summary["output_audit"]["head_face_sqlite"]["deleted_after_audit"])
            cleaned_audit = audit_run_dir(run_dir)
            self.assertEqual(cleaned_audit["output_audit"]["detector_contract"]["detections"], 1)


if __name__ == "__main__":
    unittest.main()
