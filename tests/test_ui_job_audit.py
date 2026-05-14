from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from apps.qt_ui.run_ui_job import build_output_audit, runtime_snapshot


class UiJobAuditTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
