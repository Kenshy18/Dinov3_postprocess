from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.schemas.postprocess_coordinate_space import (
    make_letterbox_transform,
    transform_polygons_to_source,
    transform_polygons_to_work,
    write_postprocess_work_jsonl,
    write_source_space_mask_sqlite,
)
from backend.pipeline.run_integrated_pipeline import restore_postprocess_outputs_to_source_space


class PostprocessCoordinateSpaceTests(unittest.TestCase):
    def test_letterbox_transform_scales_4k_to_1080p_work_space(self) -> None:
        transform = make_letterbox_transform(3840, 2160)

        self.assertEqual(transform.work_width, 1920)
        self.assertEqual(transform.work_height, 1080)
        self.assertAlmostEqual(transform.scale, 0.5)
        self.assertEqual(transform.pad_left, 0.0)
        self.assertEqual(transform.pad_top, 0.0)

        polygons = [[1000, 1200, 1400, 1200, 1400, 1600, 1000, 1600]]
        work_polygons = transform_polygons_to_work(polygons, transform)
        self.assertEqual(work_polygons, [[500.0, 600.0, 700.0, 600.0, 700.0, 800.0, 500.0, 800.0]])
        self.assertEqual(transform_polygons_to_source(work_polygons, transform), polygons)

    def test_work_jsonl_preserves_contract_but_uses_fixed_postprocess_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "detector.jsonl"
            dst = root / "work.jsonl"
            src.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "width": 3840,
                        "height": 2160,
                        "detections": [
                            {
                                "class_name": "男性器",
                                "bbox_xyxy": [1000, 1200, 1400, 1600],
                                "bbox": [1000, 1200, 400, 400],
                                "polygons": [[1000, 1200, 1400, 1200, 1400, 1600, 1000, 1600]],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            summary = write_postprocess_work_jsonl(src, dst)

            self.assertFalse(summary["is_identity"])
            row = json.loads(dst.read_text(encoding="utf-8").strip())
            self.assertEqual(row["width"], 1920)
            self.assertEqual(row["height"], 1080)
            det = row["detections"][0]
            self.assertEqual(det["bbox_xyxy"], [500.0, 600.0, 700.0, 800.0])
            self.assertEqual(det["bbox"], [500.0, 600.0, 200.0, 200.0])
            self.assertEqual(det["polygons"], [[500.0, 600.0, 700.0, 600.0, 700.0, 800.0, 500.0, 800.0]])
            self.assertEqual(det["segmentation"], det["polygons"])

    def test_sqlite_restore_scales_masks_and_raw_audit_geometry_to_source_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_sqlite = root / "work.sqlite"
            source_sqlite = root / "source.sqlite"
            conn = sqlite3.connect(str(work_sqlite))
            try:
                conn.execute("create table masks(frame integer, track_id text, polygons text, primary key(frame, track_id))")
                conn.execute(
                    "create table raw_tracked_masks(frame integer, raw_track_id text, polygons text, bbox_xyxy_json text, bbox_json text)"
                )
                conn.execute(
                    "insert into masks values(0, '1', ?)",
                    (json.dumps([[[500.0, 600.0], [700.0, 600.0], [700.0, 800.0]]]),),
                )
                conn.execute(
                    "insert into raw_tracked_masks values(0, 'r1', ?, ?, ?)",
                    (
                        json.dumps([[500.0, 600.0, 700.0, 600.0, 700.0, 800.0]]),
                        json.dumps([500.0, 600.0, 700.0, 800.0]),
                        json.dumps([500.0, 600.0, 200.0, 200.0]),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            summary = {"source_width": 3840, "source_height": 2160, "work_width": 1920, "work_height": 1080}
            write_source_space_mask_sqlite(work_sqlite, source_sqlite, summary)

            conn = sqlite3.connect(str(source_sqlite))
            try:
                polygons = json.loads(conn.execute("select polygons from masks").fetchone()[0])
                self.assertEqual(polygons, [[[1000.0, 1200.0], [1400.0, 1200.0], [1400.0, 1600.0]]])
                raw_row = conn.execute("select polygons, bbox_xyxy_json, bbox_json from raw_tracked_masks").fetchone()
                self.assertEqual(json.loads(raw_row[0]), [[1000.0, 1200.0, 1400.0, 1200.0, 1400.0, 1600.0]])
                self.assertEqual(json.loads(raw_row[1]), [1000.0, 1200.0, 1400.0, 1600.0])
                self.assertEqual(json.loads(raw_row[2]), [1000.0, 1200.0, 400.0, 400.0])
                tables = {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}
                self.assertNotIn("coordinate_space_metadata", tables)
            finally:
                conn.close()

    def test_identity_restore_prefers_gui_compatible_tracked_sqlite_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            work_tracked = run_dir / "postprocess" / "preprocess" / "preprocess" / "video_postprocess_1920x1080.tracked.sqlite"
            linked_tracked = run_dir / "sqlite" / "video_tracked.sqlite"
            pred_link = run_dir / "sqlite" / "video_int_3_predictions.sqlite"
            for path in (work_tracked, linked_tracked, pred_link):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"sqlite placeholder")

            summary = restore_postprocess_outputs_to_source_space(
                run_dir=run_dir,
                video=Path("video.mp4"),
                postprocess_summary={
                    "tracked_sqlite": str(work_tracked),
                    "tracked_sqlite_link": str(linked_tracked),
                    "prediction_sqlite_links": {"int_3": str(pred_link)},
                },
                coordinate_summary={
                    "source_width": 1920,
                    "source_height": 1080,
                    "work_width": 1920,
                    "work_height": 1080,
                },
            )

            self.assertEqual(summary["tracked_sqlite"], str(linked_tracked))
            self.assertEqual(summary["tracked_sqlite_link"], str(linked_tracked))
            self.assertEqual(summary["work_tracked_sqlite"], str(work_tracked))
            self.assertEqual(summary["prediction_sqlite_links"], {"int_3": str(pred_link)})
            self.assertEqual(summary["source_prediction_sqlite_links"], {"int_3": str(pred_link)})
            self.assertEqual(summary["coordinate_restore_summaries"], {})


if __name__ == "__main__":
    unittest.main()
