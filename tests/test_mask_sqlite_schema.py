from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.schemas.mask_sqlite import jsonl_to_raw_sqlite


class MaskSqliteSchemaTests(unittest.TestCase):
    def test_detector_jsonl_converts_to_common_raw_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "detector.jsonl"
            jsonl.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "width": 1280,
                        "height": 720,
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
            sqlite_path = root / "raw.sqlite"

            summary = jsonl_to_raw_sqlite(jsonl, sqlite_path, detector="dinov3", video=root / "video.mp4")

            self.assertEqual(summary["schema"], "raw_mask_sqlite_v1")
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["masks"], 1)
            conn = sqlite3.connect(str(sqlite_path))
            try:
                tables = {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}
                self.assertTrue({"metadata", "frames", "masks"}.issubset(tables))
                row = conn.execute("select frame, mask_id, label, score, bbox_xyxy, polygons from masks").fetchone()
                self.assertEqual(row[0], 0)
                self.assertEqual(row[1], "0:0")
                self.assertEqual(row[2], "女性器")
                self.assertAlmostEqual(float(row[3]), 0.9)
                self.assertEqual(json.loads(row[4]), [1, 2, 11, 22])
                self.assertEqual(json.loads(row[5]), [[1, 2, 11, 2, 11, 22, 1, 22]])
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
