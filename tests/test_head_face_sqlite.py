from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.schemas.head_face_sqlite import enrich_head_face_sqlite, merge_ai_and_head_face_sqlite


class HeadFaceSqliteTests(unittest.TestCase):
    def test_enrich_and_merge_head_face_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head_face = root / "head_face.sqlite"
            conn = sqlite3.connect(str(head_face))
            try:
                conn.execute(
                    """
                    create table detections(
                        id integer primary key,
                        frame_index integer,
                        timestamp_sec real,
                        class_id integer,
                        class_name text,
                        score real,
                        x1 real,
                        y1 real,
                        x2 real,
                        y2 real,
                        track_id integer,
                        source text
                    )
                    """
                )
                conn.execute(
                    "insert into detections values(1, 0, 0.0, 2, 'Face', 0.9, 10, 20, 30, 50, 7, 'track')"
                )
                conn.execute(
                    "insert into detections values(2, 0, 0.0, 1, 'Head', 0.8, 5, 10, 35, 55, 8, 'track')"
                )
                conn.commit()
            finally:
                conn.close()

            summary = enrich_head_face_sqlite(head_face)
            self.assertEqual(summary["detections"], 2)
            self.assertEqual(summary["faces"], 1)
            self.assertEqual(summary["heads"], 1)
            self.assertEqual(summary["face_ellipse_masks"], 1)

            ai = root / "ai.sqlite"
            conn = sqlite3.connect(str(ai))
            try:
                conn.execute("create table masks(frame integer, track_id text, polygons text, label text)")
                conn.execute("create table tracks(track_id text, label text)")
                conn.execute(
                    "insert into masks values(0, 'a1', ?, '女性器')",
                    (json.dumps([[[1, 2], [3, 2], [3, 4], [1, 4]]]),),
                )
                conn.execute("insert into tracks values('a1', '女性器')")
                conn.commit()
            finally:
                conn.close()

            combined = root / "combined.sqlite"
            merged = merge_ai_and_head_face_sqlite(
                ai_sqlites={"女性器": str(ai)},
                head_face_sqlite=head_face,
                output_sqlite=combined,
            )

            self.assertEqual(merged["ai_mask_rows"], 1)
            self.assertEqual(merged["head_face_detections"], 2)
            self.assertEqual(merged["studio_face_masks"], 1)
            conn = sqlite3.connect(str(combined))
            try:
                mask_columns = [row[1] for row in conn.execute("pragma table_info(masks)")]
                self.assertEqual(
                    mask_columns,
                    [
                        "frame",
                        "track_id",
                        "polygons",
                        "shape_type",
                        "control_points",
                        "dilate_px",
                        "feather_px",
                        "mosaic_block",
                        "mosaic_alias",
                        "label",
                    ],
                )
                self.assertEqual(conn.execute("select count(*) from masks").fetchone()[0], 2)
                self.assertEqual(conn.execute("select count(*) from tracks").fetchone()[0], 2)
                self.assertEqual(
                    conn.execute("select track_id, label from tracks order by track_id").fetchall(),
                    [("a1", "女性器"), ("f_7", "顔")],
                )
                self.assertEqual(
                    conn.execute(
                        """
                        select count(*) from masks
                        where polygons is null
                           or trim(polygons) in ('', '[]', 'null')
                           or shape_type is null
                           or dilate_px is null
                           or feather_px is null
                           or mosaic_block is null
                           or mosaic_alias is null
                        """
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    conn.execute("select count(*) from masks where label = '顔' and track_id like 'f_%'").fetchone()[0],
                    1,
                )
                self.assertEqual(conn.execute("select count(*) from head_face_detections").fetchone()[0], 2)
                self.assertEqual(conn.execute("select count(*) from head_face_masks").fetchone()[0], 1)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
