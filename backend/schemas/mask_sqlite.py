"""Common SQLite helpers for raw and postprocessed mask artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .detection_jsonl import normalize_frame_record


RAW_SCHEMA_VERSION = 1


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _bbox_json(det: dict[str, Any]) -> str | None:
    bbox = det.get("bbox_xyxy")
    return _json_dumps(bbox) if isinstance(bbox, list) else None


def _polygons_json(det: dict[str, Any]) -> str | None:
    polygons = det.get("polygons") or det.get("segmentation")
    return _json_dumps(polygons) if isinstance(polygons, list) else None


def create_raw_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS frames(
            frame INTEGER PRIMARY KEY,
            time_sec REAL,
            width INTEGER,
            height INTEGER
        );
        CREATE TABLE IF NOT EXISTS masks(
            frame INTEGER NOT NULL,
            mask_id TEXT NOT NULL,
            detection_index INTEGER NOT NULL,
            label TEXT,
            class_name TEXT,
            category_id INTEGER,
            score REAL,
            detector_score REAL,
            class_score REAL,
            bbox_xyxy TEXT,
            polygons TEXT,
            source_json TEXT,
            PRIMARY KEY(frame, mask_id)
        );
        CREATE INDEX IF NOT EXISTS idx_raw_masks_frame ON masks(frame);
        CREATE INDEX IF NOT EXISTS idx_raw_masks_label ON masks(label);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("schema", f"raw_mask_sqlite_v{RAW_SCHEMA_VERSION}"),
    )


def jsonl_to_raw_sqlite(
    jsonl_path: Path,
    sqlite_path: Path,
    *,
    detector: str | None = None,
    video: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_every: int = 500,
) -> dict[str, Any]:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(str(sqlite_path))
    frame_count = 0
    mask_count = 0
    try:
        create_raw_schema(conn)
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("source_jsonl", str(jsonl_path)))
        if detector:
            conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("detector", detector))
        if video is not None:
            conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("video", str(video)))
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                record = normalize_frame_record(json.loads(text))
                frame = int(record["frame_index"])
                frame_count += 1
                conn.execute(
                    "INSERT OR REPLACE INTO frames(frame, time_sec, width, height) VALUES (?, ?, ?, ?)",
                    (
                        frame,
                        record.get("time_sec"),
                        record.get("width"),
                        record.get("height"),
                    ),
                )
                for det_index, det in enumerate(record["detections"]):
                    mask_id = f"{frame}:{det_index}"
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO masks(
                            frame, mask_id, detection_index, label, class_name, category_id,
                            score, detector_score, class_score, bbox_xyxy, polygons, source_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            frame,
                            mask_id,
                            det_index,
                            det.get("label") or det.get("class_name"),
                            det.get("class_name") or det.get("label"),
                            det.get("category_id"),
                            det.get("score"),
                            det.get("detector_score"),
                            det.get("class_score"),
                            _bbox_json(det),
                            _polygons_json(det),
                            _json_dumps(det),
                        ),
                    )
                    mask_count += 1
                if progress_callback is not None and frame_count % max(1, int(progress_every)) == 0:
                    progress_callback(frame_count, mask_count)
        if progress_callback is not None:
            progress_callback(frame_count, mask_count)
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("frames", str(frame_count)))
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("masks", str(mask_count)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "path": str(sqlite_path),
        "schema": f"raw_mask_sqlite_v{RAW_SCHEMA_VERSION}",
        "frames": frame_count,
        "masks": mask_count,
        "source_jsonl": str(jsonl_path),
    }


def sqlite_table_names(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}
    finally:
        conn.close()
