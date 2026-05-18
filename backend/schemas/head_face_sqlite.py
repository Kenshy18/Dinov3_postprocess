"""SQLite helpers for optional RT-DETR Head/Face sidecar outputs."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any


HEAD_FACE_SCHEMA_VERSION = 1


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _safe_track_part(value: object) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "track"


def ellipse_polygon_from_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    points: int = 48,
) -> list[list[float]]:
    """Create an inscribed ellipse polygon for a detection bbox."""

    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    if width <= 0.0 or height <= 0.0:
        return []
    cx = left + width * 0.5
    cy = top + height * 0.5
    rx = width * 0.5
    ry = height * 0.5
    count = max(12, int(points))
    return [
        [cx + math.cos(theta) * rx, cy + math.sin(theta) * ry]
        for theta in (2.0 * math.pi * index / count for index in range(count))
    ]


def _is_face(class_name: object) -> bool:
    return str(class_name).strip().lower() == "face"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def enrich_head_face_sqlite(sqlite_path: Path, *, source: str = "rtdetr") -> dict[str, Any]:
    """Add normalized Head/Face tables and generated Face ellipse masks.

    The RT-DETR runtime writes bbox rows to ``detections``.  This helper keeps
    those rows intact and adds stable tables consumed by combined SQLite and
    overlay generation.
    """

    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)
    conn = _connect(sqlite_path)
    detection_count = 0
    face_count = 0
    head_count = 0
    mask_count = 0
    track_ids: set[str] = set()
    try:
        tables = {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}
        if "detections" not in tables:
            raise RuntimeError(f"Head/Face sqlite does not contain detections table: {sqlite_path}")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS head_face_detections(
                detection_id INTEGER PRIMARY KEY,
                frame INTEGER NOT NULL,
                time_sec REAL,
                class_id INTEGER,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                bbox_xyxy TEXT NOT NULL,
                track_id TEXT,
                source TEXT NOT NULL,
                ellipse_polygon TEXT,
                mask_polygons TEXT
            );
            CREATE TABLE IF NOT EXISTS head_face_masks(
                frame INTEGER NOT NULL,
                mask_id TEXT NOT NULL,
                detection_id INTEGER NOT NULL,
                track_id TEXT,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                polygons TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY(frame, mask_id)
            );
            CREATE TABLE IF NOT EXISTS head_face_tracks(
                track_id TEXT PRIMARY KEY,
                class_name TEXT NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                rows INTEGER NOT NULL,
                avg_score REAL
            );
            CREATE TABLE IF NOT EXISTS head_face_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_head_face_detections_frame ON head_face_detections(frame);
            CREATE INDEX IF NOT EXISTS idx_head_face_detections_track ON head_face_detections(track_id);
            CREATE INDEX IF NOT EXISTS idx_head_face_masks_frame ON head_face_masks(frame);
            CREATE INDEX IF NOT EXISTS idx_head_face_masks_track ON head_face_masks(track_id);
            """
        )
        conn.execute("DELETE FROM head_face_detections")
        conn.execute("DELETE FROM head_face_masks")
        conn.execute("DELETE FROM head_face_tracks")
        conn.execute("DELETE FROM head_face_metadata")
        conn.execute(
            "INSERT OR REPLACE INTO head_face_metadata(key, value) VALUES (?, ?)",
            ("schema", f"head_face_sqlite_v{HEAD_FACE_SCHEMA_VERSION}"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO head_face_metadata(key, value) VALUES (?, ?)",
            ("source", source),
        )

        detection_rows: list[tuple[object, ...]] = []
        mask_rows: list[tuple[object, ...]] = []
        query = """
            SELECT id, frame_index, timestamp_sec, class_id, class_name, score,
                   x1, y1, x2, y2, track_id, source
            FROM detections
            ORDER BY frame_index, id
        """
        for row in conn.execute(query):
            detection_id = int(row["id"])
            frame = int(row["frame_index"])
            class_name = str(row["class_name"])
            bbox = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
            track_id = None if row["track_id"] is None else str(row["track_id"])
            if track_id:
                track_ids.add(track_id)
            ellipse = ellipse_polygon_from_bbox(*bbox) if _is_face(class_name) else []
            ellipse_json = _json_dumps([ellipse]) if ellipse else None
            detection_rows.append(
                (
                    detection_id,
                    frame,
                    row["timestamp_sec"],
                    row["class_id"],
                    class_name,
                    float(row["score"]),
                    *bbox,
                    _json_dumps(bbox),
                    track_id,
                    str(row["source"] or source),
                    ellipse_json,
                    ellipse_json,
                )
            )
            detection_count += 1
            if _is_face(class_name):
                face_count += 1
            elif class_name.strip().lower() == "head":
                head_count += 1
            if ellipse:
                mask_id = f"face:{track_id or detection_id}:{detection_id}"
                mask_rows.append(
                    (
                        frame,
                        mask_id,
                        detection_id,
                        track_id,
                        class_name,
                        float(row["score"]),
                        _json_dumps([ellipse]),
                        "face_bbox_ellipse",
                    )
                )
                mask_count += 1

        if detection_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO head_face_detections(
                    detection_id, frame, time_sec, class_id, class_name, score,
                    x1, y1, x2, y2, bbox_xyxy, track_id, source,
                    ellipse_polygon, mask_polygons
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                detection_rows,
            )
        if mask_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO head_face_masks(
                    frame, mask_id, detection_id, track_id, class_name, score, polygons, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                mask_rows,
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO head_face_tracks(track_id, class_name, first_frame, last_frame, rows, avg_score)
            SELECT
                track_id,
                class_name,
                min(frame),
                max(frame),
                count(*),
                avg(score)
            FROM head_face_detections
            WHERE track_id IS NOT NULL
            GROUP BY track_id, class_name
            """
        )
        for key, value in {
            "detections": detection_count,
            "heads": head_count,
            "faces": face_count,
            "face_ellipse_masks": mask_count,
            "tracks": len(track_ids),
        }.items():
            conn.execute("INSERT OR REPLACE INTO head_face_metadata(key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {
        "path": str(sqlite_path),
        "schema": f"head_face_sqlite_v{HEAD_FACE_SCHEMA_VERSION}",
        "detections": detection_count,
        "heads": head_count,
        "faces": face_count,
        "face_ellipse_masks": mask_count,
        "tracks": len(track_ids),
    }


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"pragma table_info({table})")]


def merge_ai_and_head_face_sqlite(
    *,
    ai_sqlites: dict[str, str],
    head_face_sqlite: Path,
    output_sqlite: Path,
) -> dict[str, Any]:
    """Create a combined final SQLite with AI masks plus Head/Face tables."""

    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists() or output_sqlite.is_symlink():
        output_sqlite.unlink()

    conn = sqlite3.connect(str(output_sqlite))
    ai_mask_rows = 0
    ai_track_rows = 0
    head_face_rows = 0
    face_mask_rows = 0
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT,
                label TEXT,
                source_label TEXT,
                source_track_id TEXT,
                source_sqlite TEXT,
                source_json TEXT,
                PRIMARY KEY(frame, track_id)
            );
            CREATE TABLE tracks(
                track_id TEXT PRIMARY KEY,
                label TEXT,
                source_label TEXT,
                source_track_id TEXT,
                source_sqlite TEXT
            );
            CREATE TABLE cuts(frame INTEGER PRIMARY KEY);
            CREATE TABLE head_face_detections(
                detection_id INTEGER PRIMARY KEY,
                frame INTEGER NOT NULL,
                time_sec REAL,
                class_id INTEGER,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                bbox_xyxy TEXT NOT NULL,
                track_id TEXT,
                source TEXT NOT NULL,
                ellipse_polygon TEXT,
                mask_polygons TEXT
            );
            CREATE TABLE head_face_masks(
                frame INTEGER NOT NULL,
                mask_id TEXT NOT NULL,
                detection_id INTEGER NOT NULL,
                track_id TEXT,
                class_name TEXT NOT NULL,
                score REAL NOT NULL,
                polygons TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY(frame, mask_id)
            );
            CREATE TABLE head_face_tracks(
                track_id TEXT PRIMARY KEY,
                class_name TEXT NOT NULL,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                rows INTEGER NOT NULL,
                avg_score REAL
            );
            CREATE TABLE head_face_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX idx_combined_masks_frame ON masks(frame);
            CREATE INDEX idx_combined_head_face_detections_frame ON head_face_detections(frame);
            CREATE INDEX idx_combined_head_face_masks_frame ON head_face_masks(frame);
            """
        )
        conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("schema", "combined_ai_head_face_sqlite_v1"))

        for source_label, raw_path in sorted(ai_sqlites.items()):
            source = Path(str(raw_path))
            if not source.is_file():
                continue
            src = _connect(source)
            try:
                tables = _sqlite_tables(src)
                if "masks" not in tables:
                    continue
                columns = _sqlite_columns(src, "masks")
                select_cols = ", ".join(f'"{name}"' for name in columns)
                for row in src.execute(f"SELECT {select_cols} FROM masks"):
                    row_dict = {name: row[name] for name in columns}
                    frame = int(row_dict.get("frame") or 0)
                    source_track_id = str(row_dict.get("track_id") or f"row_{ai_mask_rows}")
                    label = str(row_dict.get("label") or source_label)
                    track_id = f"ai:{_safe_track_part(source_label)}:{_safe_track_part(source_track_id)}"
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO masks(
                            frame, track_id, polygons, label, source_label,
                            source_track_id, source_sqlite, source_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            frame,
                            track_id,
                            row_dict.get("polygons"),
                            label,
                            source_label,
                            source_track_id,
                            str(source),
                            _json_dumps(row_dict),
                        ),
                    )
                    ai_mask_rows += 1
                if "tracks" in tables:
                    track_columns = _sqlite_columns(src, "tracks")
                    if "track_id" in track_columns:
                        label_column = "label" if "label" in track_columns else "NULL as label"
                        for track_id_raw, label_raw in src.execute(f"SELECT track_id, {label_column} FROM tracks"):
                            source_track_id = str(track_id_raw)
                            track_id = f"ai:{_safe_track_part(source_label)}:{_safe_track_part(source_track_id)}"
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO tracks(
                                    track_id, label, source_label, source_track_id, source_sqlite
                                )
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (track_id, str(label_raw or source_label), source_label, source_track_id, str(source)),
                            )
                            ai_track_rows += 1
                if "cuts" in tables and "frame" in _sqlite_columns(src, "cuts"):
                    for (frame,) in src.execute("SELECT frame FROM cuts"):
                        conn.execute("INSERT OR IGNORE INTO cuts(frame) VALUES (?)", (int(frame),))
            finally:
                src.close()

        if head_face_sqlite.is_file():
            src = _connect(head_face_sqlite)
            try:
                tables = _sqlite_tables(src)
                for table in ("head_face_detections", "head_face_masks", "head_face_tracks", "head_face_metadata"):
                    if table not in tables:
                        continue
                    columns = _sqlite_columns(src, table)
                    select_cols = ", ".join(f'"{name}"' for name in columns)
                    placeholders = ", ".join("?" for _ in columns)
                    insert_cols = ", ".join(f'"{name}"' for name in columns)
                    rows = [tuple(row[name] for name in columns) for row in src.execute(f"SELECT {select_cols} FROM {table}")]
                    if rows:
                        conn.executemany(f"INSERT OR REPLACE INTO {table}({insert_cols}) VALUES ({placeholders})", rows)
                    if table == "head_face_detections":
                        head_face_rows += len(rows)
                    elif table == "head_face_masks":
                        face_mask_rows += len(rows)
            finally:
                src.close()

        for key, value in {
            "ai_mask_rows": ai_mask_rows,
            "ai_track_rows": ai_track_rows,
            "head_face_detections": head_face_rows,
            "head_face_masks": face_mask_rows,
            "head_face_sqlite": str(head_face_sqlite),
        }.items():
            conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
    except Exception:
        conn.rollback()
        if output_sqlite.exists():
            output_sqlite.unlink()
        raise
    finally:
        conn.close()

    return {
        "path": str(output_sqlite),
        "schema": "combined_ai_head_face_sqlite_v1",
        "ai_mask_rows": ai_mask_rows,
        "ai_track_rows": ai_track_rows,
        "head_face_detections": head_face_rows,
        "head_face_masks": face_mask_rows,
    }
