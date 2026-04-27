"""Small synthetic datasets for local development."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def rectangle(x: float, y: float, width: float, height: float) -> list[list[float]]:
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def moving_rectangle_polygons(frame: int) -> list[list[list[float]]]:
    x = 100.0 + frame * 5.0
    y = 120.0 + frame * 2.0
    return [rectangle(x, y, 80.0, 60.0)]


def write_sample_sqlite(path: Path, *, frames: int = 8, label: str = "sample") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE masks(
                frame INTEGER NOT NULL,
                track_id TEXT NOT NULL,
                polygons TEXT,
                shape_type TEXT,
                dilate_px INTEGER NOT NULL DEFAULT 0,
                feather_px INTEGER NOT NULL DEFAULT 0,
                mosaic_block INTEGER NOT NULL DEFAULT 0,
                mosaic_alias REAL NOT NULL DEFAULT 0,
                label TEXT,
                PRIMARY KEY(frame, track_id)
            )
            """
        )
        conn.execute("CREATE TABLE tracks(track_id TEXT PRIMARY KEY, label TEXT)")
        conn.execute("CREATE TABLE cuts(frame INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO tracks(track_id, label) VALUES ('1', ?)", (label,))
        rows = [
            (
                frame,
                "1",
                json.dumps(moving_rectangle_polygons(frame)),
                "polygon",
                0,
                0,
                0,
                0.0,
                label,
            )
            for frame in range(int(frames))
        ]
        conn.executemany(
            """
            INSERT INTO masks(
                frame, track_id, polygons, shape_type,
                dilate_px, feather_px, mosaic_block, mosaic_alias, label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return path
