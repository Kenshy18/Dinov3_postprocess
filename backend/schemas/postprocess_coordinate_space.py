"""Coordinate-space adapters for the fixed-size Atosyori postprocess engine."""

from __future__ import annotations

import copy
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


POSTPROCESS_WORK_WIDTH = 1920
POSTPROCESS_WORK_HEIGHT = 1080


class CoordinateSpaceError(ValueError):
    """Raised when a coordinate-space conversion cannot be applied safely."""


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    work_width: int = POSTPROCESS_WORK_WIDTH
    work_height: int = POSTPROCESS_WORK_HEIGHT
    scale: float = 1.0
    pad_left: float = 0.0
    pad_top: float = 0.0
    resized_width: float = 0.0
    resized_height: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.source_width == self.work_width
            and self.source_height == self.work_height
            and math.isclose(self.scale, 1.0)
            and math.isclose(self.pad_left, 0.0)
            and math.isclose(self.pad_top, 0.0)
        )

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_identity"] = self.is_identity
        data["mode"] = "letterbox"
        return data

    def to_work_point(self, x: float, y: float) -> tuple[float, float]:
        return (float(x) * self.scale + self.pad_left, float(y) * self.scale + self.pad_top)

    def to_source_point(self, x: float, y: float) -> tuple[float, float]:
        return ((float(x) - self.pad_left) / self.scale, (float(y) - self.pad_top) / self.scale)


def make_letterbox_transform(
    source_width: int,
    source_height: int,
    *,
    work_width: int = POSTPROCESS_WORK_WIDTH,
    work_height: int = POSTPROCESS_WORK_HEIGHT,
) -> LetterboxTransform:
    if source_width <= 0 or source_height <= 0:
        raise CoordinateSpaceError(f"invalid source size: {source_width}x{source_height}")
    if work_width <= 0 or work_height <= 0:
        raise CoordinateSpaceError(f"invalid work size: {work_width}x{work_height}")
    scale = min(float(work_width) / float(source_width), float(work_height) / float(source_height))
    resized_width = float(source_width) * scale
    resized_height = float(source_height) * scale
    return LetterboxTransform(
        source_width=int(source_width),
        source_height=int(source_height),
        work_width=int(work_width),
        work_height=int(work_height),
        scale=float(scale),
        pad_left=(float(work_width) - resized_width) * 0.5,
        pad_top=(float(work_height) - resized_height) * 0.5,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _transform_polygon_value(
    value: object,
    point_transform: Callable[[float, float], tuple[float, float]],
) -> object:
    if not isinstance(value, list):
        return value
    if not value:
        return []
    if all(_is_number(item) for item in value):
        if len(value) % 2 != 0:
            return value
        transformed: list[float] = []
        for index in range(0, len(value), 2):
            x, y = point_transform(float(value[index]), float(value[index + 1]))
            transformed.extend([x, y])
        return transformed
    return [_transform_polygon_value(item, point_transform) for item in value]


def transform_polygons_to_work(value: object, transform: LetterboxTransform) -> object:
    return _transform_polygon_value(value, transform.to_work_point)


def transform_polygons_to_source(value: object, transform: LetterboxTransform) -> object:
    return _transform_polygon_value(value, transform.to_source_point)


def _transform_bbox_xyxy(
    value: object,
    point_transform: Callable[[float, float], tuple[float, float]],
) -> object:
    if not isinstance(value, list) or len(value) < 4 or not all(_is_number(item) for item in value[:4]):
        return value
    x1, y1 = point_transform(float(value[0]), float(value[1]))
    x2, y2 = point_transform(float(value[2]), float(value[3]))
    rest = list(value[4:])
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), *rest]


def _transform_bbox_xywh(
    value: object,
    point_transform: Callable[[float, float], tuple[float, float]],
) -> object:
    if not isinstance(value, list) or len(value) < 4 or not all(_is_number(item) for item in value[:4]):
        return value
    x1, y1 = float(value[0]), float(value[1])
    x2, y2 = x1 + max(0.0, float(value[2])), y1 + max(0.0, float(value[3]))
    tx1, ty1 = point_transform(x1, y1)
    tx2, ty2 = point_transform(x2, y2)
    rest = list(value[4:])
    return [min(tx1, tx2), min(ty1, ty2), abs(tx2 - tx1), abs(ty2 - ty1), *rest]


def _transform_detection_to_work(det: dict[str, Any], transform: LetterboxTransform) -> dict[str, Any]:
    out = copy.deepcopy(det)
    if "bbox_xyxy" in out:
        out["bbox_xyxy"] = _transform_bbox_xyxy(out["bbox_xyxy"], transform.to_work_point)
    if "bbox" in out:
        out["bbox"] = _transform_bbox_xywh(out["bbox"], transform.to_work_point)
    polygons = out.get("polygons", out.get("segmentation"))
    if isinstance(polygons, list):
        transformed = transform_polygons_to_work(polygons, transform)
        out["polygons"] = transformed
        out["segmentation"] = transformed
    return out


def write_postprocess_work_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    work_width: int = POSTPROCESS_WORK_WIDTH,
    work_height: int = POSTPROCESS_WORK_HEIGHT,
) -> dict[str, Any]:
    """Write detector JSONL in Atosyori's canonical 1920x1080 coordinate space."""

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    transform: LetterboxTransform | None = None
    frame_records = 0
    detections = 0
    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise CoordinateSpaceError(f"{input_jsonl}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise CoordinateSpaceError(f"{input_jsonl}:{line_no}: frame record must be an object")
            try:
                source_width = int(record["width"])
                source_height = int(record["height"])
            except Exception as exc:
                raise CoordinateSpaceError(
                    f"{input_jsonl}:{line_no}: width and height are required for postprocess coordinate conversion"
                ) from exc

            row_transform = make_letterbox_transform(
                source_width,
                source_height,
                work_width=work_width,
                work_height=work_height,
            )
            if transform is None:
                transform = row_transform
            elif (transform.source_width, transform.source_height) != (row_transform.source_width, row_transform.source_height):
                raise CoordinateSpaceError(
                    "postprocess coordinate conversion requires constant frame dimensions; "
                    f"first={transform.source_width}x{transform.source_height}, "
                    f"line {line_no}={row_transform.source_width}x{row_transform.source_height}"
                )

            out = copy.deepcopy(record)
            out["width"] = int(work_width)
            out["height"] = int(work_height)

            raw_detections = out.get("detections")
            raw_instances = out.get("instances")
            if isinstance(raw_detections, list):
                converted = [_transform_detection_to_work(det, row_transform) if isinstance(det, dict) else det for det in raw_detections]
                out["detections"] = converted
                if raw_instances is raw_detections or "instances" not in out:
                    out["instances"] = converted
            if isinstance(raw_instances, list) and raw_instances is not raw_detections:
                out["instances"] = [
                    _transform_detection_to_work(det, row_transform) if isinstance(det, dict) else det for det in raw_instances
                ]

            if isinstance(out.get("detections"), list):
                detections += len(out["detections"])
            elif isinstance(out.get("instances"), list):
                detections += len(out["instances"])
            frame_records += 1
            dst.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")

    if transform is None:
        raise CoordinateSpaceError(f"{input_jsonl}: no frame records found")
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "frame_records": frame_records,
        "detections": detections,
        "transform": transform.as_dict(),
        "source_width": transform.source_width,
        "source_height": transform.source_height,
        "work_width": transform.work_width,
        "work_height": transform.work_height,
        "is_identity": transform.is_identity,
        "strategy": "source_to_1920x1080_letterbox_for_atosyori_then_restore_outputs",
    }


def transform_from_summary(summary: dict[str, Any]) -> LetterboxTransform:
    raw = summary.get("transform") if isinstance(summary.get("transform"), dict) else summary
    return make_letterbox_transform(
        int(raw["source_width"]),
        int(raw["source_height"]),
        work_width=int(raw.get("work_width", POSTPROCESS_WORK_WIDTH)),
        work_height=int(raw.get("work_height", POSTPROCESS_WORK_HEIGHT)),
    )


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _json_transform(raw: object, transformer: Callable[[object], object]) -> object:
    if raw is None:
        return raw
    try:
        value = json.loads(str(raw))
    except Exception:
        return raw
    transformed = transformer(value)
    return json.dumps(transformed, ensure_ascii=False, separators=(",", ":"))


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})")]


def _transform_table_to_source(conn: sqlite3.Connection, table: str, transform: LetterboxTransform) -> dict[str, int]:
    columns = _table_columns(conn, table)
    polygon_columns = [name for name in ("polygons", "mask_polygons", "ellipse_polygon", "control_points") if name in columns]
    xyxy_columns = [name for name in ("bbox_xyxy", "bbox_xyxy_json") if name in columns]
    xywh_columns = [name for name in ("bbox", "bbox_json") if name in columns]
    target_columns = polygon_columns + xyxy_columns + xywh_columns
    if not target_columns:
        return {"rows": 0, "cells": 0}

    quoted_targets = ", ".join(_quote_identifier(name) for name in target_columns)
    quoted_table = _quote_identifier(table)
    rows = list(conn.execute(f"SELECT rowid, {quoted_targets} FROM {quoted_table}"))
    changed_rows = 0
    changed_cells = 0
    for row in rows:
        rowid = row[0]
        updates: dict[str, object] = {}
        values = list(row[1:])
        for column, raw in zip(target_columns, values):
            if column in polygon_columns:
                converted = _json_transform(raw, lambda value: transform_polygons_to_source(value, transform))
            elif column in xyxy_columns:
                converted = _json_transform(raw, lambda value: _transform_bbox_xyxy(value, transform.to_source_point))
            else:
                converted = _json_transform(raw, lambda value: _transform_bbox_xywh(value, transform.to_source_point))
            if converted != raw:
                updates[column] = converted
        if not updates:
            continue
        assignments = ", ".join(f"{_quote_identifier(name)} = ?" for name in updates)
        conn.execute(f"UPDATE {quoted_table} SET {assignments} WHERE rowid = ?", [*updates.values(), rowid])
        changed_rows += 1
        changed_cells += len(updates)
    return {"rows": changed_rows, "cells": changed_cells}


def write_source_space_mask_sqlite(
    input_sqlite: Path,
    output_sqlite: Path,
    transform_summary: dict[str, Any] | LetterboxTransform,
) -> dict[str, Any]:
    """Copy a postprocess SQLite DB and restore mask geometry to source-video coordinates."""

    transform = transform_summary if isinstance(transform_summary, LetterboxTransform) else transform_from_summary(transform_summary)
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    if output_sqlite.exists() or output_sqlite.is_symlink():
        output_sqlite.unlink()

    src = sqlite3.connect(str(input_sqlite))
    dst = sqlite3.connect(str(output_sqlite))
    table_updates: dict[str, dict[str, int]] = {}
    try:
        src.backup(dst)
        tables = [
            str(row[0])
            for row in dst.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        ]
        for table in tables:
            try:
                updates = _transform_table_to_source(dst, table, transform)
            except sqlite3.OperationalError:
                continue
            if updates["rows"] or updates["cells"]:
                table_updates[table] = updates
        dst.commit()
    finally:
        src.close()
        dst.close()

    return {
        "input_sqlite": str(input_sqlite),
        "output_sqlite": str(output_sqlite),
        "transform": transform.as_dict(),
        "table_updates": table_updates,
    }
