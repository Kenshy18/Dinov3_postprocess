from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
import json
import sqlite3
import sys
from pathlib import Path
union2sqlite_ROOT = Path(__file__).resolve().parents[1]
if str(union2sqlite_ROOT) not in sys.path:
    sys.path.insert(0, str(union2sqlite_ROOT))
import standalone_runtime_fst as fst

def union2sqlite_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert interpolated_union.json into renderer-friendly prediction SQLite.')
    parser.add_argument('--input-union-json', required=True)
    parser.add_argument('--output-sqlite', required=True)
    parser.add_argument('--reference-sqlite', default=None)
    return parser.parse_args()

def union2sqlite_main() -> None:
    args = union2sqlite_parse_args()
    input_union = Path(args.input_union_json)
    output_sqlite = Path(args.output_sqlite)
    output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(input_union.read_text(encoding='utf-8'))
    sqlite_rows: list[tuple[int, str, str]] = []
    for row in rows:
        polygons_json = None
        ellipse_params = row.get('ellipse_params')
        polygon_points = row.get('polygon')
        if ellipse_params:
            ellipses = [tuple(map(float, ellipse)) for ellipse in ellipse_params]
            polygons_json = fst.make_polygons_json(ellipses)
        elif polygon_points:
            polygon = [[float(point[0]), float(point[1])] for point in polygon_points]
            polygons_json = json.dumps([polygon], ensure_ascii=False, separators=(',', ':'))
        if not polygons_json:
            continue
        sqlite_rows.append((int(row['frame']), str(row['track_id']), polygons_json))
    ref_sqlite = None if not args.reference_sqlite else Path(args.reference_sqlite)
    fst.write_sqlite(sqlite_rows, output_sqlite, reference_sqlite=ref_sqlite)
