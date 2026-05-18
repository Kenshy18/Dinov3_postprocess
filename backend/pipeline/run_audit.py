"""Audit helpers for completed detector/postprocess run directories."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.schemas.detection_jsonl import summarize_detection_jsonl


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}


def sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("select name from sqlite_master where type='table'")}


def table_count(conn: sqlite3.Connection, table: str) -> int | None:
    if table not in sqlite_tables(conn):
        return None
    return int(conn.execute(f"select count(*) from {table}").fetchone()[0])


def sqlite_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists():
        return info
    conn = sqlite3.connect(str(path))
    try:
        tables = sorted(sqlite_tables(conn))
        info["tables"] = tables
        info["row_counts"] = {
            table: count
            for table in (
                "masks",
                "tracks",
                "raw_tracked_masks",
                "raw_tracks",
                "detections",
                "head_face_detections",
                "head_face_masks",
                "head_face_tracks",
                "head_face_metadata",
            )
            if (count := table_count(conn, table)) is not None
        }
        if "masks" in tables:
            columns = sqlite_columns(conn, "masks")
            info["masks_columns"] = sorted(columns)
            if "frame" in columns:
                min_frame, max_frame = conn.execute("select min(frame), max(frame) from masks").fetchone()
                info["frame_range"] = [min_frame, max_frame]
            if "track_id" in columns:
                info["unique_tracks_in_masks"] = int(
                    conn.execute("select count(distinct track_id) from masks").fetchone()[0]
                )
            if "label" in columns:
                info["labels"] = [
                    str(row[0])
                    for row in conn.execute(
                        "select distinct label from masks where label is not null order by label"
                    )
                ]
        if "head_face_detections" in tables:
            columns = sqlite_columns(conn, "head_face_detections")
            info["head_face_detections_columns"] = sorted(columns)
            if "frame" in columns:
                min_frame, max_frame = conn.execute("select min(frame), max(frame) from head_face_detections").fetchone()
                info["head_face_frame_range"] = [min_frame, max_frame]
            if "track_id" in columns:
                info["unique_head_face_tracks"] = int(
                    conn.execute(
                        "select count(distinct track_id) from head_face_detections where track_id is not null"
                    ).fetchone()[0]
                )
            if "class_name" in columns:
                info["head_face_classes"] = [
                    str(row[0])
                    for row in conn.execute(
                        "select distinct class_name from head_face_detections where class_name is not null order by class_name"
                    )
                ]
    finally:
        conn.close()
    return info


def file_summary(path: Path) -> dict[str, Any]:
    if str(path) in {"", "."}:
        return {"path": None, "exists": False, "size_bytes": 0}
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def build_output_audit(
    *,
    detector_jsonl: Path,
    tracked_sqlite: Path | None,
    sqlite_outputs: dict[str, str],
    overlay_outputs: dict[str, str],
    pipeline_summary: dict[str, Any],
    raw_detector_sqlite: Path | None = None,
    head_face_sqlite: Path | None = None,
    combined_final_sqlite: Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    detector_contract: dict[str, Any] | None = None
    head_face_only = bool(pipeline_summary.get("head_face_only"))
    detector_jsonl_exists = str(detector_jsonl) not in {"", "."} and detector_jsonl.is_file()
    if detector_jsonl_exists:
        detector_contract = summarize_detection_jsonl(detector_jsonl).as_dict()
        if detector_contract.get("detections", 0) == 0:
            warnings.append("detector_jsonl_has_zero_detections")
        if detector_contract.get("detections_with_mask", 0) == 0:
            warnings.append("detector_jsonl_has_zero_masks")
    elif not head_face_only:
        warnings.append("detector_jsonl_missing")

    sqlite_audit = {label: sqlite_summary(Path(path)) for label, path in sorted(sqlite_outputs.items())}
    tracked_audit = sqlite_summary(tracked_sqlite)
    raw_detector_audit = sqlite_summary(raw_detector_sqlite)
    head_face_audit = sqlite_summary(head_face_sqlite)
    combined_final_audit = sqlite_summary(combined_final_sqlite)
    if pipeline_summary.get("postprocess") and not sqlite_outputs:
        warnings.append("postprocess_enabled_but_no_final_sqlite")
    if raw_detector_sqlite is not None and raw_detector_audit and not raw_detector_audit.get("exists"):
        warnings.append("raw_detector_sqlite_missing")
    if bool(pipeline_summary.get("head_face_enabled")) and head_face_sqlite is None:
        warnings.append("head_face_enabled_but_sqlite_missing")
    if head_face_sqlite is not None and head_face_audit and not head_face_audit.get("exists"):
        warnings.append("head_face_sqlite_missing")
    if combined_final_sqlite is not None and combined_final_audit and not combined_final_audit.get("exists"):
        warnings.append("combined_final_sqlite_missing")
    if tracked_audit and tracked_audit.get("exists"):
        row_counts = dict(tracked_audit.get("row_counts") or {})
        if "raw_tracked_masks" not in row_counts or "raw_tracks" not in row_counts:
            warnings.append("tracked_sqlite_missing_raw_audit_tables")

    overlay_audit = {label: file_summary(Path(path)) for label, path in sorted(overlay_outputs.items())}
    for label, info in overlay_audit.items():
        if not info["exists"] or int(info["size_bytes"]) <= 0:
            warnings.append(f"overlay_empty_or_missing:{label}")

    return {
        "detector_jsonl": file_summary(detector_jsonl),
        "detector_contract": detector_contract,
        "raw_detector_sqlite": raw_detector_audit,
        "tracked_sqlite": tracked_audit,
        "head_face_sqlite": head_face_audit,
        "combined_final_sqlite": combined_final_audit,
        "final_sqlite": sqlite_audit,
        "overlays": overlay_audit,
        "warnings": warnings,
    }


def _path_or_none(value: object) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def audit_run_dir(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    final_summary_path = run_dir / "最終成果物.json"
    final_summary = load_json(final_summary_path) if final_summary_path.is_file() else {}
    summary_candidates = [run_dir / "summary.json", run_dir / "logs" / "pipeline_summary.json"]
    if isinstance(final_summary, dict) and final_summary.get("pipeline_summary"):
        summary_candidates.insert(1, Path(str(final_summary["pipeline_summary"])))
    summary_path = next((path for path in summary_candidates if path.is_file()), None)
    if summary_path is None:
        raise FileNotFoundError(run_dir / "summary.json")
    summary = load_json(summary_path)
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    postprocess = summary.get("postprocess") or {}
    if not isinstance(postprocess, dict):
        postprocess = {}

    detector_jsonl = _path_or_none(artifacts.get("detector_jsonl") or artifacts.get("dinov3_jsonl")) or Path("")
    raw_summary_obj = summary.get("raw_sqlite") or {}
    raw_summary = raw_summary_obj if isinstance(raw_summary_obj, dict) else {}
    raw_detector_sqlite = _path_or_none(artifacts.get("raw_sqlite") or raw_summary.get("path"))
    head_face_obj = summary.get("head_face") or {}
    head_face_summary = head_face_obj if isinstance(head_face_obj, dict) else {}
    combined_obj = summary.get("combined_final_sqlite") or {}
    combined_summary = combined_obj if isinstance(combined_obj, dict) else {}
    head_face_sqlite = _path_or_none(artifacts.get("head_face_sqlite") or head_face_summary.get("path"))
    combined_final_sqlite = _path_or_none(artifacts.get("combined_final_sqlite") or combined_summary.get("path"))
    tracked_sqlite = _path_or_none(postprocess.get("tracked_sqlite") or postprocess.get("tracked_sqlite_link"))
    sqlite_outputs = postprocess.get("prediction_sqlite_links") or {}
    overlay_outputs = postprocess.get("overlay_links") or {}
    if isinstance(final_summary, dict):
        sqlite_outputs = final_summary.get("final_sqlite") or sqlite_outputs
        overlay_outputs = final_summary.get("overlays") or overlay_outputs
        head_face_sqlite = _path_or_none(final_summary.get("head_face_sqlite")) or head_face_sqlite
        combined_final_sqlite = _path_or_none(final_summary.get("combined_final_sqlite")) or combined_final_sqlite
    if not isinstance(sqlite_outputs, dict):
        sqlite_outputs = {}
    if not isinstance(overlay_outputs, dict):
        overlay_outputs = {}

    cached_output_audit = final_summary.get("output_audit") if isinstance(final_summary, dict) else None
    if isinstance(cached_output_audit, dict) and not detector_jsonl.is_file():
        output_audit = cached_output_audit
    else:
        output_audit = build_output_audit(
            detector_jsonl=detector_jsonl,
            raw_detector_sqlite=raw_detector_sqlite,
            tracked_sqlite=tracked_sqlite,
            head_face_sqlite=head_face_sqlite,
            combined_final_sqlite=combined_final_sqlite,
            sqlite_outputs={str(key): str(value) for key, value in sqlite_outputs.items()},
            overlay_outputs={str(key): str(value) for key, value in overlay_outputs.items()},
            pipeline_summary=summary,
        )
    return {
        "run_dir": str(run_dir),
        "summary": str(summary_path),
        "final_summary": str(final_summary_path) if final_summary_path.is_file() else None,
        "detector": summary.get("detector"),
        "video": summary.get("video"),
        "postprocess_enabled": summary.get("postprocess") is not None,
        "output_audit": output_audit,
    }


def audit_to_markdown(audit: dict[str, Any]) -> str:
    output = dict(audit.get("output_audit") or {})
    contract = dict(output.get("detector_contract") or {})
    tracked = output.get("tracked_sqlite") or {}
    head_face = output.get("head_face_sqlite") or {}
    warnings = list(output.get("warnings") or [])
    lines = [
        "# Run Audit",
        "",
        f"- run_dir: `{audit.get('run_dir')}`",
        f"- detector: `{audit.get('detector')}`",
        f"- video: `{audit.get('video')}`",
        f"- warnings: `{len(warnings)}`",
        "",
        "## Detector JSONL",
        "",
        f"- frames: `{contract.get('frame_records', 0)}`",
        f"- detections: `{contract.get('detections', 0)}`",
        f"- detections_with_mask: `{contract.get('detections_with_mask', 0)}`",
        "",
        "## Tracked SQLite",
        "",
        f"- path: `{tracked.get('path')}`",
        f"- row_counts: `{tracked.get('row_counts')}`",
        "",
        "## Head/Face SQLite",
        "",
        f"- path: `{head_face.get('path')}`",
        f"- row_counts: `{head_face.get('row_counts')}`",
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{item}`" for item in warnings)
    return "\n".join(lines) + "\n"
