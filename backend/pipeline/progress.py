"""Lightweight progress line helpers shared by CLI and GUI entrypoints."""

from __future__ import annotations

import json
import shlex
import time
from pathlib import Path


PROGRESS_PREFIX = "[phase-progress]"
BATCH_PROGRESS_PREFIX = "[batch-progress]"


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    value = max(0, int(float(seconds)))
    minutes, sec = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    if not text:
        text = "-"
    return shlex.quote(text)


def encode_progress_fields(fields: dict[str, object]) -> str:
    return " ".join(f"{key}={_format_value(value)}" for key, value in fields.items() if value is not None)


def parse_progress_line(line: str) -> dict[str, str]:
    if line.startswith(PROGRESS_PREFIX):
        text = line.removeprefix(PROGRESS_PREFIX).strip()
    elif line.startswith(BATCH_PROGRESS_PREFIX):
        text = line.removeprefix(BATCH_PROGRESS_PREFIX).strip()
    else:
        return {}
    label = None
    if ":" in text:
        possible_label, rest = text.split(":", 1)
        if possible_label and "=" not in possible_label:
            label = possible_label.strip()
            text = rest.strip()
    fields: dict[str, str] = {}
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    if label and "phase" not in fields:
        fields["phase"] = label
    return fields


def append_progress_audit(audit_path: Path | None, event: str, fields: dict[str, object]) -> None:
    if audit_path is None:
        return
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class ProgressReporter:
    """Time-gated progress printer.

    The caller supplies cheap counters only.  The reporter prints at most once
    per interval unless forced, keeping tight video loops essentially
    unaffected.
    """

    def __init__(
        self,
        phase: str,
        *,
        total: int | None = None,
        unit: str = "items",
        interval_sec: float = 5.0,
        audit_path: Path | None = None,
        static_fields: dict[str, object] | None = None,
        prefix: str = PROGRESS_PREFIX,
    ) -> None:
        self.phase = phase
        self.total = int(total) if total is not None and int(total) > 0 else None
        self.unit = unit
        self.interval_sec = max(0.5, float(interval_sec))
        self.audit_path = audit_path
        self.static_fields = dict(static_fields or {})
        self.prefix = prefix
        self.started_at = time.perf_counter()
        self.last_emit_at = 0.0
        self.last_current = -1

    def emit(
        self,
        current: int,
        *,
        force: bool = False,
        fps: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        current = max(0, int(current))
        now = time.perf_counter()
        if (
            not force
            and current != self.total
            and self.last_current >= 0
            and now - self.last_emit_at < self.interval_sec
        ):
            return
        self.last_emit_at = now
        self.last_current = current

        elapsed = max(now - self.started_at, 1e-9)
        effective_fps = float(fps) if fps is not None else current / elapsed
        percent = None
        eta_sec = None
        if self.total:
            percent = min(100.0, current / self.total * 100.0)
            if effective_fps > 0:
                eta_sec = max(0.0, (self.total - current) / effective_fps)
        fields: dict[str, object] = {
            "phase": self.phase,
            **self.static_fields,
            "current": current,
            "total": self.total,
            "unit": self.unit,
            "percent": percent,
            "fps": effective_fps if effective_fps > 0 else None,
            "eta": format_duration(eta_sec) if eta_sec is not None else None,
            "elapsed": format_duration(elapsed),
        }
        if extra:
            fields.update(extra)
        print(f"{self.prefix} {encode_progress_fields(fields)}", flush=True)
        append_progress_audit(self.audit_path, "phase_progress", fields)


def limited_total(total: int | None, limit: int | None = None) -> int | None:
    normalized_total = int(total) if total is not None and int(total) > 0 else None
    if limit is None:
        return normalized_total
    limit = max(0, int(limit))
    return min(normalized_total, limit) if normalized_total is not None else limit
