#!/usr/bin/env python3
"""Shared helpers for user-facing flow entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]

BUILTIN_CLASS_LABELS = ("男性器", "女性器", "結合部分", "結合")
CLASS_ALIASES = {
    "male": ("男性器",),
    "penis": ("男性器",),
    "男性器": ("男性器",),
    "female": ("女性器",),
    "vulva": ("女性器",),
    "女性器": ("女性器",),
    "junction": ("結合部分", "結合"),
    "join": ("結合部分", "結合"),
    "coupling": ("結合部分", "結合"),
    "結合部分": ("結合部分",),
    "結合": ("結合",),
}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def strip_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def validate_interval(value: int) -> int:
    if int(value) <= 0:
        raise argparse.ArgumentTypeError("keyframe interval must be a positive integer")
    return int(value)


def validate_recall(value: float) -> float:
    recall = float(value)
    if not 0.0 < recall <= 1.0:
        raise argparse.ArgumentTypeError("recall target must be in the range (0, 1]")
    return recall


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shape-mode",
        choices=("ellipse", "polygon"),
        default="ellipse",
        help="Default approximation mode for all classes unless overridden.",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=validate_interval,
        default=3,
        help="Default target keyframe interval. 3 means roughly one keyframe every 3 frames.",
    )
    parser.add_argument(
        "--recall-target",
        type=validate_recall,
        default=0.96,
        help="Default recall constraint. Used as ellipse dense recall and polygon recall floor.",
    )
    parser.add_argument(
        "--class-policy",
        action="append",
        default=[],
        metavar="CLASS:MODE[:INTERVAL[:RECALL]]",
        help=(
            "Override one class. CLASS can be female, male, junction, or an exact label. "
            "Example: --class-policy female:polygon:5:0.97"
        ),
    )
    parser.add_argument(
        "--class-policy-json",
        type=Path,
        default=None,
        help="Use an existing class policy JSON instead of generating one from the simple options.",
    )


def normalize_class_labels(name: str) -> tuple[str, ...]:
    key = name.strip()
    return CLASS_ALIASES.get(key, (key,))


def policy_entry(shape_mode: str, interval: int, recall: float) -> dict[str, Any]:
    return {
        "shape_mode": shape_mode,
        "target_interval": int(interval),
        "dense_recall_target": float(recall),
        "polygon_recall_min": float(recall),
    }


def parse_class_policy_override(
    value: str,
    *,
    default_interval: int,
    default_recall: float,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    text = value.strip()
    if not text:
        raise ValueError("empty --class-policy value")
    text = text.replace("=", ":", 1)
    parts = [part.strip() for part in text.split(":")]
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(
            "--class-policy must be CLASS:MODE[:INTERVAL[:RECALL]], "
            f"got: {value!r}"
        )

    class_name, shape_mode = parts[0], parts[1]
    if shape_mode not in {"ellipse", "polygon"}:
        raise ValueError(f"invalid shape mode in --class-policy {value!r}: {shape_mode!r}")

    interval = default_interval if len(parts) < 3 or parts[2] == "" else validate_interval(int(parts[2]))
    recall = default_recall if len(parts) < 4 or parts[3] == "" else validate_recall(float(parts[3]))
    return normalize_class_labels(class_name), policy_entry(shape_mode, interval, recall)


def build_policy_dict(
    *,
    shape_mode: str,
    keyframe_interval: int,
    recall_target: float,
    class_policy: list[str],
) -> dict[str, Any]:
    default = policy_entry(shape_mode, keyframe_interval, recall_target)
    classes = {label: dict(default) for label in BUILTIN_CLASS_LABELS}
    for override in class_policy:
        labels, entry = parse_class_policy_override(
            override,
            default_interval=keyframe_interval,
            default_recall=recall_target,
        )
        for label in labels:
            classes[label] = dict(entry)
    return {"default": default, "classes": classes}


def resolve_policy_path(args: argparse.Namespace, config_dir: Path) -> Path:
    if args.class_policy_json is not None:
        if args.class_policy:
            raise ValueError("--class-policy-json cannot be combined with --class-policy")
        return Path(args.class_policy_json).expanduser().resolve()

    policy = build_policy_dict(
        shape_mode=args.shape_mode,
        keyframe_interval=args.keyframe_interval,
        recall_target=args.recall_target,
        class_policy=list(args.class_policy or []),
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "class_policy.generated.json"
    path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def command_to_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"[cwd] {cwd or Path.cwd()}\n")
        log.write("[cmd] " + command_to_text(command) + "\n")
        log.flush()

        print(f"[log] {log_path}", flush=True)
        print("[cmd] " + command_to_text(command), flush=True)
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        returncode = process.wait()
        elapsed = time.perf_counter() - start

        status = "done" if returncode == 0 else "error"
        footer = f"[{status}] returncode={returncode} elapsed={elapsed:.2f}s\n"
        print(footer.rstrip(), flush=True)
        log.write(footer)
        return returncode, elapsed


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
