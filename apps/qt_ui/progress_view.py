from __future__ import annotations

import time

from PyQt5 import QtCore, QtWidgets


PHASE_LABELS = {
    "normalize_input": "入力正規化",
    "inference": "推論",
    "detector": "推論",
    "raw_sqlite": "推論結果SQLite作成",
    "postprocess": "後処理",
    "raw_overlay": "後処理前オーバーレイ",
    "detailed_overlay": "後処理後オーバーレイ（詳細）",
    "simple_overlay": "後処理後オーバーレイ（簡易）",
    "command": "コマンド",
    "unknown": "処理中",
}


def progress_float(fields: dict[str, str], key: str) -> float | None:
    value = fields.get(key)
    if value in (None, "", "-"):
        return None
    value = value.rstrip("%")
    try:
        return float(value)
    except ValueError:
        return None


def phase_key(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "unknown"
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if "normalize" in normalized:
        return "normalize_input"
    if "detailed_overlay" in normalized or ("overlay" in normalized and "detailed" in normalized):
        return "detailed_overlay"
    if "simple_overlay" in normalized or ("overlay" in normalized and "simple" in normalized):
        return "simple_overlay"
    if "raw_overlay" in normalized or ("overlay" in normalized and "raw" in normalized):
        return "raw_overlay"
    if "raw_sqlite" in normalized or ("sqlite" in normalized and "raw" in normalized):
        return "raw_sqlite"
    if "postprocess" in normalized or "atosyori" in normalized:
        return "postprocess"
    if (
        "inference" in normalized
        or "infer" in normalized
        or "detect" in normalized
        or "detector" in normalized
        or "codino" in normalized
        or "dinov3" in normalized
        or "eva02" in normalized
    ):
        return "inference"
    if "command" in normalized:
        return "command"
    return normalized


def phase_label(value: str | None) -> str:
    key = phase_key(value)
    return PHASE_LABELS.get(key, (value or "-").strip() or "-")


def progress_percent(fields: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = progress_float(fields, key)
        if value is not None:
            return max(0.0, min(100.0, value))
    return None


def progress_percent_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def progress_status_text(fields: dict[str, str]) -> str:
    raw_phase = fields.get("phase", "-")
    phase = fields.get("stage") or phase_label(raw_phase)
    raw_phase_key = phase_key(raw_phase)
    current = fields.get("current")
    total = fields.get("total")
    unit = fields.get("unit", "")
    fps = fields.get("fps")
    compute_fps = fields.get("compute_fps")
    eta = fields.get("eta")
    overall_eta = fields.get("overall_eta")
    phase_percent = progress_percent(fields, "percent", "phase_percent")
    overall = progress_percent(fields, "overall", "overall_percent")
    pieces = [phase]
    if raw_phase and raw_phase != "-" and raw_phase_key not in {raw_phase, phase}:
        pieces.append(f"内部={raw_phase}")
    if fields.get("item"):
        pieces.append(f"動画={fields['item']}")
    if overall is not None:
        pieces.append(f"全体={progress_percent_text(overall)}")
    if phase_percent is not None:
        pieces.append(f"フェーズ={progress_percent_text(phase_percent)}")
    if current and total:
        pieces.append(f"{current}/{total} {unit}".strip())
    elif current:
        pieces.append(f"{current} {unit}".strip())
    if fps:
        pieces.append(f"fps={fps}")
    if compute_fps:
        pieces.append(f"compute={compute_fps}")
    if overall_eta:
        pieces.append(f"残り={overall_eta}")
    if eta:
        pieces.append(f"フェーズ残り={eta}")
    if fields.get("detail"):
        pieces.append(fields["detail"])
    return " | ".join(pieces)


class ProgressDashboard(QtWidgets.QGroupBox):
    """Compact progress panel for CLI progress lines.

    The widget intentionally exposes the legacy label names used by app.py so
    the window can delegate progress rendering without rewriting job control.
    """

    EXPORTS = (
        "status_banner",
        "total_progress",
        "phase_progress",
        "current_video_value",
        "speed_value",
        "detail_info_value",
        "elapsed_value",
        "process_value",
        "stage_value",
        "frame_value",
        "phase_value",
        "phase_elapsed_value",
        "status_value_label",
        "count_value",
        "heartbeat_value",
        "remaining_value",
        "overall_remaining_value",
        "output_value",
        "command_value",
        "summary_text",
    )

    def __init__(self) -> None:
        super().__init__("実行状況")
        self._build()

    def _build(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 7)
        layout.setSpacing(6)
        self.status_banner = QtWidgets.QLabel("待機中")
        self.status_banner.setObjectName("statusBanner")
        self.status_banner.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_banner)

        progress_grid = QtWidgets.QGridLayout()
        progress_grid.setHorizontalSpacing(8)
        progress_grid.setVerticalSpacing(4)
        self.total_progress = QtWidgets.QProgressBar()
        self.total_progress.setRange(0, 1)
        self.total_progress.setValue(0)
        self.phase_progress = QtWidgets.QProgressBar()
        self.phase_progress.setRange(0, 1)
        self.phase_progress.setValue(0)
        progress_grid.addWidget(QtWidgets.QLabel("全体"), 0, 0)
        progress_grid.addWidget(self.total_progress, 0, 1)
        progress_grid.addWidget(QtWidgets.QLabel("フェーズ"), 1, 0)
        progress_grid.addWidget(self.phase_progress, 1, 1)
        layout.addLayout(progress_grid)

        metrics = QtWidgets.QGridLayout()
        metrics.setHorizontalSpacing(6)
        metrics.setVerticalSpacing(6)
        self.phase_value = self.metric_value("待機")
        self.frame_value = self.metric_value("-")
        self.speed_value = self.metric_value("-")
        self.overall_remaining_value = self.metric_value("-")
        metric_items = [
            ("フェーズ", self.phase_value),
            ("フレーム", self.frame_value),
            ("FPS", self.speed_value),
            ("残り", self.overall_remaining_value),
        ]
        for col, (label, widget) in enumerate(metric_items):
            card = QtWidgets.QFrame()
            card.setObjectName("metricCard")
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(7, 5, 7, 5)
            card_layout.setSpacing(2)
            key = QtWidgets.QLabel(label)
            key.setObjectName("metricKey")
            card_layout.addWidget(key)
            card_layout.addWidget(widget)
            metrics.addWidget(card, 0, col)
        metrics.setColumnStretch(0, 1)
        metrics.setColumnStretch(1, 1)
        metrics.setColumnStretch(2, 1)
        metrics.setColumnStretch(3, 1)
        layout.addLayout(metrics)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(0)
        form.setVerticalSpacing(0)
        self.current_video_value = self.status_value("-")
        self.detail_info_value = self.status_value("-")
        self.elapsed_value = self.status_value("-")
        self.process_value = self.status_value("NotRunning")
        self.stage_value = self.status_value("-")
        self.phase_elapsed_value = self.status_value("-")
        self.status_value_label = self.status_value("-")
        self.count_value = self.status_value("Inference 0 / 0 | Postprocess 0 / 0")
        self.heartbeat_value = self.status_value("-")
        self.remaining_value = self.status_value("-")
        self.output_value = self.status_value("-")
        self.command_value = self.status_value("-")

        rows: list[tuple[str, QtWidgets.QWidget, str, QtWidgets.QWidget]] = [
            ("動画", self.current_video_value, "経過", self.elapsed_value),
            ("詳細", self.detail_info_value, "状態", self.status_value_label),
            ("プロセス", self.process_value, "動画数", self.count_value),
            ("ステージ", self.stage_value, "更新", self.heartbeat_value),
            ("フェーズ経過", self.phase_elapsed_value, "残本数", self.remaining_value),
            ("実行Dir", self.output_value, "", self.status_value("")),
        ]
        for row, (left_label, left_widget, right_label, right_widget) in enumerate(rows):
            form.addWidget(self.status_key(left_label), row, 0)
            form.addWidget(left_widget, row, 1)
            form.addWidget(self.status_key(right_label), row, 2)
            form.addWidget(right_widget, row, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 2)
        layout.addLayout(form)

        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(40)
        self.summary_text.setPlainText("status: idle")
        layout.addWidget(self.summary_text)

    def status_key(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("statusKey")
        label.setMinimumWidth(44)
        return label

    def status_value(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("statusValue")
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        label.setMinimumWidth(60)
        label.setWordWrap(False)
        label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        return label

    def metric_value(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("metricValue")
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        return label

    def bind_legacy_attributes(self, owner: object) -> None:
        for name in self.EXPORTS:
            setattr(owner, name, getattr(self, name))

    def set_phase_indeterminate(self) -> None:
        self.phase_progress.setRange(0, 0)
        self.phase_progress.setValue(0)

    def set_phase_done(self, success: bool = True) -> None:
        self.phase_progress.setRange(0, 1)
        self.phase_progress.setValue(1 if success else 0)

    def set_overall_percent(self, percent: float | None) -> None:
        if percent is None:
            return
        value = max(0, min(1000, int(round(percent * 10))))
        self.total_progress.setRange(0, 1000)
        self.total_progress.setValue(value)

    def apply_progress_fields(self, fields: dict[str, str]) -> None:
        phase = fields.get("stage") or phase_label(fields.get("phase"))
        self.phase_value.setText(phase)
        raw_phase = fields.get("phase")
        if raw_phase and raw_phase != phase:
            self.stage_value.setText(f"{phase} ({raw_phase})")
        else:
            self.stage_value.setText(phase)

        percent = progress_percent(fields, "percent", "phase_percent")
        if percent is not None:
            self.phase_progress.setRange(0, 1000)
            self.phase_progress.setValue(max(0, min(1000, int(round(percent * 10)))))
        else:
            self.set_phase_indeterminate()
        self.set_overall_percent(progress_percent(fields, "overall", "overall_percent"))

        current = fields.get("current")
        total = fields.get("total")
        unit = fields.get("unit", "")
        if current and total:
            self.frame_value.setText(f"{current}/{total} {unit}".strip())
        elif current:
            self.frame_value.setText(f"{current} {unit}".strip())
        fps = fields.get("fps")
        compute_fps = fields.get("compute_fps")
        if fps and compute_fps:
            self.speed_value.setText(f"{fps} / {compute_fps}")
        elif fps:
            self.speed_value.setText(f"{fps}")
        elif compute_fps:
            self.speed_value.setText(f"compute {compute_fps}")

        text = progress_status_text(fields)
        self.status_banner.setText(f"実行中: {phase}")
        self.summary_text.setPlainText("status: running\n" + text)
        self.detail_info_value.setText(text)
        if fields.get("elapsed"):
            self.phase_elapsed_value.setText(fields["elapsed"])
        if fields.get("overall_eta"):
            self.overall_remaining_value.setText(fields["overall_eta"])
        elif fields.get("eta"):
            self.overall_remaining_value.setText(f"フェーズ {fields['eta']}")
        elif progress_percent(fields, "overall", "overall_percent", "percent", "phase_percent") == 100:
            self.overall_remaining_value.setText("0:00")
        self.heartbeat_value.setText(time.strftime("%Y-%m-%d %H:%M:%S"))
