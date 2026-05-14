#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets


ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_RAW_REMOVE_SHORT_TRACKS_MAX_FRAMES = 10
DEFAULT_OVERLAY_ENCODER = "nvenc"
GUI_RUNTIME_ENV = ROOT / "configs" / "gui_runtime.env"
DEFAULT_RUNTIME_PROFILE = ROOT / "configs" / "runtime_profile.json"
FALLBACK_TRT_ENGINE = ROOT / "checkpoints" / "trt" / "dinov3_backbone_fp32_1280x720_dynamic_bf16_forced_b1_8_8.engine"


def configure_qt_environment() -> None:
    plugin_root = Path(QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_root / "platforms")
    if "cv2" in os.environ.get("QT_PLUGIN_PATH", ""):
        os.environ.pop("QT_PLUGIN_PATH", None)
    if not os.environ.get("QT_QPA_PLATFORM") and os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    elif not os.environ.get("QT_QPA_PLATFORM") and os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"


configure_qt_environment()


def _valid_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def load_gui_runtime_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if not GUI_RUNTIME_ENV.is_file():
        return values
    try:
        for raw_line in GUI_RUNTIME_ENV.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError:
                tokens = [line]
            if not tokens:
                continue
            key, value = tokens[0].split("=", 1)
            values[key] = value
    except Exception:
        return {}
    return values


def runtime_profile_path() -> Path:
    runtime_env = load_gui_runtime_env()
    raw = os.environ.get("DINOV3_RUNTIME_PROFILE") or runtime_env.get("DINOV3_RUNTIME_PROFILE")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return DEFAULT_RUNTIME_PROFILE


def load_runtime_profile() -> dict:
    try:
        path = runtime_profile_path()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def default_python() -> Path:
    runtime_env = load_gui_runtime_env()
    profile = load_runtime_profile()
    candidates = [
        os.environ.get("GUI_RUNTIME_PYTHON"),
        runtime_env.get("GUI_RUNTIME_PYTHON"),
        profile.get("runtime", {}).get("python") if isinstance(profile.get("runtime"), dict) else None,
        profile.get("python"),
        str(ROOT / ".venv_integrated" / "bin" / "python"),
        sys.executable,
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if _valid_executable(path):
            return path
    integrated = ROOT / ".venv_integrated" / "bin" / "python"
    return integrated if integrated.is_file() else Path(sys.executable)


def profile_recommendations() -> dict:
    recommendations = load_runtime_profile().get("recommendations", {})
    return recommendations if isinstance(recommendations, dict) else {}


def profile_int(section: str, key: str) -> int | None:
    try:
        value = profile_recommendations().get(section, {}).get(key)
        return int(value) if value is not None else None
    except Exception:
        return None


def profile_path(section: str, key: str) -> Path | None:
    try:
        value = profile_recommendations().get(section, {}).get(key)
        if not value:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else ROOT / path
    except Exception:
        return None


def selected_trt_engine() -> Path:
    runtime_env = load_gui_runtime_env()
    raw = os.environ.get("DINOV3_TRT_BACKBONE_ENGINE") or runtime_env.get("DINOV3_TRT_BACKBONE_ENGINE")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else ROOT / path
    return profile_path("dinov3", "trt_backbone_engine") or FALLBACK_TRT_ENGINE


def runtime_summary_text() -> str:
    profile = load_runtime_profile()
    rec = profile.get("recommendations", {})
    rec = rec if isinstance(rec, dict) else {}
    benchmark = profile.get("batch_benchmark", {})
    benchmark_exists = bool(benchmark.get("exists")) if isinstance(benchmark, dict) else False
    dinov3 = rec.get("dinov3", {}) if isinstance(rec.get("dinov3"), dict) else {}
    eva02 = rec.get("eva02", {}) if isinstance(rec.get("eva02"), dict) else {}
    engine = selected_trt_engine()
    return (
        f"python={default_python()} | "
        f"profile={runtime_profile_path()} | "
        f"benchmark={'ok' if benchmark_exists else 'none'} | "
        f"DINOv3 batch={dinov3.get('batch_size', '既定')} | "
        f"EVA02 batch={eva02.get('batch_size', '既定')} | "
        f"EVA02 cls={eva02.get('classifier_batch_size', '既定')} | "
        f"engine={'ok' if engine.is_file() else 'missing'}"
    )


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def clean_run_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "video"


def as_command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


class ClosingComboBox(QtWidgets.QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.activated.connect(lambda _index: self.hidePopup())


class SegmentedOption(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal(object)

    def __init__(self, options: list[tuple[str, object]], *, columns: int | None = None) -> None:
        super().__init__()
        self._data_by_button: dict[QtWidgets.QAbstractButton, object] = {}
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        layout: QtWidgets.QBoxLayout
        if columns is None:
            layout = QtWidgets.QHBoxLayout(self)
        else:
            layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for index, (label, data) in enumerate(options):
            button = QtWidgets.QPushButton(label)
            button.setObjectName("segmentButton")
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setMinimumHeight(24)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self._group.addButton(button, index)
            self._data_by_button[button] = data
            layout.addWidget(button)
            if index == 0:
                button.setChecked(True)
        self._group.buttonClicked.connect(self._emit_changed)

    def currentData(self) -> object:
        button = self._group.checkedButton()
        return self._data_by_button.get(button)

    def setCurrentData(self, data: object) -> None:
        for button, value in self._data_by_button.items():
            if value == data:
                button.setChecked(True)
                self.changed.emit(value)
                return

    def _emit_changed(self, button: QtWidgets.QAbstractButton) -> None:
        self.changed.emit(self._data_by_button.get(button))


class PipelineUiWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SOD推論システム - フロントエンド")
        self.resize(798, 747)
        self.setMinimumSize(760, 640)

        self.queue_paths: list[Path] = []
        self.run_queue: list[Path] = []
        self.current_index = -1
        self.current_run_name = ""
        self.current_summary_path: Path | None = None
        self.process: QtCore.QProcess | None = None
        self.process_start_time = 0.0
        self.active_log_buffer = ""
        self.stopping = False
        self.workflow_running = False
        self.completed_steps = 0
        self.total_steps = 0
        self.steps_per_item = 1

        self.elapsed_timer = QtCore.QTimer(self)
        self.elapsed_timer.setInterval(1000)
        self.elapsed_timer.timeout.connect(self.update_elapsed)

        self.build_ui()
        self.apply_style()
        self.refresh_wsl_distros()
        self.update_queue_state()
        self.update_running_state(False)

    def build_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setObjectName("centralRoot")
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        self.setCentralWidget(central)

        root.addWidget(self.build_run_settings())
        main_label = QtWidgets.QLabel("メイン")
        main_label.setObjectName("mainSectionLabel")
        root.addWidget(main_label)

        body = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        body.setChildrenCollapsible(False)
        root.addWidget(body, 1)

        upper = QtWidgets.QWidget()
        upper_layout = QtWidgets.QHBoxLayout(upper)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(6)
        upper_layout.addWidget(self.build_queue_panel(), 7)
        upper_layout.addWidget(self.build_status_panel(), 11)
        body.addWidget(upper)

        lower = QtWidgets.QWidget()
        lower_layout = QtWidgets.QHBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(6)
        lower_layout.addWidget(self.build_log_panel(), 7)
        lower_layout.addWidget(self.build_postprocess_panel(), 11)
        body.addWidget(lower)
        body.setSizes([270, 230])

    def build_run_settings(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("実行設定")
        self.run_settings_box = box
        box.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        root = QtWidgets.QVBoxLayout(box)
        root.setContentsMargins(7, 9, 7, 8)
        root.setSpacing(5)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.wsl_combo = ClosingComboBox()
        self.refresh_wsl_button = QtWidgets.QPushButton("WSL一覧更新")
        self.refresh_wsl_button.clicked.connect(self.refresh_wsl_distros)
        self.output_edit = QtWidgets.QLineEdit(str(ROOT / "output" / "runs"))
        self.output_browse_button = QtWidgets.QPushButton("参照")
        self.output_browse_button.clicked.connect(self.browse_output_dir)
        self.detector_combo = ClosingComboBox()
        self.detector_combo.addItem("DINOv3", "dinov3")
        self.detector_combo.addItem("EVA02", "eva02")
        self.detector_combo.setCurrentIndex(1)

        self.detailed_overlay_check = QtWidgets.QCheckBox("詳細オーバーレイ（元マスク + 後処理輪郭 + ID/クラス）")
        self.detector_overlay_check = QtWidgets.QCheckBox("AI生成カバーオーバーレイ（推論JSONLのみ）")
        self.simple_overlay_check = QtWidgets.QCheckBox("簡易オーバーレイ（後処理後マスクのみ）")
        for check in (self.detailed_overlay_check, self.detector_overlay_check, self.simple_overlay_check):
            check.setObjectName("largeCheck")
            check.setCursor(QtCore.Qt.PointingHandCursor)
        self.detailed_overlay_check.setChecked(True)

        self.start_button = QtWidgets.QPushButton("推論開始")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_queue)
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_process)
        self.check_artifacts_button = QtWidgets.QPushButton("チェックポイント更新")
        self.check_artifacts_button.setObjectName("checkpointButton")
        self.check_artifacts_button.setFlat(True)
        self.check_artifacts_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.check_artifacts_button.clicked.connect(self.check_artifacts)
        self.advanced_button = QtWidgets.QToolButton()
        self.advanced_button.setText("▸ 詳細を開く")
        self.advanced_button.setCheckable(True)
        self.advanced_button.toggled.connect(self.toggle_advanced)

        left = QtWidgets.QWidget()
        left.setFixedWidth(296)
        left_grid = QtWidgets.QGridLayout(left)
        left_grid.setContentsMargins(0, 0, 0, 0)
        left_grid.setHorizontalSpacing(6)
        left_grid.setVerticalSpacing(4)
        left_grid.addWidget(QtWidgets.QLabel("WSLディストリ"), 0, 0)
        left_grid.addWidget(self.wsl_combo, 0, 1)
        left_grid.addWidget(self.refresh_wsl_button, 0, 2)
        left_grid.addWidget(QtWidgets.QLabel("Backend"), 1, 0)
        left_grid.addWidget(self.detector_combo, 1, 1, 1, 2)
        left_grid.setColumnStretch(1, 1)

        overlay_layout = QtWidgets.QVBoxLayout()
        overlay_layout.setSpacing(1)
        overlay_layout.addWidget(self.detailed_overlay_check)
        overlay_layout.addWidget(self.detector_overlay_check)
        overlay_layout.addWidget(self.simple_overlay_check)
        left_grid.addLayout(overlay_layout, 2, 0, 1, 3)

        checkpoint_layout = QtWidgets.QVBoxLayout()
        checkpoint_layout.setContentsMargins(0, 1, 0, 0)
        checkpoint_layout.setSpacing(2)
        checkpoint_layout.addWidget(self.check_artifacts_button, 0, QtCore.Qt.AlignLeft)
        checkpoint_layout.addWidget(self.advanced_button, 0, QtCore.Qt.AlignLeft)
        left_grid.addLayout(checkpoint_layout, 3, 0, 1, 3)

        right = QtWidgets.QWidget()
        right_grid = QtWidgets.QGridLayout(right)
        right_grid.setContentsMargins(0, 0, 0, 0)
        right_grid.setHorizontalSpacing(7)
        right_grid.setVerticalSpacing(5)
        right_grid.setAlignment(QtCore.Qt.AlignTop)
        right_grid.addWidget(QtWidgets.QLabel("結果保存先"), 0, 0)
        right_grid.addWidget(self.output_edit, 0, 1)
        right_grid.addWidget(self.output_browse_button, 0, 2)
        right_grid.addWidget(self.start_button, 0, 3)
        right_grid.addWidget(self.stop_button, 1, 3)
        right_grid.setColumnStretch(1, 1)

        top.addWidget(left)
        top.addWidget(right, 1)
        root.addLayout(top)

        self.advanced_box = self.build_advanced_box()
        root.addWidget(self.advanced_box)
        self.advanced_box.setVisible(False)
        box.setMaximumHeight(182)
        return box

    def build_advanced_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("詳細")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)

        self.python_edit = QtWidgets.QLineEdit(str(default_python()))
        self.python_browse_button = QtWidgets.QPushButton("参照")
        self.python_browse_button.clicked.connect(self.browse_python)
        self.run_prefix_edit = QtWidgets.QLineEdit("ui_run")
        self.force_check = QtWidgets.QCheckBox("既存結果を上書き")
        self.force_check.setChecked(True)
        self.recursive_check = QtWidgets.QCheckBox("フォルダ入力を再帰検索")
        self.raw_cut_detect_check = QtWidgets.QCheckBox("カット検出")
        self.raw_cut_detect_check.setChecked(True)

        self.max_frames_spin = QtWidgets.QSpinBox()
        self.max_frames_spin.setRange(0, 100_000_000)
        self.max_frames_spin.setSpecialValueText("既定")
        self.batch_size_spin = QtWidgets.QSpinBox()
        self.batch_size_spin.setRange(0, 4096)
        self.batch_size_spin.setSpecialValueText("既定")
        self.warmup_spin = QtWidgets.QSpinBox()
        self.warmup_spin.setRange(-1, 100_000)
        self.warmup_spin.setValue(-1)
        self.warmup_spin.setSpecialValueText("既定")
        self.score_enable = QtWidgets.QCheckBox("score-thresh指定")
        self.score_spin = QtWidgets.QDoubleSpinBox()
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.01)
        self.score_spin.setDecimals(3)
        self.score_spin.setValue(0.300)
        self.score_spin.setEnabled(False)
        self.score_enable.toggled.connect(self.score_spin.setEnabled)
        self.runtime_info_label = QtWidgets.QLabel(runtime_summary_text())
        self.runtime_info_label.setObjectName("runtimeInfoLabel")
        self.runtime_info_label.setWordWrap(True)
        self.runtime_info_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        grid.addWidget(QtWidgets.QLabel("実行Python"), 0, 0)
        grid.addWidget(self.python_edit, 0, 1, 1, 3)
        grid.addWidget(self.python_browse_button, 0, 4)
        grid.addWidget(QtWidgets.QLabel("Run名Prefix"), 1, 0)
        grid.addWidget(self.run_prefix_edit, 1, 1)
        grid.addWidget(self.force_check, 1, 2)
        grid.addWidget(self.recursive_check, 1, 3)
        grid.addWidget(self.raw_cut_detect_check, 1, 4)
        grid.addWidget(QtWidgets.QLabel("最大フレーム"), 2, 0)
        grid.addWidget(self.max_frames_spin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("batch-size"), 2, 2)
        grid.addWidget(self.batch_size_spin, 2, 3)
        grid.addWidget(QtWidgets.QLabel("warmup"), 2, 4)
        grid.addWidget(self.warmup_spin, 2, 5)
        grid.addWidget(self.score_enable, 3, 0)
        grid.addWidget(self.score_spin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("Runtime"), 3, 2)
        grid.addWidget(self.runtime_info_label, 3, 3, 1, 3)
        return box

    def build_queue_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("入力動画キュー")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)
        self.queue_stack = QtWidgets.QStackedWidget()
        self.empty_queue_label = QtWidgets.QLabel("動画がありません")
        self.empty_queue_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.queue_list = QtWidgets.QListWidget()
        self.queue_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.queue_stack.addWidget(self.empty_queue_label)
        self.queue_stack.addWidget(self.queue_list)
        layout.addWidget(self.queue_stack, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.add_video_button = QtWidgets.QPushButton("追加")
        self.add_video_button.clicked.connect(self.add_videos)
        self.add_folder_button = QtWidgets.QPushButton("フォルダ")
        self.add_folder_button.clicked.connect(self.add_folder)
        self.remove_button = QtWidgets.QPushButton("削除")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QtWidgets.QPushButton("全削除")
        self.clear_button.clicked.connect(self.clear_queue)
        buttons.addWidget(self.add_video_button)
        buttons.addWidget(self.add_folder_button)
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return box

    def build_status_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("実行状況")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 7)
        layout.setSpacing(3)
        self.status_banner = QtWidgets.QLabel("待機中")
        self.status_banner.setObjectName("statusBanner")
        self.status_banner.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_banner)

        progress_grid = QtWidgets.QGridLayout()
        self.total_progress = QtWidgets.QProgressBar()
        self.total_progress.setRange(0, 1)
        self.total_progress.setValue(0)
        self.phase_progress = QtWidgets.QProgressBar()
        self.phase_progress.setRange(0, 1)
        self.phase_progress.setValue(0)
        progress_grid.addWidget(QtWidgets.QLabel("全体進行"), 0, 0)
        progress_grid.addWidget(self.total_progress, 0, 1)
        progress_grid.addWidget(QtWidgets.QLabel("フェーズ"), 1, 0)
        progress_grid.addWidget(self.phase_progress, 1, 1)
        layout.addLayout(progress_grid)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(0)
        form.setVerticalSpacing(0)
        self.current_video_value = self.status_value("-")
        self.speed_value = self.status_value("-")
        self.detail_info_value = self.status_value("-")
        self.elapsed_value = self.status_value("-")
        self.process_value = self.status_value("NotRunning")
        self.stage_value = self.status_value("-")
        self.frame_value = self.status_value("-")
        self.phase_value = self.status_value("-")
        self.phase_elapsed_value = self.status_value("-")
        self.status_value_label = self.status_value("-")
        self.count_value = self.status_value("Inference 0 / 0 | Postprocess 0 / 0")
        self.heartbeat_value = self.status_value("-")
        self.remaining_value = self.status_value("-")
        self.overall_remaining_value = self.status_value("-")
        self.output_value = self.status_value("-")
        self.command_value = self.status_value("-")

        rows: list[tuple[str, QtWidgets.QWidget, str, QtWidgets.QWidget]] = [
            ("動画", self.current_video_value, "経過", self.elapsed_value),
            ("速度", self.speed_value, "", self.status_value("")),
            ("詳細情報", self.detail_info_value, "ステータス", self.status_value_label),
            ("プロセス", self.process_value, "動画数", self.count_value),
            ("ステージ", self.stage_value, "ハートビート", self.heartbeat_value),
            ("フレーム", self.frame_value, "残り", self.remaining_value),
            ("フェーズ", self.phase_value, "", self.status_value("")),
            ("フェーズ経過", self.phase_elapsed_value, "", self.status_value("")),
            ("全体残り", self.overall_remaining_value, "", self.status_value("")),
            ("実行Dir", self.output_value, "", self.status_value("")),
        ]
        for row, (left_label, left_widget, right_label, right_widget) in enumerate(rows):
            form.addWidget(self.status_key(left_label), row, 0)
            form.addWidget(left_widget, row, 1)
            form.addWidget(self.status_key(right_label), row, 2)
            form.addWidget(right_widget, row, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 3)
        layout.addLayout(form)

        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(34)
        self.summary_text.setPlainText("status: idle")
        layout.addWidget(self.summary_text)
        return box

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

    def build_log_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("実行ログ")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 8)
        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setObjectName("logEdit")
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        layout.addWidget(self.log_edit)
        return box

    def build_postprocess_panel(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("自動後処理設定")
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(8, 10, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(7)

        self.postprocess_check = QtWidgets.QCheckBox("推論後に自動で後処理を実行")
        self.postprocess_check.setObjectName("largeCheck")
        self.postprocess_check.setChecked(True)
        self.postprocess_check.toggled.connect(self.update_postprocess_enabled)

        self.class_tabs = QtWidgets.QTabWidget()
        self.class_tabs.setFixedHeight(34)
        for name in ("女性器", "男性器", "結合部分"):
            page = QtWidgets.QWidget()
            self.class_tabs.addTab(page, name)

        self.shape_combo = ClosingComboBox()
        self.shape_combo.addItem("楕円近似", "ellipse")
        self.shape_combo.addItem("ポリゴン", "polygon")
        self.keyframe_spin = QtWidgets.QSpinBox()
        self.keyframe_spin.setRange(1, 300)
        self.keyframe_spin.setValue(3)
        self.recall_spin = QtWidgets.QDoubleSpinBox()
        self.recall_spin.setRange(0.001, 1.0)
        self.recall_spin.setDecimals(3)
        self.recall_spin.setSingleStep(0.005)
        self.recall_spin.setValue(0.960)
        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(3)
        self.confidence_spin.setSingleStep(0.005)
        self.confidence_spin.setValue(0.350)

        grid.addWidget(self.postprocess_check, 0, 0, 1, 3)
        grid.addWidget(self.class_tabs, 1, 0, 1, 3)
        grid.addWidget(QtWidgets.QLabel("マスクタイプ"), 2, 0)
        grid.addWidget(self.shape_combo, 2, 1)
        grid.addWidget(QtWidgets.QLabel("キーフレーム間隔"), 3, 0)
        grid.addWidget(self.keyframe_spin, 3, 1)
        grid.addWidget(QtWidgets.QLabel("recall閾値"), 4, 0)
        grid.addWidget(self.recall_spin, 4, 1)
        grid.addWidget(QtWidgets.QLabel("confidence閾値"), 5, 0)
        grid.addWidget(self.confidence_spin, 5, 1)
        grid.setColumnStretch(2, 1)
        self.update_postprocess_enabled(True)
        return box

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#centralRoot {
                background: #f3f5f8;
            }
            QWidget {
                font-size: 12px;
                color: #1f2933;
            }
            QLabel#mainSectionLabel {
                padding: 1px 0 0 2px;
                min-height: 16px;
            }
            QGroupBox {
                border: 1px solid #c9d3df;
                border-radius: 2px;
                margin-top: 9px;
                font-weight: 600;
                background: #f9fafc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
                background: #f3f5f8;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QPlainTextEdit {
                border: 1px solid #cfd8e3;
                border-radius: 2px;
                padding: 3px 5px;
                background: #ffffff;
                min-height: 22px;
            }
            QPushButton {
                min-height: 25px;
                padding: 2px 10px;
                border: 1px solid #c8d2df;
                border-radius: 2px;
                background: #ffffff;
            }
            QPushButton:hover {
                border-color: #8ea8c8;
                background: #f6faff;
            }
            QPushButton#checkpointButton {
                border: none;
                background: transparent;
                padding: 0 0 0 0;
                min-height: 16px;
                text-align: left;
                font-weight: 400;
            }
            QPushButton#checkpointButton:hover {
                color: #1c5fd1;
                background: transparent;
            }
            QPushButton#segmentButton {
                text-align: left;
                min-height: 23px;
                padding: 2px 9px;
                font-weight: 600;
                background: #ffffff;
            }
            QPushButton#segmentButton:checked {
                background: #e8f1ff;
                border: 1px solid #3b82f6;
                color: #114fba;
            }
            QCheckBox#largeCheck {
                spacing: 8px;
                min-height: 19px;
                padding: 1px 4px;
                font-weight: 600;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QPushButton#startButton {
                background: #2672f3;
                color: white;
                border: 1px solid #1c5fd1;
                border-radius: 3px;
                font-weight: 700;
            }
            QPushButton#stopButton {
                background: #6b7280;
                color: white;
                border: 1px solid #4b5563;
                border-radius: 3px;
                font-weight: 700;
            }
            QLabel#statusBanner {
                background: #687386;
                color: #ffffff;
                min-height: 34px;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#statusKey, QLabel#statusValue {
                border: 1px solid #d9e0ea;
                background: rgba(255, 255, 255, 0.55);
                padding: 1px 4px;
                min-height: 15px;
                font-size: 10px;
            }
            QLabel#statusKey {
                background: #f7f9fc;
            }
            QPlainTextEdit#logEdit {
                background: #151b22;
                color: #e6edf3;
                font-family: "Noto Sans Mono", "Consolas", monospace;
                font-size: 11px;
            }
            QProgressBar {
                border: 1px solid #cfd8e3;
                background: #ffffff;
                text-align: center;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background: #86b7ff;
            }
            """
        )

    def toggle_advanced(self, checked: bool) -> None:
        self.advanced_box.setVisible(checked)
        self.run_settings_box.setMaximumHeight(420 if checked else 182)
        self.advanced_button.setText("▾ 詳細を閉じる" if checked else "▸ 詳細を開く")

    def sync_overlay_checks(self, source: str, checked: bool) -> None:
        return

    def update_postprocess_enabled(self, enabled: bool) -> None:
        widgets = [
            self.class_tabs,
            self.shape_combo,
            self.keyframe_spin,
            self.recall_spin,
            self.confidence_spin,
            self.detailed_overlay_check,
            self.simple_overlay_check,
        ]
        for widget in widgets:
            widget.setEnabled(enabled)

    def refresh_wsl_distros(self) -> None:
        current = self.wsl_combo.currentText() if self.wsl_combo.count() else ""
        distros = ["Ubuntu"]
        if os.name == "nt":
            try:
                completed = subprocess.run(
                    ["wsl.exe", "-l", "-q"],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=5,
                )
                found = [line.strip("\x00 \r") for line in completed.stdout.splitlines()]
                distros = [line for line in found if line] or distros
            except Exception:
                pass
        self.wsl_combo.clear()
        self.wsl_combo.addItems(distros)
        if current:
            index = self.wsl_combo.findText(current)
            if index >= 0:
                self.wsl_combo.setCurrentIndex(index)

    def browse_output_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "結果保存先", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def browse_python(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "実行Python", str(ROOT), "Python (*)")
        if path:
            self.python_edit.setText(path)

    def add_videos(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "動画を追加",
            str(ROOT / "input"),
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;All files (*)",
        )
        self.add_paths([Path(file) for file in files])

    def add_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "フォルダを追加", str(ROOT / "input"))
        if folder:
            self.add_paths([Path(folder)])

    def add_paths(self, paths: list[Path]) -> None:
        existing = {path.resolve() for path in self.queue_paths if path.exists()}
        for path in paths:
            resolved = path.expanduser().resolve()
            if resolved in existing:
                continue
            if not resolved.exists():
                self.append_log(f"[skip] not found: {resolved}")
                continue
            if resolved.is_file() and resolved.suffix.lower() not in VIDEO_EXTS:
                self.append_log(f"[skip] unsupported file: {resolved}")
                continue
            self.queue_paths.append(resolved)
            existing.add(resolved)
        self.update_queue_state()

    def remove_selected(self) -> None:
        selected_rows = sorted((index.row() for index in self.queue_list.selectedIndexes()), reverse=True)
        for row in selected_rows:
            if 0 <= row < len(self.queue_paths):
                del self.queue_paths[row]
        self.update_queue_state()

    def clear_queue(self) -> None:
        if self.process_is_running():
            return
        self.queue_paths.clear()
        self.update_queue_state()

    def update_queue_state(self) -> None:
        self.queue_list.clear()
        for path in self.queue_paths:
            item = QtWidgets.QListWidgetItem(str(path))
            item.setToolTip(str(path))
            self.queue_list.addItem(item)
        self.queue_stack.setCurrentWidget(self.queue_list if self.queue_paths else self.empty_queue_label)
        self.start_button.setEnabled(bool(self.queue_paths) and not self.process_is_running())
        self.remove_button.setEnabled(bool(self.queue_paths) and not self.process_is_running())
        self.clear_button.setEnabled(bool(self.queue_paths) and not self.process_is_running())

    def check_artifacts(self) -> None:
        if self.process_is_running():
            return
        command = [self.python_edit.text().strip() or str(default_python()), str(ROOT / "tools" / "check_artifacts.py")]
        self.append_log("[artifact-check] " + as_command_text(command))
        self.start_process(command, label="チェックポイント確認", tool_only=True)

    def start_queue(self) -> None:
        if self.process_is_running() or not self.queue_paths:
            return
        expanded_inputs = self.expand_queue_paths()
        if not expanded_inputs:
            self.status_banner.setText("エラー")
            self.status_value_label.setText("no videos")
            self.summary_text.setPlainText("status: error\nqueue: no supported videos")
            return
        output_root = Path(self.output_edit.text()).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        self.run_queue = expanded_inputs
        self.current_index = -1
        self.current_summary_path = None
        self.stopping = False
        self.workflow_running = True
        self.completed_steps = 0
        self.steps_per_item = 2 if self.postprocess_check.isChecked() else 1
        self.total_steps = max(1, len(self.run_queue) * self.steps_per_item)
        self.log_edit.clear()
        self.total_progress.setRange(0, self.total_steps)
        self.total_progress.setValue(0)
        self.status_banner.setText("実行中")
        self.status_value_label.setText("running")
        self.summary_text.setPlainText("status: running")
        self.start_next_item()

    def expand_queue_paths(self) -> list[Path]:
        expanded: list[Path] = []
        seen: set[Path] = set()
        for path in self.queue_paths:
            if path.is_file():
                candidates = [path]
            elif path.is_dir():
                iterator = path.rglob("*") if self.recursive_check.isChecked() else path.iterdir()
                candidates = sorted(
                    candidate for candidate in iterator if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTS
                )
            else:
                self.append_log(f"[skip] not found: {path}")
                continue
            for candidate in candidates:
                resolved = candidate.expanduser().resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                expanded.append(resolved)
        return expanded

    def start_next_item(self) -> None:
        self.current_index += 1
        if self.current_index >= len(self.run_queue):
            self.finish_queue(success=True)
            return

        path = self.run_queue[self.current_index]
        command = self.build_command(path, self.current_index)
        self.current_run_name = self.extract_run_name(command)
        self.current_summary_path = Path(self.output_edit.text()).expanduser() / self.current_run_name / "summary.json"
        self.current_video_value.setText(path.name)
        self.output_value.setText(str(self.current_summary_path.parent))
        self.command_value.setText(as_command_text(command))
        self.phase_value.setText("起動中")
        self.status_value_label.setText("running")
        self.count_value.setText(
            f"Inference {self.current_index + 1} / {len(self.run_queue)} | "
            f"Postprocess {self.current_index + 1 if self.postprocess_check.isChecked() else 0} / {len(self.run_queue)}"
        )
        self.remaining_value.setText(str(len(self.run_queue) - self.current_index - 1))
        self.append_log("")
        self.append_log(f"[queue] {self.current_index + 1}/{len(self.run_queue)} {path}")
        self.start_process(command, label=path.name, tool_only=False)

    def build_command(self, input_path: Path, index: int) -> list[str]:
        script = ROOT / "scripts" / "run_integrated_pipeline.py"
        job_script = ROOT / "apps" / "qt_ui" / "run_ui_job.py"
        detector = str(self.detector_combo.currentData())
        output_root = Path(self.output_edit.text()).expanduser()
        prefix = clean_run_part(self.run_prefix_edit.text() or "ui_run")
        run_name = f"{prefix}_{timestamp()}_{index + 1:02d}_{clean_run_part(input_path.stem)}"
        postprocess = self.postprocess_check.isChecked()
        detailed_overlay = postprocess and self.detailed_overlay_check.isChecked()
        simple_overlay = postprocess and self.simple_overlay_check.isChecked()
        if detailed_overlay and simple_overlay:
            overlay_mode = "both"
        elif detailed_overlay:
            overlay_mode = "detailed"
        elif simple_overlay:
            overlay_mode = "simple"
        else:
            overlay_mode = "none"
        shape_mode = str(self.shape_combo.currentData())
        keyframe_interval = self.keyframe_spin.value()
        recall = self.recall_spin.value()
        pipeline_command = [
            self.python_edit.text().strip() or str(default_python()),
            str(script),
            "--input",
            str(input_path),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--detector",
            detector,
            "--postprocess" if postprocess else "--no-postprocess",
        ]
        if postprocess:
            embed_original_masks = self.detailed_overlay_check.isChecked()
            policy_path = self.write_policy_file(output_root, run_name, shape_mode, keyframe_interval, recall)
            pipeline_command.extend(
                [
                    "--intervals",
                    str(keyframe_interval),
                    "--default-shape-mode",
                    shape_mode,
                    "--class-policy-json",
                    str(policy_path),
                    "--no-render-overlays",
                    "--overlay-encoder",
                    DEFAULT_OVERLAY_ENCODER,
                    "--raw-remove-short-tracks-max-frames",
                    str(DEFAULT_RAW_REMOVE_SHORT_TRACKS_MAX_FRAMES),
                    "--raw-cut-detect" if self.raw_cut_detect_check.isChecked() else "--no-raw-cut-detect",
                ]
            )
        if self.force_check.isChecked():
            pipeline_command.append("--force")
        if self.max_frames_spin.value() > 0:
            pipeline_command.extend(["--max-frames", str(self.max_frames_spin.value())])
        if detector == "eva02":
            eva02_batch = self.batch_size_spin.value() if self.batch_size_spin.value() > 0 else profile_int("eva02", "batch_size")
            eva02_warmup = self.warmup_spin.value() if self.warmup_spin.value() >= 0 else profile_int("eva02", "warmup_frames")
            eva02_classifier_batch = profile_int("eva02", "classifier_batch_size")
            if eva02_batch:
                pipeline_command.extend(["--eva02-batch-size", str(eva02_batch)])
            if eva02_warmup is not None:
                pipeline_command.extend(["--eva02-warmup-frames", str(eva02_warmup)])
            if eva02_classifier_batch:
                pipeline_command.extend(["--eva02-classifier-batch-size", str(eva02_classifier_batch)])
        else:
            dinov3_batch = self.batch_size_spin.value() if self.batch_size_spin.value() > 0 else profile_int("dinov3", "batch_size")
            dinov3_warmup = self.warmup_spin.value() if self.warmup_spin.value() >= 0 else profile_int("dinov3", "warmup_frames")
            trt_engine = selected_trt_engine()
            if dinov3_batch:
                pipeline_command.extend(["--batch-size", str(dinov3_batch)])
            if dinov3_warmup is not None:
                pipeline_command.extend(["--warmup-frames", str(dinov3_warmup)])
            if trt_engine.is_file():
                pipeline_command.extend(["--trt-backbone-engine", str(trt_engine)])
        if self.score_enable.isChecked():
            score_flag = "--eva02-score-thresh" if detector == "eva02" else "--score-thresh"
            pipeline_command.extend([score_flag, f"{self.score_spin.value():.3f}"])
        if postprocess:
            pipeline_command.extend(
                [
                    "--postprocess-extra-args",
                    "--embed-original-masks" if embed_original_masks else "--no-embed-original-masks",
                    "--raw-det-score-min",
                    f"{self.confidence_spin.value():.3f}",
                    "--dense-recall-target",
                    f"{recall:.3f}",
                    "--polygon-recall-min",
                    f"{recall:.3f}",
                    "--progress-interval-sec",
                    "5",
                ]
            )
        command = [
            self.python_edit.text().strip() or str(default_python()),
            str(job_script),
            "--input",
            str(input_path),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--overlay-mode",
            overlay_mode,
            "--raw-overlay" if self.detector_overlay_check.isChecked() else "--no-raw-overlay",
            "--encoder",
            DEFAULT_OVERLAY_ENCODER,
        ]
        if self.force_check.isChecked():
            command.append("--force")
        command.append("--")
        command.extend(pipeline_command)
        return command

    def write_policy_file(
        self,
        output_root: Path,
        run_name: str,
        shape_mode: str,
        keyframe_interval: int,
        recall: float,
    ) -> Path:
        entry = {
            "shape_mode": shape_mode,
            "target_interval": int(keyframe_interval),
            "dense_recall_target": float(recall),
            "polygon_recall_min": float(recall),
        }
        policy = {
            "default": entry,
            "classes": {
                "男性器": dict(entry),
                "女性器": dict(entry),
                "結合部分": dict(entry),
                "結合": dict(entry),
            },
        }
        config_dir = output_root / run_name / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "class_policy.ui.json"
        path.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def extract_run_name(self, command: list[str]) -> str:
        try:
            return command[command.index("--run-name") + 1]
        except (ValueError, IndexError):
            return f"ui_run_{timestamp()}"

    def start_process(self, command: list[str], *, label: str, tool_only: bool) -> None:
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("DINOV3_RUNTIME_PROFILE", str(runtime_profile_path()))
        benchmark_path = load_gui_runtime_env().get("DINOV3_BATCH_BENCHMARK") or os.environ.get("DINOV3_BATCH_BENCHMARK")
        environment.insert("DINOV3_BATCH_BENCHMARK", benchmark_path or str(ROOT / "configs" / "runtime_benchmark.json"))
        environment.insert("DINOV3_TRT_BACKBONE_ENGINE", str(selected_trt_engine()))
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(lambda code, status: self.process_finished(code, status, tool_only))
        self.process.errorOccurred.connect(self.process_error)
        self.active_log_buffer = ""
        self.process_start_time = time.perf_counter()
        self.elapsed_timer.start()
        self.update_running_state(True)
        self.process_value.setText("Running")
        self.phase_progress.setRange(0, 0)
        self.phase_progress.setValue(0)
        self.append_log("[runtime] " + runtime_summary_text())
        self.append_log("[cmd] " + as_command_text(command))
        self.process.start(command[0], command[1:])
        if not self.process.waitForStarted(3000):
            self.append_log(f"[error] failed to start: {label}")
            self.handle_start_failure(tool_only=tool_only)

    def handle_start_failure(self, *, tool_only: bool) -> None:
        self.elapsed_timer.stop()
        self.phase_progress.setRange(0, 1)
        self.phase_progress.setValue(0)
        self.process_value.setText("NotRunning")
        self.status_banner.setText("エラー")
        self.status_value_label.setText("failed to start")
        self.summary_text.setPlainText("status: error\nprocess: failed to start")
        self.process = None
        self.update_running_state(False)
        if self.workflow_running and not tool_only:
            self.finish_queue(success=False, exit_code=-1)

    def read_process_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if not data:
            return
        self.active_log_buffer += data
        while "\n" in self.active_log_buffer:
            line, self.active_log_buffer = self.active_log_buffer.split("\n", 1)
            self.handle_process_line(line.rstrip())
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_edit.setTextCursor(cursor)

    def handle_process_line(self, line: str) -> None:
        self.append_log(line)
        if line.startswith("[run]"):
            self.phase_value.setText(line.removeprefix("[run]").strip())
        elif line.startswith("[done]"):
            self.phase_value.setText(line.removeprefix("[done]").strip())
            if self.workflow_running and self.total_steps:
                self.completed_steps = min(self.completed_steps + 1, self.total_steps)
                self.total_progress.setValue(self.completed_steps)
        elif line.startswith("[summary]"):
            self.summary_text.setPlainText("status: completed\n" + line)
        elif line.startswith("[index]"):
            self.summary_text.setPlainText("status: completed\n" + line)
        elif line.startswith("[phase-start]"):
            self.phase_value.setText(line.removeprefix("[phase-start]").split(":", 1)[0].strip())
        elif line.startswith("[phase-progress]"):
            text = line.removeprefix("[phase-progress]").strip()
            self.phase_value.setText(text.split(":", 1)[0].strip())
            self.summary_text.setPlainText("status: running\n" + text)
        elif line.startswith("[phase-done]"):
            self.phase_value.setText(line.removeprefix("[phase-done]").strip())
        elif line.startswith("[DONE]"):
            self.summary_text.setPlainText("status: running\n" + line)
        elif line.startswith("[error]") or "failed" in line.lower():
            self.status_value_label.setText("error")
            self.status_banner.setText("エラー")
        elif line.startswith("[log]"):
            self.phase_value.setText(line.removeprefix("[log]").strip())
        self.heartbeat_value.setText(time.strftime("%Y-%m-%d %H:%M:%S"))

    def append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)

    def update_elapsed(self) -> None:
        if self.process_start_time <= 0:
            self.elapsed_value.setText("-")
            return
        elapsed = int(time.perf_counter() - self.process_start_time)
        self.elapsed_value.setText(f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")

    def process_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
        tool_only: bool,
    ) -> None:
        if self.active_log_buffer:
            self.handle_process_line(self.active_log_buffer.rstrip())
            self.active_log_buffer = ""
        self.elapsed_timer.stop()
        self.phase_progress.setRange(0, 1)
        self.phase_progress.setValue(1)
        self.process_value.setText("NotRunning")
        self.process = None
        self.update_running_state(False)

        success = exit_status == QtCore.QProcess.NormalExit and exit_code == 0 and not self.stopping
        if tool_only:
            self.status_banner.setText("待機中" if success else "エラー")
            self.status_value_label.setText("completed" if success else f"exit {exit_code}")
            self.summary_text.setPlainText(f"status: {'completed' if success else 'error'}\nreturncode: {exit_code}")
            return

        if success:
            expected_steps = min((self.current_index + 1) * self.steps_per_item, self.total_steps)
            self.completed_steps = max(self.completed_steps, expected_steps)
            self.total_progress.setValue(self.completed_steps)
            self.status_value_label.setText("completed")
            self.start_next_item()
            return

        self.finish_queue(success=False, exit_code=exit_code)

    def process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        self.append_log(f"[process-error] {error}")
        self.status_value_label.setText("error")
        self.status_banner.setText("エラー")

    def finish_queue(self, *, success: bool, exit_code: int = 0) -> None:
        self.elapsed_timer.stop()
        self.phase_progress.setRange(0, 1)
        self.phase_progress.setValue(1 if success else 0)
        self.update_running_state(False)
        self.process_value.setText("NotRunning")
        if success:
            self.status_banner.setText("完了")
            self.phase_value.setText("完了")
            self.status_value_label.setText("completed")
            self.summary_text.setPlainText("status: completed\nqueue: done")
        else:
            self.status_banner.setText("停止" if self.stopping else "エラー")
            if self.stopping:
                detail = "stopped"
            elif exit_code < 0:
                detail = "failed to start"
            else:
                detail = f"exit {exit_code}"
            self.status_value_label.setText(detail)
            self.summary_text.setPlainText(f"status: {'stopped' if self.stopping else 'error'}\nreturncode: {exit_code}")
        self.run_queue = []
        self.current_index = -1
        self.workflow_running = False
        self.stopping = False
        self.update_queue_state()

    def stop_process(self) -> None:
        if self.process is None:
            return
        self.stopping = True
        self.status_banner.setText("停止中")
        self.status_value_label.setText("terminating")
        self.process.terminate()
        QtCore.QTimer.singleShot(5000, self.kill_if_running)

    def kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            self.process.kill()

    def process_is_running(self) -> bool:
        return self.process is not None and self.process.state() != QtCore.QProcess.NotRunning

    def update_running_state(self, running: bool) -> None:
        self.start_button.setEnabled(bool(self.queue_paths) and not running)
        self.stop_button.setEnabled(running)
        self.check_artifacts_button.setEnabled(not running)
        self.add_video_button.setEnabled(not running)
        self.add_folder_button.setEnabled(not running)
        self.remove_button.setEnabled(bool(self.queue_paths) and not running)
        self.clear_button.setEnabled(bool(self.queue_paths) and not running)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.process_is_running():
            self.stop_process()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SOD推論システム")
    window = PipelineUiWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
