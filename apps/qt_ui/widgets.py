from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


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
        if columns is None:
            layout: QtWidgets.QBoxLayout = QtWidgets.QHBoxLayout(self)
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
