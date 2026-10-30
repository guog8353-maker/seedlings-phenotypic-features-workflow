"""Reusable UI widgets."""

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FileSelector(QWidget):
    """Line edit + browse button for selecting a file or folder."""

    path_changed = Signal(str)

    def __init__(
        self,
        label: str = "文件",
        mode: str = "file",
        filter_str: str = "All Files (*.*)",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.filter_str = filter_str

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.button = QPushButton("浏览...")
        self.button.clicked.connect(self._browse)

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def _browse(self):
        if self.mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", self.filter_str)
        else:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")

        if path:
            self.edit.setText(path)
            self.path_changed.emit(path)

    def get_path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str):
        self.edit.setText(path)


class InfoCard(QGroupBox):
    """Display key-value pairs in a card."""

    def __init__(self, title: str = "信息", parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self._layout = QVBoxLayout(self)
        self._labels = {}

    def set_data(self, data: dict):
        for key in list(self._labels.keys()):
            if key not in data:
                self._labels[key].deleteLater()
                del self._labels[key]

        for key, value in data.items():
            text = f"{key}: {value}"
            if key in self._labels:
                self._labels[key].setText(text)
            else:
                label = QLabel(text)
                label.setWordWrap(True)
                self._labels[key] = label
                self._layout.addWidget(label)

    def clear(self):
        for label in self._labels.values():
            label.setText("")


def show_message(
    parent, title: str, message: str, icon: QMessageBox.Icon = QMessageBox.Information
):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.setIcon(icon)
    msg.exec()


def show_error(parent, title: str, message: str):
    show_message(parent, title, message, QMessageBox.Critical)
