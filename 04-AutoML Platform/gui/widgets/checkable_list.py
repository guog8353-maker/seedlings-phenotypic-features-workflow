"""可勾选方法列表 + 一键全选复选框。"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QListWidget, QListWidgetItem,
)


class CheckableListWidget(QWidget):
    """带全选复选框的可勾选列表。"""

    selection_changed = pyqtSignal(list)

    def __init__(self, title: str = "选择方法", parent=None):
        super().__init__(parent)
        self._items: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        self.lbl_title = QLabel(f"<b>{title}</b>")
        top.addWidget(self.lbl_title)
        top.addStretch()
        self.check_all = QCheckBox("全选")
        self.check_all.setChecked(True)
        self.check_all.stateChanged.connect(self._on_all_changed)
        top.addWidget(self.check_all)
        layout.addLayout(top)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self.list_widget)

    def add_items(self, items: list[str]):
        self._items = list(items)
        self.list_widget.clear()
        for name in items:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_widget.addItem(item)
        self._emit()

    def set_checked(self, names: list[str]):
        names_set = set(names)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            checked = item.text() in names_set
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._sync_all_state()
        self._emit()

    def get_checked(self) -> list[str]:
        checked = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked.append(item.text())
        return checked

    def get_all_items(self) -> list[str]:
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]

    def _on_all_changed(self, state):
        check = Qt.CheckState.Checked if state == Qt.CheckState.Checked.value else Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(check)
        self._emit()

    def _sync_all_state(self):
        checked_count = sum(
            1 for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.CheckState.Checked
        )
        total = self.list_widget.count()
        self.check_all.blockSignals(True)
        if checked_count == total:
            self.check_all.setCheckState(Qt.CheckState.Checked)
        elif checked_count == 0:
            self.check_all.setCheckState(Qt.CheckState.Unchecked)
        else:
            self.check_all.setCheckState(Qt.CheckState.PartiallyChecked)
        self.check_all.blockSignals(False)

    def _emit(self):
        self.selection_changed.emit(self.get_checked())
