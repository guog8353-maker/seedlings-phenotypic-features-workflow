"""可折叠的 QGroupBox。"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
)


class CollapsibleGroupBox(QWidget):
    """带展开/收起按钮的折叠面板。"""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self.btn_toggle = QPushButton("▼")
        self.btn_toggle.setFixedWidth(30)
        self.btn_toggle.setFlat(True)
        self.btn_toggle.setStyleSheet("font-weight: bold;")
        self.btn_toggle.clicked.connect(self.toggle)
        header.addWidget(self.btn_toggle)

        self.lbl_title = QLabel(f"<b>{title}</b>")
        header.addWidget(self.lbl_title)
        header.addStretch()
        layout.addLayout(header)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 4, 8, 8)
        layout.addWidget(self.content)

    def set_content(self, widget: QWidget):
        """将任意 widget 放入折叠区。"""
        self.content_layout.addWidget(widget)

    def toggle(self):
        visible = not self.content.isVisible()
        self.content.setVisible(visible)
        self.btn_toggle.setText("▼" if visible else "▶")

    def collapse(self):
        self.content.setVisible(False)
        self.btn_toggle.setText("▶")

    def expand(self):
        self.content.setVisible(True)
        self.btn_toggle.setText("▼")
