"""Draggable label widget used to equip a stamp tool for image annotation."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class DraggableLabel(QFrame):
    """A colored label block that starts a drag when pressed.

    The actual drag tracking is handled by the parent window via an application
    event filter, so this widget does NOT call grabMouse(). This keeps buttons,
    menus, and other UI elements responsive during a drag.

    Signals:
        drag_started: Emitted when the user presses this label to begin dragging.
                      Carries the label name (e.g. "1" or "0").
    """

    drag_started = Signal(str)

    def __init__(self, name: str, display_name: str, color_rgba: tuple, parent=None):
        super().__init__(parent)
        self.name = name
        self.display_name = display_name
        self.color_rgba = color_rgba

        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.OpenHandCursor)
        self.setFixedHeight(80)

        # Background color from the label RGBA (drop alpha for the widget bg).
        r, g, b, _ = color_rgba
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: rgba({r}, {g}, {b}, 160);
                border: 2px solid rgba(255, 255, 255, 180);
                border-radius: 8px;
            }}
            QFrame:hover {{
                background-color: rgba({r}, {g}, {b}, 200);
                border: 2px solid rgba(255, 255, 255, 240);
            }}
            QLabel {{
                color: white;
                background-color: transparent;
                font-weight: bold;
            }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self._name_label = QLabel(f"<h2>{name}</h2>")
        self._name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._name_label)

        self._desc_label = QLabel(display_name)
        self._desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._desc_label)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            event.accept()
            self.setCursor(Qt.ClosedHandCursor)
            self.drag_started.emit(self.name)
        else:
            super().mousePressEvent(event)

    def reset_cursor(self) -> None:
        """Restore the open-hand cursor after a drag finishes."""
        self.setCursor(Qt.OpenHandCursor)
