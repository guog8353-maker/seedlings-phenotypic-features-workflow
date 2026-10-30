"""Image canvas with grid and cell selection support."""

from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .grid_manager import GridCell, GridManager


class ImageCanvas(QWidget):
    """Widget that displays an image, a grid, and supports cell selection."""

    cell_selected = Signal(int, int)  # row, col
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_manager = GridManager()

        self._image: Optional[np.ndarray] = None          # ORIGINAL, full resolution
        self._preview_image: Optional[np.ndarray] = None  # display-only thumbnail
        self._preview_ratio: float = 1.0
        self._pixmap: Optional[QPixmap] = None
        self._selected_cell: Optional[Tuple[int, int]] = None

        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def load_image(self, image_path: str) -> bool:
        """Load an image (Unicode-path safe).  The ORIGINAL is kept for all
        computation; a <=2048px preview is built for display only."""
        from .error_reporting import StageTracker, log_exception
        from .image_io import imread_unicode

        StageTracker.set("LOAD_IMAGE")
        try:
            image = imread_unicode(image_path, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"无法解码图片（路径不存在或格式不支持）: {image_path}")

            h, w = image.shape[:2]
            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            self._image = image
            max_preview = 2048
            if max(w, h) > max_preview:
                r = max_preview / max(w, h)
                self._preview_image = cv2.resize(
                    image, (max(1, int(w * r)), max(1, int(h * r))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                self._preview_image = image
            self._preview_ratio = self._preview_image.shape[1] / w

            self._selected_cell = None
            self._build_pixmap()
            self._fit_to_window()
            self.update()
            self.status_message.emit(
                f"已加载图片: {w}×{h}（预览 {self._preview_image.shape[1]}×"
                f"{self._preview_image.shape[0]}，计算使用原始分辨率）"
            )
            return True
        except Exception as e:
            StageTracker.fail("LOAD_IMAGE")
            log_exception("LOAD_IMAGE", e, context=image_path)
            return False

    def get_image(self) -> Optional[np.ndarray]:
        return self._image

    def set_grid(self, cols: int, rows: int) -> None:
        """Rebuild the grid with the given M×N dimensions."""
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        self.grid_manager.update(w, h, cols, rows)
        self._fit_to_window()
        self.update()
        self.status_message.emit(f"已分块: {cols}×{rows} = {cols * rows} 个小块")

    def select_cell(self, row: int, col: int) -> None:
        """Highlight a specific cell."""
        self._selected_cell = (row, col)
        self.update()

    def clear_selection(self) -> None:
        self._selected_cell = None
        self.update()

    def _build_pixmap(self) -> None:
        # Pixmap comes from the preview only - never the full-size original.
        if self._preview_image is None:
            self._pixmap = None
            return
        rgb = cv2.cvtColor(self._preview_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image.copy())

    def _fit_to_window(self) -> None:
        if self._pixmap is None:
            return
        widget_w = self.width()
        widget_h = self.height()
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()
        self._scale = min(widget_w / img_w, widget_h / img_h)
        self._offset_x = (widget_w - img_w * self._scale) / 2.0
        self._offset_y = (widget_h - img_h * self._scale) / 2.0

    def resizeEvent(self, event) -> None:
        self._fit_to_window()
        super().resizeEvent(event)

    def _widget_to_image(self, wx: float, wy: float) -> Tuple[int, int]:
        s = self._scale * self._preview_ratio
        if s <= 1e-12:
            return 0, 0
        ix = int((wx - self._offset_x) / s)
        iy = int((wy - self._offset_y) / s)
        return ix, iy

    def _image_to_widget(self, ix: float, iy: float) -> QPoint:
        s = self._scale * self._preview_ratio
        wx = self._offset_x + ix * s
        wy = self._offset_y + iy * s
        return QPoint(int(wx), int(wy))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._pixmap is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "请导入图片")
            return

        target = QRect(
            int(self._offset_x),
            int(self._offset_y),
            int(self._pixmap.width() * self._scale),
            int(self._pixmap.height() * self._scale),
        )
        painter.drawPixmap(target, self._pixmap, self._pixmap.rect())

        # Highlight selected cell.
        if self._selected_cell is not None:
            cell = self.grid_manager.get_cell(*self._selected_cell)
            if cell is not None:
                top_left = self._image_to_widget(cell.x1, cell.y1)
                bottom_right = self._image_to_widget(cell.x2, cell.y2)
                rect = QRect(top_left, bottom_right)
                painter.fillRect(rect, QBrush(QColor(255, 235, 59, 80)))
                pen = QPen(QColor(255, 235, 59))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.setBrush(QBrush())
                painter.drawRect(rect)

        # Draw grid borders.
        pen = QPen(QColor(255, 255, 255, 180))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(QBrush())
        for cell in self.grid_manager.cells:
            top_left = self._image_to_widget(cell.x1, cell.y1)
            bottom_right = self._image_to_widget(cell.x2, cell.y2)
            rect = QRect(top_left, bottom_right)
            painter.drawRect(rect)

        # Draw cell indices (skipped for cells too small on screen).
        for cell in self.grid_manager.cells:
            top_left = self._image_to_widget(cell.x1, cell.y1)
            bottom_right = self._image_to_widget(cell.x2, cell.y2)
            cell_w = bottom_right.x() - top_left.x()
            cell_h = bottom_right.y() - top_left.y()
            if cell_w < 44 or cell_h < 26:
                continue
            font = QFont("Microsoft YaHei", max(6, min(int(8 * self._scale), int(cell_h * 0.3))))
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            text = f"{cell.row},{cell.col}"
            painter.drawText(top_left.x() + 2, top_left.y() + 10, text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._image is not None:
            ix, iy = self._widget_to_image(event.position().x(), event.position().y())
            cell = self.grid_manager.get_cell_at(ix, iy)
            if cell is not None:
                self._selected_cell = (cell.row, cell.col)
                self.cell_selected.emit(cell.row, cell.col)
                self.status_message.emit(f"选中: ({cell.row}, {cell.col})")
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            super().mouseMoveEvent(event)
            return
        ix, iy = self._widget_to_image(event.position().x(), event.position().y())
        cell = self.grid_manager.get_cell_at(ix, iy)
        if cell is not None:
            self.status_message.emit(f"位置: ({cell.row}, {cell.col})")
        super().mouseMoveEvent(event)
