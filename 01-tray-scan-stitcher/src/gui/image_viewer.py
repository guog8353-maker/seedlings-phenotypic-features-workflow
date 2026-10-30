"""Image viewer widget with zoom and pan support."""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget


def numpy_to_qimage(frame: np.ndarray) -> QImage:
    """Convert OpenCV BGR numpy array to QImage."""
    if frame is None or frame.size == 0:
        return QImage()

    if len(frame.shape) == 2:
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format_Grayscale8)

    h, w, c = frame.shape
    if c == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    elif c == 4:
        bytes_per_line = 4 * w
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format_ARGB32)

    return QImage()


class ImageViewer(QGraphicsView):
    """A graphics view for displaying images with zoom and pan."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self._current_pixmap = QPixmap()

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._scale = 1.0
        self._min_scale = 0.05
        self._max_scale = 20.0

    def set_image(self, frame: np.ndarray):
        # Safety net: very large images can stall the main thread during
        # QImage/Pixmap conversion or fitInView.  Downsample before display.
        display_frame = self._limit_frame_size(frame, max_dim=4096)
        qimage = numpy_to_qimage(display_frame)
        if qimage.isNull():
            self._current_pixmap = QPixmap()
            self._pixmap_item.setPixmap(self._current_pixmap)
            return

        self._current_pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item.setPixmap(self._current_pixmap)

        if not self._current_pixmap.isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._scale = self.transform().m11()

    def _limit_frame_size(self, frame: np.ndarray, max_dim: int) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        if max(w, h) <= max_dim:
            return frame
        scale = max_dim / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def clear(self):
        self._current_pixmap = QPixmap()
        self._pixmap_item.setPixmap(self._current_pixmap)
        self._scene.setSceneRect(0, 0, 0, 0)

    def wheelEvent(self, event: QWheelEvent):
        if self._current_pixmap.isNull():
            return

        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = self._scale * zoom_factor

        if new_scale < self._min_scale or new_scale > self._max_scale:
            return

        self._scale = new_scale
        self.scale(zoom_factor, zoom_factor)

    def reset_view(self):
        self.resetTransform()
        self._scale = 1.0
        if not self._current_pixmap.isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._scale = self.transform().m11()
