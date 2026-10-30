"""Custom image canvas with grid and drag-to-annotate support."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from .grid_manager import GridCell, GridManager
from .label_manager import LabelManager


class ImageCanvas(QWidget):
    """Widget that displays an image, a grid, and annotation overlays."""

    # Emitted when the user annotates one or more cells.
    cells_annotated = Signal(int)
    status_message = Signal(str)
    # Emitted whenever the brush mask content changes.
    mask_changed = Signal()

    def __init__(self, label_manager: LabelManager, parent=None):
        super().__init__(parent)
        self.label_manager = label_manager
        self.grid_manager = GridManager()

        self._image: Optional[np.ndarray] = None          # ORIGINAL, full resolution
        self._preview_image: Optional[np.ndarray] = None  # display-only thumbnail
        self._preview_ratio: float = 1.0                  # preview_w / original_w
        self._pixmap: Optional[QPixmap] = None
        self._labels: Dict[Tuple[int, int], str] = {}

        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        self._hover_cell: Optional[GridCell] = None
        self._stamp_label_name: Optional[str] = None
        # {(row, col): confidence} from the last guttation detection run.
        self.cell_scores: Dict[Tuple[int, int], float] = {}

        # Tray-region point picking (4 corner points before rectification).
        self._point_pick_mode: bool = False
        self._picked_points: List[Tuple[int, int]] = []

        # Brush mask annotation (primary labeling mode).
        self._mask: Optional[np.ndarray] = None   # uint8 0/255, original resolution
        self.brush_radius: int = 25               # in original-image pixels
        self.brush_erase: bool = False            # False=paint, True=erase
        self._strokes: List[Tuple[List[Tuple[int, int]], int, bool]] = []
        self._current_stroke: Optional[List[Tuple[int, int]]] = None
        self.mask_view_mode: int = 0              # 0=overlay, 1=pure mask, 2=hidden
        self.grid_visible: bool = True
        self.clean_view: bool = False             # True = photo only (no overlays)
        self.mask_threshold: float = 0.01         # cell = 吐水 if mask >= 1% of cell
        self._mask_version: int = 0
        self._mask_overlay_cache: Optional[QPixmap] = None
        self._mask_pure_cache: Optional[QPixmap] = None
        self._mask_cache_version: int = -1
        self._mask_cache_time: float = 0.0

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def labels(self) -> Dict[Tuple[int, int], str]:
        return dict(self._labels)

    def set_labels(self, labels: Dict[Tuple[int, int], str]) -> None:
        self._labels = dict(labels)
        self.update()

    def load_image(self, image_path: str) -> bool:
        """Load an image from disk (Unicode-path safe).

        The ORIGINAL full-resolution image is kept for all processing
        (grid / annotation / export).  A small preview (longest side <= 1600)
        is built for display only; coordinate converters map between widget,
        preview and original spaces.  Raises on failure - callers must show
        the detailed error dialog (no silent False).
        """
        from .error_reporting import StageTracker, log_exception
        from .image_io import imread_unicode

        StageTracker.set("LOAD_IMAGE")
        try:
            image = imread_unicode(image_path, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"无法解码图片（路径不存在或格式不支持）: {image_path}")

            StageTracker.set("VALIDATE_IMAGE")
            h, w = image.shape[:2]
            if w <= 0 or h <= 0:
                raise ValueError(f"图片尺寸异常: {w}×{h}")
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                # Stitched panoramas may carry an alpha channel; convert to BGR.
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if image.dtype != np.uint8:
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            est_mb = w * h * 3 / 1e6
            mode = "LARGE_IMAGE_MODE" if w * h > 100_000_000 else "NORMAL_MODE"

            # Original for processing; thumbnail for display only.
            self._image = image
            max_preview = 1600
            if max(w, h) > max_preview:
                r = max_preview / max(w, h)
                new_w = max(1, int(w * r))
                new_h = max(1, int(h * r))
                self._preview_image = cv2.resize(
                    image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
            else:
                self._preview_image = image
            self._preview_ratio = self._preview_image.shape[1] / w

            self._labels.clear()
            self._build_pixmap()
            self._fit_to_window()
            self.update()
            self.status_message.emit(
                f"已加载图片: {w}×{h}（约 {est_mb:.0f}MB，{mode}，"
                f"预览 {self._preview_image.shape[1]}×{self._preview_image.shape[0]}）"
            )
            return True
        except Exception as e:
            StageTracker.fail("LOAD_IMAGE")
            log_exception("LOAD_IMAGE", e, context=image_path)
            raise

    def get_image(self) -> Optional[np.ndarray]:
        return self._image

    def set_image(self, image: np.ndarray) -> None:
        """Set the working image from an array (e.g. the rectified tray).

        All downstream work (grid / annotation / export) uses this image's
        coordinate system.  Preview separation works exactly like load_image.
        """
        if image is None or image.size == 0:
            return
        h, w = image.shape[:2]
        self._image = image
        max_preview = 1600
        if max(w, h) > max_preview:
            r = max_preview / max(w, h)
            self._preview_image = cv2.resize(
                image, (max(1, int(w * r)), max(1, int(h * r))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            self._preview_image = image
        self._preview_ratio = self._preview_image.shape[1] / w

        self._labels.clear()
        self._picked_points.clear()
        self._point_pick_mode = False
        self._build_pixmap()
        self._fit_to_window()
        self.update()
        self.status_message.emit(f"已设置工作图像: {w}×{h}")

    # ------------------------------------------------------------------
    # Tray-region point picking (before perspective rectification)
    # ------------------------------------------------------------------
    def set_point_pick_mode(self, enabled: bool) -> None:
        self._point_pick_mode = enabled
        self.update()

    def point_pick_mode(self) -> bool:
        return self._point_pick_mode

    @property
    def picked_points(self) -> List[Tuple[int, int]]:
        return list(self._picked_points)

    def undo_point(self) -> None:
        if self._picked_points:
            self._picked_points.pop()
            self.update()

    def clear_points(self) -> None:
        self._picked_points.clear()
        self.update()

    # ------------------------------------------------------------------
    # Brush mask annotation (primary labeling mode)
    # ------------------------------------------------------------------
    def set_brush(self, radius: int, erase: bool) -> None:
        self.brush_radius = max(2, int(radius))
        self.brush_erase = erase

    def set_mask_view_mode(self, mode: int) -> None:
        """0 = red overlay on photo, 1 = pure black/white mask, 2 = hidden."""
        self.mask_view_mode = int(mode)
        self.update()

    def set_grid_visible(self, visible: bool) -> None:
        self.grid_visible = bool(visible)
        self.update()

    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        """Restore a mask from saved state (matches original resolution)."""
        if mask is not None and self._image is not None:
            h, w = self._image.shape[:2]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self._mask = (mask > 0).astype(np.uint8) * 255
        else:
            self._mask = mask
        self._strokes.clear()
        self._mask_version += 1
        self.update()
        self.mask_changed.emit()

    def has_mask_content(self) -> bool:
        return self._mask is not None and bool((self._mask > 0).any())

    def clear_mask(self) -> None:
        if self._image is not None:
            self._mask = np.zeros(self._image.shape[:2], dtype=np.uint8)
        else:
            self._mask = None
        self._strokes.clear()
        self.update()
        self.mask_changed.emit()

    def undo_stroke(self) -> bool:
        """Remove the most recent stroke and repaint the mask."""
        if not self._strokes or self._image is None:
            return False
        self._strokes.pop()
        self._mask = np.zeros(self._image.shape[:2], dtype=np.uint8)
        for points, radius, erase in self._strokes:
            for px, py in points:
                cv2.circle(
                    self._mask, (px, py), radius,
                    0 if erase else 255, -1, cv2.LINE_AA,
                )
        self.update()
        self.mask_changed.emit()
        return True

    def labels_from_mask(self) -> Dict[Tuple[int, int], str]:
        """Derive per-cell labels from the mask (>= threshold of cell area)."""
        labels: Dict[Tuple[int, int], str] = {}
        if self._mask is None:
            return labels
        for cell in self.grid_manager.cells:
            crop = self._mask[cell.y1 : cell.y2, cell.x1 : cell.x2]
            if crop.size == 0:
                labels[(cell.row, cell.col)] = "0"
                continue
            ratio = float(np.count_nonzero(crop)) / float(crop.size)
            labels[(cell.row, cell.col)] = (
                "1" if ratio >= self.mask_threshold else "0"
            )
        return labels

    def _begin_stroke(self, ix: int, iy: int, erase: bool) -> None:
        if self._image is None:
            return
        if self._mask is None:
            self._mask = np.zeros(self._image.shape[:2], dtype=np.uint8)
        self._current_stroke = [(ix, iy)]
        self._stroke_erase = erase
        self._stamp_brush(ix, iy, erase)

    def _extend_stroke(self, ix: int, iy: int) -> None:
        if self._current_stroke is None or self._mask is None:
            return
        x0, y0 = self._current_stroke[-1]
        # Interpolate so fast drags stay continuous.
        dist = max(abs(ix - x0), abs(iy - y0), 1)
        for t in range(1, int(dist) + 1):
            px = int(x0 + (ix - x0) * t / dist)
            py = int(y0 + (iy - y0) * t / dist)
            self._stamp_brush(px, py, self._stroke_erase)
        self._current_stroke.append((ix, iy))
        self.update()

    def _end_stroke(self) -> None:
        if self._current_stroke is None:
            return
        if len(self._current_stroke) >= 1:
            self._strokes.append(
                (self._current_stroke, self.brush_radius, self._stroke_erase)
            )
            if len(self._strokes) > 20:
                self._strokes.pop(0)
        self._current_stroke = None
        self.mask_changed.emit()

    def _stamp_brush(self, ix: int, iy: int, erase: bool) -> None:
        if self._mask is None:
            return
        h, w = self._mask.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(
                self._mask, (ix, iy), self.brush_radius,
                0 if erase else 255, -1, cv2.LINE_AA,
            )
            self._mask_version += 1

    # ------------------------------------------------------------------
    # Mask rendering (cached, rebuilt at most ~10x/sec while painting)
    # ------------------------------------------------------------------
    def _mask_overlay_pixmap(self) -> Optional[QPixmap]:
        return self._build_mask_cache(pure=False)

    def _mask_pixmap(self) -> Optional[QPixmap]:
        return self._build_mask_cache(pure=True)

    def _build_mask_cache(self, pure: bool) -> Optional[QPixmap]:
        import time as _time

        if self._mask is None or self._preview_image is None:
            return None
        now = _time.time()
        fresh = (
            self._mask_cache_version == self._mask_version
            and now - self._mask_cache_time < 0.1
        )
        if fresh:
            return self._mask_pure_cache if pure else self._mask_overlay_cache

        ph, pw = self._preview_image.shape[:2]
        m = cv2.resize(self._mask, (pw, ph), interpolation=cv2.INTER_NEAREST)

        rgba = np.zeros((ph, pw, 4), np.uint8)
        rgba[m > 0] = (255, 40, 40, 110)
        q1 = QImage(rgba.data, pw, ph, 4 * pw, QImage.Format_RGBA8888)
        self._mask_overlay_cache = QPixmap.fromImage(q1.copy())

        gray_rgb = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
        q2 = QImage(gray_rgb.data, pw, ph, 3 * pw, QImage.Format_RGB888)
        self._mask_pure_cache = QPixmap.fromImage(q2.copy())

        self._mask_cache_version = self._mask_version
        self._mask_cache_time = now
        return self._mask_pure_cache if pure else self._mask_overlay_cache

    def _add_picked_point(self, ix: int, iy: int) -> None:
        if len(self._picked_points) >= 4:
            return
        if self._image is not None:
            h, w = self._image.shape[:2]
            ix = max(0, min(ix, w - 1))
            iy = max(0, min(iy, h - 1))
        self._picked_points.append((ix, iy))
        n = len(self._picked_points)
        self.status_message.emit(f"已选角点 {n}/4" + ("，可确认校正" if n == 4 else ""))
        self.update()

    def get_image_name(self) -> str:
        return "image"

    def set_grid(self, cols: int, rows: int) -> None:
        """Rebuild the grid with the given M×N dimensions."""
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        self.grid_manager.update(w, h, cols, rows)
        self._fit_to_window()
        self.update()
        self.status_message.emit(f"已分块: {cols}×{rows} = {cols * rows} 个小方块")

    def set_grid_auto(self, target_cells: int = 100) -> None:
        """Rebuild the grid with automatically adjusted dimensions to keep cells square."""
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        cols, rows = self.grid_manager.update_auto(w, h, target_cells)
        self._fit_to_window()
        self.update()
        self.status_message.emit(f"已自适应分块: {cols}×{rows} = {cols * rows} 个小方块")

    def fit_to_window(self) -> None:
        """Scale and center the image inside the widget."""
        self._fit_to_window()
        self.update()

    def actual_size(self) -> None:
        """Display the image at its actual size (100% scale)."""
        if self._pixmap is None:
            return
        self._scale = 1.0
        self._offset_x = (self.width() - self._pixmap.width()) / 2.0
        self._offset_y = (self.height() - self._pixmap.height()) / 2.0
        self.update()

    def clear_annotations(self) -> None:
        """Remove all annotations."""
        self._labels.clear()
        self.update()
        self.cells_annotated.emit(0)
        self.status_message.emit("已清除所有标注")

    def apply_auto_labels(self, labels: Dict[Tuple[int, int], str]) -> None:
        """Apply labels produced by the automatic detector and refresh."""
        self._labels = dict(labels)
        self.update()
        self.cells_annotated.emit(len(self._labels))

    def set_stamp_label(self, label_name: str) -> None:
        """Equip a label as the current stamp tool."""
        self._stamp_label_name = label_name
        self.update()

    def clear_stamp_label(self) -> None:
        """Unequip the stamp tool."""
        self._stamp_label_name = None
        self.update()

    def stamp_at(self, widget_x: float, widget_y: float) -> None:
        """Stamp a single cell at the given widget coordinates with the equipped label."""
        if self._image is None or self._stamp_label_name is None:
            return
        ix, iy = self._widget_to_image(widget_x, widget_y)
        cell = self.grid_manager.get_cell_at(ix, iy)
        if cell is None:
            return
        key = (cell.row, cell.col)
        if self._labels.get(key) == self._stamp_label_name:
            return
        self._labels[key] = self._stamp_label_name
        self.update()
        self.cells_annotated.emit(len(self._labels))

    def _build_pixmap(self) -> None:
        # The pixmap is built from the small preview image only, never from
        # the full-resolution original (memory guard for huge panoramas).
        if self._preview_image is None:
            self._pixmap = None
            return
        rgb = cv2.cvtColor(self._preview_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image.copy())

    def _fit_to_window(self) -> None:
        """Scale and center the image inside the widget."""
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

    def _clamp_view(self) -> None:
        """Keep at least a 60px strip of the image inside the widget.

        Without this, one hard middle-button drag flings the image completely
        off-screen and it looks like it "disappeared".
        """
        if self._pixmap is None:
            return
        disp_w = self._pixmap.width() * self._scale
        disp_h = self._pixmap.height() * self._scale
        margin = 60
        max_x = self.width() - margin
        max_y = self.height() - margin
        self._offset_x = min(max(self._offset_x, margin - disp_w), max_x)
        self._offset_y = min(max(self._offset_y, margin - disp_h), max_y)

    def _min_scale(self) -> float:
        """Never zoom out beyond a recognizable thumbnail (>=80px long side)."""
        if self._pixmap is None:
            return 0.02
        return max(0.02, 80.0 / max(self._pixmap.width(), self._pixmap.height()))

    def _widget_to_image(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert widget coordinates to ORIGINAL image coordinates."""
        s = self._scale * self._preview_ratio
        if s <= 1e-12:
            return 0, 0
        ix = int((wx - self._offset_x) / s)
        iy = int((wy - self._offset_y) / s)
        return ix, iy

    def _image_to_widget(self, ix: float, iy: float) -> QPoint:
        """Convert ORIGINAL image coordinates to widget coordinates."""
        s = self._scale * self._preview_ratio
        wx = self._offset_x + ix * s
        wy = self._offset_y + iy * s
        return QPoint(int(wx), int(wy))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fill background.
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._pixmap is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "请导入图片")
            return

        # Draw image (or the pure mask in mask-view mode 1).
        target = QRect(
            int(self._offset_x),
            int(self._offset_y),
            int(self._pixmap.width() * self._scale),
            int(self._pixmap.height() * self._scale),
        )
        if self.mask_view_mode == 1 and self._mask is not None and not self.clean_view:
            painter.drawPixmap(target, self._mask_pixmap(), self._pixmap.rect())
        else:
            painter.drawPixmap(target, self._pixmap, self._pixmap.rect())
            # Red translucent mask overlay (mode 0).
            if self.mask_view_mode == 0 and self._mask is not None and not self.clean_view:
                overlay = self._mask_overlay_pixmap()
                if overlay is not None:
                    painter.drawPixmap(target, overlay, self._pixmap.rect())

        # Draw label overlays (hidden in clean view).
        if not self.clean_view:
            for cell in self.grid_manager.cells:
                label_name = self._labels.get((cell.row, cell.col))
                if label_name is None:
                    continue
                label = self.label_manager.get_label(label_name)
                if label is None:
                    continue
                top_left = self._image_to_widget(cell.x1, cell.y1)
                bottom_right = self._image_to_widget(cell.x2, cell.y2)
                rect = QRect(top_left, bottom_right)
                painter.fillRect(rect, QBrush(QColor(*label.color)))

        # Draw hover highlight (semi-transparent white fill) for the cell under cursor.
        if self._hover_cell is not None:
            hover_top_left = self._image_to_widget(self._hover_cell.x1, self._hover_cell.y1)
            hover_bottom_right = self._image_to_widget(
                self._hover_cell.x2, self._hover_cell.y2
            )
            hover_rect = QRect(hover_top_left, hover_bottom_right)
            painter.fillRect(hover_rect, QBrush(QColor(255, 255, 255, 60)))

        # Grid borders + indices can be hidden while brushing.
        if self.grid_visible:
            # Draw grid borders with a black shadow + white inner line.
            self._draw_grid_borders(painter)

        # Draw a yellow border around the hovered cell.
        if self._hover_cell is not None:
            hover_pen = QPen(QColor(255, 235, 59))
            hover_pen.setWidth(2)
            painter.setPen(hover_pen)
            painter.setBrush(QBrush())
            painter.drawRect(hover_rect)

        if self.grid_visible:
            # Draw row/column indices inside each cell.
            self._draw_cell_indices(painter)

        # Draw picked tray-corner points and the quad preview (topmost).
        self._draw_picked_points(painter)

    def _draw_picked_points(self, painter: QPainter) -> None:
        """Draw numbered corner markers + quad preview for tray selection."""
        pts = [self._image_to_widget(ix, iy) for ix, iy in self._picked_points]
        if not pts:
            return

        # Quad outline + translucent fill once complete.
        if len(pts) >= 2:
            pen = QPen(QColor(255, 235, 59))
            pen.setWidth(3)
            painter.setPen(pen)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            if len(pts) == 4:
                painter.drawLine(pts[3], pts[0])
                painter.setBrush(QBrush(QColor(255, 235, 59, 40)))
                painter.drawPolygon(QPolygon(pts))

        # Numbered markers.
        font = QFont("Microsoft YaHei", 10, QFont.Bold)
        painter.setFont(font)
        for i, p in enumerate(pts, 1):
            painter.setBrush(QBrush(QColor(255, 60, 60)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(p, 10, 10)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(p.x() - 4, p.y() + 5, str(i))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MiddleButton, Qt.RightButton) and self._image is not None \
                and not self._point_pick_mode:
            # Pan with middle or right button drag (clamped view).
            self._pan_start = event.position()
            self._pan_offset_x = self._offset_x
            self._pan_offset_y = self._offset_y
            event.accept()
        elif self._point_pick_mode and self._image is not None:
            # Tray-corner picking takes priority over brushing.
            if event.button() == Qt.LeftButton:
                ix, iy = self._widget_to_image(event.position().x(), event.position().y())
                self._add_picked_point(ix, iy)
                event.accept()
            elif event.button() == Qt.RightButton:
                self.undo_point()
                event.accept()
        elif event.button() == Qt.LeftButton and self._image is not None:
            # Begin a brush stroke (paint or erase per current mode).
            event.accept()
            ix, iy = self._widget_to_image(event.position().x(), event.position().y())
            self._begin_stroke(ix, iy, self.brush_erase)
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            super().mouseMoveEvent(event)
            return

        if (event.buttons() & (Qt.MiddleButton | Qt.RightButton)) and hasattr(self, "_pan_start"):
            # Pan the image (clamped: the image can never leave the view).
            delta = event.position() - self._pan_start
            self._offset_x = self._pan_offset_x + delta.x()
            self._offset_y = self._pan_offset_y + delta.y()
            self._clamp_view()
            self.update()
            event.accept()
            return

        # Extend an active brush stroke.
        if self._current_stroke is not None and (
            event.buttons() & (Qt.LeftButton | Qt.RightButton)
        ):
            ix, iy = self._widget_to_image(event.position().x(), event.position().y())
            self._extend_stroke(ix, iy)
            event.accept()
            return

        event.accept()
        ix, iy = self._widget_to_image(event.position().x(), event.position().y())
        cell = self.grid_manager.get_cell_at(ix, iy)

        # Update hover highlight whenever the cursor moves over a different cell.
        if cell != self._hover_cell:
            self._hover_cell = cell
            self.update()

        if cell is not None:
            label = self._labels.get((cell.row, cell.col), "未标注")
            mode = " [已装备，点击标注]" if self._stamp_label_name else ""
            score = self.cell_scores.get((cell.row, cell.col))
            score_text = f" | 置信度: {score:.2f}" if score is not None else ""
            self.status_message.emit(
                f"位置: ({cell.row}, {cell.col}) 标签: {label}{mode}{score_text}"
            )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._end_stroke()
        super().mouseReleaseEvent(event)

    def _draw_grid_borders(self, painter: QPainter) -> None:
        """Draw each cell border as a dark shadow + bright inner line."""
        for cell in self.grid_manager.cells:
            top_left = self._image_to_widget(cell.x1, cell.y1)
            bottom_right = self._image_to_widget(cell.x2, cell.y2)
            rect = QRect(top_left, bottom_right)

            # Outer dark shadow for contrast on bright backgrounds.
            shadow_pen = QPen(QColor(0, 0, 0, 180))
            shadow_pen.setWidth(4)
            painter.setPen(shadow_pen)
            painter.setBrush(QBrush())
            painter.drawRect(rect)

            # Inner bright line for contrast on dark backgrounds.
            light_pen = QPen(QColor(255, 255, 255, 220))
            light_pen.setWidth(2)
            painter.setPen(light_pen)
            painter.drawRect(rect)

    def _draw_cell_indices(self, painter: QPainter) -> None:
        """Draw row,col text in the top-left corner of each cell.

        Skipped entirely for cells that are too small on screen - otherwise
        dense grids smear into unreadable stacked text.
        """
        for cell in self.grid_manager.cells:
            top_left = self._image_to_widget(cell.x1, cell.y1)
            bottom_right = self._image_to_widget(cell.x2, cell.y2)
            cell_w = bottom_right.x() - top_left.x()
            cell_h = bottom_right.y() - top_left.y()
            if cell_w < 44 or cell_h < 30:
                continue

            font_size = max(6, min(int(10 * self._scale), int(cell_h * 0.3)))
            font = QFont("Microsoft YaHei", font_size)
            painter.setFont(font)
            text = f"{cell.row},{cell.col}"
            # Draw black outline/shadow.
            painter.setPen(QColor(0, 0, 0))
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                painter.drawText(top_left.x() + 3 + dx, top_left.y() + 12 + dy, text)
            # Draw white text.
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(top_left.x() + 3, top_left.y() + 12, text)

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Simple zoom with mouse wheel, centered on cursor.
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        new_scale = self._scale * factor
        if new_scale < self._min_scale() or new_scale > 20:
            return

        mouse_pos = event.position()
        # Keep the point under the cursor stable (original-image space).
        ix, iy = self._widget_to_image(mouse_pos.x(), mouse_pos.y())
        self._scale = new_scale
        s = self._scale * self._preview_ratio
        self._offset_x = mouse_pos.x() - ix * s
        self._offset_y = mouse_pos.y() - iy * s
        self._clamp_view()
        self.update()
