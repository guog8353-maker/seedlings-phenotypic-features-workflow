"""Feature analysis tab: interactive pixel-level feature map explorer.

Left: controls (feature list, overlay alpha, parameter sliders, ROI, save).
Center: original image viewer (with rubber-band ROI) | feature map viewer.
All computation happens at ORIGINAL resolution; viewers show downscaled copies.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .feature_maps import FeatureParams, available_features, compute_feature_map
from .image_io import imwrite_unicode


def _np_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    q = QImage(img.copy().data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


def _fit_scale(pw: int, ph: int, ww: int, wh: int) -> float:
    return min(ww / max(pw, 1), wh / max(ph, 1), 1.0) if True else 1.0


class ImageViewer(QWidget):
    """Fit-to-window image viewer with optional rubber-band ROI."""

    roi_selected = Signal(tuple)  # (x1, y1, x2, y2) in displayed-image coords

    def __init__(self, parent=None, roi_enabled: bool = False):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._roi_enabled = roi_enabled
        self._roi_rect: Optional[QRect] = None
        self._drag_start: Optional[QPoint] = None
        self.setMinimumSize(300, 300)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        self._roi_rect = None
        self.update()

    def clear_roi(self) -> None:
        self._roi_rect = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 35))
        if self._pixmap is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "无图像")
            return
        s = _fit_scale(self._pixmap.width(), self._pixmap.height(),
                       self.width(), self.height())
        tw = int(self._pixmap.width() * s)
        th = int(self._pixmap.height() * s)
        ox = int((self.width() - tw) / 2)
        oy = int((self.height() - th) / 2)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(QRect(ox, oy, tw, th), self._pixmap, self._pixmap.rect())

        if self._roi_rect is not None:
            painter.setPen(QPen(QColor(255, 235, 59), 2))
            painter.drawRect(self._roi_rect)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._roi_enabled and event.button() == Qt.LeftButton and self._pixmap:
            self._drag_start = event.position().toPoint()
            self._roi_rect = None

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            self._roi_rect = QRect(self._drag_start, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None and self._roi_rect is not None:
            rect = self._roi_rect
            if rect.width() > 8 and rect.height() > 8:
                # Convert widget coords to displayed-image coords.
                s = _fit_scale(self._pixmap.width(), self._pixmap.height(),
                               self.width(), self.height())
                ox = (self.width() - self._pixmap.width() * s) / 2
                oy = (self.height() - self._pixmap.height() * s) / 2
                x1 = int((rect.left() - ox) / s)
                y1 = int((rect.top() - oy) / s)
                x2 = int((rect.right() - ox) / s)
                y2 = int((rect.bottom() - oy) / s)
                self.roi_selected.emit((x1, y1, x2, y2))
        self._drag_start = None


class FeatureAnalysisTab(QWidget):
    """Interactive feature-map explorer for guttation research."""

    def __init__(self, get_image: Callable[[], Optional[np.ndarray]], parent=None):
        super().__init__(parent)
        self._get_image = get_image
        self._current_map: Optional[np.ndarray] = None
        self._roi: Optional[Tuple[int, int, int, int]] = None  # original coords
        self._setup_ui()

    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        controls = QWidget()
        controls.setFixedWidth(230)
        cl = QVBoxLayout(controls)
        cl.setAlignment(Qt.AlignTop)

        cl.addWidget(QLabel("<b>特征选择</b>"))
        self.feature_list = QListWidget()
        for name in available_features():
            self.feature_list.addItem(name)
        self.feature_list.setCurrentRow(0)
        self.feature_list.currentTextChanged.connect(self._on_feature_changed)
        cl.addWidget(self.feature_list)

        cl.addWidget(QLabel("<b>叠加透明度</b>"))
        self.overlay_slider = QSlider(Qt.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(55)
        self.overlay_slider.valueChanged.connect(self._refresh_view)
        cl.addWidget(self.overlay_slider)

        cl.addWidget(QLabel("<b>参数</b>"))
        self.delta_slider, self.delta_label = self._add_slider(cl, "亮度差阈值", 5, 80, 25)
        self.sigma_slider, self.sigma_label = self._add_slider(cl, "高斯sigma", 3, 40, 12)
        self.area_slider, self.area_label = self._add_slider(cl, "Blob最小面积", 4, 500, 12)
        self.circ_slider, self.circ_label = self._add_slider(cl, "圆度下限", 20, 90, 45)

        self.roi_btn = QPushButton("框选 ROI 后仅分析该区域")
        self.roi_btn.setCheckable(True)
        self.roi_btn.toggled.connect(self._on_roi_toggled)
        cl.addWidget(self.roi_btn)

        clear_roi_btn = QPushButton("清除 ROI（整图分析）")
        clear_roi_btn.clicked.connect(self._clear_roi)
        cl.addWidget(clear_roi_btn)

        save_btn = QPushButton("保存全部特征图 + 报告")
        save_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        save_btn.clicked.connect(self._save_all)
        cl.addWidget(save_btn)

        layout.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.orig_view = ImageViewer(roi_enabled=True)
        self.orig_view.roi_selected.connect(self._on_roi_selected)
        self.feat_view = ImageViewer()
        splitter.addWidget(self._titled("原图", self.orig_view))
        splitter.addWidget(self._titled("特征图", self.feat_view))
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

    def _titled(self, title: str, viewer: ImageViewer) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel(f"<b>{title}</b>"))
        v.addWidget(viewer)
        return w

    def _add_slider(self, parent_layout, label, lo, hi, val):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        value_label = QLabel(str(val))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(val)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        slider.valueChanged.connect(self._on_params_changed)
        row.addWidget(slider)
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider, value_label

    # ------------------------------------------------------------------
    def _params(self) -> FeatureParams:
        return FeatureParams(
            bright_delta=self.delta_slider.value(),
            gaussian_sigma=float(self.sigma_slider.value()),
            blob_min_area=self.area_slider.value(),
            min_circularity=self.circ_slider.value() / 100.0,
        )

    def _analysis_image(self) -> Optional[np.ndarray]:
        img = self._get_image()
        if img is None:
            return None
        if self._roi is not None:
            x1, y1, x2, y2 = self._roi
            h, w = img.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                return img[y1:y2, x1:x2]
        return img

    def _on_feature_changed(self, _name: str) -> None:
        self._recompute()

    def _on_params_changed(self, _v) -> None:
        self._recompute()

    def _on_roi_toggled(self, checked: bool) -> None:
        self.orig_view._roi_enabled = checked
        if not checked:
            self._clear_roi()

    def _on_roi_selected(self, rect) -> None:
        img = self._get_image()
        if img is None:
            return
        # Viewer displays the image downscaled; map back to original coords.
        disp = self._display_image()
        if disp is None:
            return
        ratio = img.shape[1] / disp.shape[1]
        x1, y1, x2, y2 = rect
        self._roi = (
            int(x1 * ratio), int(y1 * ratio), int(x2 * ratio), int(y2 * ratio)
        )
        self._recompute()

    def _clear_roi(self) -> None:
        self._roi = None
        self.orig_view.clear_roi()
        if self.roi_btn.isChecked():
            self.roi_btn.setChecked(False)
        self._recompute()

    def _display_image(self) -> Optional[np.ndarray]:
        img = self._analysis_image()
        if img is None:
            return None
        if max(img.shape[:2]) > 2048:
            r = 2048 / max(img.shape[:2])
            return cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
        return img

    # ------------------------------------------------------------------
    def refresh_image(self) -> None:
        """Called by the main window after a new image is loaded."""
        self._roi = None
        self.orig_view.clear_roi()
        disp = self._display_image()
        self.orig_view.set_pixmap(_np_to_qpixmap(disp) if disp is not None else None)
        self._recompute()

    def _recompute(self) -> None:
        img = self._analysis_image()
        if img is None:
            return
        name = self.feature_list.currentItem().text() if self.feature_list.currentItem() else None
        if not name:
            return
        fmap = compute_feature_map(name, img, self._params())
        self._current_map = fmap
        self._refresh_view()

    def _refresh_view(self) -> None:
        disp = self._display_image()
        if disp is not None:
            self.orig_view.set_pixmap(_np_to_qpixmap(disp))
        if self._current_map is None or disp is None:
            return

        m = self._current_map
        if m.shape[:2] != disp.shape[:2]:
            m = cv2.resize(m, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_LINEAR)

        alpha = self.overlay_slider.value() / 100.0
        heat = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        if alpha <= 0.0:
            out = disp
        elif alpha >= 1.0:
            out = heat
        else:
            out = cv2.addWeighted(disp, 1.0 - alpha, heat, alpha, 0)
        self.feat_view.set_pixmap(_np_to_qpixmap(out))

    # ------------------------------------------------------------------
    def _save_all(self) -> None:
        img = self._analysis_image()
        if img is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", "feature_output")
        if not out_dir:
            return
        out = Path(out_dir)
        params = self._params()

        report = {
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "roi": self._roi,
            "features": {},
        }
        saved = 0
        for name in available_features():
            fmap = compute_feature_map(name, img, params)
            if fmap is None:
                continue
            fname = f"feature_{name.lower()}.png"
            if imwrite_unicode(str(out / fname), (np.clip(fmap, 0, 1) * 255).astype(np.uint8)):
                saved += 1
            idx = int(np.argmax(fmap))
            y, x = divmod(idx, fmap.shape[1])
            report["features"][name] = {
                "max_response": round(float(fmap.max()), 4),
                "max_position": [int(x), int(y)],
                "mean_response": round(float(fmap.mean()), 4),
                "pixels_above_0.5": int((fmap > 0.5).sum()),
            }

        import json
        with open(out / "feature_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        QMessageBox.information(
            self, "保存完成", f"已保存 {saved} 张特征图 + feature_report.json\n{out}"
        )
