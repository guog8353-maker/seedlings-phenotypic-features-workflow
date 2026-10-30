"""Main application window: workflow-driven guttation annotation platform.

Workflow:
  1) choose a workspace folder of images;
  2) mark 4 tray corners -> perspective-rectify (auto-saved);
  3) brush over droplet regions (mask overlay, adjustable radius);
  4) confirm grid (presets / custom / remembered);
  5) stash state and continue, or export the dataset now / in batch.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .error_reporting import StageTracker, log_exception, show_error_dialog
from .exporter import export_dataset
from .guttation_detector import GuttationConfig, GuttationDetector
from .image_canvas import ImageCanvas
from .image_io import imwrite_unicode
from .label_manager import Label, LabelManager
from .rectify import rectify_image, save_rectify_info
from . import state_store


class ExportWorker(QThread):
    """Run a single-image export in a background thread."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, canvas, output_dir: str, image_name: str,
                 mask_areas: Optional[Dict] = None, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.output_dir = output_dir
        self.image_name = image_name
        self.mask_areas = mask_areas

    def run(self) -> None:
        StageTracker.set("EXPORT")
        try:
            image = self.canvas.get_image()
            if image is None:
                self.error.emit("没有可导出的图片")
                return
            csv_path = export_dataset(
                image=image,
                grid_manager=self.canvas.grid_manager,
                labels=self.canvas.labels,
                output_dir=self.output_dir,
                image_name=self.image_name,
                mask_areas=self.mask_areas,
            )
            if csv_path is None:
                self.error.emit("导出失败")
            else:
                self.finished.emit(csv_path)
        except Exception as exc:  # noqa: BLE001
            tb = log_exception("EXPORT", exc, context=self.output_dir)
            StageTracker.fail("EXPORT")
            self.error.emit(f"{type(exc).__name__}: {exc}\n\nTRACEBACK:\n{tb}")


class MainWindow(QMainWindow):
    """Workflow-driven annotation main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("水稻吐水标注平台")
        self.setMinimumSize(1280, 800)

        self._app_dir = Path(__file__).resolve().parent.parent
        self._config = state_store.load_config(self._app_dir)

        self._workspace = self._config.get("workspace", "")
        self._images: List[Path] = []
        self._current_index = -1
        self._current_image_path = ""
        self._rectify_info: Optional[Dict] = None
        self._rectify_skipped = False
        self._last_detector: Optional[GuttationDetector] = None

        self.label_manager = LabelManager()
        self.label_manager.add_label(Label("1", "吐水", color=(255, 0, 0, 100)))
        self.label_manager.add_label(Label("0", "未吐水", color=(0, 255, 0, 100)))
        self.label_manager.set_current("1")

        self.canvas = ImageCanvas(self.label_manager)
        self.canvas.mask_threshold = float(self._config.get("mask_threshold", 0.01))
        self.canvas.status_message.connect(self._show_status)
        self.canvas.mask_changed.connect(self._on_mask_changed)

        self._init_ui()
        # Apply persisted widget values to the canvas itself, not only to the
        # controls; otherwise a custom brush radius/grid visibility is ignored
        # until the user toggles the corresponding control.
        self.canvas.set_brush(self.radius_spin.value(), self.erase_btn.isChecked())
        self.canvas.set_grid_visible(self.grid_visible_check.isChecked())
        self._init_menu()
        self._apply_styles()

        if self._workspace and Path(self._workspace).exists():
            self._scan_workspace(self._workspace)

    # ==================================================================
    # UI construction
    # ==================================================================
    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        sidebar = QWidget()
        sidebar.setMinimumWidth(270)
        sb = QVBoxLayout(sidebar)
        sb.setAlignment(Qt.AlignTop)

        # ---- ① Workspace ------------------------------------------------
        sb.addWidget(QLabel("<b>① 工作文件夹</b>"))
        ws_btn = QPushButton("选择工作文件夹")
        ws_btn.clicked.connect(self._choose_workspace)
        sb.addWidget(ws_btn)
        self.ws_label = QLabel(self._workspace or "（未选择）")
        self.ws_label.setWordWrap(True)
        self.ws_label.setStyleSheet("color: #888;")
        sb.addWidget(self.ws_label)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("‹ 上一张")
        self.prev_btn.clicked.connect(lambda: self._navigate(-1))
        nav.addWidget(self.prev_btn)
        self.nav_label = QLabel("0/0")
        self.nav_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.nav_label)
        self.next_btn = QPushButton("下一张 ›")
        self.next_btn.clicked.connect(lambda: self._navigate(1))
        nav.addWidget(self.next_btn)
        sb.addLayout(nav)
        self.img_name_label = QLabel("")
        self.img_name_label.setStyleSheet("color: #888;")
        self.img_name_label.setWordWrap(True)
        sb.addWidget(self.img_name_label)

        # ---- ② Tray rectification ---------------------------------------
        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>② 秧盘区域校正</b>"))
        self.pick_btn = QPushButton("选择秧盘区域（点4个角点）")
        self.pick_btn.setCheckable(True)
        self.pick_btn.toggled.connect(self._toggle_pick_mode)
        sb.addWidget(self.pick_btn)

        pick_row = QHBoxLayout()
        undo_pt = QPushButton("撤销点")
        undo_pt.clicked.connect(lambda: self.canvas.undo_point())
        pick_row.addWidget(undo_pt)
        clear_pt = QPushButton("重选")
        clear_pt.clicked.connect(lambda: self.canvas.clear_points())
        pick_row.addWidget(clear_pt)
        sb.addLayout(pick_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("校正尺寸:"))
        self.rect_w_edit = QLineEdit()
        self.rect_w_edit.setPlaceholderText("宽(自动)")
        size_row.addWidget(self.rect_w_edit)
        self.rect_h_edit = QLineEdit()
        self.rect_h_edit.setPlaceholderText("高(自动)")
        size_row.addWidget(self.rect_h_edit)
        sb.addLayout(size_row)

        rect_row = QHBoxLayout()
        confirm_btn = QPushButton("确认校正")
        confirm_btn.setStyleSheet("background-color: #1565c0; color: white;")
        confirm_btn.clicked.connect(self._confirm_rectify)
        rect_row.addWidget(confirm_btn)
        skip_btn = QPushButton("跳过校正")
        skip_btn.clicked.connect(self._skip_rectify)
        rect_row.addWidget(skip_btn)
        sb.addLayout(rect_row)
        self.rect_status = QLabel("未校正")
        self.rect_status.setStyleSheet("color: #888;")
        sb.addWidget(self.rect_status)

        # ---- ③ Brush annotation ------------------------------------------
        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>③ 笔刷标注（左键涂抹；橡皮模式擦除；右键/中键平移）</b>"))
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("笔刷半径:"))
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(2, 400)
        self.radius_spin.setValue(int(self._config.get("brush_radius", 25)))
        self.radius_spin.valueChanged.connect(self._apply_brush)
        radius_row.addWidget(self.radius_spin)
        sb.addLayout(radius_row)
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(2, 400)
        self.radius_slider.setValue(self.radius_spin.value())
        self.radius_slider.valueChanged.connect(self.radius_spin.setValue)
        self.radius_spin.valueChanged.connect(self.radius_slider.setValue)
        sb.addWidget(self.radius_slider)

        brush_row = QHBoxLayout()
        self.erase_btn = QPushButton("画笔模式")
        self.erase_btn.setCheckable(True)
        self.erase_btn.toggled.connect(self._toggle_erase)
        brush_row.addWidget(self.erase_btn)
        undo_stroke_btn = QPushButton("撤销笔画")
        undo_stroke_btn.clicked.connect(self._undo_stroke)
        brush_row.addWidget(undo_stroke_btn)
        clear_mask_btn = QPushButton("清空掩膜")
        clear_mask_btn.clicked.connect(lambda: self.canvas.clear_mask())
        brush_row.addWidget(clear_mask_btn)
        sb.addLayout(brush_row)

        mask_view_row = QHBoxLayout()
        mask_view_row.addWidget(QLabel("掩膜视图:"))
        self.mask_view_combo = QComboBox()
        self.mask_view_combo.addItem("半透明叠加", 0)
        self.mask_view_combo.addItem("纯掩膜（黑白）", 1)
        self.mask_view_combo.addItem("隐藏", 2)
        self.mask_view_combo.currentIndexChanged.connect(
            lambda i: self.canvas.set_mask_view_mode(self.mask_view_combo.currentData())
        )
        mask_view_row.addWidget(self.mask_view_combo)
        sb.addLayout(mask_view_row)

        self.clean_view_check = QCheckBox("只显示底图（隐藏掩膜与标注）")
        self.clean_view_check.setToolTip("滚轮缩放，右键拖拽平移，左键笔刷")
        self.clean_view_check.toggled.connect(self._toggle_clean_view)
        sb.addWidget(self.clean_view_check)

        auto_btn = QPushButton("自动预标注（多特征检测）")
        auto_btn.clicked.connect(self._auto_annotate)
        sb.addWidget(auto_btn)

        # ---- ④ Grid --------------------------------------------------------
        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>④ 分块</b>"))
        preset_row = QHBoxLayout()
        for cols, rows in self._config.get("presets", [[36, 18], [41, 20], [48, 28]]):
            b = QPushButton(f"{cols}×{rows}")
            b.clicked.connect(lambda _=False, c=cols, r=rows: self._apply_preset(c, r))
            preset_row.addWidget(b)
        sb.addLayout(preset_row)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("列(M):"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 200)
        self.cols_spin.setValue(int(self._config.get("cols", 36)))
        grid_row.addWidget(self.cols_spin)
        grid_row.addWidget(QLabel("行(N):"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 200)
        self.rows_spin.setValue(int(self._config.get("rows", 18)))
        grid_row.addWidget(self.rows_spin)
        swap_btn = QPushButton("交换")
        swap_btn.setToolTip("交换行列数")
        swap_btn.clicked.connect(self._swap_grid)
        grid_row.addWidget(swap_btn)
        sb.addLayout(grid_row)

        grid_row2 = QHBoxLayout()
        apply_grid_btn = QPushButton("应用分块")
        apply_grid_btn.clicked.connect(self._apply_grid)
        grid_row2.addWidget(apply_grid_btn)
        self.grid_visible_check = QCheckBox("显示网格线")
        self.grid_visible_check.setChecked(bool(self._config.get("show_grid", True)))
        self.grid_visible_check.toggled.connect(
            lambda v: self.canvas.set_grid_visible(v)
        )
        grid_row2.addWidget(self.grid_visible_check)
        sb.addLayout(grid_row2)
        self.grid_info_label = QLabel("网格信息: 未设置")
        self.grid_info_label.setStyleSheet("color: #888;")
        sb.addWidget(self.grid_info_label)

        # ---- ⑤ Save / export -----------------------------------------------
        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>⑤ 保存与导出</b>"))
        stash_btn = QPushButton("暂存并下一张")
        stash_btn.setStyleSheet("background-color: #1565c0; color: white;")
        stash_btn.clicked.connect(self._stash_and_next)
        sb.addWidget(stash_btn)

        export_btn = QPushButton("导出当前数据集")
        export_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        export_btn.clicked.connect(self._export_current)
        sb.addWidget(export_btn)

        export_all_btn = QPushButton("批量导出全部暂存（按多分块）")
        export_all_btn.clicked.connect(self._export_all_stashed)
        sb.addWidget(export_all_btn)

        sb.addSpacing(6)
        sb.addWidget(QLabel("多分块策略（逗号分隔 列x行，可增删）:"))
        self.grids_edit = QLineEdit(self._config.get("multi_grids_text", "36x18,41x20,48x28"))
        sb.addWidget(self.grids_edit)
        multi_btn = QPushButton("按多分块导出当前图片")
        multi_btn.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold;")
        multi_btn.clicked.connect(self._export_multi_current)
        sb.addWidget(multi_btn)

        sb.addWidget(QLabel("输出目录:"))
        self.output_edit = QLineEdit(self._config.get("output_dir", "") or "output")
        sb.addWidget(self.output_edit)
        browse_btn = QPushButton("浏览输出目录")
        browse_btn.clicked.connect(self._browse_output_dir)
        sb.addWidget(browse_btn)

        sb.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(scroll)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 980])

        self.statusBar().showMessage("就绪：请选择工作文件夹")

    def _init_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")

        open_ws = QAction("打开工作文件夹", self)
        open_ws.setShortcut("Ctrl+O")
        open_ws.triggered.connect(self._choose_workspace)
        file_menu.addAction(open_ws)

        file_menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1e1e1e; }
            QWidget { font-size: 14px; color: #eeeeee; }
            QPushButton {
                background-color: #3c3c3c; border: 1px solid #555;
                padding: 8px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:checked { background-color: #1565c0; color: white; }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #2a2a2a; border: 1px solid #555; padding: 4px;
            }
            QLabel { color: #eeeeee; }
        """
        )

    # ==================================================================
    # Workspace management
    # ==================================================================
    def _choose_workspace(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择工作文件夹", self._workspace or str(Path.home())
        )
        if directory:
            self._scan_workspace(directory)

    def _scan_workspace(self, directory: str) -> None:
        self._workspace = directory
        self._config["workspace"] = directory
        state_store.save_config(self._app_dir, self._config)
        self._images = state_store.list_workspace_images(directory)
        self.ws_label.setText(directory)
        if not self._images:
            self.nav_label.setText("0/0")
            QMessageBox.information(self, "提示", "该文件夹中没有图片文件")
            return
        self._load_image_at(0)

    def _navigate(self, delta: int) -> None:
        if not self._images:
            return
        target = self._current_index + delta
        if 0 <= target < len(self._images):
            self._stash_current(silent=True)  # auto-stash before switching
            self._load_image_at(target)

    def _load_image_at(self, index: int) -> None:
        path = self._images[index]
        self._current_index = index
        name = path.stem
        self._current_image_path = str(path)
        self.nav_label.setText(f"{index + 1}/{len(self._images)}")
        self.img_name_label.setText(path.name)

        # Reset per-image state.
        self.canvas.clear_points()
        self.canvas.set_point_pick_mode(False)
        self.pick_btn.setChecked(False)
        self._last_detector = None
        self._rectify_skipped = False

        state = state_store.load_state(self._workspace, name) if self._workspace else None

        try:
            if state and state.get("rectified"):
                # Already rectified before: load the saved rectified image.
                rect_path = state_store.rectified_path(self._workspace, name)
                img = None
                if rect_path.exists():
                    from .image_io import imread_unicode
                    img = imread_unicode(str(rect_path))
                if img is not None:
                    self.canvas.set_image(img)
                    r = state.get("rectify", {})
                    self._rectify_info = {
                        "pts": r.get("quad_points_original"),
                        "H": np.array(r["H_original_to_rectified"]) if r.get("H_original_to_rectified") else None,
                        "out_size": tuple(r.get("out_size") or (img.shape[1], img.shape[0])),
                        "src_path": str(path),
                    }
                    self.rect_status.setText(f"已校正 {img.shape[1]}×{img.shape[0]}")
                else:
                    self.canvas.load_image(str(path))
                    self._rectify_info = None
                    self.rect_status.setText("未校正")
            else:
                self.canvas.load_image(str(path))
                self._rectify_info = None
                self.rect_status.setText("未校正")
        except Exception as e:
            show_error_dialog(self, "加载失败", "LOAD_IMAGE", e, image_path=str(path))
            return

        # Restore grid + mask from state, else defaults from config.
        grid = (state or {}).get("grid") or {}
        cols = int(grid.get("cols", self._config.get("cols", 36)))
        rows = int(grid.get("rows", self._config.get("rows", 18)))
        self.cols_spin.setValue(cols)
        self.rows_spin.setValue(rows)
        self._apply_grid()

        mask = (state or {}).get("mask")
        if mask is not None:
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            self.canvas.set_mask(mask)
        else:
            self.canvas.clear_mask()
        self._on_mask_changed()

    # ==================================================================
    # Rectification
    # ==================================================================
    def _toggle_pick_mode(self, checked: bool) -> None:
        self.canvas.set_point_pick_mode(checked)
        self.statusBar().showMessage(
            "选点模式：左键点选秧盘 4 个角点，右键撤销" if checked else "已退出选点模式"
        )

    def _confirm_rectify(self) -> None:
        pts = self.canvas.picked_points
        if len(pts) < 4:
            QMessageBox.warning(self, "提示", f"还需要 {4 - len(pts)} 个角点才能校正")
            return
        image = self.canvas.get_image()
        if image is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return

        out_size = None
        try:
            w_text, h_text = self.rect_w_edit.text().strip(), self.rect_h_edit.text().strip()
            if w_text and h_text:
                out_size = (int(w_text), int(h_text))
        except ValueError:
            QMessageBox.warning(self, "提示", "校正尺寸必须是整数（或留空自动）")
            return

        StageTracker.set("RECTIFY_IMAGE")
        try:
            rectified, H, size = rectify_image(image, pts, out_size=out_size)
        except Exception as e:
            StageTracker.fail("RECTIFY_IMAGE")
            log_exception("RECTIFY_IMAGE", e)
            show_error_dialog(self, "校正失败", "RECTIFY_IMAGE", e,
                              image_path=self._current_image_path)
            return

        self._rectify_info = {
            "pts": pts, "H": H, "out_size": size,
            "src_path": self._current_image_path,
        }
        self.canvas.set_image(rectified)
        self.pick_btn.setChecked(False)
        self._apply_grid()

        # Auto-save the rectified image into the workspace.
        if self._workspace:
            name = Path(self._current_image_path).stem
            saved = state_store.save_rectified(self._workspace, name, rectified)
            self.rect_status.setText(f"已校正 {size[0]}×{size[1]}（已保存）")
            self.statusBar().showMessage(f"校正图已保存: {saved}")
        else:
            self.rect_status.setText(f"已校正 {size[0]}×{size[1]}")

    def _skip_rectify(self) -> None:
        self._rectify_skipped = True
        self._rectify_info = None
        self.rect_status.setText("已跳过校正（使用原图）")
        self.statusBar().showMessage("已跳过校正，直接使用原图标注")

    # ==================================================================
    # Brush / mask
    # ==================================================================
    def _apply_brush(self) -> None:
        self.canvas.set_brush(self.radius_spin.value(), self.erase_btn.isChecked())
        self._config["brush_radius"] = self.radius_spin.value()
        state_store.save_config(self._app_dir, self._config)

    def _toggle_erase(self, checked: bool) -> None:
        self.erase_btn.setText("橡皮模式" if checked else "画笔模式")
        self._apply_brush()

    def _undo_stroke(self) -> None:
        if not self.canvas.undo_stroke():
            self.statusBar().showMessage("没有可撤销的笔画")

    def _toggle_clean_view(self, checked: bool) -> None:
        self.canvas.clean_view = checked
        self.canvas.update()

    def _on_mask_changed(self) -> None:
        labels = self.canvas.labels_from_mask()
        # Display only 吐水 cells; "0" cells stay uncolored so the photo is
        # not drowned in green while brushing.  Export still uses the full
        # dict via labels_from_mask()/defaults.
        self.canvas.set_labels({k: v for k, v in labels.items() if v == "1"})
        n1 = sum(1 for v in labels.values() if v == "1")
        area = int(np.count_nonzero(self.canvas.mask())) if self.canvas.mask() is not None else 0
        self.statusBar().showMessage(
            f"掩膜像素: {area} | 吐水格: {n1} / {len(labels)}"
        )

    def _auto_annotate(self) -> None:
        image = self.canvas.get_image()
        if image is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        if not self.canvas.grid_manager.cells:
            QMessageBox.warning(self, "提示", "请先应用分块")
            return

        StageTracker.set("DETECT_GUTTATION")
        try:
            detector = GuttationDetector(GuttationConfig(confirm_threshold=0.70))
            detector.detect(image, self.canvas.grid_manager)
            self._last_detector = detector

            # Paint confirmed candidates into the mask as circles.
            if self.canvas.mask() is None:
                self.canvas.clear_mask()
            mask = self.canvas.mask()
            cell_lookup = {(c.row, c.col): c for c in self.canvas.grid_manager.cells}
            n = 0
            for (row, col), cands in detector.last_candidates.items():
                cell = cell_lookup.get((row, col))
                if cell is None:
                    continue
                for cand in cands:
                    if cand.verdict != "confirmed":
                        continue
                    r = max(3, int(np.sqrt(cand.area / np.pi)))
                    cv2.circle(mask, (cell.x1 + int(cand.cx), cell.y1 + int(cand.cy)),
                               r, 255, -1, cv2.LINE_AA)
                    n += 1
            self.canvas.set_mask(mask)
            self.statusBar().showMessage(f"自动预标注完成：{n} 个候选点已加入掩膜，请用笔刷修正")
        except Exception as e:
            StageTracker.fail("DETECT_GUTTATION")
            log_exception("DETECT_GUTTATION", e)
            show_error_dialog(self, "检测失败", "DETECT_GUTTATION", e,
                              image_path=self._current_image_path)

    # ==================================================================
    # Grid
    # ==================================================================
    def _apply_preset(self, cols: int, rows: int) -> None:
        self.cols_spin.setValue(cols)
        self.rows_spin.setValue(rows)
        self._apply_grid()

    def _swap_grid(self) -> None:
        cols, rows = self.cols_spin.value(), self.rows_spin.value()
        self.cols_spin.setValue(rows)
        self.rows_spin.setValue(cols)
        self._apply_grid()

    def _apply_grid(self) -> None:
        self.canvas.set_grid(self.cols_spin.value(), self.rows_spin.value())
        self._config["cols"] = self.cols_spin.value()
        self._config["rows"] = self.rows_spin.value()
        state_store.save_config(self._app_dir, self._config)
        self._update_grid_info()
        self._on_mask_changed()  # re-derive labels for the new grid

    def _update_grid_info(self) -> None:
        if self.canvas.get_image() is None:
            self.grid_info_label.setText("网格信息: 未设置")
            return
        h, w = self.canvas.get_image().shape[:2]
        cols, rows = self.canvas.grid_manager.cols, self.canvas.grid_manager.rows
        if cols > 0 and rows > 0:
            self.grid_info_label.setText(
                f"图像: {w}×{h} | 网格: {cols}×{rows} | 单元: {w // cols}×{h // rows}"
            )

    # ==================================================================
    # Stash / export
    # ==================================================================
    def _current_name(self) -> str:
        return Path(self._current_image_path).stem if self._current_image_path else ""

    def _stash_current(self, silent: bool = False) -> bool:
        if not self._workspace or not self._current_image_path:
            return False
        name = self._current_name()
        grid = {"cols": self.canvas.grid_manager.cols,
                "rows": self.canvas.grid_manager.rows}
        brush = {"radius": self.radius_spin.value(),
                 "mask_threshold": self.canvas.mask_threshold}
        try:
            state_store.save_state(
                self._workspace, name, self.canvas.mask(), grid,
                self._rectify_info, brush, self._current_image_path,
            )
            if not silent:
                self.statusBar().showMessage(f"已暂存: {name}")
            return True
        except Exception as e:
            log_exception("SAVE_STATE", e)
            if not silent:
                show_error_dialog(self, "暂存失败", "SAVE_STATE", e)
            return False

    def _stash_and_next(self) -> None:
        if not self._images:
            QMessageBox.warning(self, "提示", "请先选择工作文件夹")
            return
        if self._stash_current():
            if self._current_index + 1 < len(self._images):
                self._load_image_at(self._current_index + 1)
            else:
                QMessageBox.information(self, "完成", "已是最后一张，全部暂存完成")
                self.statusBar().showMessage("全部图片已暂存，可批量导出")

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.output_edit.text() or str(Path.home())
        )
        if directory:
            self.output_edit.setText(directory)
            self._config["output_dir"] = directory
            state_store.save_config(self._app_dir, self._config)

    def _mask_areas(self) -> Optional[Dict[Tuple[int, int], int]]:
        mask = self.canvas.mask()
        if mask is None:
            return None
        areas = {}
        for cell in self.canvas.grid_manager.cells:
            areas[(cell.row, cell.col)] = int(
                np.count_nonzero(mask[cell.y1:cell.y2, cell.x1:cell.x2])
            )
        return areas

    def _export_current(self) -> None:
        if self.canvas.get_image() is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        if not self.canvas.grid_manager.cells:
            QMessageBox.warning(self, "提示", "请先应用分块")
            return

        output_dir = self.output_edit.text().strip() or "output"
        output_dir = os.path.abspath(output_dir)
        self._on_mask_changed()  # ensure labels are current

        progress = QProgressDialog("正在导出...", "取消", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        self.worker = ExportWorker(
            self.canvas, output_dir, self._current_name(), self._mask_areas()
        )
        self.worker.finished.connect(lambda p: self._on_export_finished(p, progress))
        self.worker.error.connect(lambda e: self._on_export_error(e, progress))
        self.worker.start()

    @Slot(str)
    def _on_export_finished(self, csv_path: str, progress) -> None:
        progress.close()
        output_dir = os.path.dirname(csv_path)
        if self._rectify_info:
            try:
                save_rectify_info(
                    str(Path(output_dir) / "rectify_info.json"),
                    self._rectify_info["pts"], self._rectify_info["H"],
                    self._rectify_info["out_size"], self._rectify_info["src_path"],
                )
            except Exception:
                pass
        QMessageBox.information(self, "导出成功", f"CSV 文件:\n{csv_path}\n\n输出目录:\n{output_dir}")
        self.statusBar().showMessage(f"导出完成: {csv_path}")

    @Slot(str)
    def _on_export_error(self, message: str, progress) -> None:
        progress.close()
        from PySide6.QtWidgets import QApplication

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("导出失败")
        box.setText(f"阶段: EXPORT\n{message.split(chr(10))[0]}\n详细日志: logs/errors.log")
        copy_btn = box.addButton("复制错误信息", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is copy_btn:
            QApplication.clipboard().setText(message)
        self.statusBar().showMessage("导出失败（详见 logs/errors.log）")

    # ------------------------------------------------------------------
    # Multi-grid export
    # ------------------------------------------------------------------
    def _parse_grids(self) -> Optional[List[Tuple[int, int]]]:
        """Parse '36x18, 41x20, 48x28' into [(cols, rows), ...]; remember it."""
        import re

        text = self.grids_edit.text().strip()
        grids = []
        for token in re.split(r"[,，;；\s]+", text):
            if not token:
                continue
            m = re.match(r"^(\d+)\s*[xX×*]\s*(\d+)$", token)
            if not m:
                QMessageBox.warning(self, "格式错误",
                                    f"无法识别的分块: {token}\n正确格式如 36x18,41x20")
                return None
            grids.append((int(m.group(1)), int(m.group(2))))
        if not grids:
            QMessageBox.warning(self, "提示", "请至少输入一种分块策略")
            return None
        self._config["multi_grids_text"] = text
        state_store.save_config(self._app_dir, self._config)
        return grids

    @staticmethod
    def _labels_areas_for_grid(mask, gm, threshold):
        labels, areas = {}, {}
        for cell in gm.cells:
            cnt = 0
            if mask is not None:
                crop = mask[cell.y1:cell.y2, cell.x1:cell.x2]
                cnt = int(np.count_nonzero(crop)) if crop.size else 0
            areas[(cell.row, cell.col)] = cnt
            ratio = cnt / max(1, (cell.y2 - cell.y1) * (cell.x2 - cell.x1))
            labels[(cell.row, cell.col)] = "1" if ratio >= threshold else "0"
        return labels, areas

    def _export_one_grid(self, image, name, cols, rows, mask, threshold,
                         out_dir, merged_records):
        """Export one image under one grid strategy; collect merged records."""
        from .grid_manager import GridManager
        import csv as _csv

        gm = GridManager()
        gm.update(image.shape[1], image.shape[0], cols, rows)
        labels, areas = self._labels_areas_for_grid(mask, gm, threshold)

        sub_dir = Path(out_dir) / f"grid_{cols}x{rows}"
        csv_path = export_dataset(
            image=image, grid_manager=gm, labels=labels,
            output_dir=str(sub_dir), image_name=name, mask_areas=areas,
        )
        if csv_path and Path(csv_path).exists():
            with open(csv_path, encoding="utf-8") as f:
                for rec in _csv.DictReader(f):
                    rec["grid"] = f"{cols}x{rows}"
                    merged_records.append(rec)
        return csv_path

    @staticmethod
    def _write_merged_csv(out_dir, merged_records):
        if not merged_records:
            return None
        import csv as _csv

        fieldnames = ["image_path", "label", "row", "col",
                      "original_image", "mask_area_px", "grid"]
        merged_path = Path(out_dir) / "labels_merged.csv"
        with open(merged_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_records)
        return str(merged_path)

    def _export_multi_current(self) -> None:
        """Export the CURRENT image under every grid strategy at once."""
        image = self.canvas.get_image()
        if image is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        grids = self._parse_grids()
        if not grids:
            return

        out_dir = os.path.abspath(self.output_edit.text().strip() or "output")
        mask = self.canvas.mask()
        threshold = self.canvas.mask_threshold
        name = self._current_name()

        merged: List[Dict] = []
        try:
            for cols, rows in grids:
                self._export_one_grid(image, name, cols, rows, mask,
                                      threshold, out_dir, merged)
            merged_path = self._write_merged_csv(out_dir, merged)
            QMessageBox.information(
                self, "多分块导出完成",
                f"{name} 已按 {len(grids)} 种分块导出到:\n{out_dir}\n"
                f"每种策略一个 grid_列x行 子目录\n合并表: {merged_path}",
            )
            self.statusBar().showMessage(f"多分块导出完成: {len(grids)} 种策略")
        except Exception as e:
            log_exception("MULTI_EXPORT", e)
            show_error_dialog(self, "多分块导出失败", "MULTI_EXPORT", e)

    def _export_all_stashed(self) -> None:
        if not self._workspace:
            QMessageBox.warning(self, "提示", "请先选择工作文件夹")
            return
        grids = self._parse_grids()
        if not grids:
            return
        ann_dir = state_store._ann_dir(self._workspace)
        states = sorted(ann_dir.glob("*_state.json")) if ann_dir.exists() else []
        if not states:
            QMessageBox.information(self, "提示", "没有暂存的标注结果")
            return

        output_dir = os.path.abspath(self.output_edit.text().strip() or "output")
        progress = QProgressDialog("批量导出中...", "取消", 0, len(states) * len(grids), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        from .image_io import imread_unicode

        merged_records: List[Dict] = []
        n_done = 0
        try:
            step = 0
            for state_file in states:
                if progress.wasCanceled():
                    break
                name = state_file.stem.replace("_state", "")
                state = state_store.load_state(self._workspace, name)
                if state is None:
                    continue

                img = None
                if state.get("rectified"):
                    rp = state_store.rectified_path(self._workspace, name)
                    if rp.exists():
                        img = imread_unicode(str(rp))
                if img is None:
                    src = state.get("source_image", "")
                    if src and Path(src).exists():
                        img = imread_unicode(src)
                if img is None:
                    continue

                mask = state.get("mask")
                threshold = float((state.get("brush") or {}).get(
                    "mask_threshold", self.canvas.mask_threshold))

                for cols, rows in grids:
                    step += 1
                    progress.setValue(step)
                    if progress.wasCanceled():
                        break
                    self._export_one_grid(img, name, cols, rows, mask,
                                          threshold, output_dir, merged_records)
                n_done += 1

            merged_path = self._write_merged_csv(output_dir, merged_records)
            progress.setValue(len(states) * len(grids))
            QMessageBox.information(
                self, "批量导出完成",
                f"已导出 {n_done} 张图片 × {len(grids)} 种分块到:\n{output_dir}\n"
                f"合并表: {merged_path}（{len(merged_records)} 行）",
            )
        except Exception as e:
            progress.close()
            log_exception("BATCH_EXPORT", e)
            show_error_dialog(self, "批量导出失败", "BATCH_EXPORT", e)

    @Slot(str)
    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def closeEvent(self, event) -> None:
        # Auto-stash on exit so no work is lost.
        if self._workspace and self._current_image_path:
            self._stash_current(silent=True)
        state_store.save_config(self._app_dir, self._config)
        super().closeEvent(event)
