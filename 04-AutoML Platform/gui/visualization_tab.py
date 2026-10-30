"""可视化选项卡：独立生成实验级综合图，支持配色方案选择。"""

import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QGroupBox, QProgressBar, QScrollArea,
    QColorDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.figure_generator import generate_experiment_figures
from core.plot_style import (
    list_palettes, PALETTES, make_custom_palette, get_palette,
    save_custom_palettes, load_custom_palettes, delete_custom_palette,
)
from gui.widgets import load_modeling_data

# 自定义调色盘持久化文件
CUSTOM_PALETTE_FILE = PROJECT_ROOT / "config" / "custom_palettes.json"


class FigureWorker(QThread):
    """后台生成图片，避免 GUI 卡死。"""
    finished_signal = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, exp_dir: Path, config: dict, X: pd.DataFrame,
                 output_dir: Path, shap_indicator: str = None, palette: str = "default",
                 top_n: int = 3):
        super().__init__()
        self.exp_dir = exp_dir
        self.config = config
        self.X = X
        self.output_dir = output_dir
        self.shap_indicator = shap_indicator
        self.palette = palette
        self.top_n = top_n

    def run(self):
        try:
            paths = generate_experiment_figures(
                self.exp_dir, self.config, self.X, self.output_dir,
                shap_indicator=self.shap_indicator, palette=self.palette,
                top_n=self.top_n,
            )
            self.finished_signal.emit(paths)
        except Exception as e:
            self.error.emit(str(e))


class VisualizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>可视化</h2>"))

        # 配色选择
        palette_group = QGroupBox("配色方案")
        palette_layout = QVBoxLayout(palette_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("选择调色盘："))
        self.combo_palette = QComboBox()
        # 先加载持久化的自定义调色盘
        load_custom_palettes(CUSTOM_PALETTE_FILE)
        for key, name in list_palettes():
            self.combo_palette.addItem(name, key)
        self.combo_palette.setCurrentIndex(0)
        self.combo_palette.currentIndexChanged.connect(self._on_palette_changed)
        row1.addWidget(self.combo_palette)
        self.btn_save_palette = QPushButton("保存当前自定义色")
        self.btn_save_palette.setToolTip("把当前应用的自定义配色保存到本地，重启后仍可用")
        self.btn_save_palette.clicked.connect(self.save_current_palette)
        self.btn_delete_palette = QPushButton("删除选中自定义色")
        self.btn_delete_palette.setToolTip("删除下拉框中选中的自定义调色盘（内置不可删）")
        self.btn_delete_palette.clicked.connect(self.delete_selected_palette)
        row1.addWidget(self.btn_save_palette)
        row1.addWidget(self.btn_delete_palette)
        row1.addStretch()
        palette_layout.addLayout(row1)

        # 色块预览
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("层颜色预览："))
        self.lbl_swatches = QLabel("Base  Ensemble  AutoML  Transformer")
        self.lbl_swatches.setStyleSheet("padding: 4px; border: 1px solid #ccc; background: white;")
        row2.addWidget(self.lbl_swatches)
        row2.addStretch()
        palette_layout.addLayout(row2)

        # 主/副色调选择与可视化
        row3 = QHBoxLayout()
        self.btn_primary = QPushButton("选择主色调")
        self.btn_primary.clicked.connect(self._pick_primary)
        self.lbl_primary = QLabel("#1f77b4")
        self.lbl_primary.setStyleSheet("background: #1f77b4; color: white; padding: 4px 10px; border: 1px solid #888;")
        self.lbl_primary.setMinimumWidth(80)

        self.btn_secondary = QPushButton("选择副色调")
        self.btn_secondary.clicked.connect(self._pick_secondary)
        self.lbl_secondary = QLabel("#aec7e8")
        self.lbl_secondary.setStyleSheet("background: #aec7e8; color: black; padding: 4px 10px; border: 1px solid #888;")
        self.lbl_secondary.setMinimumWidth(80)

        self.btn_custom = QPushButton("应用自定义色")
        self.btn_custom.clicked.connect(self.apply_custom_palette)

        row3.addWidget(self.btn_primary)
        row3.addWidget(self.lbl_primary)
        row3.addWidget(self.btn_secondary)
        row3.addWidget(self.lbl_secondary)
        row3.addWidget(self.btn_custom)
        row3.addStretch()
        palette_layout.addLayout(row3)

        self.layout.addWidget(palette_group)

        # 参数与运行
        param_group = QGroupBox("生成参数")
        param_layout = QHBoxLayout(param_group)
        param_layout.addWidget(QLabel("SHAP 关注指标："))
        self.combo_indicator = QComboBox()
        self.combo_indicator.setEditable(False)
        self.combo_indicator.setMinimumWidth(150)
        self.combo_indicator.setPlaceholderText("请选择共识特征")
        param_layout.addWidget(self.combo_indicator)

        param_layout.addWidget(QLabel("SHAP Top-N："))
        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(1, 10)
        self.spin_topn.setValue(3)
        param_layout.addWidget(self.spin_topn)

        self.btn_generate = QPushButton("生成实验综合图")
        self.btn_generate.clicked.connect(self.generate_figures)
        param_layout.addWidget(self.btn_generate)

        self.btn_open_dir = QPushButton("打开图片目录")
        self.btn_open_dir.clicked.connect(self.open_figures_dir)
        param_layout.addWidget(self.btn_open_dir)

        param_layout.addStretch()
        self.layout.addWidget(param_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.layout.addWidget(self.progress)

        self.lbl_status = QLabel("未生成")
        self.layout.addWidget(self.lbl_status)

        # 图片预览
        self.scroll_area = QScrollArea()
        self.image_label = QLabel("图片将显示在这里")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        self.layout.addWidget(self.scroll_area)

        # 已生成图片列表
        self.layout.addWidget(QLabel("<b>已生成图片与配套原始数据</b>"))
        self.figure_table = QTableWidget()
        self.figure_table.setColumnCount(3)
        self.figure_table.setHorizontalHeaderLabels(["图注", "PNG 路径", "CSV 路径"])
        self.figure_table.cellClicked.connect(self._preview_figure_from_table)
        self.layout.addWidget(self.figure_table)

        self._figure_worker = None
        self._current_fig_paths = {}

        self._update_swatches()

    def _on_palette_changed(self):
        """下拉选择变化时，更新主/副色显示与色块预览。"""
        key = self.combo_palette.currentData() or "default"
        palette = get_palette(key)
        primary = palette.get("primary", "#1f77b4")
        secondary = palette.get("secondary", "#aec7e8")
        self.lbl_primary.setText(primary)
        self.lbl_primary.setStyleSheet(
            f"background: {primary}; color: {'white' if self._is_dark(primary) else 'black'}; "
            f"padding: 4px 10px; border: 1px solid #888;"
        )
        self.lbl_secondary.setText(secondary)
        self.lbl_secondary.setStyleSheet(
            f"background: {secondary}; color: {'white' if self._is_dark(secondary) else 'black'}; "
            f"padding: 4px 10px; border: 1px solid #888;"
        )
        self._update_swatches()

    @staticmethod
    def _is_dark(hex_color: str) -> bool:
        """判断颜色是否偏暗，以决定使用白字或黑字。"""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return False
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return (r * 0.299 + g * 0.587 + b * 0.114) < 128

    def _update_swatches(self):
        """根据当前选中的调色盘刷新层颜色色块。"""
        key = self.combo_palette.currentData() or "default"
        palette = get_palette(key)
        layer = palette.get("layer", {})
        html = " ".join(
            f'<span style="background:{c}; color:{"white" if self._is_dark(c) else "black"}; '
            f'padding:2px 8px; margin-right:4px; border:1px solid #888;">{name}</span>'
            for name, c in layer.items()
        )
        self.lbl_swatches.setText(f"<html>{html}</html>")

    def _pick_primary(self):
        color = QColorDialog.getColor(QColor(self.lbl_primary.text()), self, "选择主色调")
        if color.isValid():
            hex_color = color.name()
            self.lbl_primary.setText(hex_color)
            self.lbl_primary.setStyleSheet(
                f"background: {hex_color}; color: {'white' if self._is_dark(hex_color) else 'black'}; "
                f"padding: 4px 10px; border: 1px solid #888;"
            )

    def _pick_secondary(self):
        color = QColorDialog.getColor(QColor(self.lbl_secondary.text()), self, "选择副色调")
        if color.isValid():
            hex_color = color.name()
            self.lbl_secondary.setText(hex_color)
            self.lbl_secondary.setStyleSheet(
                f"background: {hex_color}; color: {'white' if self._is_dark(hex_color) else 'black'}; "
                f"padding: 4px 10px; border: 1px solid #888;"
            )

    def showEvent(self, event):
        """进入选项卡时刷新 SHAP 指标下拉框（跟随当前数据）。"""
        super().showEvent(event)
        try:
            self._populate_indicator_combo()
        except Exception:
            pass

    def _populate_indicator_combo(self):
        """从当前实验的共识特征填充 SHAP 指标下拉框。"""
        current = self.combo_indicator.currentText()
        self.combo_indicator.clear()
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        try:
            X, _, _ = load_modeling_data(exp, use_feature_selected=True)
        except Exception:
            try:
                X, _, _ = load_modeling_data(exp, use_feature_selected=False)
            except Exception:
                return
        self.combo_indicator.addItems(list(X.columns))
        if current:
            self.combo_indicator.setCurrentText(current)

    def apply_custom_palette(self):
        primary = self.lbl_primary.text().strip()
        secondary = self.lbl_secondary.text().strip()
        if not primary.startswith("#") or not secondary.startswith("#"):
            QMessageBox.warning(self, "警告", "请先选择有效的主色调和副色调")
            return
        custom = make_custom_palette(primary, secondary)
        # 注册到 combo 中
        key = f"custom_{primary.lstrip('#')}_{secondary.lstrip('#')}"
        PALETTES[key] = custom
        # 持久化到本地文件
        try:
            save_custom_palettes(CUSTOM_PALETTE_FILE)
        except Exception as e:
            self.parent_window.logger.warning(f"自定义调色盘保存失败：{e}")
        # 避免重复添加
        idx = self.combo_palette.findData(key)
        if idx < 0:
            self.combo_palette.addItem(custom["name"], key)
            idx = self.combo_palette.count() - 1
        self.combo_palette.setCurrentIndex(idx)
        self._update_swatches()
        self.lbl_status.setText(f"已应用并保存自定义配色：{primary} / {secondary}")

    def save_current_palette(self):
        """把当前选中的自定义调色盘显式保存到本地文件。"""
        key = self.combo_palette.currentData()
        if not key or not str(key).startswith("custom_"):
            QMessageBox.warning(self, "警告", "当前选中的不是自定义调色盘。\n请先「选择主/副色调」并点击「应用自定义色」。")
            return
        try:
            save_custom_palettes(CUSTOM_PALETTE_FILE)
            self.lbl_status.setText(f"自定义调色盘已保存：{self.combo_palette.currentText()}")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"保存失败：{e}")

    def delete_selected_palette(self):
        """删除下拉框中选中的自定义调色盘。"""
        key = self.combo_palette.currentData()
        if not key or not str(key).startswith("custom_"):
            QMessageBox.warning(self, "警告", "内置调色盘不可删除，只能删除自定义调色盘。")
            return
        name = self.combo_palette.currentText()
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除自定义调色盘「{name}」？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if delete_custom_palette(key, filepath=CUSTOM_PALETTE_FILE):
            idx = self.combo_palette.currentIndex()
            self.combo_palette.removeItem(idx)
            self.combo_palette.setCurrentIndex(0)
            self._update_swatches()
            self.lbl_status.setText(f"已删除自定义调色盘：{name}")
        else:
            QMessageBox.warning(self, "警告", "删除失败")

    def generate_figures(self, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")
        if not modeling_config or not modeling_config.get("output_dir"):
            QMessageBox.warning(self, "警告", "请先完成模型训练")
            return

        try:
            X, _, _ = load_modeling_data(exp, use_feature_selected=True)
        except Exception:
            try:
                X, _, _ = load_modeling_data(exp, use_feature_selected=False)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载数据失败：{e}")
                return

        exp_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
        fig_dir = exp_dir / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        palette_key = self.combo_palette.currentData() or "default"
        indicator = self.combo_indicator.currentText().strip() or None
        top_n = self.spin_topn.value()

        self._figure_silent = silent
        self.btn_generate.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("正在生成图片...")

        self._figure_worker = FigureWorker(
            exp_dir, config, X, fig_dir,
            shap_indicator=indicator, palette=palette_key,
            top_n=top_n,
        )
        self._figure_worker.finished_signal.connect(self._on_figures_finished)
        self._figure_worker.error.connect(self._on_figures_error)
        self._figure_worker.start()

    def _on_figures_finished(self, paths: dict):
        self.btn_generate.setEnabled(True)
        self.progress.setVisible(False)
        self._current_fig_paths = paths
        self.lbl_status.setText(f"图片生成完成：{len(paths)} 张，配套 CSV 已同步导出")
        self.parent_window.log_message(f"可视化图片生成完成：{len(paths)} 张，配套 CSV 已同步导出")
        self._update_figure_table(paths)
        if paths:
            first_entry = list(paths.values())[0]
            first_png = first_entry.get("png") if isinstance(first_entry, dict) else first_entry
            if first_png:
                self._preview_figure(first_png)
        if not getattr(self, "_figure_silent", False):
            QMessageBox.information(self, "成功", f"已生成 {len(paths)} 张图片及配套 CSV 数据")

    def _on_figures_error(self, msg: str):
        self.btn_generate.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText(f"图片生成失败：{msg}")
        self.parent_window.logger.error(f"可视化图片生成失败：{msg}")
        if not getattr(self, "_figure_silent", False):
            QMessageBox.critical(self, "失败", msg)

    def _update_figure_table(self, paths: dict):
        self.figure_table.setRowCount(len(paths))
        for i, (caption, entry) in enumerate(paths.items()):
            if isinstance(entry, dict):
                png_path = entry.get("png", "")
                csv_path = entry.get("csv", "")
            else:
                png_path = str(entry)
                csv_path = ""
            self.figure_table.setItem(i, 0, QTableWidgetItem(caption))
            self.figure_table.setItem(i, 1, QTableWidgetItem(png_path))
            self.figure_table.setItem(i, 2, QTableWidgetItem(csv_path))
        self.figure_table.resizeColumnsToContents()

    def _preview_figure_from_table(self, row, col):
        # 第 1 列为 PNG 路径，第 2 列为 CSV 路径；点击任意列均预览 PNG
        path = self.figure_table.item(row, 1).text()
        self._preview_figure(path)

    def _preview_figure(self, path: str):
        if Path(path).exists():
            pixmap = QPixmap(path)
            # 缩放以适应预览区域宽度
            w = self.scroll_area.width() - 40
            if pixmap.width() > w:
                pixmap = pixmap.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setText("")

    def open_figures_dir(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        fig_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        import os
        if hasattr(os, "startfile"):
            os.startfile(str(fig_dir))

    def load_results(self, exp: dict):
        """实验切换时自动刷新指标下拉框。"""
        self._populate_indicator_combo()
