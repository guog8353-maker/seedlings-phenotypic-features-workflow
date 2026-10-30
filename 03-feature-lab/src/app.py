"""Main window: crop-level feature library builder.

Import a folder of crop images (the dataset exported by software 2),
compute numeric features for every crop (one table row each, guttation
label from the 0/1 parent folder), and export CSV for downstream
TabTransformer classification / ablation / interpretability work.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .batch_features import (
    META_COLUMNS,
    FeatureBatchWorker,
    write_csv,
    write_features_meta,
)
from .error_reporting import StageTracker, log_exception, show_error_dialog
from .feature_tab import FeatureAnalysisTab
from .image_canvas import ImageCanvas

MAX_TABLE_ROWS = 5000  # display cap; the CSV always contains everything


class MainWindow(QMainWindow):
    """Crop feature-library builder main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("水稻吐水特征库构建平台")
        self.setMinimumSize(1300, 850)

        self._last_dir = str(Path.home())
        self._rows: List[Dict] = []
        self._columns: List[str] = []
        self._worker: Optional[FeatureBatchWorker] = None
        self._ref_canvas = ImageCanvas()

        self._init_ui()
        self._init_menu()
        self._apply_styles()

    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        sidebar = QWidget()
        sidebar.setFixedWidth(290)
        sb = QVBoxLayout(sidebar)
        sb.setAlignment(Qt.AlignTop)

        sb.addWidget(QLabel("<b>① 子块文件夹</b>"))
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("例如 .../raw-pic-label/grid_18x36")
        sb.addWidget(self.folder_edit)
        browse_btn = QPushButton("浏览文件夹")
        browse_btn.clicked.connect(self._browse_folder)
        sb.addWidget(browse_btn)

        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>② 特征组（勾选参与计算）</b>"))
        self.group_checks = {}
        for key, text in [
            ("geometry", "几何特征（CC/GPA/IGR/LGA/GPN）"),
            ("texture", "纹理特征（GLCM/LBP/Gabor）"),
            ("color", "色彩特征（多色彩空间/GCV/YR）"),
            ("vegetation", "RGB植被指数（21个×mean/std/range）"),
            ("spatial", "空间分布特征（GCCV/MI/CE/LAC，母图级）"),
        ]:
            cb = QCheckBox(text)
            cb.setChecked(True)
            sb.addWidget(cb)
            self.group_checks[key] = cb

        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>③ 计算</b>"))
        self.compute_btn = QPushButton("开始计算特征")
        self.compute_btn.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold;"
        )
        self.compute_btn.clicked.connect(self._start_compute)
        sb.addWidget(self.compute_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        sb.addWidget(self.progress_bar)

        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>④ 导出</b>"))
        export_btn = QPushButton("导出 CSV + 特征说明")
        export_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        export_btn.clicked.connect(self._export_csv)
        sb.addWidget(export_btn)

        sb.addSpacing(10)
        sb.addWidget(QLabel("<b>⑤ 参考大图（特征图研究用）</b>"))
        ref_btn = QPushButton("打开大图到特征图分析")
        ref_btn.clicked.connect(self._open_reference_image)
        sb.addWidget(ref_btn)

        sb.addStretch()
        splitter.addWidget(sidebar)

        self.tabs = QTabWidget()
        self.table = QTableWidget()
        self.tabs.addTab(self.table, "特征数据表")
        self.feature_tab = FeatureAnalysisTab(get_image=self._ref_canvas.get_image)
        self.tabs.addTab(self.feature_tab, "特征图分析")
        splitter.addWidget(self.tabs)
        splitter.setSizes([290, 1010])

        self.statusBar().showMessage("就绪：请选择子块文件夹")

    def _init_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        open_action = QAction("打开子块文件夹", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._browse_folder)
        file_menu.addAction(open_action)
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
            QLineEdit {
                background-color: #2a2a2a; border: 1px solid #555; padding: 4px;
            }
            QTableWidget {
                background-color: #2a2a2a; border: 1px solid #555; gridline-color: #555;
            }
            QHeaderView::section {
                background-color: #3c3c3c; padding: 4px; border: 1px solid #555;
            }
            QCheckBox { color: #eeeeee; }
            QTabBar::tab {
                background-color: #3c3c3c; padding: 8px 16px; border: 1px solid #555;
            }
            QTabBar::tab:selected { background-color: #505050; }
        """
        )

    # ------------------------------------------------------------------
    def _browse_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择子块文件夹", self._last_dir)
        if directory:
            self._last_dir = directory
            self.folder_edit.setText(directory)

    def _selected_groups(self) -> List[str]:
        return [k for k, cb in self.group_checks.items() if cb.isChecked()]

    def _start_compute(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).exists():
            QMessageBox.warning(self, "提示", "请选择有效的子块文件夹")
            return
        groups = self._selected_groups()
        if not groups:
            QMessageBox.warning(self, "提示", "请至少勾选一个特征组")
            return

        self.compute_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("正在计算特征...")

        self._worker = FeatureBatchWorker(folder, groups, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(int, int)
    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    @Slot(list)
    def _on_finished(self, rows: list) -> None:
        self._rows = rows
        self.compute_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self._fill_table()
        n1 = sum(1 for r in rows if r.get("label") == "1")
        self.statusBar().showMessage(
            f"计算完成: {len(rows)} 行 | 吐水 {n1} / 未吐水 {len(rows) - n1} | "
            f"特征列 {max(0, len(self._columns) - len(META_COLUMNS))}"
        )

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.compute_btn.setEnabled(True)
        self.statusBar().showMessage(message)

    def _fill_table(self) -> None:
        if not self._rows:
            return
        feat_cols: List[str] = []
        seen = set()
        for r in self._rows[:200]:
            for k in r.keys():
                if k not in META_COLUMNS and k not in seen:
                    seen.add(k)
                    feat_cols.append(k)
        self._columns = META_COLUMNS + feat_cols

        display_rows = self._rows[:MAX_TABLE_ROWS]
        self.table.setUpdatesEnabled(False)
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setRowCount(len(display_rows))
        for i, row in enumerate(display_rows):
            for j, col in enumerate(self._columns):
                v = row.get(col, "")
                text = f"{v:.4f}" if isinstance(v, float) else str(v)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, item)
        self.table.setUpdatesEnabled(True)
        self.table.resizeColumnsToContents()

    def _export_csv(self) -> None:
        if not self._rows:
            QMessageBox.warning(self, "提示", "请先计算特征")
            return
        default = str(Path(self.folder_edit.text()) / "features.csv") \
            if self.folder_edit.text() else "features.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出特征表", default, "CSV 文件 (*.csv)"
        )
        if not path:
            return
        try:
            write_csv(self._rows, path)
            meta_path = str(Path(path).with_name("features_meta.json"))
            write_features_meta(meta_path, self._selected_groups())
            QMessageBox.information(
                self, "导出完成",
                f"特征表: {path}\n特征说明: {meta_path}\n共 {len(self._rows)} 行",
            )
            self.statusBar().showMessage(f"已导出: {path}")
        except Exception as e:
            StageTracker.fail("EXPORT_CSV")
            log_exception("EXPORT_CSV", e)
            show_error_dialog(self, "导出失败", "EXPORT_CSV", e)

    def _open_reference_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开参考大图", self._last_dir,
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        if not self._ref_canvas.load_image(path):
            QMessageBox.critical(self, "错误", f"无法打开图片:\n{path}")
            return
        self.feature_tab.refresh_image()
        self.tabs.setCurrentWidget(self.feature_tab)
