"""PyQt6 主窗口：包含首页、数据、预处理、特征选择、模型、结果、日志、报告等标签页。"""

import json
import sys
from pathlib import Path

import yaml
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox,
    QComboBox, QListWidget, QListWidgetItem, QSplitter,
    QAbstractItemView, QGroupBox, QCheckBox, QScrollArea,
)

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger, GUILogHandler
from core.experiment_db import ExperimentDB
from core.data_manager import DataManager
from core.preprocessor import Preprocessor, apply_imbalance_handling
from core.table1_generator import generate_table1
from gui.feature_tab import FeatureTab
from gui.model_tab import ModelTab
from gui.ensemble_tab import EnsembleTab
from gui.transformer_tab import TransformerTab
from gui.automl_tab import AutoMLTab
from gui.workflow_tab import WorkflowTab
from gui.survival_tab import SurvivalTab
from gui.ablation_tab import AblationTab
from gui.minimal_subset_tab import MinimalSubsetTab
from gui.result_tab import ResultTab
from gui.visualization_tab import VisualizationTab
from gui.report_tab import ReportTab


class HomeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h1>AutoML-Platform</h1>"))
        self.layout.addWidget(QLabel("半自动化表格机器学习工作流平台"))
        self.layout.addSpacing(20)

        # 新建实验
        new_group = QWidget()
        new_layout = QHBoxLayout(new_group)
        self.exp_name_input = QLineEdit()
        self.exp_name_input.setPlaceholderText("输入实验名称")
        self.exp_desc_input = QLineEdit()
        self.exp_desc_input.setPlaceholderText("输入实验描述（可选）")
        self.btn_new = QPushButton("新建实验")
        self.btn_new.clicked.connect(self.create_experiment)
        new_layout.addWidget(QLabel("名称:"))
        new_layout.addWidget(self.exp_name_input)
        new_layout.addWidget(QLabel("描述:"))
        new_layout.addWidget(self.exp_desc_input)
        new_layout.addWidget(self.btn_new)
        self.layout.addWidget(new_group)

        # 实验列表
        self.layout.addWidget(QLabel("<b>实验列表</b>"))
        self.exp_list = QListWidget()
        self.exp_list.itemClicked.connect(self.load_experiment)
        self.layout.addWidget(self.exp_list)

        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self.refresh_experiments)
        self.layout.addWidget(self.btn_refresh)

        self.refresh_experiments()

    def refresh_experiments(self):
        self.exp_list.clear()
        if self.parent_window is None:
            return
        for exp in self.parent_window.db.list_experiments():
            item_text = f"[{exp['id']}] {exp['name']} - {exp['status']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, exp['id'])
            self.exp_list.addItem(item)

    def create_experiment(self):
        name = self.exp_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入实验名称")
            return
        desc = self.exp_desc_input.text().strip()
        try:
            exp_id = self.parent_window.db.create_experiment(
                name=name,
                description=desc,
                config=self.parent_window.config,
                output_dir=str(self.parent_window.output_dir),
            )
            self.parent_window.set_current_experiment(exp_id)
            QMessageBox.information(self, "成功", f"实验已创建：{name} (ID={exp_id})")
            self.refresh_experiments()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def load_experiment(self, item: QListWidgetItem):
        exp_id = item.data(Qt.ItemDataRole.UserRole)
        self.parent_window.set_current_experiment(exp_id)


class DataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.dm = DataManager()
        self.layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_load = QPushButton("导入数据")
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_path = QLabel("未选择文件")
        top.addWidget(self.btn_load)
        top.addWidget(self.lbl_path, stretch=1)
        self.layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # 数据预览表格
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(300)
        splitter.addWidget(self.preview_table)

        # 变量摘要表格
        self.summary_table = QTableWidget()
        splitter.addWidget(self.summary_table)

        self.layout.addWidget(splitter)

        # 批量操作区域
        batch_layout = QHBoxLayout()
        self.combo_batch_role = QComboBox()
        from core.data_manager import VARIABLE_ROLES
        self.combo_batch_role.addItems(VARIABLE_ROLES)
        self.btn_batch_apply = QPushButton("批量设置选中行")
        self.btn_batch_apply.clicked.connect(self.batch_apply_role)
        self.btn_auto_infer = QPushButton("智能推断角色")
        self.btn_auto_infer.clicked.connect(self.auto_infer_roles)
        self.btn_save_roles = QPushButton("保存角色标注")
        self.btn_save_roles.clicked.connect(self.save_roles)
        batch_layout.addWidget(QLabel("批量角色:"))
        batch_layout.addWidget(self.combo_batch_role)
        batch_layout.addWidget(self.btn_batch_apply)
        batch_layout.addWidget(self.btn_auto_infer)
        batch_layout.addStretch()
        batch_layout.addWidget(self.btn_save_roles)
        self.layout.addLayout(batch_layout)

        # Table 1 基线表
        table1_group = QGroupBox("Table 1 基线特征汇总表（基于原始数据，按 outcome 分组）")
        table1_layout = QHBoxLayout(table1_group)
        self.check_table1_nonpara = QCheckBox("连续变量使用非参数检验 (Mann-Whitney U)")
        self.check_table1_nonpara.setToolTip("默认使用 t 检验；勾选后使用 Mann-Whitney U 检验")
        self.check_table1_median = QCheckBox("同时展示中位数 (IQR)")
        self.check_table1_median.setToolTip("在均值±SD 后追加 [中位数 (Q1-Q3)]")
        self.btn_generate_table1 = QPushButton("生成 Table 1")
        self.btn_generate_table1.setToolTip("生成后保存到当前实验根目录：table1_baseline.xlsx")
        self.btn_generate_table1.clicked.connect(self.generate_table1)
        self.lbl_table1_status = QLabel("未生成")
        self.lbl_table1_status.setStyleSheet("color: gray;")
        table1_layout.addWidget(self.check_table1_nonpara)
        table1_layout.addWidget(self.check_table1_median)
        table1_layout.addWidget(self.btn_generate_table1)
        table1_layout.addWidget(self.lbl_table1_status, stretch=1)
        self.layout.addWidget(table1_group)

        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.summary_table.cellChanged.connect(self.on_role_changed)
        self._block_cell_change = False

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "",
            "Data Files (*.csv *.xlsx *.xls *.sas7bdat *.sav *.dta);;All Files (*)"
        )
        if not path:
            return
        try:
            self.dm.load(path)
            self.lbl_path.setText(path)
            self.parent_window.log_message(f"已导入数据：{path}，形状：{self.dm.df.shape}")
            self.update_preview()
            self.update_summary()
            if self.parent_window.current_experiment_id is not None:
                # 数据变更：归档全部下游结果（旧结果保留在 archive/ 中）
                from gui.widgets.result_utils import archive_downstream_results
                archived = archive_downstream_results(
                    self.parent_window.db, self.parent_window.current_experiment_id,
                    trigger="data", logger=self.parent_window.logger,
                )
                if archived:
                    self.parent_window.log_message(f"数据变更：已归档上一轮下游结果 {archived}")
                self.parent_window.db.update_experiment(
                    self.parent_window.current_experiment_id,
                    {"data_path": path, "status": "data_loaded"}
                )
                # 刷新下游视图
                self.parent_window._refresh_downstream_views()
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def update_preview(self):
        preview = self.dm.get_preview(100)
        self.preview_table.setRowCount(preview.shape[0])
        self.preview_table.setColumnCount(preview.shape[1])
        self.preview_table.setHorizontalHeaderLabels(preview.columns.tolist())
        for i in range(preview.shape[0]):
            for j in range(preview.shape[1]):
                val = preview.iat[i, j]
                self.preview_table.setItem(i, j, QTableWidgetItem(str(val)))

    def update_summary(self):
        summary = self.dm.get_summary()
        self._block_cell_change = True
        self.summary_table.setRowCount(summary.shape[0])
        self.summary_table.setColumnCount(summary.shape[1] + 1)
        headers = summary.columns.tolist() + ["设置角色"]
        self.summary_table.setHorizontalHeaderLabels(headers)

        from core.data_manager import VARIABLE_ROLES
        for i, row in summary.iterrows():
            for j, val in enumerate(row):
                self.summary_table.setItem(i, j, QTableWidgetItem(str(val)))
            combo = QComboBox()
            combo.addItems(VARIABLE_ROLES)
            combo.setCurrentText(row['role'])
            combo.currentTextChanged.connect(lambda text, r=i: self.role_combo_changed(r, text))
            self.summary_table.setCellWidget(i, len(row), combo)
        self._block_cell_change = False

    def role_combo_changed(self, row: int, text: str):
        col_name = self.summary_table.item(row, 0).text()
        self.dm.set_role(col_name, text)

    def batch_apply_role(self):
        selected_rows = sorted(set(idx.row() for idx in self.summary_table.selectedIndexes()))
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先在表格中选中要批量设置的行")
            return
        role = self.combo_batch_role.currentText()
        for row in selected_rows:
            col_name = self.summary_table.item(row, 0).text()
            self.dm.set_role(col_name, role)
            combo = self.summary_table.cellWidget(row, len(self.dm.get_summary().columns))
            if combo:
                combo.setCurrentText(role)
        self.parent_window.log_message(f"批量设置 {len(selected_rows)} 个变量为 {role}")

    def auto_infer_roles(self):
        if self.dm.df is None:
            QMessageBox.warning(self, "警告", "请先导入数据")
            return
        # 基于列名和数据特征推断角色
        for col in self.dm.df.columns:
            name = col.lower()
            if name in {"wtmec2yr", "mec_wt", "sample_weight"}:
                self.dm.set_role(col, "weight")
            elif name in {"sdmvstra", "strata"}:
                self.dm.set_role(col, "strata")
            elif name in {"sdmvpsu", "psu"}:
                self.dm.set_role(col, "psu")
            elif name in {"time", "permth_int", "follow_up", "survival_time", "month", "months"}:
                self.dm.set_role(col, "time")
            elif name in {"id", "seqn", "participant_id", "pid"}:
                self.dm.set_role(col, "id")
            elif name in {"outcome", "target", "y", "death", "mortality", "event"}:
                self.dm.set_role(col, "outcome")
            else:
                s = self.dm.df[col]
                n_unique = s.nunique(dropna=True)
                if pd.api.types.is_numeric_dtype(s):
                    if n_unique <= 2:
                        self.dm.set_role(col, "binary")
                    else:
                        self.dm.set_role(col, "continuous")
                else:
                    if n_unique <= 2:
                        self.dm.set_role(col, "binary")
                    else:
                        self.dm.set_role(col, "nominal")
        self.update_summary()
        self.parent_window.log_message("已智能推断角色，请检查并手动修正")

    def on_role_changed(self, row, column):
        pass

    def load_experiment_data(self, exp_id: int):
        """从数据库加载实验的数据和角色。"""
        exp = self.parent_window.db.get_experiment(exp_id)
        if not exp:
            return
        data_path = exp.get("data_path")
        config = exp.get("config") or {}
        saved_roles = config.get("roles")

        if data_path and Path(data_path).exists():
            try:
                self.dm.load(data_path)
                if saved_roles:
                    self.dm.roles = saved_roles
                self.lbl_path.setText(data_path)
                self.update_preview()
                self.update_summary()
                self.parent_window.log_message(f"已加载实验数据：{data_path}")
            except Exception as e:
                self.parent_window.logger.error(f"加载实验数据失败：{e}")

        # 恢复 Table 1 设置与生成状态
        table1_config = config.get("table1") or {}
        if table1_config:
            self.check_table1_nonpara.setChecked(table1_config.get("non_parametric", False))
            self.check_table1_median.setChecked(table1_config.get("show_median", False))
            path = table1_config.get("path")
            if path and Path(path).exists():
                self.lbl_table1_status.setText(f"历史结果：{Path(path).name}")
            else:
                self.lbl_table1_status.setText("历史结果文件不存在")

    def save_roles(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get('config') or {}
        config['roles'] = self.dm.roles
        self.parent_window.db.update_experiment(
            self.parent_window.current_experiment_id,
            {"config": config, "status": "role_annotated"}
        )
        self.parent_window.log_message("角色标注已保存")
        QMessageBox.information(self, "成功", "角色标注已保存")

    def generate_table1(self):
        """基于原始数据生成 Table 1 基线特征汇总表并保存为 Excel。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        if self.dm.df is None:
            QMessageBox.warning(self, "警告", "请先导入数据")
            return

        outcome_cols = [c for c, r in self.dm.roles.items() if r == "outcome"]
        if not outcome_cols:
            QMessageBox.warning(self, "警告", "未设置 outcome 角色，请将结局变量设为 outcome")
            return

        try:
            exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
            output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "table1_baseline.xlsx"

            result = generate_table1(
                self.dm.df,
                self.dm.roles,
                outcome_col=outcome_cols[0],
                output_path=output_path,
                non_parametric=self.check_table1_nonpara.isChecked(),
                show_median=self.check_table1_median.isChecked(),
            )

            # 保存配置与路径
            config = exp.get("config") or {}
            config["table1"] = {
                "path": str(output_path),
                "non_parametric": self.check_table1_nonpara.isChecked(),
                "show_median": self.check_table1_median.isChecked(),
                "outcome_col": outcome_cols[0],
                "n_features": result.shape[0] - 1,  # 扣除 N 行
            }
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id,
                {"config": config}
            )

            self.lbl_table1_status.setText(f"已保存：{output_path.name}")
            self.parent_window.log_message(f"Table 1 已生成：{output_path}")
            QMessageBox.information(self, "成功", f"Table 1 已保存到：\n{output_path}")
        except Exception as e:
            import traceback
            detail = traceback.format_exc()
            self.parent_window.logger.error(f"生成 Table 1 失败：{detail}")
            QMessageBox.critical(self, "失败", f"生成 Table 1 失败：{e}")


class PreprocessTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>预处理</h2>"))

        # 参数配置
        form = QHBoxLayout()

        self.combo_missing = QComboBox()
        self.combo_missing.addItems(["median", "mean", "knn", "drop"])
        self.combo_missing.setCurrentText("median")
        form.addWidget(QLabel("缺失值:"))
        form.addWidget(self.combo_missing)

        self.combo_outlier = QComboBox()
        self.combo_outlier.addItems(["iqr", "winsorize", "none"])
        self.combo_outlier.setCurrentText("iqr")
        form.addWidget(QLabel("异常值:"))
        form.addWidget(self.combo_outlier)

        self.combo_encoding = QComboBox()
        self.combo_encoding.addItems(["ordinal", "label"])
        self.combo_encoding.setCurrentText("ordinal")
        form.addWidget(QLabel("编码:"))
        form.addWidget(self.combo_encoding)

        self.combo_scaling = QComboBox()
        self.combo_scaling.addItems(["standard", "minmax", "robust", "none"])
        self.combo_scaling.setCurrentText("standard")
        form.addWidget(QLabel("缩放:"))
        form.addWidget(self.combo_scaling)

        self.combo_imbalance = QComboBox()
        self.combo_imbalance.addItems(["none", "smote", "adasyn", "undersample", "class_weight"])
        self.combo_imbalance.setCurrentText("none")
        form.addWidget(QLabel("不平衡:"))
        form.addWidget(self.combo_imbalance)

        self.layout.addLayout(form)

        self.btn_run = QPushButton("运行预处理")
        self.btn_run.clicked.connect(self.run_preprocessing)
        self.layout.addWidget(self.btn_run)

        self.lbl_status = QLabel("未运行")
        self.layout.addWidget(self.lbl_status)

        self.preview_table = QTableWidget()
        self.layout.addWidget(self.preview_table)

    def run_preprocessing(self, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        data_path = exp.get("data_path")
        config = exp.get("config") or {}
        roles = config.get("roles")

        if not data_path or not roles:
            QMessageBox.warning(self, "警告", "请先导入数据并保存角色标注")
            return

        # 上游变动：归档全部下游结果（旧结果保留在 archive/ 中）
        from gui.widgets.result_utils import archive_downstream_results
        archived = archive_downstream_results(
            self.parent_window.db, self.parent_window.current_experiment_id,
            trigger="preprocessing", logger=self.parent_window.logger,
        )
        if archived:
            self.parent_window.log_message(f"预处理重跑：已归档上一轮下游结果 {archived}")

        try:
            dm = DataManager()
            dm.load(data_path)
            dm.roles = roles

            feature_cols = dm.get_feature_columns(include_time=False)
            numeric_cols = [c for c in feature_cols if dm.roles.get(c) == "continuous"]
            categorical_cols = [c for c in feature_cols if dm.roles.get(c) in {"binary", "ordinal", "nominal"}]
            passthrough_cols = dm.get_columns_by_role(["weight", "strata", "psu", "time"])
            y_col = dm.get_columns_by_role(["outcome"])

            if not y_col:
                QMessageBox.warning(self, "警告", "未设置 outcome 角色，请在数据页将结局变量设为 outcome")
                return
            if not feature_cols:
                QMessageBox.warning(self, "警告", "没有可用于建模的特征列，请将至少一个变量设为 continuous/binary/ordinal/nominal")
                return

            self.parent_window.log_message(
                f"预处理参数：连续 {len(numeric_cols)} 个，分类 {len(categorical_cols)} 个，透传 {len(passthrough_cols)} 个"
            )

            prep = Preprocessor(
                numeric_columns=numeric_cols,
                categorical_columns=categorical_cols,
                passthrough_columns=passthrough_cols,
                missing_strategy=self.combo_missing.currentText(),
                outlier_method=self.combo_outlier.currentText(),
                encoding=self.combo_encoding.currentText(),
                scaling=self.combo_scaling.currentText(),
            )

            # 透传列也要包含在 X 中，否则 ColumnTransformer 找不到这些列
            all_input_cols = list(dict.fromkeys(feature_cols + passthrough_cols))
            X = dm.df[all_input_cols].copy()
            y = dm.df[y_col[0]].copy()

            X_processed = prep.fit_transform(X)

            # 不平衡处理
            method = self.combo_imbalance.currentText()
            if method != "none" and method != "class_weight":
                X_processed, y = apply_imbalance_handling(X_processed, y, method=method)

            # 保存
            output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "preprocessing"
            output_dir.mkdir(parents=True, exist_ok=True)

            processed_df = X_processed.copy()
            processed_df[y_col[0]] = y.values
            processed_path = output_dir / "preprocessed.csv"
            processed_df.to_csv(processed_path, index=False)

            import joblib
            pipeline_path = output_dir / "preprocessor.pkl"
            joblib.dump(prep.pipeline_, pipeline_path)

            # 保存预处理详细报告
            report = {
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "passthrough_columns": passthrough_cols,
                "n_numeric": len(numeric_cols),
                "n_categorical": len(categorical_cols),
                "n_passthrough": len(passthrough_cols),
                "missing_strategy": self.combo_missing.currentText(),
                "outlier_method": self.combo_outlier.currentText(),
                "encoding": self.combo_encoding.currentText(),
                "scaling": self.combo_scaling.currentText(),
                "imbalance_method": self.combo_imbalance.currentText(),
                "output_shape": list(processed_df.shape),
                "class_distribution": y.value_counts().to_dict(),
            }
            report_path = output_dir / "preprocessing_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            config["preprocessing"] = {
                "missing_strategy": self.combo_missing.currentText(),
                "outlier_method": self.combo_outlier.currentText(),
                "encoding": self.combo_encoding.currentText(),
                "scaling": self.combo_scaling.currentText(),
                "imbalance_method": self.combo_imbalance.currentText(),
                "processed_path": str(processed_path),
                "pipeline_path": str(pipeline_path),
                "report_path": str(report_path),
            }
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id,
                {"config": config, "status": "preprocessed"}
            )

            self.lbl_status.setText(f"预处理完成：{processed_df.shape}，已保存到 {output_dir}")
            self.parent_window.log_message(f"预处理完成：{processed_df.shape}")
            self.update_preview(processed_df)
            if not silent:
                QMessageBox.information(self, "成功", "预处理完成")
            # 上游变动后：刷新所有下游选项卡（下拉框与结果）
            self.parent_window._refresh_downstream_views()

        except Exception as e:
            import traceback
            detail = traceback.format_exc()
            self.parent_window.logger.error(f"预处理失败：{detail}")
            if not silent:
                QMessageBox.critical(self, "预处理失败", f"{e}\n\n详细错误已写入日志。")

    def update_preview(self, df: pd.DataFrame):
        preview = df.head(100)
        self.preview_table.setRowCount(preview.shape[0])
        self.preview_table.setColumnCount(preview.shape[1])
        self.preview_table.setHorizontalHeaderLabels(preview.columns.tolist())
        for i in range(preview.shape[0]):
            for j in range(preview.shape[1]):
                val = preview.iat[i, j]
                self.preview_table.setItem(i, j, QTableWidgetItem(f"{val:.4f}" if isinstance(val, float) else str(val)))


class LogTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoML-Platform")
        self.setGeometry(100, 100, 1400, 900)
        # 允许自由缩放，但保留一个可用的最小尺寸
        self.setMinimumSize(1000, 650)

        # 加载配置
        self.config = self._load_config()
        self.output_dir = Path(self.config['project']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志
        self.logger = setup_logger(
            self.config['logging']['log_dir'],
            self.config['logging']['level']
        )

        # 初始化数据库
        self.db = ExperimentDB(str(self.output_dir / "experiments.db"))
        self.current_experiment_id = None

        # 主窗口组件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.home_tab = HomeTab(self)
        self.data_tab = DataTab(self)
        self.log_tab = LogTab(self)

        # 重定向 logger 到 GUI 日志页
        self.logger.add(
            GUILogHandler(self.log_tab.text_edit),
            level=self.config['logging']['level'],
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )

        self._add_tab(self.home_tab, "首页")
        self._add_tab(self.data_tab, "数据")
        self.preprocess_tab = PreprocessTab(self)
        self._add_tab(self.preprocess_tab, "预处理")
        self.feature_tab = FeatureTab(self)
        self._add_tab(self.feature_tab, "特征选择")
        self.model_tab = ModelTab(self)
        self._add_tab(self.model_tab, "模型")
        self.ensemble_tab = EnsembleTab(self)
        self._add_tab(self.ensemble_tab, "集成")
        self.transformer_tab = TransformerTab(self)
        self._add_tab(self.transformer_tab, "Transformer")
        self.automl_tab = AutoMLTab(self)
        self._add_tab(self.automl_tab, "AutoML")
        self.survival_tab = SurvivalTab(self)
        self._add_tab(self.survival_tab, "生存分析")
        self.ablation_tab = AblationTab(self)
        self._add_tab(self.ablation_tab, "消融分析")
        self.minimal_subset_tab = MinimalSubsetTab(self)
        self._add_tab(self.minimal_subset_tab, "最小子集")
        self.workflow_tab = WorkflowTab(self)
        self._add_tab(self.workflow_tab, "工作流")
        self.result_tab = ResultTab(self)
        self._add_tab(self.result_tab, "结果")
        self.visualization_tab = VisualizationTab(self)
        self._add_tab(self.visualization_tab, "可视化")
        self.report_tab = ReportTab(self)
        self._add_tab(self.report_tab, "报告")
        self._add_tab(self.log_tab, "日志")

        self.main_layout.addWidget(self.tabs)

        # 状态栏
        self.statusBar().showMessage("就绪")
        self.logger.info("AutoML-Platform 已启动")

    def _add_tab(self, widget: QWidget, title: str):
        """将选项卡页包进 QScrollArea：窗口缩小/内容超高时可滚轮滑动，
        窗口也可以自由缩小（不再被内容的最小高度卡死）。"""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        self.tabs.addTab(area, title)

    def _load_config(self) -> dict:
        config_path = PROJECT_ROOT / "config" / "default.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def log_message(self, message: str):
        self.logger.info(message)

    def _load_historical_results(self, exp: dict):
        """实验切换时自动加载已保存的各模块结果。"""
        config = exp.get("config") or {}
        loaded = []

        # 预处理
        prep_config = config.get("preprocessing")
        if prep_config and prep_config.get("processed_path") and Path(prep_config["processed_path"]).exists():
            loaded.append(f"预处理 ({Path(prep_config['processed_path']).name})")

        # 特征选择
        fs_config = config.get("feature_selection")
        if fs_config and fs_config.get("consensus_table_path") and Path(fs_config["consensus_table_path"]).exists():
            self.feature_tab.load_results(exp)
            loaded.append(f"特征选择 ({int(pd.read_csv(fs_config['consensus_table_path'])['consensus'].sum()) if 'consensus' in pd.read_csv(fs_config['consensus_table_path']).columns else '?'} 个共识特征)")

        # 模型训练
        modeling_config = config.get("modeling")
        if modeling_config and modeling_config.get("leaderboard_path") and Path(modeling_config["leaderboard_path"]).exists():
            self.model_tab.load_results(exp)
            loaded.append(f"基础模型 ({len(pd.read_csv(modeling_config['leaderboard_path']))} 个模型)")

        # 集成学习
        ensemble_config = config.get("ensemble")
        if ensemble_config and ensemble_config.get("result_path") and Path(ensemble_config["result_path"]).exists():
            self.ensemble_tab.load_results(exp)
            loaded.append("集成学习")

        # Transformer
        transformer_config = config.get("transformer")
        if transformer_config and transformer_config.get("result_path") and Path(transformer_config["result_path"]).exists():
            self.transformer_tab.load_results(exp)
            loaded.append("Transformer")

        # AutoML
        automl_config = config.get("automl")
        if automl_config and automl_config.get("result_path") and Path(automl_config["result_path"]).exists():
            self.automl_tab.load_results(exp)
            loaded.append("AutoML")

        # SHAP / 结果
        result_dir = Path(exp.get("output_dir") or self.output_dir) / f"exp_{exp['id']}" / "shap"
        if result_dir.exists() and any(result_dir.glob("*_shap_summary.csv")):
            self.result_tab.load_results(exp)
            loaded.append("SHAP 分析")

        # 可视化选项卡刷新指标下拉框
        self.visualization_tab.load_results(exp)

        if loaded:
            self.log_message(f"已自动加载历史结果：{', '.join(loaded)}")
        else:
            self.log_message("该实验暂无已保存的结果文件")

    def set_current_experiment(self, exp_id: int):
        self.current_experiment_id = exp_id
        exp = self.db.get_experiment(exp_id)
        self.setWindowTitle(f"AutoML-Platform - [{exp_id}] {exp['name']}")
        self.statusBar().showMessage(f"当前实验：{exp['name']} (ID={exp_id})")
        self.logger.info(f"切换到实验：{exp['name']} (ID={exp_id})")
        # 通知数据页加载该实验的数据和角色
        self.data_tab.load_experiment_data(exp_id)
        # 恢复各模块超参数与设置（即使该模块尚未产生结果）
        self.feature_tab.load_settings(exp)
        self.model_tab.load_settings(exp)
        self.ensemble_tab.load_settings(exp)
        self.transformer_tab.load_settings(exp)
        self.automl_tab.load_settings(exp)
        self.workflow_tab.load_settings(exp)
        # 消融分析：刷新候选模型/变量并恢复设置与历史结果
        self.ablation_tab.refresh_candidates()
        self.ablation_tab.load_results(exp)
        # 最小子集分析：刷新模型并恢复设置与历史结果
        self.minimal_subset_tab.refresh_models()
        self.minimal_subset_tab.load_results(exp)
        # 报告：刷新 SHAP 指标下拉框并恢复设置
        self.report_tab.refresh_indicator_options()
        self.report_tab.load_settings(exp)
        # 加载各模块历史结果
        self._load_historical_results(exp)

    def _refresh_downstream_views(self):
        """上游（数据/预处理/特征选择/模型）变动后，统一刷新所有下游选项卡。

        归档清空了下游 config，load_results 会显示"无结果"；下拉框从当前共识特征重新填充。
        """
        if self.current_experiment_id is None:
            return
        exp = self.db.get_experiment(self.current_experiment_id)
        # 结果/可视化/报告：刷新 SHAP 指标下拉框
        try:
            self.result_tab.load_results(exp)
        except Exception as e:
            self.logger.warning(f"刷新结果页失败：{e}")
        try:
            self.visualization_tab.load_results(exp)
        except Exception as e:
            self.logger.warning(f"刷新可视化页失败：{e}")
        try:
            self.report_tab.refresh_indicator_options()
        except Exception as e:
            self.logger.warning(f"刷新报告页指标失败：{e}")
        # 消融/最小子集：刷新候选与结果
        try:
            self.ablation_tab.refresh_candidates()
            self.ablation_tab.load_results(exp)
        except Exception as e:
            self.logger.warning(f"刷新消融页失败：{e}")
        try:
            self.minimal_subset_tab.refresh_models()
            self.minimal_subset_tab.load_results(exp)
        except Exception as e:
            self.logger.warning(f"刷新最小子集页失败：{e}")
        # 各建模模块：清空过期结果显示
        for tab in (self.model_tab, self.ensemble_tab, self.transformer_tab,
                    self.automl_tab, self.feature_tab):
            try:
                tab.load_results(exp)
            except Exception:
                pass
        self.log_message("已刷新下游视图（上游结果变动，旧结果已归档）")
