"""特征选择标签页（支持超参数配置、折叠面板、勾选全选、后处理）。"""

import json
import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QMessageBox, QGroupBox, QSplitter, QLineEdit, QFormLayout,
    QTextEdit, QFileDialog,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_selector import FeatureSelectionRunner
from gui.widgets import CheckableListWidget, CollapsibleGroupBox


DEFAULT_PARAMS = {
    "LASSO": {"C_min": 1e-4, "C_max": 1e4, "n_Cs": 20, "cv": 5, "max_iter": 1000, "coef_threshold": 1e-8},
    "Boruta": {"max_iter": 100},
    "RFE": {"step": 1, "cv": 5, "scoring": "roc_auc"},
    "MutualInformation": {"k": 20},
    "stepAIC": {"direction": "forward", "threshold_in": 0.05, "threshold_out": 0.10, "max_iter": 100},
    "SCAD": {"C": 1.0, "max_iter": 1000, "coef_threshold": 1e-8},
    "GA": {"population_size": 50, "generations": 30, "cx_prob": 0.5, "mut_prob": 0.2},
}

# 超参数悬停提示：{方法: {参数: 提示文本}}
PARAM_TOOLTIPS = {
    "LASSO": {
        "C_min": "正则化路径中 C 的最小值。C 越小正则化越强，入选特征越少。建议 1e-5 ~ 1e-3，若设置过大则会保留过多特征。",
        "C_max": "正则化路径中 C 的最大值。C 越大正则化越弱，越接近无惩罚 Logistic 回归。建议 1e2 ~ 1e4。",
        "n_Cs": "正则化路径上 C 值的采样点数。越大搜索越精细，但计算越久。建议 10 ~ 50。",
        "cv": "交叉验证折数。建议 3 ~ 10。",
        "max_iter": "优化器最大迭代次数。数据量大或特征多时可增大到 5000 ~ 10000。",
        "coef_threshold": "系数绝对值入选阈值。越大筛选越严格。默认 1e-8，若希望更严格可设为 1e-6 或 1e-4。",
    },
    "Boruta": {
        "max_iter": "Boruta 最大迭代轮数。越大模型越稳定但越慢。建议 50 ~ 300。",
    },
    "RFE": {
        "step": "每次迭代剔除的特征比例/数量。越大越快但越粗糙；设为 1 最精确但最慢。",
        "cv": "RFECV 交叉验证折数。建议 3 ~ 10。",
        "scoring": "模型选择评分指标。建议 roc_auc；类别不平衡严重可尝试 auprc。",
    },
    "MutualInformation": {
        "k": "最终保留的 Top-K 特征数。建议根据预期重要特征数量设置，如 10 ~ 50。",
    },
    "stepAIC": {
        "direction": "逐步回归方向：forward（前向）、backward（后向）、both（双向）。",
        "threshold_in": "变量进入模型的 p 值阈值。越小越严格。建议 0.01 ~ 0.10。",
        "threshold_out": "变量被剔除的 p 值阈值。应略大于 threshold_in。建议 0.05 ~ 0.15。",
        "max_iter": "最大迭代步数。防止无限循环。建议 100 ~ 500。",
    },
    "SCAD": {
        "C": "L1 正则化强度的倒数。C 越小惩罚越强，入选特征越少。建议 0.01 ~ 10.0。",
        "max_iter": "优化器最大迭代次数。建议 1000 ~ 10000。",
        "coef_threshold": "系数绝对值入选阈值。越大筛选越严格。默认 1e-8。",
    },
    "GA": {
        "population_size": "遗传算法种群大小。越大搜索越充分但越慢。建议 30 ~ 100。",
        "generations": "进化代数。越大收敛越充分。建议 20 ~ 100。",
        "cx_prob": "交叉概率。建议 0.3 ~ 0.7。",
        "mut_prob": "变异概率。建议 0.1 ~ 0.3。",
    },
}


class FeatureTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.main_layout = QVBoxLayout(self)

        self.main_layout.addWidget(QLabel("<h2>特征选择</h2>"))

        # 数据血缘信息
        self.info_layout = QHBoxLayout()
        self.lbl_data_path = QLabel("数据路径：未运行")
        self.lbl_output_path = QLabel("结果路径：未运行")
        self.lbl_data_path.setStyleSheet("color: gray;")
        self.lbl_output_path.setStyleSheet("color: gray;")
        self.info_layout.addWidget(self.lbl_data_path)
        self.info_layout.addStretch()
        self.info_layout.addWidget(self.lbl_output_path)
        self.main_layout.addLayout(self.info_layout)

        # 导入已有结果：跳过数据、预处理、特征选择
        import_group = QGroupBox("导入已有结果（跳过数据、预处理、特征选择，直接跑模型）")
        import_layout = QFormLayout(import_group)

        self._import_processed_path = None
        self._import_consensus_path = None

        btn_proc_layout = QHBoxLayout()
        self.lbl_import_processed = QLabel("未选择")
        self.lbl_import_processed.setStyleSheet("color: gray;")
        btn_proc = QPushButton("选择预处理数据 CSV")
        btn_proc.clicked.connect(self._select_import_processed)
        btn_proc_layout.addWidget(btn_proc)
        btn_proc_layout.addWidget(self.lbl_import_processed, stretch=1)
        import_layout.addRow("预处理数据：", btn_proc_layout)

        btn_cons_layout = QHBoxLayout()
        self.lbl_import_consensus = QLabel("未选择")
        self.lbl_import_consensus.setStyleSheet("color: gray;")
        btn_cons = QPushButton("选择共识特征表 CSV")
        btn_cons.clicked.connect(self._select_import_consensus)
        btn_cons_layout.addWidget(btn_cons)
        btn_cons_layout.addWidget(self.lbl_import_consensus, stretch=1)
        import_layout.addRow("共识特征表：", btn_cons_layout)

        self.line_outcome_col = QLineEdit()
        self.line_outcome_col.setPlaceholderText("例如：All_cause_death 或 y")
        import_layout.addRow("结局变量列名：", self.line_outcome_col)

        self.btn_import = QPushButton("导入并保存到当前实验")
        self.btn_import.clicked.connect(self._import_precomputed_results)
        import_layout.addRow(self.btn_import)

        self.main_layout.addWidget(import_group)

        # 主分割器：上半部分配置，下半部分结果
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ---------- 上半部分 ----------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 方法选择 + 共识阈值 + 稳定性
        method_group = QGroupBox("选择方法")
        method_layout = QHBoxLayout(method_group)

        self.method_list = CheckableListWidget(title="特征选择方法")
        self.method_list.add_items(list(DEFAULT_PARAMS.keys()))
        method_layout.addWidget(self.method_list, stretch=2)

        right_layout = QVBoxLayout()
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("共识阈值 (≥k):"))
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 7)
        self.spin_threshold.setValue(3)
        thresh_layout.addWidget(self.spin_threshold)
        right_layout.addLayout(thresh_layout)

        self.check_stability = QPushButton("运行稳定性分析（5-fold Jaccard）")
        self.check_stability.setCheckable(True)
        right_layout.addWidget(self.check_stability)
        method_layout.addLayout(right_layout, stretch=1)
        top_layout.addWidget(method_group)

        # 可折叠超参数配置
        self.params_collapsible = CollapsibleGroupBox("方法超参数配置（点击展开/收起）")
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.param_widgets = {}

        for method, params in DEFAULT_PARAMS.items():
            gb = QGroupBox(method)
            form = QFormLayout(gb)
            widgets = {}
            for k, v in params.items():
                if isinstance(v, int):
                    w = QSpinBox()
                    w.setRange(1, 100000)
                    w.setValue(v)
                elif isinstance(v, float):
                    w = QDoubleSpinBox()
                    # 根据参数语义设置合理范围
                    if "threshold" in k.lower():
                        w.setRange(0.0, 1.0)
                        w.setSingleStep(1e-6)
                        w.setDecimals(8)
                    elif "C_min" in k or "C_max" in k or "C" == k:
                        w.setRange(1e-10, 1e10)
                        w.setSingleStep(0.001)
                        w.setDecimals(10)
                    elif k in {"cx_prob", "mut_prob"}:
                        w.setRange(0.0, 1.0)
                        w.setSingleStep(0.01)
                        w.setDecimals(3)
                    elif k in {"threshold_in", "threshold_out"}:
                        w.setRange(0.0, 1.0)
                        w.setSingleStep(0.01)
                        w.setDecimals(3)
                    else:
                        w.setRange(0.0, 1000.0)
                        w.setSingleStep(0.01)
                        w.setDecimals(3)
                    w.setValue(v)
                else:
                    w = QLineEdit(str(v))
                # 绑定悬停提示
                tooltip = PARAM_TOOLTIPS.get(method, {}).get(k, "")
                if tooltip:
                    w.setToolTip(tooltip)
                widgets[k] = w
                form.addRow(k, w)
            self.param_widgets[method] = widgets
            self.params_layout.addWidget(gb)

        self.params_collapsible.set_content(self.params_widget)
        top_layout.addWidget(self.params_collapsible)

        # 运行按钮
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("运行特征选择")
        self.btn_run.clicked.connect(self.run_feature_selection)
        self.btn_show_config = QPushButton("查看当前配置")
        self.btn_show_config.clicked.connect(self.show_config)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_show_config)
        run_layout.addStretch()
        top_layout.addLayout(run_layout)

        self.lbl_status = QLabel("未运行")
        top_layout.addWidget(self.lbl_status)

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(120)
        top_layout.addWidget(self.config_text)

        splitter.addWidget(top_widget)

        # ---------- 下半部分 ----------
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # 后处理面板
        post_group = QGroupBox("共识结果后处理（可勾选“手动包含”列以单独保留/剔除特征，再点击“应用后处理”覆盖原结果）")
        post_layout = QHBoxLayout(post_group)
        post_layout.addWidget(QLabel("新共识阈值:"))
        self.spin_post_threshold = QSpinBox()
        self.spin_post_threshold.setRange(1, 7)
        self.spin_post_threshold.setValue(3)
        post_layout.addWidget(self.spin_post_threshold)
        self.btn_recompute = QPushButton("仅按阈值重新计算")
        self.btn_recompute.clicked.connect(self.recompute_consensus)
        self.btn_apply_post = QPushButton("应用后处理（阈值+手动勾选）")
        self.btn_apply_post.clicked.connect(self.apply_postprocessing)
        self.btn_save_consensus = QPushButton("保存当前共识结果")
        self.btn_save_consensus.clicked.connect(self.save_current_consensus)
        self.btn_manual_save = QPushButton("手动保存到...")
        self.btn_manual_save.clicked.connect(self.manual_save_consensus)
        post_layout.addWidget(self.btn_recompute)
        post_layout.addWidget(self.btn_apply_post)
        post_layout.addWidget(self.btn_save_consensus)
        post_layout.addWidget(self.btn_manual_save)
        post_layout.addStretch()
        bottom_layout.addWidget(post_group)

        self.result_table = QTableWidget()
        bottom_layout.addWidget(self.result_table)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([500, 400])

        self.main_layout.addWidget(splitter)

    def get_method_params(self) -> dict:
        """收集各方法参数。"""
        params = {}
        for method, widgets in self.param_widgets.items():
            method_params = {}
            for k, w in widgets.items():
                if isinstance(w, QSpinBox):
                    method_params[k] = w.value()
                elif isinstance(w, QDoubleSpinBox):
                    method_params[k] = w.value()
                else:
                    val = w.text().strip()
                    try:
                        method_params[k] = int(val)
                    except ValueError:
                        try:
                            method_params[k] = float(val)
                        except ValueError:
                            method_params[k] = val
            params[method] = method_params
        return params

    def show_config(self):
        config = {
            "methods": self.method_list.get_checked(),
            "consensus_threshold": self.spin_threshold.value(),
            "method_params": self.get_method_params(),
        }
        self.config_text.setText(json.dumps(config, indent=2, ensure_ascii=False))

    def _get_data_paths(self):
        """返回预处理数据路径与特征选择输出目录。"""
        if self.parent_window.current_experiment_id is None:
            return None, None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        prep_config = config.get("preprocessing")
        if not prep_config or not prep_config.get("processed_path"):
            return None, None
        processed_path = prep_config["processed_path"]
        output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "feature_selection"
        return processed_path, output_dir

    def _load_X_y(self):
        """加载特征选择用的 X, y。"""
        processed_path, _ = self._get_data_paths()
        if processed_path is None:
            return None, None, None

        df = pd.read_csv(processed_path)
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        roles = exp.get("config", {}).get("roles", {})
        y_col = [c for c, r in roles.items() if r == "outcome"]
        if not y_col:
            return None, None, None
        y_col = y_col[0]
        feature_cols = [c for c in df.columns if c != y_col]
        return df[feature_cols], df[y_col], y_col

    def run_feature_selection(self, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        processed_path, output_dir = self._get_data_paths()
        if processed_path is None:
            QMessageBox.warning(self, "警告", "请先完成预处理")
            return

        X, y, y_col = self._load_X_y()
        if X is None:
            QMessageBox.warning(self, "警告", "未设置 outcome 角色")
            return

        methods = self.method_list.get_checked()
        if not methods:
            QMessageBox.warning(self, "警告", "请至少选择一种特征选择方法")
            return

        # 上游变动：归档全部下游结果（旧结果保留在 archive/ 中）
        from gui.widgets.result_utils import archive_downstream_results
        archived = archive_downstream_results(
            self.parent_window.db, self.parent_window.current_experiment_id,
            trigger="feature_selection", logger=self.parent_window.logger,
        )
        if archived:
            self.parent_window.log_message(f"特征选择重跑：已归档上一轮下游结果 {archived}")

        try:
            method_params = self.get_method_params()
            method_params = {m: method_params.get(m, {}) for m in methods}

            self.lbl_data_path.setText(f"数据路径：{processed_path}")
            self.lbl_output_path.setText(f"结果路径：{output_dir}")
            self.lbl_status.setText("特征选择运行中...")
            self.parent_window.log_message(f"特征选择开始：使用数据 {processed_path}，方法 {methods}")

            runner = FeatureSelectionRunner(
                methods=methods,
                consensus_threshold=self.spin_threshold.value(),
                random_state=42,
                method_params=method_params,
            )
            table = runner.run(X, y)

            stability_report = None
            if self.check_stability.isChecked():
                self.lbl_status.setText("正在计算稳定性...")
                stability_report = self._compute_stability(X, y, methods, method_params)

            output_dir.mkdir(parents=True, exist_ok=True)
            table_path = output_dir / "consensus_table.csv"
            table.to_csv(table_path, index=False)

            # 保存各方法选择摘要
            summary_records = []
            for method in methods:
                selected = runner.results.get(method, [])
                summary_records.append({
                    "method": method,
                    "n_selected": len(selected),
                    "selected_features": "|".join(selected),
                    "hyperparameters": json.dumps(method_params.get(method, {}), ensure_ascii=False),
                })
            summary_path = output_dir / "feature_selection_summary.csv"
            pd.DataFrame(summary_records).to_csv(summary_path, index=False)

            fs_config = {
                "methods": methods,
                "consensus_threshold": self.spin_threshold.value(),
                "method_params": method_params,
                "random_state": 42,
                "processed_path": processed_path,
                "consensus_table_path": str(table_path),
                "feature_selection_summary_path": str(summary_path),
            }
            if stability_report is not None:
                stability_path = output_dir / "stability_report.csv"
                stability_report.to_csv(stability_path, index=False)
                fs_config["stability_report_path"] = str(stability_path)

            self._save_consensus_features(table, output_dir, fs_config, threshold=self.spin_threshold.value())

            self.lbl_status.setText(
                f"特征选择完成：{len(table)} 个特征，共识特征 {table['consensus'].sum()} 个"
                + (f"，稳定性报告已生成" if stability_report is not None else "")
            )
            self.parent_window.log_message(f"特征选择完成：共识特征 {table['consensus'].sum()} 个，保存至 {output_dir}")
            self.show_config()
            self.update_table(table)
            if not silent:
                QMessageBox.information(self, "成功", "特征选择完成")
            # 上游变动后：刷新所有下游选项卡
            self.parent_window._refresh_downstream_views()

        except Exception as e:
            import traceback
            detail = traceback.format_exc()
            self.parent_window.logger.error(f"特征选择失败：{detail}")
            if not silent:
                QMessageBox.critical(self, "失败", f"{e}\n\n详细错误已写入日志。")

    def _compute_stability(self, X, y, methods, method_params, n_splits=5):
        """计算特征选择稳定性：多次 CV 下各方法选中特征的 Jaccard 相似度。"""
        from sklearn.model_selection import StratifiedKFold
        import numpy as np

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        records = []

        for method in methods:
            selected_sets = []
            for train_idx, _ in skf.split(X, y):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                runner = FeatureSelectionRunner(
                    methods=[method],
                    random_state=42,
                    method_params={method: method_params.get(method, {})},
                )
                runner.run(X_train, y_train)
                selected = set(runner.results.get(method, []))
                selected_sets.append(selected)

            jaccards = []
            for i in range(len(selected_sets)):
                for j in range(i + 1, len(selected_sets)):
                    a, b = selected_sets[i], selected_sets[j]
                    inter = len(a & b)
                    union = len(a | b)
                    jaccards.append(inter / union if union > 0 else 0.0)

            avg_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
            records.append({
                "method": method,
                "avg_jaccard": round(avg_jaccard, 4),
                "mean_n_features": round(float(np.mean([len(s) for s in selected_sets])), 2),
            })

        return pd.DataFrame(records)

    def _get_table_from_ui(self) -> pd.DataFrame:
        """从当前 UI 表格读取共识表（含手动包含列）。"""
        n_rows = self.result_table.rowCount()
        n_cols = self.result_table.columnCount()
        headers = [self.result_table.horizontalHeaderItem(j).text() for j in range(n_cols)]
        records = []
        for i in range(n_rows):
            row = {}
            for j in range(n_cols):
                item = self.result_table.item(i, j)
                header = headers[j]
                if header == "手动包含":
                    row[header] = 1 if item.checkState() == Qt.CheckState.Checked else 0
                else:
                    val = item.text()
                    # 尽量恢复数值类型
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    row[header] = val
            records.append(row)
        return pd.DataFrame(records)

    def _apply_manual_consensus(self, table: pd.DataFrame) -> pd.DataFrame:
        """根据 UI 表格的手动包含列覆盖 consensus 列。"""
        if "手动包含" in table.columns:
            table = table.copy()
            table["consensus"] = table["手动包含"].astype(int)
            table = table.drop(columns=["手动包含"])
        return table

    def recompute_consensus(self):
        """基于已有共识表，用新阈值重新计算共识特征。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        _, output_dir = self._get_data_paths()
        table_path = output_dir / "consensus_table.csv"
        if not table_path.exists():
            QMessageBox.warning(self, "警告", "尚未生成共识表，请先运行特征选择")
            return

        table = pd.read_csv(table_path)
        new_threshold = self.spin_post_threshold.value()
        table["consensus"] = (table["n_methods_selected"] >= new_threshold).astype(int)
        table.to_csv(table_path, index=False)

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        fs_config = config.get("feature_selection") or {}
        fs_config["consensus_threshold"] = new_threshold
        self._save_consensus_features(table, output_dir, fs_config, threshold=new_threshold)

        self.lbl_status.setText(f"已重新计算共识：阈值={new_threshold}，共识特征 {table['consensus'].sum()} 个")
        self.parent_window.log_message(f"共识阈值调整为 {new_threshold}，共识特征 {table['consensus'].sum()} 个")
        self.update_table(table)

    def apply_postprocessing(self):
        """根据当前阈值 + 表格手动勾选状态，重新生成并覆盖共识结果。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        _, output_dir = self._get_data_paths()
        table_path = output_dir / "consensus_table.csv"
        if not table_path.exists():
            QMessageBox.warning(self, "警告", "尚未生成共识表，请先运行特征选择")
            return

        # 1. 先按阈值更新 consensus
        table = pd.read_csv(table_path)
        new_threshold = self.spin_post_threshold.value()
        table["consensus"] = (table["n_methods_selected"] >= new_threshold).astype(int)

        # 2. 用 UI 表格中的手动勾选状态覆盖
        ui_table = self._get_table_from_ui()
        if "手动包含" in ui_table.columns and not ui_table.empty:
            # 按 feature 名合并手动勾选状态
            manual_map = dict(zip(ui_table["feature"].astype(str), ui_table["手动包含"].astype(int)))
            table["consensus"] = table["feature"].astype(str).map(manual_map).fillna(table["consensus"]).astype(int)

        # 3. 覆盖保存
        table.to_csv(table_path, index=False)

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        fs_config = config.get("feature_selection") or {}
        fs_config["consensus_threshold"] = new_threshold
        # 记录手动覆盖信息
        fs_config["postprocessed"] = True
        if "手动包含" in ui_table.columns:
            fs_config["manual_included"] = ui_table.loc[ui_table["手动包含"] == 1, "feature"].astype(str).tolist()
            fs_config["manual_excluded"] = ui_table.loc[ui_table["手动包含"] == 0, "feature"].astype(str).tolist()
        self._save_consensus_features(table, output_dir, fs_config, threshold=new_threshold)

        n_consensus = int(table["consensus"].sum())
        self.lbl_status.setText(f"后处理已应用：阈值={new_threshold}，共识特征 {n_consensus} 个")
        self.parent_window.log_message(f"共识后处理已应用：阈值={new_threshold}，共识特征 {n_consensus} 个")
        self.update_table(table)
        QMessageBox.information(self, "成功", f"后处理已应用，共识特征 {n_consensus} 个，已覆盖原共识结果。")

    def save_current_consensus(self):
        """保存当前阈值/手动勾选下的共识特征到实验目录。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        _, output_dir = self._get_data_paths()
        table_path = output_dir / "consensus_table.csv"
        if not table_path.exists():
            QMessageBox.warning(self, "警告", "尚未生成共识表")
            return
        # 优先使用 UI 表格中的手动勾选状态
        ui_table = self._get_table_from_ui()
        if "手动包含" in ui_table.columns and not ui_table.empty:
            table = self._apply_manual_consensus(ui_table)
        else:
            table = pd.read_csv(table_path)
        self._save_consensus_features(table, output_dir, None)
        QMessageBox.information(self, "成功", f"共识特征已保存至 {output_dir / 'consensus_features.csv'}")

    def manual_save_consensus(self):
        """手动选择路径保存共识特征列表。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        _, output_dir = self._get_data_paths()
        table_path = output_dir / "consensus_table.csv"
        if not table_path.exists():
            QMessageBox.warning(self, "警告", "尚未生成共识表")
            return

        path, _ = QFileDialog.getSaveFileName(self, "保存共识特征", str(output_dir / "consensus_features.csv"), "CSV (*.csv)")
        if not path:
            return
        # 优先使用 UI 表格中的手动勾选状态
        ui_table = self._get_table_from_ui()
        if "手动包含" in ui_table.columns and not ui_table.empty:
            table = self._apply_manual_consensus(ui_table)
        else:
            table = pd.read_csv(table_path)
        consensus_features = table.loc[table["consensus"] == 1, "feature"].tolist()
        pd.DataFrame({"feature": consensus_features}).to_csv(path, index=False)
        QMessageBox.information(self, "成功", f"已保存至 {path}")

    def _save_consensus_features(self, table: pd.DataFrame, output_dir: Path, fs_config: dict = None, threshold: int = None):
        """保存共识特征列表并更新数据库。"""
        consensus_features = table.loc[table["consensus"] == 1, "feature"].tolist()
        feature_path = output_dir / "consensus_features.csv"
        pd.DataFrame({"feature": consensus_features}).to_csv(feature_path, index=False)

        if fs_config is not None and self.parent_window.current_experiment_id is not None:
            fs_config["consensus_features_path"] = str(feature_path)
            fs_config["consensus_threshold"] = threshold if threshold is not None else self.spin_threshold.value()
            exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
            config = exp.get("config") or {}
            config["feature_selection"] = fs_config
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id,
                {"config": config, "status": "feature_selected"}
            )

    def _select_import_processed(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择预处理数据 CSV", "", "CSV (*.csv)")
        if path:
            self._import_processed_path = path
            self.lbl_import_processed.setText(Path(path).name)
            self.lbl_import_processed.setStyleSheet("color: black;")

    def _select_import_consensus(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择共识特征表 CSV", "", "CSV (*.csv)")
        if path:
            self._import_consensus_path = path
            self.lbl_import_consensus.setText(Path(path).name)
            self.lbl_import_consensus.setStyleSheet("color: black;")

    def _import_precomputed_results(self):
        """导入已有的预处理数据和共识特征表，直接用于后续建模。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        if not self._import_processed_path or not self._import_consensus_path:
            QMessageBox.warning(self, "警告", "请选择预处理数据 CSV 和共识特征表 CSV")
            return

        processed_path = Path(self._import_processed_path)
        consensus_path = Path(self._import_consensus_path)
        outcome_col = self.line_outcome_col.text().strip()

        if not processed_path.exists():
            QMessageBox.warning(self, "警告", f"预处理数据不存在：{processed_path}")
            return
        if not consensus_path.exists():
            QMessageBox.warning(self, "警告", f"共识特征表不存在：{consensus_path}")
            return
        if not outcome_col:
            QMessageBox.warning(self, "警告", "请输入结局变量列名")
            return

        try:
            df = pd.read_csv(processed_path)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"读取预处理数据失败：{e}")
            return

        if outcome_col not in df.columns:
            QMessageBox.warning(self, "警告", f"预处理数据中找不到结局变量列：{outcome_col}\n现有列：{list(df.columns)}")
            return

        try:
            fs_table = pd.read_csv(consensus_path)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"读取共识特征表失败：{e}")
            return

        # 支持两种格式：只有 feature 列，或带 consensus 标志的完整表
        if "feature" not in fs_table.columns:
            QMessageBox.warning(self, "警告", "共识特征表必须包含 'feature' 列")
            return
        if "consensus" in fs_table.columns:
            feature_cols = fs_table.loc[fs_table["consensus"] == 1, "feature"].tolist()
        else:
            feature_cols = fs_table["feature"].dropna().astype(str).tolist()

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            QMessageBox.warning(self, "警告", f"以下共识特征不在预处理数据中：{missing[:10]}")
            return

        # 复制到当前实验目录，避免原文件被移动/删除后失效
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
        prep_dir = output_dir / "preprocessing"
        fs_dir = output_dir / "feature_selection"
        prep_dir.mkdir(parents=True, exist_ok=True)
        fs_dir.mkdir(parents=True, exist_ok=True)

        local_processed = prep_dir / "preprocessed.csv"
        local_consensus = fs_dir / "consensus_table.csv"
        local_features = fs_dir / "consensus_features.csv"

        df.to_csv(local_processed, index=False)

        # 生成标准共识表格式
        consensus_df = pd.DataFrame({"feature": feature_cols, "consensus": 1})
        consensus_df.to_csv(local_consensus, index=False)
        consensus_df[["feature"]].to_csv(local_features, index=False)

        # 更新配置
        config = exp.get("config") or {}
        config["preprocessing"] = {
            "processed_path": str(local_processed),
            "n_features": len([c for c in df.columns if c != outcome_col]),
            "n_samples": len(df),
        }
        config["feature_selection"] = {
            "methods": ["imported"],
            "consensus_threshold": 1,
            "consensus_table_path": str(local_consensus),
            "consensus_features_path": str(local_features),
        }
        roles = config.get("roles", {})
        roles[outcome_col] = "outcome"
        config["roles"] = roles

        self.parent_window.db.update_experiment(
            self.parent_window.current_experiment_id,
            {"config": config, "status": "feature_selected"}
        )

        self.lbl_data_path.setText(f"数据路径：{local_processed}")
        self.lbl_output_path.setText(f"结果路径：{fs_dir}")
        self.lbl_status.setText(f"已导入：共识特征 {len(feature_cols)} 个，样本 {len(df)} 例")
        self.parent_window.log_message(
            f"导入已有结果：预处理数据 {local_processed}，共识特征 {len(feature_cols)} 个，结局变量 {outcome_col}"
        )
        QMessageBox.information(
            self, "成功",
            f"已导入并保存到当前实验：\n预处理：{local_processed}\n共识特征：{len(feature_cols)} 个\n结局变量：{outcome_col}\n\n现在可以直接去「模型」标签页跑模型。"
        )

    def update_table(self, df: pd.DataFrame):
        """更新可勾选共识表。"""
        display_df = df.head(500)
        self.result_table.setRowCount(display_df.shape[0])
        # 额外增加一列“手动包含”复选框
        cols = display_df.columns.tolist() + ["手动包含"]
        self.result_table.setColumnCount(len(cols))
        self.result_table.setHorizontalHeaderLabels(cols)
        for i in range(display_df.shape[0]):
            for j in range(display_df.shape[1]):
                val = display_df.iat[i, j]
                item = QTableWidgetItem(str(val))
                # feature 名与方法名列只读
                if display_df.columns[j] in {"feature", "selected_methods"}:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.result_table.setItem(i, j, item)
            # 手动包含复选框
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            is_consensus = bool(display_df.iat[i, display_df.columns.get_loc("consensus")]) if "consensus" in display_df.columns else False
            checkbox_item.setCheckState(Qt.CheckState.Checked if is_consensus else Qt.CheckState.Unchecked)
            self.result_table.setItem(i, len(display_df.columns), checkbox_item)

    def load_settings(self, exp: dict):
        """从实验配置中恢复特征选择方法和超参数设置。"""
        config = exp.get("config") or {}
        fs_config = config.get("feature_selection")
        if not fs_config:
            return
        # 恢复方法勾选
        methods = fs_config.get("methods", list(DEFAULT_PARAMS.keys()))
        self.method_list.set_checked(methods)
        # 恢复共识阈值
        if "consensus_threshold" in fs_config:
            self.spin_threshold.setValue(fs_config["consensus_threshold"])
            self.spin_post_threshold.setValue(fs_config["consensus_threshold"])
        # 恢复超参数
        saved_params = fs_config.get("method_params", {})
        for method, widgets in self.param_widgets.items():
            params = saved_params.get(method, DEFAULT_PARAMS.get(method, {}))
            for k, w in widgets.items():
                if k not in params:
                    continue
                val = params[k]
                if isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QLineEdit):
                    w.setText(str(val))

    def load_results(self, exp: dict):
        """加载历史特征选择结果到表格。"""
        config = exp.get("config") or {}
        fs_config = config.get("feature_selection")
        if not fs_config or not fs_config.get("consensus_table_path"):
            return
        # 先恢复设置
        self.load_settings(exp)
        table_path = Path(fs_config["consensus_table_path"])
        if not table_path.exists():
            return
        try:
            table = pd.read_csv(table_path)
            self.update_table(table)
            n_consensus = int(table["consensus"].sum()) if "consensus" in table.columns else 0
            self.lbl_status.setText(f"已加载历史结果：{len(table)} 个特征，共识 {n_consensus} 个")
            self.parent_window.log_message(f"特征选择历史结果已加载：{table_path}")
        except Exception as e:
            self.parent_window.logger.warning(f"加载特征选择历史结果失败：{e}")
