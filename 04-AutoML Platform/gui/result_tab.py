"""结果查看标签页（增加全模型 SHAP 分析）。"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog,
    QTextEdit, QGroupBox, QSpinBox, QComboBox, QProgressBar, QCheckBox,
    QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.shap_engine import run_full_shap_analysis, analyze_feature_shap, plot_shap_feature_ranking
from gui.widgets import load_modeling_data
from gui.widgets.result_utils import delete_module_results, archive_module_results


class ShapWorker(QThread):
    """后台运行全模型 SHAP 分析，避免 GUI 冻结。"""
    progress = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_dir: Path, X: pd.DataFrame, exp_dir: Path,
                 sample_size: int = 200, top_n_start: int = 3, top_n_end: int = 3,
                 indicator: str = None, config: dict = None,
                 h2o_enabled: bool = True, h2o_mode: str = "native", y=None,
                 max_parallel: int = 1, skip_existing: bool = True):
        super().__init__()
        self.model_dir = model_dir
        self.X = X
        self.exp_dir = exp_dir
        self.sample_size = sample_size
        self.top_n_start = top_n_start
        self.top_n_end = top_n_end
        self.indicator = indicator
        self.config = config
        self.h2o_enabled = h2o_enabled
        self.h2o_mode = h2o_mode
        self.y = y
        self.max_parallel = max_parallel
        self.skip_existing = skip_existing

    def run(self):
        try:
            def _progress(cur, total, name):
                self.progress.emit(cur, total, name)

            result = run_full_shap_analysis(
                self.model_dir, self.X, self.exp_dir,
                sample_size=self.sample_size,
                top_n_start=self.top_n_start, top_n_end=self.top_n_end,
                indicator=self.indicator, progress_callback=_progress,
                config=self.config,
                h2o_enabled=self.h2o_enabled, h2o_mode=self.h2o_mode, y=self.y,
                max_parallel=self.max_parallel, skip_existing=self.skip_existing,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ResultTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>结果</h2>"))

        # ----------------- 结果刷新区 -----------------
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新结果")
        self.btn_refresh.clicked.connect(self.refresh_results)
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addStretch()

        self.lbl_status = QLabel("点击刷新查看模型排行榜")

        self.result_table = QTableWidget()

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setPlaceholderText("特征选择配置与稳定性报告将显示在这里")

        # ----------------- SHAP 分析区 -----------------
        shap_group = QGroupBox("全模型 SHAP 可解释性分析")
        shap_layout = QVBoxLayout(shap_group)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("样本数:"))
        self.spin_shap_sample = QSpinBox()
        self.spin_shap_sample.setRange(50, 2000)
        self.spin_shap_sample.setSingleStep(50)
        self.spin_shap_sample.setValue(100)
        ctrl_layout.addWidget(self.spin_shap_sample)

        ctrl_layout.addWidget(QLabel("并行模型数:"))
        self.spin_shap_parallel = QSpinBox()
        self.spin_shap_parallel.setRange(1, 4)
        self.spin_shap_parallel.setValue(2)
        self.spin_shap_parallel.setToolTip(
            "同时计算几个模型的 SHAP（进程级并行）。1=串行逐个计算。\n"
            "含 GPU 模型（Transformer）时建议不超过 2。"
        )
        ctrl_layout.addWidget(self.spin_shap_parallel)

        self.check_shap_incremental = QCheckBox("增量计算（跳过已有结果）")
        self.check_shap_incremental.setChecked(True)
        self.check_shap_incremental.setToolTip(
            "勾选后：已有 SHAP summary 文件的模型会被跳过，只计算缺失的模型；\n"
            "取消勾选则重算所有模型。"
        )
        ctrl_layout.addWidget(self.check_shap_incremental)

        ctrl_layout.addWidget(QLabel("Top-N 范围:"))
        self.spin_shap_topn_start = QSpinBox()
        self.spin_shap_topn_start.setRange(1, 10)
        self.spin_shap_topn_start.setValue(1)
        self.spin_shap_topn_start.setToolTip("起始 Top-N：将输出效果最好的前 1 到前 N 个模型的 SHAP 聚合图")
        ctrl_layout.addWidget(self.spin_shap_topn_start)
        ctrl_layout.addWidget(QLabel("到"))
        self.spin_shap_topn_end = QSpinBox()
        self.spin_shap_topn_end.setRange(1, 10)
        self.spin_shap_topn_end.setValue(3)
        self.spin_shap_topn_end.setToolTip("结束 Top-N：例如设置为 10 将同时输出 Top-1 到 Top-10 的图")
        ctrl_layout.addWidget(self.spin_shap_topn_end)

        ctrl_layout.addWidget(QLabel("关注指标:"))
        self.combo_shap_feature = QComboBox()
        self.combo_shap_feature.setEditable(False)
        self.combo_shap_feature.setMinimumWidth(150)
        self.combo_shap_feature.setPlaceholderText("请选择共识特征")
        ctrl_layout.addWidget(self.combo_shap_feature)

        ctrl_layout.addStretch()
        shap_layout.addLayout(ctrl_layout)

        # H2O SHAP 选项
        h2o_layout = QHBoxLayout()
        self.check_h2o_shap = QCheckBox("对 H2O 模型进行 SHAP 分析（默认关闭，H2O 较慢）")
        self.check_h2o_shap.setChecked(False)
        h2o_layout.addWidget(self.check_h2o_shap)
        h2o_layout.addWidget(QLabel("H2O SHAP 方式:"))
        self.combo_h2o_mode = QComboBox()
        self.combo_h2o_mode.addItem("快速（H2O 原生 TreeSHAP / predict_contributions）", "native")
        self.combo_h2o_mode.addItem("排列重要性（快速近似）", "permutation")
        self.combo_h2o_mode.addItem("原始（KernelExplainer，极慢）", "kernel")
        self.combo_h2o_mode.setToolTip(
            "native：树型 H2O 模型用 Java 端 TreeSHAP，最快最准（StackedEnsemble 自动回退排列重要性）；\n"
            "permutation：整表预测、次数有界，快但为近似；\n"
            "kernel：标准 KernelExplainer，每次 predict 需 H2OFrame 转换，极慢。"
        )
        h2o_layout.addWidget(self.combo_h2o_mode)
        h2o_layout.addStretch()
        shap_layout.addLayout(h2o_layout)

        shap_btn_layout = QHBoxLayout()
        self.btn_run_shap = QPushButton("运行全模型 SHAP 分析")
        self.btn_run_shap.clicked.connect(self.run_shap_analysis)
        self.btn_analyze_feature = QPushButton("分析选中指标")
        self.btn_analyze_feature.clicked.connect(self.analyze_selected_feature)
        self.btn_open_shap_dir = QPushButton("打开 SHAP 输出目录")
        self.btn_open_shap_dir.clicked.connect(self.open_shap_dir)
        shap_btn_layout.addWidget(self.btn_run_shap)
        shap_btn_layout.addWidget(self.btn_analyze_feature)
        shap_btn_layout.addWidget(self.btn_open_shap_dir)
        shap_btn_layout.addStretch()
        shap_layout.addLayout(shap_btn_layout)

        # SHAP 结果管理
        shap_manage_layout = QHBoxLayout()
        self.btn_delete_shap = QPushButton("删除 SHAP 结果")
        self.btn_delete_shap.clicked.connect(self.delete_shap_results)
        self.btn_archive_shap = QPushButton("归档 SHAP 结果")
        self.btn_archive_shap.clicked.connect(self.archive_shap_results)
        shap_manage_layout.addWidget(self.btn_delete_shap)
        shap_manage_layout.addWidget(self.btn_archive_shap)
        shap_manage_layout.addStretch()
        shap_layout.addLayout(shap_manage_layout)

        self.shap_progress = QProgressBar()
        self.shap_progress.setRange(0, 100)
        self.shap_progress.setValue(0)
        self.shap_progress.setTextVisible(True)
        shap_layout.addWidget(self.shap_progress)

        self.lbl_shap_status = QLabel("SHAP 分析未运行")
        shap_layout.addWidget(self.lbl_shap_status)

        self.shap_ranking_table = QTableWidget()
        shap_layout.addWidget(QLabel("<b>特征 SHAP 聚合排名</b>"))
        shap_layout.addWidget(self.shap_ranking_table)

        rank_btn_layout = QHBoxLayout()
        self.btn_plot_shap_ranking = QPushButton("绘制特征排名图")
        self.btn_plot_shap_ranking.clicked.connect(self.plot_shap_ranking)
        rank_btn_layout.addWidget(self.btn_plot_shap_ranking)
        rank_btn_layout.addStretch()
        shap_layout.addLayout(rank_btn_layout)

        self.shap_feature_table = QTableWidget()
        shap_layout.addWidget(QLabel("<b>选中指标跨模型 SHAP 分析</b>"))
        shap_layout.addWidget(self.shap_feature_table)

        # 上下可拖拽调节：刷新与状态 / 结果表 / 配置区 / SHAP 区
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addLayout(btn_layout)
        top_layout.addWidget(self.lbl_status)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_container)
        splitter.addWidget(self.result_table)
        splitter.addWidget(self.config_text)
        splitter.addWidget(shap_group)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 2)
        splitter.setChildrenCollapsible(False)
        self.layout.addWidget(splitter)
        self._shap_worker = None

    def refresh_results(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}

        records = []
        metric_cols = ["auc", "auprc", "accuracy", "sensitivity", "specificity",
                       "ppv", "npv", "f1", "kappa", "logloss", "brier"]

        def _append(prefix: str, id_col: str, result_path: str):
            if not result_path or not Path(result_path).exists():
                return
            df = pd.read_csv(result_path)
            for _, row in df.iterrows():
                rec = {"model": f"{prefix}_{row[id_col]}"}
                for m in metric_cols:
                    rec[m] = row.get(m, "")
                # 保留模块特有的额外列（如 c_index）
                for c in df.columns:
                    if c not in rec and c != id_col:
                        rec[c] = row.get(c, "")
                records.append(rec)

        # 基础模型结果
        modeling_config = config.get("modeling")
        if modeling_config and modeling_config.get("output_dir"):
            model_dir = Path(modeling_config["output_dir"])
            for f in sorted(model_dir.glob("*_oof.csv")):
                model_name = f.stem.replace("_oof", "")
                df = pd.read_csv(f)
                from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, brier_score_loss,
                                             average_precision_score, recall_score, precision_score,
                                             cohen_kappa_score, log_loss)
                y_true = df["y_true"].values
                proba = df["oof_proba"].values
                pred = df["oof_pred"].values
                records.append({
                    "model": model_name,
                    "auc": round(roc_auc_score(y_true, proba), 4),
                    "auprc": round(average_precision_score(y_true, proba), 4),
                    "accuracy": round(accuracy_score(y_true, pred), 4),
                    "sensitivity": round(recall_score(y_true, pred, zero_division=0), 4),
                    "specificity": round(recall_score(y_true, pred, pos_label=0, zero_division=0), 4),
                    "ppv": round(precision_score(y_true, pred, zero_division=0), 4),
                    "npv": round(precision_score(1 - y_true, 1 - pred, zero_division=0), 4),
                    "f1": round(f1_score(y_true, pred, zero_division=0), 4),
                    "kappa": round(cohen_kappa_score(y_true, pred), 4),
                    "logloss": round(log_loss(y_true, np.clip(proba, 1e-8, 1 - 1e-8)), 4),
                    "brier": round(brier_score_loss(y_true, proba), 4),
                })

        # 集成结果
        if config.get("ensemble") and config["ensemble"].get("result_path"):
            _append("ensemble", "method", config["ensemble"]["result_path"])

        # Transformer 结果
        if config.get("transformer") and config["transformer"].get("result_path"):
            _append("transformer", "model", config["transformer"]["result_path"])

        # AutoML 结果
        if config.get("automl") and config["automl"].get("result_path"):
            _append("automl", "framework", config["automl"]["result_path"])

        # 生存结果
        if config.get("survival") and config["survival"].get("result_path"):
            _append("survival", "model", config["survival"]["result_path"])

        if not records:
            self.lbl_status.setText("暂无结果，请先运行模型训练或其他模块")
            self.result_table.setRowCount(0)
            return

        df_result = pd.DataFrame(records)
        # 统一列顺序：model + 11 指标 + 其他
        ordered_cols = ["model"] + metric_cols
        other_cols = [c for c in df_result.columns if c not in ordered_cols]
        df_result = df_result[ordered_cols + other_cols]

        # 按 auc 降序
        try:
            df_result["_auc_sort"] = pd.to_numeric(df_result["auc"], errors="coerce")
            df_result = df_result.sort_values("_auc_sort", ascending=False, na_position="last").drop(columns=["_auc_sort"])
        except Exception:
            pass

        self.lbl_status.setText(f"共 {len(records)} 条结果")
        self.update_table(df_result)
        self._show_feature_selection_config(config)
        self._populate_feature_combo()

    def showEvent(self, event):
        """进入选项卡时刷新 SHAP 关注指标下拉框（跟随当前数据）。"""
        super().showEvent(event)
        try:
            self._populate_feature_combo()
        except Exception:
            pass

    def _populate_feature_combo(self):
        """从数据中填充 SHAP 关注指标下拉框。"""
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        try:
            X, _, _ = load_modeling_data(exp, use_feature_selected=True)
        except Exception:
            try:
                X, _, _ = load_modeling_data(exp, use_feature_selected=False)
            except Exception:
                return
        current = self.combo_shap_feature.currentText()
        self.combo_shap_feature.clear()
        self.combo_shap_feature.addItems(list(X.columns))
        self.combo_shap_feature.setCurrentText(current if current else "")

    def _show_feature_selection_config(self, config: dict):
        import json
        fs_config = config.get("feature_selection")
        if not fs_config:
            self.config_text.setText("未找到特征选择配置")
            return

        lines = []
        lines.append("=" * 40)
        lines.append("特征选择配置（已保存，可复现）")
        lines.append("=" * 40)
        lines.append(f"方法：{', '.join(fs_config.get('methods', []))}")
        lines.append(f"共识阈值：{fs_config.get('consensus_threshold')}")
        lines.append(f"随机种子：{fs_config.get('random_state')}")
        lines.append("")
        lines.append("各方法超参数：")
        for method, params in fs_config.get("method_params", {}).items():
            lines.append(f"  {method}: {params}")

        stability_path = fs_config.get("stability_report_path")
        if stability_path and Path(stability_path).exists():
            df = pd.read_csv(stability_path)
            lines.append("")
            lines.append("特征选择稳定性报告（5-fold Jaccard）：")
            lines.append(df.to_string(index=False))
            lines.append("")
            lines.append("说明：avg_jaccard 越接近 1，表示该方法选中的特征在不同数据折上越稳定。")
        else:
            lines.append("")
            lines.append("未运行稳定性分析。")

        self.config_text.setText("\n".join(lines))

    def update_table(self, df: pd.DataFrame):
        self.result_table.setRowCount(df.shape[0])
        self.result_table.setColumnCount(df.shape[1])
        self.result_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.result_table.setItem(i, j, QTableWidgetItem(text))

    # ----------------- SHAP 分析槽函数 -----------------

    def run_shap_analysis(self, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")
        if not modeling_config or not modeling_config.get("output_dir"):
            QMessageBox.warning(self, "警告", "请先完成基础模型训练")
            return

        model_dir = Path(modeling_config["output_dir"])
        if not any(model_dir.glob("*.pkl")):
            QMessageBox.warning(self, "警告", f"未找到模型文件：{model_dir}")
            return

        try:
            X, y, _ = load_modeling_data(exp, use_feature_selected=True)
        except Exception:
            try:
                X, y, _ = load_modeling_data(exp, use_feature_selected=False)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载数据失败：{e}")
                return

        exp_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
        indicator = self.combo_shap_feature.currentText().strip() or None

        self._shap_silent = silent
        self.btn_run_shap.setEnabled(False)
        self.lbl_shap_status.setText("SHAP 分析开始...")
        self.shap_progress.setValue(0)

        start_n = self.spin_shap_topn_start.value()
        end_n = self.spin_shap_topn_end.value()
        if start_n > end_n:
            start_n, end_n = end_n, start_n
        self._shap_worker = ShapWorker(
            model_dir, X, exp_dir,
            sample_size=self.spin_shap_sample.value(),
            top_n_start=start_n, top_n_end=end_n,
            indicator=indicator,
            config=config,
            h2o_enabled=self.check_h2o_shap.isChecked(),
            h2o_mode=self.combo_h2o_mode.currentData(),
            y=y,
            max_parallel=self.spin_shap_parallel.value(),
            skip_existing=self.check_shap_incremental.isChecked(),
        )
        self._shap_worker.progress.connect(self._on_shap_progress)
        self._shap_worker.finished_signal.connect(self._on_shap_finished)
        self._shap_worker.error.connect(self._on_shap_error)
        self._shap_worker.start()

    def _on_shap_progress(self, cur: int, total: int, name: str):
        pct = int(cur / total * 100)
        self.shap_progress.setValue(pct)
        self.lbl_shap_status.setText(f"正在计算 SHAP：{name} ({cur}/{total})")
        self.parent_window.log_message(f"SHAP 进度：{name} ({cur}/{total})")

    def _on_shap_finished(self, result: dict):
        self.btn_run_shap.setEnabled(True)
        self.shap_progress.setValue(100)
        self.lbl_shap_status.setText(
            f"SHAP 分析完成：{result['meta']['n_models_success']}/"
            f"{result['meta']['n_models_total']} 个模型成功，输出至 {result['shap_dir']}"
        )
        self.parent_window.log_message(f"SHAP 分析完成：{result['shap_dir']}")
        self._load_shap_ranking(result["shap_dir"])
        if result["meta"].get("indicator"):
            self._load_feature_analysis(result["shap_dir"], result["meta"]["indicator"])
        QMessageBox.information(self, "成功", "全模型 SHAP 分析完成")

    def _on_shap_error(self, msg: str):
        self.btn_run_shap.setEnabled(True)
        self.lbl_shap_status.setText(f"SHAP 分析失败：{msg}")
        self.parent_window.logger.error(f"SHAP 分析失败：{msg}")
        QMessageBox.critical(self, "失败", msg)

    def _load_shap_ranking(self, shap_dir: str):
        path = Path(shap_dir) / "shap_ranking_all_models.csv"
        if not path.exists():
            # 兼容旧版本命名
            path = Path(shap_dir) / "shap_ranking_features.csv"
        if path.exists():
            df = pd.read_csv(path)
            self._update_shap_ranking_table(df)

    def _update_shap_ranking_table(self, df: pd.DataFrame):
        self.shap_ranking_table.setRowCount(df.shape[0])
        self.shap_ranking_table.setColumnCount(df.shape[1])
        self.shap_ranking_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.shap_ranking_table.setItem(i, j, QTableWidgetItem(text))

    def plot_shap_ranking(self):
        """根据当前聚合排名表绘制特征 SHAP 排名图。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        shap_dir = self._get_shap_dir()
        if shap_dir is None or not shap_dir.exists():
            QMessageBox.warning(self, "警告", "请先运行全模型 SHAP 分析")
            return

        ranking_path = shap_dir / "shap_ranking_all_models.csv"
        if not ranking_path.exists():
            ranking_path = shap_dir / "shap_ranking_features.csv"
        if not ranking_path.exists():
            QMessageBox.warning(self, "警告", "未找到特征 SHAP 聚合排名表")
            return

        output_path = shap_dir / "figure_shap_feature_ranking.png"
        try:
            fig_path = plot_shap_feature_ranking(ranking_path, output_path)
            self.lbl_shap_status.setText(f"特征排名图已保存：{fig_path}")
            self.parent_window.log_message(f"特征 SHAP 聚合排名图：{fig_path}")
            # 尝试用系统默认程序打开图片
            import os
            if hasattr(os, "startfile"):
                os.startfile(str(fig_path))
        except Exception as e:
            QMessageBox.critical(self, "失败", f"绘制特征排名图失败：{e}")
            self.parent_window.logger.error(f"绘制特征排名图失败：{e}")

    def analyze_selected_feature(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        feature = self.combo_shap_feature.currentText().strip()
        if not feature:
            QMessageBox.warning(self, "警告", "请选择一个指标")
            return

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        exp_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
        shap_dir = exp_dir / "shap"
        modeling_config = exp.get("config", {}).get("modeling")
        if not modeling_config or not modeling_config.get("output_dir"):
            QMessageBox.warning(self, "警告", "请先完成基础模型训练")
            return
        model_dir = Path(modeling_config["output_dir"])

        if not (shap_dir / "shap_ranking_all_models.csv").exists() and not (shap_dir / "shap_ranking_features.csv").exists():
            QMessageBox.warning(self, "警告", "请先运行全模型 SHAP 分析")
            return

        try:
            X, _, _ = load_modeling_data(exp, use_feature_selected=True)
        except Exception:
            try:
                X, _, _ = load_modeling_data(exp, use_feature_selected=False)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载数据失败：{e}")
                return

        try:
            fig_path = analyze_feature_shap(shap_dir, model_dir, feature, shap_dir, feature_columns=list(X.columns))
            self.lbl_shap_status.setText(f"指标 {feature} 分析完成：{fig_path}")
            self._load_feature_analysis(str(shap_dir), feature)
        except Exception as e:
            QMessageBox.critical(self, "失败", str(e))

    def _load_feature_analysis(self, shap_dir: str, feature: str):
        path = Path(shap_dir) / f"shap_analysis_{feature}.csv"
        if path.exists():
            df = pd.read_csv(path)
            self._update_shap_feature_table(df)

    def _update_shap_feature_table(self, df: pd.DataFrame):
        self.shap_feature_table.setRowCount(df.shape[0])
        self.shap_feature_table.setColumnCount(df.shape[1])
        self.shap_feature_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.shap_feature_table.setItem(i, j, QTableWidgetItem(text))

    def _get_shap_dir(self) -> Optional[Path]:
        if self.parent_window.current_experiment_id is None:
            return None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        exp_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
        return exp_dir / "shap"

    def _clear_shap_tables(self):
        self.shap_ranking_table.setRowCount(0)
        self.shap_ranking_table.setColumnCount(0)
        self.shap_feature_table.setRowCount(0)
        self.shap_feature_table.setColumnCount(0)

    def delete_shap_results(self):
        shap_dir = self._get_shap_dir()
        if shap_dir is None or not shap_dir.exists():
            QMessageBox.warning(self, "警告", "当前实验没有可删除的 SHAP 结果")
            return
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除以下目录及其所有内容？\n{shap_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        ok, msg = delete_module_results(
            shap_dir, table_widget=None,
            status_label=self.lbl_shap_status, logger=self.parent_window.logger,
        )
        if ok:
            self._clear_shap_tables()
            self.parent_window.log_message(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def archive_shap_results(self):
        shap_dir = self._get_shap_dir()
        if shap_dir is None or not shap_dir.exists():
            QMessageBox.warning(self, "警告", "当前实验没有可归档的 SHAP 结果")
            return
        ok, msg = archive_module_results(shap_dir, logger=self.parent_window.logger)
        if ok:
            self._clear_shap_tables()
            self.lbl_shap_status.setText("SHAP 结果已归档")
            self.parent_window.log_message(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def load_results(self, exp: dict):
        """加载历史 SHAP 分析结果。"""
        self._populate_feature_combo()
        exp_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}"
        shap_dir = exp_dir / "shap"
        if not shap_dir.exists():
            return
        self._load_shap_ranking(str(shap_dir))
        meta_path = shap_dir / "shap_meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            indicator = meta.get("indicator")
            if indicator:
                self.combo_shap_feature.setCurrentText(indicator)
                self._load_feature_analysis(str(shap_dir), indicator)
            self.lbl_shap_status.setText(
                f"已加载历史 SHAP：{meta.get('n_models_success', 0)}/"
                f"{meta.get('n_models_total', 0)} 个模型成功"
            )

    def open_shap_dir(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        shap_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        import os
        os.startfile(str(shap_dir)) if hasattr(os, "startfile") else None
