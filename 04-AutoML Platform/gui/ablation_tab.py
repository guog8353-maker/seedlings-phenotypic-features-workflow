"""消融分析选项卡：单变量回加消融（add-back ablation）。

检验被丢弃（drop）或未被共识选中的变量中，是否存在能提升当前最优
模型性能的被忽略协变量。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox,
    QSpinBox, QComboBox, QGroupBox, QRadioButton, QProgressBar,
    QTextEdit, QSplitter, QTabWidget, QFileDialog,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import ablation as abl


class AblationWorker(QThread):
    """后台执行消融分析，避免 GUI 冻结。"""
    progress = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, exp, models, variables, hyperparam_strategy, use_delong,
                 cv_folds, n_trials, pred_dir, skip_pairs=None):
        super().__init__()
        self.exp = exp
        self.models = models
        self.variables = variables
        self.hyperparam_strategy = hyperparam_strategy
        self.use_delong = use_delong
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.pred_dir = pred_dir
        self.skip_pairs = skip_pairs or set()
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            def _cb(cur, total, msg):
                if self._abort:
                    raise InterruptedError("用户取消")
                self.progress.emit(cur, total, msg)

            result = abl.run_ablation(
                self.exp, self.models, self.variables,
                hyperparam_strategy=self.hyperparam_strategy,
                use_delong=self.use_delong,
                cv_folds=self.cv_folds,
                n_trials=self.n_trials,
                pred_dir=self.pred_dir,
                skip_pairs=self.skip_pairs,
                progress_callback=_cb,
            )
            self.finished_signal.emit(result)
        except InterruptedError:
            self.error.emit("已取消")
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class AblationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)
        self._worker = None
        self._result = pd.DataFrame()
        self._exp = None
        self._roc_ok = None

        self.layout.addWidget(QLabel("<h2>消融分析（单变量回加）</h2>"))
        intro = QLabel(
            "用途：检验被丢弃或未被共识选中的变量是否能提升当前最优模型。"
            "建议先小范围试跑（如 Top-5 模型 × 5 个变量），N 与 X 不宜同时过大。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        self.layout.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ---------------- 上半部分：配置 ----------------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Base 模型选择
        model_group = QGroupBox("1. 选择 Base 模型（按原 AUC 排序；集成模型不支持消融已禁用）")
        model_layout = QVBoxLayout(model_group)
        model_toolbar = QHBoxLayout()
        model_toolbar.addWidget(QLabel("一键选择 Top-N:"))
        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(1, 20)
        self.spin_topn.setValue(5)
        model_toolbar.addWidget(self.spin_topn)
        self.btn_topn = QPushButton("选择 Top-N")
        self.btn_topn.clicked.connect(self._select_topn)
        model_toolbar.addWidget(self.btn_topn)
        self.btn_model_all = QPushButton("全选")
        self.btn_model_all.clicked.connect(lambda: self._set_table_checked(self.model_table, True))
        self.btn_model_clear = QPushButton("清空")
        self.btn_model_clear.clicked.connect(lambda: self._set_table_checked(self.model_table, False))
        model_toolbar.addWidget(self.btn_model_all)
        model_toolbar.addWidget(self.btn_model_clear)
        model_toolbar.addStretch()
        model_layout.addLayout(model_toolbar)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["选择", "模型", "来源", "原 AUC"])
        self.model_table.setMaximumHeight(160)
        model_layout.addWidget(self.model_table)
        top_layout.addWidget(model_group)

        # 候选变量选择
        var_group = QGroupBox("2. 选择候选变量（drop + 未被共识选中）")
        var_layout = QVBoxLayout(var_group)
        var_toolbar = QHBoxLayout()
        var_toolbar.addWidget(QLabel("一键选择 X 个:"))
        self.spin_x = QSpinBox()
        self.spin_x.setRange(1, 100)
        self.spin_x.setValue(5)
        var_toolbar.addWidget(self.spin_x)
        self.btn_select_x = QPushButton("选择 X 个")
        self.btn_select_x.clicked.connect(self._select_x_vars)
        var_toolbar.addWidget(self.btn_select_x)
        self.btn_var_nc = QPushButton("全选非共识")
        self.btn_var_nc.clicked.connect(lambda: self._set_vars_by_source(abl.NON_CONSENSUS, True))
        self.btn_var_drop = QPushButton("全选 drop")
        self.btn_var_drop.clicked.connect(lambda: self._set_vars_by_source(abl.DROP, True))
        self.btn_var_clear = QPushButton("清空")
        self.btn_var_clear.clicked.connect(lambda: self._set_table_checked(self.var_table, False))
        var_toolbar.addWidget(self.btn_var_nc)
        var_toolbar.addWidget(self.btn_var_drop)
        var_toolbar.addWidget(self.btn_var_clear)
        var_toolbar.addStretch()
        var_layout.addLayout(var_toolbar)

        self.var_table = QTableWidget()
        self.var_table.setColumnCount(4)
        self.var_table.setHorizontalHeaderLabels(["选择", "变量", "来源", "已在预处理CSV"])
        self.var_table.setMaximumHeight(160)
        var_layout.addWidget(self.var_table)
        top_layout.addWidget(var_group)

        # 评估配置
        cfg_group = QGroupBox("3. 评估配置")
        cfg_layout = QHBoxLayout(cfg_group)
        cfg_layout.addWidget(QLabel("超参数策略:"))
        self.radio_reuse = QRadioButton("沿用原最优超参数（默认，推荐）")
        self.radio_reuse.setChecked(True)
        self.radio_reuse.setToolTip("直接复用已保存模型的超参数重训，排除调参带来的方差")
        self.radio_optuna = QRadioButton("每次重新 Optuna 调参")
        self.radio_optuna.setToolTip("对每个候选变量重新调参，更公平但慢很多")
        cfg_layout.addWidget(self.radio_reuse)
        cfg_layout.addWidget(self.radio_optuna)
        cfg_layout.addWidget(QLabel("trials:"))
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(5, 100)
        self.spin_trials.setValue(20)
        cfg_layout.addWidget(self.spin_trials)
        self.check_delong = QCheckBox("DeLong 检验 + BH 校正")
        self.check_delong.setChecked(True)
        cfg_layout.addWidget(self.check_delong)
        self.check_incremental = QCheckBox("增量运行（跳过已成功组合）")
        self.check_incremental.setChecked(True)
        self.check_incremental.setToolTip(
            "勾选后：已在历史消融结果中成功的「变量×模型」组合会被跳过，\n"
            "新结果与历史结果合并；取消勾选则全部重跑。"
        )
        cfg_layout.addWidget(self.check_incremental)
        self.lbl_gpu = QLabel("")
        self.lbl_gpu.setStyleSheet("color: gray;")
        cfg_layout.addWidget(self.lbl_gpu)
        cfg_layout.addStretch()
        top_layout.addWidget(cfg_group)

        # 运行控制
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("运行消融分析")
        self.btn_run.clicked.connect(self.run_ablation)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        self.btn_refresh = QPushButton("刷新结果")
        self.btn_refresh.setToolTip("重新从磁盘加载消融结果；若上游模型/数据已更新会提示过期")
        self.btn_refresh.clicked.connect(self.refresh_results)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_cancel)
        run_layout.addWidget(self.btn_refresh)
        run_layout.addWidget(self.progress, stretch=1)
        top_layout.addLayout(run_layout)

        self.lbl_status = QLabel("未运行")
        top_layout.addWidget(self.lbl_status)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        top_layout.addWidget(self.log_text)

        # 结果管理按钮
        manage_layout = QHBoxLayout()
        self.btn_delete_results = QPushButton("删除本次结果")
        self.btn_delete_results.setToolTip("删除当前实验的消融分析结果目录（不可恢复）")
        self.btn_delete_results.clicked.connect(self.delete_results)
        self.btn_archive_results = QPushButton("归档本次结果")
        self.btn_archive_results.setToolTip("将当前消融分析结果移动到 archive/ 下并附加时间戳")
        self.btn_archive_results.clicked.connect(self.archive_results)
        manage_layout.addWidget(self.btn_delete_results)
        manage_layout.addWidget(self.btn_archive_results)
        manage_layout.addStretch()
        top_layout.addLayout(manage_layout)

        splitter.addWidget(top_widget)

        # ---------------- 下半部分：结果 ----------------
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.result_tabs = QTabWidget()

        # ΔAUC 矩阵
        matrix_widget = QWidget()
        matrix_layout = QVBoxLayout(matrix_widget)
        self.matrix_table = QTableWidget()
        matrix_layout.addWidget(self.matrix_table)
        self.result_tabs.addTab(matrix_widget, "ΔAUC 矩阵")

        # 变量榜单
        lb_widget = QWidget()
        lb_layout = QVBoxLayout(lb_widget)
        self.lb_table = QTableWidget()
        lb_layout.addWidget(self.lb_table)
        self.result_tabs.addTab(lb_widget, "变量榜单")

        # ROC 绘制
        roc_widget = QWidget()
        roc_layout = QVBoxLayout(roc_widget)
        roc_toolbar = QHBoxLayout()
        roc_toolbar.addWidget(QLabel("模型:"))
        self.combo_roc_model = QComboBox()
        self.combo_roc_model.currentTextChanged.connect(self._update_roc_vars)
        roc_toolbar.addWidget(self.combo_roc_model)
        roc_toolbar.addWidget(QLabel("变量:"))
        self.combo_roc_var = QComboBox()
        roc_toolbar.addWidget(self.combo_roc_var)
        self.btn_roc = QPushButton("绘制 ROC（base vs +变量）")
        self.btn_roc.clicked.connect(self._plot_roc)
        roc_toolbar.addWidget(self.btn_roc)
        self.btn_roc_save = QPushButton("导出 ROC PNG")
        self.btn_roc_save.clicked.connect(self._save_roc)
        roc_toolbar.addWidget(self.btn_roc_save)
        roc_toolbar.addStretch()
        roc_layout.addLayout(roc_toolbar)
        self.roc_preview = QLabel("选择模型与变量后点击绘制")
        self.roc_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roc_preview.setMinimumHeight(320)
        self.roc_preview.setStyleSheet("border: 1px solid #ccc;")
        roc_layout.addWidget(self.roc_preview)
        self._roc_fig = None
        self.result_tabs.addTab(roc_widget, "ROC 曲线")

        bottom_layout.addWidget(self.result_tabs)

        # 导出
        export_layout = QHBoxLayout()
        self.btn_export_csv = QPushButton("导出结果 CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_heatmap = QPushButton("导出热图 PNG")
        self.btn_export_heatmap.clicked.connect(self._export_heatmap)
        self.btn_export_matrix = QPushButton("导出矩阵/榜单 CSV")
        self.btn_export_matrix.clicked.connect(self._export_summary_csv)
        export_layout.addWidget(self.btn_export_csv)
        export_layout.addWidget(self.btn_export_heatmap)
        export_layout.addWidget(self.btn_export_matrix)
        export_layout.addStretch()
        bottom_layout.addLayout(export_layout)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([520, 480])
        self.layout.addWidget(splitter)

        self._update_gpu_label()

    # ---------------- 辅助 ----------------
    def _update_gpu_label(self):
        try:
            from core.transformers import get_transformer_device_info
            self.lbl_gpu.setText(get_transformer_device_info())
        except Exception:
            self.lbl_gpu.setText("")

    def _log(self, msg: str):
        self.log_text.append(msg)
        self.parent_window.log_message(msg)

    def _get_ablation_dir(self) -> Path:
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        d = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "ablation"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _set_table_checked(self, table: QTableWidget, checked: bool):
        for i in range(table.rowCount()):
            item = table.item(i, 0)
            if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    # ---------------- 模型表 ----------------
    def refresh_candidates(self):
        """根据当前实验刷新模型表与变量表。"""
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        self._exp = exp
        config = exp.get("config") or {}

        models = []
        # Base
        modeling = config.get("modeling") or {}
        lb_path = modeling.get("leaderboard_path")
        if lb_path and Path(lb_path).exists():
            try:
                lb = pd.read_csv(lb_path)
                for _, r in lb.iterrows():
                    models.append({"name": r["model"], "source": "Base", "auc": r.get("auc", np.nan), "enabled": True})
            except Exception:
                pass
        # Ensemble（禁用）
        ens = config.get("ensemble") or {}
        ens_path = ens.get("result_path")
        if ens_path and Path(ens_path).exists():
            try:
                edf = pd.read_csv(ens_path)
                for _, r in edf.iterrows():
                    models.append({"name": r["method"], "source": "Ensemble", "auc": r.get("auc", np.nan), "enabled": False})
            except Exception:
                pass
        # Transformer
        tr = config.get("transformer") or {}
        tr_path = tr.get("result_path")
        if tr_path and Path(tr_path).exists():
            try:
                tdf = pd.read_csv(tr_path)
                for _, r in tdf.iterrows():
                    models.append({"name": r["model"], "source": "Transformer", "auc": r.get("auc", np.nan), "enabled": True})
            except Exception:
                pass
        # AutoML
        au = config.get("automl") or {}
        au_path = au.get("result_path")
        if au_path and Path(au_path).exists():
            try:
                adf = pd.read_csv(au_path)
                for _, r in adf.iterrows():
                    models.append({"name": r["framework"], "source": "AutoML", "auc": r.get("auc", np.nan), "enabled": True})
            except Exception:
                pass

        # 排序：可用的在前，按 AUC 降序
        models.sort(key=lambda m: (not m["enabled"], -(m["auc"] if np.isfinite(m["auc"]) else -1)))
        self._all_models = models

        self.model_table.setRowCount(len(models))
        for i, m in enumerate(models):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | (Qt.ItemFlag.ItemIsEnabled if m["enabled"] else Qt.ItemFlag.NoItemFlags))
            chk.setCheckState(Qt.CheckState.Unchecked)
            self.model_table.setItem(i, 0, chk)
            name_item = QTableWidgetItem(m["name"])
            src_item = QTableWidgetItem(m["source"])
            auc_item = QTableWidgetItem(f"{m['auc']:.4f}" if np.isfinite(m["auc"]) else "-")
            if not m["enabled"]:
                for it in (name_item, src_item, auc_item):
                    it.setForeground(QColor("gray"))
                name_item.setToolTip("集成模型输入为基模型 OOF，消融需重训全部基模型，暂不支持")
            self.model_table.setItem(i, 1, name_item)
            self.model_table.setItem(i, 2, src_item)
            self.model_table.setItem(i, 3, auc_item)
        self.model_table.resizeColumnsToContents()

        # 候选变量
        try:
            cand = abl.collect_candidate_variables(exp)
        except Exception as e:
            cand = pd.DataFrame(columns=["variable", "source", "in_preprocessed"])
            self.parent_window.logger.warning(f"收集候选变量失败：{e}")
        self._cand_df = cand
        self.var_table.setRowCount(len(cand))
        for i, r in cand.iterrows():
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Unchecked)
            self.var_table.setItem(i, 0, chk)
            self.var_table.setItem(i, 1, QTableWidgetItem(r["variable"]))
            self.var_table.setItem(i, 2, QTableWidgetItem(r["source"]))
            self.var_table.setItem(i, 3, QTableWidgetItem("是" if r["in_preprocessed"] else "否（需重预处理）"))
        self.var_table.resizeColumnsToContents()

        # 恢复已保存的设置
        self.load_settings(exp)

    def _select_topn(self):
        n = self.spin_topn.value()
        count = 0
        for i in range(self.model_table.rowCount()):
            item = self.model_table.item(i, 0)
            enabled = item.flags() & Qt.ItemFlag.ItemIsEnabled
            item.setCheckState(Qt.CheckState.Unchecked)
            if enabled and count < n:
                item.setCheckState(Qt.CheckState.Checked)
                count += 1

    def _select_x_vars(self):
        n = self.spin_x.value()
        count = 0
        for i in range(self.var_table.rowCount()):
            item = self.var_table.item(i, 0)
            item.setCheckState(Qt.CheckState.Checked if count < n else Qt.CheckState.Unchecked)
            count += 1

    def _set_vars_by_source(self, source: str, checked: bool):
        for i in range(self.var_table.rowCount()):
            src = self.var_table.item(i, 2).text()
            item = self.var_table.item(i, 0)
            item.setCheckState(Qt.CheckState.Checked if (src == source and checked) else Qt.CheckState.Unchecked)

    def _get_selected_models(self) -> List[Dict[str, str]]:
        out = []
        for i in range(self.model_table.rowCount()):
            item = self.model_table.item(i, 0)
            if item.checkState() == Qt.CheckState.Checked and (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                out.append({
                    "name": self.model_table.item(i, 1).text(),
                    "source": self.model_table.item(i, 2).text(),
                })
        return out

    def _get_selected_variables(self) -> List[str]:
        out = []
        for i in range(self.var_table.rowCount()):
            item = self.var_table.item(i, 0)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(self.var_table.item(i, 1).text())
        return out

    # ---------------- 设置持久化 ----------------
    def get_settings(self) -> dict:
        return {
            "models": self._get_selected_models(),
            "variables": self._get_selected_variables(),
            "hyperparam_strategy": "optuna" if self.radio_optuna.isChecked() else "reuse",
            "use_delong": self.check_delong.isChecked(),
            "incremental": self.check_incremental.isChecked(),
            "n_trials": self.spin_trials.value(),
            "topn": self.spin_topn.value(),
            "x": self.spin_x.value(),
        }

    def load_settings(self, exp: dict):
        config = exp.get("config") or {}
        ab_cfg = config.get("ablation")
        if not ab_cfg:
            return
        saved_models = {m["name"] for m in ab_cfg.get("models", [])}
        for i in range(self.model_table.rowCount()):
            item = self.model_table.item(i, 0)
            name = self.model_table.item(i, 1).text()
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if name in saved_models else Qt.CheckState.Unchecked)
        saved_vars = set(ab_cfg.get("variables", []))
        for i in range(self.var_table.rowCount()):
            item = self.var_table.item(i, 0)
            var = self.var_table.item(i, 1).text()
            item.setCheckState(Qt.CheckState.Checked if var in saved_vars else Qt.CheckState.Unchecked)
        if ab_cfg.get("hyperparam_strategy") == "optuna":
            self.radio_optuna.setChecked(True)
        else:
            self.radio_reuse.setChecked(True)
        self.check_delong.setChecked(ab_cfg.get("use_delong", True))
        self.check_incremental.setChecked(ab_cfg.get("incremental", True))
        self.spin_trials.setValue(int(ab_cfg.get("n_trials", 20)))
        self.spin_topn.setValue(int(ab_cfg.get("topn", 5)))
        self.spin_x.setValue(int(ab_cfg.get("x", 5)))

    def load_results(self, exp: dict):
        """加载历史消融结果。"""
        config = exp.get("config") or {}
        ab_cfg = config.get("ablation") or {}
        result_path = ab_cfg.get("result_path")
        if not result_path or not Path(result_path).exists():
            return
        try:
            self._result = pd.read_csv(result_path)
            self._render_results(self._result)
            status = f"已加载历史消融结果：{len(self._result)} 条"
            # 过期检测：上游模型/数据更新后结果可能失效
            modeling = config.get("modeling") or {}
            prep = config.get("preprocessing") or {}
            upstream = [modeling.get("leaderboard_path"), prep.get("processed_path")]
            from gui.widgets.result_utils import is_result_stale
            stale, reason = is_result_stale(result_path, upstream)
            if stale:
                status += f"　⚠️ 结果可能已过期（{reason} 已更新），建议重新运行"
            self.lbl_status.setText(status)
        except Exception as e:
            self.parent_window.logger.warning(f"加载消融历史结果失败：{e}")

    def refresh_results(self):
        """手动刷新：重新从磁盘加载消融结果。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        self.load_results(exp)
        self._log("已刷新消融结果")

    # ---------------- 结果管理（删除/归档） ----------------
    def _get_result_dir(self) -> Optional[Path]:
        """返回当前实验的消融结果目录。"""
        if self.parent_window.current_experiment_id is None:
            return None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        ab_cfg = config.get("ablation") or {}
        result_path = ab_cfg.get("result_path")
        if result_path and Path(result_path).exists():
            return Path(result_path).parent
        # 兜底：默认消融目录
        d = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "ablation"
        return d if d.exists() else None

    def _clear_config(self):
        """清空当前实验的消融配置。"""
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        for key in ("ablation",):
            if key in config:
                del config[key]
        self.parent_window.db.update_experiment(self.parent_window.current_experiment_id, {"config": config})

    def delete_results(self):
        """删除当前实验的消融分析结果。"""
        from gui.widgets.result_utils import delete_module_results
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可删除的消融结果")
            return
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除以下消融结果目录及其所有内容？\n{result_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        ok, msg = delete_module_results(
            result_dir, status_label=self.lbl_status, logger=self.parent_window.logger,
        )
        if ok:
            self._clear_config()
            self._result = pd.DataFrame()
            self.matrix_table.setRowCount(0)
            self.lb_table.setRowCount(0)
            self.combo_roc_model.clear()
            self.combo_roc_var.clear()
            self.lbl_status.setText("结果已删除")
            self._log(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def archive_results(self):
        """归档当前实验的消融分析结果。"""
        from gui.widgets.result_utils import archive_module_results
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可归档的消融结果")
            return
        ok, msg = archive_module_results(result_dir, logger=self.parent_window.logger)
        if ok:
            self._clear_config()
            self._result = pd.DataFrame()
            self.matrix_table.setRowCount(0)
            self.lb_table.setRowCount(0)
            self.combo_roc_model.clear()
            self.combo_roc_var.clear()
            self.lbl_status.setText("结果已归档")
            self._log(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    # ---------------- 运行 ----------------
    def run_ablation_sync(self, silent: bool = True):
        """同步执行消融分析（供工作流调用，阻塞至完成，无对话框）。"""
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        self._exp = exp
        models = self._get_selected_models()
        variables = self._get_selected_variables()
        if not models or not variables:
            self.parent_window.log_message("消融分析：未选择模型或变量，跳过")
            return

        from core import ablation as abl
        abl_dir = self._get_ablation_dir()
        pred_dir = abl_dir / "preds"

        # 增量跳过
        skip_pairs = set()
        existing_result = None
        result_path = abl_dir / "ablation_result.csv"
        if self.check_incremental.isChecked() and result_path.exists():
            try:
                existing_result = pd.read_csv(result_path)
                ok = existing_result[existing_result["status"] == "ok"]
                skip_pairs = set(zip(ok["variable"].astype(str), ok["model"].astype(str)))
            except Exception:
                pass

        # 全部组合均已完成：直接加载历史结果返回，避免用空结果覆盖历史文件
        n_todo = len(models) * len(variables) - len(
            [p for p in skip_pairs
             if p[0] in set(variables) and p[1] in {m["name"] for m in models}]
        )
        if n_todo <= 0 and existing_result is not None:
            self._result = existing_result
            self._render_results(existing_result)
            self.parent_window.log_message("消融分析：所有选中组合均已有成功结果，跳过")
            return

        result = abl.run_ablation(
            exp, models, variables,
            hyperparam_strategy="optuna" if self.radio_optuna.isChecked() else "reuse",
            use_delong=self.check_delong.isChecked(),
            cv_folds=int((exp.get("config", {}).get("modeling") or {}).get("cv_folds", 5)),
            n_trials=self.spin_trials.value(),
            pred_dir=pred_dir,
            skip_pairs=skip_pairs,
            progress_callback=lambda c, t, m: self.parent_window.log_message(f"消融 {c}/{t} {m}"),
        )
        # 合并历史结果
        if result_path.exists() and not result.empty:
            try:
                existing = pd.read_csv(result_path)
                combined = pd.concat([existing, result], ignore_index=True)
                result = combined.drop_duplicates(subset=["variable", "model"], keep="last").reset_index(drop=True)
            except Exception:
                pass
        # 增量合并后按完整检验集合重算 BH-FDR q 值
        if "delong_p" in result.columns and result["delong_p"].notna().any():
            from core.ablation import bh_fdr
            result["q_value"] = bh_fdr(result["delong_p"].fillna(1.0).values)
        result.to_csv(result_path, index=False)

        config = exp.get("config") or {}
        config["ablation"] = {**self.get_settings(), "result_path": str(result_path)}
        self.parent_window.db.update_experiment(self.parent_window.current_experiment_id, {"config": config})
        self._result = result
        self._render_results(result)
        self.parent_window.log_message(f"消融分析完成：{len(result)} 条")

    def run_ablation(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        self._exp = exp

        models = self._get_selected_models()
        variables = self._get_selected_variables()
        if not models:
            QMessageBox.warning(self, "警告", "请至少选择一个 Base 模型")
            return
        if not variables:
            QMessageBox.warning(self, "警告", "请至少选择一个候选变量")
            return

        n_runs = len(models) * len(variables)
        if n_runs > 100:
            reply = QMessageBox.question(
                self, "确认", f"将执行 {n_runs} 次模型重训，可能耗时较长。继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        abl_dir = self._get_ablation_dir()
        pred_dir = abl_dir / "preds"

        # 增量运行：计算已成功的 (variable, model) 组合并跳过
        skip_pairs = set()
        existing_result = None
        if self.check_incremental.isChecked():
            result_path = abl_dir / "ablation_result.csv"
            if result_path.exists():
                try:
                    existing_result = pd.read_csv(result_path)
                    ok = existing_result[existing_result["status"] == "ok"]
                    skip_pairs = set(zip(ok["variable"].astype(str), ok["model"].astype(str)))
                except Exception as e:
                    self.parent_window.logger.warning(f"读取历史消融结果失败：{e}")
        n_todo = n_runs - len([p for p in skip_pairs if p[0] in set(variables) and p[1] in {m["name"] for m in models}])
        if n_todo <= 0 and existing_result is not None:
            self._result = existing_result
            self._render_results(existing_result)
            self.lbl_status.setText("所有选中组合均已有成功结果，无需重跑")
            self._log("所有选中组合均已有成功结果，无需重跑")
            QMessageBox.information(self, "提示", "所有选中组合均已有成功结果，无需重跑")
            return

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self._log(f"消融开始：{len(models)} 个模型 × {len(variables)} 个变量 = {n_runs} 次重训"
                  + (f"，增量跳过 {len(skip_pairs)} 个已成功组合" if skip_pairs else ""))

        self._worker = AblationWorker(
            exp, models, variables,
            hyperparam_strategy="optuna" if self.radio_optuna.isChecked() else "reuse",
            use_delong=self.check_delong.isChecked(),
            cv_folds=int((exp.get("config", {}).get("modeling") or {}).get("cv_folds", 5)),
            n_trials=self.spin_trials.value(),
            pred_dir=pred_dir,
            skip_pairs=skip_pairs,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.abort()
            self._log("正在取消...")
            self.btn_cancel.setEnabled(False)

    def _on_progress(self, cur: int, total: int, msg: str):
        self.progress.setValue(int(cur / max(total, 1) * 100))
        self.lbl_status.setText(f"运行中 ({cur}/{total})：{msg}")

    def _on_finished(self, result: pd.DataFrame):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)

        # 与历史结果合并：新结果覆盖同名 (variable, model) 组合
        abl_dir = self._get_ablation_dir()
        result_path = abl_dir / "ablation_result.csv"
        if result_path.exists() and not result.empty:
            try:
                existing = pd.read_csv(result_path)
                combined = pd.concat([existing, result], ignore_index=True)
                combined = combined.drop_duplicates(subset=["variable", "model"], keep="last")
                result = combined.reset_index(drop=True)
            except Exception as e:
                self.parent_window.logger.warning(f"合并历史消融结果失败：{e}")
        # 增量合并后按完整检验集合重算 BH-FDR q 值，避免新旧两套口径混排
        if "delong_p" in result.columns and result["delong_p"].notna().any():
            from core.ablation import bh_fdr
            result["q_value"] = bh_fdr(result["delong_p"].fillna(1.0).values)
        self._result = result

        # 保存结果与配置
        result.to_csv(result_path, index=False)

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        config["ablation"] = {
            **self.get_settings(),
            "result_path": str(result_path),
        }
        self.parent_window.db.update_experiment(
            self.parent_window.current_experiment_id, {"config": config}
        )

        n_ok = int((result["status"] == "ok").sum()) if len(result) else 0
        n_fail = int((result["status"] == "failed").sum()) if len(result) else 0
        self.lbl_status.setText(f"消融完成：成功 {n_ok}，失败 {n_fail}")
        self._log(f"消融完成：成功 {n_ok}，失败 {n_fail}，结果已保存至 {result_path}")
        self._render_results(result)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("消融失败")
        self._log(f"消融失败：{msg.splitlines()[0] if msg else ''}")
        self.parent_window.logger.error(f"消融失败：{msg}")
        QMessageBox.critical(self, "失败", "消融分析失败，详见日志")

    # ---------------- 结果渲染 ----------------
    def _render_results(self, result: pd.DataFrame):
        # ΔAUC 矩阵
        mat = abl.make_delta_matrix(result)
        self.matrix_table.setRowCount(mat.shape[0])
        self.matrix_table.setColumnCount(mat.shape[1])
        self.matrix_table.setHorizontalHeaderLabels(list(mat.columns))
        self.matrix_table.setVerticalHeaderLabels([str(v) for v in mat.index])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iat[i, j]
                item = QTableWidgetItem("" if pd.isna(val) else f"{val:.4f}")
                if pd.notna(val):
                    if val > 0.005:
                        item.setBackground(QColor(198, 239, 206))  # 绿=提升
                    elif val < -0.005:
                        item.setBackground(QColor(255, 199, 206))  # 红=下降
                self.matrix_table.setItem(i, j, item)
        self.matrix_table.resizeColumnsToContents()

        # 榜单
        summary = abl.summarize_variables(result)
        self.lb_table.setRowCount(summary.shape[0])
        self.lb_table.setColumnCount(summary.shape[1])
        self.lb_table.setHorizontalHeaderLabels(list(summary.columns))
        for i in range(summary.shape[0]):
            for j, col in enumerate(summary.columns):
                val = summary.iat[i, j]
                item = QTableWidgetItem(str(val))
                if col == "recommended" and val is True:
                    item.setBackground(QColor(198, 239, 206))
                self.lb_table.setItem(i, j, item)
        self.lb_table.resizeColumnsToContents()

        # ROC 下拉框
        ok = result[result["status"] == "ok"]
        self._roc_ok = ok
        self.combo_roc_model.blockSignals(True)
        self.combo_roc_model.clear()
        models = sorted(set(ok["source"] + ":" + ok["model"])) if len(ok) else []
        self.combo_roc_model.addItems(models)
        self.combo_roc_model.blockSignals(False)
        self._update_roc_vars()

    def _update_roc_vars(self):
        """根据所选模型，更新可绘制 ROC 的变量下拉框。"""
        self.combo_roc_var.clear()
        sel = self.combo_roc_model.currentText()
        if not sel or self._roc_ok is None or len(self._roc_ok) == 0:
            return
        source, name = sel.split(":", 1)
        sub = self._roc_ok[(self._roc_ok["source"] == source) & (self._roc_ok["model"] == name)]
        vars_ = list(dict.fromkeys(sub["variable"].astype(str).tolist()))
        self.combo_roc_var.addItems(vars_)

    def _plot_roc(self):
        if self._result.empty:
            QMessageBox.warning(self, "警告", "请先运行消融分析")
            return
        sel = self.combo_roc_model.currentText()
        var = self.combo_roc_var.currentText()
        if not sel or not var:
            QMessageBox.warning(self, "警告", "请选择模型与变量")
            return
        source, name = sel.split(":", 1)

        abl_dir = self._get_ablation_dir()
        safe_var = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in var)
        safe_name = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in name)
        pred_path = abl_dir / "preds" / f"{source}_{safe_name}__{safe_var}.csv"
        if not pred_path.exists():
            QMessageBox.warning(self, "警告", f"未找到预测文件：{pred_path.name}\n请重新运行消融（将自动保存预测值）")
            return

        df = pd.read_csv(pred_path)
        fig = self._draw_roc(df["y_true"].values, df["proba_base"].values, df["proba_new"].values,
                             f"{name} - base", f"{name} - +{var}")
        self._roc_fig = fig
        png = abl_dir / "_roc_preview.png"
        fig.savefig(str(png), dpi=150, bbox_inches="tight")
        pix = QPixmap(str(png))
        self.roc_preview.setPixmap(pix.scaled(
            self.roc_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _draw_roc(self, y_true, p_base, p_new, label_base, label_new):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc as sk_auc

        fpr_b, tpr_b, _ = roc_curve(y_true, p_base)
        fpr_n, tpr_n, _ = roc_curve(y_true, p_new)
        auc_b = sk_auc(fpr_b, tpr_b)
        auc_n = sk_auc(fpr_n, tpr_n)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr_b, tpr_b, label=f"{label_base} (AUC={auc_b:.4f})", color="#1f77b4")
        ax.plot(fpr_n, tpr_n, label=f"{label_new} (AUC={auc_n:.4f})", color="#d62728")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC: base vs +variable")
        ax.legend(loc="lower right")
        fig.tight_layout()
        return fig

    def _save_roc(self):
        if self._roc_fig is None:
            QMessageBox.warning(self, "警告", "请先绘制 ROC")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存 ROC 图", "roc_ablation.png", "PNG (*.png)")
        if path:
            self._roc_fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "成功", f"已保存：{path}")

    # ---------------- 导出 ----------------
    def _export_csv(self):
        if self._result.empty:
            QMessageBox.warning(self, "警告", "暂无消融结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出消融结果", "ablation_result.csv", "CSV (*.csv)")
        if path:
            self._result.to_csv(path, index=False)
            QMessageBox.information(self, "成功", f"已保存：{path}")

    def _export_summary_csv(self):
        if self._result.empty:
            QMessageBox.warning(self, "警告", "暂无消融结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出矩阵与榜单", "ablation_summary.csv", "CSV (*.csv)")
        if path:
            summary = abl.summarize_variables(self._result)
            summary.to_csv(path, index=False)
            mat_path = Path(path).with_name(Path(path).stem + "_matrix.csv")
            abl.make_delta_matrix(self._result).to_csv(mat_path)
            QMessageBox.information(self, "成功", f"榜单：{path}\n矩阵：{mat_path}")

    def _export_heatmap(self):
        if self._result.empty:
            QMessageBox.warning(self, "警告", "暂无消融结果")
            return
        mat = abl.make_delta_matrix(self._result)
        if mat.empty:
            QMessageBox.warning(self, "警告", "暂无可绘制的矩阵")
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(6, mat.shape[1] * 1.2), max(4, mat.shape[0] * 0.5)))
            im = ax.imshow(mat.values, cmap="RdYlGn", aspect="auto",
                           vmin=-max(0.02, abs(mat.values).max()), vmax=max(0.02, abs(mat.values).max()))
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels(list(mat.columns), rotation=45, ha="right")
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels([str(v) for v in mat.index])
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.iat[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="ΔAUC")
            ax.set_title("Ablation ΔAUC Heatmap")
            fig.tight_layout()
        except Exception as e:
            QMessageBox.critical(self, "失败", f"热图生成失败：{e}")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出热图", "ablation_heatmap.png", "PNG (*.png)")
        if path:
            fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "成功", f"已保存：{path}")
