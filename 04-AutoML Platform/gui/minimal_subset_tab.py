"""最小协变量子集分析选项卡。

对每个选中模型做 SHAP 引导的向后剔除：逐步删除贡献最低的协变量，
直到性能出现统计学显著下降（DeLong p < 阈值），取前一个子集作为最小子集，
并在子集上重算 SHAP 贡献度排名。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QProgressBar,
    QTextEdit, QSplitter, QTabWidget, QFileDialog,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import minimal_subset as ms


class MinimalSubsetWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, exp, models, p_thresh, cv_folds, resample_shap, shap_sample_size):
        super().__init__()
        self.exp = exp
        self.models = models
        self.p_thresh = p_thresh
        self.cv_folds = cv_folds
        self.resample_shap = resample_shap
        self.shap_sample_size = shap_sample_size
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            def _cb(cur, total, msg):
                if self._abort:
                    raise InterruptedError("用户取消")
                self.progress.emit(cur, total, msg)

            results = ms.run_minimal_subset(
                self.exp, self.models,
                p_thresh=self.p_thresh, cv_folds=self.cv_folds,
                resample_shap=self.resample_shap,
                shap_sample_size=self.shap_sample_size,
                progress_callback=_cb,
            )
            self.finished_signal.emit(results)
        except InterruptedError:
            self.error.emit("已取消")
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MinimalSubsetTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)
        self._worker = None
        self._results = []
        self._all_models = []

        self.layout.addWidget(QLabel("<h2>最小协变量子集分析</h2>"))
        intro = QLabel(
            "对每个选中模型做 SHAP 引导的向后剔除：逐步删除贡献最低的协变量，"
            "直到性能显著下降（DeLong p<阈值），找到保持性能的最小协变量子集。"
            "集成模型不支持（输入为基模型 OOF，已置灰）。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        self.layout.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ---------------- 上半部分：配置 ----------------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 模型选择
        model_group = QGroupBox("1. 选择模型（按原 AUC 排序）")
        model_layout = QVBoxLayout(model_group)
        model_toolbar = QHBoxLayout()
        model_toolbar.addWidget(QLabel("一键选择 Top-N:"))
        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(1, 30)
        self.spin_topn.setValue(5)
        model_toolbar.addWidget(self.spin_topn)
        self.btn_topn = QPushButton("选择 Top-N")
        self.btn_topn.clicked.connect(self._select_topn)
        model_toolbar.addWidget(self.btn_topn)
        self.btn_model_all = QPushButton("全选")
        self.btn_model_all.clicked.connect(lambda: self._set_checked(True))
        self.btn_model_clear = QPushButton("清空")
        self.btn_model_clear.clicked.connect(lambda: self._set_checked(False))
        model_toolbar.addWidget(self.btn_model_all)
        model_toolbar.addWidget(self.btn_model_clear)
        model_toolbar.addStretch()
        model_layout.addLayout(model_toolbar)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["选择", "模型", "来源", "原 AUC"])
        self.model_table.setMaximumHeight(180)
        model_layout.addWidget(self.model_table)
        top_layout.addWidget(model_group)

        # 配置
        cfg_group = QGroupBox("2. 分析配置")
        cfg_layout = QHBoxLayout(cfg_group)
        cfg_layout.addWidget(QLabel("DeLong p 阈值:"))
        self.spin_p = QDoubleSpinBox()
        self.spin_p.setRange(0.001, 0.2)
        self.spin_p.setSingleStep(0.01)
        self.spin_p.setDecimals(3)
        self.spin_p.setValue(0.05)
        self.spin_p.setToolTip("当某一步子集与全特征模型的 DeLong p < 该值时停止，取前一个子集")
        cfg_layout.addWidget(self.spin_p)
        self.check_reshap = QCheckBox("在最小子集上重算 SHAP（推荐）")
        self.check_reshap.setChecked(True)
        cfg_layout.addWidget(self.check_reshap)
        cfg_layout.addWidget(QLabel("SHAP 样本数:"))
        self.spin_shap_sample = QSpinBox()
        self.spin_shap_sample.setRange(50, 1000)
        self.spin_shap_sample.setValue(200)
        cfg_layout.addWidget(self.spin_shap_sample)
        self.lbl_gpu = QLabel("")
        self.lbl_gpu.setStyleSheet("color: gray;")
        cfg_layout.addWidget(self.lbl_gpu)
        cfg_layout.addStretch()
        top_layout.addWidget(cfg_group)

        # 运行
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("运行最小子集分析")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)
        self.btn_refresh = QPushButton("刷新结果")
        self.btn_refresh.setToolTip("重新从磁盘加载最小子集结果；若上游模型/SHAP 已更新会提示过期")
        self.btn_refresh.clicked.connect(self.refresh_results)
        self.progress = QProgressBar()
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

        splitter.addWidget(top_widget)

        # ---------------- 下半部分：结果 ----------------
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.result_tabs = QTabWidget()

        # 轨迹
        traj_widget = QWidget()
        traj_layout = QVBoxLayout(traj_widget)
        traj_toolbar = QHBoxLayout()
        traj_toolbar.addWidget(QLabel("模型:"))
        self.combo_traj_model = QComboBox()
        traj_toolbar.addWidget(self.combo_traj_model)
        self.btn_traj = QPushButton("绘制精简轨迹")
        self.btn_traj.clicked.connect(self._plot_trajectory)
        traj_toolbar.addWidget(self.btn_traj)
        traj_toolbar.addStretch()
        traj_layout.addLayout(traj_toolbar)
        self.traj_preview = QLabel("选择模型后点击绘制")
        self.traj_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.traj_preview.setMinimumHeight(320)
        self.traj_preview.setStyleSheet("border: 1px solid #ccc;")
        traj_layout.addWidget(self.traj_preview)
        self._traj_fig = None
        self.result_tabs.addTab(traj_widget, "精简轨迹")

        # 最小子集清单
        subset_widget = QWidget()
        subset_layout = QVBoxLayout(subset_widget)
        self.subset_table = QTableWidget()
        subset_layout.addWidget(self.subset_table)
        self.result_tabs.addTab(subset_widget, "最小子集清单")

        # 子集内贡献度排名
        shap_widget = QWidget()
        shap_layout = QVBoxLayout(shap_widget)
        shap_toolbar = QHBoxLayout()
        shap_toolbar.addWidget(QLabel("模型:"))
        self.combo_shap_model = QComboBox()
        shap_toolbar.addWidget(self.combo_shap_model)
        self.btn_shap_plot = QPushButton("绘制贡献度条形图")
        self.btn_shap_plot.clicked.connect(self._plot_subset_shap)
        shap_toolbar.addWidget(self.btn_shap_plot)
        shap_toolbar.addStretch()
        shap_layout.addLayout(shap_toolbar)
        shap_split = QSplitter(Qt.Orientation.Horizontal)
        self.shap_table = QTableWidget()
        shap_split.addWidget(self.shap_table)
        self.shap_preview = QLabel("选择模型查看贡献度")
        self.shap_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shap_preview.setStyleSheet("border: 1px solid #ccc;")
        self.shap_preview.setMinimumWidth(380)
        shap_split.addWidget(self.shap_preview)
        shap_split.setSizes([450, 450])
        shap_layout.addWidget(shap_split)
        self._shap_fig = None
        self.result_tabs.addTab(shap_widget, "子集贡献度")

        # 对比汇总
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        self.summary_table = QTableWidget()
        summary_layout.addWidget(self.summary_table)
        self.result_tabs.addTab(summary_widget, "对比汇总")

        bottom_layout.addWidget(self.result_tabs)

        # 回写与导出
        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("回写特征集:"))
        self.combo_wb_model = QComboBox()
        action_layout.addWidget(self.combo_wb_model)
        self.btn_writeback = QPushButton("一键回写为新特征集")
        self.btn_writeback.clicked.connect(self._write_back)
        action_layout.addWidget(self.btn_writeback)
        self.btn_view_wb = QPushButton("查看回写特征集")
        self.btn_view_wb.clicked.connect(self._view_writeback)
        action_layout.addWidget(self.btn_view_wb)
        action_layout.addStretch()
        self.btn_export = QPushButton("导出汇总 CSV")
        self.btn_export.clicked.connect(self._export_csv)
        action_layout.addWidget(self.btn_export)
        bottom_layout.addLayout(action_layout)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([500, 500])
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

    def _get_ms_dir(self) -> Path:
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        d = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "minimal_subset"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _set_checked(self, checked: bool):
        for i in range(self.model_table.rowCount()):
            item = self.model_table.item(i, 0)
            if item and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    # ---------------- 模型表 ----------------
    def refresh_models(self):
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}

        models = []
        modeling = config.get("modeling") or {}
        lb_path = modeling.get("leaderboard_path")
        if lb_path and Path(lb_path).exists():
            try:
                lb = pd.read_csv(lb_path)
                for _, r in lb.iterrows():
                    models.append({"name": r["model"], "source": "Base", "auc": r.get("auc", np.nan), "enabled": True})
            except Exception:
                pass
        ens = config.get("ensemble") or {}
        ens_path = ens.get("result_path")
        if ens_path and Path(ens_path).exists():
            try:
                edf = pd.read_csv(ens_path)
                for _, r in edf.iterrows():
                    models.append({"name": r["method"], "source": "Ensemble", "auc": r.get("auc", np.nan), "enabled": False})
            except Exception:
                pass
        tr = config.get("transformer") or {}
        tr_path = tr.get("result_path")
        if tr_path and Path(tr_path).exists():
            try:
                tdf = pd.read_csv(tr_path)
                for _, r in tdf.iterrows():
                    models.append({"name": r["model"], "source": "Transformer", "auc": r.get("auc", np.nan), "enabled": True})
            except Exception:
                pass
        au = config.get("automl") or {}
        au_path = au.get("result_path")
        if au_path and Path(au_path).exists():
            try:
                adf = pd.read_csv(au_path)
                for _, r in adf.iterrows():
                    models.append({"name": r["framework"], "source": "AutoML", "auc": r.get("auc", np.nan), "enabled": True})
            except Exception:
                pass

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
                name_item.setToolTip("集成模型输入为基模型 OOF，协变量级剔除需重训全部基模型，不支持")
            self.model_table.setItem(i, 1, name_item)
            self.model_table.setItem(i, 2, src_item)
            self.model_table.setItem(i, 3, auc_item)
        self.model_table.resizeColumnsToContents()
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

    def _get_selected_models(self) -> List[Dict[str, str]]:
        out = []
        for i in range(self.model_table.rowCount()):
            item = self.model_table.item(i, 0)
            if item.checkState() == Qt.CheckState.Checked and (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                out.append({"name": self.model_table.item(i, 1).text(), "source": self.model_table.item(i, 2).text()})
        return out

    # ---------------- 持久化 ----------------
    def get_settings(self) -> dict:
        return {
            "models": self._get_selected_models(),
            "p_thresh": self.spin_p.value(),
            "resample_shap": self.check_reshap.isChecked(),
            "shap_sample_size": self.spin_shap_sample.value(),
            "topn": self.spin_topn.value(),
        }

    def load_settings(self, exp: dict):
        config = exp.get("config") or {}
        cfg = config.get("minimal_subset")
        if not cfg:
            return
        saved = {m["name"] for m in cfg.get("models", [])}
        for i in range(self.model_table.rowCount()):
            item = self.model_table.item(i, 0)
            name = self.model_table.item(i, 1).text()
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if name in saved else Qt.CheckState.Unchecked)
        self.spin_p.setValue(float(cfg.get("p_thresh", 0.05)))
        self.check_reshap.setChecked(cfg.get("resample_shap", True))
        self.spin_shap_sample.setValue(int(cfg.get("shap_sample_size", 200)))
        self.spin_topn.setValue(int(cfg.get("topn", 5)))

    def load_results(self, exp: dict):
        config = exp.get("config") or {}
        cfg = config.get("minimal_subset") or {}
        path = cfg.get("summary_path")
        if not path or not Path(path).exists():
            return
        try:
            results = self._load_saved_results(Path(path).parent)
            if results:
                self._results = results
                self._render_results(results)
                status = f"已加载历史结果：{len(results)} 个模型"
                # 过期检测：模型 leaderboard 或 SHAP meta 更新后结果可能失效
                modeling = config.get("modeling") or {}
                model_dir = modeling.get("output_dir")
                shap_meta = None
                if model_dir:
                    shap_meta = str(Path(model_dir).parent / "shap" / "shap_meta.json")
                upstream = [modeling.get("leaderboard_path"), shap_meta]
                from gui.widgets.result_utils import is_result_stale
                stale, reason = is_result_stale(path, upstream)
                if stale:
                    status += f"　⚠️ 结果可能已过期（{reason} 已更新），建议重新运行"
                self.lbl_status.setText(status)
        except Exception as e:
            self.parent_window.logger.warning(f"加载最小子集历史结果失败：{e}")

    def refresh_results(self):
        """手动刷新：重新从磁盘加载最小子集结果。"""
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        self.load_results(exp)
        self._log("已刷新最小子集结果")

    def _load_saved_results(self, ms_dir: Path) -> List[Dict]:
        """从磁盘重建结果对象（用于历史加载与轨迹绘制）。"""
        summary_path = ms_dir / "minimal_subset_summary.csv"
        if not summary_path.exists():
            return []
        summary = pd.read_csv(summary_path)
        results = []
        for _, r in summary.iterrows():
            name = r["model"]
            traj_path = ms_dir / f"{name}_trajectory.csv"
            subset_path = ms_dir / f"{name}_subset_features.csv"
            shap_path = ms_dir / f"{name}_subset_shap.csv"
            traj = pd.read_csv(traj_path) if traj_path.exists() else pd.DataFrame()
            subset_feats = []
            if subset_path.exists():
                subset_feats = pd.read_csv(subset_path)["feature"].astype(str).tolist()
            subset_shap = pd.read_csv(shap_path) if shap_path.exists() else None
            results.append({
                "name": name, "source": r["source"], "status": r.get("status", "ok"),
                "auc_full": r["auc_full"], "auc_subset": r["auc_subset"],
                "delta_auc": r["delta_auc"], "delong_p_final": r.get("delong_p"),
                "n_subset": r["n_subset"], "subset_features": subset_feats,
                "n_full": r.get("n_full") if pd.notna(r.get("n_full")) else None,
                "strictly_better": bool(r.get("strictly_better", False)),
                "ranking_source": r.get("ranking_source", ""),
                "trajectory": traj, "subset_shap": subset_shap,
                "ordered_features": None, "error": r.get("error", ""),
            })
        return results

    # ---------------- 运行 ----------------
    def run_analysis_sync(self, silent: bool = True):
        """同步执行最小子集分析（供工作流调用，阻塞至完成，无对话框）。"""
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        models = self._get_selected_models()
        if not models:
            self.parent_window.log_message("最小子集分析：未选择模型，跳过")
            return

        cv_folds = int((exp.get("config", {}).get("modeling") or {}).get("cv_folds", 5))
        results = ms.run_minimal_subset(
            exp, models,
            p_thresh=self.spin_p.value(), cv_folds=cv_folds,
            resample_shap=self.check_reshap.isChecked(),
            shap_sample_size=self.spin_shap_sample.value(),
            progress_callback=lambda c, t, m: self.parent_window.log_message(f"最小子集 {c}/{t} {m}"),
        )
        # 保存结果
        ms_dir = self._get_ms_dir()
        summary = ms.make_summary_table(results)
        summary_path = ms_dir / "minimal_subset_summary.csv"
        summary.to_csv(summary_path, index=False)
        for r in results:
            name = r["name"]
            if isinstance(r.get("trajectory"), pd.DataFrame) and not r["trajectory"].empty:
                r["trajectory"].to_csv(ms_dir / f"{name}_trajectory.csv", index=False)
            if r.get("subset_features"):
                pd.DataFrame({"feature": r["subset_features"]}).to_csv(ms_dir / f"{name}_subset_features.csv", index=False)
            if isinstance(r.get("subset_shap"), pd.DataFrame):
                r["subset_shap"].to_csv(ms_dir / f"{name}_subset_shap.csv", index=False)

        config = exp.get("config") or {}
        config["minimal_subset"] = {**self.get_settings(), "summary_path": str(summary_path)}
        self.parent_window.db.update_experiment(self.parent_window.current_experiment_id, {"config": config})
        self._results = results
        self._render_results(results)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        self.parent_window.log_message(f"最小子集分析完成：成功 {n_ok}/{len(results)}")

    def run_analysis(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        models = self._get_selected_models()
        if not models:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return

        n_feats = None
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self._log(f"最小子集分析开始：{len(models)} 个模型（向后剔除）")

        cv_folds = int((exp.get("config", {}).get("modeling") or {}).get("cv_folds", 5))
        self._worker = MinimalSubsetWorker(
            exp, models,
            p_thresh=self.spin_p.value(),
            cv_folds=cv_folds,
            resample_shap=self.check_reshap.isChecked(),
            shap_sample_size=self.spin_shap_sample.value(),
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

    def _on_finished(self, results: list):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)
        self._results = results

        # 保存结果
        ms_dir = self._get_ms_dir()
        summary = ms.make_summary_table(results)
        summary_path = ms_dir / "minimal_subset_summary.csv"
        summary.to_csv(summary_path, index=False)
        for r in results:
            name = r["name"]
            if isinstance(r.get("trajectory"), pd.DataFrame) and not r["trajectory"].empty:
                r["trajectory"].to_csv(ms_dir / f"{name}_trajectory.csv", index=False)
            if r.get("subset_features"):
                pd.DataFrame({"feature": r["subset_features"]}).to_csv(ms_dir / f"{name}_subset_features.csv", index=False)
            if isinstance(r.get("subset_shap"), pd.DataFrame):
                r["subset_shap"].to_csv(ms_dir / f"{name}_subset_shap.csv", index=False)

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        config["minimal_subset"] = {**self.get_settings(), "summary_path": str(summary_path)}
        self.parent_window.db.update_experiment(self.parent_window.current_experiment_id, {"config": config})

        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_fail = sum(1 for r in results if r["status"] == "failed")
        self.lbl_status.setText(f"分析完成：成功 {n_ok}，失败 {n_fail}")
        self._log(f"最小子集分析完成：成功 {n_ok}，失败 {n_fail}")
        self._render_results(results)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("分析失败")
        self._log(f"最小子集分析失败：{msg.splitlines()[0] if msg else ''}")
        self.parent_window.logger.error(f"最小子集分析失败：{msg}")
        QMessageBox.critical(self, "失败", "最小子集分析失败，详见日志")

    # ---------------- 渲染 ----------------
    def _render_results(self, results: list):
        ok_results = [r for r in results if r["status"] == "ok"]

        # 最小子集清单
        cols = ["model", "source", "n_full", "n_subset", "auc_full", "auc_subset", "delta_auc", "delong_p", "strictly_better", "subset_features"]
        self.subset_table.setRowCount(len(ok_results))
        self.subset_table.setColumnCount(len(cols))
        self.subset_table.setHorizontalHeaderLabels(cols)
        for i, r in enumerate(ok_results):
            vals = [
                r["name"], r["source"],
                r.get("n_full") or len(r.get("ordered_features") or []) or len(r.get("subset_features") or []),
                r["n_subset"], r["auc_full"], r["auc_subset"], r["delta_auc"],
                f"{r['delong_p_final']:.4f}" if np.isfinite(r.get("delong_p_final", np.nan)) else "-",
                "是" if r["strictly_better"] else "否",
                ", ".join(r["subset_features"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if cols[j] == "strictly_better" and r["strictly_better"]:
                    item.setBackground(QColor(198, 239, 206))
                self.subset_table.setItem(i, j, item)
        self.subset_table.resizeColumnsToContents()

        # 对比汇总
        summary = ms.make_summary_table(results)
        self.summary_table.setRowCount(summary.shape[0])
        self.summary_table.setColumnCount(summary.shape[1])
        self.summary_table.setHorizontalHeaderLabels(list(summary.columns))
        for i in range(summary.shape[0]):
            for j in range(summary.shape[1]):
                self.summary_table.setItem(i, j, QTableWidgetItem(str(summary.iat[i, j])))
        self.summary_table.resizeColumnsToContents()

        # 下拉框
        names = [r["name"] for r in ok_results]
        self.combo_traj_model.clear()
        self.combo_traj_model.addItems(names)
        self.combo_shap_model.clear()
        self.combo_shap_model.addItems(names)
        self.combo_wb_model.clear()
        self.combo_wb_model.addItems(names)
        # 渲染第一个模型的贡献度表
        if names:
            self._show_subset_shap(names[0])

    def _show_subset_shap(self, model_name: str):
        r = next((x for x in self._results if x["name"] == model_name and x["status"] == "ok"), None)
        if r is None or not isinstance(r.get("subset_shap"), pd.DataFrame):
            self.shap_table.setRowCount(0)
            return
        df = r["subset_shap"]
        self.shap_table.setRowCount(df.shape[0])
        self.shap_table.setColumnCount(3)
        self.shap_table.setHorizontalHeaderLabels(["rank", "feature", "mean_abs_shap"])
        for i in range(df.shape[0]):
            self.shap_table.setItem(i, 0, QTableWidgetItem(str(df.iloc[i]["rank"])))
            self.shap_table.setItem(i, 1, QTableWidgetItem(str(df.iloc[i]["feature"])))
            self.shap_table.setItem(i, 2, QTableWidgetItem(f"{df.iloc[i]['mean_abs_shap']:.5f}"))
        self.shap_table.resizeColumnsToContents()

    # ---------------- 绘图 ----------------
    def _plot_trajectory(self):
        name = self.combo_traj_model.currentText()
        r = next((x for x in self._results if x["name"] == name and x["status"] == "ok"), None)
        if r is None or r["trajectory"].empty:
            QMessageBox.warning(self, "警告", "该模型无轨迹数据")
            return
        traj = r["trajectory"].sort_values("k")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(traj["k"], traj["auc"], "o-", color="#1f77b4", label="Subset AUC")
        ax.axhline(r["auc_full"], color="gray", ls="--", alpha=0.6, label=f"Full AUC={r['auc_full']:.4f}")
        ax.axvline(r["n_subset"], color="#d62728", ls=":", alpha=0.8, label=f"Minimal subset (k={r['n_subset']})")
        ax.scatter([r["n_subset"]], [r["auc_subset"]], color="#d62728", zorder=5, s=60)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("AUC")
        ax.set_title(f"{name}: backward elimination trajectory")
        ax.invert_xaxis()
        ax.legend()
        fig.tight_layout()
        self._traj_fig = fig
        ms_dir = self._get_ms_dir()
        png = ms_dir / f"_traj_{name}.png"
        fig.savefig(str(png), dpi=150, bbox_inches="tight")
        pix = QPixmap(str(png))
        self.traj_preview.setPixmap(pix.scaled(self.traj_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _plot_subset_shap(self):
        name = self.combo_shap_model.currentText()
        r = next((x for x in self._results if x["name"] == name and x["status"] == "ok"), None)
        if r is None or not isinstance(r.get("subset_shap"), pd.DataFrame):
            QMessageBox.warning(self, "警告", "该模型无子集 SHAP 数据")
            return
        self._show_subset_shap(name)
        df = r["subset_shap"].sort_values("mean_abs_shap")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, max(3, len(df) * 0.4)))
        ax.barh(df["feature"], df["mean_abs_shap"], color="#2ca02c")
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(f"{name}: 子集内贡献度排名")
        fig.tight_layout()
        self._shap_fig = fig
        ms_dir = self._get_ms_dir()
        png = ms_dir / f"_shap_{name}.png"
        fig.savefig(str(png), dpi=150, bbox_inches="tight")
        pix = QPixmap(str(png))
        self.shap_preview.setPixmap(pix.scaled(self.shap_preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    # ---------------- 回写与导出 ----------------
    def _write_back(self):
        name = self.combo_wb_model.currentText()
        r = next((x for x in self._results if x["name"] == name and x["status"] == "ok"), None)
        if r is None or not r.get("subset_features"):
            QMessageBox.warning(self, "警告", "该模型无最小子集")
            return
        feats = r["subset_features"]
        ms_dir = self._get_ms_dir()
        out_path = ms_dir / f"minimal_features_{name}.csv"
        pd.DataFrame({"feature": feats}).to_csv(out_path, index=False)

        reply = QMessageBox.question(
            self, "回写确认",
            f"模型 {name} 的最小子集（{len(feats)} 个特征）已保存到：\n{out_path}\n\n"
            f"是否同时写入实验 config 作为后续重训的指定特征集？\n（不会影响现有共识表，仅记录为 minimal_features）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
            config = exp.get("config") or {}
            config["minimal_features"] = {
                "model": name, "features": feats, "path": str(out_path),
            }
            self.parent_window.db.update_experiment(self.parent_window.current_experiment_id, {"config": config})
            self._log(f"已回写最小子集特征集：{name}（{len(feats)} 个特征）→ config.minimal_features")
        QMessageBox.information(
            self, "成功",
            f"特征集已保存：{out_path}\n\n"
            "现在可以在 模型 / Transformer / AutoML 选项卡的数据源中\n"
            "选择「最小子集特征数据」用该特征集训练新模型。"
        )

    def _view_writeback(self):
        """查看当前回写到实验 config 的最小子集特征集。"""
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        minimal = (exp.get("config") or {}).get("minimal_features")
        if not minimal or not minimal.get("features"):
            QMessageBox.information(self, "提示", "当前实验还没有回写的最小子集特征集。\n请先在下方选择模型并点击「一键回写为新特征集」。")
            return
        feats = minimal["features"]
        QMessageBox.information(
            self, "当前回写的特征集",
            f"来源模型：{minimal.get('model', '未知')}\n"
            f"特征数：{len(feats)}\n"
            f"保存路径：{minimal.get('path', '')}\n\n"
            f"特征列表：\n{', '.join(feats)}\n\n"
            "在 模型 / Transformer / AutoML 选项卡的数据源中选择\n"
            "「最小子集特征数据」即可用该特征集训练。",
        )

    def _export_csv(self):
        if not self._results:
            QMessageBox.warning(self, "警告", "暂无分析结果")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出汇总", "minimal_subset_summary.csv", "CSV (*.csv)")
        if path:
            summary = ms.make_summary_table(self._results)
            summary.to_csv(path, index=False)
            QMessageBox.information(self, "成功", f"已保存：{path}")
