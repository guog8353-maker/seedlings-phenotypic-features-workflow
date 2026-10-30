"""AutoML 标签页（多选 + 数据源 + 最优配置展示）。"""

import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from core.model_trainer import compute_classification_metrics, compute_metrics_optimal_threshold
from sklearn.model_selection import train_test_split
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QSpinBox, QTextEdit, QCheckBox, QSplitter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.automl import AutoMLRunner, AUTOML_REGISTRY, is_gpu_available
from gui.widgets import CheckableListWidget, DataSourceSelector, load_modeling_data
from gui.widgets.result_utils import (
    delete_module_results, archive_module_results,
    load_successful_results, merge_result_rows,
)


PARAM_TOOLTIPS = {
    "data_source": "选择使用预处理后的完整数据，还是特征选择后的共识特征数据。",
    "framework_list": "选择需要运行的 AutoML 框架；可按住 Ctrl 多选。",
    "time_limit": "每个框架自动搜索模型与超参数的时间上限（秒）。建议 300~1800。",
    "check_gpu": "勾选后允许框架使用 GPU（需框架与驱动支持）。H2O/FLAML 通常 CPU 即可。",
}


class AutoMLTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>AutoML</h2>"))

        note = QLabel("说明：H2O/AutoGluon/FLAML 框架内部已自动搜索模型与超参数。")
        note.setStyleSheet("color: gray;")
        self.layout.addWidget(note)

        gpu_note = QLabel(f"GPU 检测：{'检测到 NVIDIA GPU' if is_gpu_available() else '未检测到可用 GPU'}")
        gpu_note.setStyleSheet("color: gray;")
        self.layout.addWidget(gpu_note)

        # 数据源选择
        self.data_source = DataSourceSelector()
        self.data_source.setToolTip(PARAM_TOOLTIPS["data_source"])
        self.layout.addWidget(self.data_source)

        # 框架选择
        self.framework_list = CheckableListWidget(title="AutoML 框架")
        self.framework_list.setToolTip(PARAM_TOOLTIPS["framework_list"])
        self.framework_list.add_items(list(AUTOML_REGISTRY.keys()))

        # 时间限制与 GPU
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("时间限制(秒):"))
        self.spin_time = QSpinBox()
        self.spin_time.setRange(60, 3600)
        self.spin_time.setValue(300)
        self.spin_time.setToolTip(PARAM_TOOLTIPS["time_limit"])
        time_layout.addWidget(self.spin_time)

        self.check_gpu = QCheckBox("使用 GPU（若可用）")
        self.check_gpu.setChecked(False)
        self.check_gpu.setToolTip(PARAM_TOOLTIPS["check_gpu"])
        time_layout.addWidget(self.check_gpu)

        time_layout.addWidget(QLabel("内存预算(GB):"))
        self.spin_mem_budget = QSpinBox()
        self.spin_mem_budget.setRange(0, 128)
        self.spin_mem_budget.setValue(0)
        self.spin_mem_budget.setToolTip(
            "仅 GPU 模式生效：AutoGluon 会按此预算自动选择配置档位\n"
            "（满配 / 轻配(无bagging) / 极简 / CPU 保底），只动 bagging 与并发，不动模型池。\n"
            "0 = 不限内存（按默认满配运行）。"
        )
        time_layout.addWidget(self.spin_mem_budget)

        self.check_skip_existing = QCheckBox("增量训练（跳过已有结果的框架）")
        self.check_skip_existing.setChecked(True)
        self.check_skip_existing.setToolTip(
            "勾选后：已有成功结果（结果表或已保存的预测文件）的框架会被跳过，\n"
            "只运行缺失的框架；新结果会与历史结果合并，不会覆盖丢失。\n"
            "取消勾选则重跑所有选中框架。"
        )
        time_layout.addWidget(self.check_skip_existing)
        time_layout.addStretch()

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清空选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_run = QPushButton("运行 AutoML")
        self.btn_run.clicked.connect(self.run_automl)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_run)

        # 结果管理按钮
        manage_layout = QHBoxLayout()
        self.btn_delete_results = QPushButton("删除本次结果")
        self.btn_delete_results.clicked.connect(self.delete_results)
        self.btn_archive_results = QPushButton("归档本次结果")
        self.btn_archive_results.clicked.connect(self.archive_results)
        manage_layout.addWidget(self.btn_delete_results)
        manage_layout.addWidget(self.btn_archive_results)
        manage_layout.addStretch()

        self.lbl_status = QLabel("未运行")

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: gray;")

        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setStyleSheet("color: gray;")

        # 结果表
        self.result_table = QTableWidget()

        # 最优配置展示
        config_label = QLabel("<b>各框架最优配置</b>")
        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(200)

        # 上下可拖拽调节：框架选择区 / 参数与按钮区 / 结果表与配置区
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(time_layout)
        controls_layout.addLayout(btn_layout)
        controls_layout.addLayout(manage_layout)
        controls_layout.addWidget(self.lbl_status)
        controls_layout.addWidget(self.lbl_info)
        controls_layout.addWidget(self.lbl_estimate)

        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self.result_table)
        bottom_layout.addWidget(config_label)
        bottom_layout.addWidget(self.config_text)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.framework_list)
        splitter.addWidget(controls_container)
        splitter.addWidget(bottom_container)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        splitter.setChildrenCollapsible(False)
        self.layout.addWidget(splitter)

    def clear_selection(self):
        self.framework_list.set_checked([])

    def _update_memory_estimate(self, info: dict):
        """根据当前数据规模与预算，刷新 AutoGluon 内存档位预估显示。"""
        if not self.check_gpu.isChecked():
            self.lbl_estimate.setText("内存预估：未启用 GPU，AutoGluon 走 CPU（预估峰值 ≈6.5GB）")
            return
        from core.automl import estimate_ag_memory_gb, select_ag_tier, AG_CPU_PEAK_GB
        n, m = info["n_samples"], info["n_features"]
        data_mb = n * m * 4 / 1e6
        budget = self.spin_mem_budget.value()
        ests = {t: estimate_ag_memory_gb(n, m, t)
                for t in ["L1_GPU_FULL", "L2_GPU_LIGHT", "L3_GPU_MIN"]}
        if budget <= 0:
            chosen = "L1_GPU_FULL（不限预算，满配）"
        else:
            chosen = select_ag_tier(n, m, budget)
        tier_cn = {
            "L1_GPU_FULL": "GPU 满配", "L2_GPU_LIGHT": "GPU 轻配(无bagging)",
            "L3_GPU_MIN": "GPU 极简", "L4_CPU": "CPU 保底",
        }
        self.lbl_estimate.setText(
            f"内存预估（数据 {n} 行 × {m} 列 ≈ {data_mb:.0f}MB）："
            f"满配 ≈{ests['L1_GPU_FULL']:.1f}GB｜轻配 ≈{ests['L2_GPU_LIGHT']:.1f}GB｜"
            f"极简 ≈{ests['L3_GPU_MIN']:.1f}GB｜CPU ≈{AG_CPU_PEAK_GB}GB"
            f" → 将采用【{tier_cn[chosen]}】"
        )

    def get_settings(self) -> dict:
        """收集当前 AutoML 配置。"""
        return {
            "frameworks": self.framework_list.get_checked(),
            "time_limit": self.spin_time.value(),
            "use_gpu": self.check_gpu.isChecked(),
            "memory_budget_gb": self.spin_mem_budget.value(),
            "skip_existing": self.check_skip_existing.isChecked(),
            "use_feature_selected": self.data_source.use_feature_selected(),
        }

    def load_settings(self, exp: dict):
        """从实验配置恢复 AutoML 设置。"""
        config = exp.get("config") or {}
        automl_config = config.get("automl")
        if not automl_config:
            return
        if "frameworks" in automl_config:
            self.framework_list.set_checked(automl_config["frameworks"])
        if "time_limit" in automl_config:
            self.spin_time.setValue(int(automl_config["time_limit"]))
        if "use_gpu" in automl_config:
            self.check_gpu.setChecked(automl_config["use_gpu"])
        if "memory_budget_gb" in automl_config:
            self.spin_mem_budget.setValue(int(automl_config["memory_budget_gb"]))
        if "skip_existing" in automl_config:
            self.check_skip_existing.setChecked(bool(automl_config["skip_existing"]))
        if "use_feature_selected" in automl_config:
            fs_available = bool(config.get("feature_selection", {}).get("consensus_table_path"))
            self.data_source.set_feature_selected_available(fs_available)
            self.data_source.set_use_feature_selected(bool(automl_config["use_feature_selected"]))

    def run_automl(self, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        prep_config = exp.get("config", {}).get("preprocessing")
        if not prep_config or not prep_config.get("processed_path"):
            QMessageBox.warning(self, "警告", "请先完成预处理")
            return

        fs_config = exp.get("config", {}).get("feature_selection")
        self.data_source.set_feature_selected_available(
            bool(fs_config and fs_config.get("consensus_table_path"))
        )
        self.data_source.set_minimal_available(
            bool((exp.get("config") or {}).get("minimal_features"))
        )

        frameworks = self.framework_list.get_checked()
        if not frameworks:
            QMessageBox.warning(self, "警告", "请至少选择一个 AutoML 框架")
            return

        try:
            X, y, info = load_modeling_data(
                exp,
                use_feature_selected=self.data_source.use_feature_selected(),
                use_minimal_subset=self.data_source.use_minimal_subset(),
            )
        except Exception as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        # 内存估算：GPU 模式按当前预算显示各档预估与将采用档位
        self._update_memory_estimate(info)

        output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "automl"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 增量训练：跳过已有成功结果的框架（结果表缺失时从 val 文件重建指标行）
        result_path = output_dir / "automl_result.csv"
        existing = pd.DataFrame()
        if self.check_skip_existing.isChecked():
            existing = load_successful_results(
                result_path, "framework",
                artifact_dir=output_dir, artifact_suffix="_val.csv", proba_col="val_proba",
            )
            done = set(existing["framework"].astype(str)) if not existing.empty else set()
            skipped = [f for f in frameworks if f in done]
            frameworks = [f for f in frameworks if f not in done]
            if skipped:
                self.parent_window.log_message(f"增量训练：跳过已有结果的 {len(skipped)} 个框架")

        if not frameworks and not existing.empty:
            existing = existing.sort_values("auc", ascending=False).reset_index(drop=True)
            self.lbl_status.setText(f"所有选中框架均已有结果：{len(existing)} 个")
            self.update_table(existing)
            self.parent_window.log_message("所有选中框架均已有结果，无需重跑")
            if not silent:
                QMessageBox.information(self, "提示", "所有选中框架均已有结果，无需重跑")
            return

        self.lbl_info.setText(
            f"数据：{info['data_source']} | 样本数：{info['n_samples']} | 特征数：{info['n_features']} | 输出：{output_dir}"
        )
        self.parent_window.log_message(f"AutoML 开始：{info['data_source']}，{len(frameworks)} 个框架")

        records = []
        best_configs = {}
        run_details = {}

        # 划分训练/验证集，避免在训练集上评估导致 AUC 虚高
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        for framework in frameworks:
            start_time = time.time()
            try:
                self.lbl_status.setText(f"正在运行：{framework}...")
                runner = AutoMLRunner(
                    framework,
                    params={
                        "time_limit": self.spin_time.value(),
                        "use_gpu": self.check_gpu.isChecked(),
                        "memory_budget_gb": self.spin_mem_budget.value(),
                    },
                )
                runner.fit(X_train, y_train)
                proba = runner.predict_proba(X_val)
                metrics = compute_metrics_optimal_threshold(y_val.values, proba, "youden")
                val_pred = (proba >= metrics["threshold"]).astype(int)
                elapsed = round(time.time() - start_time, 2)

                # 保存验证集预测，供后续绘图/历史记录使用
                val_df = pd.DataFrame({
                    "y_true": y_val.values,
                    "val_proba": proba,
                    "val_pred": val_pred,
                })
                val_path = output_dir / f"{framework}_val.csv"
                val_df.to_csv(val_path, index=False)

                # 保存 AutoML 模型/权重
                model_paths = self._save_automl_model(framework, runner, output_dir)

                # 提取最优配置
                best_config = self._extract_best_config(framework, runner)
                best_configs[framework] = best_config

                records.append({
                    "framework": framework,
                    **metrics,
                    "train_time_sec": elapsed,
                    "time_limit": self.spin_time.value(),
                    "use_gpu": self.check_gpu.isChecked(),
                    "best_model_summary": str(best_config.get("model", "N/A")),
                    **model_paths,
                })
                run_details[framework] = {"status": "success", "auc": round(metrics['auc'], 4), "train_time_sec": elapsed, "val_path": str(val_path), "best_config": best_config, "model_paths": model_paths}
                self.parent_window.log_message(f"AutoML {framework}：AUC={metrics['auc']:.4f}，耗时 {elapsed}s")
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                self.parent_window.logger.warning(f"AutoML {framework} 失败：{e}")
                records.append({"framework": framework, "auc": 0.0, "auprc": 0.0, "accuracy": 0.0,
                                "sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0, "npv": 0.0,
                                "f1": 0.0, "kappa": 0.0, "logloss": 0.0, "brier": 0.0,
                                "train_time_sec": elapsed, "time_limit": self.spin_time.value(), "use_gpu": self.check_gpu.isChecked(), "error": str(e)})
                best_configs[framework] = {"error": str(e)}
                run_details[framework] = {"status": "failed", "error": str(e), "train_time_sec": elapsed}

        result = merge_result_rows(existing, records, "framework")
        result.to_csv(result_path, index=False)

        details_path = output_dir / "automl_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump({
                "frameworks": frameworks,
                "data_source": info["data_source"],
                "n_samples": info["n_samples"],
                "n_features": info["n_features"],
                "time_limit": self.spin_time.value(),
                "use_gpu": self.check_gpu.isChecked(),
                "gpu_available": is_gpu_available(),
                "best_configs": best_configs,
                "run_details": run_details,
            }, f, ensure_ascii=False, indent=2, default=str)

        config = exp.get("config") or {}
        config["automl"] = {
            **self.get_settings(),
            "best_configs": best_configs,
            "result_path": str(result_path),
            "details_path": str(details_path),
            "data_source": info["data_source"],
            "processed_path": info["processed_path"],
        }
        self.parent_window.db.update_experiment(
            self.parent_window.current_experiment_id,
            {"config": config}
        )

        self.lbl_status.setText(f"AutoML 完成：{len(result)} 个框架")
        self.update_table(result)
        self.config_text.setText(self._format_configs(best_configs))
        if not silent:
            QMessageBox.information(self, "成功", "AutoML 完成")

    def _save_automl_model(self, framework: str, runner: AutoMLRunner, output_dir: Path) -> dict:
        """按框架保存 AutoML 模型/权重到实验目录。"""
        paths = {}
        try:
            if framework == "H2O":
                import h2o
                save_path = output_dir / f"{framework}_model"
                h2o.save_model(runner.wrapper_.model_, path=str(save_path), force=True)
                paths["model_dir"] = str(save_path)
            elif framework == "AutoGluon":
                # AutoGluon 已保存在 predictor.path；复制到实验目录便于归档
                src = Path(runner.wrapper_.model_.path)
                dst = output_dir / f"{framework}_model"
                if src.exists():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                paths["model_dir"] = str(dst)
            elif framework == "FLAML":
                pkl_path = output_dir / f"{framework}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(runner.wrapper_.model_, f)
                paths["model_path"] = str(pkl_path)
        except Exception as e:
            paths["save_error"] = str(e)
            self.parent_window.logger.warning(f"AutoML {framework} 模型保存失败：{e}")
        return paths

    def _extract_best_config(self, framework: str, runner: AutoMLRunner) -> dict:
        """从 AutoML 运行器中提取最优模型配置摘要。"""
        try:
            if framework == "H2O":
                model = runner.wrapper_.model_
                return {"model": model.model_id if hasattr(model, "model_id") else str(model)}
            if framework == "AutoGluon":
                leaderboard = runner.wrapper_.model_.leaderboard(silent=True)
                best = leaderboard.iloc[0]
                return {
                    "model": str(best.get("model", "N/A")),
                    "score_val": float(best.get("score_val", 0)),
                }
            if framework == "FLAML":
                automl = runner.wrapper_.model_
                return {
                    "best_estimator": str(automl.best_estimator),
                    "best_config": automl.best_config,
                }
        except Exception as e:
            return {"error": str(e)}
        return {}

    def _format_configs(self, configs: dict) -> str:
        import json
        return json.dumps(configs, indent=2, ensure_ascii=False, default=str)

    def update_table(self, df: pd.DataFrame):
        self.result_table.setRowCount(df.shape[0])
        self.result_table.setColumnCount(df.shape[1])
        self.result_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.result_table.setItem(i, j, QTableWidgetItem(text))

    def _get_result_dir(self) -> Optional[Path]:
        if self.parent_window.current_experiment_id is None:
            return None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        automl_config = config.get("automl")
        if not automl_config or not automl_config.get("result_path"):
            return None
        return Path(automl_config["result_path"]).parent

    def _clear_config(self):
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        if "automl" in config:
            del config["automl"]
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id, {"config": config}
            )

    def delete_results(self):
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可删除的 AutoML 结果")
            return
        reply = QMessageBox.question(
            self, "确认删除", f"确定删除以下目录及其所有内容？\n{result_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        ok, msg = delete_module_results(
            result_dir, table_widget=self.result_table,
            status_label=self.lbl_status, logger=self.parent_window.logger,
        )
        if ok:
            self._clear_config()
            self.config_text.clear()
            self.parent_window.log_message(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def archive_results(self):
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可归档的 AutoML 结果")
            return
        ok, msg = archive_module_results(result_dir, logger=self.parent_window.logger)
        if ok:
            self._clear_config()
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            self.config_text.clear()
            self.lbl_status.setText("结果已归档")
            self.parent_window.log_message(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def load_results(self, exp: dict):
        """加载历史 AutoML 结果到表格和配置区。"""
        config = exp.get("config") or {}
        automl_config = config.get("automl")
        if not automl_config:
            return
        # 先恢复设置
        self.load_settings(exp)
        if not automl_config.get("result_path"):
            return
        result_path = Path(automl_config["result_path"])
        if not result_path.exists():
            return
        try:
            result = pd.read_csv(result_path)
            self.update_table(result)
            best_configs = automl_config.get("best_configs", {})
            self.config_text.setText(self._format_configs(best_configs))
            self.lbl_status.setText(f"已加载历史结果：{len(result)} 个框架")
            self.parent_window.log_message(f"AutoML 历史结果已加载：{result_path}")
        except Exception as e:
            self.parent_window.logger.warning(f"加载 AutoML 历史结果失败：{e}")
