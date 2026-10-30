"""集成学习标签页（多选 + 折叠超参 + Optuna）。"""

import sys
from pathlib import Path
from typing import Optional

import pickle
import pandas as pd
import numpy as np
from core.model_trainer import compute_classification_metrics, compute_metrics_optimal_threshold
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox,
    QSpinBox, QComboBox, QFormLayout, QGroupBox, QTextEdit, QSplitter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ensemble import EnsembleRunner, ENSEMBLE_REGISTRY
from core.ensemble_optimizer import EnsembleOptimizer
from gui.widgets import CheckableListWidget, CollapsibleGroupBox, DataSourceSelector
from gui.widgets.result_utils import (
    delete_module_results, archive_module_results,
    load_successful_results, merge_result_rows,
)


META_LEARNER_NAMES = ["LR", "GBM", "RF", "MLP"]


PARAM_TOOLTIPS = {
    "data_source": "集成学习基于基础模型的 OOF 预测进行二次组合，通常不需要再选择数据源。",
    "method_list": "选择集成策略；建议至少选 2~3 种进行对比。",
    "ga_population_size": "遗传算法种群大小。越大搜索越充分但越慢，建议 30~100。",
    "ga_generations": "遗传算法进化代数。建议 20~100。",
    "stacking_meta_learner": "Stacking 的元学习器，LR 最不易过拟合，GBM/RF 更强但需防过拟合。",
    "stacking_cv": "Stacking 生成元特征时的 CV 折数。建议 3~10。",
    "caruana_max_models": "Caruana 集成最多可挑选的基模型数。建议 5~20。",
    "superlearner_meta_learner": "SuperLearner 的元学习器，推荐 LR。",
    "superlearner_cv": "SuperLearner 的 CV 折数。建议 3~10。",
    "check_optuna": "对支持的方法使用 Optuna 自动搜索超参数。",
    "spin_trials": "Optuna 搜索 trial 数。建议 20~100。",
}


class EnsembleTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>集成学习</h2>"))

        note = QLabel("说明：集成学习基于基础模型的 OOF 预测进行二次组合。")
        note.setStyleSheet("color: gray;")
        self.layout.addWidget(note)

        # 数据源选择
        self.data_source = DataSourceSelector()
        self.data_source.setToolTip(PARAM_TOOLTIPS["data_source"])
        self.layout.addWidget(self.data_source)

        # 方法选择
        self.method_list = CheckableListWidget(title="集成方法")
        self.method_list.setToolTip(PARAM_TOOLTIPS["method_list"])
        self.method_list.add_items(list(ENSEMBLE_REGISTRY.keys()))

        # 可折叠超参数面板
        self.params_collapsible = CollapsibleGroupBox("集成方法超参数配置")
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self.param_widgets = {}

        # GA
        gb_ga = QGroupBox("GASelection")
        form_ga = QFormLayout(gb_ga)
        self.spin_ga_pop = QSpinBox(); self.spin_ga_pop.setRange(10, 200); self.spin_ga_pop.setValue(50)
        self.spin_ga_pop.setToolTip(PARAM_TOOLTIPS["ga_population_size"])
        self.spin_ga_gen = QSpinBox(); self.spin_ga_gen.setRange(10, 200); self.spin_ga_gen.setValue(30)
        self.spin_ga_gen.setToolTip(PARAM_TOOLTIPS["ga_generations"])
        form_ga.addRow("population_size", self.spin_ga_pop)
        form_ga.addRow("generations", self.spin_ga_gen)
        params_layout.addWidget(gb_ga)
        self.param_widgets["GASelection"] = {"population_size": self.spin_ga_pop, "generations": self.spin_ga_gen}

        # Stacking
        gb_stack = QGroupBox("Stacking")
        form_stack = QFormLayout(gb_stack)
        self.combo_stack_meta = QComboBox(); self.combo_stack_meta.addItems(META_LEARNER_NAMES); self.combo_stack_meta.setCurrentText("LR")
        self.combo_stack_meta.setToolTip(PARAM_TOOLTIPS["stacking_meta_learner"])
        self.spin_stack_cv = QSpinBox(); self.spin_stack_cv.setRange(2, 10); self.spin_stack_cv.setValue(5)
        self.spin_stack_cv.setToolTip(PARAM_TOOLTIPS["stacking_cv"])
        form_stack.addRow("meta_learner", self.combo_stack_meta)
        form_stack.addRow("cv", self.spin_stack_cv)
        params_layout.addWidget(gb_stack)
        self.param_widgets["Stacking"] = {"meta_learners": self.combo_stack_meta, "cv": self.spin_stack_cv}

        # Caruana / CaruanaOptimized
        gb_caruana = QGroupBox("CaruanaEnsembleSelection / CaruanaOptimized")
        form_caruana = QFormLayout(gb_caruana)
        self.spin_caruana_max = QSpinBox(); self.spin_caruana_max.setRange(2, 100); self.spin_caruana_max.setValue(10)
        self.spin_caruana_max.setToolTip(PARAM_TOOLTIPS["caruana_max_models"])
        form_caruana.addRow("max_models", self.spin_caruana_max)
        params_layout.addWidget(gb_caruana)
        self.param_widgets["CaruanaEnsembleSelection"] = {"max_models": self.spin_caruana_max}
        self.param_widgets["CaruanaOptimized"] = {"max_models": self.spin_caruana_max}

        # SuperLearner
        gb_sl = QGroupBox("SuperLearner")
        form_sl = QFormLayout(gb_sl)
        self.combo_sl_meta = QComboBox(); self.combo_sl_meta.addItems(META_LEARNER_NAMES); self.combo_sl_meta.setCurrentText("LR")
        self.combo_sl_meta.setToolTip(PARAM_TOOLTIPS["superlearner_meta_learner"])
        self.spin_sl_cv = QSpinBox(); self.spin_sl_cv.setRange(2, 10); self.spin_sl_cv.setValue(5)
        self.spin_sl_cv.setToolTip(PARAM_TOOLTIPS["superlearner_cv"])
        form_sl.addRow("meta_learner", self.combo_sl_meta)
        form_sl.addRow("cv", self.spin_sl_cv)
        params_layout.addWidget(gb_sl)
        self.param_widgets["SuperLearner"] = {"meta_learner": self.combo_sl_meta, "cv": self.spin_sl_cv}

        # DES-KNORA（动态集成选择，v2.1）
        gb_des = QGroupBox("DES-KNORA（动态集成选择）")
        form_des = QFormLayout(gb_des)
        self.spin_des_k = QSpinBox(); self.spin_des_k.setRange(3, 25); self.spin_des_k.setValue(7)
        self.spin_des_k.setToolTip("邻域大小 k：在原始特征空间中取多少个近邻评估基模型局部胜任力。建议 5~11。")
        self.combo_des_algo = QComboBox(); self.combo_des_algo.addItems(["KNORA-E", "KNORA-U"])
        self.combo_des_algo.setToolTip("KNORA-E：邻域内全部预测正确的模型才入选（无则退化为 KNORA-U）；\nKNORA-U：邻域内至少一个预测正确即入选。")
        self.spin_des_cv = QSpinBox(); self.spin_des_cv.setRange(2, 10); self.spin_des_cv.setValue(5)
        self.spin_des_cv.setToolTip("生成诚实 OOF 预测的交叉验证折数。")
        form_des.addRow("k（邻居数）", self.spin_des_k)
        form_des.addRow("algorithm", self.combo_des_algo)
        form_des.addRow("cv", self.spin_des_cv)
        params_layout.addWidget(gb_des)
        self.param_widgets["DES-KNORA"] = {
            "k": self.spin_des_k, "algorithm": self.combo_des_algo, "cv": self.spin_des_cv,
        }

        # 无参方法说明
        gb_none = QGroupBox("SimpleAverage / WeightedAverage / HardVoting")
        form_none = QFormLayout(gb_none)
        form_none.addRow(QLabel("无超参数"))
        params_layout.addWidget(gb_none)

        self.params_collapsible.set_content(params_widget)
        self.layout.addWidget(self.params_collapsible)

        # Optuna
        optuna_layout = QHBoxLayout()
        self.check_optuna = QCheckBox("使用 Optuna 自动调参")
        self.check_optuna.setChecked(False)
        self.check_optuna.setToolTip(PARAM_TOOLTIPS["check_optuna"])
        optuna_layout.addWidget(self.check_optuna)
        optuna_layout.addWidget(QLabel("trials:"))
        self.spin_trials = QSpinBox(); self.spin_trials.setRange(5, 200); self.spin_trials.setValue(20)
        self.spin_trials.setToolTip(PARAM_TOOLTIPS["spin_trials"])
        optuna_layout.addWidget(self.spin_trials)

        self.check_skip_existing = QCheckBox("增量训练（跳过已有结果的方法）")
        self.check_skip_existing.setChecked(True)
        self.check_skip_existing.setToolTip(
            "勾选后：已有成功结果（结果表或已保存的 OOF 文件）的集成方法会被跳过，\n"
            "只运行缺失的方法；新结果会与历史结果合并，不会覆盖丢失。\n"
            "取消勾选则重跑所有选中方法。"
        )
        optuna_layout.addWidget(self.check_skip_existing)
        optuna_layout.addStretch()

        # 运行按钮
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清空选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_run = QPushButton("运行集成学习")
        self.btn_run.clicked.connect(self.run_ensemble)
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

        self.result_table = QTableWidget()

        # 上下可拖拽调节：方法选择区 / 参数与按钮区 / 结果表
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(optuna_layout)
        controls_layout.addLayout(btn_layout)
        controls_layout.addLayout(manage_layout)
        controls_layout.addWidget(self.lbl_status)
        controls_layout.addWidget(self.lbl_info)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.method_list)
        splitter.addWidget(controls_container)
        splitter.addWidget(self.result_table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        splitter.setChildrenCollapsible(False)
        self.layout.addWidget(splitter)

    def clear_selection(self):
        self.method_list.set_checked([])

    def get_settings(self) -> dict:
        """收集当前集成学习配置。"""
        return {
            "methods": self.method_list.get_checked(),
            "use_optuna": self.check_optuna.isChecked(),
            "n_trials": self.spin_trials.value(),
            "skip_existing": self.check_skip_existing.isChecked(),
            "params": {
                "GASelection": {"population_size": self.spin_ga_pop.value(), "generations": self.spin_ga_gen.value()},
                "Stacking": {"meta_learner": self.combo_stack_meta.currentText(), "cv": self.spin_stack_cv.value()},
                "CaruanaEnsembleSelection": {"max_models": self.spin_caruana_max.value()},
                "CaruanaOptimized": {"max_models": self.spin_caruana_max.value()},
                "SuperLearner": {"meta_learner": self.combo_sl_meta.currentText(), "cv": self.spin_sl_cv.value()},
            },
        }

    def load_settings(self, exp: dict):
        """从实验配置恢复集成学习设置。"""
        config = exp.get("config") or {}
        ensemble_config = config.get("ensemble")
        if not ensemble_config:
            return
        if "methods" in ensemble_config:
            self.method_list.set_checked(ensemble_config["methods"])
        if "use_optuna" in ensemble_config:
            self.check_optuna.setChecked(ensemble_config["use_optuna"])
        if "n_trials" in ensemble_config:
            self.spin_trials.setValue(int(ensemble_config["n_trials"]))
        if "skip_existing" in ensemble_config:
            self.check_skip_existing.setChecked(bool(ensemble_config["skip_existing"]))
        saved_params = ensemble_config.get("params", {})
        ga = saved_params.get("GASelection", {})
        if "population_size" in ga:
            self.spin_ga_pop.setValue(int(ga["population_size"]))
        if "generations" in ga:
            self.spin_ga_gen.setValue(int(ga["generations"]))
        stack = saved_params.get("Stacking", {})
        if "meta_learner" in stack:
            self.combo_stack_meta.setCurrentText(str(stack["meta_learner"]))
        if "cv" in stack:
            self.spin_stack_cv.setValue(int(stack["cv"]))
        caruana = saved_params.get("CaruanaEnsembleSelection", {})
        if "max_models" in caruana:
            self.spin_caruana_max.setValue(int(caruana["max_models"]))
        sl = saved_params.get("SuperLearner", {})
        if "meta_learner" in sl:
            self.combo_sl_meta.setCurrentText(str(sl["meta_learner"]))
        if "cv" in sl:
            self.spin_sl_cv.setValue(int(sl["cv"]))

    def _collect_params(self, method: str) -> dict:
        widgets = self.param_widgets.get(method, {})
        params = {}
        for k, w in widgets.items():
            if isinstance(w, QSpinBox):
                params[k] = w.value()
            elif isinstance(w, QComboBox):
                val = w.currentText()
                if method == "Stacking":
                    params[k] = [val]
                elif method == "SuperLearner":
                    params[k] = val  # 保存字符串，运行时再解析为元学习器
                else:
                    params[k] = val
        return params

    def _load_oof_data(self):
        if self.parent_window.current_experiment_id is None:
            raise ValueError("请先创建或加载实验")

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")

        if not modeling_config or not modeling_config.get("output_dir"):
            raise ValueError("请先完成模型训练")

        model_dir = Path(modeling_config["output_dir"])
        oof_files = sorted(model_dir.glob("*_oof.csv"))
        if len(oof_files) < 2:
            raise ValueError("至少需要 2 个模型的 OOF 预测结果")

        oof_data = {}
        y_true = None
        for f in oof_files:
            model_name = f.stem.replace("_oof", "")
            df = pd.read_csv(f)
            oof_data[model_name] = df["oof_proba"].values
            if y_true is None:
                y_true = df["y_true"].values

        return pd.DataFrame(oof_data), y_true

    def run_ensemble(self, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        methods = self.method_list.get_checked()
        if not methods:
            QMessageBox.warning(self, "警告", "请至少选择一种集成方法")
            return

        try:
            oof_df, y_true = self._load_oof_data()
        except Exception as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        output_dir = Path(self.parent_window.output_dir) / f"exp_{self.parent_window.current_experiment_id}" / "ensemble"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 增量训练：跳过已有成功结果的方法（结果表缺失时从 OOF 文件重建指标行）
        result_path = output_dir / "ensemble_result.csv"
        existing = pd.DataFrame()
        if self.check_skip_existing.isChecked():
            existing = load_successful_results(
                result_path, "method",
                artifact_dir=output_dir, artifact_suffix="_oof.csv", proba_col="oof_proba",
            )
            done = set(existing["method"].astype(str)) if not existing.empty else set()
            skipped = [m for m in methods if m in done]
            methods = [m for m in methods if m not in done]
            if skipped:
                self.parent_window.log_message(f"增量训练：跳过已有结果的 {len(skipped)} 种方法")

        if not methods and not existing.empty:
            existing = existing.sort_values("auc", ascending=False).reset_index(drop=True)
            self.lbl_status.setText(f"所有选中方法均已有结果：{len(existing)} 种")
            self.update_table(existing)
            self.parent_window.log_message("所有选中方法均已有结果，无需重跑")
            if not silent:
                QMessageBox.information(self, "提示", "所有选中方法均已有结果，无需重跑")
            return

        records = []
        best_configs = {}

        self.parent_window.log_message(f"集成学习开始：{len(methods)} 种方法，基模型数 {oof_df.shape[1]}")
        self.lbl_info.setText(f"基模型数：{oof_df.shape[1]}，样本数：{len(y_true)}，输出：{output_dir}")

        for method in methods:
            try:
                params = self._collect_params(method)

                # DES-KNORA 需要原始特征矩阵做邻域计算（行序与 OOF 一致）
                if method == "DES-KNORA":
                    try:
                        from gui.widgets import load_modeling_data
                        exp = self.parent_window.db.get_experiment(
                            self.parent_window.current_experiment_id)
                        try:
                            X_full_df, _, _ = load_modeling_data(exp, use_feature_selected=True)
                        except Exception:
                            X_full_df, _, _ = load_modeling_data(exp, use_feature_selected=False)
                        params["X_full"] = X_full_df.values
                    except Exception as e:
                        raise ValueError(f"DES-KNORA 需要可加载的原始特征数据：{e}")

                if self.check_optuna.isChecked() and method in {"GASelection", "Stacking", "CaruanaEnsembleSelection", "CaruanaOptimized", "SuperLearner"}:
                    self.lbl_status.setText(f"正在用 Optuna 调参：{method}...")
                    optimizer = EnsembleOptimizer(oof_df, y_true, random_state=42)
                    best_params = optimizer.optimize(method, n_trials=self.spin_trials.value())
                    params = {**params, **best_params}
                    best_configs[method] = params

                runner = EnsembleRunner(method, method_params=params)
                runner.fit(oof_df, pd.Series(y_true))
                # Stacking / SuperLearner 在 fit 中已生成 OOF 预测，用 OOF 评估避免过拟合
                if hasattr(runner.ensemble_, "oof_proba_"):
                    proba = runner.ensemble_.oof_proba_
                else:
                    proba = runner.predict_proba(oof_df)

                metrics = compute_metrics_optimal_threshold(y_true, proba, "youden")
                pred = (proba >= metrics["threshold"]).astype(int)

                # 保存集成方法的 OOF 预测，供后续绘图/历史记录使用
                oof_df_method = pd.DataFrame({
                    "y_true": y_true,
                    "oof_proba": proba,
                    "oof_pred": pred,
                })
                method_oof_path = output_dir / f"{method}_oof.csv"
                oof_df_method.to_csv(method_oof_path, index=False)

                # 保存集成模型对象（pkl）
                model_save_path = output_dir / f"{method}.pkl"
                with open(model_save_path, "wb") as f:
                    pickle.dump(runner, f)

                records.append({
                    "method": method,
                    **metrics,
                    "oof_path": str(method_oof_path),
                    "model_path": str(model_save_path),
                })
                self.parent_window.log_message(f"集成 {method}：AUC={metrics['auc']:.4f}")
            except Exception as e:
                warnings_msg = f"{method} 失败：{e}"
                self.parent_window.logger.warning(warnings_msg)
                records.append({"method": method, "auc": 0.0, "auprc": 0.0, "accuracy": 0.0,
                                "sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0, "npv": 0.0,
                                "f1": 0.0, "kappa": 0.0, "logloss": 0.0, "brier": 0.0, "error": str(e)})

        result = merge_result_rows(existing, records, "method")
        result.to_csv(result_path, index=False)

        config = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id).get("config") or {}
        config["ensemble"] = {
            **self.get_settings(),
            "result_path": str(result_path),
            "best_configs": best_configs,
            "data_source": "OOF_predictions",
        }
        self.parent_window.db.update_experiment(
            self.parent_window.current_experiment_id,
            {"config": config}
        )

        self.lbl_status.setText(f"集成学习完成：{len(result)} 种方法")
        self.update_table(result)
        if not silent:
            QMessageBox.information(self, "成功", "集成学习完成")

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
        """返回当前实验的集成学习结果目录。"""
        if self.parent_window.current_experiment_id is None:
            return None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        ensemble_config = config.get("ensemble")
        if not ensemble_config or not ensemble_config.get("result_path"):
            return None
        return Path(ensemble_config["result_path"]).parent

    def _clear_config(self):
        """清空当前实验的集成学习配置。"""
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        if "ensemble" in config:
            del config["ensemble"]
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id, {"config": config}
            )

    def delete_results(self):
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可删除的集成学习结果")
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
            self.parent_window.log_message(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def archive_results(self):
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可归档的集成学习结果")
            return
        ok, msg = archive_module_results(result_dir, logger=self.parent_window.logger)
        if ok:
            self._clear_config()
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)
            self.lbl_status.setText("结果已归档")
            self.parent_window.log_message(msg)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "失败", msg)

    def load_results(self, exp: dict):
        """加载历史集成学习结果到表格。"""
        config = exp.get("config") or {}
        ensemble_config = config.get("ensemble")
        if not ensemble_config:
            return
        # 先恢复设置
        self.load_settings(exp)
        if not ensemble_config.get("result_path"):
            return
        result_path = Path(ensemble_config["result_path"])
        if not result_path.exists():
            return
        try:
            result = pd.read_csv(result_path)
            self.update_table(result)
            self.lbl_status.setText(f"已加载历史结果：{len(result)} 种方法")
            self.parent_window.log_message(f"集成学习历史结果已加载：{result_path}")
        except Exception as e:
            self.parent_window.logger.warning(f"加载集成学习历史结果失败：{e}")
