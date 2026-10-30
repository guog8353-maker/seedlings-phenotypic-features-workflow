"""生存分析标签页（多选 + 折叠超参 + Optuna）。"""

import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.survival import SurvivalRunner, SURVIVAL_REGISTRY
from core.survival_optimizer import SurvivalOptimizer
from gui.widgets import CheckableListWidget, CollapsibleGroupBox, DataSourceSelector, load_survival_data


class SurvivalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>生存分析</h2>"))

        # 数据源选择
        self.data_source = DataSourceSelector()
        self.layout.addWidget(self.data_source)

        # 模型选择
        self.model_list = CheckableListWidget(title="生存模型")
        self.model_list.add_items(list(SURVIVAL_REGISTRY.keys()))
        self.layout.addWidget(self.model_list)

        # 可折叠超参数
        self.params_collapsible = CollapsibleGroupBox("生存模型超参数配置")
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)

        # CoxPH / WeightedCoxPH
        gb_cox = QGroupBox("CoxPH / WeightedCoxPH")
        form_cox = QFormLayout(gb_cox)
        self.spin_penalizer = QDoubleSpinBox()
        self.spin_penalizer.setRange(1e-4, 100.0)
        self.spin_penalizer.setSingleStep(0.1)
        self.spin_penalizer.setDecimals(4)
        self.spin_penalizer.setValue(0.1)
        form_cox.addRow("penalizer", self.spin_penalizer)
        params_layout.addWidget(gb_cox)

        # DeepSurv
        gb_ds = QGroupBox("DeepSurv")
        form_ds = QFormLayout(gb_ds)
        self.spin_ds_dropout = QDoubleSpinBox(); self.spin_ds_dropout.setRange(0.0, 0.8); self.spin_ds_dropout.setSingleStep(0.05); self.spin_ds_dropout.setValue(0.2)
        self.spin_ds_lr = QDoubleSpinBox(); self.spin_ds_lr.setRange(1e-5, 1e-1); self.spin_ds_lr.setSingleStep(0.0001); self.spin_ds_lr.setDecimals(6); self.spin_ds_lr.setValue(1e-3)
        self.spin_ds_epochs = QSpinBox(); self.spin_ds_epochs.setRange(10, 500); self.spin_ds_epochs.setValue(100)
        self.spin_ds_batch = QSpinBox(); self.spin_ds_batch.setRange(16, 1024); self.spin_ds_batch.setValue(256)
        self.spin_ds_hidden1 = QSpinBox(); self.spin_ds_hidden1.setRange(16, 256); self.spin_ds_hidden1.setValue(64)
        self.spin_ds_hidden2 = QSpinBox(); self.spin_ds_hidden2.setRange(16, 256); self.spin_ds_hidden2.setValue(64)
        form_ds.addRow("dropout", self.spin_ds_dropout)
        form_ds.addRow("lr", self.spin_ds_lr)
        form_ds.addRow("epochs", self.spin_ds_epochs)
        form_ds.addRow("batch_size", self.spin_ds_batch)
        form_ds.addRow("hidden_dim_1", self.spin_ds_hidden1)
        form_ds.addRow("hidden_dim_2", self.spin_ds_hidden2)
        params_layout.addWidget(gb_ds)

        self.params_collapsible.set_content(params_widget)
        self.layout.addWidget(self.params_collapsible)

        # Optuna
        optuna_layout = QHBoxLayout()
        self.check_optuna = QCheckBox("使用 Optuna 自动调参")
        self.check_optuna.setChecked(False)
        optuna_layout.addWidget(self.check_optuna)
        optuna_layout.addWidget(QLabel("trials:"))
        self.spin_trials = QSpinBox(); self.spin_trials.setRange(3, 100); self.spin_trials.setValue(10)
        optuna_layout.addWidget(self.spin_trials)
        optuna_layout.addStretch()
        self.layout.addLayout(optuna_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清空选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_run = QPushButton("运行生存分析")
        self.btn_run.clicked.connect(self.run_survival)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_run)
        self.layout.addLayout(btn_layout)

        self.lbl_status = QLabel("未运行")
        self.layout.addWidget(self.lbl_status)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: gray;")
        self.layout.addWidget(self.lbl_info)

        self.result_table = QTableWidget()
        self.layout.addWidget(self.result_table)

    def clear_selection(self):
        self.model_list.set_checked([])

    def _collect_params(self, model_name: str) -> dict:
        if model_name in {"CoxPH", "WeightedCoxPH"}:
            return {"penalizer": self.spin_penalizer.value()}
        if model_name == "DeepSurv":
            return {
                "hidden_dims": (self.spin_ds_hidden1.value(), self.spin_ds_hidden2.value()),
                "dropout": self.spin_ds_dropout.value(),
                "lr": self.spin_ds_lr.value(),
                "num_epochs": self.spin_ds_epochs.value(),
                "batch_size": self.spin_ds_batch.value(),
            }
        return {}

    def run_survival(self, silent: bool = False):
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

        models = self.model_list.get_checked()
        if not models:
            QMessageBox.warning(self, "警告", "请至少选择一个生存模型")
            return

        try:
            X, time, event, weights, info = load_survival_data(
                exp, use_feature_selected=self.data_source.use_feature_selected()
            )
        except Exception as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "survival"
        output_dir.mkdir(parents=True, exist_ok=True)

        self.lbl_info.setText(
            f"数据：{info['data_source']} | 样本数：{info['n_samples']} | 特征数：{info['n_features']} | 输出：{output_dir}"
        )
        self.parent_window.log_message(f"生存分析开始：{info['data_source']}，{len(models)} 个模型")

        records = []
        best_configs = {}

        for model_name in models:
            try:
                params = self._collect_params(model_name)

                if self.check_optuna.isChecked() and model_name in {"CoxPH", "WeightedCoxPH", "DeepSurv"}:
                    self.lbl_status.setText(f"正在用 Optuna 调参：{model_name}...")
                    optimizer = SurvivalOptimizer(
                        model_name, X, time, event, weights,
                        random_state=42,
                    )
                    best_params = optimizer.optimize(n_trials=self.spin_trials.value())
                    params = {**params, **best_params}
                    best_configs[model_name] = params

                self.lbl_status.setText(f"正在训练：{model_name}...")
                runner = SurvivalRunner(model_name, params=params)
                runner.fit(X, time, event, weights)
                metrics = runner.evaluate(X, time, event)

                records.append({
                    "model": model_name,
                    "c_index": metrics.get("c_index"),
                    "brier": metrics.get("brier"),
                    "calibration_slope": metrics.get("calibration_slope"),
                })
                self.parent_window.log_message(
                    f"生存 {model_name}：C-index={metrics.get('c_index')}, Brier={metrics.get('brier')}"
                )
            except Exception as e:
                self.parent_window.logger.warning(f"生存 {model_name} 失败：{e}")
                records.append({"model": model_name, "c_index": 0.0, "error": str(e)})

        result = pd.DataFrame(records)
        result_path = output_dir / "survival_result.csv"
        result.to_csv(result_path, index=False)

        config = exp.get("config") or {}
        config["survival"] = {
            "models": models,
            "use_optuna": self.check_optuna.isChecked(),
            "n_trials": self.spin_trials.value(),
            "best_configs": best_configs,
            "result_path": str(result_path),
            "data_source": info["data_source"],
            "processed_path": info["processed_path"],
        }
        self.parent_window.db.update_experiment(
            self.parent_window.current_experiment_id,
            {"config": config}
        )

        self.lbl_status.setText(f"生存分析完成：{len(result)} 个模型")
        self.update_table(result)
        if not silent:
            QMessageBox.information(self, "成功", "生存分析完成")

    def update_table(self, df: pd.DataFrame):
        self.result_table.setRowCount(df.shape[0])
        self.result_table.setColumnCount(df.shape[1])
        self.result_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = f"{val:.4f}" if isinstance(val, float) else str(val)
                self.result_table.setItem(i, j, QTableWidgetItem(text))
