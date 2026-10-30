"""Transformer 标签页（多选 + 折叠超参 + Optuna）。"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from core.model_trainer import compute_classification_metrics, compute_metrics_optimal_threshold
from sklearn.model_selection import train_test_split
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSplitter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.transformers import TransformerTrainer, TRANSFORMER_REGISTRY, get_transformer_device_info
from core.transformer_optimizer import TransformerOptimizer
from gui.widgets import CheckableListWidget, CollapsibleGroupBox, DataSourceSelector, load_modeling_data
from gui.widgets.result_utils import (
    delete_module_results, archive_module_results,
    load_successful_results, merge_result_rows,
)


PARAM_TOOLTIPS = {
    "data_source": "选择使用预处理后的完整数据，还是特征选择后的共识特征数据。",
    "model_list": "选择需要训练的 Transformer 架构；可按住 Ctrl 多选。",
    "epochs": "训练轮数。数据量小或特征少时可适当减少，建议 20~100。",
    "dim": "嵌入维度 / 隐藏层维度。越大表达能力越强但参数越多，建议 16~128。",
    "depth": "Transformer 编码器层数。建议 1~6。",
    "heads": "多头注意力头数。需能被 dim 整除，建议 4~16。",
    "dropout": "Dropout 比例。用于防止过拟合，建议 0.0~0.5。",
    "learning_rate": "学习率。建议 1e-4 ~ 1e-2；过大容易震荡，过小收敛慢。",
    "weight_decay": "权重衰减 (L2 惩罚)。建议 1e-6 ~ 1e-3。",
    "batch_size": "训练批次大小。建议 32~512，根据显存/内存调整。",
    "check_optuna": "勾选后使用 Optuna 自动搜索 Transformer 超参数。",
    "spin_trials": "Optuna 搜索 trial 数。建议 5~50。",
}


class TransformerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>Transformer</h2>"))

        device_info = QLabel(f"设备状态：{get_transformer_device_info()}")
        device_info.setStyleSheet("color: gray;")
        self.layout.addWidget(device_info)

        # 数据源选择
        self.data_source = DataSourceSelector()
        self.data_source.setToolTip(PARAM_TOOLTIPS["data_source"])
        self.layout.addWidget(self.data_source)

        # 模型选择
        self.model_list = CheckableListWidget(title="Transformer 架构")
        self.model_list.setToolTip(PARAM_TOOLTIPS["model_list"])
        self.model_list.add_items(list(TRANSFORMER_REGISTRY.keys()))

        # 可折叠超参数
        self.params_collapsible = CollapsibleGroupBox("Transformer 超参数配置")
        params_widget = QWidget()
        params_layout = QFormLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)

        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(5, 500); self.spin_epochs.setValue(30)
        self.spin_epochs.setToolTip(PARAM_TOOLTIPS["epochs"])
        self.spin_dim = QSpinBox(); self.spin_dim.setRange(8, 128); self.spin_dim.setValue(32)
        self.spin_dim.setToolTip(PARAM_TOOLTIPS["dim"])
        self.spin_depth = QSpinBox(); self.spin_depth.setRange(1, 10); self.spin_depth.setValue(2)
        self.spin_depth.setToolTip(PARAM_TOOLTIPS["depth"])
        self.spin_heads = QSpinBox(); self.spin_heads.setRange(1, 16); self.spin_heads.setValue(8)
        self.spin_heads.setToolTip(PARAM_TOOLTIPS["heads"])
        self.spin_dropout = QDoubleSpinBox(); self.spin_dropout.setRange(0.0, 0.8); self.spin_dropout.setSingleStep(0.05); self.spin_dropout.setValue(0.1)
        self.spin_dropout.setToolTip(PARAM_TOOLTIPS["dropout"])
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(1e-5, 1e-1); self.spin_lr.setSingleStep(0.0001); self.spin_lr.setDecimals(6); self.spin_lr.setValue(1e-3)
        self.spin_lr.setToolTip(PARAM_TOOLTIPS["learning_rate"])
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setRange(1e-6, 1e-2); self.spin_wd.setSingleStep(0.0001); self.spin_wd.setDecimals(6); self.spin_wd.setValue(1e-5)
        self.spin_wd.setToolTip(PARAM_TOOLTIPS["weight_decay"])
        self.spin_batch = QSpinBox(); self.spin_batch.setRange(16, 1024); self.spin_batch.setValue(256)
        self.spin_batch.setToolTip(PARAM_TOOLTIPS["batch_size"])

        params_layout.addRow("epochs", self.spin_epochs)
        params_layout.addRow("dim", self.spin_dim)
        params_layout.addRow("depth", self.spin_depth)
        params_layout.addRow("heads", self.spin_heads)
        params_layout.addRow("dropout", self.spin_dropout)
        params_layout.addRow("learning_rate", self.spin_lr)
        params_layout.addRow("weight_decay", self.spin_wd)
        params_layout.addRow("batch_size", self.spin_batch)

        self.params_collapsible.set_content(params_widget)
        self.layout.addWidget(self.params_collapsible)

        # Optuna
        optuna_layout = QHBoxLayout()
        self.check_optuna = QCheckBox("使用 Optuna 自动调参")
        self.check_optuna.setChecked(False)
        self.check_optuna.setToolTip(PARAM_TOOLTIPS["check_optuna"])
        optuna_layout.addWidget(self.check_optuna)
        optuna_layout.addWidget(QLabel("trials:"))
        self.spin_trials = QSpinBox(); self.spin_trials.setRange(3, 100); self.spin_trials.setValue(10)
        self.spin_trials.setToolTip(PARAM_TOOLTIPS["spin_trials"])
        optuna_layout.addWidget(self.spin_trials)

        self.check_soup = QCheckBox("Greedy Soup 权重平均")
        self.check_soup.setChecked(False)
        self.check_soup.setToolTip(
            "勾选后：每个 torch 原生架构（FT-Transformer/TabTransformer/SAINT/sTFT）\n"
            "用多个随机种子各训一遍，按验证集 AUC 贪心做权重平均（Greedy Soup），\n"
            "通常零成本提升 1~2 个点的 AUC。训练耗时 ×种子数。\n"
            "外部模型（TabM/RealMLP/TabPFN/TabICL 等）本身已是集成实现，不受影响。"
        )
        optuna_layout.addWidget(self.check_soup)
        optuna_layout.addWidget(QLabel("种子数:"))
        self.spin_soup = QSpinBox(); self.spin_soup.setRange(2, 5); self.spin_soup.setValue(3)
        optuna_layout.addWidget(self.spin_soup)

        self.check_skip_existing = QCheckBox("增量训练（跳过已有结果的架构）")
        self.check_skip_existing.setChecked(True)
        self.check_skip_existing.setToolTip(
            "勾选后：已有成功结果（结果表或已保存的模型/预测文件）的架构会被跳过，\n"
            "只训练缺失的架构；新结果会与历史结果合并，不会覆盖丢失。\n"
            "取消勾选则重训所有选中架构。"
        )
        optuna_layout.addWidget(self.check_skip_existing)
        optuna_layout.addStretch()

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清空选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_run = QPushButton("运行 Transformer")
        self.btn_run.clicked.connect(self.run_transformer)
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

        # 上下可拖拽调节：架构选择区 / 参数与按钮区 / 结果表
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(optuna_layout)
        controls_layout.addLayout(btn_layout)
        controls_layout.addLayout(manage_layout)
        controls_layout.addWidget(self.lbl_status)
        controls_layout.addWidget(self.lbl_info)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.model_list)
        splitter.addWidget(controls_container)
        splitter.addWidget(self.result_table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        splitter.setChildrenCollapsible(False)
        self.layout.addWidget(splitter)

    def clear_selection(self):
        self.model_list.set_checked([])

    def get_settings(self) -> dict:
        """收集当前 Transformer 配置。"""
        return {
            "models": self.model_list.get_checked(),
            "use_optuna": self.check_optuna.isChecked(),
            "n_trials": self.spin_trials.value(),
            "use_feature_selected": self.data_source.use_feature_selected(),
            "skip_existing": self.check_skip_existing.isChecked(),
            "use_soup": self.check_soup.isChecked(),
            "n_soups": self.spin_soup.value(),
            "params": self._collect_params(),
        }

    def load_settings(self, exp: dict):
        """从实验配置恢复 Transformer 设置。"""
        config = exp.get("config") or {}
        transformer_config = config.get("transformer")
        if not transformer_config:
            return
        if "models" in transformer_config:
            self.model_list.set_checked(transformer_config["models"])
        if "use_optuna" in transformer_config:
            self.check_optuna.setChecked(transformer_config["use_optuna"])
        if "n_trials" in transformer_config:
            self.spin_trials.setValue(int(transformer_config["n_trials"]))
        if "use_feature_selected" in transformer_config:
            fs_available = bool(config.get("feature_selection", {}).get("consensus_table_path"))
            self.data_source.set_feature_selected_available(fs_available)
            self.data_source.set_use_feature_selected(bool(transformer_config["use_feature_selected"]))
        if "skip_existing" in transformer_config:
            self.check_skip_existing.setChecked(bool(transformer_config["skip_existing"]))
        if "use_soup" in transformer_config:
            self.check_soup.setChecked(bool(transformer_config["use_soup"]))
        if "n_soups" in transformer_config:
            self.spin_soup.setValue(int(transformer_config["n_soups"]))
        saved_params = transformer_config.get("params", {})
        mapping = {
            "num_epochs": self.spin_epochs,
            "dim": self.spin_dim,
            "depth": self.spin_depth,
            "heads": self.spin_heads,
            "dropout": self.spin_dropout,
            "lr": self.spin_lr,
            "weight_decay": self.spin_wd,
            "batch_size": self.spin_batch,
        }
        for k, w in mapping.items():
            if k in saved_params:
                val = saved_params[k]
                if isinstance(w, QSpinBox):
                    w.setValue(int(val))
                else:
                    w.setValue(float(val))

    def _collect_params(self) -> dict:
        return {
            "num_epochs": self.spin_epochs.value(),
            "dim": self.spin_dim.value(),
            "depth": self.spin_depth.value(),
            "heads": self.spin_heads.value(),
            "dropout": self.spin_dropout.value(),
            "lr": self.spin_lr.value(),
            "weight_decay": self.spin_wd.value(),
            "batch_size": self.spin_batch.value(),
        }

    def run_transformer(self, silent: bool = False):
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

        models = self.model_list.get_checked()
        if not models:
            QMessageBox.warning(self, "警告", "请至少选择一个 Transformer 架构")
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

        output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "transformer"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 增量训练：跳过已有成功结果的架构（结果表缺失时从 val 文件重建指标行）
        result_path = output_dir / "transformer_result.csv"
        existing = pd.DataFrame()
        if self.check_skip_existing.isChecked():
            existing = load_successful_results(
                result_path, "model",
                artifact_dir=output_dir, artifact_suffix="_val.csv", proba_col="val_proba",
            )
            done = set(existing["model"].astype(str)) if not existing.empty else set()
            skipped = [m for m in models if m in done]
            models = [m for m in models if m not in done]
            if skipped:
                self.parent_window.log_message(f"增量训练：跳过已有结果的 {len(skipped)} 个架构")

        if not models and not existing.empty:
            existing = existing.sort_values("auc", ascending=False).reset_index(drop=True)
            self.lbl_status.setText(f"所有选中架构均已有结果：{len(existing)} 个")
            self.update_table(existing)
            self.parent_window.log_message("所有选中架构均已有结果，无需重训")
            if not silent:
                QMessageBox.information(self, "提示", "所有选中架构均已有结果，无需重训")
            return

        self.lbl_info.setText(
            f"数据：{info['data_source']} | 样本数：{info['n_samples']} | 特征数：{info['n_features']} | 输出：{output_dir}"
        )
        self.parent_window.log_message(f"Transformer 开始：{info['data_source']}，{len(models)} 个架构")

        records = []
        best_configs = {}
        run_details = {}

        # 划分训练/验证集，避免在训练集上评估导致 AUC 虚高
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        for name in models:
            start_time = time.time()
            try:
                params = self._collect_params()
                if self.check_optuna.isChecked():
                    self.lbl_status.setText(f"正在用 Optuna 调参：{name}...")
                    optimizer = TransformerOptimizer(name, X_train, y_train, random_state=42)
                    best_params = optimizer.optimize(n_trials=self.spin_trials.value())
                    params = {**params, **best_params}
                    best_configs[name] = params

                self.lbl_status.setText(f"正在训练：{name}...")
                trainer = TransformerTrainer(name, params=params)
                # Greedy Soup：仅对本项目 torch 原生架构做多种子权重平均
                is_external = getattr(TRANSFORMER_REGISTRY[name], "is_external_wrapper", False)
                if self.check_soup.isChecked() and not is_external:
                    from core.transformers import train_greedy_soup
                    self.lbl_status.setText(f"正在训练（Soup×{self.spin_soup.value()}）：{name}...")
                    trainer.model_, soup_info = train_greedy_soup(
                        TRANSFORMER_REGISTRY[name], params,
                        X_train, y_train, X_val, y_val,
                        n_soups=self.spin_soup.value(),
                    )
                    best_configs.setdefault(name, {})["soup"] = soup_info
                    self.parent_window.log_message(
                        f"Soup [{name}]：成员种子 {soup_info['soup_members']}，"
                        f"验证 AUC={soup_info['val_auc']}"
                    )
                else:
                    trainer.fit(X_train, y_train)
                proba = trainer.predict_proba(X_val)
                metrics = compute_metrics_optimal_threshold(y_val.values, proba, "youden")
                val_pred = (proba >= metrics["threshold"]).astype(int)
                elapsed = round(time.time() - start_time, 2)

                # 保存验证集预测，供后续绘图/历史记录使用
                val_df = pd.DataFrame({
                    "y_true": y_val.values,
                    "val_proba": proba,
                    "val_pred": val_pred,
                })
                val_path = output_dir / f"{name}_val.csv"
                val_df.to_csv(val_path, index=False)

                # 保存 Transformer 模型：完整对象、state_dict、网络结构
                pkl_path = output_dir / f"{name}.pkl"
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(trainer.model_, f)
                except Exception:
                    # 部分第三方模型含 lambda/闭包（如 pytabkit TabR），标准 pickle
                    # 无法序列化；cloudpickle 可以，且产物仍可用标准 pickle 加载
                    try:
                        import cloudpickle
                        with open(pkl_path, "wb") as f:
                            cloudpickle.dump(trainer.model_, f)
                        self.parent_window.log_message(
                            f"Transformer {name}：标准 pickle 失败，已用 cloudpickle 保存"
                        )
                    except Exception as e2:
                        self.parent_window.logger.warning(
                            f"Transformer {name} 模型对象保存失败：{e2}"
                        )
                        pkl_path = None
                # 外部包装模型（pytabkit/TabPFN/TabICL）没有本项目的 state_dict，跳过 .pt
                if getattr(trainer.model_, "is_external_wrapper", False):
                    pt_path = None
                else:
                    pt_path = output_dir / f"{name}.pt"
                    torch.save(trainer.model_.model_.state_dict(), pt_path)
                arch_path = output_dir / f"{name}_architecture.json"
                arch_info = {
                    "name": name,
                    "num_dim": getattr(trainer.model_, "num_dim_", None),
                    "cat_dims": getattr(trainer.model_, "cat_dims_", None),
                    "cat_cols": getattr(trainer.model_, "cat_cols_", None),
                    "num_cols": getattr(trainer.model_, "num_cols_", None),
                    "num_classes": len(trainer.model_.le_.classes_) if hasattr(trainer.model_, "le_") else None,
                    "params": params,
                }
                with open(arch_path, "w", encoding="utf-8") as f:
                    json.dump(arch_info, f, ensure_ascii=False, indent=2, default=str)

                records.append({
                    "model": name,
                    **metrics,
                    "train_time_sec": elapsed,
                    "device": get_transformer_device_info(),
                    "model_path": str(pkl_path) if pkl_path else None,
                    "state_dict_path": str(pt_path) if pt_path else None,
                    "architecture_path": str(arch_path),
                    **{k: v for k, v in params.items()},
                })
                run_details[name] = {"status": "success", "auc": round(metrics['auc'], 4), "train_time_sec": elapsed, "val_path": str(val_path), "params": params}
                self.parent_window.log_message(f"Transformer {name}：AUC={metrics['auc']:.4f}，耗时 {elapsed}s")
            except Exception as e:
                elapsed = round(time.time() - start_time, 2)
                self.parent_window.logger.warning(f"Transformer {name} 失败：{e}")
                records.append({"model": name, "auc": 0.0, "auprc": 0.0, "accuracy": 0.0,
                                "sensitivity": 0.0, "specificity": 0.0, "ppv": 0.0, "npv": 0.0,
                                "f1": 0.0, "kappa": 0.0, "logloss": 0.0, "brier": 0.0,
                                "train_time_sec": elapsed, "device": get_transformer_device_info(), "error": str(e)})
                run_details[name] = {"status": "failed", "error": str(e), "train_time_sec": elapsed}

        result = merge_result_rows(existing, records, "model")
        result.to_csv(result_path, index=False)

        details_path = output_dir / "transformer_details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump({
                "models": models,
                "data_source": info["data_source"],
                "n_samples": info["n_samples"],
                "n_features": info["n_features"],
                "use_optuna": self.check_optuna.isChecked(),
                "n_trials": self.spin_trials.value(),
                "best_configs": best_configs,
                "run_details": run_details,
            }, f, ensure_ascii=False, indent=2, default=str)

        config = exp.get("config") or {}
        config["transformer"] = {
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

        self.lbl_status.setText(f"Transformer 完成：{len(result)} 个架构")
        self.update_table(result)
        if not silent:
            QMessageBox.information(self, "成功", "Transformer 训练完成")

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
        transformer_config = config.get("transformer")
        if not transformer_config or not transformer_config.get("result_path"):
            return None
        return Path(transformer_config["result_path"]).parent

    def _clear_config(self):
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        if "transformer" in config:
            del config["transformer"]
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id, {"config": config}
            )

    def delete_results(self):
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可删除的 Transformer 结果")
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
            QMessageBox.warning(self, "警告", "当前实验没有可归档的 Transformer 结果")
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
        """加载历史 Transformer 结果到表格。"""
        config = exp.get("config") or {}
        transformer_config = config.get("transformer")
        if not transformer_config:
            return
        # 先恢复设置
        self.load_settings(exp)
        if not transformer_config.get("result_path"):
            return
        result_path = Path(transformer_config["result_path"])
        if not result_path.exists():
            return
        try:
            result = pd.read_csv(result_path)
            self.update_table(result)
            self.lbl_status.setText(f"已加载历史结果：{len(result)} 个架构")
            self.parent_window.log_message(f"Transformer 历史结果已加载：{result_path}")
        except Exception as e:
            self.parent_window.logger.warning(f"加载 Transformer 历史结果失败：{e}")
