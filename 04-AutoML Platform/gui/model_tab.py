"""模型训练标签页。"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox,
    QSpinBox, QSplitter,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.base_models import get_model_names
from core.model_trainer import ModelTrainer
from gui.widgets import CheckableListWidget, DataSourceSelector, load_modeling_data
from gui.widgets.result_utils import delete_module_results, archive_module_results


PARAM_TOOLTIPS = {
    "check_optuna": "勾选后将使用 Optuna 对各基础模型进行贝叶斯优化调参；不勾选则使用默认参数。",
    "spin_trials": "Optuna 搜索的 trial 数量。越大搜索越充分但耗时越久；建议 20~100。",
    "spin_cv": "交叉验证折数。用于模型内部 CV 评估与 Optuna 目标函数，建议 3~10。",
    "data_source": "选择使用预处理后的完整数据，还是特征选择后的共识特征数据。",
    "model_list": "选择需要训练的基础模型；可按住 Ctrl 多选。",
}


class ModelTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>模型训练</h2>"))

        # 数据源选择
        self.data_source = DataSourceSelector()
        self.data_source.setToolTip(PARAM_TOOLTIPS["data_source"])
        self.layout.addWidget(self.data_source)

        # 模型选择
        self.model_list = CheckableListWidget(title="选择模型")
        self.model_list.setToolTip(PARAM_TOOLTIPS["model_list"])
        self.model_list.add_items(get_model_names())

        # 参数
        param_layout = QHBoxLayout()
        self.check_optuna = QCheckBox("使用 Optuna 调参")
        self.check_optuna.setChecked(False)
        self.check_optuna.setToolTip(PARAM_TOOLTIPS["check_optuna"])
        param_layout.addWidget(self.check_optuna)

        param_layout.addWidget(QLabel("Optuna trials:"))
        self.spin_trials = QSpinBox()
        self.spin_trials.setToolTip(PARAM_TOOLTIPS["spin_trials"])
        self.spin_trials.setRange(5, 500)
        self.spin_trials.setValue(20)
        param_layout.addWidget(self.spin_trials)

        param_layout.addWidget(QLabel("CV folds:"))
        self.spin_cv = QSpinBox()
        self.spin_cv.setToolTip(PARAM_TOOLTIPS["spin_cv"])
        self.spin_cv.setRange(2, 10)
        self.spin_cv.setValue(5)
        param_layout.addWidget(self.spin_cv)

        param_layout.addWidget(QLabel("并行模型数:"))
        self.spin_parallel = QSpinBox()
        self.spin_parallel.setRange(1, 4)
        self.spin_parallel.setValue(2)
        self.spin_parallel.setToolTip(
            "同时训练几个模型（进程级并行）。\n"
            "1=串行逐个训练；2~4=并发训练，每个模型分配总核数的一部分。\n"
            "内存占用随并发数增加（每个进程持有一份数据副本）。"
        )
        param_layout.addWidget(self.spin_parallel)

        self.check_skip_existing = QCheckBox("增量训练（跳过已有结果的模型）")
        self.check_skip_existing.setChecked(True)
        self.check_skip_existing.setToolTip(
            "勾选后：已在 leaderboard 中有结果的模型会被跳过，只训练缺失的模型；\n"
            "新结果会与历史 leaderboard 合并，不会覆盖丢失。\n"
            "取消勾选则重训所有选中模型（结果仍会合并保存）。"
        )
        param_layout.addWidget(self.check_skip_existing)

        self.check_class_weight = QCheckBox("类别加权训练（缓解不平衡）")
        self.check_class_weight.setChecked(False)
        self.check_class_weight.setToolTip(
            "勾选后：给支持类别权重的模型设置 balanced 权重（XGBoost 用 scale_pos_weight），\n"
            "让模型更关注少数类（正类），缓解不平衡导致的敏感度=0。\n"
            "不平衡严重时建议勾选；平衡数据可不勾。"
        )
        param_layout.addWidget(self.check_class_weight)

        self.check_gpu = QCheckBox("使用 GPU 加速（XGBoost/LightGBM）")
        self.check_gpu.setChecked(True)
        self.check_gpu.setToolTip(
            "勾选后：XGBoost/LightGBM 优先使用 NVIDIA GPU 训练，\n"
            "首次运行会用迷你数据实测 GPU 是否可用，不可用自动回退 CPU。\n"
            "CatBoost 实测 GPU 更慢，固定走 CPU；其余模型没有 GPU 实现。"
        )
        param_layout.addWidget(self.check_gpu)
        param_layout.addStretch()

        # 运行按钮
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("清空选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        self.btn_run = QPushButton("运行模型训练")
        self.btn_run.clicked.connect(self.run_models)
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

        # 信息标签
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: gray;")

        # 结果表
        self.result_table = QTableWidget()

        # 上下可拖拽调节：模型选择区 / 参数与按钮区 / 结果表
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(param_layout)
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
        """收集当前模型训练配置。"""
        return {
            "models": self.model_list.get_checked(),
            "use_optuna": self.check_optuna.isChecked(),
            "n_trials": self.spin_trials.value(),
            "cv_folds": self.spin_cv.value(),
            "use_feature_selected": self.data_source.use_feature_selected(),
            "skip_existing": self.check_skip_existing.isChecked(),
            "class_weight": self.check_class_weight.isChecked(),
            "use_gpu": self.check_gpu.isChecked(),
            "max_parallel": self.spin_parallel.value(),
        }

    def load_settings(self, exp: dict):
        """从实验配置恢复模型训练设置。"""
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")
        if not modeling_config:
            return
        if "models" in modeling_config:
            self.model_list.set_checked(modeling_config["models"])
        if "use_optuna" in modeling_config:
            self.check_optuna.setChecked(modeling_config["use_optuna"])
        if "n_trials" in modeling_config:
            self.spin_trials.setValue(int(modeling_config["n_trials"]))
        if "cv_folds" in modeling_config:
            self.spin_cv.setValue(int(modeling_config["cv_folds"]))
        if "use_feature_selected" in modeling_config:
            fs_available = bool(config.get("feature_selection", {}).get("consensus_table_path"))
            self.data_source.set_feature_selected_available(fs_available)
            self.data_source.set_use_feature_selected(bool(modeling_config["use_feature_selected"]))
        if "skip_existing" in modeling_config:
            self.check_skip_existing.setChecked(bool(modeling_config["skip_existing"]))
        if "class_weight" in modeling_config:
            self.check_class_weight.setChecked(bool(modeling_config["class_weight"]))
        if "use_gpu" in modeling_config:
            self.check_gpu.setChecked(bool(modeling_config["use_gpu"]))
        if "max_parallel" in modeling_config:
            self.spin_parallel.setValue(int(modeling_config["max_parallel"]))

    def run_models(self, silent: bool = False):
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

        try:
            X, y, info = load_modeling_data(
                exp,
                use_feature_selected=self.data_source.use_feature_selected(),
                use_minimal_subset=self.data_source.use_minimal_subset(),
            )
        except Exception as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        selected = self.model_list.get_checked()
        if not selected:
            QMessageBox.warning(self, "警告", "请至少选择一个模型")
            return

        try:
            output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "models"
            output_dir.mkdir(parents=True, exist_ok=True)
            leaderboard_path = output_dir / "model_leaderboard.csv"

            # 增量训练：跳过 leaderboard 中已有结果的模型
            if self.check_skip_existing.isChecked() and leaderboard_path.exists():
                try:
                    existing = pd.read_csv(leaderboard_path)
                    existing_models = set(existing["model"].astype(str))
                    skipped = [m for m in selected if m in existing_models]
                    selected = [m for m in selected if m not in existing_models]
                    if skipped:
                        self.parent_window.log_message(f"增量训练：跳过已有结果的 {len(skipped)} 个模型")
                except Exception as e:
                    self.parent_window.logger.warning(f"读取已有 leaderboard 失败：{e}")

            if not selected:
                if leaderboard_path.exists():
                    leaderboard = pd.read_csv(leaderboard_path)
                    self.lbl_status.setText(f"所有选中模型均已有结果：{len(leaderboard)} 个")
                    self.update_table(leaderboard)
                    self.parent_window.log_message("所有选中模型均已有结果，无需重训")
                    if not silent:
                        QMessageBox.information(self, "提示", "所有选中模型均已有结果，无需重训")
                return

            # 上游变动（全量重跑时）：归档依赖基础模型的下游结果
            # 增量训练（勾选且已有 leaderboard）不归档，仅追加模型
            is_full_rerun = not (self.check_skip_existing.isChecked() and leaderboard_path.exists())
            if is_full_rerun:
                from gui.widgets.result_utils import archive_downstream_results
                archived = archive_downstream_results(
                    self.parent_window.db, self.parent_window.current_experiment_id,
                    trigger="modeling", logger=self.parent_window.logger,
                )
                if archived:
                    self.parent_window.log_message(f"模型全量重跑：已归档上一轮下游结果 {archived}")

            self.lbl_info.setText(
                f"数据：{info['data_source']} | 样本数：{info['n_samples']} | 特征数：{info['n_features']} | 输出：{output_dir}"
            )
            self.parent_window.log_message(
                f"模型训练开始：{info['data_source']}，{len(selected)} 个模型，输出 {output_dir}"
            )

            trainer = ModelTrainer(
                X, y,
                cv_folds=self.spin_cv.value(),
                output_dir=str(output_dir),
                n_jobs=-1,
                class_weight=self.check_class_weight.isChecked(),
                use_gpu=self.check_gpu.isChecked(),
            )
            new_leaderboard = trainer.train_multiple(
                names=selected,
                use_optuna=self.check_optuna.isChecked(),
                n_trials=self.spin_trials.value(),
                max_parallel=self.spin_parallel.value(),
            )

            # 与已有 leaderboard 合并：新结果覆盖同名模型，其余保留
            if leaderboard_path.exists():
                try:
                    old = pd.read_csv(leaderboard_path)
                    new_names = set(new_leaderboard["model"].astype(str))
                    old = old[~old["model"].astype(str).isin(new_names)]
                    leaderboard = pd.concat([old, new_leaderboard], ignore_index=True)
                    if "auc" in leaderboard.columns:
                        leaderboard = leaderboard.sort_values("auc", ascending=False).reset_index(drop=True)
                except Exception as e:
                    self.parent_window.logger.warning(f"合并 leaderboard 失败：{e}")
                    leaderboard = new_leaderboard
            else:
                leaderboard = new_leaderboard

            leaderboard.to_csv(leaderboard_path, index=False)

            config = exp.get("config") or {}
            config["modeling"] = {
                **self.get_settings(),
                "output_dir": str(output_dir),
                "leaderboard_path": str(leaderboard_path),
                "data_source": info["data_source"],
                "processed_path": info["processed_path"],
            }
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id,
                {"config": config, "status": "modeled"}
            )

            self.lbl_status.setText(f"模型训练完成：本次 {len(new_leaderboard)} 个，累计 {len(leaderboard)} 个模型")
            self.parent_window.log_message(f"模型训练完成：本次 {len(new_leaderboard)} 个，累计 {len(leaderboard)} 个模型")
            self.update_table(leaderboard)
            if not silent:
                QMessageBox.information(self, "成功", "模型训练完成")
            # 上游变动后：刷新所有下游选项卡
            self.parent_window._refresh_downstream_views()

        except Exception as e:
            self.parent_window.logger.exception("模型训练失败")
            if not silent:
                QMessageBox.critical(self, "失败", str(e))

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
        """返回当前实验的模型结果目录。"""
        if self.parent_window.current_experiment_id is None:
            return None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")
        if not modeling_config or not modeling_config.get("output_dir"):
            return None
        return Path(modeling_config["output_dir"])

    def _clear_config(self):
        """清空当前实验的模型训练配置。"""
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        if "modeling" in config:
            del config["modeling"]
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id, {"config": config}
            )

    def delete_results(self):
        """删除当前实验的模型训练结果。"""
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可删除的模型训练结果")
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
        """归档当前实验的模型训练结果。"""
        result_dir = self._get_result_dir()
        if result_dir is None:
            QMessageBox.warning(self, "警告", "当前实验没有可归档的模型训练结果")
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
        """加载历史模型训练结果到表格。"""
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")
        if not modeling_config:
            return
        # 先恢复设置
        self.load_settings(exp)
        if not modeling_config.get("leaderboard_path"):
            return
        leaderboard_path = Path(modeling_config["leaderboard_path"])
        if not leaderboard_path.exists():
            return
        try:
            leaderboard = pd.read_csv(leaderboard_path)
            self.update_table(leaderboard)
            self.lbl_status.setText(f"已加载历史结果：{len(leaderboard)} 个模型")
            self.parent_window.log_message(f"模型训练历史结果已加载：{leaderboard_path}")
        except Exception as e:
            self.parent_window.logger.warning(f"加载模型历史结果失败：{e}")
