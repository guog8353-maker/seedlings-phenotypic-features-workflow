"""工作流选项卡：按用户配置顺序依次运行多个模块。"""

import sys
import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QTextEdit, QMessageBox, QProgressBar,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class WorkflowTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>工作流</h2>"))

        intro = QLabel("在此配置需要依次运行的模块顺序。各模块的超参数请在对应选项卡中设置，"
                       "本选项卡仅负责按顺序调用。结果仍保存到各模块默认路径。"
                       "注意：由于各模块当前为同步运行，工作流执行期间界面会暂时无响应，请等待完成。")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        self.layout.addWidget(intro)

        # 模块选择
        group = QGroupBox("选择要运行的模块（按从上到下的顺序执行）")
        g_layout = QVBoxLayout(group)

        self.chk_preprocess = QCheckBox("1. 预处理")
        self.chk_feature = QCheckBox("2. 特征选择")
        self.chk_model = QCheckBox("3. 基础模型训练")
        self.chk_ensemble = QCheckBox("4. 集成学习")
        self.chk_transformer = QCheckBox("5. Transformer")
        self.chk_automl = QCheckBox("6. AutoML")
        self.chk_survival = QCheckBox("7. 生存分析")
        self.chk_shap = QCheckBox("8. 全模型 SHAP 分析")
        self.chk_ablation = QCheckBox("9. 消融分析")
        self.chk_minimal_subset = QCheckBox("10. 最小子集分析")
        self.chk_viz = QCheckBox("11. 可视化")
        self.chk_report = QCheckBox("12. Word 报告")

        for chk in [
            self.chk_preprocess, self.chk_feature, self.chk_model,
            self.chk_ensemble, self.chk_transformer, self.chk_automl,
            self.chk_survival, self.chk_shap, self.chk_ablation,
            self.chk_minimal_subset, self.chk_viz, self.chk_report,
        ]:
            g_layout.addWidget(chk)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("运行工作流")
        self.btn_run.clicked.connect(self.run_workflow)
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._clear)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()
        g_layout.addLayout(btn_layout)
        self.layout.addWidget(group)

        # 进度与日志
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("工作流运行日志将显示在这里...")
        self.layout.addWidget(self.log)

    def _select_all(self):
        for chk in self.findChildren(QCheckBox):
            chk.setChecked(True)

    def _clear(self):
        for chk in self.findChildren(QCheckBox):
            chk.setChecked(False)

    def get_settings(self) -> dict:
        """收集当前工作流配置。"""
        return {
            "preprocess": self.chk_preprocess.isChecked(),
            "feature": self.chk_feature.isChecked(),
            "model": self.chk_model.isChecked(),
            "ensemble": self.chk_ensemble.isChecked(),
            "transformer": self.chk_transformer.isChecked(),
            "automl": self.chk_automl.isChecked(),
            "survival": self.chk_survival.isChecked(),
            "shap": self.chk_shap.isChecked(),
            "ablation": self.chk_ablation.isChecked(),
            "minimal_subset": self.chk_minimal_subset.isChecked(),
            "viz": self.chk_viz.isChecked(),
            "report": self.chk_report.isChecked(),
        }

    def load_settings(self, exp: dict):
        """从实验配置恢复工作流设置。"""
        config = exp.get("config") or {}
        workflow_config = config.get("workflow")
        if not workflow_config:
            return
        self.chk_preprocess.setChecked(bool(workflow_config.get("preprocess", False)))
        self.chk_feature.setChecked(bool(workflow_config.get("feature", False)))
        self.chk_model.setChecked(bool(workflow_config.get("model", False)))
        self.chk_ensemble.setChecked(bool(workflow_config.get("ensemble", False)))
        self.chk_transformer.setChecked(bool(workflow_config.get("transformer", False)))
        self.chk_automl.setChecked(bool(workflow_config.get("automl", False)))
        self.chk_survival.setChecked(bool(workflow_config.get("survival", False)))
        self.chk_shap.setChecked(bool(workflow_config.get("shap", False)))
        self.chk_ablation.setChecked(bool(workflow_config.get("ablation", False)))
        self.chk_minimal_subset.setChecked(bool(workflow_config.get("minimal_subset", False)))
        self.chk_viz.setChecked(bool(workflow_config.get("viz", False)))
        self.chk_report.setChecked(bool(workflow_config.get("report", False)))

    def _log(self, msg: str):
        self.log.append(msg)
        self.parent_window.log_message(msg)
        # 刷新事件循环，使日志实时显示
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

    def run_workflow(self):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return

        mw = self.parent_window
        steps = []

        if self.chk_preprocess.isChecked():
            steps.append(("预处理", lambda: mw.preprocess_tab.run_preprocessing(silent=True)))
        if self.chk_feature.isChecked():
            steps.append(("特征选择", lambda: mw.feature_tab.run_feature_selection(silent=True)))
        if self.chk_model.isChecked():
            steps.append(("基础模型训练", lambda: mw.model_tab.run_models(silent=True)))
        if self.chk_ensemble.isChecked():
            steps.append(("集成学习", lambda: mw.ensemble_tab.run_ensemble(silent=True)))
        if self.chk_transformer.isChecked():
            steps.append(("Transformer", lambda: mw.transformer_tab.run_transformer(silent=True)))
        if self.chk_automl.isChecked():
            steps.append(("AutoML", lambda: mw.automl_tab.run_automl(silent=True)))
        if self.chk_survival.isChecked():
            steps.append(("生存分析", lambda: mw.survival_tab.run_survival(silent=True)))
        if self.chk_shap.isChecked():
            steps.append(("全模型 SHAP 分析", lambda: mw.result_tab.run_shap_analysis(silent=True)))
        if self.chk_ablation.isChecked():
            steps.append(("消融分析", lambda: mw.ablation_tab.run_ablation_sync(silent=True)))
        if self.chk_minimal_subset.isChecked():
            steps.append(("最小子集分析", lambda: mw.minimal_subset_tab.run_analysis_sync(silent=True)))
        if self.chk_viz.isChecked():
            steps.append(("可视化", lambda: mw.visualization_tab.generate_figures(silent=True)))
        if self.chk_report.isChecked():
            steps.append(("Word 报告", lambda: mw.report_tab._generate("docx", silent=True)))

        if not steps:
            QMessageBox.warning(self, "警告", "请至少选择一个模块")
            return

        # 持久化工作流配置
        try:
            exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
            config = exp.get("config") or {}
            config["workflow"] = self.get_settings()
            self.parent_window.db.update_experiment(
                self.parent_window.current_experiment_id,
                {"config": config}
            )
        except Exception as e:
            self.parent_window.logger.warning(f"工作流配置持久化失败：{e}")

        # 依赖检查：给出警告但不阻止
        warnings = []
        checked_names = {s[0] for s in steps}
        if "基础模型训练" in checked_names and "特征选择" not in checked_names and "预处理" not in checked_names:
            warnings.append("运行基础模型训练前建议先完成预处理与特征选择。")
        if "集成学习" in checked_names and "基础模型训练" not in checked_names:
            warnings.append("集成学习依赖基础模型训练结果。")
        if "Transformer" in checked_names and "特征选择" not in checked_names:
            warnings.append("Transformer 依赖特征选择结果。")
        if "AutoML" in checked_names and "特征选择" not in checked_names:
            warnings.append("AutoML 依赖特征选择结果。")
        if "全模型 SHAP 分析" in checked_names and "基础模型训练" not in checked_names:
            warnings.append("全模型 SHAP 分析依赖基础模型训练结果。")
        if "消融分析" in checked_names and "基础模型训练" not in checked_names:
            warnings.append("消融分析依赖基础模型训练结果（需先有模型与共识特征）。")
        if "最小子集分析" in checked_names and "全模型 SHAP 分析" not in checked_names:
            warnings.append("最小子集分析建议先做全模型 SHAP 分析（需 SHAP 贡献排名）。")
        if "可视化" in checked_names and "基础模型训练" not in checked_names:
            warnings.append("可视化依赖基础模型训练结果。")
        if "Word 报告" in checked_names and "基础模型训练" not in checked_names:
            warnings.append("Word 报告依赖基础模型训练结果。")

        if warnings:
            self._log("依赖检查警告：")
            for w in warnings:
                self._log(f"  • {w}")
            self._log("将继续执行工作流，依赖不足的模块可能会失败。")

        self.log.clear()
        self._log(f"工作流开始：共 {len(steps)} 个模块")
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)

        try:
            for i, (name, runner) in enumerate(steps, 1):
                self._log(f"[开始] {name}")
                try:
                    runner()
                    self._log(f"[完成] {name}")
                except Exception as e:
                    detail = traceback.format_exc()
                    self._log(f"[错误] {name}:\n{detail}")
                    # 遇到错误继续后续步骤
                self.progress.setValue(int(i / len(steps) * 100))
                from PyQt6.QtCore import QCoreApplication
                QCoreApplication.processEvents()

            self._log("工作流执行结束。")
            QMessageBox.information(self, "完成", "工作流已执行完毕，请查看日志了解各模块状态。")
        finally:
            self.btn_run.setEnabled(True)
            self.progress.setValue(100)
