"""报告标签页：自定义章节，生成 Markdown 与 Word 报告。"""

import sys
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QFileDialog, QComboBox, QCheckBox, QGroupBox,
    QGridLayout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.report_generator import ReportGenerator


# (key, 中文标签)
SECTION_ITEMS = [
    ("overview", "实验概述"),
    ("preprocessing", "预处理"),
    ("feature_selection", "特征选择"),
    ("base_models", "基础模型"),
    ("ensemble", "集成学习"),
    ("transformer", "Transformer"),
    ("automl", "AutoML"),
    ("survival", "生存分析"),
    ("shap", "SHAP 分析"),
    ("ablation", "消融分析"),
    ("minimal_subset", "最小子集"),
    ("conclusion", "结论与建议"),
]


class ReportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.layout = QVBoxLayout(self)

        self.layout.addWidget(QLabel("<h2>报告</h2>"))
        self.lbl_info = QLabel("自定义要包含的内容，生成实验报告（Markdown / Word），仅含文字与表格。")
        self.layout.addWidget(self.lbl_info)

        # ---- SHAP 关注指标（下拉） ----
        shap_layout = QHBoxLayout()
        shap_layout.addWidget(QLabel("SHAP 关注指标（可选）："))
        self.combo_shap_indicator = QComboBox()
        self.combo_shap_indicator.setMinimumWidth(220)
        shap_layout.addWidget(self.combo_shap_indicator)
        self.btn_refresh_indicator = QPushButton("刷新指标")
        self.btn_refresh_indicator.clicked.connect(self.refresh_indicator_options)
        shap_layout.addWidget(self.btn_refresh_indicator)
        shap_layout.addStretch()
        self.layout.addLayout(shap_layout)

        # ---- 章节选择 ----
        section_group = QGroupBox("选择要包含到报告中的内容")
        grid = QGridLayout(section_group)
        self.section_checks = {}
        for idx, (key, label) in enumerate(SECTION_ITEMS):
            chk = QCheckBox(label)
            chk.setChecked(True)
            self.section_checks[key] = chk
            grid.addWidget(chk, idx // 4, idx % 4)
        btn_row = QHBoxLayout()
        self.btn_all = QPushButton("全选")
        self.btn_all.clicked.connect(lambda: self._set_all_sections(True))
        self.btn_none = QPushButton("全不选")
        self.btn_none.clicked.connect(lambda: self._set_all_sections(False))
        self.btn_default = QPushButton("默认（全部）")
        self.btn_default.clicked.connect(lambda: self._set_all_sections(True))
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        btn_row.addStretch()
        grid.addLayout(btn_row, (len(SECTION_ITEMS) // 4) + 1, 0, 1, 4)
        self.layout.addWidget(section_group)

        # ---- 生成按钮 ----
        btn_layout = QHBoxLayout()
        self.btn_md = QPushButton("生成 Markdown 报告")
        self.btn_md.clicked.connect(self.generate_md)
        self.btn_word = QPushButton("生成 Word 报告")
        self.btn_word.clicked.connect(self.generate_word)
        btn_layout.addWidget(self.btn_md)
        btn_layout.addWidget(self.btn_word)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        self.lbl_status = QLabel("未生成")
        self.layout.addWidget(self.lbl_status)
        self.layout.addStretch()

    def _set_all_sections(self, checked: bool):
        for chk in self.section_checks.values():
            chk.setChecked(checked)

    def showEvent(self, event):
        """进入选项卡时刷新 SHAP 关注指标下拉框（跟随当前数据）。"""
        super().showEvent(event)
        try:
            self.refresh_indicator_options()
        except Exception:
            pass

    def refresh_indicator_options(self):
        """从共识特征表加载 SHAP 关注指标下拉选项。"""
        self.combo_shap_indicator.clear()
        self.combo_shap_indicator.addItem("（无）", None)
        if self.parent_window.current_experiment_id is None:
            return
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        fs_config = config.get("feature_selection") or {}
        consensus_path = fs_config.get("consensus_table_path")
        feats = []
        if consensus_path and Path(consensus_path).exists():
            try:
                table = pd.read_csv(consensus_path)
                if "consensus" in table.columns:
                    feats = table.loc[table["consensus"] == 1, "feature"].astype(str).tolist()
                else:
                    feats = table["feature"].astype(str).tolist()
            except Exception as e:
                self.parent_window.logger.warning(f"加载共识特征失败：{e}")
        for f in feats:
            self.combo_shap_indicator.addItem(f, f)
        # 预选已保存的指标
        saved = (config.get("report") or {}).get("shap_indicator")
        if saved:
            idx = self.combo_shap_indicator.findText(saved)
            if idx >= 0:
                self.combo_shap_indicator.setCurrentIndex(idx)
        else:
            # 尝试从 shap_meta 读取上次使用的指标
            try:
                shap_dir = Path((config.get("modeling") or {}).get("output_dir", "")).parent / "shap"
                meta_path = shap_dir / "shap_meta.json"
                if meta_path.exists():
                    import json
                    ind = json.loads(meta_path.read_text(encoding="utf-8")).get("indicator")
                    if ind:
                        idx = self.combo_shap_indicator.findText(ind)
                        if idx >= 0:
                            self.combo_shap_indicator.setCurrentIndex(idx)
            except Exception:
                pass

    def _get_selected_sections(self) -> dict:
        return {k: chk.isChecked() for k, chk in self.section_checks.items()}

    def get_settings(self) -> dict:
        return {
            "sections": self._get_selected_sections(),
            "shap_indicator": self.combo_shap_indicator.currentData(),
        }

    def load_settings(self, exp: dict):
        config = exp.get("config") or {}
        cfg = config.get("report")
        if not cfg:
            return
        sections = cfg.get("sections") or {}
        for k, chk in self.section_checks.items():
            chk.setChecked(bool(sections.get(k, True)))
        saved_ind = cfg.get("shap_indicator")
        if saved_ind:
            idx = self.combo_shap_indicator.findText(saved_ind)
            if idx >= 0:
                self.combo_shap_indicator.setCurrentIndex(idx)

    def _collect_leaderboard(self):
        if self.parent_window.current_experiment_id is None:
            return None
        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        config = exp.get("config") or {}
        modeling_config = config.get("modeling")
        records = []
        if modeling_config and modeling_config.get("output_dir"):
            model_dir = Path(modeling_config["output_dir"])
            for f in sorted(model_dir.glob("*_oof.csv")):
                model_name = f.stem.replace("_oof", "")
                df = pd.read_csv(f)
                from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
                y_true = df["y_true"].values
                proba = df["oof_proba"].values
                pred = df["oof_pred"].values
                records.append({
                    "model": model_name,
                    "auc": round(roc_auc_score(y_true, proba), 4),
                    "accuracy": round(accuracy_score(y_true, pred), 4),
                    "f1": round(f1_score(y_true, pred, zero_division=0), 4),
                    "brier": round(brier_score_loss(y_true, proba), 4),
                })
        if not records:
            return None
        return pd.DataFrame(records).sort_values("auc", ascending=False)

    def generate_md(self):
        path = self._generate("md", silent=False)
        if path:
            self.lbl_status.setText(f"Markdown 报告已生成：{path}")

    def generate_word(self):
        path = self._generate("word", silent=False)
        if path:
            self.lbl_status.setText(f"Word 报告已生成：{path}")

    def _generate(self, fmt: str, silent: bool = False):
        if self.parent_window.current_experiment_id is None:
            QMessageBox.warning(self, "警告", "请先创建或加载实验")
            return None

        exp = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
        leaderboard = self._collect_leaderboard()
        sections = self._get_selected_sections()
        # 若未选任何模块且没有基础模型结果，提示
        if leaderboard is None and not any(sections.values()):
            QMessageBox.warning(self, "警告", "暂无可用结果，且未选择任何报告内容")
            return None

        try:
            self.lbl_status.setText("正在整合报告内容...")
            output_dir = Path(exp.get("output_dir") or self.parent_window.output_dir) / f"exp_{exp['id']}" / "reports"
            gen = ReportGenerator(exp["name"], str(output_dir))
            gen.build_from_results(
                config=exp.get("config", {}),
                leaderboard=leaderboard,
                shap_indicator=self.combo_shap_indicator.currentData(),
                sections=sections,
            )
            if fmt == "md":
                path = gen.save_markdown()
            else:
                path = gen.save_word()

            # 持久化设置
            exp2 = self.parent_window.db.get_experiment(self.parent_window.current_experiment_id)
            config = exp2.get("config") or {}
            config["report"] = self.get_settings()
            self.parent_window.db.update_experiment(self.parent_window.current_experiment_id, {"config": config})

            if not silent:
                QMessageBox.information(self, "成功", f"报告已生成：\n{path}")
            return path
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "失败", str(e))
            self.parent_window.logger.exception("报告生成失败")
            return None
