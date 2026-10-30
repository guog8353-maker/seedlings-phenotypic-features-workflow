"""数据源选择器：预处理后完整数据 vs 特征选择后共识特征数据 vs 最小子集特征数据。"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QRadioButton, QButtonGroup,
)


class DataSourceSelector(QWidget):
    """通用数据源选择控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("<b>训练数据:</b>"))
        self.group = QButtonGroup(self)

        self.radio_prep = QRadioButton("预处理后的完整数据")
        self.radio_fs = QRadioButton("特征选择后的共识特征数据")
        self.radio_fs.setChecked(True)
        self.radio_minimal = QRadioButton("最小子集特征数据")
        self.radio_minimal.setEnabled(False)
        self.radio_minimal.setToolTip("使用最小子集分析回写的特征集（需先在最小子集选项卡回写）")

        self.group.addButton(self.radio_prep, 0)
        self.group.addButton(self.radio_fs, 1)
        self.group.addButton(self.radio_minimal, 2)

        layout.addWidget(self.radio_fs)
        layout.addWidget(self.radio_prep)
        layout.addWidget(self.radio_minimal)
        layout.addStretch()

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_info)

    def use_feature_selected(self) -> bool:
        return self.radio_fs.isChecked()

    def use_minimal_subset(self) -> bool:
        return self.radio_minimal.isChecked()

    def set_feature_selected_available(self, available: bool):
        self.radio_fs.setEnabled(available)
        if not available and self.radio_fs.isChecked():
            self.radio_prep.setChecked(True)

    def set_minimal_available(self, available: bool):
        self.radio_minimal.setEnabled(available)
        if not available and self.radio_minimal.isChecked():
            self.radio_fs.setChecked(True) if self.radio_fs.isEnabled() else self.radio_prep.setChecked(True)

    def set_use_feature_selected(self, use_fs: bool):
        """设置数据源选择状态；若特征选择数据不可用则强制使用预处理数据。"""
        if use_fs and self.radio_fs.isEnabled():
            self.radio_fs.setChecked(True)
        else:
            self.radio_prep.setChecked(True)

    def set_info(self, text: str):
        self.lbl_info.setText(text)
