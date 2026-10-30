"""Main application window."""

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from .batch_tab import BatchTab
from .single_video_tab import SingleVideoTab


class MainWindow(QMainWindow):
    """Main window of the video stitching application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频抽帧与拼接工具")
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.single_tab = SingleVideoTab()
        self.batch_tab = BatchTab(get_params_callback=self.single_tab.get_current_params)

        self.tabs.addTab(self.single_tab, "单视频处理")
        self.tabs.addTab(self.batch_tab, "批量处理")
        layout.addWidget(self.tabs)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        self.statusBar().showMessage("就绪")

    def _show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "<h2>视频抽帧与拼接工具</h2>"
            "<p>用于从稳定单向移动的视频中抽取帧并拼接成全景图。</p>"
            "<p>支持均匀抽帧、关键帧抽取、顺序拼接、条带拼接和批量处理。</p>",
        )
