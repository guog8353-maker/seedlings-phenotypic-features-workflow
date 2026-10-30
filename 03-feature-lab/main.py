"""Entry point for the image feature analysis tool."""

import sys

from PySide6.QtWidgets import QApplication

from src.app import MainWindow
from src.error_reporting import init_logging


def main() -> int:
    init_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("水稻吐水特征可视化分析平台")
    app.setApplicationVersion("2.0.0")

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
