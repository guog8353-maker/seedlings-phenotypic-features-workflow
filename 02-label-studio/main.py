"""Entry point for the rice seedling tray annotation tool."""

import sys

from PySide6.QtWidgets import QApplication

from src.app import MainWindow
from src.error_reporting import init_logging


def main() -> int:
    init_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("秧盘图像分块与标注工具")
    app.setApplicationVersion("1.1.0")

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
