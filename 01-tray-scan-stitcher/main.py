"""Entry point for the video frame extraction and stitching application."""

import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

APP_VERSION = "0.5.0"
BUILD_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main():
    # Print build info so the user can verify which code is actually running.
    pipeline_file = Path(project_root) / "src" / "core" / "stitching" / "pipeline.py"
    print(f"APP VERSION = {APP_VERSION}", flush=True)
    print(f"BUILD TIME = {BUILD_ID}", flush=True)
    print(f"pipeline.py = {pipeline_file}", flush=True)

    app = QApplication(sys.argv)
    app.setApplicationName("视频抽帧与拼接工具")
    app.setApplicationDisplayName("视频抽帧与拼接工具")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
