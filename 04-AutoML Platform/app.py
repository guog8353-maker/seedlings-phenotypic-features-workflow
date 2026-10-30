"""AutoML-Platform 桌面应用入口。"""

import os
import sys
from pathlib import Path

# 确定性算法模式（部分第三方库会开启）下 CuBLAS 运算必须设置工作区配置，
# 否则 RealTabR/XRFM/TabPFN/TabICL 等模型在 GPU 上直接报错中断
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# 确保项目根目录在 Python 路径中，并设为工作目录（使 config 中相对路径可靠）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AutoML-Platform")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
