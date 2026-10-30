"""日志系统：基于 loguru，支持 GUI 实时日志窗口与文件双输出。"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = None, level: str = "INFO"):
    """
    配置 loguru 日志。
    - 移除默认 sink
    - 添加 stdout 输出
    - 添加文件输出（按大小轮转）
    """
    logger.remove()

    # stdout
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "automl_{time:YYYYMMDD}.log",
            level=level,
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    return logger


class GUILogHandler:
    """桥接 loguru 与 PyQt 文本框：将日志实时输出到 GUI。"""

    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, message: str):
        if message.strip():
            self.text_edit.append(message.strip())

    def flush(self):
        pass
