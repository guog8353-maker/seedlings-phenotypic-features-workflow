"""Centralized error reporting: logging, stage tracking, detailed dialogs.

Rules enforced project-wide:
- no bare ``except: pass``;
- no exception swallowed into an empty dialog;
- every failure keeps the full traceback in logs/errors.log.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

DEBUG_MODE = True

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_initialized = False


def init_logging() -> Path:
    """Initialize the logging system (logs/app.log + logs/errors.log)."""
    global _initialized
    if _initialized:
        return _LOG_DIR
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    app_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(fmt)

    err_handler = logging.FileHandler(_LOG_DIR / "errors.log", encoding="utf-8")
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(fmt)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    console.setFormatter(fmt)

    root.addHandler(app_handler)
    root.addHandler(err_handler)
    root.addHandler(console)

    _initialized = True
    logging.info("日志系统初始化: %s", _LOG_DIR)
    return _LOG_DIR


def log_dir() -> Path:
    return _LOG_DIR


class StageTracker:
    """Track the current processing stage so failures are locatable."""

    _current = "INIT"

    @classmethod
    def set(cls, stage: str) -> None:
        cls._current = stage
        print(f"[STAGE] {stage}", flush=True)
        logging.debug("[STAGE] %s", stage)

    @classmethod
    def get(cls) -> str:
        return cls._current

    @classmethod
    def fail(cls, stage: Optional[str] = None) -> str:
        s = stage or cls._current
        print(f"[FAILED STAGE] {s}", flush=True)
        logging.error("[FAILED STAGE] %s", s)
        return s


def log_exception(stage: str, exc: BaseException, context: str = "") -> str:
    """Print + log the full traceback; returns the traceback text."""
    tb = traceback.format_exc()
    print("========== ERROR ==========", flush=True)
    print(tb, flush=True)
    print("===========================", flush=True)
    logging.error(
        "STAGE=%s | TYPE=%s | %s\n%s", stage, type(exc).__name__, context, tb
    )
    return tb


def build_error_text(
    stage: str,
    exc: BaseException,
    image_path: str = "",
    image_info: str = "",
    tb: str = "",
) -> str:
    """Build the full human-readable error text (stage/type/message/inputs)."""
    lines = [
        f"阶段: {stage}",
        f"错误类型: {type(exc).__name__}",
        f"错误信息: {exc}",
    ]
    if image_path:
        lines.append(f"输入图片: {image_path}")
    if image_info:
        lines.append(f"图片信息: {image_info}")
    lines.append(f"详细日志: {_LOG_DIR / 'errors.log'}")
    if tb:
        lines.append("")
        lines.append("TRACEBACK:")
        lines.append(tb)
    return "\n".join(lines)


def show_error_dialog(
    parent,
    title: str,
    stage: str,
    exc: BaseException,
    image_path: str = "",
    image_info: str = "",
    tb: str = "",
) -> None:
    """Show a detailed error dialog with a '复制错误信息' button.

    Never an empty OK-only box: body always carries stage / error type /
    error message / log path; the full traceback is one click away.
    """
    from PySide6.QtWidgets import QApplication, QMessageBox

    if not tb:
        tb = traceback.format_exc()
    short_text = build_error_text(stage, exc, image_path, image_info)
    full_text = build_error_text(stage, exc, image_path, image_info, tb)

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(short_text)
    copy_btn = box.addButton("复制错误信息", QMessageBox.ActionRole)
    box.addButton(QMessageBox.Ok)
    box.exec()
    if box.clickedButton() is copy_btn:
        QApplication.clipboard().setText(full_text)
