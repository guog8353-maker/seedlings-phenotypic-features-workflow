"""Batch processing tab: automatically loops the shared SmartStitcher."""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.batch_processor import BatchProcessor
from ..core.stitching.config import StitchConfig
from ..core.stitching.video_analyzer import VideoAnalyzer
from .widgets import FileSelector, show_error


class BatchWorker(QThread):
    """Worker thread for batch processing."""

    progress = Signal(int, int)
    log = Signal(str)
    file_done = Signal(dict)
    finished_signal = Signal(list)

    def __init__(self, processor: BatchProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor

    def run(self):
        results = self.processor.run(
            progress_callback=self._on_progress,
            log_callback=self._on_log,
            file_done_callback=self._on_file_done,
        )
        self.finished_signal.emit(results)

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total)

    def _on_log(self, message: str):
        self.log.emit(message)

    def _on_file_done(self, result: dict):
        self.file_done.emit(result)


class AnalyzeWorker(QThread):
    """Worker thread for analysis-only runs over the batch input folder."""

    log = Signal(str)
    finished_signal = Signal()

    def __init__(self, videos, parent=None):
        super().__init__(parent)
        self.videos = videos

    def run(self):
        analyzer = VideoAnalyzer()
        for v in self.videos:
            try:
                a = analyzer.analyze(str(v))
                reasons = "；".join(a.reasons) if a.reasons else ""
                self.log.emit(
                    f"{v.name}: 推荐 {a.recommended_keyframes} 帧 | "
                    f"方向 {a.motion_direction} | 跨度 {a.cumulative_span_px:.0f}px | "
                    f"模糊 {a.blur_ratio:.0%} | 难度 {a.stitchability}"
                    + (f" | {reasons}" if reasons else "")
                )
            except Exception as e:
                self.log.emit(f"{v.name}: 分析失败 {e}")
        self.finished_signal.emit()


class BatchTab(QWidget):
    """Tab for batch processing multiple videos."""

    def __init__(self, get_params_callback=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[BatchWorker] = None
        self._analyze_worker: Optional[AnalyzeWorker] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.input_selector = FileSelector("输入文件夹", mode="folder")
        self.output_selector = FileSelector("输出文件夹", mode="folder")
        layout.addWidget(self.input_selector)
        layout.addWidget(self.output_selector)

        settings_group = QGroupBox("批量处理设置（每个视频独立分析、独立拼接、独立报告）")
        settings_layout = QVBoxLayout(settings_group)

        kf_layout = QHBoxLayout()
        kf_layout.addWidget(QLabel("关键帧模式:"))
        self.kf_mode_combo = QComboBox()
        self.kf_mode_combo.addItem("自动推荐（按运动/重叠/覆盖逐视频计算）", True)
        self.kf_mode_combo.addItem("固定数量", False)
        self.kf_mode_combo.setCurrentIndex(0)
        self.kf_mode_combo.currentIndexChanged.connect(self._on_kf_mode_changed)
        kf_layout.addWidget(self.kf_mode_combo)
        kf_layout.addWidget(QLabel("固定值:"))
        self.kf_fixed_spin = QSpinBox()
        self.kf_fixed_spin.setRange(2, 30)
        self.kf_fixed_spin.setValue(12)
        self.kf_fixed_spin.setEnabled(False)
        kf_layout.addWidget(self.kf_fixed_spin)
        settings_layout.addLayout(kf_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("输出画质:"))
        self.quality_mode_combo = QComboBox()
        self.quality_mode_combo.addItem("自然（推荐，无任何后处理）", "natural")
        self.quality_mode_combo.addItem("标准（完成后一次极轻去噪）", "standard")
        self.quality_mode_combo.addItem("增强（去噪 + 极轻锐化）", "enhanced")
        self.quality_mode_combo.setCurrentIndex(0)
        quality_layout.addWidget(self.quality_mode_combo)
        settings_layout.addLayout(quality_layout)

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("输出格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("JPEG", "jpg")
        self.format_combo.addItem("TIFF", "tif")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_layout.addWidget(self.format_combo)

        format_layout.addWidget(QLabel("JPEG 质量:"))
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(80, 100)
        self.jpeg_quality_spin.setValue(95)
        self.jpeg_quality_spin.setEnabled(False)
        format_layout.addWidget(self.jpeg_quality_spin)
        settings_layout.addLayout(format_layout)

        settings_layout.addWidget(
            QLabel("提示：批量 = 自动循环调用与单体完全相同的智能拼接核心；"
                   "质量不足会自动重试（最多 3 次），单视频失败不影响整批。")
        )
        layout.addWidget(settings_group)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始批量处理")
        self.start_btn.clicked.connect(self._start_batch)
        button_layout.addWidget(self.start_btn)

        self.analyze_btn = QPushButton("仅分析不拼接")
        self.analyze_btn.clicked.connect(self._analyze_only)
        button_layout.addWidget(self.analyze_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._stop_batch)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("处理日志将显示在这里...")
        layout.addWidget(self.log_text)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

    def _log(self, message: str):
        self.log_text.append(message)

    def _build_config(self) -> StitchConfig:
        auto = self.kf_mode_combo.currentData()
        return StitchConfig(
            auto_keyframes=auto,
            target_keyframes=self.kf_fixed_spin.value(),
            quality_mode=self.quality_mode_combo.currentData(),
        )

    @Slot()
    def _on_kf_mode_changed(self):
        self.kf_fixed_spin.setEnabled(not self.kf_mode_combo.currentData())

    @Slot()
    def _on_format_changed(self):
        self.jpeg_quality_spin.setEnabled(self.format_combo.currentData() == "jpg")

    def _get_videos(self):
        input_dir = self.input_selector.get_path()
        if not input_dir:
            show_error(self, "错误", "请选择输入文件夹")
            return None
        input_path = Path(input_dir)
        if not input_path.exists():
            show_error(self, "错误", "输入文件夹不存在")
            return None
        videos = sorted(input_path.glob("*.mp4"))
        if not videos:
            show_error(self, "错误", "输入文件夹中没有 .mp4 文件")
            return None
        return videos

    @Slot()
    def _start_batch(self):
        videos = self._get_videos()
        if not videos:
            return
        output_dir = self.output_selector.get_path()
        if not output_dir:
            show_error(self, "错误", "请选择输出文件夹")
            return

        input_dir = self.input_selector.get_path()
        processor = BatchProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            config=self._build_config(),
            output_format=self.format_combo.currentData(),
            jpeg_quality=self.jpeg_quality_spin.value(),
        )

        self.log_text.clear()
        self.summary_label.setText("")
        self.progress_bar.setValue(0)
        self._log(f"开始批量处理，共 {len(videos)} 个视频（共享单体智能拼接核心）")

        self._worker = BatchWorker(processor)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._log)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished_signal.connect(self._on_finished)

        self.start_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._worker.start()

    @Slot()
    def _analyze_only(self):
        videos = self._get_videos()
        if not videos:
            return
        self.log_text.clear()
        self._log(f"批量预分析 {len(videos)} 个视频（不拼接）...")
        self._analyze_worker = AnalyzeWorker(videos)
        self._analyze_worker.log.connect(self._log)
        self._analyze_worker.finished_signal.connect(
            lambda: self._log("预分析完成")
        )
        self.analyze_btn.setEnabled(False)
        self._analyze_worker.finished_signal.connect(
            lambda: self.analyze_btn.setEnabled(True)
        )
        self._analyze_worker.start()

    @Slot()
    def _stop_batch(self):
        if self._worker and self._worker.processor:
            self._worker.processor.cancel()
            self._log("正在停止...")

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    @Slot(dict)
    def _on_file_done(self, result: dict):
        if result.get("success"):
            self._log(
                f"  [完成] {result.get('file', '')} | "
                f"推荐 {result.get('recommended_keyframes', '?')} / "
                f"实际 {result.get('selected_keyframes', '?')} / "
                f"有效 {result.get('active_frames', '?')} / "
                f"贡献 {result.get('contributing_frames', '?')} | "
                f"质量 {result.get('quality_score', '?')} 分 | "
                f"尝试 {len(result.get('attempts', []))} 次 | "
                f"状态: {result.get('status', '')}"
            )
            if result.get("output"):
                self._log(f"        -> {result['output']}")
        else:
            self._log(f"  [失败] {result.get('file', '')}: {result.get('error', '')}")

    @Slot(list)
    def _on_finished(self, results: list):
        self.start_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        success = sum(1 for r in results if r.get("success"))
        failed = len(results) - success
        self.summary_label.setText(f"处理完成: 成功 {success} 个，失败 {failed} 个")
        self._log(f"批量处理结束: 成功 {success} 个，失败 {failed} 个")
        if results:
            out = Path(self.output_selector.get_path()) / "batch_report.csv"
            self._log(f"批量报告: {out}")

        if failed > 0:
            self._log("失败的文件:")
            for r in results:
                if not r.get("success"):
                    self._log(f"  - {r.get('file')}: {r.get('error')}")
