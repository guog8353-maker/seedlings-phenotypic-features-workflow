"""Single video processing tab."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.frame_extractor import FrameList
from ..core.stitcher import MotionInfo, StitchDirection, StitchMode, analyze_motion
from ..core.stitching.config import StitchConfig
from ..core.utils import save_image
from ..core.video_info import VideoInfo
from .image_viewer import ImageViewer
from .widgets import FileSelector, InfoCard, show_error, show_message
from .workers import ExtractWorker, SaveWorker, StitchWorker


class SingleVideoTab(QWidget):
    """Tab for processing a single video."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._video_info: Optional[VideoInfo] = None
        self._extracted_indexed_frames: FrameList = []
        self._extract_stats: dict = {}
        self._last_extract_ui_target: int = 12
        self._last_stitch_result: Optional[dict] = None
        self._stitched_image: Optional[np.ndarray] = None
        self._original_frame: Optional[np.ndarray] = None
        self._extract_worker: Optional[ExtractWorker] = None
        self._stitch_worker: Optional[StitchWorker] = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)

        self.file_selector = FileSelector(
            "视频文件", mode="file", filter_str="MP4 Videos (*.mp4);;All Files (*.*)"
        )
        self.file_selector.path_changed.connect(self._load_video)
        left_layout.addWidget(self.file_selector)

        self.info_card = InfoCard("视频基本信息")
        left_layout.addWidget(self.info_card)

        self.detail_card = InfoCard("视频详细信息")
        left_layout.addWidget(self.detail_card)

        self.thumbnail_label = QLabel("首帧缩略图")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setMinimumHeight(120)
        self.thumbnail_label.setStyleSheet("border: 1px solid #cccccc; background: #f5f5f5;")
        left_layout.addWidget(self.thumbnail_label)

        # Extraction settings (simplified: smart keyframes only)
        extract_group = QGroupBox("抽帧设置")
        extract_layout = QVBoxLayout(extract_group)

        self.extract_mode_combo = QComboBox()
        self.extract_mode_combo.addItem("智能关键帧", "smart")
        self.extract_mode_combo.addItem("均匀抽帧", "uniform")
        self.extract_mode_combo.setCurrentIndex(0)
        self.extract_mode_combo.setVisible(False)  # Hidden; always smart
        extract_layout.addWidget(self.extract_mode_combo)

        ssim_layout = QHBoxLayout()
        ssim_layout.addWidget(QLabel("SSIM 阈值:"))
        self.ssim_spin = QDoubleSpinBox()
        self.ssim_spin.setRange(0.80, 0.99)
        self.ssim_spin.setValue(0.95)
        self.ssim_spin.setDecimals(2)
        self.ssim_spin.setSingleStep(0.01)
        ssim_layout.addWidget(self.ssim_spin)
        extract_layout.addLayout(ssim_layout)

        shift_layout = QHBoxLayout()
        shift_layout.addWidget(QLabel("最小位移:"))
        self.min_shift_spin = QSpinBox()
        self.min_shift_spin.setRange(10, 200)
        self.min_shift_spin.setValue(30)
        shift_layout.addWidget(self.min_shift_spin)
        extract_layout.addLayout(shift_layout)

        # Hidden widgets kept for backward compatibility
        self.step_spin = QSpinBox()
        self.step_spin.setVisible(False)
        self.auto_step_check = QCheckBox()
        self.auto_step_check.setVisible(False)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setVisible(False)
        self.keyframe_scene_widget = QWidget()
        self.keyframe_scene_widget.setVisible(False)
        self.keyframe_motion_widget = QWidget()
        self.keyframe_motion_widget.setVisible(False)
        self.uniform_widget = self.step_spin.parentWidget()

        extract_layout.addWidget(QLabel("提示: SSIM 阈值越低，抽到的关键帧越少；最小位移保证帧间有足够运动"))
        left_layout.addWidget(extract_group)

        # Extracted frames preview
        frames_group = QGroupBox("已抽取帧预览")
        frames_layout = QVBoxLayout(frames_group)
        self.frames_count_label = QLabel("尚未抽取")
        frames_layout.addWidget(self.frames_count_label)

        self.frames_scroll = QScrollArea()
        self.frames_scroll.setWidgetResizable(True)
        self.frames_container = QWidget()
        self.frames_grid = QHBoxLayout(self.frames_container)
        self.frames_grid.setAlignment(Qt.AlignLeft)
        self.frames_scroll.setWidget(self.frames_container)
        self.frames_scroll.setMinimumHeight(140)
        self.frames_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        frames_layout.addWidget(self.frames_scroll)
        left_layout.addWidget(frames_group)

        # Stitching settings (hidden; use internal defaults)
        stitch_group = QGroupBox("拼接设置")
        stitch_layout = QVBoxLayout(stitch_group)

        self.stitch_mode_combo = QComboBox()
        self.stitch_mode_combo.addItem("顺序拼接", StitchMode.SEQUENTIAL)
        self.stitch_mode_combo.addItem("自动拼接", StitchMode.AUTO)
        self.stitch_mode_combo.addItem("条带拼接（推荐）", StitchMode.STRIP)
        self.stitch_mode_combo.setCurrentIndex(2)  # Default to strip
        self.stitch_mode_combo.setVisible(False)
        stitch_layout.addWidget(self.stitch_mode_combo)

        self.auto_direction_check = QCheckBox("自动检测运动方向")
        self.auto_direction_check.setChecked(True)
        self.auto_direction_check.setVisible(False)
        stitch_layout.addWidget(self.auto_direction_check)

        self.direction_combo = QComboBox()
        self.direction_combo.addItem("自动", StitchDirection.AUTO)
        self.direction_combo.addItem("水平", StitchDirection.HORIZONTAL)
        self.direction_combo.addItem("垂直", StitchDirection.VERTICAL)
        self.direction_combo.setVisible(False)
        stitch_layout.addWidget(self.direction_combo)

        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setValue(0.2)
        self.overlap_spin.setVisible(False)
        stitch_layout.addWidget(self.overlap_spin)

        self.preview_scale_spin = QDoubleSpinBox()
        self.preview_scale_spin.setValue(0.75)
        self.preview_scale_spin.setVisible(False)
        stitch_layout.addWidget(self.preview_scale_spin)

        self.output_scale_spin = QDoubleSpinBox()
        self.output_scale_spin.setValue(1.0)
        self.output_scale_spin.setVisible(False)
        stitch_layout.addWidget(self.output_scale_spin)

        self.blend_check = QCheckBox("融合重叠区域")
        self.blend_check.setChecked(True)
        self.blend_check.setVisible(False)
        stitch_layout.addWidget(self.blend_check)

        self.exposure_check = QCheckBox("曝光补偿")
        self.exposure_check.setChecked(True)
        self.exposure_check.setVisible(False)
        stitch_layout.addWidget(self.exposure_check)

        self.high_quality_check = QCheckBox("高质量模式")
        self.high_quality_check.setChecked(False)
        self.high_quality_check.setVisible(False)
        stitch_layout.addWidget(self.high_quality_check)

        self.fast_blend_check = QCheckBox("极速融合")
        self.fast_blend_check.setChecked(False)
        self.fast_blend_check.setVisible(False)
        stitch_layout.addWidget(self.fast_blend_check)

        self.motion_info_label = QLabel("运动信息：尚未分析")
        self.motion_info_label.setWordWrap(True)
        self.motion_info_label.setStyleSheet("color: #666666;")
        stitch_layout.addWidget(self.motion_info_label)

        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("目标关键帧:"))
        self.target_kf_spin = QSpinBox()
        self.target_kf_spin.setRange(2, 30)
        self.target_kf_spin.setValue(12)
        self.target_kf_spin.setToolTip("算法会尽最大努力让最终参与拼接的关键帧接近该数量")
        self.target_kf_spin.setEnabled(False)
        target_layout.addWidget(self.target_kf_spin)
        self.auto_kf_check = QCheckBox("自动推荐")
        self.auto_kf_check.setChecked(True)
        self.auto_kf_check.setToolTip("按每个视频的运动距离、重叠率和覆盖范围自动决定关键帧数量")
        self.auto_kf_check.toggled.connect(
            lambda checked: self.target_kf_spin.setEnabled(not checked)
        )
        target_layout.addWidget(self.auto_kf_check)
        stitch_layout.addLayout(target_layout)

        self.strict_check = QCheckBox("严格多帧模式（帧数不足时明确警告，不假装成功）")
        self.strict_check.setChecked(False)
        stitch_layout.addWidget(self.strict_check)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("输出画质:"))
        self.quality_mode_combo = QComboBox()
        self.quality_mode_combo.addItem("自然（推荐，无任何后处理）", "natural")
        self.quality_mode_combo.addItem("标准（完成后一次极轻去噪）", "standard")
        self.quality_mode_combo.addItem("增强（去噪 + 极轻锐化）", "enhanced")
        self.quality_mode_combo.setCurrentIndex(0)
        self.quality_mode_combo.setToolTip("自然模式不做任何锐化/CLAHE/增强，保留原始照片质感")
        quality_layout.addWidget(self.quality_mode_combo)
        stitch_layout.addLayout(quality_layout)

        self.counts_label = QLabel("帧数统计：尚未拼接")
        self.counts_label.setWordWrap(True)
        self.counts_label.setStyleSheet("color: #333333;")
        stitch_layout.addWidget(self.counts_label)

        stitch_layout.addWidget(QLabel("提示: 默认使用线扫条带拼接，输出清晰长图，无重影、无比例突变"))
        left_layout.addWidget(stitch_group)

        # Buttons
        button_layout = QHBoxLayout()
        self.extract_btn = QPushButton("1. 抽取帧")
        self.extract_btn.clicked.connect(self._extract_frames)
        button_layout.addWidget(self.extract_btn)

        self.cancel_extract_btn = QPushButton("取消")
        self.cancel_extract_btn.clicked.connect(self._cancel_extract)
        self.cancel_extract_btn.setEnabled(False)
        button_layout.addWidget(self.cancel_extract_btn)

        self.preview_btn = QPushButton("2. 预览拼接")
        self.preview_btn.clicked.connect(self._preview_stitch)
        self.preview_btn.setEnabled(False)
        button_layout.addWidget(self.preview_btn)

        self.save_btn = QPushButton("3. 保存拼接图")
        self.save_btn.clicked.connect(self._save_stitched)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        left_layout.addLayout(button_layout)

        debug_button_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("分析视频")
        self.analyze_btn.clicked.connect(self._analyze_video)
        self.analyze_btn.setToolTip("只分析不拼接：运动方向/运动量/模糊/推荐关键帧数/拼接难度")
        debug_button_layout.addWidget(self.analyze_btn)

        self.export_debug_btn = QPushButton("导出关键帧调试")
        self.export_debug_btn.clicked.connect(self._export_debug_frames)
        self.export_debug_btn.setToolTip(
            "导出候选帧/选中帧/拼接帧图片和 frame_pipeline.json，用于核对每一级真实帧数"
        )
        debug_button_layout.addWidget(self.export_debug_btn)
        left_layout.addLayout(debug_button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("请选择视频文件")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.image_viewer = ImageViewer()
        right_layout.addWidget(self.image_viewer)

        viewer_buttons = QHBoxLayout()
        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.image_viewer.reset_view)
        viewer_buttons.addWidget(self.reset_view_btn)

        self.toggle_view_btn = QPushButton("查看原始帧")
        self.toggle_view_btn.clicked.connect(self._toggle_view)
        self.toggle_view_btn.setEnabled(False)
        viewer_buttons.addWidget(self.toggle_view_btn)

        viewer_buttons.addStretch()
        right_layout.addLayout(viewer_buttons)

        splitter.addWidget(right_widget)
        splitter.setSizes([450, 750])

    @Slot(str)
    def _load_video(self, path: str):
        if not path:
            return
        try:
            self._video_info = VideoInfo.from_path(path)
            self.info_card.set_data(self._video_info.get_summary())
            self.detail_card.set_data(self._video_info.get_details())

            if self._video_info.thumbnail is not None:
                self._original_frame = self._video_info.thumbnail.copy()
                thumb = self._make_thumbnail(self._video_info.thumbnail, 240)
                self.thumbnail_label.setPixmap(self._numpy_to_pixmap(thumb))
            else:
                self._original_frame = None
                self.thumbnail_label.setText("无法读取首帧")

            self._extracted_indexed_frames = []
            self._stitched_image = None
            self._clear_frames_preview()
            self.image_viewer.clear()
            self.preview_btn.setEnabled(True)  # Enable preview button when video is loaded
            self.save_btn.setEnabled(False)
            self.toggle_view_btn.setEnabled(False)
            self.motion_info_label.setText("运动信息：尚未分析")
            self.status_label.setText(f"已加载: {self._video_info.filename}")

        except Exception as e:
            show_error(self, "加载失败", f"无法读取视频信息:\n{e}")
            self._video_info = None

    def _make_thumbnail(self, frame: np.ndarray, max_size: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))

    def _limit_display_size(self, frame: np.ndarray, max_dim: int = 1920) -> np.ndarray:
        """Resize image so the largest dimension fits within max_dim for display."""
        h, w = frame.shape[:2]
        max_current = max(w, h)
        if max_current <= max_dim:
            return frame
        scale = max_dim / max_current
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _numpy_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        if frame is None or frame.size == 0:
            return QPixmap()
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(image)
        return QPixmap()

    def _clear_frames_preview(self):
        while self.frames_grid.count():
            item = self.frames_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.frames_count_label.setText("尚未抽取")

    def _update_frames_preview(self):
        self._clear_frames_preview()
        frames = [f for _, f in self._extracted_indexed_frames]
        if not frames:
            return

        self.frames_count_label.setText(f"共 {len(frames)} 帧")
        for i, frame in enumerate(frames[:30]):
            thumb = self._make_thumbnail(frame, 80)
            label = QLabel()
            label.setPixmap(self._numpy_to_pixmap(thumb))
            label.setToolTip(f"帧 {i + 1}")
            label.setStyleSheet("border: 1px solid #cccccc; margin: 2px;")
            self.frames_grid.addWidget(label)

        if len(frames) > 30:
            more_label = QLabel(f"+{len(frames) - 30} 帧")
            self.frames_grid.addWidget(more_label)

    @Slot()
    def _on_extract_mode_changed(self):
        # Keyframe modes are hidden; always use uniform extraction.
        pass

    @Slot()
    def _on_auto_step_changed(self):
        self.step_spin.setEnabled(not self.auto_step_check.isChecked())

    @Slot()
    def _on_auto_direction_changed(self):
        self.direction_combo.setEnabled(not self.auto_direction_check.isChecked())

    @Slot()
    def _on_high_quality_changed(self):
        # High quality requires blending and conflicts with fast blend.
        if self.high_quality_check.isChecked():
            self.blend_check.setChecked(True)
            self.fast_blend_check.setChecked(False)

    @Slot()
    def _on_fast_blend_changed(self):
        # Fast blend overrides high quality.
        if self.fast_blend_check.isChecked():
            self.high_quality_check.setChecked(False)

    @Slot()
    def _extract_frames(self):
        if self._video_info is None:
            show_error(self, "错误", "请先选择视频文件")
            return

        self.cancel_extract_btn.setEnabled(True)
        self.extract_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("正在后台抽帧...")
        self.progress_bar.setValue(0)

        mode = self.extract_mode_combo.currentData()
        params = self._get_extract_param()
        self._last_extract_ui_target = params["target_keyframes"]
        print(f"UI TARGET = {params['target_keyframes']}", flush=True)

        self._extract_worker = ExtractWorker(
            video_path=self._video_info.path,
            mode=mode,
            params=params,
            parent=self,
        )
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.progress_text.connect(self._on_extract_progress_text)
        self._extract_worker.finished_signal.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)
        self._extract_worker.start()

    @Slot()
    def _cancel_extract(self):
        if self._extract_worker and self._extract_worker.isRunning():
            self._extract_worker.cancel()
            self.status_label.setText("正在取消抽帧...")

    @Slot(int, int)
    def _on_extract_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    @Slot(str)
    def _on_extract_progress_text(self, text: str):
        self.status_label.setText(text)

    @Slot(object)
    def _on_extract_finished(self, payload):
        # Worker emits (indexed_frames, stats_dict); tolerate a bare list.
        if isinstance(payload, tuple):
            indexed_frames, stats = payload
        else:
            indexed_frames, stats = payload, {}
        self._extracted_indexed_frames = indexed_frames
        self._extract_stats = stats
        self._update_frames_preview()
        self.progress_bar.setValue(100)

        status_text = f"已抽取 {len(indexed_frames)} 帧"
        if indexed_frames:
            frames = [f for _, f in indexed_frames]
            h, w = frames[0].shape[:2]
            motion = analyze_motion(frames, StitchDirection.AUTO)
            if motion.direction == "水平":
                avg_step = abs(motion.avg_dx)
                dimension = float(w)
            elif motion.direction == "垂直":
                avg_step = abs(motion.avg_dy)
                dimension = float(h)
            else:
                avg_step = motion.avg_shift
                dimension = float(max(w, h))
            if dimension > 0 and avg_step > 0:
                overlap = 1.0 - avg_step / dimension
                if overlap < 0.3:
                    status_text += (
                        f" [警告：帧间重叠仅 {overlap * 100:.0f}%，"
                        "建议减小步长或使用“关键帧-运动位移”模式]"
                    )

        self.status_label.setText(status_text)
        self._refresh_counts_display()
        self.extract_btn.setEnabled(True)
        self.cancel_extract_btn.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.toggle_view_btn.setEnabled(False)

    @Slot(str)
    def _on_extract_error(self, message: str):
        self.progress_bar.setValue(0)
        self.status_label.setText(message)
        self.extract_btn.setEnabled(True)
        self.cancel_extract_btn.setEnabled(False)
        show_error(self, "抽帧失败", message)

    def _build_stitch_config(self) -> StitchConfig:
        """Build the authoritative config from the current UI values.

        Called at the moment the user clicks a button, so the worker always
        receives the latest UI state.
        """
        cfg = StitchConfig(
            target_keyframes=self.target_kf_spin.value(),
            auto_keyframes=self.auto_kf_check.isChecked(),
            strict_multi_frame_mode=self.strict_check.isChecked(),
            quality_mode=self.quality_mode_combo.currentData(),
        )
        cfg.log("UI")
        return cfg

    @Slot()
    def _preview_stitch(self):
        if not self._video_info:
            show_error(self, "错误", "请先选择视频文件")
            return

        self.status_label.setText("正在后台拼接...")
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.preview_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.counts_label.setText("帧数统计：拼接中...")
        self.counts_label.setStyleSheet("color: #333333;")

        self._stitch_worker = StitchWorker(
            video_path=self._video_info.path,
            config=self._build_stitch_config(),
            parent=self,
        )
        self._stitch_worker.progress.connect(self._on_stitch_progress)
        self._stitch_worker.progress_text.connect(self._on_stitch_progress_text)
        self._stitch_worker.finished_signal.connect(self._on_stitch_finished)
        self._stitch_worker.error.connect(self._on_stitch_error)
        self._stitch_worker.start()

    def _get_effective_direction(self, frames) -> StitchDirection:
        if self.auto_direction_check.isChecked():
            motion = analyze_motion(frames, StitchDirection.AUTO)
            if motion.direction == "水平":
                return StitchDirection.HORIZONTAL
            elif motion.direction == "垂直":
                return StitchDirection.VERTICAL
            return StitchDirection.AUTO
        return self.direction_combo.currentData()

    @Slot(object)
    def _on_stitch_finished(self, result):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.preview_btn.setEnabled(True)

        self._update_frame_counts(result)

        if result["success"] and result["image"] is not None:
            self._stitched_image = result["image"]
            # Avoid blocking the UI with huge images by downsampling for display.
            display_image = self._limit_display_size(result["image"], max_dim=1920)
            self.image_viewer.set_image(display_image)
            self.save_btn.setEnabled(True)
            self.toggle_view_btn.setEnabled(True)
            self.toggle_view_btn.setText("查看原始帧")

            self.status_label.setText(
                f"预览拼接成功 | {result['message']} | 尺寸: {result['image'].shape[1]} x {result['image'].shape[0]}"
            )
        else:
            self._stitched_image = None
            self.save_btn.setEnabled(False)
            self.toggle_view_btn.setEnabled(False)
            show_error(self, "拼接失败", result["message"])
            self.status_label.setText(f"拼接失败: {result['message']}")

    def _update_frame_counts(self, result: dict):
        """Store the stitch result and refresh the counts display."""
        self._last_stitch_result = result
        self._refresh_counts_display()

    def _refresh_counts_display(self):
        """Show extraction-stage and stitching-stage counts side by side.

        All numbers come from the actual lists used by the pipeline / extractor,
        never from the configured target, so the UI cannot show "12" while
        only 2 frames really exist.
        """
        lines = []

        # ---- Extraction stage (from the "抽取帧" button) ----
        stats = self._extract_stats
        if stats:
            extracted = len(self._extracted_indexed_frames)
            lines.append(
                f"抽帧 | 目标: {self._last_extract_ui_target} | "
                f"视频总帧: {stats.get('total_video_frames', '?')} | "
                f"候选采样: {stats.get('raw_sampled', '?')} | "
                f"质量过滤后: {stats.get('quality_passed', '?')} | "
                f"最终选中: {extracted}"
            )

        # ---- Stitching stage (from the "预览拼接" button) ----
        result = self._last_stitch_result
        if result:
            counts = result.get("frame_counts") or {}
            analysis = result.get("video_analysis") or {}
            status_map = {
                "SINGLE_FRAME_FALLBACK": "单帧保底",
                "LIMITED_STITCH": "关键帧不足（仅 2 帧参与）",
                "MULTI_FRAME_STITCH": "多帧拼接（帧数偏少）",
                "NORMAL_MULTI_FRAME_STITCH": "正常多帧拼接",
            }
            status = status_map.get(result.get("stitch_status", ""), "未知")
            recommended = analysis.get("recommended_keyframes")
            rec_text = f"推荐: {recommended} | " if recommended else ""
            quality = result.get("quality_score")
            quality_text = f" | 质量: {quality}分" if quality is not None else ""
            lines.append(
                f"拼接 | {rec_text}目标: {counts.get('target_keyframes', '?')} | "
                f"选中: {counts.get('selected_frames', '?')} | "
                f"匹配有效: {counts.get('active_stitch_frames', '?')} | "
                f"最终参与: {counts.get('active_stitch_frames', '?')} | "
                f"实际贡献: {counts.get('contributing_frames', '?')} | "
                f"状态: {status}{quality_text}"
            )

        if not lines:
            self.counts_label.setText("帧数统计：尚未处理")
            self.counts_label.setStyleSheet("color: #333333;")
            return

        self.counts_label.setText("\n".join(lines))

        # Red bold warning when the actual count severely mismatches the target.
        insufficient = False
        if stats:
            target = self._last_extract_ui_target
            if target >= 8 and 0 < len(self._extracted_indexed_frames) <= 2:
                insufficient = True
        if result:
            counts = result.get("frame_counts") or {}
            target = counts.get("target_keyframes", 0)
            active = counts.get("active_stitch_frames", 0)
            if result.get("stitch_status") in ("SINGLE_FRAME_FALLBACK", "LIMITED_STITCH"):
                insufficient = True
            if isinstance(target, int) and target >= 8 and isinstance(active, int) and active <= 2:
                insufficient = True

        if insufficient:
            self.counts_label.setStyleSheet("color: #cc0000; font-weight: bold;")
        else:
            self.counts_label.setStyleSheet("color: #228822;")

    @Slot(int, int)
    def _on_stitch_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(100 * current / total))

    @Slot(str)
    def _on_stitch_progress_text(self, text: str):
        self.status_label.setText(text)

    @Slot(str)
    def _on_stitch_error(self, message: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.preview_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.status_label.setText(message)
        show_error(self, "拼接失败", message)

    @Slot()
    def _toggle_view(self):
        if self._stitched_image is None:
            return

        if self.toggle_view_btn.text() == "查看原始帧":
            if self._original_frame is not None:
                self.image_viewer.set_image(self._original_frame)
                self.toggle_view_btn.setText("查看拼接图")
        else:
            self.image_viewer.set_image(self._stitched_image)
            self.toggle_view_btn.setText("查看原始帧")

    @Slot()
    def _save_stitched(self):
        if not self._video_info:
            show_error(self, "错误", "请先选择视频文件")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存拼接图",
            "stitched.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif)",
        )
        if not path:
            return

        self._save_path = path
        self.status_label.setText("正在生成高清拼接图...")
        self.progress_bar.setRange(0, 0)
        self.save_btn.setEnabled(False)

        self._save_worker = SaveWorker(
            save_path=path,
            video_path=self._video_info.path,
            config=self._build_stitch_config(),
            parent=self,
        )
        self._save_worker.progress.connect(self._on_stitch_progress)
        self._save_worker.progress_text.connect(self._on_stitch_progress_text)
        self._save_worker.finished_signal.connect(self._on_save_finished)
        self._save_worker.error.connect(self._on_save_error)
        self._save_worker.start()

    @Slot(bool, str, str)
    def _on_save_finished(self, success: bool, path: str, size_text: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.save_btn.setEnabled(True)

        if success:
            show_message(self, "保存成功", f"已保存到:\n{path}")
            self.status_label.setText(f"已保存: {path} | 尺寸: {size_text}")
        else:
            show_error(self, "保存失败", "无法写入文件")

    @Slot(str)
    def _on_save_error(self, message: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.save_btn.setEnabled(True)
        self.status_label.setText(message)
        show_error(self, "保存失败", message)

    @Slot()
    def _analyze_video(self):
        """Run the AUTO pre-analysis only (no stitching) and show the result."""
        if not self._video_info:
            show_error(self, "错误", "请先选择视频文件")
            return

        self.status_label.setText("正在分析视频（不拼接）...")
        self.analyze_btn.setEnabled(False)
        try:
            from ..core.stitching.video_analyzer import VideoAnalyzer

            analyzer = VideoAnalyzer()
            a = analyzer.analyze(self._video_info.path)
            direction_map = {
                "LEFT_TO_RIGHT": "左→右",
                "RIGHT_TO_LEFT": "右→左",
                "TOP_TO_BOTTOM": "上→下",
                "BOTTOM_TO_TOP": "下→上",
                "DIAGONAL": "对角",
                "FREE_2D": "自由二维",
                "STATIC": "几乎静止",
                "UNKNOWN": "未知",
            }
            text = (
                f"时长: {a.duration:.1f}s | 总帧: {a.total_frames} | FPS: {a.fps:.1f}\n"
                f"运动方向: {direction_map.get(a.motion_direction, a.motion_direction)} | "
                f"扫描跨度: {a.cumulative_span_px:.0f}px | "
                f"模糊比例: {a.blur_ratio:.0%}\n"
                f"推荐关键帧: {a.recommended_keyframes} 帧 | "
                f"拼接难度: {a.stitchability}\n"
                f"依据: {'；'.join(a.reasons) if a.reasons else '—'}"
            )
            show_message(self, "视频分析结果", text)
            self.status_label.setText(
                f"分析完成: 推荐 {a.recommended_keyframes} 帧 | 难度 {a.stitchability}"
            )
        except Exception as e:
            show_error(self, "分析失败", str(e))
            self.status_label.setText(f"分析失败: {e}")
        finally:
            self.analyze_btn.setEnabled(True)

    @Slot()
    def _export_debug_frames(self):
        """Export candidate/selected/active frames and frame_pipeline.json."""
        if not self._video_info:
            show_error(self, "错误", "请先选择视频文件")
            return

        stem = Path(self._video_info.path).stem
        out_dir = (
            Path.cwd() / "debug_frames" / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"
        )
        cand_dir = out_dir / "raw_candidates"
        sel_dir = out_dir / "selected_keyframes"
        act_dir = out_dir / "active_stitch_frames"
        for d in (cand_dir, sel_dir, act_dir):
            d.mkdir(parents=True, exist_ok=True)

        stats = self._extract_stats or {}

        # 1. Raw candidates (kept by the extractor when running from the UI).
        candidates = stats.get("candidate_frames_data", [])
        for idx, frame in candidates:
            save_image(cand_dir / f"candidate_{idx:05d}.jpg", frame)

        # 2. Selected keyframes (the real extraction result).
        for idx, frame in self._extracted_indexed_frames:
            save_image(sel_dir / f"keyframe_{idx:05d}.jpg", frame)

        # 3. Active stitch frames: re-read from the video by frame id.
        result = self._last_stitch_result or {}
        active_ids = result.get("active_frame_ids") or []
        if active_ids:
            cap = cv2.VideoCapture(self._video_info.path)
            if cap.isOpened():
                for fid in active_ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ret, frame = cap.read()
                    if ret:
                        save_image(act_dir / f"active_{int(fid):05d}.jpg", frame)
                cap.release()

        # 4. frame_pipeline.json with the real per-stage numbers.
        counts = result.get("frame_counts") or {}
        pipeline_info = {
            "ui_target": self._last_extract_ui_target,
            "worker_target": stats.get("worker_target"),
            "extractor_target": stats.get("extractor_target"),
            "raw_candidates": stats.get("raw_sampled", 0),
            "after_quality": stats.get("quality_passed", 0),
            "after_duplicate": stats.get("duplicate_after", 0),
            "selected_keyframes": len(self._extracted_indexed_frames),
            "active_stitch_frames": counts.get("active_stitch_frames", 0),
            "contributing_frames": counts.get("contributing_frames", 0),
            "stitch_status": result.get("stitch_status", ""),
            "frame_records": stats.get("frame_records", []),
        }
        with open(out_dir / "frame_pipeline.json", "w", encoding="utf-8") as f:
            json.dump(pipeline_info, f, indent=2, ensure_ascii=False)

        show_message(
            self,
            "导出完成",
            f"关键帧调试文件已保存到:\n{out_dir}\n\n"
            f"候选帧: {len(candidates)} 张 | "
            f"选中帧: {len(self._extracted_indexed_frames)} 张 | "
            f"拼接帧: {len(active_ids)} 张",
        )
        self.status_label.setText(f"已导出关键帧调试: {out_dir}")

    def get_current_params(self) -> dict:
        """Return current parameters for batch processing."""
        return {
            "extract_mode": self.extract_mode_combo.currentData(),
            "extract_param": self._get_extract_param(),
            "stitch_mode": self.stitch_mode_combo.currentData(),
            "stitch_direction": self.direction_combo.currentData(),
            "auto_direction": self.auto_direction_check.isChecked(),
            "overlap_ratio": self.overlap_spin.value(),
            "preview_scale": self.preview_scale_spin.value(),
            "output_scale": self.output_scale_spin.value(),
            "blend": self.blend_check.isChecked(),
            "fast_blend": self.fast_blend_check.isChecked(),
            "exposure_compensate": self.exposure_check.isChecked(),
            "high_quality": self.high_quality_check.isChecked(),
        }

    def _get_extract_param(self) -> dict:
        return {
            "target_keyframes": self.target_kf_spin.value(),
            "laplacian_threshold": 100,
        }
