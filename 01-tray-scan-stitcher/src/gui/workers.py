"""Worker threads for heavy video processing tasks."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core.frame_extractor import extract_dynamic_frames, extract_uniform
from ..core.keyframe_detector import detect_keyframes_by_motion, detect_keyframes_by_scene
from ..core.smart_stitcher import SmartStitcher
from ..core.stitcher import StitchDirection, StitchMode, stitch_frames
from ..core.stitching import StitchingPipeline
from ..core.stitching.config import StitchConfig, default_debug_root
from ..core.stitching.keyframe_selector import extract_keyframes
from ..core.utils import save_image


class ExtractWorker(QThread):
    """Worker thread for frame extraction."""

    progress = Signal(int, int)
    progress_text = Signal(str)
    finished_signal = Signal(object)  # emits (indexed_frames, stats_dict)
    error = Signal(str)

    def __init__(
        self,
        video_path: str,
        mode: str,
        params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.mode = mode
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.mode == "smart":
                # Unified keyframe extraction: same entry point as stitching.
                cfg = StitchConfig(
                    target_keyframes=self.params.get("target_keyframes", 12),
                    blur_threshold=self.params.get("laplacian_threshold", 100.0),
                )
                print(f"WORKER TARGET = {cfg.target_keyframes}", flush=True)
                cfg.log("WORKER")
                self.progress_text.emit("正在两阶段抽取关键帧...")
                stats: dict = {}
                indexed = extract_keyframes(
                    self.video_path,
                    cfg,
                    progress_callback=self._on_progress,
                    stats=stats,
                    keep_candidates=True,
                )
                stats["worker_target"] = cfg.target_keyframes
            elif self.mode == "keyframe_scene":
                self.progress_text.emit("正在基于场景变化检测关键帧...")
                indexed = detect_keyframes_by_scene(
                    self.video_path,
                    threshold=self.params.get("threshold", 0.25),
                    min_interval=self.params.get("min_interval", 10),
                    max_keyframes=self.params.get("max_keyframes", 200),
                    sample_step=self.params.get("sample_step", 3),
                    progress_callback=self._on_progress,
                )
                stats = {}
            elif self.mode == "keyframe_motion":
                self.progress_text.emit("正在基于运动位移检测关键帧...")
                indexed = detect_keyframes_by_motion(
                    self.video_path,
                    min_pixel_shift=self.params.get("min_pixel_shift", 30),
                    max_keyframes=self.params.get("max_keyframes", 200),
                    sample_step=self.params.get("sample_step", 3),
                    progress_callback=self._on_progress,
                )
                stats = {}
            else:
                self.progress_text.emit("正在均匀抽帧...")
                fps = self.params.get("fps")
                indexed = extract_uniform(
                    self.video_path,
                    step=self.params.get("step", 3),
                    fps=fps if fps and fps > 0 else None,
                    auto_step=self.params.get("auto_step", False),
                    progress_callback=self._on_progress,
                )
                stats = {}

            if self._cancelled:
                self.error.emit("抽帧已取消")
                return

            self.finished_signal.emit((indexed, stats))

        except Exception as e:
            self.error.emit(f"抽帧失败: {e}")

    def _on_progress(self, current: int, total: int) -> bool:
        # Throttle UI updates to avoid flooding the main thread with signals.
        if current % 5 == 0 or current == total:
            self.progress.emit(current, total)
        if current % 10 == 0 or current == total:
            self.progress_text.emit(f"已处理 {current} / {total} 帧")
        return not self._cancelled


class StitchWorker(QThread):
    """Worker thread for image stitching (preview)."""

    progress = Signal(int, int)
    progress_text = Signal(str)
    finished_signal = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        video_path: str,
        config: Optional[StitchConfig] = None,
        feature_type: str = "orb",
        debug_dir: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.config = config if config is not None else StitchConfig()
        # feature_type / debug_dir are legacy overrides on top of the config.
        self.config.apply_overrides(
            feature_type=feature_type,
            debug_dir=debug_dir if debug_dir is not None else default_debug_root(),
        )

    def run(self):
        try:
            self.config.log("WORKER")
            self.progress_text.emit("正在运行拼接流水线...")
            stitcher = SmartStitcher(config=self.config)
            result = stitcher.process_video(
                self.video_path,
                progress_callback=self._on_progress,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.error.emit(f"拼接失败: {e}")

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)
        if current % 5 == 0 or current == total:
            self.progress_text.emit(f"拼接进度 {current} / {total}")


class SaveWorker(QThread):
    """Worker thread for high-resolution stitching and saving."""

    progress = Signal(int, int)
    progress_text = Signal(str)
    finished_signal = Signal(bool, str, str)
    error = Signal(str)

    def __init__(
        self,
        save_path: str,
        video_path: str,
        config: Optional[StitchConfig] = None,
        feature_type: str = "orb",
        quality: int = 95,
        parent=None,
    ):
        super().__init__(parent)
        self.save_path = Path(save_path)
        self.video_path = video_path
        self.config = config if config is not None else StitchConfig()
        self.config.apply_overrides(feature_type=feature_type)
        self.quality = quality

    def run(self):
        try:
            self.config.log("WORKER")
            self.progress_text.emit("正在生成高清拼接图...")
            stitcher = SmartStitcher(config=self.config)
            result = stitcher.process_video(
                self.video_path,
                progress_callback=self._on_progress,
            )

            if not result["success"] or result["image"] is None:
                self.error.emit(result["message"])
                return

            # Single-frame fallback results must not masquerade as a stitch.
            save_path = self.save_path
            if result.get("level", 0) >= 3:
                save_path = save_path.with_name(
                    f"{save_path.stem}_FALLBACK{save_path.suffix}"
                )
                self.progress_text.emit("单帧保底结果，文件名追加 _FALLBACK")

            self.progress_text.emit("正在保存文件...")
            ok = save_image(save_path, result["image"], quality=self.quality)
            if not ok:
                self.error.emit("无法写入文件")
                return

            size_text = f"{result['image'].shape[1]} x {result['image'].shape[0]}"
            self.finished_signal.emit(True, str(save_path), size_text)
        except Exception as e:
            self.error.emit(f"保存失败: {e}")

    def _on_progress(self, current: int, total: int) -> None:
        self.progress.emit(current, total)
        if current % 5 == 0 or current == total:
            self.progress_text.emit(f"高清拼接进度 {current} / {total}")
