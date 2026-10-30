"""Low-cost video pre-analysis for AUTO keyframe recommendation.

The recommended keyframe count is driven by SPATIAL COVERAGE (how far the
camera actually travelled across the scene), never by video duration alone.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoAnalysis:
    """Result of the low-cost pre-analysis of one video."""

    video_path: str = ""
    total_frames: int = 0
    fps: float = 0.0
    duration: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    sample_count: int = 0
    motion_direction: str = "UNKNOWN"
    median_shift_px: float = 0.0       # per analysis step, main axis
    cumulative_span_px: float = 0.0    # estimated total scan span (main axis)
    blur_ratio: float = 0.0
    mean_response: float = 0.0
    direction_consistency: float = 0.0
    recommended_keyframes: int = 0
    stitchability: str = "NORMAL"      # GOOD / NORMAL / DIFFICULT / BAD
    reasons: List[str] = field(default_factory=list)


class VideoAnalyzer:
    """Analyze a video cheaply and recommend a keyframe count."""

    def __init__(
        self,
        target_overlap: float = 0.60,
        min_auto_keyframes: int = 4,
        max_auto_keyframes: int = 20,
        blur_threshold: float = 100.0,
    ):
        self.target_overlap = target_overlap
        self.min_auto_keyframes = min_auto_keyframes
        self.max_auto_keyframes = max_auto_keyframes
        self.blur_threshold = blur_threshold

    def analyze(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int], bool]] = None,
    ) -> VideoAnalysis:
        result = VideoAnalysis(video_path=video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result.stitchability = "BAD"
            result.reasons.append("无法打开视频")
            return result

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.total_frames = total
        result.fps = fps
        result.duration = total / fps if fps > 0 else 0.0
        result.frame_width = w
        result.frame_height = h

        if total < 3:
            cap.release()
            result.recommended_keyframes = max(2, total)
            result.stitchability = "BAD"
            result.reasons.append(f"视频过短（{total} 帧）")
            return result

        # Low-cost sampling: ~30 analysis frames, scaled by video length.
        sample_count = int(np.clip(total // 20, 12, 60))
        sample_ids = set(np.linspace(0, total - 1, sample_count, dtype=int).tolist())
        result.sample_count = len(sample_ids)

        # Sequential read, keeping only analysis frames (small, half-res).
        small_frames: List[Tuple[int, np.ndarray]] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in sample_ids:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_AREA)
                small_frames.append((idx, small))
            idx += 1
            if progress_callback and not progress_callback(idx, total):
                break
        cap.release()

        if len(small_frames) < 2:
            result.recommended_keyframes = self.min_auto_keyframes
            result.stitchability = "BAD"
            result.reasons.append("无法读取足够的分析帧")
            return result

        # Motion between consecutive analysis frames via phase correlation.
        shifts: List[Tuple[float, float]] = []
        responses: List[float] = []
        blur_flags = 0
        for k in range(1, len(small_frames)):
            prev = small_frames[k - 1][1]
            curr = small_frames[k][1]
            try:
                (dx, dy), response = cv2.phaseCorrelate(
                    np.float32(prev), np.float32(curr)
                )
            except Exception:
                dx, dy, response = 0.0, 0.0, 0.0
            shifts.append((dx * 2.0, dy * 2.0))  # back to full resolution
            responses.append(float(response))

        for _, small in small_frames:
            if float(cv2.Laplacian(small, cv2.CV_64F).var()) < self.blur_threshold:
                blur_flags += 1

        dxs = np.array([s[0] for s in shifts])
        dys = np.array([s[1] for s in shifts])
        sum_dx = float(np.sum(dxs))
        sum_dy = float(np.sum(dys))

        # Main axis = the axis with the larger cumulative travel.
        if abs(sum_dx) >= abs(sum_dy):
            main = np.abs(dxs)
            signed = dxs
            dim = float(w)
            axis = "x"
        else:
            main = np.abs(dys)
            signed = dys
            dim = float(h)
            axis = "y"

        sample_step = max((total - 1) / max(len(small_frames) - 1, 1), 1.0)
        median_shift = float(np.median(main)) if len(main) else 0.0
        per_frame = median_shift / sample_step
        span = per_frame * total
        result.median_shift_px = median_shift
        result.cumulative_span_px = span

        # Direction classification.
        if max(abs(sum_dx), abs(sum_dy)) < 1.0:
            result.motion_direction = "STATIC"
        elif abs(sum_dx) >= 2.0 * abs(sum_dy):
            result.motion_direction = "LEFT_TO_RIGHT" if sum_dx > 0 else "RIGHT_TO_LEFT"
        elif abs(sum_dy) >= 2.0 * abs(sum_dx):
            result.motion_direction = "TOP_TO_BOTTOM" if sum_dy > 0 else "BOTTOM_TO_TOP"
        else:
            result.motion_direction = "DIAGONAL" if min(abs(sum_dx), abs(sum_dy)) > 0 else "FREE_2D"

        # Direction consistency along the main axis.
        if len(signed) and np.sum(main) > 1e-6:
            dominant_sign = 1.0 if (sum_dx if axis == "x" else sum_dy) >= 0 else -1.0
            result.direction_consistency = float(
                np.mean((signed * dominant_sign) >= 0)
            )
        result.mean_response = float(np.mean(responses)) if responses else 0.0
        result.blur_ratio = blur_flags / len(small_frames)

        # ---- Recommended keyframes from spatial coverage ----
        if span < 0.5 * dim:
            recommended = 3
            result.reasons.append(
                f"几乎无运动（扫描跨度约 {span:.0f}px < 半幅 {dim / 2:.0f}px），取 3 帧"
            )
        else:
            step_px = max((1.0 - self.target_overlap) * dim, 1.0)
            recommended = int(round(span / step_px)) + 1
            recommended = int(np.clip(
                recommended, self.min_auto_keyframes, self.max_auto_keyframes
            ))
            result.reasons.append(
                f"扫描跨度约 {span:.0f}px（{span / dim:.1f} 倍帧{'宽' if axis == 'x' else '高'}），"
                f"按目标重叠 {self.target_overlap:.0%} 推荐 {recommended} 帧"
            )
        result.recommended_keyframes = recommended

        # ---- Stitchability rating ----
        if result.mean_response < 0.05 or result.blur_ratio > 0.5:
            result.stitchability = "BAD"
            result.reasons.append(
                f"相位相关置信度过低({result.mean_response:.2f})或模糊帧过多({result.blur_ratio:.0%})"
            )
        elif result.mean_response < 0.15 or result.blur_ratio > 0.25 \
                or result.direction_consistency < 0.7:
            result.stitchability = "DIFFICULT"
            result.reasons.append(
                f"匹配置信度 {result.mean_response:.2f}，模糊比例 {result.blur_ratio:.0%}，"
                f"方向一致性 {result.direction_consistency:.0%}"
            )
        elif result.mean_response >= 0.3 and result.blur_ratio <= 0.1 \
                and result.direction_consistency >= 0.9:
            result.stitchability = "GOOD"
        else:
            result.stitchability = "NORMAL"

        return result
