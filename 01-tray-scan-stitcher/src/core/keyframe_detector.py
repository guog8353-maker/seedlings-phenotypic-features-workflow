"""Keyframe detection based on frame difference or motion."""

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .frame_extractor import FrameList


ANALYSIS_SIZE = (320, 180)


def _resize_for_analysis(frame: np.ndarray) -> np.ndarray:
    """Resize frame to a small size for fast analysis."""
    h, w = frame.shape[:2]
    if w <= ANALYSIS_SIZE[0]:
        return frame
    return cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute normalized difference between two frames (0.0 ~ 1.0)."""
    small1 = _resize_for_analysis(frame1)
    small2 = _resize_for_analysis(frame2)

    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0.0, 1.0 - correlation)


def estimate_translation(
    ref: np.ndarray, curr: np.ndarray
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Estimate pure translation using phase correlation on small images."""
    try:
        ref_small = _resize_for_analysis(ref)
        curr_small = _resize_for_analysis(curr)
        ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        (dx, dy), response = cv2.phaseCorrelate(
            np.float32(ref_gray), np.float32(curr_gray)
        )
        # Scale displacement back to original resolution
        scale = ref.shape[1] / ref_small.shape[1]
        return (dx * scale, dy * scale), response
    except Exception:
        return None, 0.0


def detect_keyframes_by_scene(
    video_path: str,
    threshold: float = 0.3,
    min_interval: int = 5,
    max_keyframes: int = 200,
    sample_step: int = 2,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Detect keyframes based on scene change."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keyframes: FrameList = []
    prev_frame: Optional[np.ndarray] = None
    last_keyframe_idx = -min_interval - 1
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_step == 0:
            if prev_frame is None:
                keyframes.append((idx, frame))
                last_keyframe_idx = idx
            else:
                diff = compute_frame_difference(prev_frame, frame)
                if diff >= threshold and (idx - last_keyframe_idx) >= min_interval:
                    keyframes.append((idx, frame))
                    last_keyframe_idx = idx

            prev_frame = frame

        idx += 1

        if progress_callback and not progress_callback(idx, total):
            break
        if len(keyframes) >= max_keyframes:
            break

    cap.release()
    return keyframes


def detect_keyframes_by_motion(
    video_path: str,
    min_pixel_shift: float = 30.0,
    max_keyframes: int = 200,
    sample_step: int = 2,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Detect keyframes based on cumulative motion shift."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keyframes: FrameList = []
    prev_frame: Optional[np.ndarray] = None
    accumulated_shift = 0.0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_step == 0:
            if prev_frame is None:
                keyframes.append((idx, frame))
            else:
                shift, response = estimate_translation(prev_frame, frame)
                if shift is not None and response >= 0.1:
                    dx, dy = shift
                    accumulated_shift += np.hypot(dx, dy)

                if accumulated_shift >= min_pixel_shift:
                    keyframes.append((idx, frame))
                    accumulated_shift = 0.0

            prev_frame = frame

        idx += 1

        if progress_callback and not progress_callback(idx, total):
            break
        if len(keyframes) >= max_keyframes:
            break

    cap.release()

    # Always include last frame if we have at least one keyframe
    if len(keyframes) > 0 and keyframes[-1][0] < total - 1:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, last_frame = cap.read()
        if ret:
            keyframes.append((total - 1, last_frame))
        cap.release()

    return keyframes
