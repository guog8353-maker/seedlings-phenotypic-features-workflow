"""Frame extraction strategies."""

from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


FrameList = List[Tuple[int, np.ndarray]]


def extract_all_frames(
    video_path: str,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Extract all frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frames: FrameList = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, frame))
        idx += 1
        if progress_callback and not progress_callback(idx, total):
            break

    cap.release()
    return frames


def _estimate_motion_magnitude(video_path: str, sample_step: int = 10) -> float:
    """Estimate average frame-to-frame motion magnitude in the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= sample_step * 2:
        cap.release()
        return 0.0

    prev_gray = None
    magnitudes = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                try:
                    (dx, dy), response = cv2.phaseCorrelate(
                        np.float32(prev_gray), np.float32(gray)
                    )
                    if response >= 0.1:
                        magnitudes.append(np.hypot(dx, dy))
                except Exception:
                    pass
            prev_gray = gray
        idx += 1
        if idx > total:
            break

    cap.release()
    if not magnitudes:
        return 0.0
    return float(np.median(magnitudes))


def extract_uniform(
    video_path: str,
    step: int = 1,
    fps: Optional[float] = None,
    auto_step: bool = False,
    target_overlap: float = 0.65,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Extract frames uniformly using seek when possible.

    If auto_step is True, the step is chosen so that consecutive extracted
    frames have approximately target_overlap overlap, which helps keep the
    motion between frames small enough for translation-based stitching.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if fps is not None and fps > 0 and video_fps > 0:
        step = max(1, int(round(video_fps / fps)))

    if auto_step and total > 0 and frame_h > 0 and frame_w > 0:
        motion = _estimate_motion_magnitude(video_path)
        # motion is pixels per sample_step frames (we sampled every sample_step).
        # We want consecutive extracted frames to have approximately target_overlap,
        # but we also want a reasonable number of frames for stitching.
        dimension = min(frame_w, frame_h)
        steps = []
        if motion > 0:
            motion_per_frame = motion / 10.0
            desired_step = int(round((1.0 - target_overlap) * dimension / motion_per_frame))
            steps.append(desired_step)
        # Aim for about 50 frames total; short videos should not extract every frame.
        target_frames = 50
        steps.append(max(1, total // target_frames))
        step = max(1, min(min(steps), 20))
        # Ensure at least 30 frames if the video is long enough.
        if total >= 30:
            step = min(step, max(1, total // 30))

    step = max(1, step)

    # For very large steps or very long videos, seek to target frames
    # For typical short videos, sequential reading is faster due to H.264 keyframe decoding
    use_seek = step >= 30 and total > 1000

    frames: FrameList = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    if use_seek:
        indices = list(range(0, total, step))
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((idx, frame))
            if progress_callback and not progress_callback(i + 1, len(indices)):
                break
    else:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append((idx, frame))
            idx += 1
            if progress_callback and not progress_callback(idx, total):
                break

    cap.release()
    return frames


def extract_by_count(
    video_path: str,
    count: int,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Extract approximately N uniformly distributed frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    count = max(1, min(count, total))
    indices = set(np.linspace(0, total - 1, count, dtype=int).tolist())

    frames: FrameList = []

    for idx in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, frame))
        if progress_callback and not progress_callback(len(frames), count):
            break

    cap.release()
    return frames


def _resize_for_analysis(frame: np.ndarray, max_dim: int = 480) -> np.ndarray:
    """Resize a frame to a small size suitable for fast analysis."""
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame
    scale = max_dim / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _compute_ssim(ref: np.ndarray, curr: np.ndarray) -> float:
    """Compute SSIM between two frames."""
    try:
        from skimage.metrics import structural_similarity

        ref_gray = cv2.cvtColor(_resize_for_analysis(ref), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(_resize_for_analysis(curr), cv2.COLOR_BGR2GRAY)
        return float(structural_similarity(ref_gray, curr_gray))
    except Exception:
        return 0.0


def _compute_hist_diff(ref: np.ndarray, curr: np.ndarray) -> float:
    """Compute histogram difference between two frames (0 = identical)."""
    try:
        ref_gray = cv2.cvtColor(_resize_for_analysis(ref), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(_resize_for_analysis(curr), cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([ref_gray], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, 1.0 - correlation)
    except Exception:
        return 0.0


def _compute_laplacian_variance(frame: np.ndarray) -> float:
    """Compute Laplacian variance as a measure of frame sharpness."""
    try:
        gray = cv2.cvtColor(_resize_for_analysis(frame), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def extract_motion_keyframes(
    video_path: str,
    motion_threshold: float = 5.0,
    ssim_threshold: float = 0.95,
    hist_threshold: float = 0.1,
    min_shift: float = 30.0,
    max_keyframes: int = 200,
    sample_step: int = 1,
    blur_threshold: float = 100.0,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Extract keyframes only when the scene has actually moved and is sharp.

    Uses phase correlation to estimate the motion between consecutive frames.
    If the motion is below ``motion_threshold``, the frame is considered
    unchanged and skipped.  A secondary SSIM / histogram check ensures the
    content has truly changed.  Frames with Laplacian variance below
    ``blur_threshold`` are considered blurry and skipped.

    This avoids extracting redundant frames that would produce thin lines or
    duplicate content in the stitched panorama.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    keyframes: FrameList = []
    prev_frame: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_step != 0:
            idx += 1
            continue

        # Filter out blurry frames using Laplacian variance.
        if blur_threshold > 0:
            lap_var = _compute_laplacian_variance(frame)
            if lap_var < blur_threshold:
                idx += 1
                continue

        keep = False
        if prev_frame is None:
            keep = True
        else:
            # Estimate motion using phase correlation on small images.
            gray = cv2.cvtColor(_resize_for_analysis(frame), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                try:
                    (dx, dy), response = cv2.phaseCorrelate(
                        np.float32(prev_gray), np.float32(gray)
                    )
                    motion_mag = np.hypot(dx, dy)
                except Exception:
                    motion_mag = 0.0
                    response = 0.0

                # Only keep the frame if it has moved enough AND the content
                # has actually changed.
                if motion_mag >= motion_threshold and response >= 0.1:
                    # Secondary check: SSIM or histogram difference.
                    if ssim_threshold > 0:
                        ssim = _compute_ssim(prev_frame, frame)
                        if ssim < ssim_threshold:
                            keep = True
                    elif hist_threshold > 0:
                        hist_diff = _compute_hist_diff(prev_frame, frame)
                        if hist_diff > hist_threshold:
                            keep = True
                    else:
                        keep = True

                # Ensure minimum shift to avoid too many keyframes.
                if keep and min_shift > 0 and len(keyframes) > 0:
                    last_idx, last_frame = keyframes[-1]
                    last_gray = cv2.cvtColor(_resize_for_analysis(last_frame), cv2.COLOR_BGR2GRAY)
                    try:
                        (dx2, dy2), response2 = cv2.phaseCorrelate(
                            np.float32(last_gray), np.float32(gray)
                        )
                        if response2 >= 0.1 and np.hypot(dx2, dy2) < min_shift:
                            keep = False
                    except Exception:
                        pass

        if keep:
            keyframes.append((idx, frame))
            prev_frame = frame
            prev_gray = cv2.cvtColor(_resize_for_analysis(frame), cv2.COLOR_BGR2GRAY)

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


def extract_sharpest_frames(
    video_path: str,
    max_frames: int = 8,
    laplacian_threshold: float = 100.0,
    min_time_gap: int = 30,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Extract the sharpest frames from a video using Laplacian variance.

    For each frame, compute the Laplacian variance as a sharpness score.
    Frames with Laplacian variance below ``laplacian_threshold`` are considered
    blurry and skipped entirely.  The remaining frames are sorted by sharpness
    (descending) and the top ``max_frames`` are kept, preserving temporal order.

    ``min_time_gap`` ensures selected frames are at least this many frames
    apart, so the dynamic changes between them are visible.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Read all frames and compute Laplacian variance.
    frame_scores: List[Tuple[int, float, np.ndarray]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Only keep frames that are sharp enough.
        if lap_var >= laplacian_threshold:
            frame_scores.append((idx, lap_var, frame))

        idx += 1
        if progress_callback and not progress_callback(idx, total):
            break

    cap.release()

    if not frame_scores:
        return []

    # Sort by sharpness (descending) and take the top max_frames with time gap.
    frame_scores.sort(key=lambda x: x[1], reverse=True)

    selected = []
    last_idx = -min_time_gap - 1
    for idx, score, frame in frame_scores:
        if idx - last_idx >= min_time_gap:
            selected.append((idx, frame))
            last_idx = idx
            if len(selected) >= max_frames:
                break

    # Sort by frame index to preserve temporal order.
    selected.sort(key=lambda x: x[0])

    return selected


def extract_dynamic_frames(
    video_path: str,
    target_width: int = 1920,
    target_height: int = 1080,
    max_frames: int = 20,
    min_frames: int = 2,
    laplacian_threshold: float = 100.0,
    min_time_gap: int = 0,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> FrameList:
    """Extract the sharpest frames with dynamic count based on target output size.

    For each frame, compute the Laplacian variance as a sharpness score.
    Frames with Laplacian variance below ``laplacian_threshold`` are considered
    blurry and skipped entirely.  The remaining frames are sorted by sharpness
    (descending) and the top frames are kept, preserving temporal order.

    The number of frames is chosen dynamically so that the output image is
    close to the target size without being too long or too wide.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Read all frames and compute Laplacian variance.
    frame_scores: List[Tuple[int, float, np.ndarray]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Only keep frames that are sharp enough.
        if lap_var >= laplacian_threshold:
            frame_scores.append((idx, lap_var, frame))

        idx += 1
        if progress_callback and not progress_callback(idx, total):
            break

    cap.release()

    if not frame_scores:
        return []

    # Sort by sharpness (descending) and take the top max_frames with time gap.
    frame_scores.sort(key=lambda x: x[1], reverse=True)

    # Estimate frame dimensions after edge cropping (10% from each side).
    sample_frame = frame_scores[0][2]
    h, w = sample_frame.shape[:2]
    crop_ratio = 0.1
    frame_w = int(w * (1 - 2 * crop_ratio))
    frame_h = int(h * (1 - 2 * crop_ratio))

    # Compute optimal frame count based on target output size.
    # For horizontal: output_width = frame_w * n, output_height = frame_h
    # For vertical: output_width = frame_w, output_height = frame_h * n
    # We want output to be close to target size without being too long or too wide.
    optimal_h = max(min_frames, min(max_frames, int(target_width / frame_w)))
    optimal_v = max(min_frames, min(max_frames, int(target_height / frame_h)))
    optimal = max(optimal_h, optimal_v)

    selected = []
    last_idx = -min_time_gap - 1
    for idx, score, frame in frame_scores:
        if idx - last_idx >= min_time_gap:
            selected.append((idx, frame))
            last_idx = idx
            if len(selected) >= optimal:
                break

    # Sort by frame index to preserve temporal order.
    selected.sort(key=lambda x: x[0])

    return selected
