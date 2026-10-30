"""Unified keyframe extraction: single entry point for UI preview and stitching.

Both the "抽取帧" button and the stitching pipeline MUST use
``extract_keyframes`` so their frame counts can never drift apart.
"""

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import StitchConfig


FrameList = List[Tuple[int, np.ndarray]]


def _resize_for_analysis(frame: np.ndarray, max_dim: int = 480) -> np.ndarray:
    """Resize a frame to a small size suitable for fast analysis."""
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame
    scale = max_dim / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _compute_laplacian_variance(frame: np.ndarray) -> float:
    """Compute Laplacian variance as a measure of frame sharpness."""
    try:
        gray = cv2.cvtColor(_resize_for_analysis(frame), cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _compute_brightness(frame: np.ndarray) -> float:
    """Compute mean brightness of a frame."""
    try:
        gray = cv2.cvtColor(_resize_for_analysis(frame), cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    except Exception:
        return 0.0


def extract_keyframes(
    video_path: str,
    config: StitchConfig,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    stats: Optional[dict] = None,
    keep_candidates: bool = False,
) -> FrameList:
    """THE single keyframe extraction entry point (UI preview and stitching).

    Two-stage scheme:

    Stage A (candidate sampling)
        Uniformly sample ``raw = min(total, max(36, target_keyframes * 3))``
        frames from the video, then apply the quality filter (blur +
        brightness).  This is the candidate pool.

    Stage B (final selection)
        Pick ``target_keyframes`` frames uniformly in time from the candidate
        pool.  If the pool holds fewer frames than the target, keep all of
        them and then backfill from the rest of the video (sharpness-first,
        well spread in time) instead of silently accepting a tiny count.

    Every stage prints its before/after count and, when ``stats`` is given,
    records per-frame keep/reject decisions so the exact frame drop point is
    always visible.
    """
    target = config.target_keyframes
    print(f"[EXTRACTOR CONFIG] target_keyframes = {target} | "
          f"blur_threshold = {config.blur_threshold:.1f} | "
          f"max_keyframes = {config.max_keyframes}", flush=True)

    if stats is not None:
        stats.clear()
        stats.update({
            "extractor_target": target,
            "total_video_frames": 0,
            "fps": 0.0,
            "duration": 0.0,
            "raw_sampled": 0,
            "quality_before": 0,
            "quality_passed": 0,
            "duplicate_before": 0,
            "duplicate_after": 0,
            "motion_before": 0,
            "motion_after": 0,
            "selected_frames": 0,
            "backfilled": 0,
            "frame_records": [],
        })

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total / fps) if fps > 0 else 0.0
    if total <= 0:
        cap.release()
        return []

    print(f"[VIDEO] total_video_frames = {total} | fps = {fps:.2f} | "
          f"duration = {duration:.2f}s", flush=True)
    if stats is not None:
        stats["total_video_frames"] = total
        stats["fps"] = fps
        stats["duration"] = duration

    # ---------------- Stage A: candidate sampling ----------------
    raw_count = min(total, max(36, target * 3))
    sample_ids = set(np.linspace(0, total - 1, raw_count, dtype=int).tolist())
    print(f"[SAMPLING] raw_sampled_frames = {len(sample_ids)}", flush=True)
    if stats is not None:
        stats["raw_sampled"] = len(sample_ids)

    candidates: List[Tuple[int, float, np.ndarray]] = []  # (idx, sharpness, frame)
    quality_before = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_ids:
            quality_before += 1
            lap_var = _compute_laplacian_variance(frame)
            if lap_var < config.blur_threshold:
                if stats is not None:
                    stats["frame_records"].append({
                        "frame_id": idx, "decision": "REJECT",
                        "stage": "quality_filter", "reason": "blur",
                        "value": round(lap_var, 2),
                        "threshold": config.blur_threshold,
                    })
            else:
                brightness = _compute_brightness(frame)
                if brightness < config.min_brightness or brightness > config.max_brightness:
                    if stats is not None:
                        stats["frame_records"].append({
                            "frame_id": idx, "decision": "REJECT",
                            "stage": "quality_filter", "reason": "brightness",
                            "value": round(brightness, 2),
                            "threshold": [config.min_brightness, config.max_brightness],
                        })
                else:
                    candidates.append((idx, lap_var, frame))
        idx += 1
        if progress_callback and not progress_callback(idx, total):
            break
    cap.release()

    print(f"[QUALITY FILTER] before = {quality_before} after = {len(candidates)}", flush=True)
    # No duplicate / motion filters at extraction stage: report them honestly.
    print(f"[DUPLICATE FILTER] before = {len(candidates)} after = {len(candidates)} (none)", flush=True)
    print(f"[MOTION FILTER] before = {len(candidates)} after = {len(candidates)} (none)", flush=True)
    if stats is not None:
        stats["quality_before"] = quality_before
        stats["quality_passed"] = len(candidates)
        stats["duplicate_before"] = len(candidates)
        stats["duplicate_after"] = len(candidates)
        stats["motion_before"] = len(candidates)
        stats["motion_after"] = len(candidates)
        if keep_candidates:
            stats["candidate_frames_data"] = [(i, f) for i, _, f in candidates]

    if not candidates:
        print("[FINAL EXTRACTION] final_extracted_frames = 0", flush=True)
        return []

    # ---------------- Stage B: final selection ----------------
    candidates.sort(key=lambda x: x[0])
    if len(candidates) > target:
        pick = np.linspace(0, len(candidates) - 1, target, dtype=int)
        selected = [candidates[i] for i in pick]
    else:
        selected = list(candidates)
    selected_ids = {i for i, _, _ in selected}

    if stats is not None:
        for c_idx, lap_var, _ in candidates:
            stats["frame_records"].append({
                "frame_id": c_idx,
                "decision": "KEEP" if c_idx in selected_ids else "REJECT",
                "stage": "final_selection",
                "reason": "selected" if c_idx in selected_ids else "not_selected_by_uniform_sampling",
                "value": round(lap_var, 2),
                "threshold": target,
            })

    # ---------------- Backfill when selection falls short ----------------
    backfilled = 0
    if len(selected) < target and total > len(sample_ids):
        missing = target - len(selected)
        pool_size = min(total - len(sample_ids), missing * 3)
        if pool_size > 0:
            all_ids = np.linspace(0, total - 1, min(total, pool_size * 2), dtype=int)
            fill_ids = [int(i) for i in all_ids if int(i) not in sample_ids][:pool_size]
            cap = cv2.VideoCapture(video_path)
            fills: List[Tuple[int, float, np.ndarray]] = []
            for f_idx in fill_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                lap_var = _compute_laplacian_variance(frame)
                # Relaxed gate for backfill: 60% of the blur threshold.
                if lap_var >= config.blur_threshold * 0.6:
                    fills.append((f_idx, lap_var, frame))
            cap.release()
            # Sharpness-first backfill.
            fills.sort(key=lambda x: x[1], reverse=True)
            for f_idx, lap_var, frame in fills[:missing]:
                selected.append((f_idx, lap_var, frame))
                backfilled += 1
                if stats is not None:
                    stats["frame_records"].append({
                        "frame_id": f_idx, "decision": "KEEP",
                        "stage": "backfill", "reason": "backfilled_to_reach_target",
                        "value": round(lap_var, 2),
                        "threshold": round(config.blur_threshold * 0.6, 2),
                    })

    selected.sort(key=lambda x: x[0])
    print(f"[FINAL EXTRACTION] final_extracted_frames = {len(selected)} "
          f"(backfilled = {backfilled})", flush=True)
    if stats is not None:
        stats["selected_frames"] = len(selected)
        stats["backfilled"] = backfilled

    return [(i, frame) for i, _, frame in selected]


def select_keyframes(
    video_path: str,
    blur_threshold: float = 100.0,
    min_brightness: float = 30.0,
    max_brightness: float = 225.0,
    target_keyframes: int = 12,
    maximum_keyframes: int = 24,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
    stats: Optional[dict] = None,
) -> FrameList:
    """Backward-compatible wrapper around :func:`extract_keyframes`."""
    cfg = StitchConfig(
        target_keyframes=target_keyframes,
        max_keyframes=maximum_keyframes,
        blur_threshold=blur_threshold,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
    )
    return extract_keyframes(
        video_path, cfg,
        progress_callback=progress_callback,
        stats=stats,
    )
