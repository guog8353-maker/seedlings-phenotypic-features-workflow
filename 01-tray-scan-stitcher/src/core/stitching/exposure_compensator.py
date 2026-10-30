"""Exposure and color compensation for seamless blending."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ExposureCompensator:
    """Compensate for exposure and color differences between frames."""

    def __init__(self, method: str = "gain"):
        self.method = method

    def compensate(
        self,
        frames: List[np.ndarray],
        transforms: List[np.ndarray],
        reference_idx: int = 0,
    ) -> List[np.ndarray]:
        """Compensate exposure differences between frames.

        Uses the overlap regions between consecutive frames to estimate
        exposure differences and applies smooth gain compensation.
        """
        if len(frames) <= 1:
            return frames

        # Compute gains for each frame relative to the reference.
        gains = [np.ones(3, dtype=np.float32) for _ in frames]

        for i in range(1, len(frames)):
            # Estimate gain from the overlap region with the previous frame.
            gain = self._estimate_gain(frames[i - 1], frames[i], transforms[i - 1])
            gains[i] = gains[i - 1] * gain

        # Normalize cumulative gains to geometric mean 1.0: without this the
        # per-step gains multiply up over a long chain (1.05^9 ~ 1.55x),
        # blowing out highlights at the far end of the mosaic.
        log_mean = np.mean([np.log(np.maximum(g, 1e-3)) for g in gains], axis=0)
        norm = np.exp(-log_mean).astype(np.float32)
        gains = [g * norm for g in gains]

        # Apply gains to all frames.
        compensated = []
        for frame, gain in zip(frames, gains):
            adjusted = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            compensated.append(adjusted)

        return compensated

    def _estimate_gain(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """Estimate exposure gain from the overlap region between two frames."""
        try:
            # Compute mean brightness of each frame in the overlap region.
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            ref_mean = float(ref_gray.mean())
            curr_mean = float(curr_gray.mean())

            if curr_mean > 1.0:
                scale = ref_mean / curr_mean
                # Limit the gain to a reasonable range to avoid over/under exposure.
                scale = max(0.95, min(1.05, scale))
                return np.array([scale, scale, scale], dtype=np.float32)
            return np.ones(3, dtype=np.float32)
        except Exception:
            return np.ones(3, dtype=np.float32)

    def compensate_blocks(
        self,
        frames: List[np.ndarray],
        transforms: List[np.ndarray],
        block_size: int = 8,
    ) -> List[np.ndarray]:
        """Compensate exposure using block-based gain compensation."""
        if len(frames) <= 1:
            return frames

        # Compute block gains for each frame.
        compensated = [frames[0]]

        for i in range(1, len(frames)):
            # Estimate block gains from the overlap region.
            adjusted = self._compensate_block_frame(
                compensated[i - 1], frames[i], transforms[i - 1], block_size
            )
            compensated.append(adjusted)

        return compensated

    def _compensate_block_frame(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        transform: np.ndarray,
        block_size: int,
    ) -> np.ndarray:
        """Compensate a single frame using block-based gain compensation."""
        h, w = curr.shape[:2]
        result = curr.copy().astype(np.float32)

        # Compute mean brightness of reference and current frames.
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        ref_mean = float(ref_gray.mean())
        curr_mean = float(curr_gray.mean())

        if curr_mean > 1.0:
            scale = ref_mean / curr_mean
            scale = max(0.8, min(1.2, scale))
            result = result * scale

        return np.clip(result, 0, 255).astype(np.uint8)
