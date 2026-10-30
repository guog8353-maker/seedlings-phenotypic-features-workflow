"""Ghost detection for plant scenes.

Detects areas where the same region has two distinct edges (ghosting) and
reduces the blending weight for those areas.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class GhostDetector:
    """Detect ghosting in overlapping regions."""

    def __init__(self, ghost_threshold: float = 0.3):
        self.ghost_threshold = ghost_threshold

    def detect(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> np.ndarray:
        """Detect ghosting in the overlap region between two images.

        Returns a ghost score map where higher values indicate more ghosting.
        """
        # Compute the overlap region.
        overlap = (mask1 > 0) & (mask2 > 0)
        if not np.any(overlap):
            return np.zeros(img1.shape[:2], dtype=np.float32)

        # Compute gradient magnitude for both images.
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        grad_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)

        grad_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)

        # Compute structural difference.
        # If both images have strong edges in the same area, it's likely ghosting.
        edge_both = (grad_mag1 > 20) & (grad_mag2 > 20)
        edge_diff = np.abs(grad_mag1 - grad_mag2)

        # Ghost score: high when both images have edges but they're different.
        ghost_score = np.zeros(img1.shape[:2], dtype=np.float32)
        ghost_score[overlap] = edge_diff[overlap] / 255.0

        # Normalize to 0-1 range.
        ghost_score = np.clip(ghost_score, 0, 1)

        return ghost_score

    def reduce_ghost_weight(
        self,
        ghost_score: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Reduce the blending weight in ghosted areas.

        Returns a modified mask with reduced weight in ghosted areas.
        """
        # Reduce weight where ghost score is high.
        reduction = 1.0 - ghost_score * (1.0 - self.ghost_threshold)
        reduction = np.clip(reduction, self.ghost_threshold, 1.0)

        # Apply reduction to mask.
        reduced_mask = mask.astype(np.float32) * reduction

        return reduced_mask.astype(np.uint8)
