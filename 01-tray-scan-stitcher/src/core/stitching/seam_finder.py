"""Seam finding using GraphCut or Voronoi for optimal stitching."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class SeamFinder:
    """Find the best seam between two overlapping images."""

    def __init__(
        self,
        method: str = "graphcut",
        color_weight: float = 1.0,
        gradient_weight: float = 0.5,
        texture_weight: float = 0.3,
    ):
        self.method = method
        self.color_weight = color_weight
        self.gradient_weight = gradient_weight
        self.texture_weight = texture_weight

    def find_seam(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> np.ndarray:
        """Find the best seam between two overlapping images.

        Returns a seam mask where 1 means img1 and 0 means img2.
        """
        if self.method == "graphcut":
            return self._find_seam_graphcut(img1, img2, mask1, mask2)
        elif self.method == "voronoi":
            return self._find_seam_voronoi(img1, img2, mask1, mask2)
        else:
            return self._find_seam_dp(img1, img2, mask1, mask2)

    def _find_seam_graphcut(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> np.ndarray:
        """Find seam using GraphCut."""
        # Compute the overlap region.
        overlap = (mask1 > 0) & (mask2 > 0)
        if not np.any(overlap):
            return (mask1 > 0).astype(np.uint8)

        # Compute color difference.
        diff_color = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute gradient difference.
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        grad_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        grad_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = (
            np.sqrt(grad_x1**2 + grad_y1**2) + np.sqrt(grad_x2**2 + grad_y2**2)
        ).astype(np.float32)

        # Compute texture difference (local variance).
        texture1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
        texture2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
        texture_diff = np.full_like(gray_diff, abs(texture1 - texture2))

        # Combine differences with weights.
        cost = (
            self.color_weight * gray_diff
            + self.gradient_weight * grad_mag
            + self.texture_weight * texture_diff
        )

        # Use dynamic programming to find the lowest-cost path.
        return self._find_seam_dp(cost, mask1, mask2)

    def _find_seam_voronoi(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> np.ndarray:
        """Find seam using Voronoi diagram."""
        # Compute the overlap region.
        overlap = (mask1 > 0) & (mask2 > 0)
        if not np.any(overlap):
            return (mask1 > 0).astype(np.uint8)

        # Compute the Voronoi diagram based on distance to the seam.
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)

        # Assign each pixel to the frame with the smaller distance.
        seam = (dist1 < dist2).astype(np.uint8)

        # Apply the seam to the overlap region.
        seam_mask = np.zeros_like(mask1)
        seam_mask[overlap] = seam[overlap]

        # Fill in non-overlap regions.
        seam_mask[(mask1 > 0) & (mask2 == 0)] = 1
        seam_mask[(mask2 > 0) & (mask1 == 0)] = 0

        return seam_mask

    def _find_seam_dp(
        self,
        cost: np.ndarray,
        mask1: np.ndarray,
        mask2: np.ndarray,
    ) -> np.ndarray:
        """Find seam using dynamic programming (lowest-cost path)."""
        h, w = cost.shape[:2]
        overlap = (mask1 > 0) & (mask2 > 0)

        if not np.any(overlap):
            return (mask1 > 0).astype(np.uint8)

        # Find the overlap region bounds.
        overlap_cols = np.where(np.any(overlap, axis=0))[0]
        if len(overlap_cols) == 0:
            return (mask1 > 0).astype(np.uint8)

        start_col = overlap_cols[0]
        end_col = overlap_cols[-1]

        # Dynamic programming to find the lowest-cost vertical path.
        dp = np.full((h, w), np.inf, dtype=np.float32)
        parent = np.full((h, w), -1, dtype=np.int32)

        # Initialize first row.
        dp[0, start_col:end_col + 1] = cost[0, start_col:end_col + 1]

        # Fill DP table.
        for y in range(1, h):
            for x in range(start_col, end_col + 1):
                # Consider three possible predecessors.
                candidates = []
                for dx in [-1, 0, 1]:
                    px = x + dx
                    if start_col <= px <= end_col:
                        candidates.append((dp[y - 1, px], px))

                if candidates:
                    best_cost, best_x = min(candidates)
                    dp[y, x] = cost[y, x] + best_cost
                    parent[y, x] = best_x

        # Find the best ending point.
        end_x = start_col + np.argmin(dp[h - 1, start_col:end_col + 1])

        # Backtrack to find the path.
        seam = np.zeros((h, w), dtype=np.uint8)
        x = end_x
        for y in range(h - 1, -1, -1):
            seam[y, x] = 1
            x = parent[y, x]
            if x < 0:
                break

        # Extend seam to full image: left of seam is img1, right is img2.
        seam_mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            seam_x = np.where(seam[y] > 0)[0]
            if len(seam_x) > 0:
                seam_mask[y, : seam_x[0]] = 1

        return seam_mask
