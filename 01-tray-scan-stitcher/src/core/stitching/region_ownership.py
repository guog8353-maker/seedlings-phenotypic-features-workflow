"""Quality-based region ownership for multi-frame mosaic."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class RegionOwnership:
    """Compute quality-based region ownership for multi-frame mosaic."""

    def __init__(
        self,
        w1: float = 0.3,  # sharpness
        w2: float = 0.1,  # exposure_quality
        w3: float = 0.2,  # distance_to_image_center
        w4: float = 0.2,  # geometric_confidence
        w5: float = 0.1,  # seam_cost
        w6: float = 0.2,  # reprojection_error
        w7: float = 0.1,  # motion_blur
        w8: float = 0.2,  # parallax_conflict
    ):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7
        self.w8 = w8

    def compute_ownership(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        transforms: List[np.ndarray],
        match_infos: List[dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute quality-based region ownership for multi-frame mosaic.

        Returns:
            - ownership_mask: int32 array where each pixel's value is the frame_id
              of the owner frame (0-indexed). -1 means no owner.
            - source_map: uint8 array visualizing the ownership (each frame gets
              a different color).
        """
        if not frames:
            return None, None

        canvas_h, canvas_w = frames[0].shape[:2]
        n = len(frames)

        # Initialize quality maps for each frame.
        quality_maps = []

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            # Compute per-pixel sharpness (Laplacian variance in local windows).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness_map = self._compute_local_sharpness(gray)

            # Compute per-pixel exposure quality (brightness close to optimal).
            optimal_brightness = 128.0
            exposure_map = 1.0 - np.abs(gray.astype(np.float32) - optimal_brightness) / 128.0

            # Compute distance to image center.
            center_x, center_y = canvas_w // 2, canvas_h // 2
            y_coords, x_coords = np.mgrid[0:canvas_h, 0:canvas_w]
            dist_to_center = np.sqrt(
                (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
            )
            max_dist = np.sqrt(center_x**2 + center_y**2)
            dist_map = 1.0 - (dist_to_center / max_dist)

            # Compute geometric confidence (from match info).
            geometric_confidence = 1.0
            if i > 0 and i - 1 < len(match_infos):
                info = match_infos[i - 1]
                geometric_confidence = info.get("inlier_ratio", 1.0)
            confidence_map = np.full((canvas_h, canvas_w), geometric_confidence, dtype=np.float32)

            # Compute seam cost (from mask edges).
            mask_uint8 = (mask > 0).astype(np.uint8)
            dist_to_edge = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            max_edge_dist = dist_to_edge.max()
            if max_edge_dist > 0:
                seam_cost_map = dist_to_edge / max_edge_dist
            else:
                seam_cost_map = np.ones((canvas_h, canvas_w), dtype=np.float32)

            # Compute reprojection error (from match info).
            reprojection_error = 0.0
            if i > 0 and i - 1 < len(match_infos):
                info = match_infos[i - 1]
                reprojection_error = info.get("reprojection_error", 0.0)
            error_map = np.full((canvas_h, canvas_w), 1.0 - min(reprojection_error / 10.0, 1.0), dtype=np.float32)

            # Compute motion blur (from sharpness).
            motion_blur = 1.0 - np.clip(sharpness_map / 200.0, 0, 1)
            blur_map = 1.0 - motion_blur

            # Compute parallax conflict (from gradient difference with neighbors).
            parallax_conflict = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            if i > 0:
                # Compare with previous frame.
                prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
                grad_diff = cv2.absdiff(gray, prev_gray)
                parallax_conflict = grad_diff.astype(np.float32) / 255.0

            # Combine all quality factors.
            quality_map = (
                self.w1 * sharpness_map
                + self.w2 * exposure_map
                + self.w3 * dist_map
                + self.w4 * confidence_map
                + self.w5 * seam_cost_map
                - self.w6 * (1.0 - error_map)
                - self.w7 * (1.0 - blur_map)
                - self.w8 * (1.0 - parallax_conflict)
            )

            # Normalize to 0-1 range.
            quality_map = np.clip(quality_map, 0, 1)

            # Apply mask: only consider valid pixels.
            quality_map = quality_map * (mask > 0).astype(np.float32)

            quality_maps.append(quality_map)

        # Compute ownership: each pixel belongs to the frame with the highest quality.
        ownership_mask = np.full((canvas_h, canvas_w), -1, dtype=np.int32)
        max_quality = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for i, quality_map in enumerate(quality_maps):
            update = quality_map > max_quality
            ownership_mask[update] = i
            max_quality[update] = quality_map[update]

        # Smooth ownership: per-pixel flicker between frames turns into visible
        # grain, so reassign each pixel by local majority vote and remove tiny
        # fragments.  Every pixel still comes from exactly one raw frame.
        ownership_mask = self._smooth_ownership(ownership_mask, n)

        # Generate source map visualization.
        source_map = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        for i in range(n):
            color = colors[i % len(colors)]
            source_map[ownership_mask == i] = color

        return ownership_mask, source_map

    def _smooth_ownership(
        self,
        ownership_mask: np.ndarray,
        n_frames: int,
        vote_kernel: int = 31,
        min_fragment_ratio: float = 0.001,
    ) -> np.ndarray:
        """Majority-vote smoothing of the ownership map + fragment removal.

        1. Blur each frame's vote map and re-assign each pixel to the locally
           dominant frame (kills per-pixel flicker = full-screen grain).
        2. Connected components smaller than ``min_fragment_ratio`` of the
           canvas are reassigned to the runner-up frame.
        Pixels with no votes at all stay -1 (no owner).
        """
        canvas_h, canvas_w = ownership_mask.shape[:2]

        best1_val = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
        best2_val = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
        best1_idx = np.full((canvas_h, canvas_w), -1, dtype=np.int32)
        best2_idx = np.full((canvas_h, canvas_w), -1, dtype=np.int32)
        total_vote = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for i in range(n_frames):
            vote = cv2.blur(
                (ownership_mask == i).astype(np.float32), (vote_kernel, vote_kernel)
            )
            total_vote += vote

            # Update runner-up first, then the leader.
            second_better = (vote > best2_val) & (vote <= best1_val)
            best2_val[second_better] = vote[second_better]
            best2_idx[second_better] = i

            better = vote > best1_val
            # Demote the old leader to runner-up where it is still better.
            demote = better & (best1_val > best2_val)
            best2_val[demote] = best1_val[demote]
            best2_idx[demote] = best1_idx[demote]
            best1_val[better] = vote[better]
            best1_idx[better] = i

        smoothed = np.where(total_vote > 1e-3, best1_idx, -1).astype(np.int32)

        # Remove tiny fragments: reassign them to the runner-up frame, but
        # only where a runner-up actually exists (never write -1 back).
        min_area = max(64, int(canvas_h * canvas_w * min_fragment_ratio))
        for i in range(n_frames):
            comp = (smoothed == i).astype(np.uint8)
            if not comp.any():
                continue
            num, labels, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=8)
            for c in range(1, num):
                if stats[c, cv2.CC_STAT_AREA] < min_area:
                    frag = labels == c
                    reassign = frag & (best2_idx >= 0)
                    smoothed[reassign] = best2_idx[reassign]

        return smoothed

    def _compute_local_sharpness(self, gray: np.ndarray, window_size: int = 15) -> np.ndarray:
        """Compute local sharpness using Laplacian variance in sliding windows."""
        h, w = gray.shape[:2]
        sharpness_map = np.zeros((h, w), dtype=np.float32)

        # Compute Laplacian.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Compute local variance using a box filter.
        laplacian_sq = laplacian ** 2
        local_mean = cv2.boxFilter(laplacian_sq, -1, (window_size, window_size))
        local_mean_sq = cv2.boxFilter(laplacian, -1, (window_size, window_size)) ** 2
        local_var = local_mean - local_mean_sq

        # Normalize to 0-1 range.
        if local_var.max() > 0:
            sharpness_map = local_var / local_var.max()

        return np.clip(sharpness_map, 0, 1)
