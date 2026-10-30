"""Narrow-band seam blending for multi-frame mosaic.

Each pixel comes from exactly one frame (the "owner") based on the ownership
mask.  Only a narrow band near the seam is blended to avoid hard edges.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class NarrowBandBlender:
    """Blend multiple images using narrow-band seam blending.

    Each pixel is assigned to exactly one frame (the "owner") based on the
    ownership mask.  Only a narrow band near the seam is blended to avoid
    hard edges.  Non-boundary regions strictly use a single source frame.
    """

    def __init__(self, seam_blend_width: int = 12):
        self.seam_blend_width = seam_blend_width

    def blend(
        self,
        images: List[np.ndarray],
        ownership_mask: np.ndarray,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """Blend multiple images using narrow-band seam blending.

        Args:
            images: List of images to blend (same size).
            ownership_mask: int32 array where each pixel's value is the frame_id
                of the owner frame (0-indexed). -1 means no owner.
            masks: List of binary masks indicating valid pixels for each image.

        Returns:
            Blended image where each pixel comes from exactly one frame.
        """
        if not images:
            return None

        if len(images) == 1:
            return images[0]

        canvas_h, canvas_w = images[0].shape[:2]
        result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Assign each pixel to its owner frame.
        for i, img in enumerate(images):
            result[ownership_mask == i] = img[ownership_mask == i]

        # Apply narrow-band blending near the seams.
        if self.seam_blend_width > 0:
            result = self._apply_narrow_band_blend(result, images, ownership_mask, masks)

        return result

    def _apply_narrow_band_blend(
        self,
        result: np.ndarray,
        images: List[np.ndarray],
        ownership_mask: np.ndarray,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """Blend only a narrow band around ownership boundaries.

        Single final convex combination (no compounding across pairs):
        alpha peaks at 0.5 exactly on the seam and tapers to 0 at
        ``seam_blend_width`` pixels away.  Pixels where either frame is
        invalid get weight exactly 0 - black invalid pixels never blend.
        """
        canvas_h, canvas_w = result.shape[:2]
        W = max(int(self.seam_blend_width), 1)
        n = len(images)

        # Seam map: pixels where ownership changes between valid neighbors.
        seam = np.zeros((canvas_h, canvas_w), dtype=bool)
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = np.roll(ownership_mask, (dy, dx), axis=(0, 1))
            seam |= (ownership_mask != shifted) & (ownership_mask >= 0) & (shifted >= 0)

        if not np.any(seam):
            return result

        dist_from_seam = cv2.distanceTransform(
            (~seam).astype(np.uint8), cv2.DIST_L2, 5
        )
        band = dist_from_seam < W

        # Dilated ownership regions tell which frames meet near each pixel.
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * W + 1, 2 * W + 1))
        dilated = [
            cv2.dilate((ownership_mask == i).astype(np.uint8), dil_kernel)
            for i in range(n)
        ]

        alpha_max = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        blend_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        base_alpha = np.clip(0.5 * (1.0 - dist_from_seam / W), 0.0, 0.5)

        for i in range(n):
            for j in range(i + 1, n):
                near = (
                    (dilated[i] > 0) & (dilated[j] > 0) & band
                    & (masks[i] > 0) & (masks[j] > 0)
                )
                if not np.any(near):
                    continue
                a = base_alpha * near.astype(np.float32)
                upd = a > alpha_max
                if not np.any(upd):
                    continue
                alpha_max[upd] = a[upd]
                blend_img[upd] = (
                    (images[i].astype(np.float32) + images[j].astype(np.float32)) * 0.5
                )[upd]

        out = result.astype(np.float32)
        a3 = alpha_max[:, :, None]
        out = out * (1.0 - a3) + blend_img * a3
        return np.clip(out, 0, 255).astype(np.uint8)
