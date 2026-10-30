"""Spatial ownership blending for seamless panorama stitching.

Each pixel comes from exactly one frame (the "owner"), not a blend of multiple
frames.  Only a tiny feather is applied near the seam to avoid hard edges.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class SpatialOwnershipBlender:
    """Blend multiple images using spatial ownership blending.

    Each pixel is assigned to exactly one frame (the "owner") based on the
    seam and mask.  The owner frame provides the pixel value directly.
    Only a tiny feather is applied near the seam to avoid hard edges.
    """

    def __init__(self, feather_width: int = 5):
        self.feather_width = feather_width

    def blend(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        seam_masks: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Blend multiple images using spatial ownership blending.

        Args:
            images: List of images to blend (same size).
            masks: List of binary masks indicating valid pixels for each image.
            seam_masks: Optional list of seam masks for finer control.

        Returns:
            Blended image where each pixel comes from exactly one frame.
        """
        if not images:
            return None

        if len(images) == 1:
            return images[0]

        canvas_h, canvas_w = images[0].shape[:2]
        result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        ownership = np.zeros((canvas_h, canvas_w), dtype=np.int32)

        # Assign each pixel to the frame with the highest mask value.
        for i, (img, mask) in enumerate(zip(images, masks)):
            # Use the mask as the ownership score.
            score = mask.astype(np.float32)
            # Pixels with higher score belong to this frame.
            update = score > ownership.astype(np.float32)
            ownership[update] = i
            result[update] = img[update]

        # Apply a tiny feather near the seams to avoid hard edges.
        if self.feather_width > 0:
            result = self._apply_feather(result, images, ownership, masks)

        return result

    def _apply_feather(
        self,
        result: np.ndarray,
        images: List[np.ndarray],
        ownership: np.ndarray,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """Apply a tiny feather near the seams to avoid hard edges."""
        canvas_h, canvas_w = result.shape[:2]
        feathered = result.copy().astype(np.float32)

        # Find seam pixels (where ownership changes).
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                # Find pixels where ownership is between frame i and frame j.
                seam = ((ownership == i) & (masks[j] > 0)) | (
                    (ownership == j) & (masks[i] > 0)
                )

                if not np.any(seam):
                    continue

                # Apply a tiny feather to the seam region.
                seam_float = seam.astype(np.float32)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.feather_width * 2 + 1, self.feather_width * 2 + 1)
                )
                seam_blur = cv2.filter2D(seam_float, -1, kernel)
                seam_blur = np.clip(seam_blur, 0, 1)

                # Blend the two frames at the seam.
                for c in range(3):
                    feathered[:, :, c] = (
                        feathered[:, :, c] * (1 - seam_blur)
                        + (
                            images[i][:, :, c].astype(np.float32) * seam_blur
                            + images[j][:, :, c].astype(np.float32) * seam_blur
                        )
                        / 2
                    )

        return np.clip(feathered, 0, 255).astype(np.uint8)


class MultiBandBlender:
    """Legacy multi-band blender for low-texture scenes."""

    def __init__(self, num_bands: int = 5):
        self.num_bands = num_bands

    def blend(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        seam_masks: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Blend multiple images using multi-band blending."""
        if not images:
            return None

        if len(images) == 1:
            return images[0]

        # Initialize accumulator.
        canvas_h, canvas_w = images[0].shape[:2]
        result = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for img, mask in zip(images, masks):
            # Ensure mask is float32 and has 3 channels for broadcasting.
            if len(mask.shape) == 2:
                mask_3ch = mask[:, :, np.newaxis].astype(np.float32)
            else:
                mask_3ch = mask.astype(np.float32)

            # Accumulate weighted image.
            result += img.astype(np.float32) * mask_3ch
            weight_sum += mask_3ch.squeeze()

        # Normalize by weight.
        weight_sum = np.maximum(weight_sum, 1e-6)
        result = result / weight_sum[:, :, np.newaxis]

        return np.clip(result, 0, 255).astype(np.uint8)
