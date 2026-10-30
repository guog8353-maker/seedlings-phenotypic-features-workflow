"""Automatic cropping of the final panorama."""

from typing import Optional, Tuple

import cv2
import numpy as np


class Cropper:
    """Automatically crop black borders from the final panorama."""

    def __init__(self, mode: str = "rect"):
        """Initialize with crop mode.

        Args:
            mode: "rect" for maximum inscribed rectangle, "content" for all content.
        """
        self.mode = mode

    def crop(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Crop black borders from the panorama.

        Args:
            image: The panorama image.
            mask: Optional binary mask indicating valid pixels. If None, uses
                non-black pixels as the mask.

        Returns:
            Cropped image.
        """
        if image is None or image.size == 0:
            return image

        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        if self.mode == "rect":
            rect = self.find_crop_rect(mask)
            if rect is None:
                return image
            x, y, w, h = rect
            return image[y : y + h, x : x + w]
        else:
            return self._crop_content(image, mask)

    def find_crop_rect(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Largest all-valid rectangle inside the main region (RECTANGULAR_CROP).

        Steps: close small holes -> keep the largest connected component ->
        compute the maximum inscribed axis-aligned rectangle (histogram
        method).  The output contains no black holes, like a phone panorama.
        """
        if mask is None or mask.size == 0:
            return None

        mask_u8 = (mask > 0).astype(np.uint8)

        # Close small holes / gaps inside the main content.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        # Keep only the largest connected component.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num <= 1:
            return None
        # stats[0] is the background; find the largest foreground component.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_mask = (labels == largest).astype(np.uint8)

        return self._max_inscribed_rect(largest_mask)

    @staticmethod
    def _max_inscribed_rect(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Maximum-area axis-aligned rectangle fully inside a binary mask.

        Classic "largest rectangle in histogram" sweep over rows.
        The mask is downscaled first for speed; the result is scaled back.
        Returns (x, y, w, h) or None.
        """
        h0, w0 = mask.shape[:2]
        scale = 1.0
        m = mask
        max_dim = max(h0, w0)
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            m = cv2.resize(
                mask,
                (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                interpolation=cv2.INTER_NEAREST,
            )

        h, w = m.shape[:2]
        heights = np.zeros(w, dtype=np.int32)
        best_area = 0
        best_rect: Optional[Tuple[int, int, int, int]] = None

        for y in range(h):
            row = m[y]
            for x in range(w):
                heights[x] = heights[x] + 1 if row[x] > 0 else 0

            # Largest rectangle in the histogram ``heights`` (monotone stack).
            stack: List[int] = []
            for x in range(w + 1):
                cur = int(heights[x]) if x < w else 0
                while stack and int(heights[stack[-1]]) > cur:
                    top = stack.pop()
                    rect_h = int(heights[top])
                    left = stack[-1] + 1 if stack else 0
                    rect_w = x - left
                    area = rect_h * rect_w
                    if area > best_area:
                        best_area = area
                        best_rect = (left, y - rect_h + 1, rect_w, rect_h)
                if x < w:
                    stack.append(x)

        if best_rect is None or best_area <= 0:
            return None

        # Scale back to original mask coordinates.
        inv = 1.0 / scale
        x, y, w_r, h_r = best_rect
        return (
            int(x * inv),
            int(y * inv),
            min(int(w_r * inv), w0 - int(x * inv)),
            min(int(h_r * inv), h0 - int(y * inv)),
        )

    def _crop_content(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop to include all content (no black borders)."""
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w]
