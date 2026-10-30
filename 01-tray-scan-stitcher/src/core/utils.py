"""Utility helpers."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def save_image(
    path: Union[str, Path],
    image: np.ndarray,
    quality: int = 95,
    png_compression: int = 1,
) -> bool:
    """Save an image, supporting Unicode paths on Windows.

    OpenCV's cv2.imwrite can fail with non-ASCII paths on Windows.
    This function uses cv2.imencode + numpy.tofile as a workaround.

    The image is automatically converted to RGB mode (3 channels) if it has
    an alpha channel, ensuring compatibility with other software.

    Args:
        path: Output file path.
        quality: JPEG quality (0-100).  Higher is better/larger.
        png_compression: PNG compression level (0-9).  Lower is faster and
            produces slightly larger files but the image is still lossless.
    """
    try:
        path = Path(path)
        ext = path.suffix.lower()

        # Ensure RGB mode (3 channels) for compatibility.
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Force a supported format; unknown extensions fall back to PNG.
        if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            path = path.with_suffix(".png")
            ext = ".png"

        if ext in (".jpg", ".jpeg"):
            flag = ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, quality))]
        elif ext == ".png":
            flag = ".png"
            params = [
                cv2.IMWRITE_PNG_COMPRESSION,
                max(0, min(9, png_compression)),
            ]
        else:
            # Preserve the TIFF format selected in the single/batch UI.
            flag = ext
            params = []

        success, encoded = cv2.imencode(flag, image, params)
        if not success:
            return False

        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.tofile(str(path))
        return True
    except Exception:
        return False
