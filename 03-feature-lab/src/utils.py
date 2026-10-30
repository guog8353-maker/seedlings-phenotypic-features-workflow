"""Utility helpers."""

import os
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_images(directory: str) -> List[str]:
    """Collect all image file paths in a directory."""
    image_paths: List[str] = []
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return image_paths

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(str(p) for p in directory_path.glob(f"*{ext}"))
        image_paths.extend(str(p) for p in directory_path.glob(f"*{ext.upper()}"))

    return sorted(image_paths)


def is_image_file(path: str) -> bool:
    """Check whether a file path is an image."""
    return os.path.splitext(path)[-1].lower() in IMAGE_EXTENSIONS
