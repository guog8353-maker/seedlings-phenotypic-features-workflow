"""Export cropped grid cells and their labels for machine learning."""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .grid_manager import GridCell, GridManager
from .image_io import imwrite_unicode


def export_dataset(
    image: np.ndarray,
    grid_manager: GridManager,
    labels: Dict[Tuple[int, int], str],
    output_dir: str,
    image_name: str = "image",
    default_label: str = "0",
    include_unlabeled: bool = True,
    export_size: Optional[Tuple[int, int]] = None,
    mask_areas: Optional[Dict[Tuple[int, int], int]] = None,
) -> Optional[str]:
    """Crop grid cells and save them into label-named folders plus a CSV file.

    Args:
        image: Original BGR image as a NumPy array.
        grid_manager: GridManager containing cell coordinates.
        labels: Mapping from (row, col) to label name.
        output_dir: Root output directory.
        image_name: Base name used for cropped image file names.
        default_label: Label assigned to cells without an explicit annotation.
        include_unlabeled: If True, export cells that were never annotated using
            the default label. If False, skip them.
        export_size: If provided, resize all exported crops to this (width, height).
            Smaller images are padded with edge pixels to maintain aspect ratio.
        mask_areas: Optional {(row, col): mask pixel count} recorded into the CSV.

    Returns:
        Path to the generated CSV file, or None if export failed.
    """
    if image is None or image.size == 0:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "labels.csv"
    safe_image_name = Path(image_name).stem

    records: List[Dict[str, str]] = []

    for cell in grid_manager.cells:
        label = labels.get((cell.row, cell.col), default_label if include_unlabeled else None)
        if label is None:
            continue

        # Crop the cell from the original image (bounds-safe).
        img_h, img_w = image.shape[:2]
        x1 = max(0, min(int(cell.x1), img_w))
        y1 = max(0, min(int(cell.y1), img_h))
        x2 = max(0, min(int(cell.x2), img_w))
        y2 = max(0, min(int(cell.y2), img_h))
        if x2 <= x1 or y2 <= y1:
            logging.warning("跳过无效网格 (%d,%d): 越界 [%d,%d,%d,%d]",
                            cell.row, cell.col, cell.x1, cell.y1, cell.x2, cell.y2)
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Resize to uniform size if requested.
        if export_size is not None:
            crop = _resize_with_padding(crop, export_size)

        # Save to output_dir/<label>/<stem>_r<row>_c<col>.jpg (Unicode-safe).
        label_dir = output_path / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{safe_image_name}_r{cell.row:03d}_c{cell.col:03d}.jpg"
        file_path = label_dir / file_name

        if not imwrite_unicode(str(file_path), crop):
            raise IOError(f"无法写出裁剪图片: {file_path}")

        record = {
            "image_path": str(file_path),
            "label": label,
            "row": str(cell.row),
            "col": str(cell.col),
            "original_image": safe_image_name,
        }
        if mask_areas is not None:
            record["mask_area_px"] = str(mask_areas.get((cell.row, cell.col), 0))
        records.append(record)

    # Write CSV (header always, even with zero records).
    fieldnames = ["image_path", "label", "row", "col", "original_image"]
    if mask_areas is not None:
        fieldnames.append("mask_area_px")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if records:
            writer.writerows(records)

    return str(csv_path)


def _resize_with_padding(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image to a fixed size with edge-pixel padding to maintain aspect ratio."""
    target_w, target_h = size
    h, w = image.shape[:2]

    # Compute scale to fit within target size.
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Resize with high-quality interpolation.
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create a padded canvas with edge-pixel background.
    canvas = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
    # Fill with edge pixels.
    if new_w < target_w:
        left_pad = (target_w - new_w) // 2
        right_pad = target_w - new_w - left_pad
        canvas[:, :left_pad] = cv2.resize(resized[:, :1], (left_pad, target_h))
        canvas[:, left_pad : left_pad + new_w] = cv2.resize(resized, (new_w, target_h))
        canvas[:, left_pad + new_w :] = cv2.resize(resized[:, -1:], (right_pad, target_h))
    elif new_h < target_h:
        top_pad = (target_h - new_h) // 2
        bottom_pad = target_h - new_h - top_pad
        canvas[:top_pad, :] = cv2.resize(resized[:1, :], (target_w, top_pad))
        canvas[top_pad : top_pad + new_h, :] = cv2.resize(resized, (target_w, new_h))
        canvas[top_pad + new_h :, :] = cv2.resize(resized[-1:, :], (target_w, bottom_pad))
    else:
        canvas = cv2.resize(resized, (target_w, target_h))

    return canvas
