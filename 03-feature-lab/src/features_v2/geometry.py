"""Geometric / coverage features for one crop (thesis 表4-3).

Canopy mask: ExG (2G-R-B) enhanced + OTSU threshold, the same operator the
thesis uses for canopy segmentation (式 4-23/4-24).
"""

from typing import Dict

import cv2
import numpy as np


def canopy_mask(crop: np.ndarray) -> np.ndarray:
    """Segment canopy pixels via ExG + OTSU. Returns uint8 mask 0/255."""
    b, g, r = cv2.split(crop.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def extract_geometry_features(crop: np.ndarray) -> Dict[str, float]:
    """CC / GPA / IGR / LGA / GPN for one crop."""
    if crop is None or crop.size == 0:
        return {}

    mask = canopy_mask(crop)
    total = float(mask.size)
    canopy_px = float(np.count_nonzero(mask))

    cc = canopy_px / total
    feats: Dict[str, float] = {
        "CC": cc,
        "GPA": canopy_px,
        "IGR": 1.0 - cc,
    }

    # Gap patches (background connected components).
    gap = (mask == 0).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
        feats["LGA"] = float(areas.max())
        feats["GPN"] = float(np.count_nonzero(areas >= max(4, total * 0.0005)))
    else:
        feats["LGA"] = 0.0
        feats["GPN"] = 0.0

    return feats
