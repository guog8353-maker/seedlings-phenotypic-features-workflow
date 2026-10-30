"""Color feature extraction for grid cells."""

from typing import Dict

import cv2
import numpy as np

from ..grid_manager import GridCell


def extract_color_features(image: np.ndarray, cell: GridCell) -> Dict[str, float]:
    """Extract color features from a single grid cell.

    Args:
        image: BGR image as a NumPy array.
        cell: GridCell defining the region to analyze.

    Returns:
        Dictionary of feature names and values.
    """
    crop = image[cell.y1 : cell.y2, cell.x1 : cell.x2]
    if crop.size == 0:
        return {}

    features: Dict[str, float] = {}

    # RGB features (OpenCV stores as BGR).
    b, g, r = cv2.split(crop)
    features["rgb_r_mean"] = float(np.mean(r))
    features["rgb_g_mean"] = float(np.mean(g))
    features["rgb_b_mean"] = float(np.mean(b))
    features["rgb_r_std"] = float(np.std(r))
    features["rgb_g_std"] = float(np.std(g))
    features["rgb_b_std"] = float(np.std(b))

    # HSV features.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    features["hsv_h_mean"] = float(np.mean(h))
    features["hsv_s_mean"] = float(np.mean(s))
    features["hsv_v_mean"] = float(np.mean(v))
    features["hsv_h_std"] = float(np.std(h))
    features["hsv_s_std"] = float(np.std(s))
    features["hsv_v_std"] = float(np.std(v))

    # Lab features (perceptually uniform color space).
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b2 = cv2.split(lab)
    features["lab_l_mean"] = float(np.mean(l))
    features["lab_a_mean"] = float(np.mean(a))
    features["lab_b_mean"] = float(np.mean(b2))
    features["lab_l_std"] = float(np.std(l))
    features["lab_a_std"] = float(np.std(a))
    features["lab_b_std"] = float(np.std(b2))

    # Color entropy based on HSV 2D histogram (H-S).
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = hist.flatten() / (crop.shape[0] * crop.shape[1] + 1e-8)
    hist_nonzero = hist[hist > 0]
    features["color_entropy"] = float(
        -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-8))
    )

    # Mean brightness / value.
    features["mean_brightness"] = float(np.mean(v))

    # Dominant hue angle (circular mean of hue channel).
    hue_rad = np.deg2rad(h.astype(float) * 2.0)
    features["dominant_hue"] = float(
        np.rad2deg(np.arctan2(np.mean(np.sin(hue_rad)), np.mean(np.cos(hue_rad))))
    )

    return features
