"""Geometric feature extraction for grid cells."""

from typing import Dict

import cv2
import numpy as np

from ..grid_manager import GridCell


def extract_geometry_features(image: np.ndarray, cell: GridCell) -> Dict[str, float]:
    """Extract geometric features from a single grid cell.

    Args:
        image: BGR image as a NumPy array.
        cell: GridCell defining the region to analyze.

    Returns:
        Dictionary of feature names and values.
    """
    crop = image[cell.y1 : cell.y2, cell.x1 : cell.x2]
    if crop.size == 0:
        return {}

    h, w = crop.shape[:2]
    area = h * w
    features: Dict[str, float] = {}

    # Convert to grayscale and threshold using Otsu to separate foreground.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Coverage / occupancy ratio.
    foreground_pixels = int(np.sum(binary == 255))
    features["coverage"] = foreground_pixels / float(area)

    # Contour-based features.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Merge all contours into one big contour for global shape analysis.
        all_contours = np.vstack(contours)

        # Area and perimeter of the largest combined contour.
        contour_area = float(cv2.contourArea(all_contours))
        perimeter = float(cv2.arcLength(all_contours, True))
        features["contour_area"] = contour_area / float(area)
        features["contour_perimeter"] = perimeter / max(w, h)

        # Circularity.
        if perimeter > 0:
            features["circularity"] = 4.0 * np.pi * contour_area / (perimeter**2)
        else:
            features["circularity"] = 0.0

        # Bounding box and derived ratios.
        x, y, bw, bh = cv2.boundingRect(all_contours)
        features["bbox_aspect_ratio"] = max(bw, 1) / max(bh, 1)
        bbox_area = bw * bh
        if bbox_area > 0:
            features["rectangularity"] = contour_area / bbox_area
            features["extent"] = contour_area / bbox_area
        else:
            features["rectangularity"] = 0.0
            features["extent"] = 0.0

        # Equivalent diameter.
        if contour_area > 0:
            features["equivalent_diameter"] = np.sqrt(4.0 * contour_area / np.pi) / max(w, h)
        else:
            features["equivalent_diameter"] = 0.0

        # Centroid of foreground (normalized by cell size).
        moments = cv2.moments(binary)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            features["centroid_x"] = cx / w
            features["centroid_y"] = cy / h
        else:
            features["centroid_x"] = 0.5
            features["centroid_y"] = 0.5

        # Number of connected components.
        features["num_components"] = float(len(contours))
    else:
        for name in [
            "contour_area",
            "contour_perimeter",
            "circularity",
            "bbox_aspect_ratio",
            "rectangularity",
            "extent",
            "equivalent_diameter",
            "centroid_x",
            "centroid_y",
            "num_components",
        ]:
            features[name] = 0.0
        features["centroid_x"] = 0.5
        features["centroid_y"] = 0.5

    # Spatial distribution: compactness / standard deviation of foreground positions.
    y_idx, x_idx = np.where(binary == 255)
    if len(x_idx) > 0:
        features["spatial_std_x"] = float(np.std(x_idx)) / max(w, 1)
        features["spatial_std_y"] = float(np.std(y_idx)) / max(h, 1)
    else:
        features["spatial_std_x"] = 0.0
        features["spatial_std_y"] = 0.0

    return features
