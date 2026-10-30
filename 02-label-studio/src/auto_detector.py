"""Automatic water-sprouting (吐水) detection for grid cells."""

from typing import Dict, Tuple

import cv2
import numpy as np

from .grid_manager import GridManager


class AutoDetector:
    """Detect 吐水 cells based on brightness/luminance threshold.

    The detector assumes cells with water film reflection are brighter than
    non-吐水 cells. It computes the mean luminance of each cell and compares
    it against a configurable threshold.
    """

    def __init__(self, threshold: int = 128, method: str = "mean_luminance"):
        """Initialize the detector.

        Args:
            threshold: Brightness threshold in [0, 255]. Cells with mean
                luminance greater than this value are classified as '1' (吐水).
            method: Detection method. Currently only 'mean_luminance' is supported.
        """
        self.threshold = max(0, min(255, threshold))
        self.method = method

    def detect(
        self, image: np.ndarray, grid_manager: GridManager
    ) -> Dict[Tuple[int, int], str]:
        """Run auto-detection and return a mapping of (row, col) -> label name.

        Args:
            image: BGR image as a NumPy array.
            grid_manager: GridManager containing the cells to evaluate.

        Returns:
            Dictionary mapping (row, col) to label name ('1' or '0').
        """
        if image is None or image.size == 0 or not grid_manager.cells:
            return {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        labels: Dict[Tuple[int, int], str] = {}

        for cell in grid_manager.cells:
            crop = gray[cell.y1 : cell.y2, cell.x1 : cell.x2]
            if crop.size == 0:
                labels[(cell.row, cell.col)] = "0"
                continue

            mean_luminance = float(np.mean(crop))
            labels[(cell.row, cell.col)] = "1" if mean_luminance > self.threshold else "0"

        return labels

    def summarize(
        self, labels: Dict[Tuple[int, int], str]
    ) -> Tuple[int, int, int]:
        """Return (count_1, count_0, total) for the given labels."""
        count_1 = sum(1 for v in labels.values() if v == "1")
        total = len(labels)
        return count_1, total - count_1, total
