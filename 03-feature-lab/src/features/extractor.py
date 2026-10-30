"""Unified feature extractor for image grid cells."""

from typing import Dict, List, Tuple

import numpy as np

from ..grid_manager import GridCell, GridManager
from .color_features import extract_color_features
from .geometry_features import extract_geometry_features
from .texture_features import extract_texture_features


class FeatureExtractor:
    """Extract all features for each cell in a grid."""

    def __init__(self):
        self.feature_names: List[str] = []

    def extract(
        self, image: np.ndarray, grid_manager: GridManager
    ) -> List[Dict[str, float]]:
        """Extract features for every cell in the grid.

        Args:
            image: BGR image as a NumPy array.
            grid_manager: GridManager with cell coordinates.

        Returns:
            List of feature dictionaries, one per cell, in the same order as
            grid_manager.cells.
        """
        results: List[Dict[str, float]] = []
        for cell in grid_manager.cells:
            features = self.extract_cell(image, cell)
            results.append(features)

        # Cache feature names from the first non-empty result.
        for result in results:
            if result:
                self.feature_names = list(result.keys())
                break

        return results

    def extract_cell(
        self, image: np.ndarray, cell: GridCell
    ) -> Dict[str, float]:
        """Extract features for a single cell."""
        features: Dict[str, float] = {}
        features.update(extract_color_features(image, cell))
        features.update(extract_texture_features(image, cell))
        features.update(extract_geometry_features(image, cell))
        features["row"] = float(cell.row)
        features["col"] = float(cell.col)
        return features

    def to_dataframe_rows(
        self,
        image_name: str,
        grid_manager: GridManager,
        results: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """Convert extraction results into flat rows for pandas / CSV export."""
        rows: List[Dict[str, float]] = []
        for cell, result in zip(grid_manager.cells, results):
            row = {"image_name": image_name, "row": cell.row, "col": cell.col}
            row.update(result)
            rows.append(row)
        return rows
