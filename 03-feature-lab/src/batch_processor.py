"""Batch processing and feature export utilities."""

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .features import FeatureExtractor
from .grid_manager import GridManager


class BatchProcessor(QThread):
    """Process multiple images in a background thread."""

    progress = Signal(int, int)  # current, total
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        image_paths: List[str],
        cols: int,
        rows: int,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.image_paths = image_paths
        self.cols = cols
        self.rows = rows
        self.output_dir = output_dir
        self.extractor = FeatureExtractor()

    def run(self) -> None:
        try:
            import pandas as pd

            all_rows: List[dict] = []
            total = len(self.image_paths)

            for idx, path in enumerate(self.image_paths, start=1):
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    continue

                h, w = image.shape[:2]
                grid = GridManager()
                grid.update(w, h, self.cols, self.rows)

                results = self.extractor.extract(image, grid)
                image_name = Path(path).name
                rows = self.extractor.to_dataframe_rows(image_name, grid, results)
                all_rows.extend(rows)

                self.progress.emit(idx, total)

            if not all_rows:
                self.error.emit("没有成功处理任何图片")
                return

            df = pd.DataFrame(all_rows)
            os.makedirs(self.output_dir, exist_ok=True)
            csv_path = os.path.join(self.output_dir, "features.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            # Also save as Excel if openpyxl is available.
            try:
                xlsx_path = os.path.join(self.output_dir, "features.xlsx")
                df.to_excel(xlsx_path, index=False)
            except Exception:
                xlsx_path = None

            self.finished.emit(csv_path)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
