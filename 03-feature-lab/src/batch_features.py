"""Threaded batch feature extraction for a folder of crop images."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal

from .features_v2 import GROUPS, extract_crop_features
from .features_v2.geometry import canopy_mask
from .features_v2.spatial import spatial_features_from_coverage
from .folder_loader import CropItem, scan_folder
from .image_io import imread_unicode


META_COLUMNS = ["image_path", "original_image", "row", "col", "grid_strategy", "label"]


class FeatureBatchWorker(QThread):
    """Compute features for every crop in a folder (background thread)."""

    progress = Signal(int, int)
    finished_signal = Signal(list)
    error = Signal(str)

    def __init__(self, folder: str, groups: List[str], parent=None):
        super().__init__(parent)
        self.folder = folder
        self.groups = groups
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            items = scan_folder(self.folder)
            if not items:
                self.error.emit("该文件夹下没有找到图片")
                return

            rows: List[Dict] = []
            cc_map: Dict[int, float] = {}
            need_cc = "spatial" in self.groups

            total = len(items)
            for i, item in enumerate(items, 1):
                if self._cancelled:
                    self.error.emit("已取消")
                    return
                img = imread_unicode(str(item.path))
                row: Dict = {
                    "image_path": str(item.path),
                    "original_image": item.original_image,
                    "row": item.row if item.row is not None else "",
                    "col": item.col if item.col is not None else "",
                    "grid_strategy": item.grid_strategy,
                    "label": item.label,
                }
                if img is not None and img.size > 0:
                    if img.ndim == 3 and img.shape[2] == 4:
                        import cv2
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    row.update(extract_crop_features(img, self.groups))
                    if need_cc:
                        mask = canopy_mask(img)
                        cc_map[i - 1] = float(np.count_nonzero(mask)) / float(mask.size)
                rows.append(row)

                if i % 20 == 0 or i == total:
                    self.progress.emit(i, total)

            # Spatial aggregation: per (original_image, grid_strategy) parent.
            if need_cc:
                self._fill_spatial(items, rows, cc_map)

            self.finished_signal.emit(rows)
        except Exception as e:
            self.error.emit(f"批量计算失败: {type(e).__name__}: {e}")

    def _fill_spatial(
        self, items: List[CropItem], rows: List[Dict], cc_map: Dict[int, float]
    ) -> None:
        """Compute GCCV/MI/CE/LAC per parent and fill into all its rows."""
        parents: Dict[Tuple[str, str], List[int]] = {}
        for idx, item in enumerate(items):
            parents.setdefault((item.original_image, item.grid_strategy), []).append(idx)

        for (orig, grid), idxs in parents.items():
            coords = [(items[i].row, items[i].col) for i in idxs]
            if any(r is None or c is None for r, c in coords) or len(idxs) < 4:
                continue
            max_r = max(r for r, _ in coords)
            max_c = max(c for _, c in coords)
            cover = np.full((max_r + 1, max_c + 1), np.nan)
            for i in idxs:
                if i in cc_map:
                    cover[items[i].row, items[i].col] = cc_map[i]
            feats = spatial_features_from_coverage(cover)
            for i in idxs:
                rows[i].update(feats)


def write_csv(rows: List[Dict], path: str) -> None:
    """Write rows to CSV with stable column order (meta first, then features)."""
    import csv

    if not rows:
        return
    feat_cols: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in META_COLUMNS and k not in seen:
                seen.add(k)
                feat_cols.append(k)
    fieldnames = META_COLUMNS + feat_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_features_meta(path: str, groups: List[str]) -> None:
    """Describe every exported feature (for ablation / interpretability)."""
    meta = {
        "meta_columns": {
            "image_path": "子块图像路径",
            "original_image": "母图名称（解析自文件名 <原名>_r行_c列）",
            "row": "子块在母图中的行",
            "col": "子块在母图中的列",
            "grid_strategy": "分块策略（grid_列x行）",
            "label": "吐水判定（0=未吐水，1=吐水，来自父文件夹名）",
        },
        "groups": {
            "geometry": "几何特征: CC冠层覆盖率/GPA绿色投影面积/IGR空缺比例/LGA最大空缺斑块/GPN空缺斑块数",
            "texture": "纹理特征: GLCM七项(均值±std)+熵, LBP熵/均值/方差, Gabor均值±std",
            "color": "色彩特征: RGB/HSV/LAB/YCbCr各通道mean+std, GCV, YR黄化像素比例",
            "vegetation": "RGB植被指数: 21个指数×(mean/std/range)，公式见论文表4-2",
            "spatial": "空间分布特征(母图级): GCCV变异系数/MI(Moran's I)/CE覆盖熵/LAC空缺度",
        },
        "enabled_groups": groups,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
