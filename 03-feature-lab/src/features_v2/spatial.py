"""Spatial distribution features per parent image (thesis 4.3.2 / 4.3.5).

The grid_* folders ARE the blocks, so no sub-gridding: each crop's local
canopy coverage CC(i,j) is placed at its (row, col) of the parent tray,
and GCCV / Moran's I / coverage entropy / lacunarity are computed on that
matrix.  The resulting values describe the parent image and are written
into every row belonging to it.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def _morans_i(values: np.ndarray) -> float:
    """Moran's I with queen contiguity on the valid cells of a matrix."""
    mask = np.isfinite(values)
    n = int(mask.sum())
    if n < 4:
        return 0.0
    x = np.where(mask, values, 0.0)
    mean = x[mask].mean()
    dev = x - mean

    # Queen contiguity weights.
    w_sum = 0.0
    num = 0.0
    den = float((dev[mask] ** 2).sum())
    if den <= 1e-12:
        return 0.0
    h, w = values.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted_v = np.roll(np.roll(mask, dy, 0), dx, 1)
            shifted_d = np.roll(np.roll(dev, dy, 0), dx, 1)
            both = mask & shifted_v
            w_sum += float(both.sum())
            num += float((dev * shifted_d)[both].sum())
    if w_sum <= 0:
        return 0.0
    return (n / w_sum) * (num / den)


def spatial_features_from_coverage(cover: np.ndarray) -> Dict[str, float]:
    """GCCV / MI / CE / Lacunarity from a parent's coverage matrix."""
    valid = cover[np.isfinite(cover)]
    if valid.size < 4:
        return {}
    feats: Dict[str, float] = {}

    mean = float(valid.mean())
    std = float(valid.std())
    feats["GCCV"] = std / mean if abs(mean) > 1e-9 else 0.0
    feats["MI"] = _morans_i(cover)

    # Coverage entropy over K=10 bins of normalized coverage.
    if valid.max() > 0:
        norm = valid / valid.max()
        hist, _ = np.histogram(norm, bins=10, range=(0, 1.0), density=True)
        hist = hist[hist > 0]
        feats["CE"] = float(-np.sum(hist * np.log2(hist + 1e-8)))
    else:
        feats["CE"] = 0.0

    # Lacunarity (thesis 式4-15 simplified): (std/mean)^2 + 1.
    feats["LAC"] = (std / mean) ** 2 + 1.0 if abs(mean) > 1e-9 else 1.0

    return feats
