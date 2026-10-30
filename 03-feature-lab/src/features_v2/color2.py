"""Color features: multi color-space stats + GCV + YR (yellowing ratio)."""

from typing import Dict

import cv2
import numpy as np

from .geometry import canopy_mask


def extract_color_features(crop: np.ndarray) -> Dict[str, float]:
    """RGB/HSV/LAB/YCbCr channel stats + GCV + YR for one crop."""
    if crop is None or crop.size == 0:
        return {}
    feats: Dict[str, float] = {}

    spaces = {
        "rgb": crop,
        "hsv": cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
        "lab": cv2.cvtColor(crop, cv2.COLOR_BGR2LAB),
        "ycbcr": cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb),
    }
    for prefix, img in spaces.items():
        for i, ch_name in enumerate(("c1", "c2", "c3")):
            ch = img[:, :, i].astype(np.float32)
            feats[f"{prefix}_{ch_name}_mean"] = float(ch.mean())
            feats[f"{prefix}_{ch_name}_std"] = float(ch.std())

    # GCV: green channel coefficient of variation (thesis 4.8 yellowing basis).
    g = crop[:, :, 1].astype(np.float32)
    feats["GCV"] = float(g.std() / (g.mean() + 1e-8))

    # YR: yellowing pixels inside the canopy / canopy pixels.
    # Yellow = hue 15~35 in HSV, restricted to the canopy mask.
    hsv = spaces["hsv"]
    mask = canopy_mask(crop)
    canopy_px = float(np.count_nonzero(mask))
    if canopy_px > 0:
        yellow = (
            (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35)
            & (hsv[:, :, 1] > 40) & (mask > 0)
        )
        feats["YR"] = float(np.count_nonzero(yellow)) / canopy_px
    else:
        feats["YR"] = 0.0

    return feats
