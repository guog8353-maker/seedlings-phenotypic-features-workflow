"""Crop-level feature extraction package (v2).

Each crop image -> one flat feature dict.  Groups can be toggled
individually for ablation-friendly exports.
"""

from typing import Dict, List

import numpy as np

from .color2 import extract_color_features
from .geometry import extract_geometry_features
from .texture2 import extract_texture_features
from .vegetation import extract_vegetation_features

GROUPS = {
    "geometry": ("几何特征", extract_geometry_features),
    "texture": ("纹理特征", extract_texture_features),
    "color": ("色彩特征", extract_color_features),
    "vegetation": ("RGB植被指数", extract_vegetation_features),
}


def extract_crop_features(crop: np.ndarray, groups: List[str]) -> Dict[str, float]:
    """Compute the selected feature groups for one crop."""
    feats: Dict[str, float] = {}
    for g in groups:
        fn = GROUPS.get(g, (None, None))[1]
        if fn is not None:
            feats.update(fn(crop))
    return feats
