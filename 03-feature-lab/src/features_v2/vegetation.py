"""21 RGB vegetation indices (thesis 表4-2), each with mean/std/range."""

from typing import Dict

import numpy as np

_EPS = 1e-8


def _index_maps(crop: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute every vegetation index as a per-pixel float map."""
    b = crop[:, :, 0].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    r = crop[:, :, 2].astype(np.float32)
    s = r + g + b + _EPS
    rn, gn, bn = r / s, g / s, b / s

    exg = 2.0 * g - r - b
    exr = 1.4 * r - g
    exgr = exg - exr
    cive = 0.441 * rn - 0.811 * gn + 0.385 * bn + 18.787
    veg = g / (np.power(np.abs(r) + _EPS, 0.667) * np.power(np.abs(b) + _EPS, 0.333))

    return {
        "ExG": exg,
        "TGI": -0.5 * (190.0 * (r - g) - 120.0 * (r - b)),
        "GBDI": g - b,
        "MExG": 1.262 * g - 0.884 * r - 0.311 * b,
        "ExR": exr,
        "ExGR": exgr,
        "VEG": veg,
        "VDVI": (2.0 * g - r - b) / (2.0 * g + r + b + _EPS),
        "CIVE": cive,
        "NGRDI": (g - r) / (g + r + _EPS),
        "RGRI": r / (g + _EPS),
        "VARI": (g - r) / (g + r - b + _EPS),
        "MGRVI": (g * g - r * r) / (g * g + r * r + _EPS),
        "EGRBDI": (4.0 * g * g - b * r) / (4.0 * g * g + b * r + _EPS),
        "RGBRI": (g + r) / (b + _EPS),
        "E-NGBDI": (g - b) / (g + b + 0.5) * 1.5,
        "ExB": 1.4 * b - g,
        "IKAW": (r - b) / (r + b + _EPS),
        "COM1": 0.25 * exg + 0.30 * exgr + 0.33 * cive + 0.12 * veg,
        "COM2": 0.36 * exg + 0.47 * cive + 0.17 * veg,
        "GBRI": g / (b + _EPS),
    }


def extract_vegetation_features(crop: np.ndarray) -> Dict[str, float]:
    """mean / std / range of all 21 RGB vegetation indices for one crop."""
    feats: Dict[str, float] = {}
    for name, m in _index_maps(crop).items():
        m = m[np.isfinite(m)]
        if m.size == 0:
            continue
        feats[f"veg_{name}_mean"] = float(m.mean())
        feats[f"veg_{name}_std"] = float(m.std())
        feats[f"veg_{name}_range"] = float(m.max() - m.min())
    return feats
