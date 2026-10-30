"""Texture features: GLCM (7 props x 4 directions) + LBP entropy + Gabor."""

from typing import Dict

import cv2
import numpy as np


def extract_texture_features(crop: np.ndarray) -> Dict[str, float]:
    """GLCM / LBP / Gabor features for one crop."""
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    if crop is None or crop.size == 0:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    feats: Dict[str, float] = {}

    # ---- GLCM (32 levels, distances 1-2, 4 directions, mean ± std) ----
    gray_q = (gray.astype(np.float32) / 255.0 * 31).astype(np.uint8)
    glcm = graycomatrix(
        gray_q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True,
    )
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        vals = graycoprops(glcm, prop)
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"] = float(np.std(vals))
    # Entropy (manual).
    p = glcm.astype(np.float64)
    p_sum = p.sum(axis=(0, 1), keepdims=True)
    p_norm = p / (p_sum + 1e-12)
    ent = -np.sum(np.where(p_norm > 0, p_norm * np.log2(p_norm + 1e-12), 0), axis=(0, 1))
    feats["glcm_entropy_mean"] = float(np.mean(ent))
    feats["glcm_entropy_std"] = float(np.std(ent))

    # ---- LBP ----
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    n_bins = int(lbp.max() + 2)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    feats["lbp_entropy"] = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-8)))
    feats["lbp_mean"] = float(np.mean(lbp))
    feats["lbp_std"] = float(np.std(lbp))

    # ---- Gabor (4 orientations x 2 frequencies) ----
    responses = []
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        for freq in (0.1, 0.2):
            kernel = cv2.getGaborKernel(
                (9, 9), sigma=2.0, theta=theta, lambd=1.0 / freq, gamma=0.5, psi=0
            )
            responses.append(float(np.mean(np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel)))))
    feats["gabor_mean"] = float(np.mean(responses))
    feats["gabor_std"] = float(np.std(responses))

    return feats
