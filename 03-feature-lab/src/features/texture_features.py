"""Texture feature extraction for grid cells."""

from typing import Dict

import cv2
import numpy as np

from ..grid_manager import GridCell


def extract_texture_features(image: np.ndarray, cell: GridCell) -> Dict[str, float]:
    """Extract texture features from a single grid cell.

    Args:
        image: BGR image as a NumPy array.
        cell: GridCell defining the region to analyze.

    Returns:
        Dictionary of feature names and values.
    """
    crop = image[cell.y1 : cell.y2, cell.x1 : cell.x2]
    if crop.size == 0:
        return {}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    features: Dict[str, float] = {}

    # GLCM (Gray Level Co-occurrence Matrix) features.
    glcm_features = _glcm_features(gray)
    features.update(glcm_features)

    # LBP (Local Binary Pattern) histogram statistics.
    lbp_features = _lbp_features(gray)
    features.update(lbp_features)

    # Gabor filter bank features.
    gabor_features = _gabor_features(gray)
    features.update(gabor_features)

    # Simple gradient / edge density.
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    features["edge_density"] = float(np.mean(grad_magnitude))
    features["edge_std"] = float(np.std(grad_magnitude))

    return features


def _glcm_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute GLCM texture properties at 4 directions and average them."""
    from skimage.feature import graycomatrix, graycoprops

    features: Dict[str, float] = {}
    if gray.size < 8:
        return features

    # Quantize to 32 gray levels for efficiency and robustness.
    gray_q = (gray.astype(np.float32) / 255.0 * 31).astype(np.uint8)
    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(gray_q, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

    prop_names = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for prop_name in prop_names:
        values = graycoprops(glcm, prop_name)
        features[f"glcm_{prop_name}_mean"] = float(np.mean(values))
        features[f"glcm_{prop_name}_std"] = float(np.std(values))

    return features


def _lbp_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute uniform LBP histogram statistics."""
    from skimage.feature import local_binary_pattern

    features: Dict[str, float] = {}
    if gray.size < 8:
        return features

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = int(lbp.max() + 2)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    features["lbp_entropy"] = float(
        -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-8))
    )
    features["lbp_mean"] = float(np.mean(lbp))
    features["lbp_std"] = float(np.std(lbp))
    features["lbp_energy"] = float(np.sum(hist**2))

    return features


def _gabor_features(gray: np.ndarray) -> Dict[str, float]:
    """Compute Gabor filter bank mean responses."""
    features: Dict[str, float] = {}
    if gray.size < 8:
        return features

    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    frequencies = [0.1, 0.2]
    responses = []

    for theta in thetas:
        for freq in frequencies:
            kernel = cv2.getGaborKernel(
                (9, 9), sigma=2.0, theta=theta, lambd=1.0 / freq, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            mean_response = float(np.mean(np.abs(filtered)))
            responses.append(mean_response)
            features[f"gabor_theta{int(np.rad2deg(theta))}_f{freq:.1f}"] = mean_response

    features["gabor_mean"] = float(np.mean(responses))
    features["gabor_std"] = float(np.std(responses))

    return features
