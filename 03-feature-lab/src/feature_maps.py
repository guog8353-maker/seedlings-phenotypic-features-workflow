"""Pixel-level feature response maps for large (stitched) images.

All maps are computed at ORIGINAL resolution.  Neighborhood operations with
large kernels automatically run tile-by-tile and stitch back, so 4K/8K+
panoramas keep droplet-level detail without blowing up memory.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
@dataclass
class FeatureParams:
    """Tunable parameters exposed as UI sliders."""

    bright_delta: int = 25          # local-brightness threshold (V above bg)
    gaussian_sigma: float = 12.0    # background estimation sigma
    blob_min_area: int = 12         # min blob area in px
    blob_max_area: int = 4000       # max blob area in px
    min_circularity: float = 0.45   # blob circularity floor
    highlight_v: int = 200          # specular V threshold
    # Fusion weights (water-drop probability / guttation enhancement).
    w_brightness: float = 0.25
    w_contrast: float = 0.25
    w_blob: float = 0.25
    w_color: float = 0.25


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize a float map to [0, 1] robustly."""
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def compute_tiled(
    image: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    tile: int = 2048,
    margin: int = 64,
) -> np.ndarray:
    """Run a neighborhood-based map tile-by-tile and stitch the result.

    Each output tile is computed on a margin-expanded crop so border
    artifacts of blur/morphology never appear inside the kept region.
    Pixel-wise maps should NOT use this - just compute directly.
    """
    h, w = image.shape[:2]
    if max(h, w) <= tile:
        return func(image)

    out = np.zeros((h, w), dtype=np.float32)
    step = tile
    for y1 in range(0, h, step):
        for x1 in range(0, w, step):
            y2 = min(y1 + step, h)
            x2 = min(x1 + step, w)
            ey1 = max(y1 - margin, 0)
            ey2 = min(y2 + margin, h)
            ex1 = max(x1 - margin, 0)
            ex2 = min(x2 + margin, w)
            resp = func(image[ey1:ey2, ex1:ex2])
            if resp.ndim == 3:
                resp = cv2.cvtColor(resp, cv2.COLOR_BGR2GRAY)
            core = resp[y1 - ey1 : y2 - ey1, x1 - ex1 : x2 - ex1]
            out[y1:y2, x1:x2] = core.astype(np.float32)
    return out


def _to8(m: np.ndarray) -> np.ndarray:
    return (np.clip(m, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Color channel maps (pixel-wise; direct computation)
# ----------------------------------------------------------------------
def color_channel_maps(image: np.ndarray) -> Dict[str, np.ndarray]:
    """R/G/B, H/S/V, LAB L/A/B, Gray as 0-1 float maps."""
    b, g, r = cv2.split(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hh, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def f(x):
        return x.astype(np.float32) / 255.0

    return {
        "R": f(r), "G": f(g), "B": f(b),
        "H": hh.astype(np.float32) / 180.0, "S": f(s), "V": f(v),
        "LAB_L": f(L), "LAB_A": f(A), "LAB_B": f(B),
        "Gray": f(gray),
    }


# ----------------------------------------------------------------------
# Enhancement maps
# ----------------------------------------------------------------------
def local_brightness_map(image: np.ndarray, p: FeatureParams) -> np.ndarray:
    """Top-hat: pixels brighter than their local background (droplets pop)."""

    def _f(img):
        v = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(v, (0, 0), p.gaussian_sigma)
        return np.clip(v - bg, 0, None)

    m = compute_tiled(image, _f)
    return normalize01(np.clip(m / max(p.bright_delta * 2, 1), 0, 1))


def local_contrast_map(image: np.ndarray, p: FeatureParams) -> np.ndarray:
    """Local standard deviation of intensity (droplet rims respond high)."""

    def _f(img):
        v = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = max(3, int(p.gaussian_sigma) | 1)
        mean = cv2.boxFilter(v, -1, (k, k))
        sq = cv2.boxFilter(v * v, -1, (k, k))
        return np.sqrt(np.clip(sq - mean * mean, 0, None))

    return normalize01(compute_tiled(image, _f))


def highlight_map(image: np.ndarray, p: FeatureParams) -> np.ndarray:
    """Specular highlight soft response: very bright AND desaturated."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32)
    v_term = np.clip((v - p.highlight_v) / 55.0, 0, 1)
    return normalize01(v_term * (1.0 - s))


def edge_maps(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Canny / Sobel magnitude / Laplacian edge maps."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = normalize01(np.sqrt(sx * sx + sy * sy))
    lap = normalize01(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))
    return {"Edge_Canny": canny, "Edge_Sobel": sobel, "Edge_Laplacian": lap}


def blob_response_map(image: np.ndarray, p: FeatureParams) -> np.ndarray:
    """Multi-scale Difference-of-Gaussians max response (round small blobs)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out = np.zeros_like(gray)
    for s1, s2 in ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0)):
        dog = cv2.GaussianBlur(gray, (0, 0), s1) - cv2.GaussianBlur(gray, (0, 0), s2)
        out = np.maximum(out, np.clip(dog, 0, None))
    return normalize01(out)


def morphology_maps(
    image: np.ndarray, p: FeatureParams
) -> Dict[str, np.ndarray]:
    """Per-blob circularity / area / aspect-ratio maps (painted per blob)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    bg = cv2.GaussianBlur(v, (0, 0), p.gaussian_sigma)
    cand = (((v.astype(np.int16) - bg.astype(np.int16)) > p.bright_delta)
            & (s < 90)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    circ_map = np.zeros(v.shape, np.float32)
    area_map = np.zeros(v.shape, np.float32)
    aspect_map = np.zeros(v.shape, np.float32)

    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < p.blob_min_area or area > p.blob_max_area:
            continue
        perim = float(cv2.arcLength(cnt, True))
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, 1) / max(bh, 1)

        mask = np.zeros(v.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        circ_map[mask > 0] = max(circ_map[mask > 0].max(), float(circ))
        area_map[mask > 0] = float(np.clip(area / p.blob_max_area, 0, 1))
        aspect_map[mask > 0] = float(np.clip(1.0 / aspect, 0, 1))

    return {
        "Morph_Circularity": normalize01(circ_map),
        "Morph_Area": normalize01(area_map),
        "Morph_Aspect": normalize01(aspect_map),
    }


def hog_energy_map(image: np.ndarray) -> np.ndarray:
    """HOG-style gradient energy per 16x16 block, upsampled to full size."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    h, w = gray.shape
    bh, bw = max(h // 16, 1), max(w // 16, 1)
    blocks = cv2.resize(mag, (bw, bh), interpolation=cv2.INTER_AREA)
    return normalize01(cv2.resize(blocks, (w, h), interpolation=cv2.INTER_LINEAR))


def lbp_map(image: np.ndarray) -> np.ndarray:
    """Local Binary Pattern value map (texture fingerprint)."""
    from skimage.feature import local_binary_pattern

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    return normalize01(lbp.astype(np.float32))


def orb_density_map(image: np.ndarray) -> np.ndarray:
    """ORB keypoint density: Gaussian splat at each keypoint."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1500)
    kps = orb.detect(gray, None)
    dens = np.zeros(gray.shape, np.float32)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= y < dens.shape[0] and 0 <= x < dens.shape[1]:
            dens[y, x] += 1.0
    return normalize01(cv2.GaussianBlur(dens, (0, 0), 8.0))


# ----------------------------------------------------------------------
# Fusion / guttation
# ----------------------------------------------------------------------
def fusion_probability_map(
    image: np.ndarray, p: FeatureParams
) -> np.ndarray:
    """water_drop_probability_map: weighted fusion of droplet cues."""
    bright = local_brightness_map(image, p)
    contrast = local_contrast_map(image, p)
    blob = blob_response_map(image, p)
    s = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    low_sat = 1.0 - s
    total = p.w_brightness + p.w_contrast + p.w_blob + p.w_color
    if total <= 0:
        total = 1.0
    fused = (
        p.w_brightness * bright
        + p.w_contrast * contrast
        + p.w_blob * blob
        + p.w_color * low_sat
    ) / total
    return normalize01(fused)


def guttation_enhance_map(image: np.ndarray, p: FeatureParams) -> np.ndarray:
    """Guttation Enhancement Mode: tuned to bright, small, round, rimmed drops."""
    highlight = highlight_map(image, p)
    contrast = local_contrast_map(image, p)
    blob = blob_response_map(image, p)
    fused = 0.35 * highlight + 0.25 * contrast + 0.40 * blob
    return normalize01(fused)


def cnn_feature_map(image: np.ndarray) -> Optional[np.ndarray]:
    """Optional CNN feature map (MobileNetV2).  None when torch is absent."""
    try:
        import torch
        import torchvision.models as models
    except Exception:
        return None

    with torch.no_grad():
        net = models.mobilenet_v2(weights="DEFAULT").features.eval()
        x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224)).astype(np.float32) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        feat = net(x).mean(dim=1).squeeze(0).numpy()
        return normalize01(
            cv2.resize(feat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        )


# ----------------------------------------------------------------------
# Registry used by the UI
# ----------------------------------------------------------------------
def available_features() -> List[str]:
    """Feature names shown in the UI selector."""
    return [
        "R", "G", "B", "H", "S", "V", "LAB_L", "LAB_A", "LAB_B", "Gray",
        "LocalBrightness", "LocalContrast", "Highlight",
        "Edge_Canny", "Edge_Sobel", "Edge_Laplacian",
        "Blob", "Morph_Circularity", "Morph_Area", "Morph_Aspect",
        "HOG", "LBP", "ORB_Density",
        "Fusion", "GuttationEnhance", "CNN",
    ]


def compute_feature_map(
    name: str, image: np.ndarray, p: FeatureParams
) -> Optional[np.ndarray]:
    """Compute one named feature map at original resolution."""
    color = color_channel_maps(image)
    if name in color:
        return color[name]
    if name == "LocalBrightness":
        return local_brightness_map(image, p)
    if name == "LocalContrast":
        return local_contrast_map(image, p)
    if name == "Highlight":
        return highlight_map(image, p)
    edges = edge_maps(image)
    if name in edges:
        return edges[name]
    if name == "Blob":
        return blob_response_map(image, p)
    morph = morphology_maps(image, p)
    if name in morph:
        return morph[name]
    if name == "HOG":
        return hog_energy_map(image)
    if name == "LBP":
        return lbp_map(image)
    if name == "ORB_Density":
        return orb_density_map(image)
    if name == "Fusion":
        return fusion_probability_map(image, p)
    if name == "GuttationEnhance":
        return guttation_enhance_map(image, p)
    if name == "CNN":
        return cnn_feature_map(image)
    return None
