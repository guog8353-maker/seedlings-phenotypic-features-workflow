"""Geometric sanity validation for pairwise and global transforms.

A homography that RANSAC accepts is not necessarily visually plausible.
Every transform must pass ``validate_transform`` before it may enter the
mosaic; otherwise it is rejected with human-readable reasons.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import StitchConfig


def to_3x3(M: np.ndarray) -> np.ndarray:
    """Embed a 2x3 affine matrix into a 3x3 homogeneous matrix."""
    if M.shape == (2, 3):
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
        return H
    return M.astype(np.float64)


def project_corners(H: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Project the four frame corners through H.  Returns (4, 2) points."""
    h, w = shape[:2]
    corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H).reshape(-1, 2)


def quad_area(pts: np.ndarray) -> float:
    """Signed-area magnitude of a quadrilateral given as (4, 2) points."""
    return 0.5 * abs(
        float(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)))
        - float(np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    )


def quad_is_self_intersecting(pts: np.ndarray) -> bool:
    """Check whether the quadrilateral (4, 2) self-intersects (bowtie)."""

    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    def _intersect(a, b, c, d):
        d1 = _ccw(c, d, a)
        d2 = _ccw(c, d, b)
        d3 = _ccw(a, b, c)
        d4 = _ccw(a, b, d)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    # A simple quad self-intersects iff the two diagonals' endpoint pairs
    # straddle each other on non-adjacent edges.
    return _intersect(pts[0], pts[1], pts[2], pts[3]) or _intersect(
        pts[1], pts[2], pts[3], pts[0]
    )


def decompose(H: np.ndarray, shape: Tuple[int, int]) -> Dict:
    """Decompose a 3x3 transform into human-interpretable metrics."""
    H = to_3x3(H)
    h, w = shape[:2]
    A = H[:2, :2]

    scale_x = float(np.hypot(H[0, 0], H[1, 0]))
    scale_y = float(np.hypot(H[0, 1], H[1, 1]))
    rotation = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
    perspective_strength = float(abs(H[2, 0]) + abs(H[2, 1]))
    det = float(np.linalg.det(A))
    try:
        condition_number = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    corners = project_corners(H, shape)
    area_ratio = quad_area(corners) / float(w * h)

    diag = float(np.hypot(w, h))
    original = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corner_displacement = float(
        np.max(np.linalg.norm(corners - original, axis=1)) / max(diag, 1.0)
    )

    aspect = scale_x / scale_y if scale_y > 1e-9 else float("inf")

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "aspect_ratio": aspect,
        "rotation_deg": rotation,
        "translation": [float(H[0, 2]), float(H[1, 2])],
        "perspective_strength": perspective_strength,
        "determinant": det,
        "condition_number": condition_number,
        "projected_area_ratio": float(area_ratio),
        "corner_displacement": corner_displacement,
        "self_intersecting": quad_is_self_intersecting(corners),
    }


def validate_transform(
    H: np.ndarray,
    shape: Tuple[int, int],
    config: StitchConfig,
    is_global: bool = False,
) -> Tuple[bool, List[str], Dict]:
    """Validate a transform against geometric sanity limits.

    Returns (valid, reasons, metrics).  ``is_global=True`` uses the more
    generous accumulated-pose limits (for final per-frame global poses).
    """
    metrics = decompose(H, shape)
    reasons: List[str] = []

    if is_global:
        min_s, max_s = config.global_min_scale, config.global_max_scale
        max_rot = config.global_max_rotation_deg
    else:
        min_s, max_s = config.min_scale, config.max_scale
        max_rot = config.max_rotation_deg

    if not (min_s < metrics["scale_x"] < max_s):
        reasons.append(f"scale_x={metrics['scale_x']:.3f} 超出 [{min_s}, {max_s}]")
    if not (min_s < metrics["scale_y"] < max_s):
        reasons.append(f"scale_y={metrics['scale_y']:.3f} 超出 [{min_s}, {max_s}]")
    if not (config.min_aspect_ratio < metrics["aspect_ratio"] < config.max_aspect_ratio):
        reasons.append(f"aspect_ratio={metrics['aspect_ratio']:.3f} 异常")
    if abs(metrics["rotation_deg"]) > max_rot and not is_global:
        reasons.append(f"rotation={metrics['rotation_deg']:.2f}° 超过 ±{max_rot}°")
    elif abs(metrics["rotation_deg"]) > max_rot and is_global:
        reasons.append(f"global rotation={metrics['rotation_deg']:.2f}° 超过 ±{max_rot}°")
    if not (config.min_area_ratio < metrics["projected_area_ratio"] < config.max_area_ratio):
        reasons.append(f"projected_area_ratio={metrics['projected_area_ratio']:.2f} 异常")
    if metrics["perspective_strength"] > config.max_perspective:
        reasons.append(
            f"perspective_strength={metrics['perspective_strength']:.5f} 超过 {config.max_perspective}"
        )
    if metrics["determinant"] <= 0:
        reasons.append(f"determinant={metrics['determinant']:.3f} <= 0（镜像/翻转）")
    if metrics["condition_number"] > config.max_condition_number:
        reasons.append(f"condition_number={metrics['condition_number']:.2e} 过大")
    if metrics["corner_displacement"] > config.max_corner_displacement and not is_global:
        reasons.append(
            f"corner_displacement={metrics['corner_displacement']:.2f} 超过 {config.max_corner_displacement}"
        )
    if metrics["self_intersecting"]:
        reasons.append("projected quad 自交（蝴蝶结形）")

    return (len(reasons) == 0), reasons, metrics


def distortion_penalty(metrics: Dict, config: StitchConfig) -> float:
    """Continuous penalty measuring how far a transform is from rigid motion.

    Used for model scoring: the higher the penalty, the more the transform
    deforms the frame.  Rigid translation scores 0.
    """
    penalty = 0.0
    penalty += abs(metrics["scale_x"] - 1.0) * 100.0
    penalty += abs(metrics["scale_y"] - 1.0) * 100.0
    penalty += abs(metrics["aspect_ratio"] - 1.0) * 50.0
    penalty += abs(metrics["rotation_deg"]) * 2.0
    penalty += metrics["perspective_strength"] * 1e5
    penalty += abs(metrics["projected_area_ratio"] - 1.0) * 50.0
    return penalty
