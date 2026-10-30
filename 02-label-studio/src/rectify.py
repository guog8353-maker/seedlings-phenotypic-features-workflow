"""Perspective rectification of the seedling tray region (4 corner points).

The user marks the tray quadrilateral once; everything downstream (grid,
annotation, export) then works on the distortion-free rectangle.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def order_quad_points(pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Sort 4 points into [top-left, top-right, bottom-right, bottom-left].

    Robust to arbitrary click order: sort by polar angle around the centroid,
    then rotate the sequence so it starts at the top-left-most point.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    ordered = pts[np.argsort(angles)]

    # Rotate so the sequence starts at the point closest to the image's
    # top-left corner (min x+y), giving TL, TR, BR, BL order for quads
    # that are not rotated more than ~45 degrees.
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)

    # Ensure clockwise order (positive signed area in image coordinates).
    signed = 0.0
    for i in range(4):
        p1, p2 = ordered[i], ordered[(i + 1) % 4]
        signed += p1[0] * p2[1] - p2[0] * p1[1]
    if signed < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def compute_output_size(
    pts: np.ndarray, max_side: int = 4096
) -> Tuple[int, int]:
    """Output (width, height) from mean quad edge lengths, capped at max_side."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    w = int(round((top + bottom) / 2.0))
    h = int(round((left + right) / 2.0))
    w = max(1, min(w, max_side))
    h = max(1, min(h, max_side))
    return w, h


def _quad_is_degenerate(pts: np.ndarray, image_shape: Tuple[int, int]) -> Optional[str]:
    """Return a reason string if the quad is unusable, else None."""
    h, w = image_shape[:2]
    if ((pts[:, 0] < -0.05 * w).any() or (pts[:, 0] > 1.05 * w).any()
            or (pts[:, 1] < -0.05 * h).any() or (pts[:, 1] > 1.05 * h).any()):
        return "角点超出图像范围"
    area = float(cv2.contourArea(pts.astype(np.float32)))
    if area < 0.001 * w * h:
        return f"四边形面积过小（{area:.0f}px^2）"
    # Self-intersection check: the two diagonals must cross each other.
    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = _ccw(pts[0], pts[2], pts[1])
    d2 = _ccw(pts[0], pts[2], pts[3])
    d3 = _ccw(pts[1], pts[3], pts[0])
    d4 = _ccw(pts[1], pts[3], pts[2])
    if (d1 > 0) == (d2 > 0) or (d3 > 0) == (d4 > 0):
        return "四边形自交（请按相邻顺序点选角点）"
    return None


def rectify_image(
    image: np.ndarray,
    pts: Sequence[Sequence[float]],
    out_size: Optional[Tuple[int, int]] = None,
    max_side: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Warp the 4-point tray quad into a rectangle.

    Returns (rectified_image, H_matrix, out_size).  Raises ValueError with a
    human-readable reason for degenerate quads.
    """
    ordered = order_quad_points(pts)
    reason = _quad_is_degenerate(ordered, image.shape)
    if reason:
        raise ValueError(f"无法校正: {reason}")

    if out_size is None:
        out_w, out_h = compute_output_size(ordered, max_side)
    else:
        out_w, out_h = out_size
        out_w = max(1, min(int(out_w), max_side))
        out_h = max(1, min(int(out_h), max_side))

    src = ordered.astype(np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(
        image, H, (out_w, out_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logging.info("透视校正: out=%dx%d H=%s", out_w, out_h, H.tolist())
    return rectified, H, (out_w, out_h)


def save_rectify_info(
    path: str,
    pts: Sequence[Sequence[float]],
    H: np.ndarray,
    out_size: Tuple[int, int],
    src_path: str = "",
) -> None:
    """Persist the rectification parameters for coordinate back-mapping."""
    info = {
        "source_image": src_path,
        "quad_points_original": [list(map(float, p)) for p in order_quad_points(pts)],
        "H_original_to_rectified": H.tolist(),
        "output_size": [int(out_size[0]), int(out_size[1])],
        "note": "labels.csv 与小图坐标均基于校正图；"
                "原图坐标 = inv(H) @ 校正图坐标",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
