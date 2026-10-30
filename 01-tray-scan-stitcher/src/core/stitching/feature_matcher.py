"""Feature detection and matching with RANSAC outlier rejection.

Includes grid-uniform keypoint distribution so that structural features
(tray borders, frames, ground) are not drowned out by dense plant texture
in the image center, plus an inlier spatial-spread score.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _grid_uniform_keypoints(
    kp: List[cv2.KeyPoint],
    des: Optional[np.ndarray],
    shape: Tuple[int, int],
    grid_rows: int = 8,
    grid_cols: int = 8,
    max_per_cell: int = 40,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Keep at most ``max_per_cell`` strongest keypoints per grid cell.

    Prevents one highly-textured region (e.g. dense leaves at the center)
    from contributing the overwhelming majority of features.
    """
    if des is None or not kp:
        return kp, des

    h, w = shape[:2]
    cells: Dict[Tuple[int, int], List[int]] = {}
    for i, p in enumerate(kp):
        cx = min(int(p.pt[0] / max(w / grid_cols, 1)), grid_cols - 1)
        cy = min(int(p.pt[1] / max(h / grid_rows, 1)), grid_rows - 1)
        cells.setdefault((cy, cx), []).append(i)

    keep: List[int] = []
    for indices in cells.values():
        indices.sort(key=lambda i: kp[i].response, reverse=True)
        keep.extend(indices[:max_per_cell])

    keep.sort()
    return [kp[i] for i in keep], des[keep]


def _spread_cells(pts: np.ndarray, shape: Tuple[int, int], grid: int = 8) -> int:
    """Count how many grid cells the points cover (spatial spread score)."""
    if pts is None or len(pts) == 0:
        return 0
    h, w = shape[:2]
    p = pts.reshape(-1, 2)
    cx = np.clip((p[:, 0] / max(w / grid, 1)).astype(int), 0, grid - 1)
    cy = np.clip((p[:, 1] / max(h / grid, 1)).astype(int), 0, grid - 1)
    return int(len(set(zip(cy.tolist(), cx.tolist()))))


class FeatureMatcher:
    """Detect and match features between two frames."""

    def __init__(
        self,
        feature_type: str = "sift",
        max_features: int = 2000,
        ratio_test: float = 0.75,
        min_good_matches: int = 5,
        min_inliers: int = 3,
        min_inlier_ratio: float = 0.1,
        ransac_reprojection_threshold: float = 10.0,
        grid_rows: int = 8,
        grid_cols: int = 8,
        max_per_cell: int = 40,
    ):
        self.feature_type = feature_type
        self.max_features = max_features
        self.ratio_test = ratio_test
        self.min_good_matches = min_good_matches
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_reprojection_threshold = ransac_reprojection_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_per_cell = max_per_cell

        # Initialize feature detector.
        if feature_type == "sift":
            self.detector = cv2.SIFT_create(max_features)
            self.norm = cv2.NORM_L2
        elif feature_type == "akaze":
            self.detector = cv2.AKAZE_create()
            self.norm = cv2.NORM_HAMMING
        else:
            self.detector = cv2.ORB_create(max_features)
            self.norm = cv2.NORM_HAMMING

    def detect_and_compute(
        self, frame: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors, grid-uniformly distributed."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        kp, des = _grid_uniform_keypoints(
            kp, des, frame.shape, self.grid_rows, self.grid_cols, self.max_per_cell
        )
        return kp, des

    def match(
        self, ref: np.ndarray, curr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """Match features between two frames.

        Returns (src_pts, dst_pts, info) where src_pts are points in ``curr``
        and dst_pts are corresponding points in ``ref``.  Returns (None, None, info)
        if matching fails.
        """
        kp1, des1 = self.detect_and_compute(ref)
        kp2, des2 = self.detect_and_compute(curr)

        info = {
            "kp1_count": len(kp1),
            "kp2_count": len(kp2),
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reprojection_error": 0.0,
            "spread_cells": 0,
            "status": "failed",
        }

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            info["status"] = "insufficient_features"
            return None, None, info

        # Match using KNN + ratio test.
        if self.feature_type == "sift":
            matcher = cv2.FlannBasedMatcher()
        else:
            matcher = cv2.BFMatcher(self.norm)

        matches = matcher.knnMatch(des2, des1, k=2)

        # Apply Lowe's ratio test.
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.ratio_test * n.distance:
                    good.append(m)

        info["good_matches"] = len(good)

        if len(good) < self.min_good_matches:
            info["status"] = "insufficient_matches"
            return None, None, info

        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        info["status"] = "success"
        return src_pts, dst_pts, info

    # ------------------------------------------------------------------
    # Model estimators (can share one match set via src_pts/dst_pts)
    # ------------------------------------------------------------------
    def _inlier_stats(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        mask: Optional[np.ndarray],
        projected: Optional[np.ndarray],
        shape: Tuple[int, int],
        info: dict,
    ) -> None:
        """Fill inlier count / ratio / reprojection error / spread into info."""
        inliers = int(np.sum(mask)) if mask is not None else 0
        total = len(mask) if mask is not None else 0
        info["inliers"] = inliers
        info["inlier_ratio"] = inliers / total if total > 0 else 0.0

        if mask is not None and inliers > 0 and projected is not None:
            sel = mask.ravel() == 1
            src_in = src_pts[sel].reshape(-1, 2)
            dst_in = dst_pts[sel].reshape(-1, 2)
            # ``projected`` covers ALL source points; keep inliers only.
            proj = projected[sel].reshape(-1, 2)
            # Correct per-point Euclidean reprojection error.
            err = np.linalg.norm(proj - dst_in, axis=1)
            info["reprojection_error"] = float(np.mean(err))
            info["spread_cells"] = _spread_cells(src_in, shape)

    def estimate_homography(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Estimate homography between two frames using RANSAC."""
        if src_pts is None or dst_pts is None:
            src_pts, dst_pts, info = self.match(ref, curr)
        if info is None:
            info = {"status": "success"}

        if src_pts is None or dst_pts is None:
            return None, info

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_reprojection_threshold
        )

        if H is None:
            info["status"] = "homography_failed"
            return None, info

        projected = (
            cv2.perspectiveTransform(src_pts, H) if mask is not None else None
        )
        self._inlier_stats(src_pts, dst_pts, mask, projected, ref.shape, info)

        if info["inliers"] < self.min_inliers or info["inlier_ratio"] < self.min_inlier_ratio:
            info["status"] = "insufficient_inliers"
            return None, info

        info["status"] = "success"
        return H, info

    def estimate_affine(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Estimate full affine transform between two frames using RANSAC."""
        if src_pts is None or dst_pts is None:
            src_pts, dst_pts, info = self.match(ref, curr)
        if info is None:
            info = {"status": "success"}

        if src_pts is None or dst_pts is None:
            return None, info

        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reprojection_threshold,
        )

        if M is None:
            info["status"] = "affine_failed"
            return None, info

        if mask is not None and int(np.sum(mask)) > 0:
            src_h = np.hstack(
                [src_pts.reshape(-1, 2), np.ones((len(src_pts), 1), dtype=np.float32)]
            )
            projected = (M @ src_h.T).T.reshape(-1, 1, 2)
        else:
            projected = None
        self._inlier_stats(src_pts, dst_pts, mask, projected, ref.shape, info)

        if info["inliers"] < self.min_inliers or info["inlier_ratio"] < self.min_inlier_ratio:
            info["status"] = "insufficient_inliers"
            return None, info

        info["status"] = "success"
        return M, info

    def estimate_similarity(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        src_pts: Optional[np.ndarray] = None,
        dst_pts: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Estimate similarity transform (translation + rotation + uniform scale).

        Uses cv2.estimateAffinePartial2D - the preferred model for plane
        scanning where the camera moves above a roughly flat scene.
        """
        if src_pts is None or dst_pts is None:
            src_pts, dst_pts, info = self.match(ref, curr)
        if info is None:
            info = {"status": "success"}

        if src_pts is None or dst_pts is None:
            return None, info

        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reprojection_threshold,
        )

        if M is None:
            info["status"] = "similarity_failed"
            return None, info

        if mask is not None and int(np.sum(mask)) > 0:
            src_h = np.hstack(
                [src_pts.reshape(-1, 2), np.ones((len(src_pts), 1), dtype=np.float32)]
            )
            projected = (M @ src_h.T).T.reshape(-1, 1, 2)
        else:
            projected = None
        self._inlier_stats(src_pts, dst_pts, mask, projected, ref.shape, info)

        if info["inliers"] < self.min_inliers or info["inlier_ratio"] < self.min_inlier_ratio:
            info["status"] = "insufficient_inliers"
            return None, info

        info["status"] = "success"
        return M, info

    def estimate_translation(
        self, ref: np.ndarray, curr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Estimate pure translation via phase correlation (last-resort model)."""
        info = {
            "inliers": 0,
            "inlier_ratio": 0.0,
            "reprojection_error": 0.0,
            "spread_cells": 0,
            "status": "failed",
        }
        try:
            ref_gray = cv2.cvtColor(
                cv2.resize(ref, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY
            )
            curr_gray = cv2.cvtColor(
                cv2.resize(curr, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY
            )
            (dx, dy), response = cv2.phaseCorrelate(
                np.float32(curr_gray), np.float32(ref_gray)
            )
            # phaseCorrelate maps curr -> ref; scale back to full resolution.
            dx, dy = dx * 2.0, dy * 2.0
            if response < 0.05:
                info["status"] = "phase_correlate_failed"
                return None, info
            M = np.float64([[1, 0, dx], [0, 1, dy]])
            info["status"] = "success"
            info["inlier_ratio"] = float(response)
            return M, info
        except Exception:
            info["status"] = "phase_correlate_failed"
            return None, info
