"""Transform estimation with strict geometric validation and model fallback.

Model priority for plane scanning (camera moving above the scene):

    Translation  (last resort, rigid)
    Similarity   (preferred: translation + rotation + uniform scale)
    Affine       (when similarity cannot explain the motion)
    Homography   (only when clearly better AND geometrically plausible)

Every candidate matrix must pass ``validate_transform``; rejected candidates
are recorded with human-readable reasons so no bad matrix silently enters
the mosaic.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import StitchConfig
from .feature_matcher import FeatureMatcher
from .transform_validator import distortion_penalty, to_3x3, validate_transform


def _compute_deformation(H: np.ndarray, frame_shape: Tuple[int, int]) -> float:
    """Ratio of max to min side length of the warped frame quadrilateral."""
    h, w = frame_shape[:2]
    corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    ).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        sides = [
            float(np.linalg.norm(warped[(i + 1) % 4] - warped[i])) for i in range(4)
        ]
        if min(sides) > 0:
            return float(max(sides) / min(sides))
        return float("inf")
    except Exception:
        return float("inf")


class TransformEstimator:
    """Estimate the simplest reliable transform between two frames."""

    def __init__(
        self,
        feature_matcher: FeatureMatcher,
        max_deformation: float = 2.0,
        config: Optional[StitchConfig] = None,
    ):
        self.matcher = feature_matcher
        self.max_deformation = max_deformation
        self.config = config if config is not None else StitchConfig()

    def estimate(
        self, ref: np.ndarray, curr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str, dict]:
        """Estimate the best valid transform from ``curr`` to ``ref``.

        Returns (matrix, model_type, info).  ``info["rejected"]`` lists every
        candidate that failed validation together with its reasons.
        model_type is one of "similarity" / "affine" / "homography" /
        "translation" / "failed".
        """
        rejected: List[Dict] = []

        # Single feature-matching pass shared by all RANSAC models.
        src_pts, dst_pts, match_info = self.matcher.match(ref, curr)

        candidates: Dict[str, Tuple[Optional[np.ndarray], dict]] = {}

        if src_pts is not None and dst_pts is not None:
            # --- Similarity (preferred for plane scanning) ---
            M_sim, info_sim = self.matcher.estimate_similarity(
                ref, curr, src_pts, dst_pts, dict(match_info)
            )
            candidates["similarity"] = (M_sim, info_sim)

            # --- Full affine ---
            M_aff, info_aff = self.matcher.estimate_affine(
                ref, curr, src_pts, dst_pts, dict(match_info)
            )
            candidates["affine"] = (M_aff, info_aff)

            # --- Homography (last choice, strictest scrutiny) ---
            H_hom, info_hom = self.matcher.estimate_homography(
                ref, curr, src_pts, dst_pts, dict(match_info)
            )
            candidates["homography"] = (H_hom, info_hom)

        # Validate every candidate geometrically.
        valid: Dict[str, Tuple[np.ndarray, dict, dict]] = {}
        for model, (M, info) in candidates.items():
            if M is None:
                rejected.append({"model": model, "reasons": [info.get("status", "failed")]})
                continue
            ok, reasons, metrics = validate_transform(M, ref.shape, self.config)
            info["metrics"] = metrics
            if ok:
                valid[model] = (M, info, metrics)
            else:
                rejected.append({"model": model, "reasons": reasons})

        # ---- Choose the winning model --------------------------------
        chosen_model: Optional[str] = None

        def _score(info: dict, metrics: dict) -> float:
            return (
                info.get("inlier_ratio", 0.0) * 100.0
                - info.get("reprojection_error", 0.0)
                - distortion_penalty(metrics, self.config) * 0.5
            )

        simple = [m for m in ("similarity", "affine") if m in valid]
        if simple:
            # Prefer similarity on near-tie (simpler model wins).
            simple.sort(
                key=lambda m: _score(valid[m][1], valid[m][2]), reverse=True
            )
            chosen_model = simple[0]

            # Homography is allowed only when it is clearly superior.
            if "homography" in valid:
                _, h_info, _ = valid["homography"]
                _, s_info, _ = valid[chosen_model]
                clearly_better = (
                    h_info.get("inlier_ratio", 0.0)
                    >= s_info.get("inlier_ratio", 0.0) + 0.05
                    and h_info.get("reprojection_error", 1e9)
                    <= s_info.get("reprojection_error", 0.0) * 0.7
                )
                if clearly_better:
                    chosen_model = "homography"
                else:
                    rejected.append(
                        {
                            "model": "homography",
                            "reasons": ["几何合理但不显著优于简单模型，按优先级弃用"],
                        }
                    )
        elif "homography" in valid:
            chosen_model = "homography"

        if chosen_model is not None:
            M, info, metrics = valid[chosen_model]
            info["model"] = chosen_model
            info["rejected"] = rejected
            return to_3x3(M), chosen_model, info

        # ---- Last resort: pure translation via phase correlation ------
        M_trans, info_trans = self.matcher.estimate_translation(ref, curr)
        if M_trans is not None:
            info_trans["model"] = "translation"
            info_trans["rejected"] = rejected
            info_trans["fallback_reason"] = "all_feature_models_invalid"
            return to_3x3(M_trans), "translation", info_trans

        return None, "failed", {
            "status": "all_models_failed",
            "match_status": match_info.get("status", "unknown"),
            "rejected": rejected,
        }
