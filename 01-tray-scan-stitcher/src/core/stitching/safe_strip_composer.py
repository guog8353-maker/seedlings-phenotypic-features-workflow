"""SafeStripComposer: phone-panorama-style stable mosaic composition.

Instead of letting all frames compete pixel-by-pixel (which shatters into
thin strips and black cracks), frames are laid out along the camera's main
motion axis.  Each frame owns one contiguous slab; adjacent frames share
exactly one seam; only a 4-12px band around each seam is feathered.

Guarantees (asserted at runtime):
- every pixel with at least one valid frame gets an owner (holes == 0);
- image and mask share the same canvas shape and the same transform;
- masks are strictly binary;
- no erode / closing / inpaint / per-frame morphology anywhere.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .debug_logger import _imwrite


class SafeStripComposer:
    """Compose warped frames into one stable panorama along the motion axis."""

    def __init__(
        self,
        feather_width: int = 8,
        strip_variance_threshold: float = 0.9,
        debug_logger=None,
    ):
        self.feather_width = max(int(feather_width), 0)
        self.strip_variance_threshold = strip_variance_threshold
        self.debug_logger = debug_logger

    # ------------------------------------------------------------------
    def compose(
        self,
        warped_frames: List[np.ndarray],
        content_masks: List[np.ndarray],
        global_poses: List[np.ndarray],
        frame_shapes: List[Tuple[int, int]],
        canvas_bounds: Dict,
    ) -> Dict:
        """Compose the final mosaic.

        Returns a dict with:
            image:         final composed canvas (after seam feather)
            source_id_map: int32 owner frame index per pixel (-1 = none)
            valid_mask:    uint8 canvas validity (owner >= 0)
            mode:          "strip" or "fallback" (free 2D motion)
            stats:         axis / variance ratio / seams / holes
        """
        n = len(warped_frames)
        assert n >= 1 and n == len(content_masks) == len(global_poses) == len(frame_shapes)
        canvas_h, canvas_w = canvas_bounds["height"], canvas_bounds["width"]

        # ---- Input assertions (fail loudly, never stitch on bad data) ----
        for i, (img, mask, pose) in enumerate(
            zip(warped_frames, content_masks, global_poses)
        ):
            assert img.shape[:2] == (canvas_h, canvas_w), (
                f"frame {i}: warped image {img.shape[:2]} != canvas {(canvas_h, canvas_w)}"
            )
            assert mask.shape == (canvas_h, canvas_w), (
                f"frame {i}: warped mask {mask.shape} != canvas {(canvas_h, canvas_w)}"
            )
            uniq = set(np.unique(mask).tolist())
            assert uniq <= {0, 1, 255}, f"frame {i}: mask not binary: {sorted(uniq)[:8]}"
            assert np.isfinite(pose).all(), f"frame {i}: pose contains NaN/Inf"

        # ---- Frame centers on the canvas ----
        offset_x = -canvas_bounds["min_x"]
        offset_y = -canvas_bounds["min_y"]
        centers = np.zeros((n, 2), dtype=np.float64)
        for i, (shape, pose) in enumerate(zip(frame_shapes, global_poses)):
            h, w = shape[:2]
            H = pose.astype(np.float64).copy()
            H[0, 2] += offset_x
            H[1, 2] += offset_y
            c = np.array([w / 2.0, h / 2.0, 1.0])
            centers[i] = (H @ c)[:2]

        # ---- Main motion axis via PCA on the centers ----
        axis, variance_ratio = self._main_axis(centers)
        stats: Dict = {
            "axis": [float(axis[0]), float(axis[1])],
            "variance_ratio": float(variance_ratio),
            "holes": 0,
            "seams": [],
        }

        if n >= 2 and variance_ratio < self.strip_variance_threshold:
            # Free 2D motion: caller falls back to the generic composer.
            stats["reason"] = (
                f"主轴方差占比 {variance_ratio:.2f} < {self.strip_variance_threshold}"
            )
            return {
                "image": None,
                "source_id_map": None,
                "valid_mask": None,
                "mode": "fallback",
                "stats": stats,
            }

        # ---- Projection coordinates along the axis ----
        t_centers = centers @ axis
        order = np.argsort(t_centers)
        t_sorted = t_centers[order]

        ys, xs = np.mgrid[0:canvas_h, 0:canvas_w]
        t_map = (xs * axis[0] + ys * axis[1]).astype(np.float32)

        # ---- Hard ownership: nearest center along axis, content required ----
        best_dist = np.full((canvas_h, canvas_w), np.inf, dtype=np.float32)
        owner = np.full((canvas_h, canvas_w), -1, dtype=np.int32)
        valid_union = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        for idx in order:
            valid = content_masks[idx] > 0
            valid_union = np.maximum(valid_union, valid.astype(np.uint8))
            d = np.abs(t_map - t_centers[idx])
            upd = valid & (d < best_dist)
            owner[upd] = idx
            best_dist[upd] = d[upd]

        # ---- Coverage assertion: no valid pixel may be ownerless ----
        holes = int(((valid_union > 0) & (owner < 0)).sum())
        stats["holes"] = holes
        print(
            f"[SAFE STRIP] axis=({axis[0]:.3f},{axis[1]:.3f}) "
            f"variance={variance_ratio:.2f} | holes = {holes}",
            flush=True,
        )
        assert holes == 0, f"source_id_map 存在 {holes} 个无属主像素"

        # ---- Seam refinement between consecutive strips ----
        gray_frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in warped_frames
        ]
        seams = self._find_seams(
            order, t_sorted, gray_frames, content_masks, t_map, stats
        )

        # Rebuild ownership from seam boundaries (keeps strips contiguous),
        # with content-valid fallback so coverage stays complete.
        owner = self._assign_by_seams(
            order, t_centers, seams, content_masks, t_map, valid_union
        )

        # ---- Hard composite (winner-takes-all, zero blending) ----
        hard = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i in range(n):
            hard[owner == i] = warped_frames[i][owner == i]
        canvas_valid = (owner >= 0).astype(np.uint8)

        # ---- Narrow feather around each seam (both-valid only) ----
        final = self._feather_seams(
            hard, warped_frames, content_masks, order, seams, t_map, owner
        )

        self._save_debug(
            valid_union, owner, hard, final, seams, t_map, order, n
        )

        return {
            "image": final,
            "source_id_map": owner,
            "valid_mask": canvas_valid,
            "mode": "strip",
            "stats": stats,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _main_axis(centers: np.ndarray) -> Tuple[np.ndarray, float]:
        """Dominant motion direction + its variance share (PCA)."""
        if len(centers) < 2:
            return np.array([1.0, 0.0]), 1.0
        c = centers - centers.mean(axis=0)
        cov = c.T @ c / max(len(centers) - 1, 1)
        vals, vecs = np.linalg.eigh(cov)
        idx = int(np.argmax(vals))
        total = float(vals.sum())
        ratio = float(vals[idx] / total) if total > 1e-12 else 1.0
        axis = vecs[:, idx]
        # Canonical sign: point along +x when ambiguous.
        if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
            axis = -axis
        return axis.astype(np.float64), ratio

    def _find_seams(
        self,
        order: np.ndarray,
        t_sorted: np.ndarray,
        gray_frames: List[np.ndarray],
        masks: List[np.ndarray],
        t_map: np.ndarray,
        stats: Dict,
    ) -> List[Tuple[int, int, float]]:
        """One optimal seam per consecutive strip pair.

        Between the centers of strip k and k+1, sample candidate axis
        positions; the seam goes where the two frames' mean gray difference
        (over both-valid pixels near the candidate) is minimal.
        """
        seams: List[Tuple[int, int, float]] = []
        for k in range(len(order) - 1):
            i, j = int(order[k]), int(order[k + 1])
            t0, t1 = float(t_sorted[k]), float(t_sorted[k + 1])
            both = (masks[i] > 0) & (masks[j] > 0)

            best_s, best_cost = (t0 + t1) / 2.0, np.inf
            for frac in np.linspace(0.3, 0.7, 9):
                s = t0 + frac * (t1 - t0)
                near = both & (np.abs(t_map - s) < 8)
                if not np.any(near):
                    continue
                cost = float(np.mean(np.abs(gray_frames[i][near] - gray_frames[j][near])))
                if cost < best_cost:
                    best_cost, best_s = cost, float(s)

            seams.append((i, j, best_s))
            stats["seams"].append(
                {"pair": [i, j], "position": round(best_s, 1), "cost": round(best_cost, 2)}
            )
        return seams

    def _assign_by_seams(
        self,
        order: np.ndarray,
        t_centers: np.ndarray,
        seams: List[Tuple[int, int, float]],
        masks: List[np.ndarray],
        t_map: np.ndarray,
        valid_union: np.ndarray,
    ) -> np.ndarray:
        """Assign each pixel to its seam-delimited slab, with content fallback."""
        canvas_h, canvas_w = t_map.shape
        owner = np.full((canvas_h, canvas_w), -1, dtype=np.int32)

        boundaries = sorted(s for _, _, s in seams)
        slab_index = np.digitize(t_map, boundaries).astype(np.int32)
        slab_index = np.clip(slab_index, 0, len(order) - 1)
        slab_owner = order.astype(np.int32)

        preferred = slab_owner[slab_index]
        for pos, idx in enumerate(order):
            region = (slab_index == pos) & (masks[idx] > 0)
            owner[region] = idx

        # Fallback: preferred frame has no content -> nearest center with content.
        best_dist = np.full((canvas_h, canvas_w), np.inf, dtype=np.float32)
        missing = (owner < 0) & (valid_union > 0)
        for idx in order:
            d = np.abs(t_map - t_centers[idx])
            upd = missing & (masks[idx] > 0) & (d < best_dist)
            owner[upd] = idx
            best_dist[upd] = d[upd]

        leftover = int(((valid_union > 0) & (owner < 0)).sum())
        assert leftover == 0, f"seam 分配后仍有 {leftover} 个无属主像素"
        return owner

    def _feather_seams(
        self,
        hard: np.ndarray,
        warped_frames: List[np.ndarray],
        masks: List[np.ndarray],
        order: np.ndarray,
        seams: List[Tuple[int, int, float]],
        t_map: np.ndarray,
        owner: np.ndarray,
    ) -> np.ndarray:
        """Linear crossfade in a narrow band around each seam.

        Feather requires BOTH frames to have real content; single-sided
        pixels keep the hard owner value.  One pass per seam, no compounding.
        """
        if self.feather_width <= 0 or not seams:
            return hard

        W = float(self.feather_width)
        out = hard.astype(np.float32)
        for i, j, s in seams:
            band = (np.abs(t_map - s) < W) & ((owner == i) | (owner == j))
            both = band & (masks[i] > 0) & (masks[j] > 0)
            if not np.any(both):
                continue
            alpha = np.clip((t_map - (s - W)) / (2.0 * W), 0.0, 1.0)[both][:, None]
            a = warped_frames[i][both].astype(np.float32)
            b = warped_frames[j][both].astype(np.float32)
            out[both] = a * (1.0 - alpha) + b * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def _save_debug(
        self,
        valid_union: np.ndarray,
        owner: np.ndarray,
        hard: np.ndarray,
        final: np.ndarray,
        seams: List[Tuple[int, int, float]],
        t_map: np.ndarray,
        order: np.ndarray,
        n: int,
    ) -> None:
        """Write the A-E debug images (28th requirement)."""
        if self.debug_logger is None:
            return
        out_dir = self.debug_logger.output_dir / "14_safe_strip"
        out_dir.mkdir(exist_ok=True)

        _imwrite(str(out_dir / "A_valid_union.png"), valid_union * 255)

        colors = np.random.RandomState(7).randint(50, 255, size=(max(n, 1), 3))
        vis = np.zeros((*owner.shape, 3), dtype=np.uint8)
        for i in range(n):
            vis[owner == i] = colors[i]
        _imwrite(str(out_dir / "B_source_id_map.png"), vis)

        _imwrite(str(out_dir / "C_hard_composite.png"), hard)

        seam_vis = hard.copy()
        for _, _, s in seams:
            line = (np.abs(t_map - s) < 1).astype(np.uint8) * 255
            seam_vis[line > 0] = (0, 0, 255)
        _imwrite(str(out_dir / "D_seam_lines.png"), seam_vis)

        _imwrite(str(out_dir / "E_feather_result.png"), final)
