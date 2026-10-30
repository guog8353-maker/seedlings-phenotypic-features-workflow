"""Multi-feature fusion guttation (吐水) detection.

Replaces the single mean-brightness threshold with a per-cell pipeline:

    cell crop (original resolution)
      -> HSV leaf segmentation
      -> bright low-saturation blob candidates (multi-scale, cell-relative)
      -> shape / brightness / color / position / context feature scoring
      -> guttation_score in [0, 1]
      -> confirmed / suspect / rejected with explicit reasons

Never resizes the source image; every score is accompanied by a reason so
misclassifications can be audited in report.json.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .grid_manager import GridCell, GridManager


@dataclass
class GuttationConfig:
    """All tunables for the guttation detector."""

    # HSV leaf segmentation.
    leaf_h_min: int = 35
    leaf_h_max: int = 95
    leaf_s_min: int = 40
    leaf_v_min: int = 20

    # Candidate bright-spot extraction.
    bright_delta: int = 25        # V must exceed local background by this
    max_saturation: int = 90      # droplets are low-saturation

    # Multi-scale blob size limits (relative to cell area).
    min_area_ratio: float = 0.00008
    max_area_ratio: float = 0.02

    # Reflection rejection (large connected bright region).
    reflection_area_ratio: float = 0.20
    reflection_max_circularity: float = 0.35

    # Score weights.
    w_position: float = 0.20
    w_shape: float = 0.25
    w_brightness: float = 0.20
    w_color: float = 0.15
    w_context: float = 0.20

    # Decision thresholds.
    confirm_threshold: float = 0.70
    suspect_threshold: float = 0.45

    # Position scoring: within this fraction of the cell diagonal from the
    # leaf edge counts as "edge/tip zone".
    edge_near_ratio: float = 0.06


@dataclass
class Candidate:
    """One droplet candidate inside a cell."""

    cx: float
    cy: float
    area: float
    circularity: float
    score: float
    verdict: str          # confirmed / suspect / rejected
    reject_reason: str = ""
    features: Dict[str, float] = field(default_factory=dict)


class GuttationDetector:
    """Multi-feature guttation detector working on grid cells."""

    def __init__(self, config: Optional[GuttationConfig] = None):
        self.config = config or GuttationConfig()
        self.last_report: Dict = {}
        self.last_candidates: Dict[Tuple[int, int], List[Candidate]] = {}

    # ------------------------------------------------------------------
    def detect(
        self, image: np.ndarray, grid_manager: GridManager
    ) -> Dict[Tuple[int, int], str]:
        """Run detection; returns {(row, col): "1"/"0"} like the old detector."""
        labels: Dict[Tuple[int, int], str] = {}
        self.last_candidates = {}
        report_cells = []

        n_confirmed = 0
        n_suspect = 0
        n_rejected = 0
        reject_reasons: Dict[str, int] = {}
        total_candidates = 0

        for cell in grid_manager.cells:
            cell_img = image[cell.y1 : cell.y2, cell.x1 : cell.x2]
            if cell_img.size == 0:
                labels[(cell.row, cell.col)] = "0"
                continue

            candidates = self._analyze_cell(cell_img)
            self.last_candidates[(cell.row, cell.col)] = candidates
            total_candidates += len(candidates)

            best = max((c.score for c in candidates), default=0.0)
            label = "1" if best >= self.config.confirm_threshold else "0"
            labels[(cell.row, cell.col)] = label

            for c in candidates:
                if c.verdict == "confirmed":
                    n_confirmed += 1
                elif c.verdict == "suspect":
                    n_suspect += 1
                else:
                    n_rejected += 1
                    reject_reasons[c.reject_reason] = (
                        reject_reasons.get(c.reject_reason, 0) + 1
                    )

            report_cells.append({
                "row": cell.row,
                "col": cell.col,
                "best_score": round(best, 3),
                "label": label,
                "n_candidates": len(candidates),
            })

        self.last_report = {
            "total_candidates": total_candidates,
            "confirmed_guttation": n_confirmed,
            "suspect": n_suspect,
            "rejected": n_rejected,
            "reject_reasons": reject_reasons,
            "cells": report_cells,
        }
        logging.info(
            "吐水检测: 候选 %d | 确认 %d | 疑似 %d | 排除 %d | 原因 %s",
            total_candidates, n_confirmed, n_suspect, n_rejected, reject_reasons,
        )
        return labels

    def cell_score(self, row: int, col: int) -> float:
        """Best candidate score of a cell (0 if none) - for UI display."""
        cands = self.last_candidates.get((row, col), [])
        return max((c.score for c in cands), default=0.0)

    def write_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.last_report, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _analyze_cell(self, cell_img: np.ndarray) -> List[Candidate]:
        cfg = self.config
        ch, cw = cell_img.shape[:2]
        cell_area = float(ch * cw)
        diag = float(np.hypot(ch, cw))

        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # 1. Leaf segmentation (HSV green).
        leaf_mask = cv2.inRange(
            hsv,
            (cfg.leaf_h_min, cfg.leaf_s_min, cfg.leaf_v_min),
            (cfg.leaf_h_max, 255, 255),
        )
        leaf_ratio = float(np.count_nonzero(leaf_mask)) / cell_area

        # 2. Bright low-saturation blobs near leaves.
        sigma = max(diag * 0.05, 3.0)
        background = cv2.GaussianBlur(V, (0, 0), sigma)
        bright = ((V.astype(np.int16) - background.astype(np.int16))
                  > cfg.bright_delta)
        low_sat = S < cfg.max_saturation
        near_leaf = cv2.dilate(
            leaf_mask, np.ones((max(3, int(diag * 0.03)),) * 2, np.uint8)
        ) > 0
        cand_mask = (bright & low_sat & near_leaf).astype(np.uint8) * 255
        cand_mask = cv2.morphologyEx(
            cand_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )

        # 3. Large bright connected regions = leaf reflection / water film.
        reflection_mask = self._large_bright_regions(V, S, cell_area)

        # Distance to leaf edge (0 on the boundary).
        leaf_dist = cv2.distanceTransform(leaf_mask, cv2.DIST_L2, 3)

        min_area = max(4.0, cell_area * cfg.min_area_ratio)
        max_area = cell_area * cfg.max_area_ratio

        contours, _ = cv2.findContours(
            cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[Candidate] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 1:
                continue
            perimeter = float(cv2.arcLength(cnt, True))
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            )
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            cand = Candidate(cx, cy, area, circularity, 0.0, "rejected")

            # ---- Hard rejections with explicit reasons ----
            if area > max_area:
                cand.reject_reason = "too_large"
                candidates.append(cand)
                continue
            if area < min_area:
                cand.reject_reason = "too_small"
                candidates.append(cand)
                continue
            if circularity < 0.45:
                cand.reject_reason = "not_round"
                candidates.append(cand)
                continue

            cmask = np.zeros((ch, cw), np.uint8)
            cv2.drawContours(cmask, [cnt], -1, 255, -1)
            if reflection_mask[int(cy), int(cx)] > 0:
                cand.reject_reason = "leaf_reflection"
                candidates.append(cand)
                continue

            # ---- Feature scores ----
            inner_mean = float(cv2.mean(V, mask=cmask)[0])
            ring = cv2.dilate(cmask, np.ones((9, 9), np.uint8)) - cmask
            ring_mean = float(cv2.mean(V, mask=ring)[0]) if ring.any() else inner_mean
            contrast = inner_mean - ring_mean
            if contrast < 8:
                cand.reject_reason = "low_contrast"
                candidates.append(cand)
                continue

            shape_score = float(np.clip((circularity - 0.3) / 0.6, 0, 1))
            v_term = float(np.clip((inner_mean - 150) / 105, 0, 1))
            contrast_term = float(np.clip(contrast / 60, 0, 1))
            brightness_score = 0.6 * contrast_term + 0.4 * v_term

            mean_s = float(cv2.mean(S, mask=cmask)[0])
            s_term = 1.0 - float(np.clip(mean_s / 120, 0, 1))
            green_inside = float(np.mean((leaf_mask[cmask > 0] > 0)))
            color_score = float(np.clip(0.7 * s_term + 0.3 * (1 - green_inside), 0, 1))

            dist_edge = float(leaf_dist[int(cy), int(cx)])
            near_px = cfg.edge_near_ratio * diag
            if dist_edge <= near_px:
                position_score = 1.0
            else:
                position_score = float(np.clip(1 - dist_edge / (0.25 * diag), 0, 1))

            context_score = 1.0 if leaf_ratio >= 0.02 else 0.4

            score = (
                cfg.w_position * position_score
                + cfg.w_shape * shape_score
                + cfg.w_brightness * brightness_score
                + cfg.w_color * color_score
                + cfg.w_context * context_score
            )
            cand.score = float(np.clip(score, 0, 1))
            cand.features = {
                "shape": round(shape_score, 3),
                "brightness": round(brightness_score, 3),
                "color": round(color_score, 3),
                "position": round(position_score, 3),
                "context": round(context_score, 3),
                "contrast": round(contrast, 1),
                "dist_edge": round(dist_edge, 1),
            }

            if cand.score >= cfg.confirm_threshold:
                cand.verdict = "confirmed"
            elif cand.score >= cfg.suspect_threshold:
                cand.verdict = "suspect"
            else:
                cand.verdict = "rejected"
                cand.reject_reason = "low_score"
            candidates.append(cand)

        return candidates

    def _large_bright_regions(
        self, V: np.ndarray, S: np.ndarray, cell_area: float
    ) -> np.ndarray:
        """Mask of large contiguous bright low-sat regions (reflections)."""
        cfg = self.config
        mask = ((V > 200) & (S < cfg.max_saturation)).astype(np.uint8) * 255
        out = np.zeros_like(mask)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for c in range(1, num):
            area = stats[c, cv2.CC_STAT_AREA]
            if area < cell_area * cfg.reflection_area_ratio:
                continue
            comp = (labels == c).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            perim = float(cv2.arcLength(cnts[0], True))
            circ = 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0
            if circ < cfg.reflection_max_circularity:
                out[labels == c] = 255
        return out

    # ------------------------------------------------------------------
    def render_debug(
        self,
        image: np.ndarray,
        grid_manager: GridManager,
        max_size: int = 1600,
    ) -> np.ndarray:
        """Draw confirmed/suspect/rejected circles on a downscaled copy."""
        h, w = image.shape[:2]
        scale = min(max_size / max(w, h), 1.0)
        vis = (
            cv2.resize(image, (int(w * scale), int(h * scale)))
            if scale < 1
            else image.copy()
        )

        colors = {
            "confirmed": (0, 200, 0),
            "suspect": (0, 220, 220),
            "rejected": (0, 0, 220),
        }
        cell_lookup = {(c.row, c.col): c for c in grid_manager.cells}
        for (row, col), cands in self.last_candidates.items():
            cell = cell_lookup.get((row, col))
            if cell is None:
                continue
            for cand in cands:
                x = int((cell.x1 + cand.cx) * scale)
                y = int((cell.y1 + cand.cy) * scale)
                r = max(4, int(np.sqrt(cand.area / np.pi) * scale))
                color = colors.get(cand.verdict, (255, 255, 255))
                cv2.circle(vis, (x, y), r, color, 2)
                cv2.putText(
                    vis, f"{cand.score:.2f}", (x + r + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                )
        return vis
