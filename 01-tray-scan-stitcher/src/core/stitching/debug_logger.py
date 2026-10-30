"""Debug logging and visualization for the stitching pipeline."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _imwrite(path: str, image: np.ndarray) -> bool:
    """Unicode-safe image write (cv2.imwrite fails on non-ASCII Windows paths)."""
    try:
        ext = Path(path).suffix or ".jpg"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


class DebugLogger:
    """Save debug images and JSON reports for each stitching step."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report: Dict = {}

    def save_keyframes(self, keyframes: List[tuple]) -> None:
        """Save selected keyframes."""
        dir_path = self.output_dir / "01_keyframes"
        dir_path.mkdir(exist_ok=True)
        for idx, frame in keyframes:
            _imwrite(str(dir_path / f"frame_{idx:04d}.jpg"), frame)
        self.report["keyframes"] = len(keyframes)

    # ------------------------------------------------------------------
    # Frame-pipeline stage dumps (countable on disk)
    # ------------------------------------------------------------------
    def save_candidates(self, keyframes: List[tuple]) -> None:
        """Save candidate keyframes (selector output, before the match graph)."""
        dir_path = self.output_dir / "01_candidates"
        dir_path.mkdir(exist_ok=True)
        for idx, frame in keyframes:
            _imwrite(str(dir_path / f"frame_{idx:05d}.jpg"), frame)
        self.report["candidate_frame_count"] = len(keyframes)

    def save_active_stitch_frames(self, keyframes: List[tuple]) -> None:
        """Save the frames actually passed to the stitcher.

        Written twice: ``05_active_stitch_frames`` (stage view) and
        ``REAL_STITCH_INPUT`` (the authoritative count the user can verify by
        opening the folder and counting the images).
        """
        for name in ("05_active_stitch_frames", "REAL_STITCH_INPUT"):
            dir_path = self.output_dir / name
            dir_path.mkdir(exist_ok=True)
            for i, (idx, frame) in enumerate(keyframes):
                _imwrite(str(dir_path / f"stitch_input_{i + 1:03d}_frame_{idx:05d}.jpg"), frame)
        self.report["active_stitch_frame_count"] = len(keyframes)

    def save_contributing_frames(self, keyframes: List[tuple]) -> None:
        """Save frames that actually contributed pixels to the final image."""
        dir_path = self.output_dir / "06_contributing_frames"
        dir_path.mkdir(exist_ok=True)
        for idx, frame in keyframes:
            _imwrite(str(dir_path / f"frame_{idx:05d}.jpg"), frame)
        self.report["contributing_frame_count"] = len(keyframes)

    def save_frame_selection_report(self, records: List[Dict]) -> None:
        """Save per-frame keep/reject records as frame_selection_report.json."""
        report_path = self.output_dir / "frame_selection_report.json"
        serializable = self._make_serializable(records)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def save_transform_report(self, report: List[Dict]) -> None:
        """Save per-pair / per-frame transform metrics as transform_report.json."""
        report_path = self.output_dir / "transform_report.json"
        serializable = self._make_serializable(report)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def save_projected_boxes(self, canvas: np.ndarray) -> None:
        """Save the projected-frame-boxes visualization."""
        if canvas is None or canvas.size == 0:
            return
        dir_path = self.output_dir / "05_projected_frame_boxes"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / "projected_boxes.jpg"), canvas)

    def save_quality_stage(self, name: str, image: np.ndarray) -> None:
        """Save one A/B/C/D quality-comparison stage image (PNG, lossless)."""
        if image is None or image.size == 0:
            return
        dir_path = self.output_dir / "10_quality_check"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / f"{name}.png"), image)

    def save_quality_metrics(self, metrics: Dict) -> None:
        """Save image-quality diagnostics as quality_metrics.json."""
        report_path = self.output_dir / "10_quality_check" / "quality_metrics.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(metrics), f, indent=2, ensure_ascii=False)

    def save_warped_stage(
        self, frames: List[np.ndarray], masks: List[np.ndarray]
    ) -> None:
        """Save every warped frame and its warped mask (artifact localization)."""
        f_dir = self.output_dir / "11_warped_frames"
        m_dir = self.output_dir / "12_warped_masks"
        f_dir.mkdir(exist_ok=True)
        m_dir.mkdir(exist_ok=True)
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            _imwrite(str(f_dir / f"warped_{i:03d}.png"), frame)
            _imwrite(str(m_dir / f"mask_{i:03d}.png"), (mask > 0).astype(np.uint8) * 255)

    def save_coverage_map(self, coverage: np.ndarray) -> None:
        """Save a colorized coverage map: 0=black, 1=green, 2=yellow, 3+=red."""
        if coverage is None or coverage.size == 0:
            return
        dir_path = self.output_dir / "13_coverage_map"
        dir_path.mkdir(exist_ok=True)
        vis = np.zeros((*coverage.shape, 3), dtype=np.uint8)
        vis[coverage == 1] = (0, 200, 0)
        vis[coverage == 2] = (0, 220, 220)
        vis[coverage >= 3] = (0, 0, 230)
        _imwrite(str(dir_path / "coverage_map.png"), vis)

    def save_keyframe_timeline(
        self, total_frames: int, selected_ids: List[int], active_ids: List[int]
    ) -> None:
        """Draw keyframe distribution along the video timeline."""
        if total_frames <= 0:
            return
        dir_path = self.output_dir / "15_continuity"
        dir_path.mkdir(exist_ok=True)
        W, H = 1200, 160
        img = np.full((H, W, 3), 255, dtype=np.uint8)

        def x_of(fid: int) -> int:
            return int(20 + (W - 40) * min(max(fid, 0), total_frames - 1) / max(total_frames - 1, 1))

        cv2.line(img, (20, 80), (W - 20, 80), (180, 180, 180), 2)
        for fid in selected_ids:
            cv2.line(img, (x_of(fid), 60), (x_of(fid), 100), (200, 160, 60), 2)
        for fid in active_ids:
            cv2.line(img, (x_of(fid), 45), (x_of(fid), 115), (40, 40, 220), 3)
        cv2.putText(img, f"total={total_frames} selected={len(selected_ids)} active={len(active_ids)}",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "orange=selected red=active", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        _imwrite(str(dir_path / "selected_keyframes_timeline.png"), img)

    def save_frame_centers_path(
        self, centers: List[tuple], bounds: Dict
    ) -> None:
        """Draw the global center trajectory of participating frames."""
        if not centers:
            return
        dir_path = self.output_dir / "15_continuity"
        dir_path.mkdir(exist_ok=True)
        h, w = bounds["height"], bounds["width"]
        scale = min(1200 / max(w, 1), 800 / max(h, 1), 1.0)
        img = np.zeros((max(int(h * scale), 1), max(int(w * scale), 1), 3), dtype=np.uint8)
        pts = [(int(x * scale), int(y * scale)) for x, y in centers]
        for k in range(1, len(pts)):
            cv2.line(img, pts[k - 1], pts[k], (0, 200, 0), 2)
        for k, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 0, 255), -1)
            cv2.putText(img, str(k), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        _imwrite(str(dir_path / "frame_centers_path.png"), img)

    def save_continuity_report(self, report: Dict) -> None:
        """Save continuity_report.json."""
        dir_path = self.output_dir / "15_continuity"
        dir_path.mkdir(exist_ok=True)
        with open(dir_path / "continuity_report.json", "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(report), f, indent=2, ensure_ascii=False)

    def save_source_map(self, ownership_mask: np.ndarray) -> None:
        """Save a colorized source map: one color per source frame id."""
        dir_path = self.output_dir / "09_source_map"
        dir_path.mkdir(exist_ok=True)
        if ownership_mask is None or ownership_mask.size == 0:
            return
        vis = np.zeros((*ownership_mask.shape, 3), dtype=np.uint8)
        unique_ids = [i for i in np.unique(ownership_mask) if i >= 0]
        for i in unique_ids:
            # Deterministic distinct color per frame id.
            rng = np.random.RandomState(int(i) + 1)
            vis[ownership_mask == i] = rng.randint(50, 255, size=3)
        _imwrite(str(dir_path / "source_map.png"), vis)

    def save_feature_matches(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        matches_img: np.ndarray,
        pair_idx: int,
    ) -> None:
        """Save feature match visualization."""
        dir_path = self.output_dir / "02_feature_matches"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / f"match_{pair_idx:03d}.jpg"), matches_img)

    def save_inlier_matches(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        matches_img: np.ndarray,
        pair_idx: int,
    ) -> None:
        """Save inlier match visualization."""
        dir_path = self.output_dir / "03_inlier_matches"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / f"inlier_{pair_idx:03d}.jpg"), matches_img)

    def save_pairwise_warp(
        self, warped: np.ndarray, pair_idx: int
    ) -> None:
        """Save pairwise warp result."""
        dir_path = self.output_dir / "04_pairwise_warp"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / f"warp_{pair_idx:03d}.jpg"), warped)

    def save_global_warp(self, canvas: np.ndarray) -> None:
        """Save global warp result."""
        dir_path = self.output_dir / "05_global_warp"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / "global_warp.jpg"), canvas)

    def save_seam_masks(self, seam_masks: List[np.ndarray]) -> None:
        """Save seam masks."""
        dir_path = self.output_dir / "06_seam_masks"
        dir_path.mkdir(exist_ok=True)
        for i, mask in enumerate(seam_masks):
            _imwrite(str(dir_path / f"seam_{i:03d}.png"), mask * 255)

    def save_exposure_compensated(self, frames: List[np.ndarray]) -> None:
        """Save exposure compensated frames."""
        dir_path = self.output_dir / "07_exposure_compensated"
        dir_path.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            _imwrite(str(dir_path / f"frame_{i:03d}.jpg"), frame)

    def save_blended(self, blended: np.ndarray) -> None:
        """Save blended result."""
        dir_path = self.output_dir / "08_blended"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / "blended.jpg"), blended)

    def save_final_crop(self, cropped: np.ndarray) -> None:
        """Save final cropped result."""
        dir_path = self.output_dir / "09_final_crop"
        dir_path.mkdir(exist_ok=True)
        _imwrite(str(dir_path / "final.jpg"), cropped)

    def save_report(self, report: Dict) -> None:
        """Save JSON report (large image arrays are excluded)."""
        self.report.update(report)
        # Convert numpy arrays to lists for JSON serialization.  Drop the big
        # image payloads (final image, source map) - they are saved as image
        # files, not as giant JSON lists.
        excluded = {"image", "source_map"}
        slim = {k: v for k, v in self.report.items() if k not in excluded}
        serializable_report = self._make_serializable(slim)
        report_path = self.output_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

    def _make_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        else:
            return obj

    def log(self, key: str, value) -> None:
        """Log a key-value pair to the report."""
        self.report[key] = value
