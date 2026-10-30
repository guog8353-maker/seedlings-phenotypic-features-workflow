"""Main stitching pipeline for plane scanning mode with multi-frame mosaic."""

import heapq
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import KeyframePipelineError, StitchConfig
from .cropper import Cropper
from .debug_logger import DebugLogger
from .exposure_compensator import ExposureCompensator
from .feature_matcher import FeatureMatcher
from .global_transform_verifier import GlobalTransformVerifier
from .keyframe_graph import KeyframeGraph, select_span_path
from .keyframe_selector import select_keyframes
from .narrow_band_blender import NarrowBandBlender
from .region_ownership import RegionOwnership
from .safe_strip_composer import SafeStripComposer
from .seam_finder import SeamFinder
from .transform_estimator import TransformEstimator
from .transform_validator import project_corners, to_3x3, validate_transform
from .video_analyzer import VideoAnalyzer


class StitchingPipeline:
    """Complete stitching pipeline for plane scanning mode with multi-frame mosaic.

    All parameters come from a single StitchConfig instance (single source of
    truth).  Keyword overrides are applied on top of it for backward
    compatibility (e.g. ``StitchingPipeline(feature_type="sift")``).
    """

    def __init__(self, config: Optional[StitchConfig] = None, **overrides):
        self.config = config if config is not None else StitchConfig()
        if overrides:
            self.config.apply_overrides(**overrides)
        cfg = self.config

        # Initialize modules.
        self.matcher = FeatureMatcher(
            feature_type=cfg.feature_type,
            max_features=cfg.max_features,
            ratio_test=cfg.ratio_test,
            min_good_matches=cfg.min_good_matches,
            min_inliers=cfg.min_inliers,
            min_inlier_ratio=cfg.min_inlier_ratio,
            ransac_reprojection_threshold=cfg.ransac_reprojection_threshold,
            grid_rows=cfg.feature_grid_rows,
            grid_cols=cfg.feature_grid_cols,
            max_per_cell=cfg.max_features_per_cell,
        )
        self.estimator = TransformEstimator(
            feature_matcher=self.matcher,
            max_deformation=cfg.max_deformation,
            config=cfg,
        )
        self.verifier = GlobalTransformVerifier()
        self.graph_builder = KeyframeGraph(self.matcher)
        self.region_ownership = RegionOwnership()
        self.narrow_band_blender = NarrowBandBlender(seam_blend_width=cfg.seam_blend_width)
        self.exposure_compensator = ExposureCompensator(method=cfg.exposure_method)
        self.cropper = Cropper(mode=cfg.crop_mode)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def stitch(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int, int], bool]] = None,
    ) -> Dict:
        """Run the complete stitching pipeline on a video.

        Returns a dict with:
            - success: bool
            - image: stitched image or None
            - message: status message
            - report: detailed report dict
            - level: stitching level (1=full, 2=simple, 3=fallback)
            - stitch_status: SINGLE_FRAME_FALLBACK / LIMITED_STITCH /
                             MULTI_FRAME_STITCH / NORMAL_MULTI_FRAME_STITCH
            - frame_counts: per-stage frame counts
        """
        start_time = time.time()
        self.config.log("STITCHER")

        report = {
            "video_path": video_path,
            "success": False,
            "message": "",
            "level": 0,
            "steps": {},
            "initial_keyframe_count": 0,
            "retry_keyframe_count": 0,
            "final_keyframe_count": 0,
            "fallback_reason": "",
            "stitch_status": "",
            "frame_counts": {},
            "active_frame_ids": [],
        }

        # AUTO keyframe mode: this video's spatial coverage decides its own
        # recommended keyframe count (never a one-size-fits-all number).
        if self.config.auto_keyframes:
            analyzer = VideoAnalyzer(
                target_overlap=self.config.target_overlap,
                min_auto_keyframes=self.config.min_auto_keyframes,
                max_auto_keyframes=self.config.max_auto_keyframes,
                blur_threshold=self.config.blur_threshold,
            )
            analysis = analyzer.analyze(video_path)
            recommended = self._verify_auto_keyframes(
                video_path, analysis.recommended_keyframes, analysis.total_frames
            )
            self.config.target_keyframes = recommended
            report["video_analysis"] = {
                "recommended_keyframes": recommended,
                "analyzer_estimate": analysis.recommended_keyframes,
                "motion_direction": analysis.motion_direction,
                "cumulative_span_px": round(analysis.cumulative_span_px, 1),
                "median_shift_px": round(analysis.median_shift_px, 2),
                "blur_ratio": round(analysis.blur_ratio, 3),
                "mean_response": round(analysis.mean_response, 3),
                "direction_consistency": round(analysis.direction_consistency, 3),
                "stitchability": analysis.stitchability,
                "reasons": analysis.reasons,
                "duration": round(analysis.duration, 2),
                "total_frames": analysis.total_frames,
            }
            print(
                f"[AUTO KEYFRAMES] analyzer = {analysis.recommended_keyframes} -> "
                f"match-verified = {recommended} | "
                f"direction = {analysis.motion_direction} | "
                f"span = {analysis.cumulative_span_px:.0f}px | "
                f"blur = {analysis.blur_ratio:.0%} | "
                f"stitchability = {analysis.stitchability}",
                flush=True,
            )

        # Initialize debug logger (one folder per run).
        debug_logger = None
        if self.config.debug_dir:
            run_dir = (
                Path(self.config.debug_dir)
                / f"{Path(video_path).stem}_{datetime.now():%Y%m%d_%H%M%S}"
            )
            debug_logger = DebugLogger(str(run_dir))

        try:
            # Step 1+2: select keyframes and build the match graph, with
            # keyframe recovery when the actual stitch input count severely
            # mismatches the target.
            relax_level = 0
            selection_records: List[Dict] = []
            while True:
                keyframes, sel_stats = self._select_keyframes_with_retry(
                    video_path, report, progress_callback, relax_level
                )
                selection_records = sel_stats.get("frame_records", [])

                if len(keyframes) < 2:
                    # Level 3: fallback - use the sharpest frame.
                    return self._fallback_sharpest_frame(
                        video_path, report, debug_logger, start_time,
                        error="可用帧不足 2 张",
                    )

                report["steps"]["keyframes"] = len(keyframes)
                report["final_keyframe_count"] = len(keyframes)

                if debug_logger:
                    debug_logger.save_candidates(keyframes)

                # Build keyframe graph and select the LONGEST TEMPORAL-SPAN
                # path (never just the biggest cluster - that drops the
                # head/tail of the video and produces partial stitches).
                if progress_callback:
                    progress_callback(2, 10)
                frames = [f for _, f in keyframes]
                graph = self.graph_builder.build_graph(frames, max_gap=3)
                active_indices = select_span_path(
                    graph, keyframes, min_edge_ratio=self.config.min_inlier_ratio
                )

                if len(active_indices) < 2:
                    # Level 2: simple stable stitching - use all frames.
                    active_indices = list(range(len(frames)))

                print(
                    f"[MATCH FILTER] before = {len(keyframes)} after = {len(active_indices)}",
                    flush=True,
                )
                print(
                    f"[SPAN PATH] id range = "
                    f"{keyframes[active_indices[0]][0]}..{keyframes[active_indices[-1]][0]}",
                    flush=True,
                )

                # Bridge recovery: extend the active path along the timeline
                # so the stitch covers the video from start to end.
                keyframes, active_indices, bridge_info = self._bridge_extend(
                    video_path,
                    keyframes,
                    active_indices,
                    sel_stats.get("total_video_frames", 0),
                )
                report["steps"]["bridge"] = bridge_info
                frames = [f for _, f in keyframes]

                active_pairs = [keyframes[i] for i in active_indices]
                n_active = len(active_pairs)

                # Record graph-stage exclusions.
                excluded = set(range(len(keyframes))) - set(active_indices)
                for i in sorted(excluded):
                    selection_records.append(
                        {
                            "frame_id": keyframes[i][0],
                            "decision": "REJECT",
                            "stage": "match_graph",
                            "reason": "not_in_largest_component",
                            "value": "",
                            "threshold": "",
                        }
                    )

                # Hard mismatch check: target says many frames, reality says 1-2.
                target = self.config.target_keyframes
                hard_min = self.config.hard_min_active_frames
                severe_mismatch = target >= 8 and n_active <= 2
                below_hard_min = target >= hard_min and n_active < hard_min

                try:
                    if severe_mismatch or below_hard_min:
                        raise KeyframePipelineError(
                            f"目标关键帧数量与实际参与数量严重不符: "
                            f"target = {target}, active = {n_active}"
                        )
                except KeyframePipelineError as e:
                    if relax_level < 2:
                        relax_level += 1
                        print(
                            f"[KEYFRAME RECOVERY] {e}，触发第 {relax_level} 级关键帧恢复",
                            flush=True,
                        )
                        report["steps"][f"keyframe_recovery_{relax_level}"] = str(e)
                        continue
                    # Recovery exhausted.
                    if n_active < 2:
                        return self._fallback_sharpest_frame(
                            video_path, report, debug_logger, start_time,
                            error=f"关键帧恢复后仍不足 2 张: {e}",
                        )
                    print(f"[KEYFRAME RECOVERY] 恢复耗尽，以 {n_active} 帧继续: {e}", flush=True)

                break

            active_frames = [f for _, f in active_pairs]
            report["steps"]["active_frames"] = n_active
            report["active_frame_ids"] = [int(fid) for fid, _ in active_pairs]

            # ---------------- STITCH INPUT CHECK (hard log) ----------------
            fps = float(sel_stats.get("fps") or 0.0)
            print("=============================", flush=True)
            print("STITCH INPUT CHECK", flush=True)
            print(f"target_keyframes = {self.config.target_keyframes}", flush=True)
            print(f"candidate_frames = {sel_stats.get('quality_passed', 0)}", flush=True)
            print(f"selected_frames = {len(keyframes)}", flush=True)
            print(f"active_stitch_frames = {n_active}", flush=True)
            print("=============================", flush=True)
            for fid, _ in active_pairs:
                ts = f"{fid / fps:.2f}s" if fps > 0 else "n/a"
                print(f"[STITCH INPUT] Frame {fid} timestamp = {ts} selected = YES", flush=True)
            print(f"TOTAL INPUT FRAMES = {n_active}", flush=True)
            print(f"REAL_STITCH_INPUT_COUNT = {n_active}", flush=True)

            # Per-stage frame counts (the four numbers that must agree).
            frame_counts = {
                "target_keyframes": self.config.target_keyframes,
                "total_video_frames": sel_stats.get("total_video_frames", 0),
                "candidate_frames": sel_stats.get("quality_passed", 0),
                "selected_frames": len(keyframes),
                "active_stitch_frames": n_active,
                "contributing_frames": 0,  # filled after ownership computation
            }
            # Runtime assertion: the reported count must equal the real list.
            assert frame_counts["active_stitch_frames"] == len(active_frames), (
                f"报告的参与拼接帧数 {frame_counts['active_stitch_frames']} "
                f"与实际列表长度 {len(active_frames)} 不一致"
            )
            report["frame_counts"] = frame_counts

            if debug_logger:
                debug_logger.save_active_stitch_frames(active_pairs)
                debug_logger.save_frame_selection_report(selection_records)

            # Step 3: Build a pose graph with validated transforms.
            # Each frame tries to match i+1, i+2, i+3 so that a single bad
            # edge cannot poison the whole chain - BFS can route around it.
            if progress_callback:
                progress_callback(3, 10)
            n_frames = len(active_frames)
            edges: Dict[Tuple[int, int], Dict] = {}
            transform_report: List[Dict] = []

            for i in range(n_frames):
                for j in range(i + 1, min(i + 4, n_frames)):
                    H, model_type, info = self.estimator.estimate(
                        active_frames[i], active_frames[j]
                    )

                    # Explainable per-pair log.
                    fid_i, fid_j = active_pairs[i][0], active_pairs[j][0]
                    pair_entry = {
                        "pair": [int(fid_i), int(fid_j)],
                        "gap": j - i,
                        "model": model_type,
                        "status": info.get("status", ""),
                        "inliers": info.get("inliers", 0),
                        "inlier_ratio": round(info.get("inlier_ratio", 0.0), 3),
                        "reprojection_error": round(info.get("reprojection_error", 0.0), 3),
                        "spread_cells": info.get("spread_cells", 0),
                    }
                    if H is None:
                        reasons = []
                        for rej in info.get("rejected", []):
                            reasons.append(
                                f"{rej['model']}: {'; '.join(rej['reasons'])}"
                            )
                        pair_entry["rejected"] = reasons
                        print(
                            f"[TRANSFORM] pair {fid_i}->{fid_j}: ALL MODELS REJECTED "
                            f"({' | '.join(reasons)})",
                            flush=True,
                        )
                    else:
                        m = info.get("metrics", {})
                        pair_entry.update({
                            "scale_x": round(m.get("scale_x", 1.0), 4),
                            "scale_y": round(m.get("scale_y", 1.0), 4),
                            "rotation_deg": round(m.get("rotation_deg", 0.0), 3),
                            "perspective_strength": round(m.get("perspective_strength", 0.0), 6),
                            "projected_area_ratio": round(m.get("projected_area_ratio", 1.0), 4),
                        })
                        rejected_notes = [
                            f"{r['model']}({'; '.join(r['reasons'])})"
                            for r in info.get("rejected", [])
                            if r["model"] != model_type
                        ]
                        note = f" | rejected: {', '.join(rejected_notes)}" if rejected_notes else ""
                        print(
                            f"[TRANSFORM] pair {fid_i}->{fid_j}: {model_type} accepted "
                            f"(inliers={info.get('inliers', 0)}, "
                            f"err={info.get('reprojection_error', 0.0):.2f}, "
                            f"scale=({m.get('scale_x', 1.0):.3f},{m.get('scale_y', 1.0):.3f}), "
                            f"rot={m.get('rotation_deg', 0.0):.2f}deg, "
                            f"persp={m.get('perspective_strength', 0.0):.5f}){note}",
                            flush=True,
                        )
                        edges[(i, j)] = {
                            "H": to_3x3(H),
                            "info": info,
                            "model": model_type,
                        }
                    transform_report.append(pair_entry)

            report["steps"]["pose_graph_edges"] = len(edges)

            # Median-motion consistency check on consecutive edges:
            # a pair whose translation deviates wildly from the median motion
            # is an outlier edge and is removed (BFS will route around it).
            consec = [
                (i, i + 1) for i in range(n_frames - 1) if (i, i + 1) in edges
            ]
            if len(consec) >= 3:
                mags = []
                for e in consec:
                    t = edges[e]["H"][:2, 2]
                    mags.append(float(np.hypot(t[0], t[1])))
                med = float(np.median(mags))
                factor = self.config.motion_outlier_factor
                for e, mag in zip(consec, mags):
                    if abs(mag - med) > factor * max(med, 1.0) + 20.0:
                        fid_a, fid_b = active_pairs[e[0]][0], active_pairs[e[1]][0]
                        print(
                            f"[MOTION OUTLIER] pair {fid_a}->{fid_b}: |t|={mag:.1f}px "
                            f"vs median {med:.1f}px, edge removed",
                            flush=True,
                        )
                        del edges[e]
                        report["steps"].setdefault("motion_outlier_edges", []).append(
                            [int(fid_a), int(fid_b), round(mag, 1), round(med, 1)]
                        )

            # Step 4: Global poses via BFS from the middle reference frame.
            if progress_callback:
                progress_callback(4, 10)
            ref_idx = n_frames // 2
            global_poses = self._bfs_global_poses(edges, n_frames, ref_idx)
            report["steps"]["global_poses"] = sum(1 for p in global_poses if p is not None)
            report["steps"]["reference_frame"] = int(active_pairs[ref_idx][0])

            # Global sanity check: drop frames whose accumulated pose is
            # geometrically absurd (instead of letting them tear the mosaic).
            valid_idx = []
            for i, pose in enumerate(global_poses):
                if pose is None:
                    print(
                        f"[GLOBAL POSE] FRAME {active_pairs[i][0]}: unreachable in pose graph, excluded",
                        flush=True,
                    )
                    continue
                ok, reasons, metrics = validate_transform(
                    pose, active_frames[i].shape, self.config, is_global=True
                )
                if ok:
                    valid_idx.append(i)
                else:
                    print(
                        f"[GLOBAL POSE] FRAME {active_pairs[i][0]}: INVALID GLOBAL POSE "
                        f"({'; '.join(reasons)}), excluded",
                        flush=True,
                    )
                    transform_report.append({
                        "frame_id": int(active_pairs[i][0]),
                        "valid": False,
                        "reasons": reasons,
                        "global_metrics": {
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in metrics.items()
                        },
                    })

            n_dropped = n_frames - len(valid_idx)
            report["steps"]["geometrically_valid_frames"] = len(valid_idx)
            report["frame_counts"]["geometrically_valid"] = len(valid_idx)
            if n_dropped > 0:
                print(
                    f"[GLOBAL POSE] {n_dropped} frame(s) excluded, "
                    f"{len(valid_idx)} geometrically valid frames remain",
                    flush=True,
                )

            if len(valid_idx) < 2:
                return self._fallback_sharpest_frame(
                    video_path, report, debug_logger, start_time,
                    error="几何校验后可用帧不足 2 张",
                )

            active_pairs = [active_pairs[i] for i in valid_idx]
            active_frames = [active_frames[i] for i in valid_idx]
            global_poses = [global_poses[i] for i in valid_idx]
            n_active = len(active_frames)
            report["steps"]["active_frames"] = n_active
            report["active_frame_ids"] = [int(fid) for fid, _ in active_pairs]
            report["frame_counts"]["active_stitch_frames"] = n_active

            # ---- Completeness metrics (hard requirement) ----------------
            fps_c = float(sel_stats.get("fps") or 0.0)
            total_c = int(sel_stats.get("total_video_frames", 0))
            ids_c = [int(fid) for fid, _ in active_pairs]
            duration_c = total_c / fps_c if fps_c > 0 else 0.0
            first_t = min(ids_c) / fps_c if fps_c > 0 else 0.0
            last_t = max(ids_c) / fps_c if fps_c > 0 else 0.0
            time_cov = (last_t - first_t) / duration_c if duration_c > 0 else 0.0
            report["completeness"] = {
                "video_duration": round(duration_c, 2),
                "first_used_time": round(first_t, 2),
                "last_used_time": round(last_t, 2),
                "time_coverage_ratio": round(time_cov, 3),
                "partial_failure": bool(time_cov < 0.6),
                "continuity_breaks": 0,  # filled after composition
            }
            print(
                f"[COMPLETENESS] time_coverage_ratio = {time_cov:.2f} "
                f"(first={first_t:.2f}s last={last_t:.2f}s duration={duration_c:.2f}s)",
                flush=True,
            )

            if debug_logger:
                debug_logger.save_keyframe_timeline(
                    total_c,
                    [int(fid) for fid, _ in keyframes],
                    ids_c,
                )

            # Consecutive-pair transforms / infos for exposure + ownership.
            orig_idx = valid_idx
            transforms = []
            match_infos = []
            for k in range(1, len(orig_idx)):
                a, b = orig_idx[k - 1], orig_idx[k]
                e = edges.get((a, b))
                if e is not None:
                    transforms.append(e["H"])
                    match_infos.append(e["info"])
                else:
                    transforms.append(np.eye(3, dtype=np.float64))
                    match_infos.append({"inlier_ratio": 1.0, "reprojection_error": 0.0})

            # Step 5: Verify global transform expansion.
            if progress_callback:
                progress_callback(5, 10)
            canvas_bounds = self.verifier.compute_canvas_bounds(active_frames, global_poses)
            expansion = self.verifier.verify_expansion(active_frames, global_poses, canvas_bounds)
            report["steps"]["expansion"] = expansion

            if not expansion["is_expanded"]:
                report["steps"]["warnings"] = expansion["warnings"]

            if debug_logger:
                debug_logger.save_transform_report(transform_report)
                debug_logger.save_projected_boxes(
                    self._draw_projected_boxes(active_frames, global_poses, canvas_bounds)
                )

            # Step 6: Exposure compensation.
            if progress_callback:
                progress_callback(6, 10)
            compensated_frames = self.exposure_compensator.compensate(
                active_frames, transforms
            )
            report["steps"]["exposure_compensated"] = len(compensated_frames)

            if debug_logger:
                debug_logger.save_exposure_compensated(compensated_frames)

            # Step 7: Warp frames to canvas and generate binary masks.
            if progress_callback:
                progress_callback(7, 10)
            warped_frames, warped_masks = self._warp_frames_to_canvas(
                compensated_frames, global_poses, canvas_bounds
            )
            report["steps"]["warped_frames"] = len(warped_frames)

            if debug_logger:
                debug_logger.save_global_warp(warped_frames[0])
                # Quality-check stage A (original) and B (after warp only).
                debug_logger.save_quality_stage("01_original_frame", active_frames[0])
                debug_logger.save_quality_stage("02_warp_only", warped_frames[0])
                debug_logger.save_warped_stage(warped_frames, warped_masks)

            # Step 8+9: compose the mosaic.
            # Preferred: SafeStripComposer (strip ownership along the motion
            # axis, one seam per adjacent pair, narrow feather).
            # Fallback: generic pixel-level quality ownership + coverage fill
            # + narrow-band blending (free 2D motion only).
            if progress_callback:
                progress_callback(8, 10)
            compose_mode = "generic"
            ownership_mask = None
            source_map = None
            blended = None

            if self.config.strip_compose_enabled:
                composer = SafeStripComposer(
                    feather_width=self.config.seam_blend_width,
                    debug_logger=debug_logger,
                )
                comp = composer.compose(
                    warped_frames,
                    warped_masks,
                    global_poses,
                    [f.shape for f in active_frames],
                    canvas_bounds,
                )
                report["steps"]["compose"] = comp["stats"]
                if comp["mode"] == "strip":
                    compose_mode = "strip"
                    ownership_mask = comp["source_id_map"]
                    blended = comp["image"]
                    print("[COMPOSE] SafeStripComposer: strip mode", flush=True)
                else:
                    print(
                        f"[COMPOSE] SafeStripComposer: fallback ({comp['stats'].get('reason', '')})",
                        flush=True,
                    )

            if compose_mode != "strip":
                ownership_mask, source_map = self.region_ownership.compute_ownership(
                    warped_frames, warped_masks, transforms, match_infos
                )
                report["steps"]["ownership_computed"] = ownership_mask is not None

                if debug_logger:
                    debug_logger.save_seam_masks([source_map])

                # Coverage consistency: holes = valid content with no owner.
                ownership_mask, n_holes_before, n_holes_after = self._fill_coverage_holes(
                    ownership_mask, warped_masks
                )
                report["steps"]["coverage"] = {
                    "coverage_hole_pixels_before": n_holes_before,
                    "coverage_hole_pixels_after": n_holes_after,
                }
                print(
                    f"[COVERAGE] coverage_hole_pixels: {n_holes_before} -> {n_holes_after}",
                    flush=True,
                )

                # Step 9 (generic): narrow-band blending.
                if progress_callback:
                    progress_callback(9, 10)
                blended = self.narrow_band_blender.blend(
                    warped_frames, ownership_mask, warped_masks
                )

            report["steps"]["blended"] = blended is not None

            # ---- Continuity break detection (hard requirement) -----------
            centers_px = self._frame_centers_px(
                [f.shape for f in active_frames], global_poses, canvas_bounds
            )
            axis_c, _ = SafeStripComposer._main_axis(np.array(centers_px))
            n_breaks = self._detect_continuity_break(ownership_mask, axis_c)
            report["completeness"]["continuity_breaks"] = n_breaks
            if n_breaks > 0:
                report["completeness"]["partial_failure"] = True
            print(f"[CONTINUITY] breaks = {n_breaks}", flush=True)

            if debug_logger:
                debug_logger.save_frame_centers_path(centers_px, canvas_bounds)

            if debug_logger:
                debug_logger.save_source_map(ownership_mask)
                coverage = np.zeros(ownership_mask.shape, dtype=np.int32)
                for m in warped_masks:
                    coverage += (m > 0).astype(np.int32)
                debug_logger.save_coverage_map(coverage)
                composite = np.zeros_like(warped_frames[0])
                for i, wf in enumerate(warped_frames):
                    composite[ownership_mask == i] = wf[ownership_mask == i]
                debug_logger.save_quality_stage("05_before_blend", composite)
                debug_logger.save_blended(blended)
                # Quality-check stage C (after blend, before crop/postprocess).
                debug_logger.save_quality_stage("03_blend_only", blended)
                debug_logger.save_quality_stage("06_after_blend", blended)

            # Step 10: Auto crop to the largest valid connected region.
            if progress_callback:
                progress_callback(10, 10)
            # The final valid mask must reflect what actually landed in the
            # blended image: pixels with an owner frame.  (The union of the
            # warped masks is wider - ownership can leave holes there.)
            if ownership_mask is not None:
                final_valid_mask = (ownership_mask >= 0).astype(np.uint8)
            else:
                final_valid_mask = np.zeros_like(warped_masks[0])
                for m in warped_masks:
                    final_valid_mask = np.maximum(final_valid_mask, (m > 0).astype(np.uint8))

            crop_rect = self.cropper.find_crop_rect(final_valid_mask)
            if crop_rect is not None:
                x, y, cw, ch = crop_rect
                final = blended[y : y + ch, x : x + cw]
                ownership_cropped = ownership_mask[y : y + ch, x : x + cw]
            else:
                final = self.cropper.crop(blended)
                ownership_cropped = ownership_mask

            # Optional post-processing according to the quality mode.
            # "natural" (default) applies nothing at all.
            final = self._apply_postprocess(final)
            report["steps"]["final_size"] = final.shape if final is not None else None

            if debug_logger:
                debug_logger.save_final_crop(final)
                # Quality-check stage D (final) + diagnostics vs the original frame.
                debug_logger.save_quality_stage("04_final", final)
                if final is not None:
                    q_orig = self._quality_metrics(active_frames[0])
                    q_final = self._quality_metrics(final)
                    quality_report = {"original": q_orig, "final": q_final}
                    debug_logger.save_quality_metrics(quality_report)
                    report["steps"]["quality_metrics"] = quality_report
                    edge_ratio = (
                        q_final["edge_density"] / q_orig["edge_density"]
                        if q_orig["edge_density"] > 0
                        else 0.0
                    )
                    print(
                        f"[QUALITY] edge_density: orig={q_orig['edge_density']:.4f} "
                        f"final={q_final['edge_density']:.4f} (x{edge_ratio:.2f}) | "
                        f"local_contrast: orig={q_orig['local_contrast']:.2f} "
                        f"final={q_final['local_contrast']:.2f} | "
                        f"noise: orig={q_orig['noise_estimate']:.2f} "
                        f"final={q_final['noise_estimate']:.2f} | "
                        f"clipped_black={q_final['clipped_black_ratio'] * 100:.2f}% "
                        f"clipped_white={q_final['clipped_white_ratio'] * 100:.2f}%",
                        flush=True,
                    )
                    if edge_ratio > 1.5:
                        print(
                            "[QUALITY WARNING] 最终图 edge_density 超过原图 1.5 倍，"
                            "可能存在过度锐化或所有权碎片化",
                            flush=True,
                        )

            # Multi-frame contribution validation.
            if final is not None:
                validation = self._validate_multi_frame_contribution(
                    final, ownership_cropped, active_frames, canvas_bounds
                )
                report["steps"]["validation"] = validation

                contributing = sum(
                    1
                    for c in validation["frame_contributions"].values()
                    if c["contributed_pixels"] > 0
                )
                report["frame_counts"]["contributing_frames"] = contributing

                if debug_logger:
                    contributing_pairs = [
                        active_pairs[i]
                        for i, c in validation["frame_contributions"].items()
                        if c["contributed_pixels"] > 0 and i < len(active_pairs)
                    ]
                    debug_logger.save_contributing_frames(contributing_pairs)

                if not validation["is_valid"]:
                    report["steps"]["validation_warnings"] = validation["warnings"]
                    # If validation fails, try to fix it by resampling keyframes.
                    if len(keyframes) < self.config.target_keyframes:
                        return self._retry_with_more_keyframes(
                            video_path, report, debug_logger, start_time, final
                        )

            # Classify the result honestly.
            status = self._classify_stitch_status(n_active)
            report["stitch_status"] = status

            report["success"] = True
            if status == "LIMITED_STITCH":
                report["message"] = (
                    f"关键帧不足：目标 {self.config.target_keyframes}，"
                    f"实际参与拼接 {n_active}"
                )
                report["level"] = 2
            else:
                report["message"] = "拼接成功"
                report["level"] = 1
            report["image"] = final
            report["processing_time"] = time.time() - start_time
            report["source_map"] = source_map

            if debug_logger:
                bridge_info = report["steps"].get("bridge") or {}
                completeness = report.get("completeness") or {}
                debug_logger.save_continuity_report({
                    "video_duration": completeness.get("video_duration", 0),
                    "first_used_time": completeness.get("first_used_time", 0),
                    "last_used_time": completeness.get("last_used_time", 0),
                    "time_coverage_ratio": completeness.get("time_coverage_ratio", 0),
                    "selected_keyframes": report["frame_counts"].get("selected_frames", 0),
                    "active_stitch_frames": n_active,
                    "contributing_frames": report["frame_counts"].get("contributing_frames", 0),
                    "bridge_recovery_used": bridge_info.get("bridge_recovery_used", False),
                    "detected_breaks": bridge_info.get("detected_breaks", 0),
                    "continuity_breaks": completeness.get("continuity_breaks", 0),
                    "coverage_completeness": completeness.get("time_coverage_ratio", 0),
                    "partial_failure": completeness.get("partial_failure", False),
                })
                debug_logger.save_report(report)

            return report

        except Exception as e:
            # Level 3: fallback on any error.
            return self._fallback_sharpest_frame(
                video_path, report, debug_logger, start_time, error=str(e)
            )

    # ------------------------------------------------------------------
    # Keyframe selection with retry / recovery
    # ------------------------------------------------------------------
    def _verify_auto_keyframes(
        self, video_path: str, recommended: int, total_frames: int
    ) -> int:
        """Verify the AUTO estimate against real matcher feasibility.

        A frame count is only usable if consecutive keyframes at that spacing
        can actually be matched.  Probe pairs at the proposed spacing; if
        matching fails, densify (more keyframes = smaller gaps) and re-probe.
        """
        n = max(3, recommended)
        max_auto = self.config.max_auto_keyframes
        if total_frames <= 0:
            return n

        def _read_frame(cap, idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            return frame if ret else None

        while n <= max_auto:
            gap = max(1, (total_frames - 1) // max(n - 1, 1))
            probe_ids = [0, gap, 2 * gap, 3 * gap]
            probe_ids = [min(i, total_frames - 1) for i in probe_ids]

            cap = cv2.VideoCapture(video_path)
            frames = [_read_frame(cap, i) for i in probe_ids]
            cap.release()
            frames = [f for f in frames if f is not None]

            if len(frames) < 2:
                return n

            ok = True
            for k in range(1, len(frames)):
                M, info = self.matcher.estimate_similarity(frames[k - 1], frames[k])
                if M is None:
                    M, info = self.matcher.estimate_affine(frames[k - 1], frames[k])
                if M is None:
                    ok = False
                    break

            if ok:
                if n != recommended:
                    print(
                        f"[AUTO VERIFY] spacing OK at {n} keyframes "
                        f"(gap={gap} frames), matcher confirmed",
                        flush=True,
                    )
                return n

            new_n = min(n + 4, max_auto + 4)
            print(
                f"[AUTO VERIFY] {n} keyframes too sparse to match "
                f"(gap={gap} frames), densify -> {min(new_n, max_auto)}",
                flush=True,
            )
            n = new_n

        return max_auto

    def _bridge_extend(
        self,
        video_path: str,
        keyframes: List[Tuple[int, np.ndarray]],
        active_indices: List[int],
        total_frames: int,
        coverage_target: float = 0.8,
    ) -> Tuple[List[Tuple[int, np.ndarray]], List[int], Dict]:
        """Extend the active frame path along the timeline (bridge recovery).

        The span path often covers only part of the video.  This walks the
        tail (then the head), attaching the next usable keyframe or inserting
        intermediate video frames as bridges (each link verified by the
        geometrically-validated estimator) until the active path covers
        ``coverage_target`` of the video timeline or nothing more attaches.

        Returns (keyframes, active_indices, bridge_info).
        """
        info: Dict = {
            "bridge_recovery_used": False,
            "detected_breaks": 0,
            "bridged_frame_ids": [],
        }
        if total_frames <= 1 or len(active_indices) < 2:
            return keyframes, active_indices, info

        active_pairs = sorted(
            (keyframes[i] for i in active_indices), key=lambda p: p[0]
        )
        active_ids = {p[0] for p in active_pairs}
        excluded_pairs = sorted(
            (p for p in keyframes if p[0] not in active_ids), key=lambda p: p[0]
        )

        gaps = [
            active_pairs[k + 1][0] - active_pairs[k][0]
            for k in range(len(active_pairs) - 1)
        ]
        median_gap = max(int(np.median(gaps)) if gaps else 8, 1)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return keyframes, active_indices, info

        def read_frame(idx: int) -> Optional[np.ndarray]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            return frame if ret else None

        def link_ok(fa: np.ndarray, fb: np.ndarray) -> bool:
            M, _, _ = self.estimator.estimate(fa, fb)
            return M is not None

        def try_chain(id_a, fa, id_b, fb, n_mid):
            """Try a->mids->b chain; returns bridge (id, frame) list or None."""
            if link_ok(fa, fb):
                return []
            mids = np.linspace(id_a, id_b, n_mid + 2, dtype=int)[1:-1]
            prev = fa
            chain = []
            for mid in mids:
                mf = read_frame(int(mid))
                if mf is None or not link_ok(prev, mf):
                    return None
                chain.append((int(mid), mf))
                prev = mf
            if not link_ok(prev, fb):
                return None
            return chain

        def coverage() -> float:
            span = active_pairs[-1][0] - active_pairs[0][0]
            return span / max(total_frames - 1, 1)

        # ---- Extend the tail ----
        while coverage() < coverage_target:
            tail_id, tail_f = active_pairs[-1]
            cand = next((p for p in excluded_pairs if p[0] > tail_id), None)
            if cand is None:
                new_id = tail_id + median_gap
                if new_id >= total_frames:
                    break
                f = read_frame(new_id)
                if f is None:
                    break
                cand = (int(new_id), f)

            chain = None
            for n_mid in (0, 2, 4):
                chain = try_chain(tail_id, tail_f, cand[0], cand[1], n_mid)
                if chain is not None:
                    break
            if chain is None:
                # Candidate unusable: drop it and try the next one.
                excluded_pairs = [p for p in excluded_pairs if p[0] != cand[0]]
                info["detected_breaks"] += 1
                if cand[0] not in {p[0] for p in excluded_pairs} and cand[0] > tail_id + median_gap * 3:
                    break
                continue

            for b in chain:
                active_pairs.append(b)
                info["bridged_frame_ids"].append(b[0])
            active_pairs.append(cand)
            excluded_pairs = [p for p in excluded_pairs if p[0] != cand[0]]
            info["bridge_recovery_used"] = True

        # ---- Extend the head ----
        while coverage() < coverage_target:
            head_id, head_f = active_pairs[0]
            cand = next((p for p in reversed(excluded_pairs) if p[0] < head_id), None)
            if cand is None:
                new_id = head_id - median_gap
                if new_id < 0:
                    break
                f = read_frame(new_id)
                if f is None:
                    break
                cand = (int(new_id), f)

            chain = None
            for n_mid in (0, 2, 4):
                chain = try_chain(cand[0], cand[1], head_id, head_f, n_mid)
                if chain is not None:
                    break
            if chain is None:
                excluded_pairs = [p for p in excluded_pairs if p[0] != cand[0]]
                info["detected_breaks"] += 1
                if head_id - median_gap * 3 < 0:
                    break
                continue

            for b in chain:
                active_pairs.append(b)
                info["bridged_frame_ids"].append(b[0])
            active_pairs.append(cand)
            active_pairs.sort(key=lambda p: p[0])
            excluded_pairs = [p for p in excluded_pairs if p[0] != cand[0]]
            info["bridge_recovery_used"] = True

        cap.release()
        active_pairs.sort(key=lambda p: p[0])
        info["final_coverage"] = round(coverage(), 3)
        info["active_after_bridge"] = len(active_pairs)
        print(
            f"[BRIDGE] used = {info['bridge_recovery_used']} | "
            f"breaks = {info['detected_breaks']} | "
            f"bridged = {len(info['bridged_frame_ids'])} frames | "
            f"coverage -> {info['final_coverage']}",
            flush=True,
        )

        # Rebuild the keyframe list (active + remaining excluded, time sorted).
        active_ids = {p[0] for p in active_pairs}
        merged = sorted(
            list(active_pairs) + [p for p in excluded_pairs if p[0] not in active_ids],
            key=lambda p: p[0],
        )
        new_active_indices = [
            i for i, p in enumerate(merged) if p[0] in active_ids
        ]
        return merged, new_active_indices, info

    def _select_keyframes_with_retry(
        self,
        video_path: str,
        report: Dict,
        progress_callback: Optional[Callable[[int, int], bool]] = None,
        relax_level: int = 0,
    ) -> Tuple[List[Tuple[int, np.ndarray]], Dict]:
        """Select keyframes with retry logic for insufficient frames.

        ``relax_level`` widens the net when the previous attempt produced too
        few frames to satisfy the target:

            0 = configured thresholds
            1 = blur threshold lowered 40%, target raised to max_keyframes
            2 = blur threshold lowered 60%, uniform timeline sampling
        """
        cfg = self.config
        stats: Dict = {}

        if relax_level == 0:
            blur = cfg.blur_threshold
            target = cfg.target_keyframes
        elif relax_level == 1:
            blur = cfg.blur_threshold * 0.6
            target = cfg.max_keyframes
        else:
            blur = cfg.blur_threshold * 0.4
            target = cfg.max_keyframes

        print(f"[EXTRACTOR CONFIG] target_keyframes = {target} | "
              f"blur_threshold = {blur:.1f} | relax_level = {relax_level}", flush=True)

        keyframes = select_keyframes(
            video_path,
            blur_threshold=blur,
            min_brightness=cfg.min_brightness,
            max_brightness=cfg.max_brightness,
            target_keyframes=target,
            maximum_keyframes=cfg.max_keyframes,
            progress_callback=progress_callback,
            stats=stats,
        )

        if relax_level == 0:
            report["initial_keyframe_count"] = len(keyframes)
        else:
            report["retry_keyframe_count"] = len(keyframes)

        if len(keyframes) >= cfg.min_keyframes:
            return keyframes, stats

        # Last resort: uniformly sample sharp frames from the video timeline.
        keyframes = self._uniform_sample_sharp_frames(
            video_path, max(cfg.target_keyframes, cfg.min_keyframes), progress_callback
        )
        stats["uniform_fallback_used"] = True
        stats["selected_frames"] = len(keyframes)
        return keyframes, stats

    def _uniform_sample_sharp_frames(
        self,
        video_path: str,
        count: int,
        progress_callback: Optional[Callable[[int, int], bool]] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """Uniformly sample sharp frames from the video timeline."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        # Sample uniformly from the video timeline.
        indices = np.linspace(0, total - 1, count, dtype=int).tolist()
        keyframes = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Check if frame is sharp enough.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if lap_var >= 80.0:
                    keyframes.append((idx, frame))

            if progress_callback and not progress_callback(len(keyframes), count):
                break

        cap.release()
        return keyframes

    # ------------------------------------------------------------------
    # Canvas / validation helpers
    # ------------------------------------------------------------------
    def _warp_frames_to_canvas(
        self,
        frames: List[np.ndarray],
        poses: List[np.ndarray],
        bounds: Dict,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Warp all frames to the unified canvas with binary validity masks.

        One single resampling per frame: image INTER_LINEAR, mask
        INTER_NEAREST (same H, same canvas).  No erosion, no morphology -
        edge handling is the composer's job, not the mask's.
        """
        offset_x = -bounds["min_x"]
        offset_y = -bounds["min_y"]
        canvas_w = bounds["width"]
        canvas_h = bounds["height"]

        warped_frames = []
        warped_masks = []

        for frame, pose in zip(frames, poses):
            # Add offset to pose.
            pose_canvas = pose.copy()
            pose_canvas[0, 2] += offset_x
            pose_canvas[1, 2] += offset_y

            # Warp frame (single resampling, natural interpolation).
            warped = cv2.warpPerspective(
                frame,
                pose_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            # Warp the validity mask with NEAREST interpolation: strictly
            # binary, validity comes from the mask alone (never brightness).
            mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
            warped_mask = cv2.warpPerspective(
                mask,
                pose_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            warped_mask = np.where(warped_mask > 127, 255, 0).astype(np.uint8)

            warped_frames.append(warped)
            warped_masks.append(warped_mask)

        return warped_frames, warped_masks

    def _frame_centers_px(
        self,
        frame_shapes: List[Tuple[int, int]],
        poses: List[np.ndarray],
        bounds: Dict,
    ) -> List[Tuple[float, float]]:
        """Project each frame's center onto the canvas (for path debug/axis)."""
        offset_x = -bounds["min_x"]
        offset_y = -bounds["min_y"]
        centers = []
        for shape, pose in zip(frame_shapes, poses):
            h, w = shape[:2]
            H = pose.astype(np.float64).copy()
            H[0, 2] += offset_x
            H[1, 2] += offset_y
            c = H @ np.array([w / 2.0, h / 2.0, 1.0])
            centers.append((float(c[0]), float(c[1])))
        return centers

    def _detect_continuity_break(
        self, ownership_mask: np.ndarray, axis: np.ndarray, min_gap_px: int = 10
    ) -> int:
        """Count interior content gaps along the main axis (continuity breaks).

        Projects the valid mask onto the dominant canvas direction; any run
        of empty columns/rows >= min_gap_px inside the valid range counts as
        one break (two big blobs instead of one continuous scan).
        """
        if ownership_mask is None:
            return 0
        valid = ownership_mask >= 0
        if abs(axis[0]) >= abs(axis[1]):
            profile = valid.any(axis=0)
        else:
            profile = valid.any(axis=1)

        present = np.where(profile)[0]
        if len(present) == 0:
            return 0

        breaks = 0
        gap = 0
        for k in range(present[0], present[-1] + 1):
            if not profile[k]:
                gap += 1
                if gap == min_gap_px:
                    breaks += 1
            else:
                gap = 0
        return breaks

    def _classify_stitch_status(self, n_active: int) -> str:
        """Classify the stitch result by how many frames really participated."""
        if n_active <= 1:
            return "SINGLE_FRAME_FALLBACK"
        if n_active == 2:
            return "LIMITED_STITCH"
        if n_active < self.config.hard_min_active_frames:
            return "MULTI_FRAME_STITCH"
        return "NORMAL_MULTI_FRAME_STITCH"

    def _apply_postprocess(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply the (single) post-processing step selected by quality_mode.

        - natural:  nothing - the pixels stay exactly as blended.
        - standard: one very light denoise (fastNlMeans, h=3).
        - enhanced: standard + one very mild unsharp mask (amount=0.15).
        Never more than one pass of each, never before blending is finished.
        """
        if image is None:
            return image
        mode = getattr(self.config, "quality_mode", "natural")
        if mode == "natural":
            return image

        result = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        if mode == "enhanced":
            blurred = cv2.GaussianBlur(result, (0, 0), 2.0)
            result = cv2.addWeighted(result, 1.15, blurred, -0.15, 0)
        return result

    @staticmethod
    def _quality_metrics(image: np.ndarray) -> Dict:
        """Compute image-quality diagnostics for artifact localization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total = float(gray.size)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / total

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        local_contrast = float(np.std(lap))

        # Noise estimate: std of the high-frequency residual.
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
        noise_estimate = float(np.std(residual))

        clipped_black = float(np.count_nonzero(gray <= 2)) / total
        clipped_white = float(np.count_nonzero(gray >= 253)) / total

        return {
            "edge_density": edge_density,
            "local_contrast": local_contrast,
            "noise_estimate": noise_estimate,
            "clipped_black_ratio": clipped_black,
            "clipped_white_ratio": clipped_white,
        }

    def _fill_coverage_holes(
        self,
        ownership_mask: np.ndarray,
        content_masks: List[np.ndarray],
        max_iters: int = 64,
    ) -> Tuple[np.ndarray, int, int]:
        """Reassign every ownerless-but-covered pixel to a real source frame.

        holes = pixels inside the union of all content masks whose ownership
        is -1.  They are filled by iteratively growing the known ownership
        regions into the holes (nearest owner wins), but only into pixels
        where the candidate frame actually has content.  Leftovers (no nearby
        owner) go to any frame with content there.  Returns
        (filled_ownership, holes_before, holes_after).
        """
        if ownership_mask is None:
            return ownership_mask, 0, 0

        union = np.zeros(ownership_mask.shape, dtype=np.uint8)
        for m in content_masks:
            union = np.maximum(union, (m > 0).astype(np.uint8))

        holes = (union > 0) & (ownership_mask < 0)
        holes_before = int(holes.sum())
        if holes_before == 0:
            return ownership_mask, 0, 0

        filled = ownership_mask.copy()
        remaining = holes.copy()
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Nearest-owner dilation fill (each iteration grows regions by 1px).
        for _ in range(max_iters):
            if not remaining.any():
                break
            progressed = False
            for i, m in enumerate(content_masks):
                owned = (filled == i).astype(np.uint8)
                grow = (
                    cv2.dilate(owned, kernel, iterations=1).astype(bool)
                    & remaining
                    & (m > 0)
                )
                if grow.any():
                    filled[grow] = i
                    remaining[grow] = False
                    progressed = True
            if not progressed:
                break

        # Last resort: any frame with content at that pixel.
        if remaining.any():
            for i, m in enumerate(content_masks):
                take = remaining & (m > 0)
                if take.any():
                    filled[take] = i
                    remaining[take] = False

        holes_after = int(((union > 0) & (filled < 0)).sum())
        return filled, holes_before, holes_after

    def _bfs_global_poses(
        self,
        edges: Dict[Tuple[int, int], Dict],
        n_frames: int,
        ref_idx: int,
    ) -> List[Optional[np.ndarray]]:
        """Compute global poses by BFS from the reference frame over the pose graph.

        Edge (i, j) stores H mapping frame j into frame i's coordinates.
        Chaining: pose[j] = pose[i] @ H_ij; traversing backwards uses inv(H_ij).
        Shortest hop count wins; ties prefer the higher inlier ratio, so a
        single bad consecutive edge can always be routed around via i+2/i+3.
        """
        adjacency: Dict[int, List[Tuple[int, np.ndarray, float]]] = {
            i: [] for i in range(n_frames)
        }
        for (i, j), e in edges.items():
            H = e["H"]
            ratio = float(e["info"].get("inlier_ratio", 0.0))
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                continue
            adjacency[i].append((j, H, ratio))        # pose[j] = pose[i] @ H
            adjacency[j].append((i, H_inv, ratio))    # pose[i] = pose[j] @ H_inv

        poses: List[Optional[np.ndarray]] = [None] * n_frames
        poses[ref_idx] = np.eye(3, dtype=np.float64)
        # (hops, -inlier_ratio, node)
        heap = [(0, 0.0, ref_idx)]
        best_cost = {ref_idx: (0, 0.0)}

        while heap:
            hops, neg_ratio, node = heapq.heappop(heap)
            for neighbor, H_step, ratio in adjacency[node]:
                new_cost = (hops + 1, neg_ratio - ratio)
                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    poses[neighbor] = poses[node] @ H_step
                    heapq.heappush(heap, (hops + 1, neg_ratio - ratio, neighbor))

        return poses

    def _draw_projected_boxes(
        self,
        frames: List[np.ndarray],
        poses: List[np.ndarray],
        bounds: Dict,
    ) -> np.ndarray:
        """Draw every frame's projected quadrilateral on the canvas (debug)."""
        canvas = np.zeros((bounds["height"], bounds["width"], 3), dtype=np.uint8)
        offset_x = -bounds["min_x"]
        offset_y = -bounds["min_y"]
        colors = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80),
            (255, 80, 255), (80, 255, 255), (255, 160, 80), (160, 80, 255),
        ]
        for i, (frame, pose) in enumerate(zip(frames, poses)):
            H = pose.copy()
            H[0, 2] += offset_x
            H[1, 2] += offset_y
            pts = project_corners(H, frame.shape).astype(np.int32)
            color = colors[i % len(colors)]
            cv2.polylines(canvas, [pts], True, color, 3)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(
                canvas, str(i), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3, cv2.LINE_AA,
            )
        return canvas

    def _validate_multi_frame_contribution(
        self,
        final: np.ndarray,
        ownership_mask: np.ndarray,
        frames: List[np.ndarray],
        canvas_bounds: Dict,
    ) -> Dict:
        """Validate that multiple frames contribute to the final image."""
        canvas_h, canvas_w = final.shape[:2]
        total_area = canvas_h * canvas_w

        # Compute contribution of each frame.
        frame_contributions = {}
        for i in range(len(frames)):
            contributed_pixels = (ownership_mask == i).sum()
            contribution_ratio = contributed_pixels / total_area if total_area > 0 else 0.0
            frame_contributions[i] = {
                "contributed_pixels": int(contributed_pixels),
                "contribution_ratio": float(contribution_ratio),
            }

        # Check validation conditions.
        warnings = []

        # Condition 1: active_stitch_frames >= 2.
        if len(frames) < 2:
            warnings.append("参与拼接的帧数少于 2 张")

        # Condition 2: at least two frames contribute > 5% of final area.
        significant_frames = sum(
            1 for c in frame_contributions.values()
            if c["contribution_ratio"] >= self.config.minimum_frame_contribution_ratio
        )
        if significant_frames < 2:
            warnings.append(
                f"只有 {significant_frames} 张帧贡献超过 {self.config.minimum_frame_contribution_ratio * 100:.0f}% 的最终面积"
            )

        # Condition 3: final canvas area is significantly larger than single frame.
        # For a horizontal panning video, the panorama should be wider than a single frame,
        # not necessarily larger in area.  We check width or height expansion instead.
        single_frame_area = frames[0].shape[0] * frames[0].shape[1]
        expansion_ratio = total_area / single_frame_area if single_frame_area > 0 else 0.0

        # Check if the panorama is wider or taller than a single frame.
        single_frame_h, single_frame_w = frames[0].shape[:2]
        is_wider = canvas_bounds["width"] > single_frame_w
        is_taller = canvas_bounds["height"] > single_frame_h

        if not (is_wider or is_taller or expansion_ratio >= 1.05):
            warnings.append(
                f"最终画布面积扩展比例 {expansion_ratio:.2f} 小于 1.05，且宽度/高度没有明显扩展"
            )

        # Condition 4: source map has at least two frame_ids.
        unique_frames = len(np.unique(ownership_mask[ownership_mask >= 0]))
        if unique_frames < 2:
            warnings.append(f"source map 中只有 {unique_frames} 个不同的 frame_id")

        # Condition 5: non-reference frame contributes at least one continuous valid region.
        # (This is implicitly checked by the contribution ratio.)

        # Condition 6: no large transparent average regions.
        # (This is implicitly checked by the ownership mask.)

        is_valid = len(warnings) == 0

        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "frame_contributions": frame_contributions,
            "expansion_ratio": expansion_ratio,
            "unique_frames": unique_frames,
        }

    def _retry_with_more_keyframes(
        self,
        video_path: str,
        report: Dict,
        debug_logger: Optional[DebugLogger],
        start_time: float,
        final_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """Retry stitching with more keyframes."""
        report["steps"]["retry_reason"] = "多帧贡献验收失败，重新采样关键帧"
        # For now, just return the current result with a warning.
        report["success"] = True
        report["message"] = "拼接成功，但多帧贡献不足"
        report["level"] = 2
        report["image"] = final_image
        report["processing_time"] = time.time() - start_time
        return report

    def _fallback_sharpest_frame(
        self,
        video_path: str,
        report: Dict,
        debug_logger: Optional[DebugLogger],
        start_time: float,
        error: Optional[str] = None,
    ) -> Dict:
        """Level 3 fallback: use the sharpest frame from the video."""
        reason = error if error else "匹配不足"
        print(f"[FALLBACK ACTIVATED] reason = {reason} | "
              f"frames_before_fallback = {report.get('final_keyframe_count', 0)} | "
              f"frames_after_fallback = 1", flush=True)
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                report["message"] = f"无法打开视频: {video_path}"
                return report

            best_frame = None
            best_score = 0.0
            idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()

                idx += 1
                if idx > 100:  # Limit search to first 100 frames.
                    break

            cap.release()

            if best_frame is not None:
                report["success"] = True
                report["message"] = f"本次使用了单帧保底（{reason}）"
                report["image"] = best_frame
                report["level"] = 3
                report["stitch_status"] = "SINGLE_FRAME_FALLBACK"
                report["fallback_reason"] = reason
                report["processing_time"] = time.time() - start_time
                return report

            report["message"] = f"无法提取任何帧: {video_path}"
            return report

        except Exception as e:
            report["message"] = f"降级输出失败: {e}"
            return report
