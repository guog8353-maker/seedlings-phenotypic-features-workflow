"""Single source of truth for all stitching pipeline parameters.

Every module (UI, worker, extractor, stitcher) must read parameters from one
StitchConfig instance.  No function is allowed to re-define its own
target_keyframes / max_frames defaults that silently override the UI value.
"""

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


class KeyframePipelineError(Exception):
    """Raised when the actual stitch input count severely mismatches the target."""


@dataclass
class StitchConfig:
    """Authoritative configuration for the whole stitching pipeline."""

    # ---- Keyframe counts -------------------------------------------------
    # AUTO mode: every video is pre-analyzed (spatial coverage, overlap,
    # motion) and gets its own recommended keyframe count.  When False,
    # target_keyframes is used as a fixed value for all videos.
    auto_keyframes: bool = True
    target_keyframes: int = 12
    min_keyframes: int = 3
    max_keyframes: int = 24
    # AUTO recommendation bounds and target overlap between neighbors.
    target_overlap: float = 0.60
    min_auto_keyframes: int = 4
    max_auto_keyframes: int = 20
    # Hard lower bound: if the video is long enough but fewer frames than this
    # actually enter the stitcher, keyframe recovery is triggered.
    hard_min_active_frames: int = 6
    # Strict multi-frame mode: insufficient frames must be reported as such,
    # never silently presented as a successful stitch.
    strict_multi_frame_mode: bool = False

    # ---- Frame quality filtering -----------------------------------------
    blur_threshold: float = 100.0
    min_brightness: float = 30.0
    max_brightness: float = 225.0

    # ---- Feature matching ------------------------------------------------
    feature_type: str = "orb"
    max_features: int = 300
    ratio_test: float = 0.75
    min_good_matches: int = 12
    min_inliers: int = 8
    min_inlier_ratio: float = 0.2
    ransac_reprojection_threshold: float = 3.0
    max_deformation: float = 2.0

    # ---- Blending / compositing ------------------------------------------
    num_bands: int = 3
    seam_method: str = "voronoi"
    exposure_method: str = "gain"
    crop_mode: str = "rect"
    seam_blend_width: int = 8

    # ---- Keyframe selection thresholds ------------------------------------
    min_new_area_ratio: float = 0.12
    preferred_new_area_ratio: float = 0.20
    duplicate_similarity_threshold: float = 0.96
    minimum_overlap_ratio: float = 0.40
    maximum_overlap_ratio: float = 0.75
    minimum_frame_contribution_ratio: float = 0.05

    # ---- Geometric sanity limits (plane scanning) -------------------------
    # Per-pair transform validation: anything outside these bounds is rejected.
    min_scale: float = 0.75
    max_scale: float = 1.35
    min_aspect_ratio: float = 0.8      # scale_x / scale_y
    max_aspect_ratio: float = 1.25
    max_rotation_deg: float = 12.0     # per consecutive pair
    min_area_ratio: float = 0.6        # projected quad area / original area
    max_area_ratio: float = 1.6
    max_perspective: float = 0.0015    # |h31| + |h32|
    max_corner_displacement: float = 0.5   # fraction of frame diagonal
    max_condition_number: float = 1e6
    # Global (accumulated) pose limits, more generous than per-pair limits.
    global_max_rotation_deg: float = 25.0
    global_min_scale: float = 0.6
    global_max_scale: float = 1.6
    # Median-motion outlier factor: a pair whose translation deviates more
    # than this factor from the median motion is treated as an outlier edge.
    motion_outlier_factor: float = 3.0
    # Grid-uniform feature distribution (structure over plant texture).
    feature_grid_rows: int = 8
    feature_grid_cols: int = 8
    max_features_per_cell: int = 40
    min_spread_cells: int = 4          # inliers must cover >= N grid cells
    # ---- Composition strategy ----------------------------------------------
    # SafeStripComposer: strip ownership along the main motion axis
    # (phone-panorama style).  Set False to use the generic pixel-level
    # quality ownership instead.
    strip_compose_enabled: bool = True

    # ---- Output quality mode ----------------------------------------------
    # "natural": no post-processing at all (default, like a phone photo).
    # "standard": one very light denoise after stitching.
    # "enhanced": standard + one very mild unsharp mask.
    quality_mode: str = "natural"
    # Local warping is disabled by default: geometric stability first.
    local_warp_enabled: bool = False

    # ---- Fallback ----------------------------------------------------------
    single_frame_fallback_enabled: bool = True
    single_frame_fallback_only_after_all_retries: bool = True

    # ---- Debug -------------------------------------------------------------
    max_display_size: int = 1024
    debug_dir: Optional[str] = None  # None disables on-disk debug output

    def apply_overrides(self, **overrides) -> "StitchConfig":
        """Apply keyword overrides in-place (only known field names)."""
        known = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in known and value is not None:
                setattr(self, key, value)
        return self

    def log(self, tag: str) -> None:
        """Print the authoritative keyframe parameters with a stage tag.

        The four tags UI / WORKER / EXTRACTOR / STITCHER must all print the
        same target; a mismatch means the configuration is out of sync.
        """
        print(
            f"[{tag} CONFIG] target_keyframes = {self.target_keyframes} | "
            f"min_keyframes = {self.min_keyframes} | "
            f"max_keyframes = {self.max_keyframes} | "
            f"hard_min_active_frames = {self.hard_min_active_frames} | "
            f"strict_multi_frame_mode = {self.strict_multi_frame_mode}",
            flush=True,
        )


def default_debug_root() -> str:
    """Default on-disk debug root: <cwd>/debug (cwd is the project dir when
    launched via 启动软件.cmd)."""
    return str(Path.cwd() / "debug")
