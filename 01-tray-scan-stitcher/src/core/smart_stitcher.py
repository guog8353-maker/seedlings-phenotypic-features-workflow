"""SmartStitcher: THE single entry point for processing one video.

Single-video mode calls ``process_video`` once; batch mode calls it in a
loop.  There is exactly one stitching implementation - this class wraps the
shared StitchingPipeline with AUTO analysis, quality evaluation, automatic
retries and per-video report output.  No second stitching algorithm exists.
"""

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .stitching import StitchConfig, StitchingPipeline
from .utils import save_image


def _make_json_safe(obj):
    """Recursively convert numpy values for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    return obj


class SmartStitcher:
    """Process one video end-to-end with the shared intelligent pipeline."""

    def __init__(
        self,
        config: Optional[StitchConfig] = None,
        output_format: str = "png",
        jpeg_quality: int = 95,
    ):
        self.config = config if config is not None else StitchConfig()
        self.output_format = output_format.lower().lstrip(".")
        self.jpeg_quality = jpeg_quality

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], bool]] = None,
    ) -> Dict:
        """Analyze, stitch, evaluate and (if needed) retry one video.

        Returns a unified result dict: the pipeline report fields plus
        quality_score / attempts / output / status / failure_reason.
        """
        start = time.time()
        stem = Path(video_path).stem
        out_dir = Path(output_dir) / stem if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        attempts: List[Tuple[int, Dict, str]] = []
        # Retry order: default -> densify (more keyframes, smaller gaps; the
        # right answer to matching failures) -> strict (geometry clampdown;
        # the right answer to distortion failures).  Stop as soon as an
        # attempt scores >= 80.
        tweaks = ["default", "densify", "strict"]

        for attempt_no, tweak in enumerate(tweaks, 1):
            run_config = self._tweak_config(self.config, tweak, attempts)
            if out_dir:
                run_config = replace(
                    run_config, debug_dir=str(out_dir / "debug")
                )
            print(f"[ATTEMPT {attempt_no}] strategy = {tweak}", flush=True)

            pipeline = StitchingPipeline(config=run_config)
            result = pipeline.stitch(video_path, progress_callback=progress_callback)
            score = self.evaluate_stitch_quality(result)
            result["quality_score"] = score
            attempts.append((score, result, tweak))
            print(f"[ATTEMPT {attempt_no}] quality_score = {score}", flush=True)

            if score >= 80:
                break

        # Pick the best attempt, never simply the last one.
        attempts.sort(key=lambda a: a[0], reverse=True)
        best_score, best, best_tweak = attempts[0]
        best["attempts"] = [
            {"attempt": i + 1, "strategy": t, "quality_score": s}
            for i, (s, _, t) in enumerate(attempts)
        ]
        best["best_strategy"] = best_tweak
        best["processing_time_total"] = time.time() - start

        # Save outputs.
        if out_dir and best.get("success") and best.get("image") is not None:
            is_fallback = best.get("level", 0) >= 3
            name = f"stitched_FALLBACK.{self.output_format}" if is_fallback \
                else f"stitched.{self.output_format}"
            output_path = out_dir / name
            ok = save_image(output_path, best["image"], quality=self.jpeg_quality)
            if ok:
                best["output"] = str(output_path)

            # Keep losing attempts for debugging.
            if len(attempts) > 1:
                att_dir = out_dir / "debug_attempts"
                att_dir.mkdir(exist_ok=True)
                for i, (s, r, t) in enumerate(attempts[1:], 2):
                    if r.get("image") is not None:
                        save_image(
                            att_dir / f"attempt_{i}_{t}_score{s}.png", r["image"]
                        )

        if out_dir:
            self._write_report(best, out_dir / "report.json")

        # Flatten batch-friendly fields.
        counts = best.get("frame_counts") or {}
        analysis = best.get("video_analysis") or {}
        best.update({
            "filename": Path(video_path).name,
            "recommended_keyframes": analysis.get(
                "recommended_keyframes", counts.get("target_keyframes")
            ),
            "selected_keyframes": counts.get("selected_frames", 0),
            "active_frames": counts.get("active_stitch_frames", 0),
            "contributing_frames": counts.get("contributing_frames", 0),
            "motion_type": analysis.get("motion_direction", ""),
            "stitchability": analysis.get("stitchability", ""),
            "status": "完成" if best.get("success") else "失败",
            "failure_reason": "" if best.get("success") else best.get("message", ""),
        })
        return best

    # ------------------------------------------------------------------
    # Quality evaluation & retry strategy
    # ------------------------------------------------------------------
    def evaluate_stitch_quality(self, result: Dict) -> int:
        """Score a stitch result 0-100 (>=80 accept, <80 retry, <60 escalate)."""
        if not result.get("success") or result.get("image") is None:
            return 0

        score = 100.0
        status = result.get("stitch_status", "")
        score -= {
            "NORMAL_MULTI_FRAME_STITCH": 0,
            "MULTI_FRAME_STITCH": 10,
            "LIMITED_STITCH": 35,
            "SINGLE_FRAME_FALLBACK": 60,
        }.get(status, 10)

        counts = result.get("frame_counts") or {}
        if counts.get("contributing_frames", 0) < 3:
            score -= 15

        steps = result.get("steps") or {}
        validation = steps.get("validation") or {}
        warnings = validation.get("warnings") or []
        score -= min(8 * len(warnings), 24)

        expansion = steps.get("expansion") or {}
        if not expansion.get("is_expanded", True):
            score -= 20

        quality = steps.get("quality_metrics") or {}
        orig = quality.get("original") or {}
        fin = quality.get("final") or {}
        if orig.get("edge_density", 0) > 0:
            edge_ratio = fin.get("edge_density", 0) / orig["edge_density"]
            if edge_ratio > 1.5:
                score -= 15
            elif edge_ratio > 1.3:
                score -= 5
        clipped = fin.get("clipped_black_ratio", 0) + fin.get("clipped_white_ratio", 0)
        if clipped > 0.01:
            score -= 10

        # ---- Completeness is the TOP priority -----------------------------
        # A pretty but partial stitch is a failure, not a success.
        comp = result.get("completeness") or {}
        cov = comp.get("time_coverage_ratio", 1.0)
        if comp.get("partial_failure") or cov < 0.6 \
                or comp.get("continuity_breaks", 0) > 0:
            score = min(score, 40)  # PARTIAL_STITCH_FAILURE: forces retries
        elif cov < 0.8:
            score -= 15

        return int(max(0, min(100, round(score))))

    def _tweak_config(
        self,
        base: StitchConfig,
        tweak: str,
        attempts: List[Tuple[int, Dict, str]],
    ) -> StitchConfig:
        """Derive the config for one retry strategy."""
        if tweak == "densify":
            # More keyframes = smaller gaps between neighbors; this is the
            # correct response when matching fails because the spacing was
            # too wide.  Never reduces below the AUTO recommendation.
            target = base.target_keyframes
            if base.auto_keyframes and attempts:
                analysis = attempts[0][1].get("video_analysis") or {}
                target = analysis.get("recommended_keyframes", target)
            return replace(
                base,
                auto_keyframes=False,
                target_keyframes=min(base.max_auto_keyframes, target + 6),
            )
        if tweak == "strict":
            # Geometry clampdown: effectively similarity/affine only.
            return replace(
                base,
                max_rotation_deg=base.max_rotation_deg * 0.7,
                max_scale=1.2,
                min_scale=0.85,
                max_perspective=base.max_perspective * 0.5,
            )
        return replace(base)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _write_report(self, result: Dict, path: Path) -> None:
        """Write the per-video report.json (large arrays stripped)."""
        slim = {
            k: v for k, v in result.items()
            if k not in ("image", "source_map")
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_make_json_safe(slim), f, indent=2, ensure_ascii=False)
