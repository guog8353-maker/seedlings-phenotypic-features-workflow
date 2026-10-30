"""Batch processing: automatically loop SmartStitcher over many videos.

Batch mode owns NO stitching logic of its own.  Every video is processed by
the exact same SmartStitcher.process_video() used by single-video mode, with
the same StitchConfig.  A failing video never stops the batch; a consolidated
batch_report.json / batch_report.csv is written at the end.
"""

import csv
import json
from pathlib import Path
from typing import Callable, List, Optional

from .smart_stitcher import SmartStitcher
from .stitching.config import StitchConfig


class BatchProcessor:
    """Process multiple videos, one SmartStitcher call each."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: Optional[StitchConfig] = None,
        output_format: str = "png",
        jpeg_quality: int = 95,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config if config is not None else StitchConfig()
        self.output_format = output_format.lower().lstrip(".")
        self.jpeg_quality = jpeg_quality
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def get_video_files(self) -> List[Path]:
        if not self.input_dir.exists():
            return []
        return sorted(self.input_dir.glob("*.mp4"))

    def process_single(
        self,
        video_path: Path,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Process one video via the shared SmartStitcher (same as single mode)."""
        if log_callback:
            log_callback(f"开始处理: {video_path.name}")

        try:
            stitcher = SmartStitcher(
                config=self.config,
                output_format=self.output_format,
                jpeg_quality=self.jpeg_quality,
            )
            result = stitcher.process_video(
                str(video_path), output_dir=str(self.output_dir)
            )
            result["file"] = video_path.name
            result["success"] = bool(result.get("success"))
            if not result["success"]:
                result["error"] = result.get("message", "拼接失败")
            if log_callback:
                if result["success"]:
                    log_callback(
                        f"  完成: 推荐 {result.get('recommended_keyframes', '?')} / "
                        f"选中 {result.get('selected_keyframes', '?')} / "
                        f"参与 {result.get('active_frames', '?')} / "
                        f"贡献 {result.get('contributing_frames', '?')} | "
                        f"质量 {result.get('quality_score', '?')} 分 | "
                        f"尝试 {len(result.get('attempts', []))} 次"
                    )
                    if result.get("output"):
                        log_callback(f"  已保存: {result['output']}")
                else:
                    log_callback(f"  失败: {result.get('error', '')}")
            return result

        except Exception as e:
            # One failing video must never stop the batch.
            return {
                "file": video_path.name,
                "filename": video_path.name,
                "success": False,
                "error": f"处理异常: {e}",
                "status": "失败",
                "failure_reason": str(e),
            }

    def run(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        file_done_callback: Optional[Callable[[dict], None]] = None,
    ) -> List[dict]:
        """Run batch processing over all videos in the input directory."""
        self._cancelled = False
        videos = self.get_video_files()
        results = []

        if not videos:
            if log_callback:
                log_callback("未找到 .mp4 视频文件")
            return results

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if log_callback:
            log_callback(f"发现 {len(videos)} 个视频文件")

        for i, video_path in enumerate(videos, 1):
            if self._cancelled:
                if log_callback:
                    log_callback("批量处理已取消")
                break

            if progress_callback:
                progress_callback(i, len(videos))
            result = self.process_single(video_path, log_callback)
            results.append(result)

            if file_done_callback:
                file_done_callback(result)

        if progress_callback:
            progress_callback(len(videos), len(videos))
        self._write_batch_report(results)
        if log_callback:
            log_callback("批量处理完成")
        return results

    # ------------------------------------------------------------------
    # Batch report
    # ------------------------------------------------------------------
    _REPORT_FIELDS = [
        "filename",
        "duration",
        "recommended_keyframes",
        "selected_keyframes",
        "active_frames",
        "contributing_frames",
        "motion_type",
        "stitchability",
        "attempts",
        "quality_score",
        "output",
        "status",
        "failure_reason",
    ]

    def _write_batch_report(self, results: List[dict]) -> None:
        """Write batch_report.json and batch_report.csv."""
        rows = []
        for r in results:
            analysis = r.get("video_analysis") or {}
            rows.append({
                "filename": r.get("filename", r.get("file", "")),
                "duration": analysis.get("duration", ""),
                "recommended_keyframes": r.get("recommended_keyframes", ""),
                "selected_keyframes": r.get("selected_keyframes", ""),
                "active_frames": r.get("active_frames", ""),
                "contributing_frames": r.get("contributing_frames", ""),
                "motion_type": r.get("motion_type", ""),
                "stitchability": r.get("stitchability", ""),
                "attempts": len(r.get("attempts", [])) if r.get("attempts") else "",
                "quality_score": r.get("quality_score", ""),
                "output": r.get("output", ""),
                "status": r.get("status", "完成" if r.get("success") else "失败"),
                "failure_reason": r.get("failure_reason", r.get("error", "")),
            })

        json_path = self.output_dir / "batch_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        csv_path = self.output_dir / "batch_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self._REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
