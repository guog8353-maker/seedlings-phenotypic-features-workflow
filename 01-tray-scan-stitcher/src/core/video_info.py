"""Video metadata extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Container for video metadata."""

    path: str
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    size_mb: float
    codec: str
    fourcc: str
    aspect_ratio: str
    color_space: str
    bit_depth: str
    bitrate_kbps: Optional[float]
    thumbnail: Optional[np.ndarray] = None

    @property
    def resolution(self) -> str:
        return f"{self.width} x {self.height}"

    @property
    def duration_str(self) -> str:
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = self.duration % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:05.2f}"
        return f"{minutes:02d}:{seconds:05.2f}"

    @property
    def bitrate_str(self) -> str:
        if self.bitrate_kbps is None:
            return "未知"
        if self.bitrate_kbps >= 10000:
            return f"{self.bitrate_kbps / 1000:.2f} Mbps"
        return f"{self.bitrate_kbps:.2f} kbps"

    @classmethod
    def from_path(cls, video_path: str) -> "VideoInfo":
        """Extract metadata from a video file."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0
        duration = frame_count / fps if frame_count > 0 else 0.0

        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])

        codec_names = {
            "H264": "H.264 / AVC",
            "X264": "H.264 / AVC",
            "avc1": "H.264 / AVC",
            "HEVC": "H.265 / HEVC",
            "hvc1": "H.265 / HEVC",
            "MJPG": "Motion JPEG",
            "MP4V": "MPEG-4 Part 2",
            "DIVX": "DivX",
            "XVID": "Xvid",
            "VP80": "VP8",
            "VP90": "VP9",
            "AV01": "AV1",
            "mp4v": "MPEG-4 Part 2",
        }
        codec = codec_names.get(fourcc_str, fourcc_str)

        gcd = np.gcd(width, height)
        aspect_ratio = f"{width // gcd}:{height // gcd}"

        color_space = "BGR (OpenCV 默认)"
        bit_depth = "8-bit"

        bitrate_kbps = None
        if duration > 0:
            bitrate_kbps = (size_bytes * 8) / (duration * 1000)

        ret, frame = cap.read()
        thumbnail = frame if ret else None

        cap.release()

        return cls(
            path=str(path.resolve()),
            filename=path.name,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            size_mb=size_mb,
            codec=codec,
            fourcc=fourcc_str,
            aspect_ratio=aspect_ratio,
            color_space=color_space,
            bit_depth=bit_depth,
            bitrate_kbps=bitrate_kbps,
            thumbnail=thumbnail,
        )

    def get_summary(self) -> dict:
        return {
            "文件名": self.filename,
            "时长": self.duration_str,
            "分辨率": self.resolution,
            "帧率": f"{self.fps:.2f} fps",
            "总帧数": str(self.frame_count),
            "文件大小": f"{self.size_mb:.2f} MB",
        }

    def get_details(self) -> dict:
        return {
            "完整路径": self.path,
            "编码格式": f"{self.codec} ({self.fourcc})",
            "宽高比": self.aspect_ratio,
            "色彩空间": self.color_space,
            "位深": self.bit_depth,
            "估算码率": self.bitrate_str,
        }
