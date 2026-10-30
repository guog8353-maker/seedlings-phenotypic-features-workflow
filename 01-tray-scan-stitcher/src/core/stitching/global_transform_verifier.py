"""Global transform verification and expansion for multi-frame mosaic."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class GlobalTransformVerifier:
    """Verify that global transforms truly expand the field of view."""

    def __init__(self):
        pass

    def verify_expansion(
        self,
        frames: List[np.ndarray],
        global_poses: List[np.ndarray],
        canvas_bounds: Dict,
    ) -> Dict:
        """Verify that the global transforms truly expand the field of view.

        Returns a dict with:
            - is_expanded: bool
            - expansion_ratio: float
            - frame_corners: List of projected corners for each frame
            - frame_areas: List of areas for each frame
            - warnings: List of warning messages
        """
        if not frames or not global_poses:
            return {
                "is_expanded": False,
                "expansion_ratio": 0.0,
                "frame_corners": [],
                "frame_areas": [],
                "warnings": ["没有帧或全局变换"],
            }

        # Compute single frame area.
        h, w = frames[0].shape[:2]
        single_frame_area = w * h

        # Compute projected corners for each frame.
        frame_corners = []
        frame_areas = []

        for frame, pose in zip(frames, global_poses):
            h, w = frame.shape[:2]
            corners = np.array([
                [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]
            ], dtype=np.float64).T

            warped = (pose @ corners).T
            frame_corners.append(warped)

            # Compute area of the warped quadrilateral.
            area = 0.0
            for i in range(4):
                p1 = warped[i]
                p2 = warped[(i + 1) % 4]
                area += (p1[0] * p2[1] - p2[0] * p1[1])
            area = abs(area) / 2.0
            frame_areas.append(area)

        # Compute canvas area.
        canvas_area = canvas_bounds["width"] * canvas_bounds["height"]

        # Compute expansion ratio.
        if single_frame_area > 0:
            expansion_ratio = canvas_area / single_frame_area
        else:
            expansion_ratio = 0.0

        # Check if the field of view is truly expanded.
        is_expanded = expansion_ratio > 1.2

        # Generate warnings.
        warnings = []
        if not is_expanded:
            warnings.append(
                f"拼接未真正展开: 画布面积 {canvas_area} 与单帧面积 {single_frame_area} 的比例 {expansion_ratio:.2f} 小于 1.2"
            )

        # Check if all frames have the same global transform.
        if len(global_poses) > 1:
            first_pose = global_poses[0]
            all_same = all(
                np.allclose(first_pose, pose, atol=1e-6) for pose in global_poses[1:]
            )
            if all_same:
                warnings.append("所有帧的全局变换完全相同，没有空间展开")

        return {
            "is_expanded": is_expanded,
            "expansion_ratio": expansion_ratio,
            "frame_corners": frame_corners,
            "frame_areas": frame_areas,
            "warnings": warnings,
        }

    def compute_canvas_bounds(
        self,
        frames: List[np.ndarray],
        poses: List[np.ndarray],
    ) -> Dict:
        """Compute the bounds of the final canvas from all frame corners."""
        min_x, min_y = 0.0, 0.0
        max_x, max_y = 0.0, 0.0

        for frame, pose in zip(frames, poses):
            h, w = frame.shape[:2]
            corners = np.array([
                [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]
            ], dtype=np.float64).T

            warped_corners = (pose @ corners).T
            min_x = min(min_x, float(warped_corners[:, 0].min()))
            max_x = max(max_x, float(warped_corners[:, 0].max()))
            min_y = min(min_y, float(warped_corners[:, 1].min()))
            max_y = max(max_y, float(warped_corners[:, 1].max()))

        return {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "width": int(np.ceil(max_x - min_x)),
            "height": int(np.ceil(max_y - min_y)),
        }
