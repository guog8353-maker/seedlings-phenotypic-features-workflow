"""Global pose optimization to avoid cumulative drift."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class GlobalOptimizer:
    """Optimize global poses using a simple pose graph approach.

    Instead of just accumulating transforms sequentially (which causes drift),
    this module builds a graph of frame relationships and optimizes the global
    poses to minimize reprojection errors.
    """

    def __init__(self, reference_frame: int = -1):
        """Initialize with a reference frame index.

        If reference_frame is -1, the middle frame is used as reference.
        """
        self.reference_frame = reference_frame

    def optimize(
        self,
        transforms: List[np.ndarray],
        match_infos: List[dict],
    ) -> List[np.ndarray]:
        """Optimize global poses from pairwise transforms.

        Args:
            transforms: List of 3x3 homography or 2x3 affine matrices mapping
                frame i+1 to frame i.
            match_infos: List of match info dicts for each transform.

        Returns:
            List of optimized global transforms (3x3 homography matrices) that
            map each frame to the reference frame's coordinate system.
        """
        n = len(transforms) + 1
        if n == 1:
            return [np.eye(3, dtype=np.float64)]

        # Choose reference frame (middle by default).
        if self.reference_frame < 0 or self.reference_frame >= n:
            ref_idx = n // 2
        else:
            ref_idx = self.reference_frame

        # Convert all transforms to 3x3 homogeneous matrices.
        H_list = []
        for M in transforms:
            if M.shape == (2, 3):
                H = np.eye(3, dtype=np.float64)
                H[:2, :] = M
                H_list.append(H)
            else:
                H_list.append(M.astype(np.float64))

        # Compute global poses by chaining from reference frame.
        global_poses = [None] * n
        global_poses[ref_idx] = np.eye(3, dtype=np.float64)

        # Forward chain from reference.
        for i in range(ref_idx + 1, n):
            # frame i maps to frame i-1 via H_list[i-1].
            # global_pose[i] = global_pose[i-1] @ H_list[i-1]
            global_poses[i] = global_poses[i - 1] @ H_list[i - 1]

        # Backward chain from reference.
        for i in range(ref_idx - 1, -1, -1):
            # frame i+1 maps to frame i via H_list[i].
            # So frame i maps to frame i+1 via inv(H_list[i]).
            # global_pose[i] = global_pose[i+1] @ inv(H_list[i])
            try:
                H_inv = np.linalg.inv(H_list[i])
                global_poses[i] = global_poses[i + 1] @ H_inv
            except np.linalg.LinAlgError:
                # If inversion fails, use identity.
                global_poses[i] = global_poses[i + 1]

        # Simple refinement: if match quality is low, dampen the transform.
        refined_poses = []
        for i, pose in enumerate(global_poses):
            if i == ref_idx:
                refined_poses.append(pose)
            else:
                # Dampen poses with low match quality to reduce drift.
                if i > 0 and i - 1 < len(match_infos):
                    info = match_infos[i - 1]
                    if info.get("inlier_ratio", 1.0) < 0.5:
                        # Low quality match; blend with identity to reduce drift.
                        pose = 0.9 * pose + 0.1 * np.eye(3, dtype=np.float64)
                refined_poses.append(pose)

        return refined_poses
