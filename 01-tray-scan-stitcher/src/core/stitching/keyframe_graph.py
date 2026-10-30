"""Keyframe graph builder to find the largest reliable connected component."""

from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .feature_matcher import FeatureMatcher


class KeyframeGraph:
    """Build a graph of keyframes and find the largest reliable connected component."""

    def __init__(self, matcher: FeatureMatcher):
        self.matcher = matcher

    def build_graph(
        self,
        frames: List[np.ndarray],
        max_gap: int = 3,
    ) -> Dict:
        """Build a graph of keyframes with reliable match edges.

        Args:
            frames: List of keyframes.
            max_gap: Maximum gap between frames to try matching (e.g., 3 means
                try i with i+1, i+2, i+3).

        Returns:
            Dict with:
                - nodes: List of frame indices
                - edges: Dict of (i, j) -> match_info
                - components: List of connected components (each a list of indices)
                - largest_component: List of indices in the largest component
        """
        n = len(frames)
        nodes = list(range(n))
        edges = {}

        # Try matching each frame with frames up to max_gap ahead.
        for i in range(n):
            for j in range(i + 1, min(i + max_gap + 1, n)):
                H, info = self.matcher.estimate_homography(frames[i], frames[j])

                if H is not None and info.get("status") == "success":
                    edges[(i, j)] = info

        # Find connected components using union-find.
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in edges.keys():
            union(i, j)

        # Group nodes by component.
        components = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)

        # Find the largest component.
        largest_component = max(components.values(), key=len) if components else []

        return {
            "nodes": nodes,
            "edges": edges,
            "components": list(components.values()),
            "largest_component": largest_component,
        }

    def get_active_frames(
        self,
        frames: List[np.ndarray],
        graph: Dict,
    ) -> List[int]:
        """Get the indices of frames that should be used for stitching.

        Returns the indices of frames in the largest reliable connected component.
        """
        return graph["largest_component"]


def select_span_path(
    graph: Dict,
    keyframes: List[Tuple[int, np.ndarray]],
    min_edge_ratio: float = 0.2,
) -> List[int]:
    """Select the path with the LONGEST TEMPORAL SPAN through the match graph.

    Unlike ``largest_component`` (which keeps the biggest cluster regardless
    of where it sits on the timeline and silently drops the head/tail of the
    video), this walks the time-ordered DAG and returns the path whose first
    and last frames cover the widest time range, preferring higher mean edge
    quality on ties.

    Returns a sorted list of keyframe indices; empty list if no edges exist.
    """
    n = len(keyframes)
    edges = graph.get("edges", {})
    ids = [int(fid) for fid, _ in keyframes]

    # Forward predecessors with edge quality.
    preds: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), info in edges.items():
        if 0 <= i < j < n:
            preds[j].append((i, float(info.get("inlier_ratio", 0.0))))

    # DP over the DAG: best (max-span) path ending at each node.
    parent = [-1] * n
    start = list(range(n))          # path start index for best path ending at j
    mean_ratio = [0.0] * n          # mean edge ratio of that path
    n_edges = [0] * n

    for j in range(n):
        best_span = 0
        for i, ratio in preds[j]:
            if ratio < min_edge_ratio:
                continue
            cand_span = ids[j] - ids[start[i]]
            cand_edges = n_edges[i] + 1
            cand_mean = (mean_ratio[i] * n_edges[i] + ratio) / cand_edges
            cur_span = ids[j] - ids[start[j]] if n_edges[j] > 0 else 0
            if cand_span > best_span and (
                cand_span > cur_span
                or (cand_span == cur_span and cand_mean > mean_ratio[j])
                or n_edges[j] == 0
            ):
                best_span = cand_span
                parent[j] = i
                start[j] = start[i]
                mean_ratio[j] = cand_mean
                n_edges[j] = cand_edges

    # End node with the longest covered span (tie: higher mean quality).
    best_j = -1
    best_key = (-1, -1.0)
    for j in range(n):
        if n_edges[j] == 0:
            continue
        key = (ids[j] - ids[start[j]], mean_ratio[j])
        if key > best_key:
            best_key = key
            best_j = j

    if best_j < 0:
        return []

    # Reconstruct the path.
    path = []
    j = best_j
    while j >= 0:
        path.append(j)
        j = parent[j]
    path.reverse()
    return path
