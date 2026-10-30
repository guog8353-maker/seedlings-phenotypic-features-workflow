"""Workspace state persistence for the annotation workflow.

Workspace layout:

    <workspace>/
        img_a.jpg, img_b.png, ...           (source images)
        rectified/<name>_rect.png           (auto-saved rectified images)
        annotations/<name>_mask.png         (brush masks)
        annotations/<name>_state.json       (rectify info + grid + params)

App-level persistence (config.json in the app directory) remembers the
workspace, output dir, grid presets, brush radius, and view toggles.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .image_io import imread_unicode, imwrite_unicode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_CONFIG = {
    "workspace": "",
    "output_dir": "",
    "cols": 36,
    "rows": 18,
    "presets": [[36, 18], [41, 20], [48, 28]],
    "brush_radius": 25,
    "show_grid": True,
    "mask_threshold": 0.01,
}


# ----------------------------------------------------------------------
# App config
# ----------------------------------------------------------------------
def load_config(app_dir: Path) -> Dict:
    path = Path(app_dir) / "config.json"
    cfg = dict(DEFAULT_CONFIG)
    try:
        if path.exists():
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def save_config(app_dir: Path, cfg: Dict) -> None:
    try:
        path = Path(app_dir) / "config.json"
        path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


# ----------------------------------------------------------------------
# Workspace images
# ----------------------------------------------------------------------
def list_workspace_images(workspace: str) -> List[Path]:
    root = Path(workspace)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ----------------------------------------------------------------------
# Annotation state (per image)
# ----------------------------------------------------------------------
def _ann_dir(workspace: str) -> Path:
    return Path(workspace) / "annotations"


def _rect_dir(workspace: str) -> Path:
    return Path(workspace) / "rectified"


def rectified_path(workspace: str, name: str) -> Path:
    return _rect_dir(workspace) / f"{name}_rect.png"


def mask_path(workspace: str, name: str) -> Path:
    return _ann_dir(workspace) / f"{name}_mask.png"


def state_path(workspace: str, name: str) -> Path:
    return _ann_dir(workspace) / f"{name}_state.json"


def has_state(workspace: str, name: str) -> bool:
    return state_path(workspace, name).exists()


def save_rectified(workspace: str, name: str, image: np.ndarray) -> str:
    """Auto-save the rectified image; returns the saved path."""
    _rect_dir(workspace).mkdir(parents=True, exist_ok=True)
    path = rectified_path(workspace, name)
    imwrite_unicode(str(path), image)
    return str(path)


def save_state(
    workspace: str,
    name: str,
    mask: Optional[np.ndarray],
    grid: Dict,
    rectify_info: Optional[Dict],
    brush: Dict,
    source_image: str,
) -> str:
    """Persist mask + grid + rectify info for one image."""
    _ann_dir(workspace).mkdir(parents=True, exist_ok=True)
    if mask is not None:
        imwrite_unicode(str(mask_path(workspace, name)), mask)

    H = rectify_info.get("H") if rectify_info else None
    state = {
        "source_image": source_image,
        "grid": grid,
        "brush": brush,
        "rectified": rectify_info is not None,
        "rectify": {
            "quad_points_original": rectify_info.get("pts") if rectify_info else None,
            "H_original_to_rectified": H.tolist() if H is not None else None,
            "out_size": list(rectify_info.get("out_size", [])) if rectify_info else None,
        },
    }
    path = state_path(workspace, name)
    path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(path)


def load_state(workspace: str, name: str) -> Optional[Dict]:
    """Load mask + grid + rectify info for one image (None if absent)."""
    path = state_path(workspace, name)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    mask = None
    mp = mask_path(workspace, name)
    if mp.exists():
        mask = imread_unicode(str(mp), 0)  # grayscale
    state["mask"] = mask
    return state
