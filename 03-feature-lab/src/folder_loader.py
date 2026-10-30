"""Folder loading for crop-level feature extraction.

Scans a dataset folder (e.g. the export of software 2):

    <root>/grid_18x36/0/A10010P1121_r000_c000.jpg
    <root>/grid_18x36/1/A10010P1121_r000_c003.jpg

Each image becomes one table row.  The immediate parent folder name "0"/"1"
fills the guttation label column; the file name encodes original_image,
row and col.  Plain flat folders also work (label left empty).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_NAME_RE = re.compile(r"^(?P<stem>.+?)_r(?P<row>\d+)_c(?P<col>\d+)$")
_GRID_RE = re.compile(r"^grid_(?P<cols>\d+)x(?P<rows>\d+)$")


@dataclass
class CropItem:
    """One crop image and everything the table needs to know about it."""

    path: Path
    label: str                    # "0" / "1" / "" (unknown)
    original_image: str
    row: Optional[int]
    col: Optional[int]
    grid_strategy: str            # e.g. "18x36" or ""


def _parse_file(path: Path, root: Path) -> CropItem:
    # Label from the immediate parent folder name.
    parent = path.parent.name
    label = parent if parent in ("0", "1") else ""

    # Grid strategy from the grandparent folder name (grid_<cols>x<rows>).
    grid_strategy = ""
    for anc in (path.parent.parent.name, parent):
        m = _GRID_RE.match(anc)
        if m:
            grid_strategy = f"{m.group('cols')}x{m.group('rows')}"
            break

    # original_image / row / col from the file stem.
    stem = path.stem
    m = _NAME_RE.match(stem)
    if m:
        original = m.group("stem")
        row = int(m.group("row"))
        col = int(m.group("col"))
    else:
        original, row, col = stem, None, None

    return CropItem(path, label, original, row, col, grid_strategy)


def scan_folder(root: str) -> List[CropItem]:
    """Recursively scan a folder for crop images."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    items = []
    for p in sorted(root_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            items.append(_parse_file(p, root_path))
    return items
