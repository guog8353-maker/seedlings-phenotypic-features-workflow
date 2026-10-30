"""Unicode-safe image IO.

cv2.imread / cv2.imwrite silently fail on non-ASCII (e.g. Chinese) paths on
Windows.  These helpers go through np.fromfile / imencode instead.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from any (including non-ASCII) path. None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, image: np.ndarray, params: Optional[list] = None) -> bool:
    """Write an image to any (including non-ASCII) path. Returns success."""
    try:
        ext = Path(str(path)).suffix or ".png"
        ok, buf = cv2.imencode(ext, image, params or [])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False
