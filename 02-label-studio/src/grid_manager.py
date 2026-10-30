"""Grid splitting logic for a rectangular image."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GridCell:
    """A single cell in the grid.

    All coordinates are in original image pixel space and use the
    half-open interval [x1, x2) × [y1, y2).
    """

    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int


class GridManager:
    """Build and query an M×N grid over an image."""

    def __init__(self):
        self.width: int = 0
        self.height: int = 0
        self.cols: int = 0
        self.rows: int = 0
        self.cells: List[GridCell] = []

    def update(self, width: int, height: int, cols: int, rows: int) -> None:
        """Recreate the grid for the given image size and dimensions."""
        self.width = max(1, width)
        self.height = max(1, height)
        self.cols = max(1, cols)
        self.rows = max(1, rows)
        self.cells = []

        cell_w = self.width / self.cols
        cell_h = self.height / self.rows

        for r in range(self.rows):
            y1 = int(round(r * cell_h))
            y2 = int(round((r + 1) * cell_h)) if r < self.rows - 1 else self.height
            for c in range(self.cols):
                x1 = int(round(c * cell_w))
                x2 = int(round((c + 1) * cell_w)) if c < self.cols - 1 else self.width
                self.cells.append(GridCell(row=r, col=c, x1=x1, y1=y1, x2=x2, y2=y2))

    def update_auto(self, width: int, height: int, target_cells: int = 100) -> Tuple[int, int]:
        """Recreate the grid with automatically adjusted dimensions to keep cells square.

        Computes cols and rows so that each cell is approximately square,
        based on the image aspect ratio.  Returns the computed (cols, rows).
        """
        aspect = width / max(height, 1)
        # Solve cols * rows = target_cells and cols / rows = aspect.
        # cols = sqrt(target_cells * aspect), rows = sqrt(target_cells / aspect).
        cols = max(1, int(round((target_cells * aspect) ** 0.5)))
        rows = max(1, int(round((target_cells / aspect) ** 0.5)))
        self.update(width, height, cols, rows)
        return cols, rows

    def get_cell_at(self, x: int, y: int) -> Optional[GridCell]:
        """Return the cell containing the given image-space coordinate."""
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        for cell in self.cells:
            if cell.x1 <= x < cell.x2 and cell.y1 <= y < cell.y2:
                return cell
        return None
