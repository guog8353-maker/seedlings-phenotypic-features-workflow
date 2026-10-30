"""Matplotlib-based plotting components for feature visualization."""

from typing import List

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QSizePolicy, QWidget


class PlotCanvas(QWidget):
    """A Qt widget that displays matplotlib figures lazily.

    Matplotlib and its Qt backend are only imported when the widget is first
    used, so importing this module does not add startup cost.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._canvas = None
        self._figure = None
        self._axes = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._placeholder = QLabel("可视化区域")
        self._placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._placeholder)

    def set_placeholder_text(self, text: str) -> None:
        """Set the text shown before the matplotlib canvas is created."""
        if self._placeholder is not None:
            self._placeholder.setText(text)

    def _ensure_canvas(self) -> None:
        """Create the matplotlib canvas on first use."""
        if self._canvas is not None:
            return

        import matplotlib
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        self._figure = Figure(figsize=(5, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111)

        layout = self.layout()
        layout.removeWidget(self._placeholder)
        self._placeholder.deleteLater()
        self._placeholder = None
        layout.addWidget(self._canvas)

    def _clear(self) -> None:
        if self._axes is not None:
            self._axes.clear()

    def _draw(self) -> None:
        if self._canvas is not None:
            self._canvas.draw()
            self._canvas.update()
        self.update()

    def _reset_polar(self) -> None:
        """Switch back to a regular axes if the previous plot was polar."""
        if self._axes is not None and self._axes.name == "polar":
            self._figure.clf()
            self._axes = self._figure.add_subplot(111)

    def plot_histogram(
        self,
        values: np.ndarray,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = 32,
        color: str = "steelblue",
    ) -> None:
        self._ensure_canvas()
        self._reset_polar()
        self._clear()
        self._axes.hist(values.flatten(), bins=bins, color=color, edgecolor="black", alpha=0.8)
        self._axes.set_title(title)
        self._axes.set_xlabel(xlabel)
        self._axes.set_ylabel(ylabel)
        self._figure.tight_layout()
        self._draw()

    def plot_bar(
        self,
        labels: List[str],
        values: List[float],
        title: str = "Bar Chart",
        color: str = "coral",
    ) -> None:
        self._ensure_canvas()
        self._reset_polar()
        self._clear()
        self._axes.bar(labels, values, color=color, edgecolor="black")
        self._axes.set_title(title)
        self._axes.tick_params(axis="x", rotation=45)
        self._figure.tight_layout()
        self._draw()

    def plot_radar(
        self,
        labels: List[str],
        values: List[float],
        title: str = "Radar Chart",
        color: str = "teal",
    ) -> None:
        self._ensure_canvas()
        self._clear()
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values_plot = values + values[:1]
        angles += angles[:1]

        self._axes = self._figure.add_subplot(111, polar=True)
        self._axes.plot(angles, values_plot, color=color, linewidth=2)
        self._axes.fill(angles, values_plot, color=color, alpha=0.25)
        self._axes.set_xticks(angles[:-1])
        self._axes.set_xticklabels(labels, fontsize=8)
        self._axes.set_title(title, y=1.08)
        self._figure.tight_layout()
        self._draw()

    def plot_heatmap(
        self,
        data: np.ndarray,
        title: str = "Heatmap",
        xlabel: str = "",
        ylabel: str = "",
        cmap: str = "viridis",
    ) -> None:
        self._ensure_canvas()
        self._reset_polar()
        self._clear()
        im = self._axes.imshow(data, cmap=cmap, aspect="auto")
        self._figure.colorbar(im, ax=self._axes)
        self._axes.set_title(title)
        self._axes.set_xlabel(xlabel)
        self._axes.set_ylabel(ylabel)
        self._figure.tight_layout()
        self._draw()

    def plot_boxplot(
        self,
        data: List[np.ndarray],
        labels: List[str],
        title: str = "Boxplot",
    ) -> None:
        self._ensure_canvas()
        self._reset_polar()
        self._clear()
        self._axes.boxplot(data, labels=labels)
        self._axes.set_title(title)
        self._axes.tick_params(axis="x", rotation=45)
        self._figure.tight_layout()
        self._draw()

    def plot_correlation_heatmap(
        self,
        corr_matrix: np.ndarray,
        labels: List[str],
        title: str = "Correlation Heatmap",
    ) -> None:
        self._ensure_canvas()
        self._reset_polar()
        self._clear()
        im = self._axes.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        self._figure.colorbar(im, ax=self._axes)
        self._axes.set_xticks(np.arange(len(labels)))
        self._axes.set_yticks(np.arange(len(labels)))
        self._axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        self._axes.set_yticklabels(labels, fontsize=7)
        self._axes.set_title(title)
        self._figure.tight_layout()
        self._draw()

    def show_message(self, message: str) -> None:
        """Display a text message in place of a plot."""
        self._ensure_canvas()
        self._clear()
        self._axes.text(
            0.5,
            0.5,
            message,
            horizontalalignment="center",
            verticalalignment="center",
            transform=self._axes.transAxes,
            fontsize=14,
            color="gray",
        )
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        for spine in self._axes.spines.values():
            spine.set_visible(False)
        self._figure.tight_layout()
        self._draw()

    def save_figure(self, path: str) -> None:
        self._ensure_canvas()
        self._figure.savefig(path, dpi=150, bbox_inches="tight")
