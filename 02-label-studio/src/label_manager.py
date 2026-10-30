"""Manage annotation labels and colors."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Label:
    """A single annotation label.

    Attributes:
        name: Short identifier used in CSV outputs (e.g. "1" or "0").
        display_name: Human-readable description shown in the UI.
        color: RGBA tuple used for the semi-transparent overlay.
    """

    name: str
    display_name: str
    color: tuple = (255, 0, 0, 100)


class LabelManager:
    """Keeps the list of labels and the currently selected one."""

    def __init__(self):
        self._labels: List[Label] = []
        self._current: Optional[Label] = None

    def add_label(self, label: Label) -> None:
        """Register a new label."""
        self._labels.append(label)
        if self._current is None:
            self._current = label

    def get_label(self, name: str) -> Optional[Label]:
        """Find a label by its short name."""
        for label in self._labels:
            if label.name == name:
                return label
        return None

    def set_current(self, name: str) -> bool:
        """Set the current active label by name."""
        label = self.get_label(name)
        if label is None:
            return False
        self._current = label
        return True

    @property
    def labels(self) -> List[Label]:
        """All registered labels."""
        return list(self._labels)

    @property
    def current_label(self) -> Label:
        """The currently selected label."""
        if self._current is None:
            raise RuntimeError("No label is currently selected")
        return self._current
