"""Abstract base class for all counters."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseCounter(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.count_in = 0
        self.count_out = 0

    @abstractmethod
    def update(self, track_id: int, center: Tuple[float, float], timestamp: float) -> dict:
        """
        Process a single detection for one frame.

        Args:
            track_id:  Unique integer ID assigned by the tracker.
            center:    (cx, cy) — center of the bounding box.
            timestamp: Seconds elapsed in the video stream.

        Returns:
            dict with boolean flags, e.g. {"entered": True, "exited": False}
        """

    @abstractmethod
    def remove_track(self, track_id: int) -> None:
        """Called when a tracked object permanently disappears."""

    def get_counts(self) -> dict:
        return {"in": self.count_in, "out": self.count_out}
