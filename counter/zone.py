"""
Counting entering zone — Slide 5 / Slide 6 rules
==================================================

Rules (from spec):
  1. Use the **center** of the bounding box (slide 9).
  2. Count 1 time per person **per entry** if they remain inside the zone for
     at least `min_dwell_seconds` (slide 5).
  3. A person who enters N times is counted N times — each visit is independent
     (slide 6: Count:2 for a person who enters, leaves, and enters again).
  4. The dwell timer starts fresh on each new entry.

Counting strategy:
  - A person is counted as soon as their dwell time reaches min_dwell_seconds
    (early trigger), so counts appear promptly.
  - When the person exits the zone, their state is cleared so the next entry
    starts a new dwell measurement.
"""

from typing import Dict, List, Optional, Tuple

from .base import BaseCounter
from utils.geometry import point_in_polygon


class _ZoneState:
    """Per-track zone state."""

    __slots__ = ("entry_time", "counted")

    def __init__(self, entry_time: float):
        self.entry_time: float = entry_time
        self.counted: bool = False  # True once the dwell threshold is reached


class ZoneCounter(BaseCounter):
    """
    Counts people who enter and dwell in a polygonal zone.

    Only `count_in` is meaningful; `count_out` stays at 0 (zone entry only).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.zone_id: str = config["id"]
        self.name: str = config["name"]
        self.polygon: List[Tuple[float, float]] = [tuple(p) for p in config["points"]]
        self.min_dwell: float = float(config.get("min_dwell_seconds", 2.0))

        self._states: Dict[int, _ZoneState] = {}  # track_id → state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, track_id: int, center: Tuple[float, float], timestamp: float
    ) -> dict:
        inside = point_in_polygon(center, self.polygon)
        state = self._states.get(track_id)
        result = {"counted": False}

        if inside:
            if state is None:
                # Person just entered the zone — start dwell timer
                self._states[track_id] = _ZoneState(entry_time=timestamp)
            elif not state.counted:
                dwell = timestamp - state.entry_time
                if dwell >= self.min_dwell:
                    self.count_in += 1
                    state.counted = True
                    result["counted"] = True
        else:
            if state is not None:
                # Person left the zone — clear state so the next entry is fresh
                del self._states[track_id]

        return result

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)

    def get_counts(self) -> dict:
        # Zone counting only has an "in" count (number of valid visits)
        return {"in": self.count_in, "out": 0}
