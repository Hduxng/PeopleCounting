"""
Counting crossline — Slide 3 / Slide 4 rules
=============================================

Rules (from spec):
  1. Use the **center** of the bounding box as the tracking point (slide 9).
  2. Count 1 time per person per crossing direction.
  3. Hysteresis buffer: the center must travel at least `buffer_px` pixels away
     from the line before a crossing is confirmed — this prevents oscillating
     detections near the boundary from being counted (slide 4: Enter:0, Exit:0).
  4. "Enter" direction is configurable per line.
"""

from typing import Dict, Optional, Tuple

from .base import BaseCounter
from utils.geometry import point_side_of_line


class CrosslineCounter(BaseCounter):
    """
    Counts people crossing a single virtual line segment.

    State machine per tracked ID:
      last_confirmed_side ∈ {None, 1, -1}

      - None  : first time we see this ID; just record the side, no count.
      - 1/-1  : last confirmed side (outside the buffer zone).
                When the current side differs from last_confirmed_side, a
                crossing is registered and last_confirmed_side is updated.
      - 0     : current frame is inside the buffer zone; side is ambiguous,
                so we skip this frame without changing last_confirmed_side.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.line_id: str = config["id"]
        self.name: str = config["name"]
        pt1_raw, pt2_raw = config["points"]
        self.pt1: Tuple[float, float] = tuple(pt1_raw)
        self.pt2: Tuple[float, float] = tuple(pt2_raw)
        self.buffer_px: float = float(config.get("buffer_px", 20))
        # "positive" → crossing from negative→positive side counts as enter
        # "negative" → crossing from positive→negative side counts as enter
        self.enter_direction: str = config.get("enter_direction", "positive")

        self._states: Dict[int, Optional[int]] = {}  # track_id → last confirmed side
        self._counted_in_ids: set[int] = set()
        self._counted_out_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, track_id: int, center: Tuple[float, float], timestamp: float
    ) -> dict:
        side = point_side_of_line(center, self.pt1, self.pt2, self.buffer_px)
        result = {"entered": False, "exited": False}

        if side == 0:
            # Inside the hysteresis buffer — wait for a clear side commitment
            return result

        last = self._states.get(track_id)

        if last is None:
            # First observation: initialise side without counting
            self._states[track_id] = side
            return result

        if side == last:
            return result  # Still on the same side

        # ----- Crossing confirmed -----
        self._states[track_id] = side

        if self.enter_direction == "positive":
            entered = side == 1
        else:
            entered = side == -1

        if entered:
            if track_id not in self._counted_in_ids:
                self.count_in += 1
                self._counted_in_ids.add(track_id)
                result["entered"] = True
        else:
            if track_id not in self._counted_out_ids:
                self.count_out += 1
                self._counted_out_ids.add(track_id)
                result["exited"] = True

        return result

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)
