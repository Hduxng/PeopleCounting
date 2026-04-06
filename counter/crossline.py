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
  5. Temporal hysteresis: require `confirm_frames` consecutive frames on the
     new side before registering a crossing — prevents single-frame jitter.
"""

from typing import Dict, Optional, Tuple

from .base import BaseCounter
from utils.geometry import point_side_of_line


def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


_PendingCrossing = Tuple[int, int]
_CrosslineHandoff = Tuple[
    Optional[int],
    Optional[_PendingCrossing],
    bool,
    bool,
    Tuple[float, float],
    float,
]


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

    Temporal hysteresis:
      When a side change is first detected, enter a pending state. Only
      confirm the crossing after `confirm_frames` consecutive frames on
      the new side. If the track returns to the old side, cancel the pending.
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
        self._require_prior_in_for_out: bool = bool(config.get("require_prior_in_for_out", True))
        self._confirm_frames: int = int(config.get("confirm_frames", 1))
        self._handoff_sec: float = float(config.get("handoff_seconds", 1.0))
        self._handoff_radius: float = float(config.get("handoff_radius_px", max(self.buffer_px * 2.0, 60.0)))

        # Pre-compute line length (avoid sqrt per update call)
        import math
        dx = self.pt2[0] - self.pt1[0]
        dy = self.pt2[1] - self.pt1[1]
        self._line_len: float = math.sqrt(dx * dx + dy * dy)

        self._states: Dict[int, Optional[int]] = {}  # track_id → last confirmed side
        self._counted_in_ids: set[int] = set()
        self._counted_out_ids: set[int] = set()
        # Pending crossings: {track_id: (new_side, consecutive_count)}
        self._pending: Dict[int, _PendingCrossing] = {}
        self._handoff: Dict[int, _CrosslineHandoff] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, track_id: int, center: Tuple[float, float], timestamp: float
    ) -> dict:
        self._expire_handoffs(timestamp)
        side = point_side_of_line(center, self.pt1, self.pt2, self.buffer_px, self._line_len)
        result = {"entered": False, "exited": False}

        if side == 0:
            # Inside the hysteresis buffer — wait for a clear side commitment
            return result

        last = self._states.get(track_id)

        if last is None:
            restored = self._claim_handoff(track_id, center, timestamp)
            if restored is not None:
                restored_state, restored_pending, counted_in, counted_out = restored
                if restored_state is not None:
                    self._states[track_id] = restored_state
                if restored_pending is not None:
                    self._pending[track_id] = restored_pending
                if counted_in:
                    self._counted_in_ids.add(track_id)
                if counted_out:
                    self._counted_out_ids.add(track_id)
                last = self._states.get(track_id)

        if last is None:
            # First observation: initialise side without counting
            self._states[track_id] = side
            self._pending.pop(track_id, None)
            return result

        if side == last:
            # Still on the same side — cancel any pending crossing
            self._pending.pop(track_id, None)
            return result

        # ----- Side changed — check temporal hysteresis -----
        if self._confirm_frames <= 1:
            # No temporal hysteresis: confirm immediately
            return self._confirm_crossing(track_id, side)

        pending = self._pending.get(track_id)
        if pending is not None and pending[0] == side:
            # Continue accumulating frames on new side
            count = pending[1] + 1
            if count >= self._confirm_frames:
                self._pending.pop(track_id, None)
                return self._confirm_crossing(track_id, side)
            self._pending[track_id] = (side, count)
        else:
            # Start new pending crossing
            self._pending[track_id] = (side, 1)

        return result

    def mark_lost(
        self,
        track_id: int,
        center: Tuple[float, float] | None,
        timestamp: float,
    ) -> None:
        self._expire_handoffs(timestamp)
        if center is None:
            return

        state = self._states.pop(track_id, None)
        pending = self._pending.pop(track_id, None)
        counted_in = track_id in self._counted_in_ids
        counted_out = track_id in self._counted_out_ids
        if state is None and pending is None and not counted_in and not counted_out:
            return

        self._handoff[track_id] = (
            state,
            pending,
            counted_in,
            counted_out,
            (float(center[0]), float(center[1])),
            float(timestamp),
        )

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)
        self._pending.pop(track_id, None)
        self._handoff.pop(track_id, None)

    def transfer_state(self, from_tid: int, to_tid: int) -> None:
        """Transfer crossing state from one track ID to another (for Re-ID remap)."""
        if from_tid == to_tid:
            return

        from_state = self._states.pop(from_tid, None)
        if from_state is not None and to_tid not in self._states:
            self._states[to_tid] = from_state

        from_pending = self._pending.pop(from_tid, None)
        if from_pending is not None:
            to_pending = self._pending.get(to_tid)
            if to_pending is None:
                self._pending[to_tid] = from_pending
            elif to_pending[0] == from_pending[0]:
                self._pending[to_tid] = (to_pending[0], max(to_pending[1], from_pending[1]))

        if from_tid in self._counted_in_ids:
            self._counted_in_ids.add(to_tid)
        if from_tid in self._counted_out_ids:
            self._counted_out_ids.add(to_tid)

        from_handoff = self._handoff.pop(from_tid, None)
        if from_handoff is not None:
            state, pending, counted_in, counted_out, center, lost_ts = from_handoff
            if state is not None and to_tid not in self._states:
                self._states[to_tid] = state
            if pending is not None:
                to_pending = self._pending.get(to_tid)
                if to_pending is None or to_pending[0] == pending[0] and pending[1] > to_pending[1]:
                    self._pending[to_tid] = pending
            if counted_in:
                self._counted_in_ids.add(to_tid)
            if counted_out:
                self._counted_out_ids.add(to_tid)
            to_handoff = self._handoff.get(to_tid)
            if to_handoff is None:
                self._handoff[to_tid] = (state, pending, counted_in, counted_out, center, lost_ts)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _confirm_crossing(self, track_id: int, side: int) -> dict:
        """Register a confirmed crossing event."""
        result = {"entered": False, "exited": False}
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
            if self._require_prior_in_for_out and track_id not in self._counted_in_ids:
                return result
            if track_id not in self._counted_out_ids:
                self.count_out += 1
                self._counted_out_ids.add(track_id)
                result["exited"] = True

        return result

    def _expire_handoffs(self, timestamp: float) -> None:
        expired = [
            tid for tid, (_, _, _, _, _, lost_ts) in self._handoff.items()
            if timestamp - lost_ts >= self._handoff_sec
        ]
        for tid in expired:
            del self._handoff[tid]

    def _claim_handoff(
        self,
        track_id: int,
        center: Tuple[float, float],
        timestamp: float,
    ) -> Optional[Tuple[Optional[int], Optional[_PendingCrossing], bool, bool]]:
        best_tid = None
        best_score = None
        for candidate_tid, (state, pending, counted_in, counted_out, candidate_center, lost_ts) in self._handoff.items():
            if candidate_tid == track_id:
                continue
            if state is None and pending is None and not counted_in and not counted_out:
                continue
            dist = _center_distance(center, candidate_center)
            if dist > self._handoff_radius:
                continue
            age = max(float(timestamp) - float(lost_ts), 0.0)
            score = (dist, age)
            if best_score is None or score < best_score:
                best_tid = candidate_tid
                best_score = score

        if best_tid is None:
            return None

        state, pending, counted_in, counted_out, _, _ = self._handoff.pop(best_tid)
        return state, pending, counted_in, counted_out
