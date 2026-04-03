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
  5. Refractory period: after exiting, ignore re-entry within a short window
     to prevent boundary oscillation from double-counting.

Counting strategy:
  - A person is counted as soon as their dwell time reaches min_dwell_seconds
    (early trigger), so counts appear promptly.
  - When the person exits the zone, their state is cached briefly (refractory)
    so a quick re-entry resumes rather than restarts.
  - When tracking briefly loses a counted person during an overlap, a nearby
    replacement ID can inherit the counted state to avoid duplicate counts.
"""

from typing import Dict, List, Optional, Tuple

from .base import BaseCounter
from utils.geometry import point_in_polygon


def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


class _ZoneState:
    """Per-track zone state."""

    __slots__ = ("entry_time", "counted")

    def __init__(self, entry_time: float):
        self.entry_time: float = entry_time
        self.counted: bool = False  # True once the dwell threshold is reached


_ZoneHandoff = Tuple[_ZoneState, Tuple[float, float], float]


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
        self._refractory_sec: float = float(config.get("refractory_seconds", 0.17))
        self._handoff_sec: float = float(config.get("handoff_seconds", max(self._refractory_sec, 2.0)))
        self._handoff_radius: float = float(config.get("handoff_radius_px", 70.0))

        # Pre-compute numpy polygon (avoid array creation per update call)
        import numpy as np
        self._polygon_np = np.array(self.polygon, dtype=np.float32)

        self._states: Dict[int, _ZoneState] = {}  # track_id → state
        # Cache recently exited states for refractory re-entry
        self._exited: Dict[int, Tuple[_ZoneState, float]] = {}  # tid → (state, exit_timestamp)
        # Cache counted in-zone tracks that briefly disappeared during overlap.
        self._handoff: Dict[int, _ZoneHandoff] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, track_id: int, center: Tuple[float, float], timestamp: float
    ) -> dict:
        self._expire_exited(timestamp)
        self._expire_handoffs(timestamp)

        inside = point_in_polygon(center, self.polygon, self._polygon_np)
        state = self._states.get(track_id)
        result = {"counted": False}

        if inside:
            if state is None:
                # Check if re-entering within refractory period
                exited = self._exited.pop(track_id, None)
                if exited is not None:
                    old_state, exit_ts = exited
                    if timestamp - exit_ts < self._refractory_sec:
                        # Restore previous state (resume dwell timer)
                        self._states[track_id] = old_state
                        state = old_state

                if state is None:
                    inherited = self._claim_handoff(track_id, center, timestamp)
                    if inherited is not None:
                        self._states[track_id] = inherited
                        state = inherited

                if state is None:
                    # Genuine new entry — start dwell timer
                    self._states[track_id] = _ZoneState(entry_time=timestamp)
                    state = self._states[track_id]

            if not state.counted:
                dwell = timestamp - state.entry_time
                if dwell >= self.min_dwell:
                    self.count_in += 1
                    state.counted = True
                    result["counted"] = True
        else:
            if state is not None:
                # Person left the zone — cache state for refractory period
                self._exited[track_id] = (state, timestamp)
                del self._states[track_id]

        return result

    def mark_lost(
        self,
        track_id: int,
        center: Tuple[float, float] | None,
        timestamp: float,
    ) -> None:
        self._expire_exited(timestamp)
        self._expire_handoffs(timestamp)
        if center is None:
            return

        state = self._states.pop(track_id, None)
        if state is None or not state.counted:
            return

        self._handoff[track_id] = (
            state,
            (float(center[0]), float(center[1])),
            float(timestamp),
        )

    def remove_track(self, track_id: int) -> None:
        self._states.pop(track_id, None)
        self._exited.pop(track_id, None)
        self._handoff.pop(track_id, None)

    def transfer_state(self, from_tid: int, to_tid: int) -> None:
        """Transfer zone state from one track ID to another (for Re-ID remap)."""
        if from_tid == to_tid:
            return

        from_state = self._states.pop(from_tid, None)
        to_state = self._states.get(to_tid)
        if from_state is not None:
            if to_state is None:
                self._states[to_tid] = from_state
                to_state = from_state
            else:
                to_state.entry_time = min(to_state.entry_time, from_state.entry_time)
                to_state.counted = to_state.counted or from_state.counted

        from_exited = self._exited.pop(from_tid, None)
        if from_exited is not None and to_tid not in self._states:
            to_exited = self._exited.get(to_tid)
            if to_exited is None:
                self._exited[to_tid] = from_exited
            else:
                to_state_exited, to_exit_ts = to_exited
                from_state_exited, from_exit_ts = from_exited
                to_state_exited.entry_time = min(to_state_exited.entry_time, from_state_exited.entry_time)
                to_state_exited.counted = to_state_exited.counted or from_state_exited.counted
                self._exited[to_tid] = (to_state_exited, max(to_exit_ts, from_exit_ts))

        from_handoff = self._handoff.pop(from_tid, None)
        if from_handoff is not None:
            handoff_state, handoff_center, handoff_ts = from_handoff
            if to_tid in self._states:
                self._states[to_tid].entry_time = min(self._states[to_tid].entry_time, handoff_state.entry_time)
                self._states[to_tid].counted = self._states[to_tid].counted or handoff_state.counted
            else:
                to_handoff = self._handoff.get(to_tid)
                if to_handoff is None:
                    self._handoff[to_tid] = (handoff_state, handoff_center, handoff_ts)
                else:
                    to_state_handoff, to_center, to_ts = to_handoff
                    to_state_handoff.entry_time = min(to_state_handoff.entry_time, handoff_state.entry_time)
                    to_state_handoff.counted = to_state_handoff.counted or handoff_state.counted
                    self._handoff[to_tid] = (
                        to_state_handoff,
                        handoff_center if handoff_ts >= to_ts else to_center,
                        max(to_ts, handoff_ts),
                    )

    def get_counts(self) -> dict:
        # Zone counting only has an "in" count (number of valid visits)
        return {"in": self.count_in, "out": 0}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _expire_exited(self, timestamp: float) -> None:
        expired = [
            tid for tid, (_, exit_ts) in self._exited.items()
            if timestamp - exit_ts >= self._refractory_sec
        ]
        for tid in expired:
            del self._exited[tid]

    def _expire_handoffs(self, timestamp: float) -> None:
        expired = [
            tid for tid, (_, _, lost_ts) in self._handoff.items()
            if timestamp - lost_ts >= self._handoff_sec
        ]
        for tid in expired:
            del self._handoff[tid]

    def _claim_handoff(
        self,
        track_id: int,
        center: Tuple[float, float],
        timestamp: float,
    ) -> Optional[_ZoneState]:
        best_tid = None
        best_score = None
        for candidate_tid, (candidate_state, candidate_center, lost_ts) in self._handoff.items():
            if candidate_tid == track_id or not candidate_state.counted:
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

        state, _, _ = self._handoff.pop(best_tid)
        return state
