"""
Level 1-2: Counter unit tests.

Tests CrosslineCounter and ZoneCounter logic directly with simulated
track center points — no detection or tracking involved.
"""

import pytest

from counter.crossline import CrosslineCounter
from counter.zone import ZoneCounter


# =========================================================================
# LEVEL 1 — Basic
# =========================================================================


class TestLevel1_BasicCrossline:
    """1.4, 1.5: Single person crossing a line."""

    def test_1_4_single_person_crosses_line_enter(self, crossline_cfg):
        """One person walks top→bottom across horizontal line → IN=1."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 1

        # Start above line (y < 360), move down past it
        for y in range(100, 600, 10):
            counter.update(tid, (640, y), timestamp=y / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 1, f"Expected IN=1, got {counts}"
        assert counts["out"] == 0

    def test_1_5_single_person_crosses_line_exit(self, crossline_cfg):
        """One person walks bottom→top → OUT=1."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 1

        # Start below line (y > 360), move up
        for y in range(600, 100, -10):
            counter.update(tid, (640, y), timestamp=(600 - y) / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 0
        assert counts["out"] == 1, f"Expected OUT=1, got {counts}"

    def test_1_1_single_person_stable_id(self, crossline_cfg):
        """One person passes through — same ID throughout, counted once."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 42

        for y in range(100, 600, 5):
            counter.update(tid, (640, y), timestamp=y / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 1

    def test_1_2_multiple_separate_people(self, crossline_cfg):
        """3 people walk top→bottom separately → IN=3."""
        counter = CrosslineCounter(crossline_cfg)

        for tid in [1, 2, 3]:
            x = 200 + tid * 200
            for y in range(100, 600, 10):
                counter.update(tid, (x, y), timestamp=y / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 3


class TestLevel1_BasicZone:
    """1.3: Single person entering and dwelling in a zone."""

    def test_1_3_person_stays_in_zone(self, zone_cfg):
        """Person enters zone, stays >= min_dwell → counted once."""
        counter = ZoneCounter(zone_cfg)
        tid = 1
        # Stand in center of zone for 3 seconds (> 2.0 min_dwell)
        for frame in range(90):  # 90 frames at 30fps = 3s
            counter.update(tid, (600, 350), timestamp=frame / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 1

    def test_person_leaves_before_dwell(self, zone_cfg):
        """Person enters but leaves before min_dwell → NOT counted."""
        counter = ZoneCounter(zone_cfg)
        tid = 1
        # In zone for 1 second (< 2.0 min_dwell)
        for frame in range(30):
            counter.update(tid, (600, 350), timestamp=frame / 30.0)
        # Leave zone
        for frame in range(30, 60):
            counter.update(tid, (100, 100), timestamp=frame / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 0

    def test_person_outside_zone_not_counted(self, zone_cfg):
        """Person never enters zone → count=0."""
        counter = ZoneCounter(zone_cfg)
        for frame in range(120):
            counter.update(1, (100, 100), timestamp=frame / 30.0)

        assert counter.get_counts()["in"] == 0


# =========================================================================
# LEVEL 2 — Intermediate
# =========================================================================


class TestLevel2_CrosslineEdgeCases:
    """2.3, 2.5: Line crossing edge cases."""

    def test_2_3_cross_back_and_forth(self, crossline_cfg):
        """Person crosses line → goes back → crosses again.
        Expected: IN=1, OUT=1 (counted once per direction per ID)."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 1

        # Cross down (enter)
        for y in range(100, 500, 10):
            counter.update(tid, (640, y), timestamp=y / 30.0)

        # Cross back up (exit)
        for y in range(500, 100, -10):
            counter.update(tid, (640, y), timestamp=(500 + (500 - y)) / 30.0)

        # Cross down again — should NOT count again (already counted for this ID)
        for y in range(100, 500, 10):
            counter.update(tid, (640, y), timestamp=(1000 + y) / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 1, f"Should count IN only once per ID, got {counts}"
        assert counts["out"] == 1

    def test_2_5_oscillation_near_line(self, crossline_cfg):
        """Person oscillates within buffer zone → NO count."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 1

        # First, establish a side
        counter.update(tid, (640, 300), timestamp=0)

        # Oscillate within buffer zone (360 ± 20)
        for i in range(50):
            y = 355 + (10 if i % 2 == 0 else -10)  # 345-365, within ±20 buffer
            counter.update(tid, (640, y), timestamp=(i + 1) / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 0, f"Oscillation within buffer should not count, got {counts}"
        assert counts["out"] == 0

    def test_multiple_people_different_directions(self, crossline_cfg):
        """Person A enters, Person B exits → IN=1, OUT=1."""
        counter = CrosslineCounter(crossline_cfg)

        # Person A: top → bottom (enter)
        for y in range(100, 600, 10):
            counter.update(1, (400, y), timestamp=y / 30.0)

        # Person B: bottom → top (exit)
        for y in range(600, 100, -10):
            counter.update(2, (800, y), timestamp=y / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 1
        assert counts["out"] == 1


class TestLevel2_ZoneEdgeCases:
    """2.2: Zone re-entry and edge cases."""

    def test_2_2_person_enter_leave_reenter(self, fast_zone_cfg):
        """Person enters zone, leaves, enters again → counted 2 times."""
        counter = ZoneCounter(fast_zone_cfg)
        tid = 1
        fps = 30.0

        # First entry: stay for 1 second (> 0.5s min_dwell)
        for f in range(30):
            counter.update(tid, (600, 350), timestamp=f / fps)

        # Leave zone
        for f in range(30, 45):
            counter.update(tid, (100, 100), timestamp=f / fps)

        # Second entry: stay for 1 second
        for f in range(45, 75):
            counter.update(tid, (600, 350), timestamp=f / fps)

        counts = counter.get_counts()
        assert counts["in"] == 2, f"Expected 2 entries, got {counts}"

    def test_zone_boundary_in_out(self, fast_zone_cfg):
        """Person walks along zone boundary — should not count if never fully inside."""
        counter = ZoneCounter(fast_zone_cfg)
        # Zone is [[300,200], [900,200], [900,500], [300,500]]
        # Walk along the left edge just outside
        for f in range(60):
            counter.update(1, (299, 200 + f * 5), timestamp=f / 30.0)

        assert counter.get_counts()["in"] == 0

    def test_multiple_people_in_zone(self, fast_zone_cfg):
        """3 people in zone simultaneously, each dwelling enough → count=3."""
        counter = ZoneCounter(fast_zone_cfg)
        fps = 30.0

        for f in range(30):  # 1 second > 0.5s dwell
            t = f / fps
            counter.update(1, (400, 300), t)
            counter.update(2, (600, 300), t)
            counter.update(3, (800, 400), t)

        assert counter.get_counts()["in"] == 3


class TestLevel2_TrackRemoval:
    """Counter state cleanup after track removal."""

    def test_crossline_remove_track(self, crossline_cfg):
        """After remove_track, counter state for that ID is cleared."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 1
        counter.update(tid, (640, 300), timestamp=0)
        counter.remove_track(tid)

        # Internal state should be gone
        assert tid not in counter._states

    def test_zone_remove_track(self, fast_zone_cfg):
        """After remove_track, zone state for that ID is cleared."""
        counter = ZoneCounter(fast_zone_cfg)
        tid = 1
        counter.update(tid, (600, 350), timestamp=0)
        counter.remove_track(tid)
        assert tid not in counter._states


# =========================================================================
# LEVEL 2 — Speed variations
# =========================================================================


class TestLevel2_Speed:
    """2.4, 2.6: Fast and slow movement."""

    def test_2_4_fast_person(self, crossline_cfg):
        """Person crosses line in just 3 frames (fast) → still counted."""
        counter = CrosslineCounter(crossline_cfg)
        tid = 1
        # 3 positions: well above, on line, well below
        counter.update(tid, (640, 200), timestamp=0)
        counter.update(tid, (640, 360), timestamp=1 / 30)
        counter.update(tid, (640, 550), timestamp=2 / 30)

        counts = counter.get_counts()
        assert counts["in"] == 1

    def test_2_6_fast_and_slow_together(self, crossline_cfg):
        """Fast person + slow person crossing simultaneously → both counted."""
        counter = CrosslineCounter(crossline_cfg)

        # Slow person: 50 frames to cross
        for f in range(50):
            y = 200 + f * 6  # 200→500
            counter.update(1, (400, y), timestamp=f / 30.0)

        # Fast person: 5 frames to cross
        for f in range(5):
            y = 200 + f * 80  # 200→520
            counter.update(2, (800, y), timestamp=f / 30.0)

        counts = counter.get_counts()
        assert counts["in"] == 2


class TestLevel2_StateTransfer:
    """Counter state should survive remaps onto an existing canonical ID."""

    def test_zone_transfer_state_does_not_reset_counted_canonical(self, fast_zone_cfg):
        counter = ZoneCounter(fast_zone_cfg)

        for f in range(20):
            counter.update(55, (600, 350), timestamp=f / 30.0)
        assert counter.get_counts()["in"] == 1
        assert counter._states[55].counted is True

        counter.update(108, (600, 350), timestamp=20 / 30.0)
        assert counter._states[108].counted is False

        counter.transfer_state(108, 55)

        assert counter._states[55].counted is True
        before = counter.get_counts()["in"]
        for f in range(21, 40):
            counter.update(55, (600, 350), timestamp=f / 30.0)
        assert counter.get_counts()["in"] == before

    def test_crossline_transfer_state_does_not_reset_existing_canonical(self, crossline_cfg):
        counter = CrosslineCounter(crossline_cfg)

        for y in range(100, 600, 20):
            counter.update(55, (640, y), timestamp=y / 30.0)
        assert counter.get_counts()["in"] == 1
        assert 55 in counter._counted_in_ids

        counter.update(108, (640, 120), timestamp=1000 / 30.0)
        counter.transfer_state(108, 55)

        before = counter.get_counts()["in"]
        for y in range(120, 600, 20):
            counter.update(55, (640, y), timestamp=1100 / 30.0 + y / 30.0)
        assert counter.get_counts()["in"] == before


    def test_zone_transfer_merges_exited_state_and_preserves_refractory(self, fast_zone_cfg):
        counter = ZoneCounter(fast_zone_cfg)

        for f in range(20):
            counter.update(55, (600, 350), timestamp=f / 30.0)
        counter.update(55, (100, 100), timestamp=20 / 30.0)
        assert 55 in counter._exited
        assert counter._exited[55][0].counted is True

        for f in range(21, 24):
            counter.update(108, (600, 350), timestamp=f / 30.0)
        counter.update(108, (100, 100), timestamp=24 / 30.0)
        assert 108 in counter._exited
        assert counter._exited[108][0].counted is False

        counter.transfer_state(108, 55)

        exited_state, exit_ts = counter._exited[55]
        assert exited_state.counted is True
        assert exit_ts == 24 / 30.0
        before = counter.get_counts()["in"]

        counter.update(55, (600, 350), timestamp=25 / 30.0)

        assert counter.get_counts()["in"] == before
        assert counter._states[55].counted is True

    def test_crossline_transfer_merges_pending_same_side_progress(self, crossline_cfg):
        cfg = dict(crossline_cfg)
        cfg["confirm_frames"] = 3
        counter = CrosslineCounter(cfg)

        counter.update(55, (640, 100), timestamp=0.0)
        counter.update(55, (640, 600), timestamp=1 / 30.0)
        counter.update(55, (640, 610), timestamp=2 / 30.0)
        assert counter._pending[55] == (1, 2)

        counter.update(108, (640, 100), timestamp=0.0)
        counter.update(108, (640, 600), timestamp=1 / 30.0)
        assert counter._pending[108] == (1, 1)

        counter.transfer_state(108, 55)

        assert counter._pending[55] == (1, 2)
        result = counter.update(55, (640, 620), timestamp=3 / 30.0)
        assert result["entered"] is True
        assert counter.get_counts()["in"] == 1


class TestLevel2_OverlapHandoff:
    """Counter state should survive short overlap-induced ID churn."""

    def test_zone_handoff_keeps_counted_state_for_nearby_new_id(self, fast_zone_cfg):
        cfg = dict(fast_zone_cfg)
        cfg["handoff_seconds"] = 2.0
        cfg["handoff_radius_px"] = 80.0
        counter = ZoneCounter(cfg)

        for f in range(20):
            counter.update(22, (600, 350), timestamp=f / 30.0)

        assert counter.get_counts()["in"] == 1
        assert counter._states[22].counted is True

        counter.mark_lost(22, (604, 352), timestamp=20 / 30.0)

        before = counter.get_counts()["in"]
        for f in range(21, 40):
            counter.update(66, (607, 351), timestamp=f / 30.0)

        assert counter.get_counts()["in"] == before
        assert counter._states[66].counted is True

    def test_zone_handoff_does_not_block_far_new_person(self, fast_zone_cfg):
        cfg = dict(fast_zone_cfg)
        cfg["handoff_seconds"] = 2.0
        cfg["handoff_radius_px"] = 50.0
        counter = ZoneCounter(cfg)

        for f in range(20):
            counter.update(22, (420, 350), timestamp=f / 30.0)
        counter.mark_lost(22, (420, 350), timestamp=20 / 30.0)

        for f in range(21, 40):
            counter.update(66, (840, 350), timestamp=f / 30.0)

        assert counter.get_counts()["in"] == 2

    def test_crossline_handoff_preserves_pending_crossing(self, crossline_cfg):
        cfg = dict(crossline_cfg)
        cfg["confirm_frames"] = 3
        cfg["handoff_seconds"] = 1.0
        cfg["handoff_radius_px"] = 80.0
        counter = CrosslineCounter(cfg)

        counter.update(22, (640, 100), timestamp=0.0)
        counter.update(22, (640, 600), timestamp=1 / 30.0)
        counter.update(22, (640, 610), timestamp=2 / 30.0)
        assert counter._pending[22] == (1, 2)

        counter.mark_lost(22, (640, 610), timestamp=2 / 30.0)

        result = counter.update(66, (642, 620), timestamp=3 / 30.0)

        assert result["entered"] is True
        assert counter.get_counts()["in"] == 1
