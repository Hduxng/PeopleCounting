"""Tests for TrackAnchor with velocity EMA and adaptive drift."""

import numpy as np
import pytest

from utils.anchor import TrackAnchor


class TestVelocityEMA:
    """Velocity EMA smoothing reduces prediction noise."""

    def test_stationary_track_zero_velocity(self):
        """A track with identical bboxes across frames → zero velocity prediction."""
        anchor = TrackAnchor(stationary_threshold=0.5, velocity_window=5)
        bbox = np.array([100, 100, 200, 300])
        for _ in range(5):
            anchor.update(1, bbox)

        # Predict: should use the same position (zero velocity)
        pred = anchor._predict(1, 1280, 720)
        assert pred is not None
        np.testing.assert_allclose(pred, [100, 100, 200, 300], atol=1.0)

    def test_noisy_positions_smoothed(self):
        """Noisy positions should produce smoothed velocity closer to mean than to last delta."""
        anchor = TrackAnchor(velocity_window=5, velocity_ema_alpha=0.4, stationary_threshold=0.0)
        # Moving right at ~10px/frame with noise
        positions = [
            [100, 100, 200, 300],
            [112, 100, 212, 300],  # +12
            [118, 100, 218, 300],  # +6
            [130, 100, 230, 300],  # +12
            [138, 100, 238, 300],  # +8
        ]
        for p in positions:
            anchor.update(1, np.array(p))

        pred = anchor._predict(1, 1280, 720)
        assert pred is not None
        # Should predict roughly 10px right of last position
        expected_x1 = 138 + 10  # ~148 ± noise
        assert 140 < pred[0] < 160  # smoothed, not exact

    def test_velocity_window_limits_history(self):
        """History should not grow beyond velocity_window."""
        anchor = TrackAnchor(velocity_window=3)
        for i in range(10):
            anchor.update(1, np.array([100 + i * 10, 100, 200 + i * 10, 300]))
        assert len(anchor._history[1]) == 3


class TestAdaptiveDrift:
    """Drift cap should scale with bbox size."""

    def test_large_bbox_allows_more_drift(self):
        """A large bounding box should have a higher drift cap than default 30px."""
        anchor = TrackAnchor(
            max_drift_px=30.0,
            velocity_window=3,
            stationary_threshold=0.0,
            velocity_damping=1.0,  # no damping for this test
        )
        # Large bbox: 200x400 → diagonal ~447 → adaptive drift = 0.15*447 ≈ 67
        positions = [
            [100, 100, 300, 500],
            [100, 100, 300, 500],
            [200, 100, 400, 500],  # jumped 100px right
        ]
        for p in positions:
            anchor.update(1, np.array(p))

        pred = anchor._predict(1, 1280, 720)
        assert pred is not None
        # With adaptive drift, a 100px velocity should be capped to ~67px
        # (0.15 * sqrt(200^2 + 400^2) ≈ 67)
        assert pred[0] > 230  # more than fixed 30px drift

    def test_small_bbox_uses_baseline_drift(self):
        """A small bbox uses the baseline max_drift_px."""
        anchor = TrackAnchor(
            max_drift_px=30.0,
            velocity_window=3,
            stationary_threshold=0.0,
            velocity_damping=1.0,
        )
        # Small bbox: 30x50 → diagonal ~58 → adaptive = 0.15*58 ≈ 8.7 < 30
        # So baseline 30px is used
        positions = [
            [100, 100, 130, 150],
            [100, 100, 130, 150],
            [200, 100, 230, 150],  # jumped 100px right
        ]
        for p in positions:
            anchor.update(1, np.array(p))

        pred = anchor._predict(1, 1280, 720)
        assert pred is not None
        # Capped at 30px from last position
        assert pred[0] <= 230 + 1  # 200 + 30


class TestAugment:
    """augment() injects synthetic detections correctly."""

    def test_no_injection_when_covered(self):
        """Track covered by a real detection → no synthetic injection."""
        anchor = TrackAnchor(min_iou=0.1, min_hits=1)
        anchor.update(1, np.array([100, 100, 200, 300]))
        anchor._hits[1] = 5

        dets = np.array([[100, 100, 200, 300, 0.9, 0]], dtype=float)
        aug, syn_map = anchor.augment({1}, dets, (720, 1280, 3))
        assert len(aug) == 1  # no synthetic added
        assert syn_map == {}

    def test_injection_when_uncovered(self):
        """Track NOT covered by any detection → synthetic injected."""
        anchor = TrackAnchor(min_iou=0.1, min_hits=1)
        anchor.update(1, np.array([100, 100, 200, 300]))
        anchor._hits[1] = 5

        dets = np.array([[500, 500, 600, 700, 0.9, 0]], dtype=float)
        aug, syn_map = anchor.augment({1}, dets, (720, 1280, 3))
        assert len(aug) == 2  # original + 1 synthetic
        assert 1 in syn_map  # det_index 1 → tid 1

    def test_camera_motion_passed(self):
        """Camera motion parameter should be accepted without error."""
        anchor = TrackAnchor(min_hits=1)
        anchor.update(1, np.array([100, 100, 200, 300]))
        anchor._hits[1] = 5

        dets = np.empty((0, 6), dtype=float)
        aug, syn_map = anchor.augment({1}, dets, (720, 1280, 3), camera_motion=(5.0, 3.0))
        assert len(aug) >= 1


class TestCoverageGeometry:
    """Coverage check should not treat a large merge box as a valid single-person match."""

    def test_large_merge_box_does_not_count_as_coverage(self):
        anchor = TrackAnchor(min_iou=0.1, min_hits=1)
        track_box = [100, 100, 150, 250]
        merged_box = np.array([[75, 95, 205, 255, 0.9, 0]], dtype=float)

        assert anchor._has_overlap(track_box, merged_box) is False

    def test_same_person_box_still_counts_as_coverage(self):
        anchor = TrackAnchor(min_iou=0.1, min_hits=1)
        track_box = [100, 100, 200, 300]
        same_person_box = np.array([[108, 118, 192, 300, 0.9, 0]], dtype=float)

        assert anchor._has_overlap(track_box, same_person_box) is True
