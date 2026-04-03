"""Tests for post-tracker TrackValidator."""

import numpy as np
import pytest

from utils.validation import TrackValidator


class TestTrackValidator:

    def test_consistent_bbox_no_penalty(self):
        """Consistent bbox across frames → no confidence penalty."""
        v = TrackValidator(max_ar_change=0.3, max_area_change=0.5, min_track_age=1)
        bbox = np.array([100, 100, 200, 400])
        # First frame (young penalty)
        c1 = v.validate(1, bbox, 0.9)
        # Second frame (age >= min_track_age=1, consistent bbox)
        c2 = v.validate(1, bbox, 0.9)
        assert c2 == 0.9  # no penalty

    def test_ar_change_penalized(self):
        """Large aspect ratio change → confidence penalized."""
        v = TrackValidator(max_ar_change=0.3, confidence_penalty=0.5, min_track_age=0)
        v.validate(1, np.array([100, 100, 200, 400]), 0.9)  # AR = 300/100 = 3.0
        # Now AR changes drastically: 100/200 = 0.5
        c = v.validate(1, np.array([100, 100, 300, 200]), 0.9)
        assert c < 0.9

    def test_area_change_penalized(self):
        """Large area change → confidence penalized."""
        v = TrackValidator(max_area_change=0.5, confidence_penalty=0.5, min_track_age=0)
        v.validate(1, np.array([100, 100, 200, 300]), 0.9)  # area = 100*200 = 20000
        # Area doubles: 200*400 = 80000
        c = v.validate(1, np.array([100, 100, 300, 500]), 0.9)
        assert c < 0.9

    def test_young_track_penalty(self):
        """Tracks younger than min_track_age get penalized."""
        v = TrackValidator(min_track_age=5)
        bbox = np.array([100, 100, 200, 300])
        c = v.validate(1, bbox, 1.0)
        assert c == pytest.approx(0.8)  # young penalty

    def test_remove_cleans_state(self):
        """remove() cleans up track history."""
        v = TrackValidator()
        v.validate(1, np.array([100, 100, 200, 300]), 0.9)
        assert 1 in v._history
        v.remove(1)
        assert 1 not in v._history
