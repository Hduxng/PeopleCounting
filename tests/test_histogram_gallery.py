"""Tests for the histogram-based Re-ID gallery."""

import cv2
import numpy as np
import pytest

from reid.histogram_gallery import HistogramGallery


def _make_crop(hue: int, width: int = 64, height: int = 128) -> np.ndarray:
    """Create a solid-colour BGR crop with the given HSV hue (0-179)."""
    hsv = np.full((height, width, 3), [hue, 200, 200], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestHistogramGallery:

    def test_no_remap_when_no_lost_tracks(self):
        g = HistogramGallery(lifetime=30)
        crop = _make_crop(60)
        remap = g.update({1, 2}, {1: crop, 2: crop})
        assert remap == {}

    def test_recover_lost_track_same_colour(self):
        g = HistogramGallery(lifetime=30, match_threshold=0.7)
        red = _make_crop(0)

        # Frame 1: track 1 is active
        g.update({1}, {1: red})
        # Frame 2: track 1 disappears
        g.update(set(), {})
        # Frame 3: track 99 appears with same colour → should remap to 1
        remap = g.update({99}, {99: red})
        assert remap == {99: 1}

    def test_no_recovery_different_colour(self):
        g = HistogramGallery(lifetime=30, match_threshold=0.3)
        red = _make_crop(0)
        blue = _make_crop(120)

        g.update({1}, {1: red})
        g.update(set(), {})
        remap = g.update({99}, {99: blue})
        assert remap == {}

    def test_lost_track_expires(self):
        g = HistogramGallery(lifetime=3, match_threshold=0.7)
        red = _make_crop(0)

        g.update({1}, {1: red})
        # Lose track 1 then wait past lifetime
        for _ in range(5):
            g.update(set(), {})
        remap = g.update({99}, {99: red})
        assert remap == {}  # expired, no recovery

    def test_identity_drift_detection(self):
        g = HistogramGallery(lifetime=30, drift_threshold=0.3, match_threshold=0.7)
        red = _make_crop(0)
        blue = _make_crop(120)

        # Frame 1: track 1 is red
        g.update({1}, {1: red})
        # Frame 2: track 1 suddenly becomes blue (drift)
        g.update({1}, {1: blue})
        # Old red feature should now be in lost gallery
        assert 1 in g._lost

    def test_multiple_tracks_independent(self):
        g = HistogramGallery(lifetime=30, match_threshold=0.7)
        red = _make_crop(0)
        green = _make_crop(60)

        g.update({1, 2}, {1: red, 2: green})
        g.update(set(), {})
        # Track 99 is red → should map to 1, not 2
        remap = g.update({99}, {99: red})
        assert remap.get(99) == 1

    def test_small_crop_skipped(self):
        g = HistogramGallery(lifetime=30, min_crop_area=10000)
        tiny = _make_crop(0, width=10, height=10)  # 300 bytes < 10000
        g.update({1}, {1: tiny})
        assert 1 not in g._active

    def test_ema_update(self):
        g = HistogramGallery(lifetime=30, ema_alpha=0.5)
        red = _make_crop(0)

        g.update({1}, {1: red})
        h1 = g._active[1].copy()
        g.update({1}, {1: red})
        h2 = g._active[1]
        # After EMA with same input, histogram should be very close
        np.testing.assert_allclose(h1, h2, atol=0.01)

    def test_remove_track(self):
        g = HistogramGallery(lifetime=30)
        red = _make_crop(0)
        g.update({1}, {1: red})
        g.remove(1)
        assert 1 not in g._active
        assert 1 not in g._lost
