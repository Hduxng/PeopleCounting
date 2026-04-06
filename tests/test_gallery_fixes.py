"""
Tests for ReID gallery fixes (TDD — RED phase).

Issues covered:
  #3 — Memory leak: expired lost tracks leave orphaned metadata
  #2 — HistogramGallery greedy matching → should use Hungarian
  #5 — Merge detection false positive on crouch/stand (area-only)
  #6 — HistogramGallery lacks spatial constraint in recovery
"""

import cv2
import numpy as np
import pytest

from reid.embedder import ReIDEmbedder
from reid.gallery import TrackGallery, _normalize_feature
from reid.histogram_gallery import HistogramGallery


# ---------------------------------------------------------------------------
# Fixtures: lightweight mock embedder (no GPU needed)
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Deterministic embedder: maps crop mean color to a unit vector."""

    @property
    def feat_dim(self) -> int:
        return 16

    def __call__(self, crops: list[np.ndarray]) -> list[list[float]]:
        if not crops:
            return []
        result = []
        for crop in crops:
            mean_color = crop.mean(axis=(0, 1))  # [B, G, R]
            rng = np.random.RandomState(int(mean_color.sum()) % 2**31)
            vec = rng.randn(self.feat_dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            result.append(vec.tolist())
        return result


@pytest.fixture
def embedder():
    return MockEmbedder()


def _make_crop(color: tuple[int, int, int], w: int = 64, h: int = 128) -> np.ndarray:
    """Solid BGR crop."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def _make_hsv_crop(hue: int, w: int = 64, h: int = 128) -> np.ndarray:
    """Create a solid-colour BGR crop from HSV hue (0-179)."""
    hsv = np.full((h, w, 3), [hue, 200, 200], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _bbox(x: float, y: float, w: float, h: float) -> np.ndarray:
    """Create xyxy bbox from center + size."""
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float64)


# =========================================================================
# #3 — Memory leak: orphaned metadata after lost track expiry
# =========================================================================


class TestMemoryLeakOnExpiry:
    """_age_lost_gallery must clean ALL per-track dicts when a lost track expires."""

    def test_last_area_cleaned_on_expiry(self, embedder):
        gallery = TrackGallery(embedder=embedder, lifetime=3)
        crop = _make_crop((100, 150, 200))
        bbox = _bbox(300, 300, 50, 120)

        # Track 1 active with bbox (populates _last_area)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: bbox})
        assert 1 in gallery._last_area

        # Track 1 disappears, wait past lifetime
        for _ in range(5):
            gallery.update(set(), {})

        assert 1 not in gallery._lost, "Track should have expired"
        assert 1 not in gallery._last_area, "_last_area not cleaned on expiry"

    def test_last_bbox_cleaned_on_expiry(self, embedder):
        gallery = TrackGallery(embedder=embedder, lifetime=3)
        crop = _make_crop((100, 150, 200))
        bbox = _bbox(300, 300, 50, 120)

        gallery.update({1}, {1: crop}, bboxes_by_tid={1: bbox})
        assert 1 in gallery._last_bbox

        for _ in range(5):
            gallery.update(set(), {})

        assert 1 not in gallery._last_bbox, "_last_bbox not cleaned on expiry"

    def test_merge_state_cleaned_on_expiry(self, embedder):
        gallery = TrackGallery(embedder=embedder, lifetime=3)
        crop = _make_crop((100, 150, 200))
        small_bbox = _bbox(300, 300, 50, 120)
        big_bbox = _bbox(300, 300, 100, 120)  # 2x width → merge trigger

        # Frame 1: small bbox
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: small_bbox})
        # Frame 2: big bbox → triggers merge detection
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: big_bbox})

        # Verify merge state was created
        had_merge_state = (
            1 in gallery._pre_merge_area
            or 1 in gallery._merge_counts
            or 1 in gallery._merged_ids
        )

        # Track disappears, wait past lifetime
        for _ in range(5):
            gallery.update(set(), {})

        assert 1 not in gallery._pre_merge_area, "_pre_merge_area not cleaned"
        assert 1 not in gallery._merge_counts, "_merge_counts not cleaned"
        assert 1 not in gallery._merged_ids, "_merged_ids not cleaned"

    def test_drift_counts_cleaned_on_expiry(self, embedder):
        gallery = TrackGallery(embedder=embedder, lifetime=3)
        crop = _make_crop((100, 150, 200))

        gallery.update({1}, {1: crop})

        # Manually inject drift state (simulating partial drift detection)
        gallery._drift_counts[1] = 1

        for _ in range(5):
            gallery.update(set(), {})

        assert 1 not in gallery._drift_counts, "_drift_counts not cleaned"

    def test_no_leak_over_many_tracks(self, embedder):
        """Simulate many tracks appearing and expiring — metadata dicts should not grow."""
        lifetime = 2
        gallery = TrackGallery(embedder=embedder, lifetime=lifetime)

        for tid in range(1, 101):
            crop = _make_crop((tid % 256, 100, 200))
            bbox = _bbox(float(tid * 10 % 640), 300.0, 50.0, 120.0)
            gallery.update({tid}, {tid: crop}, bboxes_by_tid={tid: bbox})
            # Immediately lose the track
            gallery.update(set(), {})

        # Age well past lifetime (lifetime + margin)
        for _ in range(lifetime + 3):
            gallery.update(set(), {})

        # All metadata dicts should be empty
        assert len(gallery._last_area) == 0, f"_last_area leaked {len(gallery._last_area)} entries"
        assert len(gallery._last_bbox) == 0, f"_last_bbox leaked {len(gallery._last_bbox)} entries"
        assert len(gallery._pre_merge_area) == 0
        assert len(gallery._merge_counts) == 0
        assert len(gallery._drift_counts) == 0
        assert len(gallery._lost) == 0
        assert len(gallery._active) == 0


# =========================================================================
# #2 — HistogramGallery: greedy matching fails when 2 new tracks compete
# =========================================================================


class TestHistogramHungarianMatching:
    """HistogramGallery should use optimal (Hungarian) matching, not greedy."""

    def test_two_reappear_simultaneously_correct_assignment(self):
        """When two lost tracks reappear at once, each should match its own colour."""
        g = HistogramGallery(lifetime=30, match_threshold=0.7)

        red = _make_hsv_crop(0)
        blue = _make_hsv_crop(120)

        # Frame 1: both active
        g.update({1, 2}, {1: red, 2: blue})
        # Frame 2: both disappear
        g.update(set(), {})
        # Frame 3: both reappear with new IDs
        remap = g.update({90, 91}, {90: red, 91: blue})

        # Optimal: 90→1 (red→red), 91→2 (blue→blue)
        assert remap.get(90) == 1, f"Red track should map to 1, got {remap}"
        assert remap.get(91) == 2, f"Blue track should map to 2, got {remap}"

    def test_greedy_would_assign_wrong_but_hungarian_correct(self):
        """
        Scenario where greedy fails:
        - Lost: tid=1 (green), tid=2 (green-ish, slightly different)
        - New:  tid=10 (green-ish, closer to tid=2), tid=11 (green, closer to tid=1)
        - Greedy processes tid=10 first → steals tid=1 (best overall for tid=10)
        - Hungarian correctly assigns tid=10→2, tid=11→1
        """
        g = HistogramGallery(lifetime=30, match_threshold=0.7, ema_alpha=1.0)

        # Two distinct greens
        green_a = _make_hsv_crop(55)   # tid=1's colour
        green_b = _make_hsv_crop(65)   # tid=2's colour

        g.update({1, 2}, {1: green_a, 2: green_b})
        g.update(set(), {})

        # New tracks: 10 gets green_b, 11 gets green_a
        remap = g.update({10, 11}, {10: green_b, 11: green_a})

        assert remap.get(10) == 2, f"tid=10 (green_b) should map to 2, got {remap}"
        assert remap.get(11) == 1, f"tid=11 (green_a) should map to 1, got {remap}"


# =========================================================================
# #5 — Merge detection: false positive on crouch/stand (area-only)
# =========================================================================


class TestMergeDetectionAspectRatio:
    """Merge detection should consider shape change, not just area."""

    def test_width_increase_triggers_merge(self, embedder):
        """Two people merging → width doubles. Should trigger merge freeze."""
        gallery = TrackGallery(embedder=embedder, lifetime=30, merge_area_ratio=1.5)
        crop = _make_crop((100, 150, 200))

        # Normal standing bbox
        normal_bbox = _bbox(300, 300, 50, 120)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: normal_bbox})

        # Merged bbox: width doubles (two people side by side), area ~2x
        merged_bbox = _bbox(300, 300, 100, 120)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: merged_bbox})

        assert 1 in gallery._merged_ids, "Width increase (true merge) should trigger merge"

    def test_height_increase_no_merge(self, embedder):
        """Person standing up from crouch → height increases, width stays same.
        Should NOT trigger merge freeze."""
        gallery = TrackGallery(embedder=embedder, lifetime=30, merge_area_ratio=1.5)
        crop = _make_crop((100, 150, 200))

        # Crouching bbox: short and wide-ish
        crouch_bbox = _bbox(300, 300, 55, 70)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: crouch_bbox})

        # Standing bbox: taller, similar width → area ~2x but width barely changed
        stand_bbox = _bbox(300, 300, 55, 140)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: stand_bbox})

        assert 1 not in gallery._merged_ids, \
            "Height-only increase (crouch→stand) should NOT trigger merge"

    def test_proportional_scale_no_merge(self, embedder):
        """Person walks closer to camera → proportional scale increase.
        Should NOT trigger merge (width ratio matches height ratio)."""
        gallery = TrackGallery(embedder=embedder, lifetime=30, merge_area_ratio=1.5)
        crop = _make_crop((100, 150, 200))

        far_bbox = _bbox(300, 300, 44, 96)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: far_bbox})

        # ~1.25x scale in both dimensions → area ~1.56x (above 1.5 threshold)
        # but width ratio 55/44 = 1.25 < 1.3 threshold
        close_bbox = _bbox(300, 300, 55, 120)
        gallery.update({1}, {1: crop}, bboxes_by_tid={1: close_bbox})

        assert 1 not in gallery._merged_ids, \
            "Proportional scale increase should NOT trigger merge"


# =========================================================================
# #6 — HistogramGallery: no spatial constraint in recovery
# =========================================================================


class TestHistogramSpatialConstraint:
    """HistogramGallery should reject spatially impossible matches."""

    def test_same_colour_far_away_no_match(self):
        """Two red people at opposite ends — should NOT recover."""
        g = HistogramGallery(lifetime=30, match_threshold=0.7)
        red = _make_hsv_crop(0)

        bbox_left = _bbox(50, 300, 50, 120)
        bbox_right = _bbox(1200, 300, 50, 120)

        # Track 1 at left side
        g.update({1}, {1: red}, bboxes_by_tid={1: bbox_left})
        # Track 1 disappears
        g.update(set(), {}, bboxes_by_tid={})
        # Track 99 appears far right with same colour
        remap = g.update({99}, {99: red}, bboxes_by_tid={99: bbox_right})

        assert remap == {}, \
            "Same colour at far distance should NOT be recovered"

    def test_same_colour_nearby_matches(self):
        """Same colour reappearing nearby — should recover."""
        g = HistogramGallery(lifetime=30, match_threshold=0.7)
        red = _make_hsv_crop(0)

        bbox1 = _bbox(300, 300, 50, 120)
        bbox2 = _bbox(320, 310, 50, 120)  # nearby

        g.update({1}, {1: red}, bboxes_by_tid={1: bbox1})
        g.update(set(), {}, bboxes_by_tid={})
        remap = g.update({99}, {99: red}, bboxes_by_tid={99: bbox2})

        assert remap == {99: 1}, "Same colour nearby should recover"


# =========================================================================
# #4 — Protocol class for embedder interface
# =========================================================================


class TestEmbedderProtocol:
    """All embedders should satisfy the ReIDEmbedder protocol."""

    def test_mock_embedder_is_reid_embedder(self):
        """MockEmbedder (used in tests) satisfies ReIDEmbedder protocol."""
        emb = MockEmbedder()
        assert isinstance(emb, ReIDEmbedder)

    def test_protocol_feat_dim(self):
        emb = MockEmbedder()
        assert emb.feat_dim == 16

    def test_protocol_call(self):
        emb = MockEmbedder()
        crop = _make_crop((100, 150, 200))
        result = emb([crop])
        assert len(result) == 1
        assert len(result[0]) == 16

    def test_protocol_empty_input(self):
        emb = MockEmbedder()
        assert emb([]) == []
