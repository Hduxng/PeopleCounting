"""
Tests for 3 missing ID switch fixes on pc_v2:
  Fix #2: EMA blend guard — skip blending when new feature is wildly different
  Fix #4: get_feature() public API — encapsulate _active access
  Fix #7: is_synthetic_consistent() — validate synthetic associations
"""

import numpy as np
import pytest

from utils.anchor import TrackAnchor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.call_count = 0

    def __call__(self, crops: list) -> list:
        self.call_count += 1
        results = []
        for crop in crops:
            v = np.full(self.dim, float(crop.mean()), dtype=np.float32)
            norm = np.linalg.norm(v)
            results.append((v / norm if norm > 1e-6 else v).tolist())
        return results


def _make_crop(value: float = 100.0, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _unit_vec(dim: int, seed: float) -> np.ndarray:
    rng = np.random.RandomState(int(seed * 1000) % 2**31)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=512)


@pytest.fixture
def gallery(embedder):
    from reid.gallery import TrackGallery
    return TrackGallery(
        embedder=embedder,
        lifetime=10,
        ema_alpha=0.85,
        match_threshold=0.30,
        min_crop_area=100,
    )


@pytest.fixture
def anchor() -> TrackAnchor:
    return TrackAnchor(
        min_iou=0.15,
        synthetic_conf=0.45,
        min_hits=3,
        max_inject=10,
    )


# ===================================================================
# Fix #2: EMA blend guard
# ===================================================================

class TestEMABlendGuard:
    """_append_active_prototype should skip EMA blend when new feature
    is wildly different from stored representative feature."""

    @pytest.mark.unit
    def test_ema_guard_constructor_param(self, embedder):
        """TrackGallery should accept ema_min_similarity parameter."""
        from reid.gallery import TrackGallery
        g = TrackGallery(embedder=embedder, ema_min_similarity=0.6)
        assert g._ema_min_similarity == 0.6

    @pytest.mark.unit
    def test_blend_skipped_for_wildly_different_feature(self, gallery, embedder):
        """If new feature is very different, active feature should not change."""
        feat_a = _unit_vec(512, 1.0)
        feat_b = _unit_vec(512, 999.0)  # very different

        # Establish track
        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(100)},
            precomputed_embeddings={1: feat_a},
        )
        stored_before = gallery._active[1].copy()

        # Update with wildly different feature
        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(200)},
            precomputed_embeddings={1: feat_b},
        )
        stored_after = gallery._active[1]

        # Should be very close to original (blend skipped)
        sim = float(np.dot(stored_before, stored_after))
        assert sim > 0.90, (
            f"EMA should skip blend for wildly different feature, "
            f"but similarity to original={sim:.3f}"
        )

    @pytest.mark.unit
    def test_blend_proceeds_for_similar_feature(self, gallery, embedder):
        """If new feature is similar enough, EMA should blend normally."""
        feat_a = _unit_vec(512, 1.0)
        feat_similar = feat_a + np.random.RandomState(42).randn(512).astype(np.float32) * 0.01
        feat_similar = feat_similar / np.linalg.norm(feat_similar)

        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(100)},
            precomputed_embeddings={1: feat_a},
        )
        stored_before = gallery._active[1].copy()

        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(100)},
            precomputed_embeddings={1: feat_similar},
        )
        stored_after = gallery._active[1]

        diff = np.linalg.norm(stored_before - stored_after)
        assert diff > 1e-6, "EMA should blend similar features"

    @pytest.mark.unit
    def test_blend_guard_does_not_affect_first_frame(self, gallery, embedder):
        """First time a track appears, any feature should be accepted."""
        feat = _unit_vec(512, 1.0)
        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(100)},
            precomputed_embeddings={1: feat},
        )
        assert 1 in gallery._active


# ===================================================================
# Fix #4: get_feature() public API
# ===================================================================

class TestGetFeatureAPI:
    """TrackGallery should expose a get_feature() method instead of
    requiring callers to access _active directly."""

    @pytest.mark.unit
    def test_get_feature_method_exists(self, gallery):
        assert hasattr(gallery, "get_feature")
        assert callable(gallery.get_feature)

    @pytest.mark.unit
    def test_get_feature_returns_active(self, gallery, embedder):
        feat = _unit_vec(512, 1.0)
        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(100)},
            precomputed_embeddings={1: feat},
        )
        result = gallery.get_feature(1)
        assert result is not None
        sim = float(np.dot(result / np.linalg.norm(result), feat))
        assert sim > 0.95

    @pytest.mark.unit
    def test_get_feature_returns_lost(self, gallery, embedder):
        feat = _unit_vec(512, 1.0)
        gallery.update(
            confirmed_ids={1},
            crops_by_tid={1: _make_crop(100)},
            precomputed_embeddings={1: feat},
        )
        # Track disappears → goes to lost
        gallery.update(confirmed_ids=set(), crops_by_tid={})
        assert 1 not in gallery._active
        result = gallery.get_feature(1)
        assert result is not None, "get_feature should find track in lost gallery"

    @pytest.mark.unit
    def test_get_feature_returns_none_for_unknown(self, gallery):
        assert gallery.get_feature(999) is None


# ===================================================================
# Fix #7: is_synthetic_consistent()
# ===================================================================

class TestSyntheticConsistent:
    """TrackAnchor should provide is_synthetic_consistent() to verify
    feature consistency for synthetic associations."""

    @pytest.mark.unit
    def test_method_exists(self, anchor):
        assert hasattr(anchor, "is_synthetic_consistent")

    @pytest.mark.unit
    def test_consistent_features_accepted(self):
        a = TrackAnchor()
        feat = _unit_vec(512, 1.0)
        assert a.is_synthetic_consistent(
            track_feat=feat,
            gallery_feat=feat,
            threshold=0.3,
        ) is True

    @pytest.mark.unit
    def test_inconsistent_features_rejected(self):
        a = TrackAnchor()
        feat_a = _unit_vec(512, 1.0)
        feat_b = _unit_vec(512, 999.0)
        assert a.is_synthetic_consistent(
            track_feat=feat_a,
            gallery_feat=feat_b,
            threshold=0.3,
        ) is False

    @pytest.mark.unit
    def test_no_gallery_feat_accepted(self):
        a = TrackAnchor()
        feat = _unit_vec(512, 1.0)
        assert a.is_synthetic_consistent(
            track_feat=feat,
            gallery_feat=None,
            threshold=0.3,
        ) is True
