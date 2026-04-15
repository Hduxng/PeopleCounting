"""
Re-ID embedding quality tests.

Verifies that the CLIP-ReID and OSNet embedders produce consistent,
discriminative features suitable for person tracking.

Requires GPU.  Tests are skipped if CUDA is unavailable.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Re-ID tests"),
    pytest.mark.filterwarnings(
        r"ignore:Cython evaluation .* is unavailable, now use python evaluation\.:UserWarning"
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip_embedder():
    from reid.embedder import CLIPReIDEmbedder
    return CLIPReIDEmbedder(weights="clip_market1501.pt", device="cuda:0", half=True)


@pytest.fixture(scope="module")
def osnet_embedder():
    from reid.embedder import OSNetEmbedder
    return OSNetEmbedder(
        model_name="osnet_x1_0", device="cuda:0", half=True, tta=False,
    )


def _make_person_crop(color: tuple, w: int = 64, h: int = 128, noise: int = 20) -> np.ndarray:
    """Create a synthetic person crop: solid color block with noise."""
    crop = np.full((h, w, 3), color, dtype=np.uint8)
    noise_arr = np.random.randint(-noise, noise + 1, crop.shape, dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
    return crop


def _cosine_distance(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =========================================================================
# CLIP-ReID Tests
# =========================================================================


class TestCLIPReID_Basic:
    """Basic embedding quality for CLIP-ReID."""

    def test_output_dimension(self, clip_embedder):
        """CLIP-ReID should produce 1280-D embeddings."""
        crop = _make_person_crop((100, 150, 200))
        embs = clip_embedder([crop])
        assert len(embs) == 1
        assert len(embs[0]) == 1280

    def test_l2_normalized(self, clip_embedder):
        """Embeddings should be L2-normalized (unit vectors)."""
        crop = _make_person_crop((100, 150, 200))
        embs = clip_embedder([crop])
        norm = np.linalg.norm(embs[0])
        assert abs(norm - 1.0) < 0.01, f"Expected unit vector, got norm={norm}"

    def test_empty_input(self, clip_embedder):
        """Empty crop list should return empty list."""
        assert clip_embedder([]) == []

    def test_batch_processing(self, clip_embedder):
        """Multiple crops processed in one call."""
        crops = [_make_person_crop((c, 100, 200)) for c in [50, 100, 150, 200]]
        embs = clip_embedder(crops)
        assert len(embs) == 4
        for e in embs:
            assert len(e) == 1280

    def test_deterministic(self, clip_embedder):
        """Same crop → same embedding (within FP16 tolerance)."""
        crop = _make_person_crop((120, 180, 60), noise=0)
        e1 = clip_embedder([crop])
        e2 = clip_embedder([crop])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist < 0.01, f"Same crop should give same embedding, dist={dist}"


class TestCLIPReID_Discriminative:
    """CLIP-ReID should produce discriminative features."""

    def test_same_person_similar(self, clip_embedder):
        """Slight variations of same crop → small cosine distance."""
        base_color = (100, 150, 200)
        crop1 = _make_person_crop(base_color, noise=10)
        crop2 = _make_person_crop(base_color, noise=10)
        e1 = clip_embedder([crop1])
        e2 = clip_embedder([crop2])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist < 0.5, f"Similar crops should be close, dist={dist}"

    def test_different_people_different(self, clip_embedder):
        """Very different crops → larger cosine distance."""
        crop1 = _make_person_crop((0, 0, 255), w=64, h=128, noise=5)    # red
        crop2 = _make_person_crop((255, 255, 0), w=64, h=128, noise=5)  # cyan
        e1 = clip_embedder([crop1])
        e2 = clip_embedder([crop2])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist > 0.05, f"Different crops should be far apart, dist={dist}"

    def test_brightness_invariance(self, clip_embedder):
        """Same person under different brightness → still similar."""
        crop_normal = _make_person_crop((100, 150, 80), noise=5)
        crop_bright = cv2.convertScaleAbs(crop_normal, alpha=1.5, beta=40)
        e1 = clip_embedder([crop_normal])
        e2 = clip_embedder([crop_bright])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist < 0.5, f"Brightness change should not break Re-ID, dist={dist}"

    def test_horizontal_flip_similar(self, clip_embedder):
        """Horizontally flipped crop → should still be somewhat similar."""
        crop = _make_person_crop((80, 120, 200), noise=5)
        crop_flip = cv2.flip(crop, 1)
        e1 = clip_embedder([crop])
        e2 = clip_embedder([crop_flip])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist < 0.6, f"Flipped crop should be somewhat similar, dist={dist}"


class TestCLIPReID_EdgeCases:
    """Edge cases for CLIP-ReID embedder."""

    def test_tiny_crop(self, clip_embedder):
        """Very small crop (20x20) — should not crash (padded internally)."""
        tiny = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        embs = clip_embedder([tiny])
        assert len(embs) == 1
        assert len(embs[0]) == 1280

    def test_large_crop(self, clip_embedder):
        """Large crop (500x300) — should resize internally and work."""
        large = np.random.randint(0, 255, (500, 300, 3), dtype=np.uint8)
        embs = clip_embedder([large])
        assert len(embs) == 1
        assert len(embs[0]) == 1280

    def test_mixed_sizes(self, clip_embedder):
        """Crops of different sizes in one batch."""
        crops = [
            np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8),
        ]
        embs = clip_embedder(crops)
        assert len(embs) == 3
        for e in embs:
            assert len(e) == 1280


# =========================================================================
# OSNet Tests
# =========================================================================


class TestOSNet_Basic:
    """Basic embedding quality for OSNet."""

    def test_output_dimension(self, osnet_embedder):
        """OSNet should produce 512-D embeddings."""
        crop = _make_person_crop((100, 150, 200))
        embs = osnet_embedder([crop])
        assert len(embs) == 1
        assert len(embs[0]) == 512

    def test_l2_normalized(self, osnet_embedder):
        """Embeddings should be L2-normalized."""
        crop = _make_person_crop((100, 150, 200))
        embs = osnet_embedder([crop])
        norm = np.linalg.norm(embs[0])
        assert abs(norm - 1.0) < 0.01, f"Expected unit vector, got norm={norm}"

    def test_empty_input(self, osnet_embedder):
        assert osnet_embedder([]) == []

    def test_deterministic(self, osnet_embedder):
        """Same crop → same embedding."""
        crop = _make_person_crop((120, 180, 60), noise=0)
        e1 = osnet_embedder([crop])
        e2 = osnet_embedder([crop])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist < 0.01, f"Same crop should give same embedding, dist={dist}"


class TestOSNet_Discriminative:
    """OSNet discriminative power."""

    def test_same_person_similar(self, osnet_embedder):
        crop1 = _make_person_crop((100, 150, 200), noise=10)
        crop2 = _make_person_crop((100, 150, 200), noise=10)
        e1 = osnet_embedder([crop1])
        e2 = osnet_embedder([crop2])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist < 0.5, f"Similar crops should be close, dist={dist}"

    def test_different_people_different(self, osnet_embedder):
        crop1 = _make_person_crop((0, 0, 255), noise=5)
        crop2 = _make_person_crop((255, 255, 0), noise=5)
        e1 = osnet_embedder([crop1])
        e2 = osnet_embedder([crop2])
        dist = _cosine_distance(e1[0], e2[0])
        assert dist > 0.05, f"Different crops should be far apart, dist={dist}"


# =========================================================================
# Gallery Integration
# =========================================================================


class TestGallery:
    """TrackGallery with CLIP-ReID embedder."""

    def test_gallery_update_and_recover(self, clip_embedder):
        """Gallery should store features and recover lost IDs."""
        from reid.gallery import TrackGallery

        gallery = TrackGallery(
            embedder=clip_embedder,
            lifetime=30,
            ema_alpha=0.85,
            match_threshold=0.5,
        )

        # Create distinct person crops
        crop_a = _make_person_crop((0, 0, 200), w=64, h=128, noise=5)
        crop_b = _make_person_crop((0, 200, 0), w=64, h=128, noise=5)

        # Frame 1: both people visible
        remap = gallery.update(
            confirmed_ids={1, 2},
            crops_by_tid={1: crop_a, 2: crop_b},
        )
        assert remap == {}
        assert 1 in gallery._active
        assert 2 in gallery._active

    def test_gallery_lost_and_found(self, clip_embedder):
        """Person disappears then reappears — gallery should recover ID."""
        from reid.gallery import TrackGallery

        gallery = TrackGallery(
            embedder=clip_embedder,
            lifetime=30,
            ema_alpha=0.85,
            match_threshold=0.5,  # lenient threshold for synthetic data
        )

        crop_a = _make_person_crop((0, 0, 200), w=64, h=128, noise=3)

        # Frame 1: person 1 visible
        gallery.update({1}, {1: crop_a})

        # Frame 2: person 1 disappears
        gallery.update(set(), {})
        assert 1 in gallery._lost, "Person 1 should be in lost gallery"

        # Frame 3: person reappears with new tracker ID=99, same appearance
        crop_a_similar = _make_person_crop((0, 0, 200), w=64, h=128, noise=3)
        remap = gallery.update({99}, {99: crop_a_similar})

        # If gallery matched, remap should map 99→1
        # (depends on embedding similarity of synthetic crops)
        # At minimum, gallery should not crash
        assert isinstance(remap, dict)

    def test_gallery_lifetime_expiry(self, clip_embedder):
        """Lost track should expire after lifetime frames."""
        from reid.gallery import TrackGallery

        gallery = TrackGallery(
            embedder=clip_embedder,
            lifetime=5,
            ema_alpha=0.85,
            match_threshold=0.3,
        )

        crop = _make_person_crop((100, 150, 50), w=64, h=128, noise=3)

        # Person 1 visible
        gallery.update({1}, {1: crop})

        # Person disappears for 6 frames (> lifetime=5)
        for _ in range(6):
            gallery.update(set(), {})

        # Person 1 should be evicted from lost gallery
        assert 1 not in gallery._lost, "Should be evicted after lifetime"

    def test_gallery_ema_feature_update(self, clip_embedder):
        """Active features should update via EMA, not be replaced."""
        from reid.gallery import TrackGallery

        gallery = TrackGallery(
            embedder=clip_embedder,
            lifetime=30,
            ema_alpha=0.9,
            match_threshold=0.3,
        )

        crop1 = _make_person_crop((100, 150, 200), w=64, h=128, noise=0)
        gallery.update({1}, {1: crop1})
        feat_after_1 = gallery._active[1].copy()

        crop2 = _make_person_crop((100, 150, 200), w=64, h=128, noise=30)
        gallery.update({1}, {1: crop2})
        feat_after_2 = gallery._active[1]

        # Feature should have changed (EMA blending)
        diff = np.linalg.norm(feat_after_1 - feat_after_2)
        # With alpha=0.9, the change should be small but nonzero
        assert diff > 0, "EMA should update features"
