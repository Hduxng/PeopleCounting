"""CPU-friendly regression tests for gallery stability logic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reid.gallery import TrackGallery


class _UnusedEmbedder:
    def __call__(self, crops):
        raise AssertionError('precomputed embeddings should be used in this test')


def _norm(vec):
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _crop():
    return np.zeros((128, 64, 3), dtype=np.uint8)


def test_short_gap_spatial_recovery_relaxes_noisy_reid_match():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.05,
        relaxed_match_threshold=0.25,
        spatial_match_window=5,
    )
    old_feat = _norm([1.0, 0.0, 0.0])
    noisy_feat = _norm([0.96, 0.28, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: old_feat},
        bboxes_by_tid={1: np.array([100, 100, 160, 240], dtype=float)},
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)

    remap = gallery.update(
        {99},
        {99: _crop()},
        precomputed_embeddings={99: noisy_feat},
        bboxes_by_tid={99: np.array([102, 101, 161, 242], dtype=float)},
        frame_idx=2,
    )

    assert remap == {99: 1}


def test_identity_drift_requires_confirmation_frames():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.3,
        drift_threshold=0.10,
        drift_confirm_frames=2,
    )
    stable_feat = _norm([1.0, 0.0, 0.0])
    drift_feat = _norm([0.0, 1.0, 0.0])
    bbox = np.array([100, 100, 160, 240], dtype=float)

    gallery.update({1}, {1: _crop()}, precomputed_embeddings={1: stable_feat}, bboxes_by_tid={1: bbox}, frame_idx=0)
    gallery.update({1}, {1: _crop()}, precomputed_embeddings={1: drift_feat}, bboxes_by_tid={1: bbox}, frame_idx=1)

    assert 1 not in gallery._lost
    np.testing.assert_allclose(gallery._active[1], stable_feat)

    gallery.update({1}, {1: _crop()}, precomputed_embeddings={1: drift_feat}, bboxes_by_tid={1: bbox}, frame_idx=2)

    assert 1 in gallery._lost


def test_merge_frame_does_not_trigger_identity_drift():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.3,
        drift_threshold=0.10,
        drift_confirm_frames=1,
        merge_area_ratio=1.5,
    )
    stable_feat = _norm([1.0, 0.0, 0.0])
    merged_feat = _norm([0.0, 1.0, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: stable_feat},
        bboxes_by_tid={1: np.array([100, 100, 150, 250], dtype=float)},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: merged_feat},
        bboxes_by_tid={1: np.array([90, 90, 170, 280], dtype=float)},
        frame_idx=1,
    )

    assert 1 in gallery._merged_ids
    assert 1 not in gallery._lost



def test_no_recovery_when_region_is_already_occupied_by_active_track():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        relaxed_match_threshold=0.45,
        spatial_match_window=5,
    )
    old_feat = _norm([1.0, 0.0, 0.0])
    active_feat = _norm([0.0, 1.0, 0.0])
    new_feat = _norm([0.98, 0.02, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: old_feat},
        bboxes_by_tid={1: np.array([100, 100, 160, 240], dtype=float)},
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)

    remap = gallery.update(
        {2, 99},
        {2: _crop(), 99: _crop()},
        precomputed_embeddings={2: active_feat, 99: new_feat},
        bboxes_by_tid={
            2: np.array([102, 101, 161, 242], dtype=float),
            99: np.array([103, 102, 162, 244], dtype=float),
        },
        frame_idx=2,
    )

    assert remap == {}



def test_tiny_nearby_active_track_does_not_block_recovery():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        relaxed_match_threshold=0.45,
        spatial_match_window=5,
    )
    old_feat = _norm([1.0, 0.0, 0.0])
    active_feat = _norm([0.0, 1.0, 0.0])
    new_feat = _norm([0.98, 0.02, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: old_feat},
        bboxes_by_tid={1: np.array([100, 100, 160, 240], dtype=float)},
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)

    remap = gallery.update(
        {2, 99},
        {2: _crop(), 99: _crop()},
        precomputed_embeddings={2: active_feat, 99: new_feat},
        bboxes_by_tid={
            2: np.array([108, 108, 124, 144], dtype=float),
            99: np.array([102, 101, 161, 242], dtype=float),
        },
        frame_idx=2,
    )

    assert remap == {99: 1}


def test_recovery_prefers_strict_appearance_match_over_relaxed_spatial_match():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.10,
        relaxed_match_threshold=0.35,
        spatial_match_window=5,
    )
    exact_feat = _norm([1.0, 0.0, 0.0])
    relaxed_feat = _norm([0.8, 0.6, 0.0])

    gallery.update(
        {1, 2},
        {1: _crop(), 2: _crop()},
        precomputed_embeddings={1: exact_feat, 2: relaxed_feat},
        bboxes_by_tid={
            1: np.array([10, 10, 70, 150], dtype=float),
            2: np.array([100, 100, 160, 240], dtype=float),
        },
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)

    remap = gallery.update(
        {99},
        {99: _crop()},
        precomputed_embeddings={99: exact_feat},
        bboxes_by_tid={99: np.array([102, 101, 161, 242], dtype=float)},
        frame_idx=2,
    )

    assert remap == {99: 1}


def test_global_assignment_recovers_two_ids_instead_of_greedy_steal():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.35,
        spatial_match_window=0,
    )
    old_a = _norm([1.0, 0.0, 0.0])
    old_b = _norm([0.66, 0.751265, 0.0])
    new_b_like = _norm([0.98, -0.199, 0.0])

    gallery.update(
        {1, 2},
        {1: _crop(), 2: _crop()},
        precomputed_embeddings={1: old_a, 2: old_b},
        bboxes_by_tid={
            1: np.array([10, 10, 70, 150], dtype=float),
            2: np.array([200, 10, 260, 150], dtype=float),
        },
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)

    remap = gallery.update(
        {99, 100},
        {99: _crop(), 100: _crop()},
        precomputed_embeddings={99: old_a, 100: new_b_like},
        bboxes_by_tid={
            99: np.array([202, 10, 262, 150], dtype=float),
            100: np.array([12, 10, 72, 150], dtype=float),
        },
        frame_idx=2,
    )

    assert remap == {99: 2, 100: 1}


def test_long_gap_appearance_match_rejects_tiny_far_lookalike():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=300,
        match_threshold=0.35,
        spatial_match_window=5,
    )
    old_feat = _norm([1.0, 0.0, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: old_feat},
        bboxes_by_tid={1: np.array([395, 180, 471, 314], dtype=float)},
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)
    gallery._lost[1][1] = 200

    remap = gallery.update(
        {99},
        {99: _crop()},
        precomputed_embeddings={99: old_feat},
        bboxes_by_tid={99: np.array([610, 124, 640, 168], dtype=float)},
        frame_idx=202,
    )

    assert remap == {}


def test_no_recovery_when_old_region_is_still_occupied_after_split():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=300,
        match_threshold=0.35,
        spatial_match_window=5,
    )
    old_feat = _norm([1.0, 0.0, 0.0])
    active_feat = _norm([0.0, 1.0, 0.0])

    gallery._lost[20] = [old_feat, 200, np.array([391.2, 188.3, 464.3, 313.6], dtype=float)]

    remap = gallery.update(
        {30, 99},
        {30: _crop(), 99: _crop()},
        precomputed_embeddings={30: active_feat, 99: old_feat},
        bboxes_by_tid={
            30: np.array([406.4, 178.1, 459.6, 255.8], dtype=float),
            99: np.array([610.4, 123.6, 640.0, 168.2], dtype=float),
        },
        frame_idx=4462,
    )

    assert remap == {}
