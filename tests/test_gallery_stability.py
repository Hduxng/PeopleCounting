"""CPU-friendly regression tests for gallery stability logic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reid.gallery import TrackGallery


class _UnusedEmbedder:
    def __call__(self, crops):
        raise AssertionError('precomputed embeddings should be used in this test')


class _DebugCapture:
    def __init__(self):
        self.skip_events = []
        self.bank_events = []
        self.merge_events = []
        self.drift_events = []

    def log_gallery_skip(self, frame_idx, tid, reason, confidence=None, crop_area=None):
        self.skip_events.append({
            "frame_idx": frame_idx,
            "tid": tid,
            "reason": reason,
            "confidence": confidence,
            "crop_area": crop_area,
        })

    def log_gallery_prototypes(self, frame_idx, tid, count, source="update"):
        self.bank_events.append({
            "frame_idx": frame_idx,
            "tid": tid,
            "count": count,
            "source": source,
        })

    def log_merge(self, *args, **kwargs):
        self.merge_events.append((args, kwargs))

    def log_split(self, *args, **kwargs):
        pass

    def log_drift(self, *args, **kwargs):
        self.drift_events.append((args, kwargs))


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


def test_short_gap_spatial_recovery_accepts_large_contained_box():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.05,
        relaxed_match_threshold=0.25,
        spatial_match_window=5,
    )
    old_feat = _norm([1.0, 0.0, 0.0])
    noisy_feat = _norm([0.90, 0.44, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: old_feat},
        bboxes_by_tid={1: np.array([100, 100, 150, 240], dtype=float)},
        frame_idx=0,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=1)

    remap = gallery.update(
        {99},
        {99: _crop()},
        precomputed_embeddings={99: noisy_feat},
        bboxes_by_tid={99: np.array([85, 85, 175, 285], dtype=float)},
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


def test_merge_state_releases_when_area_settles_below_release_ratio():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.3,
        merge_area_ratio=1.5,
        merge_release_ratio=1.35,
    )
    feat = _norm([1.0, 0.0, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: np.array([100, 100, 150, 250], dtype=float)},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: np.array([92, 92, 172, 280], dtype=float)},
        frame_idx=1,
    )
    assert 1 in gallery._merged_ids

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: np.array([97, 97, 153, 213], dtype=float)},
        frame_idx=2,
    )

    assert 1 not in gallery._merged_ids


def test_merge_state_times_out_after_hold_budget():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.3,
        merge_area_ratio=1.5,
        merge_hold_frames=3,
    )
    feat = _norm([1.0, 0.0, 0.0])
    merged_bbox = np.array([90, 90, 170, 280], dtype=float)

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: np.array([100, 100, 150, 250], dtype=float)},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: merged_bbox},
        frame_idx=1,
    )
    assert 1 in gallery._merged_ids

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: merged_bbox},
        frame_idx=2,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: merged_bbox},
        frame_idx=3,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: merged_bbox},
        frame_idx=4,
    )

    assert 1 not in gallery._merged_ids



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


def test_prototype_bank_recovers_view_not_represented_by_centroid_alone():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.15,
        drift_threshold=1.1,
        spatial_match_window=0,
        max_prototypes=2,
    )
    feat_front = _norm([1.0, 0.0, 0.0])
    feat_side = _norm([0.0, 1.0, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat_front},
        bboxes_by_tid={1: np.array([100, 100, 160, 240], dtype=float)},
        update_confidences={1: 0.95},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat_side},
        bboxes_by_tid={1: np.array([101, 101, 161, 241], dtype=float)},
        update_confidences={1: 0.95},
        frame_idx=1,
    )
    gallery.update(set(), {}, precomputed_embeddings={}, bboxes_by_tid={}, frame_idx=2)

    remap = gallery.update(
        {99},
        {99: _crop()},
        precomputed_embeddings={99: feat_side},
        bboxes_by_tid={99: np.array([102, 102, 162, 242], dtype=float)},
        update_confidences={99: 0.95},
        frame_idx=3,
    )

    assert remap == {99: 1}


def test_low_confidence_embedding_does_not_update_prototype_bank():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        min_update_conf=0.80,
        max_prototypes=4,
    )
    feat = _norm([1.0, 0.0, 0.0])
    bbox = np.array([100, 100, 160, 240], dtype=float)

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: bbox},
        update_confidences={1: 0.40},
        frame_idx=0,
    )
    assert 1 not in gallery._active
    assert 1 not in gallery._active_banks

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: bbox},
        update_confidences={1: 0.95},
        frame_idx=1,
    )
    assert 1 in gallery._active
    assert len(gallery._active_banks[1]) == 1


def test_small_crop_does_not_update_gallery_even_with_precomputed_embedding():
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        min_crop_area=500,
    )
    feat = _norm([1.0, 0.0, 0.0])
    small_crop = np.zeros((16, 16, 3), dtype=np.uint8)

    gallery.update(
        {1},
        {1: small_crop},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: np.array([100, 100, 116, 116], dtype=float)},
        update_confidences={1: 0.95},
        frame_idx=0,
    )

    assert 1 not in gallery._active
    assert 1 not in gallery._active_banks


def test_gallery_debug_logs_low_conf_skip_and_bank_count():
    debug = _DebugCapture()
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        min_update_conf=0.80,
        max_prototypes=4,
        debug_logger=debug,
    )
    feat = _norm([1.0, 0.0, 0.0])
    bbox = np.array([100, 100, 160, 240], dtype=float)

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: bbox},
        update_confidences={1: 0.40},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: bbox},
        update_confidences={1: 0.95},
        frame_idx=1,
    )

    assert debug.skip_events == [{
        "frame_idx": 0,
        "tid": 1,
        "reason": "low_conf",
        "confidence": 0.40,
        "crop_area": 8192,
    }]
    assert debug.bank_events == [{
        "frame_idx": 1,
        "tid": 1,
        "count": 1,
        "source": "update",
    }]


def test_gallery_debug_logs_small_crop_skip():
    debug = _DebugCapture()
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        min_crop_area=500,
        debug_logger=debug,
    )
    feat = _norm([1.0, 0.0, 0.0])
    small_crop = np.zeros((16, 16, 3), dtype=np.uint8)

    gallery.update(
        {1},
        {1: small_crop},
        precomputed_embeddings={1: feat},
        bboxes_by_tid={1: np.array([100, 100, 116, 116], dtype=float)},
        update_confidences={1: 0.95},
        frame_idx=0,
    )

    assert debug.skip_events == [{
        "frame_idx": 0,
        "tid": 1,
        "reason": "small_crop",
        "confidence": None,
        "crop_area": 256,
    }]


def test_gallery_debug_logs_merged_skip_reason():
    debug = _DebugCapture()
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        merge_area_ratio=1.5,
        debug_logger=debug,
    )
    stable_feat = _norm([1.0, 0.0, 0.0])
    merged_feat = _norm([0.0, 1.0, 0.0])

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: stable_feat},
        bboxes_by_tid={1: np.array([100, 100, 150, 250], dtype=float)},
        update_confidences={1: 0.95},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: merged_feat},
        bboxes_by_tid={1: np.array([90, 90, 170, 280], dtype=float)},
        update_confidences={1: 0.95},
        frame_idx=1,
    )

    assert any(event["reason"] == "merged" and event["tid"] == 1 for event in debug.skip_events)


def test_gallery_debug_logs_drift_hold_skip_reason():
    debug = _DebugCapture()
    gallery = TrackGallery(
        embedder=_UnusedEmbedder(),
        lifetime=30,
        match_threshold=0.30,
        drift_threshold=0.10,
        drift_confirm_frames=2,
        debug_logger=debug,
    )
    stable_feat = _norm([1.0, 0.0, 0.0])
    drift_feat = _norm([0.0, 1.0, 0.0])
    bbox = np.array([100, 100, 160, 240], dtype=float)

    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: stable_feat},
        bboxes_by_tid={1: bbox},
        update_confidences={1: 0.95},
        frame_idx=0,
    )
    gallery.update(
        {1},
        {1: _crop()},
        precomputed_embeddings={1: drift_feat},
        bboxes_by_tid={1: bbox},
        update_confidences={1: 0.95},
        frame_idx=1,
    )

    assert any(event["reason"] == "drift_hold" and event["tid"] == 1 for event in debug.skip_events)
