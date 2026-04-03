"""Unit tests for duplicate-track stabilization in the nwojke path."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from main import _LiveTrackAliasResolver


class _FakeTrack:
    def __init__(self, track_id, bbox, hits, age, time_since_update=0):
        self.track_id = track_id
        self._bbox = np.asarray(bbox, dtype=float)
        self.hits = hits
        self.age = age
        self.time_since_update = time_since_update

    def to_tlbr(self):
        return self._bbox.copy()


def test_aliases_parallel_duplicate_track_to_older_id():
    resolver = _LiveTrackAliasResolver()
    stable = _FakeTrack(3, [352.9, 151.6, 444.2, 265.7], hits=80, age=120)
    duplicate = _FakeTrack(16, [352.3, 151.0, 445.7, 324.7], hits=4, age=4)

    new_aliases = resolver.alias_duplicates([duplicate, stable])

    assert new_aliases == {16: 3}
    assert resolver.resolve(16) == 3


def test_alias_persists_when_only_duplicate_track_remains():
    resolver = _LiveTrackAliasResolver()
    stable = _FakeTrack(3, [352.9, 151.6, 444.2, 265.7], hits=80, age=120)
    duplicate = _FakeTrack(16, [352.3, 151.0, 445.7, 324.7], hits=4, age=4)
    resolver.alias_duplicates([stable, duplicate])

    assert resolver.alias_duplicates([_FakeTrack(16, [351.9, 147.1, 437.6, 305.6], hits=40, age=40)]) == {}
    assert resolver.resolve(16) == 3


def test_distinct_people_are_not_aliased():
    resolver = _LiveTrackAliasResolver()
    left = _FakeTrack(3, [340.0, 150.0, 430.0, 310.0], hits=80, age=120)
    right = _FakeTrack(16, [430.0, 160.0, 520.0, 320.0], hits=5, age=5)

    assert resolver.alias_duplicates([left, right]) == {}
    assert resolver.resolve(16) == 16


def test_tracker_skips_low_confidence_new_tracks():
    metric = NearestNeighborDistanceMetric("cosine", 0.3, 100)
    tracker = Tracker(metric, n_init=1, new_track_thresh=0.5)

    tracker.predict()
    tracker.update([Detection([10, 10, 20, 40], 0.40, np.array([1.0, 0.0], dtype=np.float32))])

    assert tracker.tracks == []


def test_low_confidence_detection_can_still_update_existing_track():
    metric = NearestNeighborDistanceMetric("cosine", 0.3, 100)
    tracker = Tracker(metric, n_init=1, new_track_thresh=0.5)
    feat = np.array([1.0, 0.0], dtype=np.float32)

    tracker.predict()
    tracker.update([Detection([10, 10, 20, 40], 0.95, feat)])
    assert len(tracker.tracks) == 1
    first_tid = tracker.tracks[0].track_id

    tracker.predict()
    tracker.update([Detection([11, 11, 20, 40], 0.30, feat)])

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == first_tid


def test_live_duplicate_keeps_older_stable_id_even_if_new_box_is_fresh():
    resolver = _LiveTrackAliasResolver()
    stable = _FakeTrack(55, [352.9, 151.6, 444.2, 265.7], hits=220, age=800, time_since_update=1)
    fresh_duplicate = _FakeTrack(108, [352.3, 151.0, 445.7, 324.7], hits=8, age=12, time_since_update=0)

    new_aliases = resolver.alias_duplicates([stable, fresh_duplicate])

    assert new_aliases == {108: 55}
    assert resolver.resolve(108) == 55


def test_protected_gallery_id_cannot_be_aliased_back_to_new_track():
    resolver = _LiveTrackAliasResolver()
    restored_old = _FakeTrack(55, [352.9, 151.6, 444.2, 265.7], hits=220, age=800, time_since_update=1)
    fresh_duplicate = _FakeTrack(121, [352.3, 151.0, 445.7, 324.7], hits=3, age=3, time_since_update=0)

    new_aliases = resolver.alias_duplicates([restored_old, fresh_duplicate], protected_ids={55})

    assert new_aliases == {121: 55}
    assert resolver.resolve(121) == 55



def test_duplicate_against_already_aliased_track_maps_to_root_canonical():
    resolver = _LiveTrackAliasResolver()
    resolver._alias[121] = 55
    aliased_track = _FakeTrack(121, [352.9, 151.6, 444.2, 265.7], hits=220, age=800, time_since_update=0)
    fresh_duplicate = _FakeTrack(130, [352.3, 151.0, 445.7, 264.9], hits=6, age=8, time_since_update=0)

    new_aliases = resolver.alias_duplicates([aliased_track, fresh_duplicate])

    assert new_aliases == {130: 55}
    assert resolver.resolve(130) == 55


def test_duplicate_component_collapses_transitive_chain_to_one_canonical():
    resolver = _LiveTrackAliasResolver()
    stable = _FakeTrack(55, [100.0, 100.0, 200.0, 260.0], hits=220, age=800, time_since_update=1)
    bridge = _FakeTrack(121, [115.0, 100.0, 215.0, 260.0], hits=12, age=18, time_since_update=0)
    far_duplicate = _FakeTrack(130, [130.0, 100.0, 230.0, 260.0], hits=5, age=6, time_since_update=0)

    new_aliases = resolver.alias_duplicates([far_duplicate, bridge, stable])

    assert new_aliases == {121: 55, 130: 55}
    assert resolver.resolve(121) == 55
    assert resolver.resolve(130) == 55


def test_gallery_remap_alias_persists_across_future_frames():
    resolver = _LiveTrackAliasResolver()

    assert resolver.remember_alias(13, 12) == 12
    assert resolver.resolve(13) == 12

    assert resolver.remember_alias(42, 13) == 12
    assert resolver.resolve(42) == 12
