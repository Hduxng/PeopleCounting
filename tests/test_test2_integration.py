"""Artifact-backed integration tests for real `test2.mp4` debug runs.

These tests exercise the end-to-end pipeline through saved debug artifacts
instead of synthetic tracker fixtures. They pin down the exact failure modes
reported from `test2.mp4`:

* overlapping / merged boxes that later churn IDs and trigger repeated counts
* cross-person jumps where a canonical track switches to a far-away raw track

The artifacts live under ``debug/<timestamp>/`` and are produced by `main.py`
with ``runtime.debug: true``.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
BUG_BASELINE_RUN = "20260402_145447"
REFERENCE_RUN = "20260403_135114"


@dataclass(frozen=True)
class RemapEvent:
    frame: int
    new_tid: int
    old_tid: int


@dataclass(frozen=True)
class CountEvent:
    frame: int
    tid: int
    counter: str
    event: str


@dataclass(frozen=True)
class RawSwitch:
    track_id: int
    frame: int
    prev_raw: int | None
    new_raw: int | None
    distance_px: float


class _Test2ArtifactRun:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.base = ROOT / "debug" / run_id
        if not self.base.exists():
            pytest.skip(f"missing debug artifact: {self.base}")
        for name in ("config.log", "events.log", "tracks.csv"):
            if not (self.base / name).exists():
                pytest.skip(f"incomplete debug artifact: {(self.base / name)}")
        if "source: test2.mp4" not in self.config_log:
            pytest.skip(f"{self.base} is not a debug run for test2.mp4")

    @cached_property
    def config_log(self) -> str:
        return (self.base / "config.log").read_text(encoding="utf-8", errors="ignore")

    @cached_property
    def remaps(self) -> list[RemapEvent]:
        events: list[RemapEvent] = []
        for line in (self.base / "events.log").read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.search(r"REMAP\s+frame=(\d+)\s+(\d+) -> (\d+)", line)
            if not match:
                continue
            frame, new_tid, old_tid = map(int, match.groups())
            events.append(RemapEvent(frame=frame, new_tid=new_tid, old_tid=old_tid))
        return events

    @cached_property
    def counts(self) -> list[CountEvent]:
        events: list[CountEvent] = []
        for line in (self.base / "events.log").read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.search(
                r"COUNT\s+frame=(\d+)\s+tid=(\d+)\s+counter=(.*?)\s+event=(\w+)",
                line,
            )
            if not match:
                continue
            events.append(
                CountEvent(
                    frame=int(match.group(1)),
                    tid=int(match.group(2)),
                    counter=match.group(3),
                    event=match.group(4),
                )
            )
        return events

    @cached_property
    def track_rows(self) -> list[dict[str, str]]:
        with (self.base / "tracks.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def frame_rows(self, frame_idx: int) -> list[dict[str, str]]:
        return [row for row in self.track_rows if int(row["frame"]) == frame_idx]

    def frame_track(self, frame_idx: int, track_id: int) -> dict[str, str]:
        for row in self.frame_rows(frame_idx):
            if int(row["track_id"]) == track_id:
                return row
        raise KeyError((frame_idx, track_id))

    def same_frame_remap_cycles(self) -> set[tuple[int, int, int]]:
        by_frame: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for event in self.remaps:
            by_frame[event.frame].add((event.new_tid, event.old_tid))
        cycles: set[tuple[int, int, int]] = set()
        for frame, pairs in by_frame.items():
            for src, dst in pairs:
                if (dst, src) in pairs:
                    cycles.add((frame, src, dst))
        return cycles

    def same_frame_multi_destination_remaps(self) -> set[tuple[int, int, tuple[int, ...]]]:
        by_frame: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        for event in self.remaps:
            by_frame[event.frame][event.new_tid].add(event.old_tid)
        multi: set[tuple[int, int, tuple[int, ...]]] = set()
        for frame, mapping in by_frame.items():
            for new_tid, old_tids in mapping.items():
                if len(old_tids) > 1:
                    multi.add((frame, new_tid, tuple(sorted(old_tids))))
        return multi

    def duplicate_count_frames(self) -> dict[tuple[int, str, str], list[int]]:
        by_key: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        for event in self.counts:
            by_key[(event.tid, event.counter, event.event)].append(event.frame)
        return {key: frames for key, frames in by_key.items() if len(frames) > 1}

    def raw_switches(self, track_id: int) -> list[RawSwitch]:
        rows = [row for row in self.track_rows if int(row["track_id"]) == track_id]
        rows.sort(key=lambda row: int(row["frame"]))
        switches: list[RawSwitch] = []
        prev_row: dict[str, str] | None = None
        for row in rows:
            if prev_row is not None and row["raw_id"] != prev_row["raw_id"]:
                dist = math.hypot(
                    float(row["center_x"]) - float(prev_row["center_x"]),
                    float(row["center_y"]) - float(prev_row["center_y"]),
                )
                switches.append(
                    RawSwitch(
                        track_id=track_id,
                        frame=int(row["frame"]),
                        prev_raw=int(prev_row["raw_id"]) if prev_row["raw_id"] else None,
                        new_raw=int(row["raw_id"]) if row["raw_id"] else None,
                        distance_px=dist,
                    )
                )
            prev_row = row
        return switches


def _artifact(run_id: str) -> _Test2ArtifactRun:
    return _Test2ArtifactRun(run_id)


def test_test2_bug_baseline_contains_same_frame_cycle_and_multi_destination_remaps():
    run = _artifact(BUG_BASELINE_RUN)

    cycles = run.same_frame_remap_cycles()
    multi = run.same_frame_multi_destination_remaps()

    assert (9215, 121, 55) in cycles
    assert (9215, 55, 121) in cycles
    assert (4331, 57, (50, 55)) in multi
    assert (4814, 69, (43, 55)) in multi


def test_test2_bug_baseline_contains_duplicate_count_after_merged_box_id_churn():
    run = _artifact(BUG_BASELINE_RUN)

    duplicate_counts = run.duplicate_count_frames()
    assert duplicate_counts[(55, "Waiting Area", "zone_counted")] == [4348, 5157, 6248, 6851, 9002]
    assert duplicate_counts[(43, "Waiting Area", "zone_counted")] == [3626, 4722]

    merged_repeat = run.frame_track(5157, 55)
    assert merged_repeat["raw_id"] == "69"
    assert merged_repeat["remap_from"] == "69"
    assert merged_repeat["merged"] == "1"


def test_test2_bug_baseline_contains_cross_person_jump_for_canonical_tid_55():
    run = _artifact(BUG_BASELINE_RUN)
    switches = run.raw_switches(55)

    assert any(
        sw.frame == 6947
        and sw.prev_raw == 81
        and sw.new_raw == 71
        and sw.distance_px >= 90.0
        for sw in switches
    )
    assert any(
        sw.frame == 8738
        and sw.prev_raw == 102
        and sw.new_raw == 109
        and sw.distance_px >= 240.0
        for sw in switches
    )


def test_test2_reference_run_has_no_same_frame_cycle_or_multi_destination_remaps():
    run = _artifact(REFERENCE_RUN)

    assert run.same_frame_remap_cycles() == set()
    assert run.same_frame_multi_destination_remaps() == set()


def test_test2_reference_run_keeps_crowded_window_raw_switches_spatially_local():
    run = _artifact(REFERENCE_RUN)
    crowded_switches = [
        sw for sw in run.raw_switches(44) if 5080 <= sw.frame <= 5149
    ]

    assert crowded_switches
    assert max(sw.distance_px for sw in crowded_switches) < 30.0
