"""
Debug logger — writes per-frame, per-track state to CSV + summary log.

Usage:
    from debug import DebugLogger

    dbg = DebugLogger(enabled=True)       # logs to debug/<timestamp>/
    dbg.log_config(cfg, tracker_type)      # once at startup
    dbg.log_frame_start(frame_idx, timestamp, n_detections)
    dbg.log_track(...)                     # once per track per frame
    dbg.log_anchor_inject(frame_idx, tid, bbox)
    dbg.log_remap(frame_idx, new_tid, old_tid)
    dbg.log_merge(frame_idx, tid, area, prev_area)
    dbg.log_counter_event(frame_idx, tid, counter_name, event)
    dbg.log_lost(frame_idx, tid)
    dbg.close()

Output files (in debug/<timestamp>/):
    tracks.csv    — per-track per-frame: bbox, center, conf, zone/line side, etc.
    events.log    — human-readable event log (remaps, merges, anchor, counts, lost)
    config.log    — snapshot of active config at startup
"""

import csv
import io
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class DebugLogger:
    """Lightweight CSV + text logger for tracking diagnostics."""

    def __init__(self, enabled: bool = True, output_dir: str = "debug"):
        self._enabled = enabled
        if not enabled:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir = Path(output_dir) / ts
        self._dir.mkdir(parents=True, exist_ok=True)

        # --- tracks.csv ---
        self._csv_path = self._dir / "tracks.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "frame", "timestamp", "track_id", "raw_id",
            "x1", "y1", "x2", "y2",
            "center_x", "center_y",
            "width", "height", "area",
            "confidence", "is_synthetic",
            "line_side", "in_zone",
            "remap_from",
            "merged",
        ])

        # --- events.log ---
        self._log_path = self._dir / "events.log"
        self._log_file = open(self._log_path, "w")

        self._t0 = time.perf_counter()
        self._write_event("=== Debug session started ===")
        print(f"[debug] logging to {self._dir}/")

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def log_config(self, cfg: dict, tracker_type: str) -> None:
        if not self._enabled:
            return
        path = self._dir / "config.log"
        with open(path, "w") as f:
            import yaml
            f.write(f"tracker_type: {tracker_type}\n")
            f.write(f"logged_at: {datetime.now().isoformat()}\n")
            f.write("---\n")
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Per-frame markers
    # ------------------------------------------------------------------

    def log_frame_start(
        self,
        frame_idx: int,
        timestamp: float,
        n_detections: int,
        n_tracks: int,
    ) -> None:
        if not self._enabled:
            return
        self._write_event(
            f"[frame {frame_idx:06d}] t={timestamp:.3f}s  "
            f"dets={n_detections}  tracks={n_tracks}"
        )

    # ------------------------------------------------------------------
    # Per-track data (one row per track per frame)
    # ------------------------------------------------------------------

    def log_track(
        self,
        frame_idx: int,
        timestamp: float,
        track_id: int,
        raw_id: int,
        bbox: np.ndarray | list,
        center: tuple,
        confidence: float,
        is_synthetic: bool = False,
        line_side: int | None = None,
        in_zone: bool | None = None,
        remap_from: int | None = None,
        merged: bool = False,
    ) -> None:
        if not self._enabled:
            return
        bb = np.asarray(bbox, dtype=float)
        x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
        w = x2 - x1
        h = y2 - y1
        self._csv.writerow([
            frame_idx, f"{timestamp:.3f}", track_id, raw_id,
            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
            f"{center[0]:.1f}", f"{center[1]:.1f}",
            f"{w:.1f}", f"{h:.1f}", f"{w * h:.0f}",
            f"{confidence:.3f}", int(is_synthetic),
            line_side if line_side is not None else "",
            int(in_zone) if in_zone is not None else "",
            remap_from if remap_from is not None else "",
            int(merged),
        ])

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_anchor_inject(
        self,
        frame_idx: int,
        tid: int,
        bbox: np.ndarray | list,
        lost_frames: int,
    ) -> None:
        if not self._enabled:
            return
        bb = np.asarray(bbox, dtype=float)
        self._write_event(
            f"  ANCHOR  frame={frame_idx}  tid={tid}  "
            f"bbox=[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}]  "
            f"lost_frames={lost_frames}"
        )

    def log_remap(self, frame_idx: int, new_tid: int, old_tid: int) -> None:
        if not self._enabled:
            return
        self._write_event(
            f"  REMAP   frame={frame_idx}  {new_tid} -> {old_tid}  "
            f"(new track re-identified as old track)"
        )

    def log_merge(
        self,
        frame_idx: int,
        tid: int,
        area: float,
        prev_area: float,
    ) -> None:
        if not self._enabled:
            return
        ratio = area / prev_area if prev_area > 0 else 0
        self._write_event(
            f"  MERGE   frame={frame_idx}  tid={tid}  "
            f"area={area:.0f}  prev={prev_area:.0f}  ratio={ratio:.2f}x  "
            f"(feature frozen)"
        )

    def log_split(self, frame_idx: int, tid: int) -> None:
        if not self._enabled:
            return
        self._write_event(
            f"  SPLIT   frame={frame_idx}  tid={tid}  "
            f"(feature unfrozen, old pushed to lost gallery)"
        )

    def log_drift(
        self,
        frame_idx: int,
        tid: int,
        distance: float,
    ) -> None:
        if not self._enabled:
            return
        self._write_event(
            f"  DRIFT   frame={frame_idx}  tid={tid}  "
            f"cos_dist={distance:.3f}  (identity drift detected)"
        )

    def log_counter_event(
        self,
        frame_idx: int,
        tid: int,
        counter_name: str,
        event: str,
    ) -> None:
        """event: 'enter', 'exit', 'zone_counted', 'zone_enter', 'zone_leave'"""
        if not self._enabled:
            return
        self._write_event(
            f"  COUNT   frame={frame_idx}  tid={tid}  "
            f"counter={counter_name}  event={event}"
        )

    def log_lost(self, frame_idx: int, tid: int) -> None:
        if not self._enabled:
            return
        self._write_event(
            f"  LOST    frame={frame_idx}  tid={tid}  (track removed)"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._enabled:
            return
        elapsed = time.perf_counter() - self._t0
        self._write_event(f"=== Debug session ended ({elapsed:.1f}s) ===")
        self._csv_file.close()
        self._log_file.close()
        print(f"[debug] logs saved: {self._dir}/")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_event(self, msg: str) -> None:
        elapsed = time.perf_counter() - self._t0
        self._log_file.write(f"[{elapsed:8.3f}] {msg}\n")
        self._log_file.flush()
