"""
TrackAnchor — prevents track loss for people still visible in frame.

Problem:
  When a person is partially occluded, motion-blurred, or at an unfavourable
  angle, YOLO's confidence drops below the detection threshold. The tracker
  then has no detection to match the existing track against, and the track
  either enters a tentative state or is deleted — even though the person is
  still clearly visible.

Solution:
  Before each tracker.update() call, check every confirmed track from the
  previous frame. If no current YOLO detection overlaps with the track's
  predicted position, inject a synthetic detection there. This "anchors"
  the track so the tracker always has something to associate against.

Synthetic detection properties:
  - confidence: `synthetic_conf` (default 0.45)
      • Above tracker.min_conf (0.35) → accepted for association
      • Below detector.confidence_new_track (0.5) → does NOT trigger counter
        updates and is unlikely to spawn new tracks
  - Position: velocity-EMA extrapolated from recent bbox history
  - Clamped to frame bounds

Safety guards:
  - Only injected for tracks confirmed for ≥ `min_hits` frames
  - At most `max_inject` synthetic detections per frame
  - Injection skipped if bbox predicted outside frame

Velocity model:
  - EMA-smoothed velocity over a sliding window (default 5 frames)
  - Stationary detection: velocity < threshold → snap to zero
  - Adaptive drift cap: scales with bbox size instead of fixed pixels
  - Exponential damping per lost frame → predictions converge to last known position
"""

from collections import deque

import numpy as np


class TrackAnchor:
    """
    Args:
        min_iou:              IoU with an existing detection to consider the track
                              "covered" (no injection needed).
        synthetic_conf:       Confidence assigned to injected detections.
        min_hits:             Minimum confirmed frames before a track is eligible.
        max_inject:           Maximum synthetic detections per frame.
        velocity_damping:     Damping factor per lost frame (0.5 = halve each frame).
        max_drift_px:         Baseline max drift in pixels (scaled adaptively).
        velocity_window:      Number of frames to compute velocity over.
        velocity_ema_alpha:   EMA weight for velocity smoothing (lower = smoother).
        stationary_threshold: Velocity magnitude below which we snap to zero.
    """

    def __init__(
        self,
        min_iou: float = 0.15,
        synthetic_conf: float = 0.45,
        min_hits: int = 3,
        max_inject: int = 10,
        velocity_damping: float = 0.5,
        max_drift_px: float = 30.0,
        velocity_window: int = 5,
        velocity_ema_alpha: float = 0.4,
        stationary_threshold: float = 0.5,
        max_lost_frames: int = 8,
        edge_margin_px: float = 5.0,
        cover_area_ratio: float = 0.50,
        cover_center_ratio: float = 0.08,
        cover_containment_ratio: float = 0.65,
    ):
        self._min_iou = min_iou
        self._conf = synthetic_conf
        self._min_hits = min_hits
        self._max_inject = max_inject
        self._damping = velocity_damping
        self._max_drift = max_drift_px
        self._vel_window = velocity_window
        self._vel_alpha = velocity_ema_alpha
        self._stat_thresh = stationary_threshold
        self._max_lost = max_lost_frames
        self._edge_margin = edge_margin_px
        self._cover_area_ratio = cover_area_ratio
        self._cover_center_ratio = cover_center_ratio
        self._cover_containment_ratio = cover_containment_ratio

        # {tid: deque([bbox0, bbox1, ...], maxlen=velocity_window)}
        self._history: dict[int, deque] = {}
        # {tid: int}  number of confirmed frames
        self._hits: dict[int, int] = {}
        # {tid: int}  frames since last real detection
        self._lost_frames: dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(self, tid: int, bbox: np.ndarray) -> None:
        """Record confirmed track bbox after each frame."""
        bbox = np.asarray(bbox, dtype=float)
        if tid not in self._history:
            self._history[tid] = deque(maxlen=self._vel_window)
        self._history[tid].append(bbox)
        self._hits[tid] = self._hits.get(tid, 0) + 1

    def remove(self, tid: int) -> None:
        """Called when a track is permanently lost."""
        self._history.pop(tid, None)
        self._hits.pop(tid, None)
        self._lost_frames.pop(tid, None)

    # ------------------------------------------------------------------
    def augment(
        self,
        confirmed_tids: set[int],
        dets_np: np.ndarray,   # shape (N, 6): [x1,y1,x2,y2,conf,cls]
        frame_shape: tuple,
        camera_motion: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Augment detection array with synthetic detections for unanchored tracks.

        Args:
            camera_motion: (dx, dy) global camera motion to compensate for.

        Returns:
            (augmented_dets, synthetic_map)
        """
        if not confirmed_tids:
            return dets_np, {}

        self._camera_motion = camera_motion
        fh, fw = frame_shape[:2]
        extra = []
        tid_map = {}
        injected = 0

        for tid in confirmed_tids:
            if injected >= self._max_inject:
                break
            if self._hits.get(tid, 0) < self._min_hits:
                continue

            pred = self._predict(tid, fw, fh)
            if pred is None:
                continue
            if self._has_overlap(pred, dets_np):
                self._lost_frames.pop(tid, None)
                continue

            lost_n = self._lost_frames.get(tid, 0) + 1
            if lost_n > self._max_lost:
                continue
            self._lost_frames[tid] = lost_n

            tid_map[len(extra)] = tid
            extra.append([*pred, self._conf, 0.0])
            injected += 1

        if not extra:
            return dets_np, {}

        n_real = len(dets_np)
        synthetic = np.array(extra, dtype=float)
        augmented = np.vstack([dets_np, synthetic]) if n_real else synthetic
        global_map = {n_real + k: v for k, v in tid_map.items()}
        return augmented, global_map

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ema_velocity(self, hist: deque) -> tuple[float, float]:
        """Compute EMA-smoothed velocity from bbox history.

        Uses center-based velocity across consecutive frames, smoothed with
        exponential moving average. More robust than single-frame difference.
        """
        if len(hist) < 2:
            return 0.0, 0.0

        vx_ema, vy_ema = 0.0, 0.0
        first = True

        for i in range(len(hist) - 1):
            b0, b1 = hist[i], hist[i + 1]
            vx = ((b1[0] + b1[2]) - (b0[0] + b0[2])) * 0.5
            vy = ((b1[1] + b1[3]) - (b0[1] + b0[3])) * 0.5

            if first:
                vx_ema, vy_ema = vx, vy
                first = False
            else:
                vx_ema = self._vel_alpha * vx + (1 - self._vel_alpha) * vx_ema
                vy_ema = self._vel_alpha * vy + (1 - self._vel_alpha) * vy_ema

        # Stationary detection: snap small velocities to zero
        speed = np.sqrt(vx_ema ** 2 + vy_ema ** 2)
        if speed < self._stat_thresh:
            return 0.0, 0.0

        return vx_ema, vy_ema

    def _predict(self, tid: int, fw: int, fh: int) -> list | None:
        """EMA velocity extrapolation with adaptive drift cap."""
        hist = self._history.get(tid)
        if not hist:
            return None

        b_last = hist[-1]
        em = self._edge_margin

        # If the last known bbox is touching a frame edge, the person is
        # likely exiting.  Don't inject — let the track die naturally.
        at_edge = (b_last[0] <= em or b_last[2] >= fw - em or
                   b_last[1] <= em or b_last[3] >= fh - em)
        if at_edge and self._lost_frames.get(tid, 0) > 0:
            return None

        if len(hist) == 1:
            raw_x1, raw_y1, raw_x2, raw_y2 = b_last
        else:
            vx, vy = self._compute_ema_velocity(hist)

            # Subtract camera motion so prediction is in world-space
            cam_dx, cam_dy = getattr(self, '_camera_motion', (0.0, 0.0))
            vx -= cam_dx
            vy -= cam_dy

            # Damp velocity: each lost frame reduces velocity
            lost_n = self._lost_frames.get(tid, 0)
            damp = self._damping ** max(1, lost_n)
            vx *= damp
            vy *= damp

            # Re-add camera motion (prediction should follow camera)
            vx += cam_dx
            vy += cam_dy

            # Adaptive drift cap based on bbox size
            bw = b_last[2] - b_last[0]
            bh = b_last[3] - b_last[1]
            bbox_diag = np.sqrt(bw ** 2 + bh ** 2)
            adaptive_drift = max(self._max_drift, 0.15 * bbox_diag)

            vx = np.clip(vx, -adaptive_drift, adaptive_drift)
            vy = np.clip(vy, -adaptive_drift, adaptive_drift)

            raw_x1 = b_last[0] + vx
            raw_y1 = b_last[1] + vy
            raw_x2 = b_last[2] + vx
            raw_y2 = b_last[3] + vy

            # If the last box is on a border and extrapolation pushes farther out, stop.
            if b_last[2] >= fw and raw_x2 > b_last[2]:
                return None
            if b_last[0] <= 0.0 and raw_x1 < b_last[0]:
                return None
            if b_last[3] >= fh and raw_y2 > b_last[3]:
                return None
            if b_last[1] <= 0.0 and raw_y1 < b_last[1]:
                return None

        # Clamp to frame
        x1 = max(0.0, min(float(fw - 1), raw_x1))
        y1 = max(0.0, min(float(fh - 1), raw_y1))
        x2 = max(x1 + 1.0, min(float(fw), raw_x2))
        y2 = max(y1 + 1.0, min(float(fh), raw_y2))

        # Reject predictions mostly outside frame (> 80% outside)
        pred_area = (x2 - x1) * (y2 - y1)
        orig_area = (b_last[2] - b_last[0]) * (b_last[3] - b_last[1])
        if orig_area > 0 and pred_area / orig_area < 0.2:
            return None

        return [x1, y1, x2, y2]

    def _has_overlap(self, bbox: list, dets_np: np.ndarray) -> bool:
        """Return True if a detection plausibly covers this track.

        A large merged box can overlap multiple tracks with decent IoU. For anchoring we
        only want to treat a detection as coverage when its geometry still looks like the
        same person, not a temporary merge of two nearby people.
        """
        if len(dets_np) == 0:
            return False

        x1, y1, x2, y2 = bbox
        ix1 = np.maximum(x1, dets_np[:, 0])
        iy1 = np.maximum(y1, dets_np[:, 1])
        ix2 = np.minimum(x2, dets_np[:, 2])
        iy2 = np.minimum(y2, dets_np[:, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_bbox = max((x2 - x1) * (y2 - y1), 1.0)
        area_dets = np.maximum((dets_np[:, 2] - dets_np[:, 0]) * (dets_np[:, 3] - dets_np[:, 1]), 1.0)
        union = area_bbox + area_dets - inter
        iou = np.where(union > 0, inter / union, 0.0)

        area_ratio = np.minimum(area_bbox, area_dets) / np.maximum(area_bbox, area_dets)
        inter_over_bbox = inter / area_bbox

        bbox_cx = 0.5 * (x1 + x2)
        bbox_cy = 0.5 * (y1 + y2)
        det_cx = 0.5 * (dets_np[:, 0] + dets_np[:, 2])
        det_cy = 0.5 * (dets_np[:, 1] + dets_np[:, 3])
        center_dist = np.sqrt((det_cx - bbox_cx) ** 2 + (det_cy - bbox_cy) ** 2)
        bbox_diag = max(np.hypot(x2 - x1, y2 - y1), 1.0)

        sized_like_track = area_ratio >= self._cover_area_ratio
        centered_inside_track = (
            (inter_over_bbox >= self._cover_containment_ratio) &
            (center_dist <= self._cover_center_ratio * bbox_diag)
        )
        covered = (iou >= self._min_iou) & (sized_like_track | centered_inside_track)
        return bool(np.any(covered))
