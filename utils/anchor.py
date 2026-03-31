"""
TrackAnchor — prevents track loss for people still visible in frame.

Problem:
  When a person is partially occluded, motion-blurred, or at an unfavourable
  angle, YOLO's confidence drops below the detection threshold. StrongSORT
  then has no detection to match the existing track against, and the track
  either enters a tentative state or is deleted — even though the person is
  still clearly visible.

Solution:
  Before each tracker.update() call, check every confirmed track from the
  previous frame. If no current YOLO detection overlaps with the track's
  predicted position, inject a synthetic detection there. This "anchors"
  the track so StrongSORT always has something to associate against.

Synthetic detection properties:
  - confidence: `synthetic_conf` (default 0.45)
      • Above tracker.min_conf (0.35) → accepted for association
      • Below detector.confidence_new_track (0.5) → does NOT trigger counter
        updates and is unlikely to spawn new tracks
  - Position: linearly extrapolated from the last 2 known bboxes (velocity model)
  - Clamped to frame bounds

Safety guards:
  - Only injected for tracks confirmed for ≥ `min_hits` frames (avoids
    anchoring spurious short-lived tracks)
  - At most `max_inject` synthetic detections per frame (avoids runaway)
  - Injection skipped if bbox predicted outside frame
"""

import numpy as np


class TrackAnchor:
    """
    Args:
        min_iou:        IoU with an existing detection to consider the track
                        "covered" (no injection needed). Default 0.15 — low
                        because we want to catch ANY plausible overlap.
        synthetic_conf: Confidence assigned to injected detections.
        min_hits:       Minimum number of confirmed frames before a track is
                        eligible for anchoring.
        max_inject:     Maximum synthetic detections per frame (safety cap).
    """

    def __init__(
        self,
        min_iou:        float = 0.15,
        synthetic_conf: float = 0.45,
        min_hits:       int   = 3,
        max_inject:     int   = 10,
    ):
        self._min_iou   = min_iou
        self._conf      = synthetic_conf
        self._min_hits  = min_hits
        self._max_inject = max_inject

        # {tid: [bbox_prev, bbox_curr]}  (last 2 bboxes, each [x1,y1,x2,y2])
        self._history:  dict[int, list[np.ndarray]] = {}
        # {tid: int}  number of confirmed frames
        self._hits:     dict[int, int]               = {}

    # ------------------------------------------------------------------
    def update(self, tid: int, bbox: np.ndarray) -> None:
        """Record confirmed track bbox after each frame."""
        bbox = np.asarray(bbox, dtype=float)
        hist = self._history.setdefault(tid, [])
        if len(hist) < 2:
            hist.append(bbox)
        else:
            hist[0] = hist[1]
            hist[1] = bbox
        self._hits[tid] = self._hits.get(tid, 0) + 1

    def remove(self, tid: int) -> None:
        """Called when a track is permanently lost."""
        self._history.pop(tid, None)
        self._hits.pop(tid, None)

    # ------------------------------------------------------------------
    def augment(
        self,
        confirmed_tids: set[int],
        dets_np:        np.ndarray,   # shape (N, 6): [x1,y1,x2,y2,conf,cls]
        frame_shape:    tuple,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Augment detection array with synthetic detections for unanchored tracks.

        Args:
            confirmed_tids: track IDs confirmed in the PREVIOUS frame.
            dets_np:        current YOLO detections (may be empty).
            frame_shape:    (H, W, C) of the current frame.

        Returns:
            (augmented_dets, synthetic_map)
            - augmented_dets: original detections + synthetic ones
            - synthetic_map:  {det_index: track_id} for every injected detection.
              Used by the caller to fill appearance embeddings from gallery
              instead of zero vectors (which cause NaN in cosine normalisation).
        """
        if not confirmed_tids:
            return dets_np, {}

        fh, fw    = frame_shape[:2]
        extra     = []
        tid_map   = {}   # index-in-extra → tid
        injected  = 0

        for tid in confirmed_tids:
            if injected >= self._max_inject:
                break
            if self._hits.get(tid, 0) < self._min_hits:
                continue

            pred = self._predict(tid, fw, fh)
            if pred is None:
                continue
            if self._has_overlap(pred, dets_np):
                continue

            tid_map[len(extra)] = tid
            extra.append([*pred, self._conf, 0.0])
            injected += 1

        if not extra:
            return dets_np, {}

        n_real       = len(dets_np)
        synthetic    = np.array(extra, dtype=float)
        augmented    = np.vstack([dets_np, synthetic]) if n_real else synthetic
        # Remap local extra-index → global det-index
        global_map   = {n_real + k: v for k, v in tid_map.items()}
        return augmented, global_map

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict(self, tid: int, fw: int, fh: int) -> list | None:
        """Linear velocity extrapolation from bbox history."""
        hist = self._history.get(tid)
        if not hist:
            return None

        b1 = hist[-1]
        if len(hist) == 1:
            raw_x1, raw_y1, raw_x2, raw_y2 = b1
        else:
            b0, b1 = hist[0], hist[1]
            # Centre-based velocity
            vx = ((b1[0] + b1[2]) - (b0[0] + b0[2])) * 0.5
            vy = ((b1[1] + b1[3]) - (b0[1] + b0[3])) * 0.5
            raw_x1 = b1[0] + vx
            raw_y1 = b1[1] + vy
            raw_x2 = b1[2] + vx
            raw_y2 = b1[3] + vy

            # If the last box is already on a border and the extrapolation pushes
            # it farther out through that same border, stop anchoring.
            if b1[2] >= fw and raw_x2 > b1[2]:
                return None
            if b1[0] <= 0.0 and raw_x1 < b1[0]:
                return None
            if b1[3] >= fh and raw_y2 > b1[3]:
                return None
            if b1[1] <= 0.0 and raw_y1 < b1[1]:
                return None

        # Clamp to frame
        x1 = max(0.0, min(float(fw - 1), raw_x1))
        y1 = max(0.0, min(float(fh - 1), raw_y1))
        x2 = max(x1 + 1.0, min(float(fw), raw_x2))
        y2 = max(y1 + 1.0, min(float(fh), raw_y2))

        # Reject predictions mostly outside frame (> 80% outside)
        pred_area = (x2 - x1) * (y2 - y1)
        orig_area  = (hist[-1][2] - hist[-1][0]) * (hist[-1][3] - hist[-1][1])
        if orig_area > 0 and pred_area / orig_area < 0.2:
            return None

        return [x1, y1, x2, y2]

    def _has_overlap(self, bbox: list, dets_np: np.ndarray) -> bool:
        """Return True if bbox has IoU >= min_iou with any detection."""
        if len(dets_np) == 0:
            return False
        x1, y1, x2, y2 = bbox
        for det in dets_np:
            dx1, dy1, dx2, dy2 = det[0], det[1], det[2], det[3]
            ix1 = max(x1, dx1); iy1 = max(y1, dy1)
            ix2 = min(x2, dx2); iy2 = min(y2, dy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (x2-x1)*(y2-y1) + (dx2-dx1)*(dy2-dy1) - inter
                if union > 0 and inter / union >= self._min_iou:
                    return True
        return False
