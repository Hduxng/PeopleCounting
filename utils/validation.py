"""
Post-tracker validation — lightweight appearance-free sanity checks.

Catches tracker mismatches by verifying that a track's bounding box
properties (aspect ratio, area) remain consistent across frames.
If a track suddenly changes shape, its confidence is penalized so
counters ignore it (soft gate — the track itself is not destroyed).

Also penalizes very young tracks (< min_track_age frames) to prevent
spurious short-lived detections from triggering counts.
"""

import numpy as np


class TrackValidator:
    """
    Args:
        max_ar_change:      Max fractional change in aspect ratio per frame.
        max_area_change:    Max fractional change in bbox area per frame.
        min_track_age:      Frames before a track can trigger counts at full confidence.
        confidence_penalty: Multiplier applied when validation fails (0-1).
    """

    def __init__(
        self,
        max_ar_change: float = 0.3,
        max_area_change: float = 0.5,
        min_track_age: int = 5,
        confidence_penalty: float = 0.5,
    ):
        self._max_ar = max_ar_change
        self._max_area = max_area_change
        self._min_age = min_track_age
        self._penalty = confidence_penalty

        # {tid: (aspect_ratio, area, age)}
        self._history: dict[int, tuple[float, float, int]] = {}

    def validate(self, tid: int, bbox: np.ndarray, confidence: float) -> float:
        """
        Validate a track's bbox consistency and return adjusted confidence.

        Args:
            tid:        Track ID.
            bbox:       [x1, y1, x2, y2] bounding box.
            confidence: Original tracker confidence.

        Returns:
            Adjusted confidence (may be lower if validation fails).
        """
        w = max(bbox[2] - bbox[0], 1.0)
        h = max(bbox[3] - bbox[1], 1.0)
        ar = h / w
        area = w * h

        prev = self._history.get(tid)
        if prev is not None:
            prev_ar, prev_area, age = prev
            age += 1

            # Aspect ratio consistency check
            ar_change = abs(ar - prev_ar) / max(prev_ar, 0.01)
            if ar_change > self._max_ar:
                confidence *= self._penalty

            # Area consistency check
            area_change = abs(area - prev_area) / max(prev_area, 1.0)
            if area_change > self._max_area:
                confidence *= self._penalty

            # Young track penalty
            if age < self._min_age:
                confidence *= 0.8

            self._history[tid] = (ar, area, age)
        else:
            # First frame — apply young track penalty
            confidence *= 0.8
            self._history[tid] = (ar, area, 1)

        return confidence

    def remove(self, tid: int) -> None:
        self._history.pop(tid, None)
