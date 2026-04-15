"""
Histogram-based Re-ID gallery — lightweight alternative to model-based Re-ID.

Uses HSV color histograms (OpenCV only, no neural network) to match
lost tracks with new detections.  Much less accurate than CLIP-ReID
but requires zero GPU memory and no model weights.

Features:
  - CLAHE normalization on V-channel for lighting robustness
  - Spatial pyramid: 3 horizontal strips (head/torso/legs) for spatial structure
  - Optional V-channel histogram for brightness discrimination
  - EMA-smoothed histogram updates
  - Identity drift detection for merge-split handling

Distance metric: per-strip Bhattacharyya distance via cv2.compareHist.
Lower = more similar (0 = identical, 1 = completely different).
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class HistogramGallery:
    """
    Args:
        lifetime:        Frames to retain a lost track's histogram.
        ema_alpha:       EMA weight for active track histogram updates.
        match_threshold: Max Bhattacharyya distance to accept a re-ID match.
        min_crop_area:   Crops smaller than this (px²) are skipped.
        h_bins:          Number of hue bins (HSV histogram).
        s_bins:          Number of saturation bins.
        drift_threshold: Bhattacharyya distance to flag identity drift.
        use_clahe:       Apply CLAHE to V-channel before histogram computation.
        spatial_pyramid: Split crop into horizontal strips for spatial features.
        n_strips:        Number of horizontal strips (default 3: head/torso/legs).
        v_bins:          Value channel histogram bins (0 = disabled).
        v_weight:        Weight for V-channel distance relative to HS distance.
    """

    def __init__(
        self,
        lifetime: int = 90,
        ema_alpha: float = 0.85,
        match_threshold: float = 0.55,
        min_crop_area: int = 800,
        h_bins: int = 50,
        s_bins: int = 60,
        drift_threshold: float = 0.70,
        use_clahe: bool = True,
        spatial_pyramid: bool = True,
        n_strips: int = 3,
        v_bins: int = 32,
        v_weight: float = 0.3,
    ):
        self._lifetime = lifetime
        self._alpha = ema_alpha
        self._threshold = match_threshold
        self._min_area = min_crop_area
        self._h_bins = h_bins
        self._s_bins = s_bins
        self._drift_thresh = drift_threshold
        self._use_clahe = use_clahe
        self._spatial = spatial_pyramid
        self._n_strips = n_strips if spatial_pyramid else 1
        self._v_bins = v_bins
        self._v_weight = v_weight

        # Pre-compute CLAHE object (thread-safe, reusable)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if use_clahe else None

        # Feature size per strip: h_bins*s_bins + v_bins
        self._hs_size = h_bins * s_bins
        self._strip_size = self._hs_size + v_bins
        self._feat_size = self._strip_size * self._n_strips

        # Strip height ratios (head=20%, torso=50%, legs=30%)
        if self._n_strips == 3:
            self._strip_ratios = [0.0, 0.2, 0.7, 1.0]
        elif self._n_strips == 2:
            self._strip_ratios = [0.0, 0.4, 1.0]
        else:
            self._strip_ratios = [0.0, 1.0]

        # {track_id: np.ndarray}  — concatenated strip histograms
        self._active: dict[int, np.ndarray] = {}

        # {track_id: [histogram, frames_since_lost]}
        self._lost: dict[int, list] = {}

        # {track_id: np.ndarray} — last known bbox (xyxy) for spatial matching
        self._last_bbox: dict[int, np.ndarray] = {}

        # Max center displacement as fraction of bbox diagonal for recovery
        self._max_remap_dist_ratio: float = 3.0

    # ------------------------------------------------------------------
    def update(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
        bboxes_by_tid: dict[int, np.ndarray] | None = None,
    ) -> dict[int, int]:
        """
        Call once per frame after the tracker returns confirmed tracks.

        Returns:
            id_remap: {new_track_id: recovered_old_track_id}
        """
        if bboxes_by_tid is not None:
            for tid, bbox in bboxes_by_tid.items():
                self._last_bbox[tid] = np.asarray(bbox, dtype=float)

        self._detect_identity_drift(confirmed_ids, crops_by_tid)
        self._age_lost_gallery(confirmed_ids)
        id_remap = self._recover_ids(confirmed_ids, crops_by_tid, bboxes_by_tid)
        self._update_active(confirmed_ids, crops_by_tid, id_remap)
        return id_remap

    def remove(self, track_id: int) -> None:
        self._active.pop(track_id, None)
        self._lost.pop(track_id, None)
        self._last_bbox.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_histogram(self, crop: np.ndarray) -> np.ndarray | None:
        """Compute spatial pyramid histogram from a BGR crop."""
        if crop is None or crop.size < self._min_area:
            return None
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # CLAHE normalization on Value channel
            if self._clahe is not None:
                hsv[:, :, 2] = self._clahe.apply(hsv[:, :, 2])

            h = hsv.shape[0]
            if h < self._n_strips:
                return None

            features = []
            for i in range(self._n_strips):
                y_start = int(self._strip_ratios[i] * h)
                y_end = int(self._strip_ratios[i + 1] * h)
                if y_end <= y_start:
                    y_end = y_start + 1
                strip = hsv[y_start:y_end]

                # HS histogram
                hs_hist = cv2.calcHist(
                    [strip], [0, 1], None,
                    [self._h_bins, self._s_bins],
                    [0, 180, 0, 256],
                )
                cv2.normalize(hs_hist, hs_hist, 0, 1, cv2.NORM_MINMAX)
                features.append(hs_hist.flatten())

                # V-channel histogram
                if self._v_bins > 0:
                    v_hist = cv2.calcHist(
                        [strip], [2], None,
                        [self._v_bins],
                        [0, 256],
                    )
                    cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
                    features.append(v_hist.flatten())

            return np.concatenate(features).astype(np.float32)
        except Exception:
            return None

    def _compare(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """Weighted per-strip Bhattacharyya distance."""
        total_dist = 0.0
        n_components = 0

        for i in range(self._n_strips):
            offset = i * self._strip_size

            # HS distance
            hs1 = h1[offset:offset + self._hs_size].reshape(self._h_bins, self._s_bins)
            hs2 = h2[offset:offset + self._hs_size].reshape(self._h_bins, self._s_bins)
            hs_dist = cv2.compareHist(
                hs1.astype(np.float32),
                hs2.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            total_dist += hs_dist
            n_components += 1

            # V-channel distance (weighted)
            if self._v_bins > 0:
                v_offset = offset + self._hs_size
                v1 = h1[v_offset:v_offset + self._v_bins].reshape(self._v_bins, 1)
                v2 = h2[v_offset:v_offset + self._v_bins].reshape(self._v_bins, 1)
                v_dist = cv2.compareHist(
                    v1.astype(np.float32),
                    v2.astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                total_dist += self._v_weight * v_dist
                n_components += self._v_weight

        return total_dist / n_components if n_components > 0 else 1.0

    def _detect_identity_drift(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
    ) -> None:
        for tid in list(confirmed_ids & set(self._active)):
            crop = crops_by_tid.get(tid)
            new_hist = self._compute_histogram(crop)
            if new_hist is None:
                continue

            dist = self._compare(new_hist, self._active[tid])
            if dist > self._drift_thresh:
                self._lost[tid] = [self._active[tid], 0]
                self._active[tid] = new_hist

    def _age_lost_gallery(self, confirmed_ids: set[int]) -> None:
        for tid in set(self._active) - confirmed_ids:
            self._lost[tid] = [self._active.pop(tid), 0]

        expired = [
            tid for tid, (_, age) in self._lost.items()
            if age >= self._lifetime
        ]
        for tid in expired:
            del self._lost[tid]
            self._last_bbox.pop(tid, None)

        for entry in self._lost.values():
            entry[1] += 1

    def _spatial_match_ok(
        self,
        new_bbox: np.ndarray,
        old_bbox: np.ndarray,
    ) -> bool:
        """Check if two bboxes are spatially plausible for the same person."""
        new_bbox = np.asarray(new_bbox, dtype=float)
        old_bbox = np.asarray(old_bbox, dtype=float)

        nc = np.array([(new_bbox[0] + new_bbox[2]) * 0.5, (new_bbox[1] + new_bbox[3]) * 0.5])
        oc = np.array([(old_bbox[0] + old_bbox[2]) * 0.5, (old_bbox[1] + old_bbox[3]) * 0.5])
        center_dist = float(np.linalg.norm(nc - oc))

        diag = float(np.linalg.norm([
            old_bbox[2] - old_bbox[0],
            old_bbox[3] - old_bbox[1],
        ]))
        if diag <= 0:
            return True
        return center_dist <= self._max_remap_dist_ratio * diag

    def _recover_ids(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
        bboxes_by_tid: dict[int, np.ndarray] | None = None,
    ) -> dict[int, int]:
        if not self._lost:
            return {}

        new_ids = sorted(confirmed_ids - set(self._active))
        if not new_ids:
            return {}

        lost_ids = sorted(self._lost)

        # Compute histograms for new tracks
        new_hists: dict[int, np.ndarray] = {}
        for new_tid in new_ids:
            hist = self._compute_histogram(crops_by_tid.get(new_tid))
            if hist is not None:
                new_hists[new_tid] = hist

        valid_new = sorted(new_hists)
        if not valid_new:
            return {}

        # Build cost matrix: rows = new tracks, cols = lost tracks + dummy columns
        num_new = len(valid_new)
        num_old = len(lost_ids)
        invalid_cost = 1e9
        cost_matrix = np.full((num_new, num_old + num_new), invalid_cost, dtype=np.float64)
        cost_matrix[:, num_old:] = self._threshold  # dummy columns = "no match"

        for row, new_tid in enumerate(valid_new):
            new_hist = new_hists[new_tid]
            new_bbox = None if bboxes_by_tid is None else bboxes_by_tid.get(new_tid)
            for col, old_tid in enumerate(lost_ids):
                old_hist, _ = self._lost[old_tid]
                dist = self._compare(new_hist, old_hist)
                if dist >= self._threshold:
                    continue
                # Spatial sanity check
                old_bbox = self._last_bbox.get(old_tid) if hasattr(self, '_last_bbox') else None
                if new_bbox is not None and old_bbox is not None:
                    if not self._spatial_match_ok(new_bbox, old_bbox):
                        continue
                cost_matrix[row, col] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        id_remap: dict[int, int] = {}

        for row, col in zip(row_ind, col_ind):
            if col >= num_old or cost_matrix[row, col] >= self._threshold:
                continue
            new_tid = valid_new[row]
            old_tid = lost_ids[col]
            id_remap[new_tid] = old_tid
            self._active[new_tid] = self._lost.pop(old_tid)[0]

        return id_remap

    def _update_active(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
        id_remap: dict[int, int],
    ) -> None:
        for tid in confirmed_ids:
            crop = crops_by_tid.get(tid)
            new_hist = self._compute_histogram(crop)
            if new_hist is None:
                continue

            if tid in self._active:
                self._active[tid] = (
                    self._alpha * self._active[tid]
                    + (1 - self._alpha) * new_hist
                )
            else:
                self._active[tid] = new_hist
