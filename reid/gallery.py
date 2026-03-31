"""
Per-track feature gallery for Re-ID recovery.

Problem it solves:
  When a person is briefly occluded (or walks out/back into frame), the tracker
  loses their track and assigns a NEW ID on reappearance — causing double counting.

How it works:
  1. After each frame, store EMA-averaged feature vectors for every confirmed track.
  2. When a track is lost, keep its features in a "lost gallery" for `lifetime` frames.
  3. When a NEW track appears, compare its features against the lost gallery.
  4. If cosine distance < threshold → remap new ID to the recovered old ID.

Integration: the returned `id_remap` dict is applied BEFORE passing IDs to
counters and the bbox smoother, so counting is unaffected by the spurious ID change.
"""

import numpy as np


class TrackGallery:
    """
    Args:
        embedder:       OSNetEmbedder instance (same one used for tracking).
        lifetime:       Frames to retain a lost track's features (default 90 ≈ 3s@30fps).
        ema_alpha:      EMA weight for active track feature updates (0=replace, 1=freeze).
        match_threshold: Max cosine distance to accept a re-ID match (lower = stricter).
        min_crop_area:  Crops smaller than this (px²) are skipped — too noisy for Re-ID.
    """

    def __init__(
        self,
        embedder,
        lifetime: int   = 90,
        ema_alpha: float = 0.85,
        match_threshold: float = 0.30,
        min_crop_area: int = 800,   # ~28×28 px
    ):
        self._embedder   = embedder
        self._lifetime   = lifetime
        self._alpha      = ema_alpha
        self._threshold  = match_threshold
        self._min_area   = min_crop_area

        # {track_id: np.ndarray}  — L2-normalised 512-D feature
        self._active: dict[int, np.ndarray] = {}

        # {track_id: [feature, frames_since_lost]}
        self._lost: dict[int, list] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        confirmed_ids: set[int],
        crops_by_tid:  dict[int, np.ndarray],
    ) -> dict[int, int]:
        """
        Call once per frame after the tracker returns confirmed tracks.

        Args:
            confirmed_ids: set of track IDs that are confirmed this frame.
            crops_by_tid:  {track_id: BGR crop (numpy array)} for each confirmed track.

        Returns:
            id_remap: {new_track_id: recovered_old_track_id}
                      Apply this mapping to track IDs before updating counters.
        """
        self._age_lost_gallery(confirmed_ids)
        id_remap = self._recover_ids(confirmed_ids, crops_by_tid)
        self._update_active(confirmed_ids, crops_by_tid, id_remap)
        return id_remap

    def remove(self, track_id: int) -> None:
        """Permanently remove a track (e.g. after lifetime expires in the main loop)."""
        self._active.pop(track_id, None)
        self._lost.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _age_lost_gallery(self, confirmed_ids: set[int]) -> None:
        """Move newly-lost tracks to lost gallery; age and evict expired ones."""
        # Tracks that just disappeared
        for tid in set(self._active) - confirmed_ids:
            self._lost[tid] = [self._active.pop(tid), 0]

        # Age remaining lost tracks; evict expired ones
        expired = [
            tid for tid, (_, age) in self._lost.items()
            if age >= self._lifetime
        ]
        for tid in expired:
            del self._lost[tid]

        for entry in self._lost.values():
            entry[1] += 1

    def _recover_ids(
        self,
        confirmed_ids: set[int],
        crops_by_tid:  dict[int, np.ndarray],
    ) -> dict[int, int]:
        """For tracks that just appeared, try to match them to lost tracks."""
        if not self._lost:
            return {}

        new_ids  = confirmed_ids - set(self._active)
        id_remap = {}

        for new_tid in new_ids:
            crop = crops_by_tid.get(new_tid)
            if crop is None or crop.size < self._min_area:
                continue

            new_feat = self._embed_single(crop)
            if new_feat is None:
                continue

            # Cosine distance = 1 − dot(f1, f2) for L2-normalised vectors
            best_old, best_dist = None, self._threshold
            for old_tid, (old_feat, _) in self._lost.items():
                dist = 1.0 - float(np.dot(new_feat, old_feat))
                if dist < best_dist:
                    best_dist = dist
                    best_old  = old_tid

            if best_old is not None:
                id_remap[new_tid] = best_old
                # Promote recovered track back to active gallery
                self._active[new_tid] = self._lost.pop(best_old)[0]

        return id_remap

    def _update_active(
        self,
        confirmed_ids: set[int],
        crops_by_tid:  dict[int, np.ndarray],
        id_remap:      dict[int, int],
    ) -> None:
        """EMA-update features for every confirmed track."""
        for tid in confirmed_ids:
            crop = crops_by_tid.get(tid)
            if crop is None or crop.size < self._min_area:
                continue

            new_feat = self._embed_single(crop)
            if new_feat is None:
                continue

            if tid in self._active:
                blended = self._alpha * self._active[tid] + (1 - self._alpha) * new_feat
                norm    = np.linalg.norm(blended)
                self._active[tid] = blended / norm if norm > 1e-6 else new_feat
            else:
                self._active[tid] = new_feat

    def _embed_single(self, crop: np.ndarray):
        """Embed one crop; return L2-normalised 1-D numpy array or None on failure."""
        try:
            vecs = self._embedder([crop])
            if not vecs:
                return None
            return np.array(vecs[0], dtype=np.float32)
        except Exception:
            return None
