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

Merge-split handling:
  When two people stand very close, detector boxes may merge and split repeatedly.
  The gallery now avoids reacting to one-frame appearance spikes during merges,
  and for very short gaps it can recover IDs using strong spatial continuity even
  when Re-ID distance is temporarily noisy.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _bbox_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=float)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=float)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0))
    area_b = float(max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_inter_over_smaller(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = np.asarray(box_a, dtype=float)
    bx1, by1, bx2, by2 = np.asarray(box_b, dtype=float)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0))
    area_b = float(max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0))
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _bbox_center_distance(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    if box_a is None or box_b is None:
        return float('inf')
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=float)
    bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=float)
    return float(np.linalg.norm(ac - bc))


def _match_quality_reward(quality: tuple[float, float, float]) -> float:
    tier, score, recency = quality
    return float(tier) * 1_000_000.0 + float(score) * 1_000.0 + float(recency)


def _normalize_feature(feature: np.ndarray | list | tuple | None) -> np.ndarray | None:
    if feature is None:
        return None
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-6:
        return None
    return arr / norm


def _clone_feature_bank(bank: list[np.ndarray] | None) -> list[np.ndarray]:
    if not bank:
        return []
    return [np.asarray(feature, dtype=np.float32).copy() for feature in bank]


def _feature_bank_centroid(bank: list[np.ndarray] | None) -> np.ndarray | None:
    if not bank:
        return None
    stacked = np.asarray(bank, dtype=np.float32)
    if stacked.ndim != 2 or stacked.shape[0] == 0:
        return None
    centroid = stacked.mean(axis=0)
    return _normalize_feature(centroid)


def _feature_distance_to_bank(
    feature: np.ndarray | None,
    representative: np.ndarray | None,
    bank: list[np.ndarray] | None,
) -> float | None:
    feature = _normalize_feature(feature)
    if feature is None:
        return None

    distances = []
    rep = _normalize_feature(representative)
    if rep is not None:
        distances.append(1.0 - float(np.dot(feature, rep)))

    if bank:
        for prototype in bank:
            proto = _normalize_feature(prototype)
            if proto is None:
                continue
            distances.append(1.0 - float(np.dot(feature, proto)))

    if not distances:
        return None
    return min(distances)


class TrackGallery:
    """
    Args:
        embedder:        Re-ID embedder instance (CLIPReIDEmbedder or OSNetEmbedder).
        lifetime:        Frames to retain a lost track's features (default 90 ≈ 3s@30fps).
        ema_alpha:       EMA weight for active track feature updates (0=replace, 1=freeze).
        match_threshold: Max cosine distance to accept a re-ID match (lower = stricter).
        min_crop_area:   Crops smaller than this (px²) are skipped — too noisy for Re-ID.
        drift_threshold: If a confirmed track's current appearance is farther than this
                         from its stored feature, flag as potential identity swap after merge-split.
    """

    def __init__(
        self,
        embedder,
        lifetime: int = 90,
        ema_alpha: float = 0.85,
        match_threshold: float = 0.30,
        min_crop_area: int = 800,
        drift_threshold: float = 0.6,
        merge_area_ratio: float = 1.5,
        merge_release_ratio: float | None = None,
        merge_hold_frames: int = 45,
        drift_confirm_frames: int = 2,
        spatial_match_window: int = 12,
        spatial_iou_threshold: float = 0.45,
        spatial_area_ratio: float = 0.60,
        spatial_center_ratio: float = 0.30,
        spatial_containment_threshold: float = 0.82,
        spatial_containment_center_ratio: float = 0.55,
        relaxed_match_threshold: float | None = None,
        active_conflict_iou: float = 0.30,
        active_conflict_area_ratio: float = 0.55,
        active_conflict_center_ratio: float = 0.35,
        max_prototypes: int = 5,
        min_update_conf: float = 0.0,
        ema_min_similarity: float = 0.5,
        debug_logger=None,
    ):
        self._embedder = embedder
        self._ema_min_similarity = ema_min_similarity
        self._lifetime = lifetime
        self._alpha = ema_alpha
        self._threshold = match_threshold
        self._min_area = min_crop_area
        self._min_update_conf = float(min_update_conf)
        self._drift_thresh = drift_threshold
        self._merge_area_ratio = merge_area_ratio
        release_default = 1.0 + max(merge_area_ratio - 1.0, 0.0) * 0.5
        self._merge_release_ratio = min(
            merge_area_ratio,
            max(1.0, release_default if merge_release_ratio is None else float(merge_release_ratio)),
        )
        self._merge_hold_frames = max(1, int(merge_hold_frames))
        self._drift_confirm = max(1, int(drift_confirm_frames))
        self._spatial_window = max(0, int(spatial_match_window))
        self._spatial_iou = spatial_iou_threshold
        self._spatial_area_ratio = spatial_area_ratio
        self._spatial_center_ratio = spatial_center_ratio
        self._spatial_containment = spatial_containment_threshold
        self._spatial_containment_center_ratio = spatial_containment_center_ratio
        self._active_conflict_iou = active_conflict_iou
        self._active_conflict_area_ratio = active_conflict_area_ratio
        self._active_conflict_center_ratio = active_conflict_center_ratio
        relaxed_default = (
            match_threshold + 0.15
            if relaxed_match_threshold is None else relaxed_match_threshold
        )
        self._relaxed_threshold = max(match_threshold, relaxed_default)
        self._dbg = debug_logger
        # Max center displacement (as a multiple of bbox diagonal) allowed for an
        # appearance-only match when the track was lost within spatial_match_window.
        # A look-alike standing far away should not steal a recently-lost track's ID.
        self._max_remap_dist_ratio: float = 1.8
        self._long_gap_remap_dist_ratio: float = 3.0
        self._long_gap_area_ratio: float = 0.25
        self._active_conflict_containment: float = 0.80
        self._max_prototypes = max(1, int(max_prototypes))

        # {track_id: np.ndarray} — representative feature for active tracks
        self._active: dict[int, np.ndarray] = {}
        # {track_id: [feature0, feature1, ...]} — recent good embeddings
        self._active_banks: dict[int, list[np.ndarray]] = {}

        # {track_id: [representative_feature, frames_since_lost, last_bbox]}
        self._lost: dict[int, list] = {}
        # {track_id: [feature0, feature1, ...]} — prototype bank retained while lost
        self._lost_banks: dict[int, list[np.ndarray]] = {}

        # {track_id: float} — last known crop area, for merge detection
        self._last_area: dict[int, float] = {}

        # {track_id: np.ndarray} — last reliable bbox for short-gap spatial recovery
        self._last_bbox: dict[int, np.ndarray] = {}

        # set of track IDs currently flagged as merged (feature updates frozen)
        self._merged_ids: set[int] = set()

        # {track_id: float} — area before the merge event (for split detection)
        self._pre_merge_area: dict[int, float] = {}
        # {track_id: int} — consecutive frames spent in merge-freeze state
        self._merge_counts: dict[int, int] = {}

        # {track_id: int} — consecutive drift frames before we trust the signal
        self._drift_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
        precomputed_embeddings: dict[int, np.ndarray] | None = None,
        bboxes_by_tid: dict[int, np.ndarray] | None = None,
        update_confidences: dict[int, float] | None = None,
        frame_idx: int = 0,
    ) -> dict[int, int]:
        """Call once per frame after the tracker returns confirmed tracks."""
        self._frame_idx = frame_idx
        if bboxes_by_tid is not None:
            for tid, bbox in bboxes_by_tid.items():
                self._last_bbox[tid] = np.asarray(bbox, dtype=float)

        embeddings = self._prepare_embeddings(
            confirmed_ids,
            crops_by_tid,
            precomputed_embeddings=precomputed_embeddings,
        )

        self._detect_merges(confirmed_ids, bboxes_by_tid)
        self._detect_identity_drift(confirmed_ids, embeddings)
        self._age_lost_gallery(confirmed_ids)
        id_remap = self._recover_ids(confirmed_ids, embeddings, bboxes_by_tid)
        self._update_active(
            confirmed_ids,
            embeddings,
            id_remap,
            crops_by_tid=crops_by_tid,
            update_confidences=update_confidences,
        )
        return id_remap

    def get_feature(self, track_id: int) -> np.ndarray | None:
        """Return the stored representative feature for a track (active or lost)."""
        feat = self._active.get(track_id)
        if feat is None:
            lost_entry = self._lost.get(track_id)
            if lost_entry is not None:
                feat = lost_entry[0]
        return feat

    def remove(self, track_id: int) -> None:
        """Permanently remove a track (e.g. after lifetime expires in the main loop)."""
        self._active.pop(track_id, None)
        self._active_banks.pop(track_id, None)
        self._lost.pop(track_id, None)
        self._lost_banks.pop(track_id, None)
        self._last_area.pop(track_id, None)
        self._last_bbox.pop(track_id, None)
        self._merged_ids.discard(track_id)
        self._pre_merge_area.pop(track_id, None)
        self._merge_counts.pop(track_id, None)
        self._drift_counts.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_merges(
        self,
        confirmed_ids: set[int],
        bboxes_by_tid: dict[int, np.ndarray] | None,
    ) -> None:
        """Flag tracks whose bbox area jumped (likely absorbed another person)."""
        if bboxes_by_tid is None:
            return

        for tid in confirmed_ids:
            bb = bboxes_by_tid.get(tid)
            if bb is None:
                continue
            area = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
            prev_area = self._last_area.get(tid)

            if area < self._min_area or (prev_area is not None and prev_area < self._min_area):
                self._last_area[tid] = area
                continue

            if prev_area is not None and prev_area > 0:
                ratio = area / prev_area
                if ratio >= self._merge_area_ratio and tid not in self._merged_ids:
                    self._merged_ids.add(tid)
                    self._pre_merge_area[tid] = prev_area
                    self._merge_counts[tid] = 0
                    self._drift_counts.pop(tid, None)
                    if self._dbg is not None:
                        self._dbg.log_merge(self._frame_idx, tid, area, prev_area)
                elif tid in self._merged_ids:
                    pre_area = self._pre_merge_area.get(tid, prev_area)
                    self._merge_counts[tid] = self._merge_counts.get(tid, 0) + 1
                    release_ratio = area / pre_area if pre_area > 0 else float("inf")
                    if pre_area > 0 and release_ratio < 1.0 / self._merge_area_ratio:
                        self._merged_ids.discard(tid)
                        self._pre_merge_area.pop(tid, None)
                        self._merge_counts.pop(tid, None)
                        self._snapshot_active_to_lost(tid)
                        if self._dbg is not None:
                            self._dbg.log_split(self._frame_idx, tid)
                    elif pre_area > 0 and release_ratio <= self._merge_release_ratio:
                        self._merged_ids.discard(tid)
                        self._pre_merge_area.pop(tid, None)
                        self._merge_counts.pop(tid, None)
                    elif self._merge_counts.get(tid, 0) >= self._merge_hold_frames:
                        self._merged_ids.discard(tid)
                        self._pre_merge_area.pop(tid, None)
                        self._merge_counts.pop(tid, None)

            self._last_area[tid] = area

    def _batch_embed(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Embed ALL confirmed crops in a single batched forward pass."""
        tids = []
        crops = []
        for tid in confirmed_ids:
            crop = crops_by_tid.get(tid)
            if self._crop_area(crop) >= self._min_area:
                tids.append(tid)
                crops.append(crop)

        if not crops:
            return {}

        try:
            vecs = self._embedder(crops)
        except Exception:
            return {}

        embeddings = {}
        for tid, vec in zip(tids, vecs):
            feature = _normalize_feature(vec)
            if feature is not None:
                embeddings[tid] = feature
        return embeddings

    def _crop_area(self, crop: np.ndarray | None) -> int:
        if crop is None or crop.ndim < 2:
            return 0
        return int(crop.shape[0] * crop.shape[1])

    def _prepare_embeddings(
        self,
        confirmed_ids: set[int],
        crops_by_tid: dict[int, np.ndarray],
        precomputed_embeddings: dict[int, np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        if precomputed_embeddings is not None:
            raw_embeddings = precomputed_embeddings
        else:
            raw_embeddings = self._batch_embed(confirmed_ids, crops_by_tid)

        embeddings: dict[int, np.ndarray] = {}
        for tid in confirmed_ids:
            if self._crop_area(crops_by_tid.get(tid)) < self._min_area:
                continue
            feature = _normalize_feature(raw_embeddings.get(tid))
            if feature is not None:
                embeddings[tid] = feature
        return embeddings

    def _snapshot_active_to_lost(self, tid: int) -> None:
        representative = self._active.get(tid)
        representative = _normalize_feature(representative)
        if representative is None:
            return
        self._lost[tid] = [representative, 0, self._last_bbox.get(tid)]
        bank = self._active_banks.get(tid)
        cloned_bank = _clone_feature_bank(bank)
        self._lost_banks[tid] = cloned_bank if cloned_bank else [representative.copy()]

    def _move_active_to_lost(self, tid: int) -> None:
        self._snapshot_active_to_lost(tid)
        self._active.pop(tid, None)
        self._active_banks.pop(tid, None)
        self._drift_counts.pop(tid, None)

    def _restore_lost_to_active(self, new_tid: int, old_tid: int, fallback_feature: np.ndarray | None) -> None:
        representative, _, _ = self._lost.pop(old_tid)
        bank = self._lost_banks.pop(old_tid, None)
        cloned_bank = _clone_feature_bank(bank)
        if not cloned_bank:
            fallback = _normalize_feature(fallback_feature if fallback_feature is not None else representative)
            cloned_bank = [fallback] if fallback is not None else []
        self._active_banks[new_tid] = cloned_bank
        centroid = _feature_bank_centroid(cloned_bank)
        if centroid is None:
            centroid = _normalize_feature(representative)
        if centroid is not None:
            self._active[new_tid] = centroid

        if self._dbg is not None:
            self._dbg.log_gallery_prototypes(
                self._frame_idx,
                new_tid,
                len(cloned_bank),
                source="restore",
            )

    def _update_acceptance_status(
        self,
        tid: int,
        crops_by_tid: dict[int, np.ndarray],
        update_confidences: dict[int, float] | None,
    ) -> tuple[bool, str | None, int, float | None]:
        crop_area = self._crop_area(crops_by_tid.get(tid))
        if crop_area < self._min_area:
            return False, "small_crop", crop_area, None
        if update_confidences is None:
            return True, None, crop_area, None
        try:
            confidence = float(update_confidences.get(tid, 1.0))
        except (TypeError, ValueError):
            return False, "low_conf", crop_area, None
        if not np.isfinite(confidence):
            return False, "low_conf", crop_area, confidence
        if confidence < self._min_update_conf:
            return False, "low_conf", crop_area, confidence
        return True, None, crop_area, confidence

    def _log_update_skip(
        self,
        tid: int,
        reason: str,
        *,
        confidence: float | None = None,
        crop_area: int | None = None,
    ) -> None:
        if self._dbg is None:
            return
        self._dbg.log_gallery_skip(
            self._frame_idx,
            tid,
            reason=reason,
            confidence=confidence,
            crop_area=crop_area,
        )

    def _append_active_prototype(self, tid: int, new_feat: np.ndarray) -> None:
        new_feat = _normalize_feature(new_feat)
        if new_feat is None:
            return

        bank = _clone_feature_bank(self._active_banks.get(tid))
        if not bank and tid in self._active:
            representative = _normalize_feature(self._active.get(tid))
            if representative is not None:
                bank.append(representative)

        bank.append(new_feat)
        if len(bank) > self._max_prototypes:
            bank = bank[-self._max_prototypes:]
        self._active_banks[tid] = bank

        centroid = _feature_bank_centroid(bank)
        if centroid is None:
            return

        previous = _normalize_feature(self._active.get(tid))
        if previous is None:
            self._active[tid] = centroid
        else:
            # Similarity guard: skip blend if new centroid is wildly different
            sim = float(np.dot(previous, centroid))
            if sim < self._ema_min_similarity:
                return
            blended = self._alpha * previous + (1.0 - self._alpha) * centroid
            normalized_blended = _normalize_feature(blended)
            self._active[tid] = normalized_blended if normalized_blended is not None else centroid

        if self._dbg is not None:
            self._dbg.log_gallery_prototypes(
                self._frame_idx,
                tid,
                len(bank),
                source="update",
            )

    def _detect_identity_drift(
        self,
        confirmed_ids: set[int],
        embeddings: dict[int, np.ndarray],
    ) -> None:
        """Detect tracks whose appearance has drifted too far from stored feature."""
        for tid in list(confirmed_ids & set(self._active)):
            if tid in self._merged_ids:
                self._drift_counts.pop(tid, None)
                continue

            new_feat = embeddings.get(tid)
            if new_feat is None:
                self._drift_counts.pop(tid, None)
                continue

            dist = _feature_distance_to_bank(
                new_feat,
                self._active.get(tid),
                self._active_banks.get(tid),
            )
            if dist is None:
                self._drift_counts.pop(tid, None)
                continue
            if dist <= self._drift_thresh:
                self._drift_counts.pop(tid, None)
                continue

            drift_count = self._drift_counts.get(tid, 0) + 1
            self._drift_counts[tid] = drift_count
            if drift_count < self._drift_confirm:
                continue

            self._drift_counts.pop(tid, None)
            self._snapshot_active_to_lost(tid)
            self._active[tid] = new_feat
            self._active_banks[tid] = [new_feat.copy()]
            if self._dbg is not None:
                self._dbg.log_drift(self._frame_idx, tid, dist)

    def _age_lost_gallery(self, confirmed_ids: set[int]) -> None:
        """Move newly-lost tracks to lost gallery; age and evict expired ones."""
        for tid in set(self._active) - confirmed_ids:
            self._move_active_to_lost(tid)

        expired = [
            tid for tid, (_, age, _) in self._lost.items()
            if age >= self._lifetime
        ]
        for tid in expired:
            del self._lost[tid]
            self._lost_banks.pop(tid, None)

        for entry in self._lost.values():
            entry[1] += 1

    def _bbox_conflicts_with_active(
        self,
        candidate_bbox: np.ndarray | None,
        other_bbox: np.ndarray | None,
    ) -> bool:
        if candidate_bbox is None or other_bbox is None:
            return False

        candidate_bbox = np.asarray(candidate_bbox, dtype=float)
        other_bbox = np.asarray(other_bbox, dtype=float)
        candidate_area = float(max((candidate_bbox[2] - candidate_bbox[0]) * (candidate_bbox[3] - candidate_bbox[1]), 1.0))
        other_area = float(max((other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1]), 1.0))
        area_ratio = min(candidate_area, other_area) / max(candidate_area, other_area)

        ix1 = max(candidate_bbox[0], other_bbox[0])
        iy1 = max(candidate_bbox[1], other_bbox[1])
        ix2 = min(candidate_bbox[2], other_bbox[2])
        iy2 = min(candidate_bbox[3], other_bbox[3])
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        inter_over_smaller = inter / max(min(candidate_area, other_area), 1.0)

        iou = _bbox_iou(candidate_bbox, other_bbox)
        if iou >= self._active_conflict_iou:
            return True

        center_dist = _bbox_center_distance(candidate_bbox, other_bbox)
        diag_ref = max(
            min(
                np.linalg.norm([candidate_bbox[2] - candidate_bbox[0], candidate_bbox[3] - candidate_bbox[1]]),
                np.linalg.norm([other_bbox[2] - other_bbox[0], other_bbox[3] - other_bbox[1]]),
            ),
            1.0,
        )
        if inter_over_smaller >= self._active_conflict_containment:
            return center_dist <= max(self._active_conflict_center_ratio, 0.40) * diag_ref
        if area_ratio < self._active_conflict_area_ratio:
            return False
        return center_dist <= self._active_conflict_center_ratio * diag_ref

    def _has_active_conflict(
        self,
        new_tid: int,
        confirmed_ids: set[int],
        bboxes_by_tid: dict[int, np.ndarray] | None,
        new_bbox: np.ndarray | None,
        old_bbox: np.ndarray | None,
    ) -> bool:
        if bboxes_by_tid is None:
            return False

        for other_tid in confirmed_ids:
            if other_tid == new_tid:
                continue
            other_bbox = bboxes_by_tid.get(other_tid)
            if self._bbox_conflicts_with_active(new_bbox, other_bbox):
                return True
            if self._bbox_conflicts_with_active(old_bbox, other_bbox):
                return True
        return False

    def _long_gap_appearance_match_ok(
        self,
        new_bbox: np.ndarray | None,
        old_bbox: np.ndarray | None,
    ) -> bool:
        if new_bbox is None or old_bbox is None:
            return True

        new_bbox = np.asarray(new_bbox, dtype=float)
        old_bbox = np.asarray(old_bbox, dtype=float)
        new_area = float(max((new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]), 1.0))
        old_area = float(max((old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1]), 1.0))
        area_ratio = min(new_area, old_area) / max(new_area, old_area)
        if area_ratio < self._long_gap_area_ratio:
            return False

        center_dist = _bbox_center_distance(new_bbox, old_bbox)
        diag_ref = max(
            min(
                np.linalg.norm([new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]]),
                np.linalg.norm([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]),
            ),
            1.0,
        )
        return center_dist <= self._long_gap_remap_dist_ratio * diag_ref

    def _spatial_match_quality(
        self,
        new_bbox: np.ndarray | None,
        old_bbox: np.ndarray | None,
        age: int,
    ) -> tuple[float, float, float] | None:
        if (
            self._spatial_window <= 0 or
            age > self._spatial_window or
            new_bbox is None or
            old_bbox is None
        ):
            return None

        new_bbox = np.asarray(new_bbox, dtype=float)
        old_bbox = np.asarray(old_bbox, dtype=float)
        new_area = float(max((new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]), 1.0))
        old_area = float(max((old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1]), 1.0))
        area_ratio = min(new_area, old_area) / max(new_area, old_area)

        iou = _bbox_iou(new_bbox, old_bbox)
        inter_over_smaller = _bbox_inter_over_smaller(new_bbox, old_bbox)
        center_dist = _bbox_center_distance(new_bbox, old_bbox)
        diag_ref = max(
            min(
                np.linalg.norm([new_bbox[2] - new_bbox[0], new_bbox[3] - new_bbox[1]]),
                np.linalg.norm([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]),
            ),
            1.0,
        )
        center_limit = self._spatial_center_ratio * diag_ref
        containment_limit = self._spatial_containment_center_ratio * diag_ref
        containment_ok = (
            inter_over_smaller >= self._spatial_containment
            and center_dist <= containment_limit
        )

        if area_ratio < self._spatial_area_ratio and not containment_ok:
            return None
        if iou < self._spatial_iou and center_dist > center_limit and not containment_ok:
            return None

        center_score = max(0.0, 1.0 - center_dist / max(center_limit, 1.0))
        size_score = max(area_ratio, inter_over_smaller if containment_ok else 0.0)
        return iou, size_score, center_score

    def _recover_ids(
        self,
        confirmed_ids: set[int],
        embeddings: dict[int, np.ndarray],
        bboxes_by_tid: dict[int, np.ndarray] | None = None,
    ) -> dict[int, int]:
        """For tracks that just appeared, try to match them to lost tracks."""
        if not self._lost:
            return {}

        new_ids = sorted(confirmed_ids - set(self._active))
        if not new_ids:
            return {}

        lost_ids = sorted(self._lost)
        reward_by_pair: dict[tuple[int, int], float] = {}

        for new_tid in new_ids:
            new_feat = embeddings.get(new_tid)
            new_bbox = None if bboxes_by_tid is None else bboxes_by_tid.get(new_tid)

            for old_tid in lost_ids:
                old_feat, age, old_bbox = self._lost[old_tid]
                if self._has_active_conflict(new_tid, confirmed_ids, bboxes_by_tid, new_bbox, old_bbox):
                    continue

                dist = _feature_distance_to_bank(
                    new_feat,
                    old_feat,
                    self._lost_banks.get(old_tid),
                )

                spatial_quality = self._spatial_match_quality(new_bbox, old_bbox, age)
                quality = None
                if dist is not None and dist < self._threshold:
                    # For recently-lost tracks, also require spatial proximity.
                    # A look-alike at a different location should not match a track
                    # that disappeared only seconds ago.
                    if (
                        self._spatial_window > 0
                        and age <= self._spatial_window
                        and new_bbox is not None
                        and old_bbox is not None
                    ):
                        center_dist = _bbox_center_distance(new_bbox, old_bbox)
                        new_bbox_arr = np.asarray(new_bbox, dtype=float)
                        diag = float(np.linalg.norm([
                            new_bbox_arr[2] - new_bbox_arr[0],
                            new_bbox_arr[3] - new_bbox_arr[1],
                        ]))
                        if diag > 0 and center_dist > self._max_remap_dist_ratio * diag:
                            # Spatially impossible for this short gap — skip appearance match.
                            # Fall through to spatial-only path below.
                            pass
                        else:
                            quality = (2, self._threshold - dist, 1.0 / (age + 1))
                    else:
                        if self._long_gap_appearance_match_ok(new_bbox, old_bbox):
                            quality = (2, self._threshold - dist, 1.0 / (age + 1))
                if quality is None and spatial_quality is not None and (dist is None or dist < self._relaxed_threshold):
                    iou, area_ratio, center_score = spatial_quality
                    appearance_bonus = 0.0 if dist is None else self._relaxed_threshold - dist
                    quality = (
                        1,
                        iou + 0.25 * area_ratio + 0.10 * center_score + 0.01 * appearance_bonus,
                        1.0 / (age + 1),
                    )

                if quality is not None:
                    reward_by_pair[(new_tid, old_tid)] = _match_quality_reward(quality)

        if not reward_by_pair:
            return {}

        num_new = len(new_ids)
        num_old = len(lost_ids)
        invalid_cost = 1e9
        cost_matrix = np.full((num_new, num_old + num_new), invalid_cost, dtype=np.float64)
        cost_matrix[:, num_old:] = 0.0

        for row, new_tid in enumerate(new_ids):
            for col, old_tid in enumerate(lost_ids):
                reward = reward_by_pair.get((new_tid, old_tid))
                if reward is None:
                    continue
                cost_matrix[row, col] = -reward

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        id_remap = {}

        for row, col in zip(row_ind, col_ind):
            if col >= num_old or cost_matrix[row, col] >= 0.0:
                continue

            new_tid = new_ids[row]
            old_tid = lost_ids[col]
            old_feat, _, _ = self._lost[old_tid]
            id_remap[new_tid] = old_tid
            self._restore_lost_to_active(new_tid, old_tid, fallback_feature=old_feat)

            new_bbox = None if bboxes_by_tid is None else bboxes_by_tid.get(new_tid)
            if new_bbox is not None:
                self._last_bbox[new_tid] = np.asarray(new_bbox, dtype=float)

        return id_remap

    def _update_active(

        self,
        confirmed_ids: set[int],
        embeddings: dict[int, np.ndarray],
        id_remap: dict[int, int],
        *,
        crops_by_tid: dict[int, np.ndarray],
        update_confidences: dict[int, float] | None,
    ) -> None:
        """Update representative features using a bounded prototype bank."""
        for tid in confirmed_ids:
            accept_update, skip_reason, crop_area, confidence = self._update_acceptance_status(
                tid,
                crops_by_tid,
                update_confidences,
            )
            new_feat = embeddings.get(tid)

            if tid in self._merged_ids:
                if tid not in self._active and accept_update and new_feat is not None:
                    self._append_active_prototype(tid, new_feat)
                elif tid in self._active:
                    self._log_update_skip(tid, "merged", confidence=confidence, crop_area=crop_area)
                elif not accept_update and skip_reason is not None:
                    self._log_update_skip(tid, skip_reason, confidence=confidence, crop_area=crop_area)
                continue

            # Hold the existing feature while drift is still only a tentative signal.
            if tid in self._drift_counts and tid in self._active:
                self._log_update_skip(tid, "drift_hold", confidence=confidence, crop_area=crop_area)
                continue

            if not accept_update:
                if skip_reason is not None:
                    self._log_update_skip(tid, skip_reason, confidence=confidence, crop_area=crop_area)
                continue

            if new_feat is None:
                continue

            self._append_active_prototype(tid, new_feat)
