"""
People Counting — main entry point.

Usage:
    python main.py                              # uses config.yaml in the same folder
    python main.py --config my.yaml             # custom config path
    python main.py --source test.mp4            # override video source
    python main.py --source test.mp4 --save output/test.mp4
"""

import argparse
from collections import defaultdict
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from counter import CrosslineCounter, ZoneCounter
from debug import DebugLogger
from utils.detector import RFDETRONNXDetector, is_rfdetr_onnx_model
from utils.drawing import draw_crossline, draw_track, draw_zone
from utils.geometry import get_bbox_center, point_in_polygon, point_side_of_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_video_fps(cfg: dict, cap, fallback: float | None = None) -> float | None:
    fps_override = cfg.get("video", {}).get("fps_override")

    if fps_override is not None:
        try:
            fps = float(fps_override)
        except (TypeError, ValueError):
            sys.exit(f"Invalid video.fps_override {fps_override!r}. Must be null or a positive number.")
        if fps <= 0:
            sys.exit(f"Invalid video.fps_override {fps_override!r}. Must be > 0.")
        return fps

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps > 0:
        return fps
    return fallback


def resolve_save_path(save_path: str) -> Path:
    path = Path(save_path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".mp4")
    return path


def _is_live_source(source) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        return source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    return False


def resolve_playback_timing(cfg: dict, source, cap) -> tuple[bool, float | None]:
    if _is_live_source(source):
        print("[video] source=live  playback_sync=off")
        return False, None

    fps = resolve_video_fps(cfg, cap)
    if fps is None:
        print("[video] source=file  playback_sync=off  fps=unknown")
        return False, None

    print(f"[video] source=file  playback_sync=on  fps={fps:.3f}")
    return True, 1.0 / fps


def resolve_device(cfg: dict) -> tuple[str, bool]:
    """
    Returns (device_str, use_half).
    - Resolves "auto" to "cuda" or "cpu".
    - Forces half=False when device is CPU (FP16 unsupported on CPU).
    """
    import torch
    raw    = cfg.get("compute", {}).get("device", "auto")
    half   = cfg.get("compute", {}).get("half", False)
    device = raw if raw != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if not device.startswith("cuda"):
        half = False
    print(f"[compute] device={device}  half={half}")
    return device, half


def _is_onnx_model_path(path: str | Path | None) -> bool:
    return bool(path) and Path(path).suffix.lower() == ".onnx"


def _prefer_onnx_sibling(path: str | None) -> str | None:
    if not path:
        return path
    candidate = Path(path)
    if candidate.suffix.lower() not in {".pt", ".pth"}:
        return path
    onnx_candidate = candidate.with_suffix(".onnx")
    if onnx_candidate.exists():
        return str(onnx_candidate)
    return path


def _build_detector(detector_cfg: dict, device: str, half: bool = False, compute_cfg: dict | None = None):
    model_path = _prefer_onnx_sibling(detector_cfg["model"])

    if _is_onnx_model_path(model_path) and is_rfdetr_onnx_model(model_path):
        model = RFDETRONNXDetector(model_path, device=device, half=half, compute_cfg=compute_cfg)
        print(
            f"[Detector] RF-DETR ONNX backend: {model_path}  "
            f"device={device}  providers={list(model.providers)}"
        )
        return model, "rfdetr-onnx"

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(model_path)
    backend = "onnx" if _is_onnx_model_path(model_path) else "pytorch"
    if backend == "pytorch":
        model.to(device)
    else:
        print(f"[Detector] ONNX backend: {model_path}  device={device}")
    return model, backend


def _run_detector(model, detector_backend: str, frame, detector_cfg: dict, device: str, half: bool) -> np.ndarray:
    if detector_backend == "rfdetr-onnx":
        dets = model.predict(
            frame,
            conf=detector_cfg["confidence"],
            classes=detector_cfg.get("classes"),
            dedup_cfg=detector_cfg.get("dedup"),
        )
        return dets.astype(float, copy=False) if len(dets) else np.empty((0, 6), dtype=float)

    det_results = model(
        frame,
        conf=detector_cfg["confidence"],
        iou=detector_cfg["iou"],
        classes=detector_cfg["classes"],
        device=device,
        half=half,
        verbose=False,
    )

    raw_boxes = det_results[0].boxes
    dets_list = []
    if raw_boxes is not None and len(raw_boxes):
        for box, conf_val, cls in zip(
            raw_boxes.xyxy.cpu().numpy(),
            raw_boxes.conf.cpu().numpy(),
            raw_boxes.cls.cpu().numpy(),
        ):
            if float(conf_val) >= detector_cfg["confidence"]:
                dets_list.append([*box, float(conf_val), int(cls)])

    return np.array(dets_list, dtype=float) if dets_list else np.empty((0, 6), dtype=float)


def build_counters(cfg: dict):
    mode = cfg["counting"].get("mode", "both").lower()
    if mode not in ("line", "zone", "both"):
        sys.exit(f"Invalid counting.mode {mode!r}. Must be: line | zone | both")

    crosslines = (
        [CrosslineCounter(c) for c in cfg["counting"].get("crosslines", [])]
        if mode in ("line", "both") else []
    )
    zones = (
        [ZoneCounter(z) for z in cfg["counting"].get("zones", [])]
        if mode in ("zone", "both") else []
    )
    return crosslines, zones


def resolve_counter_retention_frames(cfg: dict, tracker_type: str) -> int:
    """Keep counter state long enough to survive short tracking gaps."""
    tcfg = cfg.get("tracker", {})
    # BoT-SORT uses track_buffer; StrongSORT / DeepSORT use max_age
    retention = int(tcfg.get("track_buffer", tcfg.get("max_age", 30)))

    if tracker_type in ("strongsort", "botsort", "deepsort", "nwojke"):
        gcfg = tcfg.get("gallery", {})
        if gcfg.get("enabled", True):
            retention = max(retention, int(gcfg.get("lifetime", retention)))
    elif tracker_type == "bytetrack":
        hcfg = tcfg.get("histogram_gallery", {})
        if hcfg.get("enabled", True):
            retention = max(retention, int(hcfg.get("lifetime", retention)))

    return max(1, retention)


def cleanup_stale_counter_tracks(
    frame_idx: int,
    active_ids: set[int],
    last_seen_frame: dict[int, int],
    counters,
    retention_frames: int,
) -> None:
    """Drop counter state only after an ID has been absent for long enough."""
    for tid in active_ids:
        last_seen_frame[tid] = frame_idx

    expired = [
        tid for tid, last_seen in last_seen_frame.items()
        if tid not in active_ids and frame_idx - last_seen > retention_frames
    ]
    for tid in expired:
        for counter in counters:
            counter.remove_track(tid)
        del last_seen_frame[tid]


def build_tracker(cfg: dict):
    """
    Returns (tracker, reid_embedder, tracker_type).

    tracker_type   — one of: "nwojke" | "botsort" | "strongsort" | "deepsort" | "bytetrack"
    tracker        — tracker instance, or None (→ use ultralytics built-in)
    reid_embedder  — OSNetEmbedder or None
    """
    tracker_type = cfg["tracker"]["type"].lower()
    device, half = resolve_device(cfg)
    compute_cfg = cfg.get("compute", {})

    if tracker_type == "nwojke":
        reid_emb = _build_reid_embedder(cfg["tracker"].get("reid", {}), device, half, compute_cfg=compute_cfg)
        return _build_nwojke_deepsort(cfg), reid_emb, "nwojke"

    if tracker_type == "botsort":
        reid_emb = _build_reid_embedder(cfg["tracker"].get("reid", {}), device, half, compute_cfg=compute_cfg)
        return _build_botsort(cfg, device, half), reid_emb, "botsort"

    if tracker_type == "strongsort":
        reid_emb = _build_reid_embedder(cfg["tracker"].get("reid", {}), device, half, compute_cfg=compute_cfg)
        return _build_strongsort(cfg, device, half), reid_emb, "strongsort"

    if tracker_type == "deepsort":
        tracker, reid_embedder = _build_deepsort(cfg, device, half, compute_cfg=compute_cfg)
        return tracker, reid_embedder, "deepsort"

    if tracker_type == "bytetrack":
        return _build_bytetrack(cfg), None, "bytetrack"

    # Unknown tracker type — fallback to ultralytics built-in
    return None, None, tracker_type


def _build_botsort(cfg: dict, device: str, half: bool):
    try:
        import torch
        from boxmot import BotSort
    except ImportError:
        sys.exit("boxmot not installed. Run: pip install boxmot")

    tcfg         = cfg["tracker"]
    reid_weights = tcfg.get("reid", {}).get("weights") or "osnet_x1_0_msmt17.pt"

    print(f"[BoT-SORT] reid={reid_weights}  device={device}  half={half}  cmc={tcfg.get('cmc_method', 'ecc')}")
    fuse_first = tcfg.get("fuse_first_associate", True)
    return BotSort(
        reid_weights     = Path(reid_weights),
        device           = torch.device(device),
        half             = half,
        track_high_thresh  = tcfg.get("track_high_thresh", 0.45),
        track_low_thresh   = tcfg.get("track_low_thresh", 0.1),
        new_track_thresh   = tcfg.get("new_track_thresh", 0.5),
        track_buffer       = tcfg.get("track_buffer", 60),
        match_thresh       = tcfg.get("match_thresh", 0.8),
        proximity_thresh   = tcfg.get("proximity_thresh", 0.5),
        appearance_thresh  = tcfg.get("appearance_thresh", 0.25),
        cmc_method         = tcfg.get("cmc_method", "ecc"),
        frame_rate         = tcfg.get("frame_rate", 30),
        fuse_first_associate = fuse_first,
        with_reid          = False,  # embeddings computed externally and passed via embs=
    )


def _build_strongsort(cfg: dict, device: str, half: bool):
    try:
        import torch
        from boxmot import StrongSort
    except ImportError:
        sys.exit("boxmot not installed. Run: pip install boxmot")

    tcfg         = cfg["tracker"]
    reid_weights = tcfg.get("reid", {}).get("weights") or "osnet_x1_0_msmt17.pt"

    print(f"[StrongSORT] reid={reid_weights}  device={device}  half={half}  ema={tcfg.get('ema_alpha', 0.9)}")
    return StrongSort(
        reid_weights = Path(reid_weights),
        device       = torch.device(device),
        half         = half,
        min_conf     = tcfg.get("min_conf", 0.3),
        max_cos_dist = tcfg.get("max_cosine_distance", 0.3),
        max_iou_dist = tcfg.get("max_iou_distance", 0.7),
        n_init       = tcfg.get("n_init", 3),
        nn_budget    = tcfg.get("nn_budget", 100),
        ema_alpha    = tcfg.get("ema_alpha", 0.9),
    )


def _build_deepsort(cfg: dict, device: str, half: bool, compute_cfg: dict | None = None):
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError:
        sys.exit("deep-sort-realtime not installed. Run: pip install deep-sort-realtime")

    tcfg          = cfg["tracker"]
    reid_embedder = _build_reid_embedder(tcfg.get("reid", {}), device, half, compute_cfg=compute_cfg)
    tracker = DeepSort(
        max_age             = tcfg.get("max_age", 60),
        n_init              = tcfg.get("n_init", 3),
        max_iou_distance    = tcfg.get("max_iou_distance", 0.7),
        max_cosine_distance = tcfg.get("max_cosine_distance", 0.3),
        nn_budget           = tcfg.get("nn_budget", 100),
        embedder            = None if reid_embedder else "mobilenet",
        bgr                 = True,
    )
    return tracker, reid_embedder


def _build_nwojke_deepsort(cfg: dict):
    """Build the original nwojke/deep_sort tracker (bundled in deep_sort/)."""
    from deep_sort.tracker import Tracker
    from deep_sort.nn_matching import NearestNeighborDistanceMetric

    tcfg = cfg["tracker"]
    max_cosine = tcfg.get("max_cosine_distance", 0.3)
    nn_budget  = tcfg.get("nn_budget", 100)
    max_iou    = tcfg.get("max_iou_distance", 0.7)
    max_age    = tcfg.get("max_age", 30)
    n_init     = tcfg.get("n_init", 3)
    new_track_thresh = cfg["detector"].get(
        "confidence_new_track",
        cfg["detector"].get("confidence", 0.0),
    )

    metric = NearestNeighborDistanceMetric("cosine", max_cosine, nn_budget)
    tracker = Tracker(
        metric,
        max_iou_distance=max_iou,
        max_age=max_age,
        n_init=n_init,
        new_track_thresh=new_track_thresh,
    )

    print(f"[nwojke DeepSORT] max_age={max_age}  n_init={n_init}  "
          f"max_cosine={max_cosine}  nn_budget={nn_budget}  "
          f"new_track_thresh={new_track_thresh}")
    return tracker


def _build_bytetrack(cfg: dict):
    try:
        from boxmot import ByteTrack
    except ImportError:
        sys.exit("boxmot not installed. Run: pip install boxmot")

    tcfg = cfg["tracker"]
    print(f"[ByteTrack] lightweight mode  track_buffer={tcfg.get('track_buffer', 60)}  "
          f"match_thresh={tcfg.get('match_thresh', 0.8)}")
    return ByteTrack(
        track_high_thresh=tcfg.get("track_high_thresh", 0.45),
        track_low_thresh=tcfg.get("track_low_thresh", 0.1),
        new_track_thresh=tcfg.get("new_track_thresh", 0.5),
        track_buffer=tcfg.get("track_buffer", 60),
        match_thresh=tcfg.get("match_thresh", 0.8),
        frame_rate=tcfg.get("frame_rate", 30),
    )


def _build_reid_embedder(
    reid_cfg: dict,
    device: str,
    half: bool,
    compute_cfg: dict | None = None,
):
    """
    Returns a Re-ID embedder based on config:
      type=clip  → CLIPReIDEmbedder (1280-D, ViT backbone, best accuracy)
      type=osnet → OSNetEmbedder    (512-D, CNN backbone, lighter)
    """
    if not reid_cfg:
        return None

    reid_type = reid_cfg.get("type", "osnet").lower()

    if reid_type == "clip":
        try:
            from reid.embedder import CLIPReIDEmbedder
            weights = reid_cfg.get("weights", "clip_market1501.pt")
            print(f"[Re-ID] CLIP-ReID embedder: {weights}  device={device}  half={half}")
            return CLIPReIDEmbedder(
                weights=weights,
                device=device,
                half=half,
            )
        except ImportError as e:
            print(f"[Re-ID] CLIP-ReID unavailable ({e}), falling back to OSNet.")
            reid_type = "osnet"

    # OSNet path
    model_name = reid_cfg.get("model", "osnet_x1_0")
    weights    = _prefer_onnx_sibling(reid_cfg.get("osnet_weights") or None)
    tta        = reid_cfg.get("tta", False)

    try:
        from reid.embedder import OSNetEmbedder, OSNetONNXEmbedder
        if _is_onnx_model_path(weights):
            embedder = OSNetONNXEmbedder(
                weights_path=weights,
                device=device,
                half=half,
                tta=tta,
                compute_cfg=compute_cfg,
            )
            print(
                f"[Re-ID] OSNet ONNX embedder: {weights}  device={device}  "
                f"tta={tta}  providers={list(getattr(embedder, 'providers', ()))}"
            )
            return embedder
        print(f"[Re-ID] OSNet embedder: {model_name}  device={device}  half={half}  tta={tta}")
        return OSNetEmbedder(
            model_name=model_name,
            weights_path=weights,
            device=device,
            half=half,
            tta=tta,
        )
    except ImportError as e:
        print(f"[Re-ID] Warning: {e}\n        Falling back to built-in embedder.")
        return None


def _extract_crops(frame: np.ndarray, boxes_xyxy: list) -> list:
    """Crop person patches from frame, clamped to frame bounds."""
    fh, fw = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(fw, int(x2)), min(fh, int(y2))
        crops.append(frame[y1:y2, x1:x2])
    return crops


class _BboxSmoother:
    """
    Per-track EMA (Exponential Moving Average) on bounding box coordinates.
    Reduces visual jitter caused by frame-to-frame detection noise.

    alpha=1.0 → no smoothing (raw Kalman output)
    alpha=0.0 → fully frozen (never updates)
    Typical: 0.5–0.7

    Area-ratio guard: if the new bbox area is very different from the
    stored state (e.g. after a merge/split or near frame exit), snap
    to the new bbox instead of blending — prevents ghost dragging.
    """
    def __init__(self, alpha: float = 0.6, snap_area_ratio: float = 0.4):
        self._alpha = alpha
        self._snap_ratio = snap_area_ratio
        self._state: dict[int, np.ndarray] = {}

    def update(self, tid: int, bbox: np.ndarray) -> np.ndarray:
        bbox = np.asarray(bbox, dtype=float)
        if tid not in self._state:
            self._state[tid] = bbox
        else:
            prev = self._state[tid]
            prev_area = max((prev[2] - prev[0]) * (prev[3] - prev[1]), 1.0)
            new_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            ratio = min(prev_area, new_area) / max(prev_area, new_area)
            if ratio < self._snap_ratio:
                # Area changed too much — snap to avoid blending stale state
                self._state[tid] = bbox
            else:
                self._state[tid] = self._alpha * bbox + (1 - self._alpha) * self._state[tid]
        return self._state[tid]

    def remove(self, tid: int) -> None:
        self._state.pop(tid, None)


def _is_valid_bbox(x1: float, y1: float, x2: float, y2: float,
                    fw: int, fh: int, min_area: float = 100.0,
                    max_edge_ratio: float = 0.8) -> bool:
    """Reject ghost/invalid bboxes from Kalman prediction.

    Returns False for:
    - inverted bbox (x1 >= x2 or y1 >= y2)
    - area smaller than min_area pixels
    - bbox center is outside the frame
    - more than max_edge_ratio of the bbox is outside the frame
    """
    if x1 >= x2 or y1 >= y2:
        return False
    w, h = x2 - x1, y2 - y1
    if w * h < min_area:
        return False
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    if cx < 0 or cx > fw or cy < 0 or cy > fh:
        return False
    # Check how much of the bbox area is outside the frame
    clipped_x1 = max(0.0, x1)
    clipped_y1 = max(0.0, y1)
    clipped_x2 = min(float(fw), x2)
    clipped_y2 = min(float(fh), y2)
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return False
    clipped_area = (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1)
    full_area = w * h
    if clipped_area / full_area < (1.0 - max_edge_ratio):
        return False
    return True


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
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


def _bbox_inter_over_smaller(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
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


def _axis_overlap_ratio(a1: float, a2: float, b1: float, b2: float) -> float:
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    denom = max(min(a2 - a1, b2 - b1), 1.0)
    return overlap / denom


def _prefer_track_candidate(candidate, current) -> bool:
    # Prefer the older, better-established track ID for duplicate collapse.
    # A fresh matched duplicate should not steal canonical ownership from a
    # long-lived stationary person just because it has time_since_update == 0.
    cand_score = (
        int(candidate.hits),
        int(candidate.age),
        candidate.time_since_update == 0,
        -int(candidate.time_since_update),
        -int(candidate.track_id),
    )
    curr_score = (
        int(current.hits),
        int(current.age),
        current.time_since_update == 0,
        -int(current.time_since_update),
        -int(current.track_id),
    )
    return cand_score > curr_score


def _track_bbox(track) -> np.ndarray | None:
    if hasattr(track, "to_tlbr"):
        return np.asarray(track.to_tlbr(), dtype=float)
    if hasattr(track, "to_ltrb"):
        return np.asarray(track.to_ltrb(), dtype=float)
    return None


def _track_latest_feature(track) -> np.ndarray | None:
    features = getattr(track, "features", None)
    if not features:
        return None
    feat = np.asarray(features[-1], dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-6:
        return None
    return feat / norm


def _feature_distance(feat_a: np.ndarray | None, feat_b: np.ndarray | None) -> float | None:
    if feat_a is None or feat_b is None:
        return None
    return 1.0 - float(np.clip(np.dot(feat_a, feat_b), -1.0, 1.0))


def _prefer_canonical_track_candidate(
    candidate,
    current,
    *,
    canonical_tid: int,
    smoother: _BboxSmoother,
    locality_ratio: float = 0.65,
    locality_margin_px: float = 12.0,
) -> bool:
    """
    Prefer the raw track that stays spatially closer to the canonical track's
    previous position. This avoids a canonical ID hopping to a farther-away
    person when duplicate raw tracks briefly coexist.
    """
    ref_bbox = smoother._state.get(canonical_tid)
    if ref_bbox is None:
        return _prefer_track_candidate(candidate, current)

    candidate_bbox = _track_bbox(candidate)
    current_bbox = _track_bbox(current)
    if candidate_bbox is None or current_bbox is None:
        return _prefer_track_candidate(candidate, current)

    ref_center = np.asarray(get_bbox_center(ref_bbox), dtype=float)
    cand_center = np.asarray(get_bbox_center(candidate_bbox), dtype=float)
    curr_center = np.asarray(get_bbox_center(current_bbox), dtype=float)
    cand_dist = float(np.linalg.norm(cand_center - ref_center))
    curr_dist = float(np.linalg.norm(curr_center - ref_center))

    diag = max(
        float(np.linalg.norm([ref_bbox[2] - ref_bbox[0], ref_bbox[3] - ref_bbox[1]])),
        1.0,
    )
    local_limit = locality_ratio * diag
    margin = max(locality_margin_px, 0.15 * diag)

    cand_local = cand_dist <= local_limit
    curr_local = curr_dist <= local_limit
    if cand_local != curr_local:
        return cand_local
    if abs(cand_dist - curr_dist) > margin:
        return cand_dist < curr_dist
    return _prefer_track_candidate(candidate, current)


def _transfer_runtime_id_state(
    from_tid: int,
    to_tid: int,
    smoother: _BboxSmoother,
    counters,
) -> None:
    if from_tid == to_tid:
        return

    # Always prefer from_tid's current physical position as the starting point
    # for the canonical ID. Keeping to_tid's stale position and blending toward
    # from_tid's location causes a visible bbox jump on the frame of the remap.
    if from_tid in smoother._state:
        smoother._state[to_tid] = smoother._state.pop(from_tid)
    # If from_tid has no state yet (spawned this frame), leave to_tid's state
    # intact — the next smoother.update() will initialise it correctly.

    for counter in counters:
        if hasattr(counter, "transfer_state"):
            counter.transfer_state(from_tid, to_tid)


def _apply_id_remap(
    id_remap: dict[int, int],
    *,
    dbg: DebugLogger,
    frame_idx: int,
    smoother: _BboxSmoother,
    counters,
    live_alias_resolver=None,
    resolve_old_tid: bool = False,
) -> dict[int, int]:
    """Canonicalize recovered IDs and transfer runtime state once."""
    if not id_remap:
        return {}

    resolved_remap: dict[int, int] = {}
    for new_tid, old_tid in id_remap.items():
        canonical_tid = old_tid
        if live_alias_resolver is not None:
            if resolve_old_tid:
                canonical_tid = live_alias_resolver.resolve(canonical_tid)
            canonical_tid = live_alias_resolver.remember_alias(new_tid, canonical_tid)
        resolved_remap[new_tid] = canonical_tid
        dbg.log_remap(frame_idx, new_tid, canonical_tid)
        _transfer_runtime_id_state(new_tid, canonical_tid, smoother, counters)

    return resolved_remap


def _resolve_runtime_tid(
    raw_tid: int,
    id_remap: dict[int, int],
    live_alias_resolver=None,
) -> int:
    """Resolve a raw tracker ID through remap and live-alias lineage."""
    tid = id_remap.get(raw_tid, raw_tid)
    if live_alias_resolver is None:
        return tid
    return live_alias_resolver.resolve(tid)


def _mark_track_lost_for_counters(
    tid: int,
    *,
    smoother: _BboxSmoother,
    counters,
    timestamp: float,
) -> None:
    """Give counters one last chance to hand off state before a track is dropped."""
    last_bbox = smoother._state.get(tid)
    last_center = get_bbox_center(last_bbox) if last_bbox is not None else None
    for counter in counters:
        counter.mark_lost(tid, last_center, timestamp)


def _update_counters_for_track(
    *,
    tid: int,
    center,
    timestamp: float,
    track_conf: float,
    conf_new_track: float,
    crosslines,
    zones,
    dbg: DebugLogger,
    frame_idx: int,
) -> tuple[int | None, bool | None]:
    """Feed counters only for sufficiently stable tracks and return debug metadata."""
    line_side = None
    in_zone = None
    if track_conf < conf_new_track:
        return line_side, in_zone

    for counter in crosslines:
        line_side = point_side_of_line(
            center,
            counter.pt1,
            counter.pt2,
            counter.buffer_px,
            counter._line_len,
        )
        result = counter.update(tid, center, timestamp)
        if result.get("entered"):
            dbg.log_counter_event(frame_idx, tid, counter.name, "enter")
        if result.get("exited"):
            dbg.log_counter_event(frame_idx, tid, counter.name, "exit")

    for counter in zones:
        in_zone = point_in_polygon(center, counter.polygon, counter._polygon_np)
        result = counter.update(tid, center, timestamp)
        if result.get("counted"):
            dbg.log_counter_event(frame_idx, tid, counter.name, "zone_counted")

    return line_side, in_zone


class _LiveTrackAliasResolver:
    """Collapse parallel raw track IDs that actually describe one person."""

    def __init__(
        self,
        iou_threshold: float = 0.55,
        containment_threshold: float = 0.82,
        horizontal_overlap_ratio: float = 0.85,
        top_edge_ratio: float = 0.28,
        center_ratio: float = 0.42,
        containment_center_ratio: float = 0.55,
        width_ratio: float = 0.75,
        appearance_distance_threshold: float = 0.20,
    ):
        self._iou_threshold = iou_threshold
        self._containment_threshold = containment_threshold
        self._horizontal_overlap_ratio = horizontal_overlap_ratio
        self._top_edge_ratio = top_edge_ratio
        self._center_ratio = center_ratio
        self._containment_center_ratio = containment_center_ratio
        self._width_ratio = width_ratio
        self._appearance_distance_threshold = appearance_distance_threshold
        self._alias: dict[int, int] = {}

    def resolve(self, tid: int) -> int:
        cur = tid
        seen = set()
        while cur in self._alias and cur not in seen:
            seen.add(cur)
            cur = self._alias[cur]
        return cur

    def remember_alias(self, source_tid: int, canonical_tid: int) -> int:
        source_tid = int(source_tid)
        canonical_tid = self.resolve(int(canonical_tid))
        source_root = self.resolve(source_tid)
        self._alias[source_tid] = canonical_tid
        if source_root != canonical_tid:
            self._alias[source_root] = canonical_tid
        return canonical_tid

    def alias_duplicates(self, tracks, protected_ids: set[int] | None = None) -> dict[int, int]:
        if len(tracks) < 2:
            return {}

        protected_ids = set() if protected_ids is None else {int(tid) for tid in protected_ids}
        ordered = sorted(
            tracks,
            key=lambda track: (-int(track.hits), -int(track.age), int(track.track_id)),
        )
        boxes = {
            int(track.track_id): np.asarray(track.to_tlbr(), dtype=float)
            for track in ordered
        }

        parents = {int(track.track_id): int(track.track_id) for track in ordered}

        def _find(tid: int) -> int:
            root = tid
            while parents[root] != root:
                root = parents[root]
            while parents[tid] != tid:
                nxt = parents[tid]
                parents[tid] = root
                tid = nxt
            return root

        def _union(left_tid: int, right_tid: int) -> None:
            left_root = _find(left_tid)
            right_root = _find(right_tid)
            if left_root == right_root:
                return
            parents[right_root] = left_root

        for idx, primary in enumerate(ordered):
            primary_tid = int(primary.track_id)
            primary_box = boxes[primary_tid]
            for secondary in ordered[idx + 1:]:
                secondary_tid = int(secondary.track_id)
                if self._looks_duplicate(primary, primary_box, secondary, boxes[secondary_tid]):
                    _union(primary_tid, secondary_tid)

        components: dict[int, list] = defaultdict(list)
        for track in ordered:
            components[_find(int(track.track_id))].append(track)

        new_aliases: dict[int, int] = {}
        for members in components.values():
            if len(members) < 2:
                continue

            keeper = members[0]
            for candidate in members[1:]:
                keeper, _ = self._pick_canonical(keeper, candidate, protected_ids)

            canonical_tid = self.resolve(int(keeper.track_id))
            for member in members:
                member_tid = int(member.track_id)
                resolved_tid = self.resolve(member_tid)
                if resolved_tid == canonical_tid:
                    continue
                self._alias[resolved_tid] = canonical_tid
                self._alias[member_tid] = canonical_tid
                new_aliases[member_tid] = canonical_tid

        return new_aliases

    def _looks_duplicate(self, track_a, box_a: np.ndarray, track_b, box_b: np.ndarray) -> bool:
        width_a = max(float(box_a[2] - box_a[0]), 1.0)
        width_b = max(float(box_b[2] - box_b[0]), 1.0)
        height_a = max(float(box_a[3] - box_a[1]), 1.0)
        height_b = max(float(box_b[3] - box_b[1]), 1.0)
        min_height = max(min(height_a, height_b), 1.0)
        iou = _bbox_iou(box_a, box_b)
        inter_over_smaller = _bbox_inter_over_smaller(box_a, box_b)

        horizontal_overlap = _axis_overlap_ratio(box_a[0], box_a[2], box_b[0], box_b[2])
        if horizontal_overlap < self._horizontal_overlap_ratio:
            return False

        center_a = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5])
        center_b = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5])
        center_dist = float(np.linalg.norm(center_a - center_b))
        feat_dist = _feature_distance(_track_latest_feature(track_a), _track_latest_feature(track_b))
        if (
            feat_dist is not None
            and feat_dist > self._appearance_distance_threshold
        ):
            return False

        if (
            inter_over_smaller >= self._containment_threshold
            and center_dist <= self._containment_center_ratio * min_height
        ):
            return True

        if iou < self._iou_threshold:
            return False
        if min(width_a, width_b) / max(width_a, width_b) < self._width_ratio:
            return False
        if abs(float(box_a[1] - box_b[1])) > self._top_edge_ratio * min_height:
            return False
        return center_dist <= self._center_ratio * min_height

    def _pick_canonical(self, track_a, track_b, protected_ids: set[int] | None = None):
        protected_ids = set() if protected_ids is None else protected_ids
        tid_a = self.resolve(int(track_a.track_id))
        tid_b = self.resolve(int(track_b.track_id))
        a_protected = tid_a in protected_ids
        b_protected = tid_b in protected_ids
        if a_protected != b_protected:
            return (track_a, track_b) if a_protected else (track_b, track_a)
        return (track_a, track_b) if _prefer_track_candidate(track_a, track_b) else (track_b, track_a)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: dict, save_path: str | None = None, display: bool = False, debug: bool = False) -> None:
    dbg = DebugLogger(enabled=debug)

    device, half = resolve_device(cfg)
    compute_cfg = cfg.get("compute", {})
    model, detector_backend = _build_detector(cfg["detector"], device, half=half, compute_cfg=compute_cfg)

    tracker, reid_embedder, tracker_type = build_tracker(cfg)
    use_custom = tracker is not None   # False → ultralytics built-in

    # Print active configuration summary
    reid_cfg = cfg["tracker"].get("reid", {})
    reid_type = reid_cfg.get("type", "none")
    reid_detail = ""
    if reid_type == "clip":
        reid_detail = f"CLIP-ReID ({reid_cfg.get('weights', 'clip_market1501.pt')})"
    elif reid_type == "osnet":
        reid_detail = f"OSNet ({reid_cfg.get('model', 'osnet_x1_0')})"
    elif tracker_type == "bytetrack":
        reid_detail = "Histogram gallery (no model)"
    print(f"\n{'='*60}")
    print(f"  Tracker : {tracker_type.upper()}")
    print(f"  Re-ID   : {reid_detail or 'none'}")
    print(f"  Detector: {cfg['detector']['model']} ({detector_backend})")
    print(f"  Device  : {device}  half={half}")
    print(f"{'='*60}\n")
    dbg.log_config(cfg, tracker_type)

    # Feature gallery for Re-ID recovery (StrongSORT / BoT-SORT / DeepSORT / nwojke path)
    gallery = None
    if tracker_type in ("strongsort", "botsort", "deepsort", "nwojke") and reid_embedder is not None:
        gcfg = cfg["tracker"].get("gallery", {})
        if gcfg.get("enabled", True):
            from reid.gallery import TrackGallery
            gallery = TrackGallery(
                embedder        = reid_embedder,
                lifetime        = gcfg.get("lifetime", 90),
                ema_alpha       = gcfg.get("ema_alpha", 0.85),
                match_threshold = gcfg.get("match_threshold", 0.28),
                drift_threshold = gcfg.get("drift_threshold", 0.6),
                merge_area_ratio = gcfg.get("merge_area_ratio", 1.5),
                debug_logger    = dbg,
            )
            print(f"[Gallery] Re-ID recovery enabled  lifetime={gallery._lifetime}  "
                  f"threshold={gallery._threshold}  merge_ratio={gallery._merge_area_ratio}")

    # Histogram gallery for lightweight Re-ID recovery (ByteTrack path)
    hist_gallery = None
    if tracker_type == "bytetrack":
        hcfg = cfg["tracker"].get("histogram_gallery", {})
        if hcfg.get("enabled", True):
            from reid.histogram_gallery import HistogramGallery
            hist_gallery = HistogramGallery(
                lifetime        = hcfg.get("lifetime", 150),
                ema_alpha       = hcfg.get("ema_alpha", 0.85),
                match_threshold = hcfg.get("match_threshold", 0.55),
                drift_threshold = hcfg.get("drift_threshold", 0.70),
                use_clahe       = hcfg.get("use_clahe", True),
                spatial_pyramid = hcfg.get("spatial_pyramid", True),
                n_strips        = hcfg.get("n_strips", 3),
                v_bins          = hcfg.get("v_bins", 32),
                v_weight        = hcfg.get("v_weight", 0.3),
            )
            print(f"[HistGallery] Histogram Re-ID recovery enabled  "
                  f"lifetime={hist_gallery._lifetime}  threshold={hist_gallery._threshold}")

    # TrackAnchor — keeps confirmed tracks alive when YOLO misses a detection
    anchor = None
    if tracker_type in ("strongsort", "botsort", "bytetrack", "deepsort", "nwojke"):
        acfg = cfg["tracker"].get("anchor", {})
        if acfg.get("enabled", True):
            from utils.anchor import TrackAnchor
            anchor = TrackAnchor(
                min_iou              = acfg.get("min_iou", 0.15),
                synthetic_conf       = acfg.get("synthetic_conf", 0.45),
                min_hits             = acfg.get("min_hits", 3),
                max_inject           = acfg.get("max_inject", 10),
                velocity_damping     = acfg.get("velocity_damping", 0.5),
                max_drift_px         = acfg.get("max_drift_px", 30.0),
                velocity_window      = acfg.get("velocity_window", 5),
                velocity_ema_alpha   = acfg.get("velocity_ema_alpha", 0.4),
                stationary_threshold = acfg.get("stationary_threshold", 0.5),
                max_lost_frames      = acfg.get("max_lost_frames", 8),
                edge_margin_px       = acfg.get("edge_margin_px", 5.0),
            )
            print(f"[Anchor] Track preservation enabled  min_iou={anchor._min_iou}  "
                  f"conf={anchor._conf}  damping={anchor._damping}  max_lost={anchor._max_lost}")

    # Post-tracker validation — catches bbox inconsistencies
    validator = None
    vcfg = cfg["tracker"].get("post_validation", {})
    if vcfg.get("enabled", True):
        from utils.validation import TrackValidator
        validator = TrackValidator(
            max_ar_change      = vcfg.get("max_ar_change", 0.3),
            max_area_change    = vcfg.get("max_area_change", 0.5),
            min_track_age      = vcfg.get("min_track_age", 5),
            confidence_penalty = vcfg.get("confidence_penalty", 0.5),
        )
        print(f"[Validator] Post-tracker validation enabled  "
              f"max_ar={validator._max_ar}  max_area={validator._max_area}  min_age={validator._min_age}")

    # Camera Motion Compensation — for non-static cameras
    cmc = None
    cmc_cfg = cfg["tracker"].get("cmc", cfg["tracker"].get("camera_motion_compensation", {}))
    if cmc_cfg.get("enabled", False):
        from utils.cmc import SimpleMotionCompensation
        cmc = SimpleMotionCompensation(
            pyr_scale = cmc_cfg.get("pyr_scale", 0.5),
            levels    = cmc_cfg.get("levels", 3),
            winsize   = cmc_cfg.get("winsize", 15),
            downsample = cmc_cfg.get("downsample", 2),
        )
        print(f"[CMC] Camera motion compensation enabled  downsample={cmc._downsample}")

    conf_new_track = cfg["detector"].get("confidence_new_track",
                                         cfg["detector"]["confidence"])

    crosslines, zones = build_counters(cfg)
    all_counters = crosslines + zones
    counter_retention_frames = resolve_counter_retention_frames(cfg, tracker_type)

    source = cfg["video"]["source"]
    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        sys.exit(f"Cannot open video source: {source!r}")

    output_path = resolve_save_path(save_path) if save_path else None
    output_fps = resolve_video_fps(cfg, cap, fallback=30.0) if output_path is not None else None
    writer = None

    disp = cfg.get("display", {})
    win = disp.get("window_name", "People Counting")
    if display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        window_size = disp.get("window_size", [1600, 900])
        if isinstance(window_size, (list, tuple)) and len(window_size) == 2:
            width, height = int(window_size[0]), int(window_size[1])
            if width > 0 and height > 0:
                try:
                    cv2.resizeWindow(win, width, height)
                except cv2.error:
                    pass
    playback_sync, frame_interval_s = resolve_playback_timing(cfg, source, cap) if display else (False, None)
    next_frame_deadline = time.perf_counter()
    smoother = _BboxSmoother(alpha=disp.get("bbox_ema_alpha", 0.6))
    live_alias_resolver = _LiveTrackAliasResolver() if use_custom else None
    counter_last_seen_frame: dict[int, int] = {}

    need_visual = display or (output_path is not None)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    prev_ids: set = set()
    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Progress log (every 100 frames when not displaying)
        if not display and frame_idx > 0 and frame_idx % 100 == 0:
            elapsed = time.perf_counter() - t_start
            fps_now = frame_idx / elapsed if elapsed > 0 else 0
            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                eta = (total_frames - frame_idx) / fps_now if fps_now > 0 else 0
                print(f"\r[progress] {frame_idx}/{total_frames} ({pct:.0f}%)  "
                      f"{fps_now:.1f} fps  ETA {eta:.0f}s", end="", flush=True)
            else:
                print(f"\r[progress] frame {frame_idx}  {fps_now:.1f} fps",
                      end="", flush=True)

        if output_path is not None and writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(output_fps),
                (fw, fh),
            )
            if not writer.isOpened():
                sys.exit(f"Cannot open output video for writing: {output_path}")
            print(f"[save] writing={output_path}  fps={float(output_fps):.3f}  size={fw}x{fh}")

        timestamp: float = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        active_ids: set = set()

        # Camera motion compensation
        camera_motion = cmc.estimate(frame) if cmc is not None else (0.0, 0.0)

        if use_custom and tracker_type in ("strongsort", "botsort"):
            # --- StrongSORT / BoT-SORT (boxmot) path -----------------------------
            dets_np = _run_detector(model, detector_backend, frame, cfg["detector"], device, half)
            xyxy_boxes = [det[:4].copy() for det in dets_np]

            # Anchor: inject synthetic detections for confirmed tracks with no coverage
            synthetic_map: dict[int, int] = {}
            if anchor is not None:
                dets_np, synthetic_map = anchor.augment(prev_ids, dets_np, frame.shape, camera_motion)
                for det_idx, tid in synthetic_map.items():
                    dbg.log_anchor_inject(frame_idx, tid, dets_np[det_idx][:4],
                                          anchor._lost_frames.get(tid, 0))

            # Build embeddings for ALL detections (real + synthetic)
            embs_np = None
            if reid_embedder is not None:
                n_total = len(dets_np)
                n_real  = len(xyxy_boxes)

                # Real detections: compute with Re-ID embedder
                real_embs = reid_embedder(_extract_crops(frame, xyxy_boxes)) if xyxy_boxes else []

                if n_total > 0:
                    feat_dim = len(real_embs[0]) if real_embs else reid_embedder.feat_dim
                    embs_list = [None] * n_total

                    for i, e in enumerate(real_embs):
                        embs_list[i] = e

                    # Synthetic detections: use gallery feature to avoid zero-norm NaN
                    for det_idx, tid in synthetic_map.items():
                        gallery_feat = gallery._active.get(tid) if gallery else None
                        if gallery_feat is not None:
                            embs_list[det_idx] = gallery_feat.tolist()
                        else:
                            # Fallback: random unit vector (never NaN)
                            v = np.random.randn(feat_dim).astype(np.float32)
                            embs_list[det_idx] = (v / np.linalg.norm(v)).tolist()

                    # Fill any remaining None slots (should not happen, but be safe)
                    for i in range(n_total):
                        if embs_list[i] is None:
                            v = np.random.randn(feat_dim).astype(np.float32)
                            embs_list[i] = (v / np.linalg.norm(v)).tolist()

                    embs_np = np.array(embs_list, dtype=float)

            tracks = tracker.update(dets_np, frame, embs=embs_np)
            # tracks: [[x1,y1,x2,y2, id, conf, cls, det_idx], ...]

            # --- Feature gallery: recover IDs after occlusion -------------------
            if gallery is not None and len(tracks):
                confirmed_ids  = {int(t[4]) for t in tracks}
                crops_by_tid   = {}
                bboxes_by_tid  = {}
                fh2, fw2       = frame.shape[:2]

                # Reuse embeddings from tracker path → avoid re-computing
                precomputed = {}
                for t in tracks:
                    tid2       = int(t[4])
                    det_idx    = int(t[7]) if len(t) > 7 else -1
                    bx1,by1,bx2,by2 = (max(0,int(t[0])), max(0,int(t[1])),
                                        min(fw2,int(t[2])), min(fh2,int(t[3])))
                    crops_by_tid[tid2] = frame[by1:by2, bx1:bx2]
                    bboxes_by_tid[tid2] = np.array([bx1, by1, bx2, by2], dtype=float)
                    if reid_embedder is not None and 0 <= det_idx < len(real_embs):
                        precomputed[tid2] = np.array(real_embs[det_idx], dtype=np.float32)

                id_remap = gallery.update(confirmed_ids, crops_by_tid,
                                          precomputed_embeddings=precomputed if precomputed else None,
                                          bboxes_by_tid=bboxes_by_tid,
                                          frame_idx=frame_idx)

                # Apply remap: transfer smoother + counter state
                id_remap = _apply_id_remap(
                    id_remap,
                    dbg=dbg,
                    frame_idx=frame_idx,
                    smoother=smoother,
                    counters=all_counters,
                    live_alias_resolver=live_alias_resolver,
                )
            else:
                id_remap = {}

            n_tracks_this_frame = len(tracks)
            dbg.log_frame_start(frame_idx, timestamp, len(dets_np), n_tracks_this_frame)

            fh, fw = frame.shape[:2]
            for t in tracks:
                raw_tid       = int(t[4])
                tid = _resolve_runtime_tid(raw_tid, id_remap, live_alias_resolver)
                track_conf    = float(t[5])
                det_idx_t     = int(t[7]) if len(t) > 7 else -1
                is_synthetic  = det_idx_t in synthetic_map
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]

                if not _is_valid_bbox(x1, y1, x2, y2, fw, fh):
                    continue

                anchor_box = np.array([x1, y1, x2, y2])
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(fw), x2), min(float(fh), y2)
                bbox   = smoother.update(tid, np.array([x1, y1, x2, y2]))
                center = get_bbox_center(bbox)
                active_ids.add(tid)

                # Post-tracker validation: penalize inconsistent bbox changes
                if validator is not None:
                    track_conf = validator.validate(tid, np.array([x1, y1, x2, y2]), track_conf)

                line_side, in_zone = _update_counters_for_track(
                    tid=tid,
                    center=center,
                    timestamp=timestamp,
                    track_conf=track_conf,
                    conf_new_track=conf_new_track,
                    crosslines=crosslines,
                    zones=zones,
                    dbg=dbg,
                    frame_idx=frame_idx,
                )

                is_merged = gallery is not None and (
                    tid in gallery._merged_ids or raw_tid in gallery._merged_ids
                )
                dbg.log_track(
                    frame_idx, timestamp, tid, raw_tid, bbox, center,
                    track_conf, is_synthetic=is_synthetic,
                    line_side=line_side, in_zone=in_zone,
                    remap_from=raw_tid if raw_tid != tid else None,
                    merged=is_merged,
                )

                # Update anchor history with unclamped bbox so velocity-based exit
                # detection works correctly (smoothed bbox never reaches fw exactly).
                if anchor is not None:
                    anchor.update(tid, anchor_box)

                if need_visual and disp.get("show_bbox", True):
                    draw_track(
                        frame, bbox, tid, center,
                        color=tuple(disp.get("bbox_color", [0, 255, 255])),
                        id_text_color=tuple(disp.get("id_text_color", [255, 255, 255])),
                        id_bg_color=tuple(disp.get("id_bg_color", [32, 32, 32])),
                        center_color=tuple(disp.get("center_color", [255, 0, 255])),
                        show_id=disp.get("show_ids", True),
                        show_center=disp.get("show_centers", True),
                        font_scale=disp.get("font_scale", 0.6),
                        thickness=disp.get("thickness", 2),
                    )

        elif use_custom and tracker_type == "deepsort":
            # --- DeepSORT path --------------------------------------------------
            dets_np = _run_detector(model, detector_backend, frame, cfg["detector"], device, half)
            deepsort_dets = []
            xyxy_boxes = []
            for det in dets_np:
                x1, y1, x2, y2, conf, cls = det
                deepsort_dets.append(([x1, y1, x2 - x1, y2 - y1], float(conf), int(cls)))
                xyxy_boxes.append(det[:4].copy())
            n_real = len(deepsort_dets)

            synthetic_map: dict[int, int] = {}
            if anchor is not None:
                dets_np, synthetic_map = anchor.augment(prev_ids, dets_np, frame.shape, camera_motion)
                for det_idx in sorted(synthetic_map):
                    sx1, sy1, sx2, sy2, sconf, scls = dets_np[det_idx]
                    deepsort_dets.append(([sx1, sy1, sx2 - sx1, sy2 - sy1], float(sconf), int(scls)))
                    xyxy_boxes.append([sx1, sy1, sx2, sy2])
                    dbg.log_anchor_inject(frame_idx, synthetic_map[det_idx],
                                          [sx1, sy1, sx2, sy2],
                                          anchor._lost_frames.get(synthetic_map[det_idx], 0))

            if reid_embedder is not None:
                feat_dim = getattr(reid_embedder, "feat_dim", 512)

                def _fallback_embed() -> list[float]:
                    vec = np.random.randn(feat_dim).astype(np.float32)
                    norm = max(float(np.linalg.norm(vec)), 1e-6)
                    return (vec / norm).tolist()

                embeds = []
                crops = _extract_crops(frame, xyxy_boxes[:n_real]) if n_real else []
                real_embs = reid_embedder(crops) if crops else []
                if real_embs:
                    feat_dim = len(real_embs[0])

                embeds.extend(real_embs)
                while len(embeds) < n_real:
                    embeds.append(_fallback_embed())

                for det_idx in sorted(synthetic_map):
                    tid = synthetic_map[det_idx]
                    gallery_feat = gallery._active.get(tid) if gallery is not None else None
                    embeds.append(gallery_feat.tolist() if gallery_feat is not None else _fallback_embed())

                while len(embeds) < len(deepsort_dets):
                    embeds.append(_fallback_embed())

                tracks = tracker.update_tracks(deepsort_dets, embeds=embeds)
            else:
                tracks = tracker.update_tracks(deepsort_dets, frame=frame)

            if gallery is not None:
                confirmed_tracks = [track for track in tracks if track.is_confirmed()]
                if confirmed_tracks:
                    confirmed_ids = {int(track.track_id) for track in confirmed_tracks}
                    crops_by_tid = {}
                    bboxes_by_tid = {}
                    fh2, fw2 = frame.shape[:2]
                    # Build precomputed embeddings from tracker's real_embs
                    track_embeddings = {}
                    for track in confirmed_tracks:
                        tid2 = int(track.track_id)
                        bx1, by1, bx2, by2 = track.to_ltrb()
                        bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                        bx2, by2 = min(fw2, int(bx2)), min(fh2, int(by2))
                        crops_by_tid[tid2] = frame[by1:by2, bx1:bx2]
                        bboxes_by_tid[tid2] = np.array([bx1, by1, bx2, by2], dtype=float)
                        # Map detection index to track embedding if available
                        det_idx = getattr(track, 'det_idx', None)
                        if det_idx is not None and det_idx < len(real_embs):
                            track_embeddings[tid2] = np.array(real_embs[det_idx], dtype=np.float32)
                    id_remap = gallery.update(
                        confirmed_ids, crops_by_tid,
                        precomputed_embeddings=track_embeddings if track_embeddings else None,
                        bboxes_by_tid=bboxes_by_tid,
                        frame_idx=frame_idx,
                    )

                    id_remap = _apply_id_remap(
                        id_remap,
                        dbg=dbg,
                        frame_idx=frame_idx,
                        smoother=smoother,
                        counters=all_counters,
                        live_alias_resolver=live_alias_resolver,
                    )
                else:
                    id_remap = {}
            else:
                id_remap = {}

            confirmed_tracks = [track for track in tracks if track.is_confirmed()]
            dbg.log_frame_start(frame_idx, timestamp, len(dets_np), len(confirmed_tracks))

            fh, fw = frame.shape[:2]
            for track in tracks:
                if not track.is_confirmed():
                    continue
                raw_tid = int(track.track_id)
                tid = _resolve_runtime_tid(raw_tid, id_remap, live_alias_resolver)
                det_conf = track.get_det_conf()
                track_conf = float(det_conf if det_conf is not None else 0.5)
                x1, y1, x2, y2 = track.to_ltrb()

                if not _is_valid_bbox(x1, y1, x2, y2, fw, fh):
                    continue

                anchor_box = np.array([x1, y1, x2, y2])
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(fw), x2), min(float(fh), y2)
                bbox = smoother.update(tid, np.array([x1, y1, x2, y2]))
                center = get_bbox_center(bbox)
                active_ids.add(tid)

                if validator is not None:
                    track_conf = validator.validate(tid, np.array([x1, y1, x2, y2]), track_conf)

                line_side, in_zone = _update_counters_for_track(
                    tid=tid,
                    center=center,
                    timestamp=timestamp,
                    track_conf=track_conf,
                    conf_new_track=conf_new_track,
                    crosslines=crosslines,
                    zones=zones,
                    dbg=dbg,
                    frame_idx=frame_idx,
                )

                is_merged = gallery is not None and (
                    tid in gallery._merged_ids or raw_tid in gallery._merged_ids
                )
                dbg.log_track(
                    frame_idx, timestamp, tid, raw_tid, bbox, center,
                    track_conf, is_synthetic=False,
                    line_side=line_side, in_zone=in_zone,
                    remap_from=raw_tid if raw_tid != tid else None,
                    merged=is_merged,
                )

                if anchor is not None:
                    anchor.update(tid, anchor_box)

                if need_visual and disp.get("show_bbox", True):
                    draw_track(
                        frame, bbox, tid, center,
                        color=tuple(disp.get("bbox_color", [0, 255, 255])),
                        id_text_color=tuple(disp.get("id_text_color", [255, 255, 255])),
                        id_bg_color=tuple(disp.get("id_bg_color", [32, 32, 32])),
                        center_color=tuple(disp.get("center_color", [255, 0, 255])),
                        show_id=disp.get("show_ids", True),
                        show_center=disp.get("show_centers", True),
                        font_scale=disp.get("font_scale", 0.6),
                        thickness=disp.get("thickness", 2),
                    )

        elif use_custom and tracker_type == "bytetrack":
            # --- ByteTrack (boxmot) — lightweight, no Re-ID model ---------------
            dets_np = _run_detector(model, detector_backend, frame, cfg["detector"], device, half)

            # Anchor: inject synthetic detections for confirmed tracks with no coverage
            synthetic_map_bt: dict[int, int] = {}
            if anchor is not None:
                dets_np, synthetic_map_bt = anchor.augment(prev_ids, dets_np, frame.shape, camera_motion)
                for det_idx, tid in synthetic_map_bt.items():
                    dbg.log_anchor_inject(frame_idx, tid, dets_np[det_idx][:4],
                                          anchor._lost_frames.get(tid, 0))

            tracks = tracker.update(dets_np, frame)
            # tracks: [[x1,y1,x2,y2, id, conf, cls, det_idx], ...]

            # Histogram gallery: recover IDs after occlusion
            if hist_gallery is not None and len(tracks):
                confirmed_ids = {int(t[4]) for t in tracks}
                crops_by_tid = {}
                fh2, fw2 = frame.shape[:2]
                for t in tracks:
                    tid2 = int(t[4])
                    bx1, by1, bx2, by2 = (
                        max(0, int(t[0])), max(0, int(t[1])),
                        min(fw2, int(t[2])), min(fh2, int(t[3])),
                    )
                    crops_by_tid[tid2] = frame[by1:by2, bx1:bx2]
                id_remap = hist_gallery.update(confirmed_ids, crops_by_tid)

                id_remap = _apply_id_remap(
                    id_remap,
                    dbg=dbg,
                    frame_idx=frame_idx,
                    smoother=smoother,
                    counters=all_counters,
                    live_alias_resolver=live_alias_resolver,
                )
            else:
                id_remap = {}

            dbg.log_frame_start(frame_idx, timestamp, len(dets_np), len(tracks))

            fh, fw = frame.shape[:2]
            for t in tracks:
                raw_tid = int(t[4])
                tid = _resolve_runtime_tid(raw_tid, id_remap, live_alias_resolver)
                track_conf = float(t[5])
                det_idx_t = int(t[7]) if len(t) > 7 else -1
                is_synthetic = det_idx_t in synthetic_map_bt
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]

                if not _is_valid_bbox(x1, y1, x2, y2, fw, fh):
                    continue

                anchor_box = np.array([x1, y1, x2, y2])
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(fw), x2), min(float(fh), y2)
                bbox = smoother.update(tid, np.array([x1, y1, x2, y2]))
                center = get_bbox_center(bbox)
                active_ids.add(tid)

                if validator is not None:
                    track_conf = validator.validate(tid, np.array([x1, y1, x2, y2]), track_conf)

                line_side, in_zone = _update_counters_for_track(
                    tid=tid,
                    center=center,
                    timestamp=timestamp,
                    track_conf=track_conf,
                    conf_new_track=conf_new_track,
                    crosslines=crosslines,
                    zones=zones,
                    dbg=dbg,
                    frame_idx=frame_idx,
                )

                dbg.log_track(
                    frame_idx, timestamp, tid, raw_tid, bbox, center,
                    track_conf, is_synthetic=is_synthetic,
                    line_side=line_side, in_zone=in_zone,
                    remap_from=raw_tid if raw_tid != tid else None,
                    merged=False,
                )

                if anchor is not None:
                    anchor.update(tid, anchor_box)

                if need_visual and disp.get("show_bbox", True):
                    draw_track(
                        frame, bbox, tid, center,
                        color=tuple(disp.get("bbox_color", [0, 255, 255])),
                        id_text_color=tuple(disp.get("id_text_color", [255, 255, 255])),
                        id_bg_color=tuple(disp.get("id_bg_color", [32, 32, 32])),
                        center_color=tuple(disp.get("center_color", [255, 0, 255])),
                        show_id=disp.get("show_ids", True),
                        show_center=disp.get("show_centers", True),
                        font_scale=disp.get("font_scale", 0.6),
                        thickness=disp.get("thickness", 2),
                    )

        elif use_custom and tracker_type == "nwojke":
            # --- nwojke/deep_sort — original DeepSORT implementation ------------
            # Key advantage: time_since_update lets us skip Kalman ghost tracks.
            from deep_sort.detection import Detection as DSDetection

            dets_np = _run_detector(model, detector_backend, frame, cfg["detector"], device, half)
            xyxy_boxes = [det[:4].copy() for det in dets_np]
            n_real = len(xyxy_boxes)

            # Anchor: inject synthetic detections for confirmed tracks with no coverage
            nwojke_synthetic_map: dict[int, int] = {}
            if anchor is not None:
                dets_np, nwojke_synthetic_map = anchor.augment(
                    prev_ids, dets_np, frame.shape, camera_motion)
                for det_idx, s_tid in nwojke_synthetic_map.items():
                    sx1, sy1, sx2, sy2 = dets_np[det_idx][:4]
                    xyxy_boxes.append([sx1, sy1, sx2, sy2])
                    dbg.log_anchor_inject(frame_idx, s_tid, [sx1, sy1, sx2, sy2],
                                         anchor._lost_frames.get(s_tid, 0))

            # Build Re-ID embeddings
            feat_dim = getattr(reid_embedder, "feat_dim", 512) if reid_embedder else 512

            def _nwojke_fallback():
                v = np.random.randn(feat_dim).astype(np.float32)
                return v / max(float(np.linalg.norm(v)), 1e-6)

            if reid_embedder is not None and xyxy_boxes:
                real_crops = _extract_crops(frame, xyxy_boxes[:n_real])
                real_embs = reid_embedder(real_crops) if real_crops else []
                if real_embs:
                    feat_dim = len(real_embs[0])
                all_embs = list(real_embs)
                while len(all_embs) < n_real:
                    all_embs.append(_nwojke_fallback().tolist())
                for det_idx in range(n_real, len(xyxy_boxes)):
                    s_tid = nwojke_synthetic_map.get(det_idx)
                    gf = gallery._active.get(s_tid) if (gallery and s_tid) else None
                    all_embs.append(gf.tolist() if gf is not None else _nwojke_fallback().tolist())
            else:
                real_embs = []
                all_embs = [_nwojke_fallback().tolist() for _ in xyxy_boxes]

            # Build Detection objects for nwojke tracker
            nwojke_dets = []
            for i, box in enumerate(xyxy_boxes):
                x1, y1, x2, y2 = box[:4]
                tlwh = [x1, y1, x2 - x1, y2 - y1]
                conf_val = float(dets_np[i][4]) if i < len(dets_np) else 0.4
                feat = np.asarray(all_embs[i], dtype=np.float32)
                nwojke_dets.append(DSDetection(tlwh, conf_val, feat))

            # Run tracker: predict → update
            tracker.predict()
            tracker.update(nwojke_dets)

            # Gallery: update with confirmed tracks
            nwojke_id_remap: dict[int, int] = {}
            if gallery is not None:
                confirmed_tracks_nw = [t for t in tracker.tracks if t.is_confirmed()]
                if confirmed_tracks_nw:
                    fh2, fw2 = frame.shape[:2]
                    confirmed_ids_nw = {t.track_id for t in confirmed_tracks_nw}
                    crops_by_tid_nw: dict[int, np.ndarray] = {}
                    bboxes_by_tid_nw: dict[int, np.ndarray] = {}
                    precomp_nw: dict[int, np.ndarray] = {}
                    for t in confirmed_tracks_nw:
                        tlbr = t.to_tlbr()
                        bx1 = max(0, int(tlbr[0])); by1 = max(0, int(tlbr[1]))
                        bx2 = min(fw2, int(tlbr[2])); by2 = min(fh2, int(tlbr[3]))
                        crops_by_tid_nw[t.track_id] = frame[by1:by2, bx1:bx2]
                        bboxes_by_tid_nw[t.track_id] = np.array([bx1, by1, bx2, by2], dtype=float)
                        if t.features:
                            precomp_nw[t.track_id] = np.asarray(t.features[-1], dtype=np.float32)
                    nwojke_id_remap = gallery.update(
                        confirmed_ids_nw, crops_by_tid_nw,
                        precomputed_embeddings=precomp_nw if precomp_nw else None,
                        bboxes_by_tid=bboxes_by_tid_nw,
                        frame_idx=frame_idx,
                    )
                    nwojke_id_remap = _apply_id_remap(
                        nwojke_id_remap,
                        dbg=dbg,
                        frame_idx=frame_idx,
                        smoother=smoother,
                        counters=all_counters,
                        live_alias_resolver=live_alias_resolver,
                        resolve_old_tid=True,
                    )

            # max display age: tracks with time_since_update > this are hidden
            # 0 = only matched-this-frame tracks (no ghost)
            # 1 = tolerate 1 missed frame (reduces flicker on brief occlusions)
            max_display_age = cfg["tracker"].get("max_display_age", 1)

            fh, fw = frame.shape[:2]
            active_tracks_nw = [
                t for t in tracker.tracks
                if t.is_confirmed() and t.time_since_update <= max_display_age
            ]
            if live_alias_resolver is not None and len(active_tracks_nw) > 1:
                protected_ids = set(nwojke_id_remap.values())
                live_alias_remap = live_alias_resolver.alias_duplicates(
                    active_tracks_nw,
                    protected_ids=protected_ids,
                )
                for duplicate_tid, canonical_tid in live_alias_remap.items():
                    dbg.log_remap(frame_idx, duplicate_tid, canonical_tid)
                    _transfer_runtime_id_state(
                        duplicate_tid,
                        canonical_tid,
                        smoother,
                        all_counters,
                    )

            if live_alias_resolver is not None and active_tracks_nw:
                selected_tracks = {}
                for track in active_tracks_nw:
                    raw_tid = int(track.track_id)
                    canonical_tid = _resolve_runtime_tid(
                        raw_tid,
                        nwojke_id_remap,
                        live_alias_resolver,
                    )
                    current = selected_tracks.get(canonical_tid)
                    if current is None or _prefer_canonical_track_candidate(
                        track,
                        current,
                        canonical_tid=canonical_tid,
                        smoother=smoother,
                    ):
                        selected_tracks[canonical_tid] = track
                active_tracks_nw = list(selected_tracks.values())

            dbg.log_frame_start(frame_idx, timestamp, len(dets_np), len(active_tracks_nw))

            for track in active_tracks_nw:
                raw_tid = track.track_id
                tid = _resolve_runtime_tid(raw_tid, nwojke_id_remap, live_alias_resolver)

                tlbr = track.to_tlbr()
                x1, y1, x2, y2 = tlbr[0], tlbr[1], tlbr[2], tlbr[3]

                if not _is_valid_bbox(x1, y1, x2, y2, fw, fh):
                    continue

                anchor_box = np.array([x1, y1, x2, y2])
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(fw), x2), min(float(fh), y2)
                bbox = smoother.update(tid, np.array([x1, y1, x2, y2]))
                center = get_bbox_center(bbox)
                active_ids.add(tid)

                # Confidence from hits ratio (nwojke track has no conf directly)
                track_conf = min(1.0, track.hits / max(track._n_init, 1)) * 0.9 + 0.1
                if validator is not None:
                    track_conf = validator.validate(tid, np.array([x1, y1, x2, y2]), track_conf)

                line_side, in_zone = _update_counters_for_track(
                    tid=tid,
                    center=center,
                    timestamp=timestamp,
                    track_conf=track_conf,
                    conf_new_track=conf_new_track,
                    crosslines=crosslines,
                    zones=zones,
                    dbg=dbg,
                    frame_idx=frame_idx,
                )

                is_merged = gallery is not None and (
                    tid in gallery._merged_ids or raw_tid in gallery._merged_ids
                )
                dbg.log_track(
                    frame_idx, timestamp, tid, raw_tid, bbox, center,
                    track_conf, is_synthetic=(raw_tid in nwojke_synthetic_map.values()),
                    line_side=line_side, in_zone=in_zone,
                    remap_from=raw_tid if raw_tid != tid else None,
                    merged=is_merged,
                )

                if anchor is not None:
                    anchor.update(tid, anchor_box)

                if need_visual and disp.get("show_bbox", True):
                    draw_track(
                        frame, bbox, tid, center,
                        color=tuple(disp.get("bbox_color", [0, 255, 255])),
                        id_text_color=tuple(disp.get("id_text_color", [255, 255, 255])),
                        id_bg_color=tuple(disp.get("id_bg_color", [32, 32, 32])),
                        center_color=tuple(disp.get("center_color", [255, 0, 255])),
                        show_id=disp.get("show_ids", True),
                        show_center=disp.get("show_centers", True),
                        font_scale=disp.get("font_scale", 0.6),
                        thickness=disp.get("thickness", 2),
                    )

        else:
            # --- Fallback: ultralytics built-in tracker -------------------------
            if detector_backend == "rfdetr-onnx":
                sys.exit(
                    "RF-DETR ONNX detector is supported only with custom trackers "
                    "(nwojke, deepsort, strongsort, botsort, bytetrack)."
                )
            results = model.track(
                frame,
                persist=True,
                conf=cfg["detector"]["confidence"],
                iou=cfg["detector"]["iou"],
                classes=cfg["detector"]["classes"],
                tracker=f"{cfg['tracker']['type']}.yaml",
                device=device,
                half=half,
                verbose=False,
            )

            boxes_obj = results[0].boxes
            if boxes_obj is not None and boxes_obj.id is not None:
                for bbox_t, id_t in zip(boxes_obj.xyxy, boxes_obj.id):
                    tid = int(id_t.item())
                    bbox = smoother.update(tid, bbox_t.cpu().numpy())
                    center = get_bbox_center(bbox)
                    active_ids.add(tid)

                    for counter in crosslines:
                        counter.update(tid, center, timestamp)
                    for counter in zones:
                        counter.update(tid, center, timestamp)

                    if disp.get("show_bbox", True):
                        draw_track(
                            frame, bbox, tid, center,
                            color=tuple(disp.get("bbox_color", [0, 255, 255])),
                            id_text_color=tuple(disp.get("id_text_color", [255, 255, 255])),
                            id_bg_color=tuple(disp.get("id_bg_color", [32, 32, 32])),
                            center_color=tuple(disp.get("center_color", [255, 0, 255])),
                            show_id=disp.get("show_ids", True),
                            show_center=disp.get("show_centers", True),
                            font_scale=disp.get("font_scale", 0.6),
                            thickness=disp.get("thickness", 2),
                        )

        # Clean up per-track state for lost tracks
        lost = prev_ids - active_ids
        for tid in lost:
            dbg.log_lost(frame_idx, tid)
            _mark_track_lost_for_counters(
                tid,
                smoother=smoother,
                counters=all_counters,
                timestamp=timestamp,
            )
            smoother.remove(tid)
            if anchor is not None:
                anchor.remove(tid)
            if hist_gallery is not None:
                hist_gallery.remove(tid)
            if validator is not None:
                validator.remove(tid)

        cleanup_stale_counter_tracks(
            frame_idx,
            active_ids,
            counter_last_seen_frame,
            all_counters,
            counter_retention_frames,
        )
        prev_ids = active_ids
        frame_idx += 1

        if need_visual:
            # Draw lines
            for c in crosslines:
                counts = c.get_counts()
                draw_crossline(
                    frame, c.pt1, c.pt2,
                    counts["in"], counts["out"], c.name,
                    color=tuple(disp.get("line_color", [0, 255, 0])),
                    text_color=tuple(disp.get("count_text_color", [255, 255, 255])),
                    label_bg_color=tuple(disp.get("line_label_bg_color", [40, 96, 40])),
                    font_scale=disp.get("font_scale", 0.6),
                    thickness=disp.get("thickness", 2),
                )

            # Draw zones
            for z in zones:
                counts = z.get_counts()
                draw_zone(
                    frame, z.polygon, counts["in"], z.name,
                    color=tuple(disp.get("zone_color", [0, 165, 255])),
                    text_color=tuple(disp.get("count_text_color", [255, 255, 255])),
                    label_bg_color=tuple(disp.get("zone_label_bg_color", [0, 96, 176])),
                    font_scale=disp.get("font_scale", 0.6),
                    thickness=disp.get("thickness", 2),
                )

            if writer is not None:
                writer.write(frame)

        if display:
            cv2.imshow(win, frame)

            wait_ms = 1
            if playback_sync and frame_interval_s is not None:
                target_time = next_frame_deadline + frame_interval_s
                now = time.perf_counter()
                if target_time > now:
                    wait_ms = max(1, int(round((target_time - now) * 1000)))
                    next_frame_deadline = target_time
                else:
                    next_frame_deadline = now

            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    elapsed_total = time.perf_counter() - t_start
    if not display and frame_idx > 0:
        print()  # newline after \r progress
    print(f"[done] {frame_idx} frames in {elapsed_total:.1f}s "
          f"({frame_idx / elapsed_total:.1f} fps)")

    for c in crosslines:
        counts = c.get_counts()
        print(f"  {c.name}: IN={counts['in']}  OUT={counts['out']}")
    for z in zones:
        counts = z.get_counts()
        print(f"  {z.name}: IN={counts['in']}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[save] output saved: {output_path}")
    if display:
        cv2.destroyAllWindows()
    dbg.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auto_save_path(source) -> str:
    """Generate output path: output/<source_name>_<timestamp>.mp4"""
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if isinstance(source, int):
        name = f"cam{source}"
    else:
        name = Path(source).stem
    return str(Path("output") / f"{name}_{ts}.mp4")


def resolve_run_options(cfg: dict, args) -> tuple[str | None, bool, bool]:
    runtime_cfg = cfg.get("runtime", {})
    output_cfg = cfg.get("output", {})

    display = bool(runtime_cfg.get("display", False)) if args.display is None else bool(args.display)
    debug = bool(runtime_cfg.get("debug", False)) if args.debug is None else bool(args.debug)

    if args.save is not None:
        save_path = args.save
    elif args.no_save is True:
        save_path = None
    else:
        should_save = bool(output_cfg.get("save", True))
        configured_path = output_cfg.get("path")
        save_path = configured_path if should_save and configured_path else (
            _auto_save_path(cfg["video"]["source"]) if should_save else None
        )

    return save_path, display, debug


def parse_args():
    parser = argparse.ArgumentParser(description="People Counting")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--source", default=None,
                        help="Override video source (path or webcam index)")
    parser.add_argument("--save", default=None,
                        help="Save path for output video (default: config output.path or auto-generated)")
    parser.add_argument("--no-save", action="store_true", default=None,
                        help="Disable saving output video")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=None,
                        help="Override preview window setting from config")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None,
                        help="Override debug logging setting from config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        try:
            cfg["video"]["source"] = int(args.source)
        except ValueError:
            cfg["video"]["source"] = args.source

    save_path, display, debug = resolve_run_options(cfg, args)
    run(cfg, save_path=save_path, display=display, debug=debug)
