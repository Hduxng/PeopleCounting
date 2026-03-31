"""
People Counting — main entry point.

Usage:
    python main.py                              # uses config.yaml in the same folder
    python main.py --config my.yaml             # custom config path
    python main.py --source test.mp4            # override video source
    python main.py --source test.mp4 --save output/test.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from counter import CrosslineCounter, ZoneCounter
from utils.drawing import draw_crossline, draw_track, draw_zone
from utils.geometry import get_bbox_center


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
    retention = int(tcfg.get("max_age", 30))

    if tracker_type == "strongsort":
        gcfg = tcfg.get("gallery", {})
        if gcfg.get("enabled", True):
            retention = max(retention, int(gcfg.get("lifetime", retention)))

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

    tracker_type   — one of: "deepsort" | "strongsort" | "bytetrack" | "botsort"
    tracker        — tracker instance, or None (→ use ultralytics built-in)
    reid_embedder  — OSNetEmbedder or None
    """
    tracker_type = cfg["tracker"]["type"].lower()
    device, half = resolve_device(cfg)

    if tracker_type == "strongsort":
        reid_emb = _build_reid_embedder(cfg["tracker"].get("reid", {}), device, half)
        return _build_strongsort(cfg, device, half), reid_emb, "strongsort"

    if tracker_type == "deepsort":
        tracker, reid_embedder = _build_deepsort(cfg, device)
        return tracker, reid_embedder, "deepsort"

    # bytetrack / botsort — ultralytics built-in
    return None, None, tracker_type


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


def _build_deepsort(cfg: dict, device: str):
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError:
        sys.exit("deep-sort-realtime not installed. Run: pip install deep-sort-realtime")

    tcfg          = cfg["tracker"]
    reid_embedder = _build_reid_embedder(tcfg.get("reid", {}), device, half=False)
    tracker = DeepSort(
        max_age             = tcfg.get("max_age", 50),
        n_init              = tcfg.get("n_init", 3),
        max_iou_distance    = tcfg.get("max_iou_distance", 0.7),
        max_cosine_distance = tcfg.get("max_cosine_distance", 0.35),
        nn_budget           = tcfg.get("nn_budget", 150),
        embedder            = None if reid_embedder else "mobilenet",
        bgr                 = True,
    )
    return tracker, reid_embedder


def _build_reid_embedder(reid_cfg: dict, device: str, half: bool):
    """
    Returns an OSNetEmbedder if reid config is present, else None.
    """
    if not reid_cfg:
        return None

    model_name = reid_cfg.get("model", "osnet_x1_0")
    weights    = reid_cfg.get("weights") or None
    tta        = reid_cfg.get("tta", False)

    try:
        from reid.embedder import OSNetEmbedder
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
    """
    def __init__(self, alpha: float = 0.6):
        self._alpha = alpha
        self._state: dict[int, np.ndarray] = {}

    def update(self, tid: int, bbox: np.ndarray) -> np.ndarray:
        bbox = np.asarray(bbox, dtype=float)
        if tid not in self._state:
            self._state[tid] = bbox
        else:
            self._state[tid] = self._alpha * bbox + (1 - self._alpha) * self._state[tid]
        return self._state[tid]

    def remove(self, tid: int) -> None:
        self._state.pop(tid, None)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(cfg: dict, save_path: str | None = None) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    device, half = resolve_device(cfg)
    model = YOLO(cfg["detector"]["model"])
    model.to(device)

    tracker, reid_embedder, tracker_type = build_tracker(cfg)
    use_custom = tracker is not None   # False → ultralytics built-in

    # Feature gallery for Re-ID recovery (StrongSORT path only)
    gallery = None
    if tracker_type == "strongsort" and reid_embedder is not None:
        gcfg = cfg["tracker"].get("gallery", {})
        if gcfg.get("enabled", True):
            from reid.gallery import TrackGallery
            gallery = TrackGallery(
                embedder        = reid_embedder,
                lifetime        = gcfg.get("lifetime", 90),
                ema_alpha       = gcfg.get("ema_alpha", 0.85),
                match_threshold = gcfg.get("match_threshold", 0.28),
            )
            print(f"[Gallery] Re-ID recovery enabled  lifetime={gallery._lifetime}  threshold={gallery._threshold}")

    # TrackAnchor — keeps confirmed tracks alive when YOLO misses a detection
    anchor = None
    if tracker_type == "strongsort":
        acfg = cfg["tracker"].get("anchor", {})
        if acfg.get("enabled", True):
            from utils.anchor import TrackAnchor
            anchor = TrackAnchor(
                min_iou        = acfg.get("min_iou", 0.15),
                synthetic_conf = acfg.get("synthetic_conf", 0.45),
                min_hits       = acfg.get("min_hits", 3),
                max_inject     = acfg.get("max_inject", 10),
            )
            print(f"[Anchor] Track preservation enabled  min_iou={anchor._min_iou}  conf={anchor._conf}")

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
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    window_size = disp.get("window_size", [1600, 900])
    if isinstance(window_size, (list, tuple)) and len(window_size) == 2:
        width, height = int(window_size[0]), int(window_size[1])
        if width > 0 and height > 0:
            try:
                cv2.resizeWindow(win, width, height)
            except cv2.error:
                pass
    playback_sync, frame_interval_s = resolve_playback_timing(cfg, source, cap)
    next_frame_deadline = time.perf_counter()
    smoother = _BboxSmoother(alpha=disp.get("bbox_ema_alpha", 0.6))
    counter_last_seen_frame: dict[int, int] = {}

    prev_ids: set = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        if use_custom and tracker_type == "strongsort":
            # --- StrongSORT (boxmot) path ---------------------------------------
            det_results = model(
                frame,
                conf=cfg["detector"]["confidence"],
                iou=cfg["detector"]["iou"],
                classes=cfg["detector"]["classes"],
                device=device,
                half=half,
                verbose=False,
            )

            raw_boxes = det_results[0].boxes
            xyxy_boxes = []
            dets_list  = []
            if raw_boxes is not None and len(raw_boxes):
                for box, conf, cls in zip(
                    raw_boxes.xyxy.cpu().numpy(),
                    raw_boxes.conf.cpu().numpy(),
                    raw_boxes.cls.cpu().numpy(),
                ):
                    c = float(conf)
                    # Two-stage: keep all detections for association, but mark
                    # low-confidence ones so the tracker won't start new tracks from them.
                    # StrongSORT respects min_conf for new track creation.
                    if c >= cfg["detector"]["confidence"]:
                        dets_list.append([*box, c, int(cls)])
                        xyxy_boxes.append(box)

            dets_np = np.array(dets_list, dtype=float) if dets_list else np.empty((0, 6))

            # Anchor: inject synthetic detections for confirmed tracks with no coverage
            synthetic_map: dict[int, int] = {}
            if anchor is not None:
                dets_np, synthetic_map = anchor.augment(prev_ids, dets_np, frame.shape)

            # Build embeddings for ALL detections (real + synthetic)
            embs_np = None
            if reid_embedder is not None:
                n_total = len(dets_np)
                n_real  = len(xyxy_boxes)

                # Real detections: compute with OSNetEmbedder
                real_embs = reid_embedder(_extract_crops(frame, xyxy_boxes)) if xyxy_boxes else []

                if n_total > 0:
                    feat_dim = len(real_embs[0]) if real_embs else 512
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
                fh2, fw2       = frame.shape[:2]
                for t in tracks:
                    tid2       = int(t[4])
                    bx1,by1,bx2,by2 = (max(0,int(t[0])), max(0,int(t[1])),
                                        min(fw2,int(t[2])), min(fh2,int(t[3])))
                    crops_by_tid[tid2] = frame[by1:by2, bx1:bx2]
                id_remap = gallery.update(confirmed_ids, crops_by_tid)

                # Apply remap: transfer smoother state old→new so display stays smooth
                for new_tid, old_tid in id_remap.items():
                    if new_tid in smoother._state and old_tid not in smoother._state:
                        smoother._state[old_tid] = smoother._state.pop(new_tid)
                    elif new_tid in smoother._state:
                        smoother._state.pop(new_tid)
            else:
                id_remap = {}

            fh, fw = frame.shape[:2]
            for t in tracks:
                raw_tid       = int(t[4])
                tid           = id_remap.get(raw_tid, raw_tid)
                track_conf    = float(t[5])
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                anchor_box = np.array([x1, y1, x2, y2])  # unclamped for exit detection
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(fw), x2), min(float(fh), y2)
                bbox   = smoother.update(tid, np.array([x1, y1, x2, y2]))
                center = get_bbox_center(bbox)
                active_ids.add(tid)

                # Two-stage: only feed counters from high-confidence detections
                if track_conf >= conf_new_track:
                    for counter in crosslines:
                        counter.update(tid, center, timestamp)
                    for counter in zones:
                        counter.update(tid, center, timestamp)

                # Update anchor history with unclamped bbox so velocity-based exit
                # detection works correctly (smoothed bbox never reaches fw exactly).
                if anchor is not None:
                    anchor.update(tid, anchor_box)

                if disp.get("show_bbox", True):
                    draw_track(
                        frame, bbox, tid, center,
                        color=tuple(disp.get("bbox_color", [0, 255, 255])),
                        show_id=disp.get("show_ids", True),
                        show_center=disp.get("show_centers", True),
                        thickness=disp.get("thickness", 2),
                    )

        elif use_custom and tracker_type == "deepsort":
            # --- DeepSORT path --------------------------------------------------
            det_results = model(
                frame,
                conf=cfg["detector"]["confidence"],
                iou=cfg["detector"]["iou"],
                classes=cfg["detector"]["classes"],
                device=device,
                half=half,
                verbose=False,
            )

            raw_boxes = det_results[0].boxes
            deepsort_dets = []
            xyxy_boxes    = []
            if raw_boxes is not None:
                for box, conf, cls in zip(
                    raw_boxes.xyxy.cpu().numpy(),
                    raw_boxes.conf.cpu().numpy(),
                    raw_boxes.cls.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = box
                    deepsort_dets.append(([x1, y1, x2-x1, y2-y1], float(conf), int(cls)))
                    xyxy_boxes.append(box)

            if reid_embedder is not None:
                crops  = _extract_crops(frame, xyxy_boxes) if xyxy_boxes else []
                embeds = reid_embedder(crops) if crops else []
                tracks = tracker.update_tracks(deepsort_dets, embeds=embeds)
            else:
                tracks = tracker.update_tracks(deepsort_dets, frame=frame)

            fh, fw = frame.shape[:2]
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = int(track.track_id)
                x1, y1, x2, y2 = track.to_ltrb()
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(fw), x2), min(float(fh), y2)
                bbox   = smoother.update(tid, np.array([x1, y1, x2, y2]))
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
                        show_id=disp.get("show_ids", True),
                        show_center=disp.get("show_centers", True),
                        thickness=disp.get("thickness", 2),
                    )

        else:
            # --- ByteTrack / BotSORT path (ultralytics built-in) ----------------
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
                            show_id=disp.get("show_ids", True),
                            show_center=disp.get("show_centers", True),
                            thickness=disp.get("thickness", 2),
                        )

        # Clean up per-track state for lost tracks
        lost = prev_ids - active_ids
        for tid in lost:
            smoother.remove(tid)
            if anchor is not None:
                anchor.remove(tid)

        cleanup_stale_counter_tracks(
            frame_idx,
            active_ids,
            counter_last_seen_frame,
            all_counters,
            counter_retention_frames,
        )
        prev_ids = active_ids
        frame_idx += 1

        # Draw lines
        for c in crosslines:
            counts = c.get_counts()
            draw_crossline(
                frame, c.pt1, c.pt2,
                counts["in"], counts["out"], c.name,
                color=tuple(disp.get("line_color", [0, 255, 0])),
                thickness=disp.get("thickness", 2),
            )

        # Draw zones
        for z in zones:
            counts = z.get_counts()
            draw_zone(
                frame, z.polygon, counts["in"], z.name,
                color=tuple(disp.get("zone_color", [0, 165, 255])),
                thickness=disp.get("thickness", 2),
            )

        if writer is not None:
            writer.write(frame)

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

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="People Counting")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--source", default=None,
                        help="Override video source (path or webcam index)")
    parser.add_argument("--save", default=None,
                        help="Save annotated output video (.mp4 added if omitted)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    if args.source is not None:
        try:
            cfg["video"]["source"] = int(args.source)
        except ValueError:
            cfg["video"]["source"] = args.source
    run(cfg, save_path=args.save)
