"""OpenCV drawing helpers for visualisation."""

from typing import Tuple, List
import cv2
import numpy as np


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw legible text with a filled background box."""
    pad_x = 6
    pad_y = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(int(origin[0]), frame_w - text_w - (2 * pad_x) - 1))
    y = max(text_h + pad_y, min(int(origin[1]), frame_h - baseline - pad_y - 1))

    top_left = (x, y - text_h - pad_y)
    bottom_right = (x + text_w + (2 * pad_x), y + baseline + pad_y)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(frame, text, (x + pad_x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_crossline(
    frame: np.ndarray,
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    count_in: int,
    count_out: int,
    name: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg_color: Tuple[int, int, int] = (40, 96, 40),
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    p1 = (int(pt1[0]), int(pt1[1]))
    p2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(frame, p1, p2, color, thickness)

    mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    label = f"{name}  In:{count_in}  Out:{count_out}"
    _draw_label(
        frame,
        label,
        (mid[0] - 80, mid[1] - 12),
        text_color=text_color,
        bg_color=label_bg_color,
        font_scale=font_scale,
        thickness=max(1, thickness - 1),
    )


def draw_zone(
    frame: np.ndarray,
    polygon: List[Tuple[float, float]],
    count: int,
    name: str = "",
    color: Tuple[int, int, int] = (0, 165, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg_color: Tuple[int, int, int] = (0, 96, 176),
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    pts = np.array(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    min_x = int(np.min(pts[:, 0]))
    min_y = int(np.min(pts[:, 1]))
    _draw_label(
        frame,
        f"{name}  Count:{count}",
        (min_x, min_y - 10),
        text_color=text_color,
        bg_color=label_bg_color,
        font_scale=font_scale,
        thickness=max(1, thickness - 1),
    )


def draw_track(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    track_id: int,
    center: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 255, 255),
    id_text_color: Tuple[int, int, int] = (255, 255, 255),
    id_bg_color: Tuple[int, int, int] = (32, 32, 32),
    center_color: Tuple[int, int, int] = (255, 0, 255),
    show_id: bool = True,
    show_center: bool = True,
    font_scale: float = 0.55,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if show_center:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), 4, center_color, -1)

    if show_id:
        _draw_label(
            frame,
            f"ID:{track_id}",
            (x1, y1 - 8),
            text_color=id_text_color,
            bg_color=id_bg_color,
            font_scale=font_scale,
            thickness=max(1, thickness - 1),
        )
