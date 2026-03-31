"""OpenCV drawing helpers for visualisation."""

from typing import Tuple, List
import cv2
import numpy as np


def draw_crossline(
    frame: np.ndarray,
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    count_in: int,
    count_out: int,
    name: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    p1 = (int(pt1[0]), int(pt1[1]))
    p2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(frame, p1, p2, color, thickness)

    mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    label = f"{name}  In:{count_in}  Out:{count_out}"
    cv2.putText(frame, label, (mid[0] - 60, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_zone(
    frame: np.ndarray,
    polygon: List[Tuple[float, float]],
    count: int,
    name: str = "",
    color: Tuple[int, int, int] = (0, 165, 255),
    thickness: int = 2,
) -> None:
    pts = np.array(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (*color, 30))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    label_origin = (int(pts[0][0]), int(pts[0][1]) - 10)
    cv2.putText(frame, f"{name}  Count:{count}", label_origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_track(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    track_id: int,
    center: Tuple[float, float],
    color: Tuple[int, int, int] = (0, 255, 255),
    show_id: bool = True,
    show_center: bool = True,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if show_center:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    if show_id:
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
