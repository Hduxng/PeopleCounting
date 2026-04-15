"""Geometry helpers for line crossing and polygon containment."""

import math
import numpy as np
from typing import Tuple, List


def point_side_of_line(
    point: Tuple[float, float],
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    buffer_px: float = 0,
    line_len: float | None = None,
) -> int:
    """
    Determine which side of a directed line (pt1→pt2) a point is on.

    Uses the 2-D cross product:
        cross = (pt2 - pt1) × (point - pt1)

    Args:
        line_len: Pre-computed line length (optional, avoids sqrt per call).

    Returns:
         1  — positive side
        -1  — negative side
         0  — inside the buffer zone around the line (treat as "on the line")
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    cross = dx * (point[1] - pt1[1]) - dy * (point[0] - pt1[0])

    if buffer_px > 0:
        if line_len is None:
            line_len = math.sqrt(dx * dx + dy * dy)
        if line_len > 0:
            perp_dist = abs(cross) / line_len
            if perp_dist <= buffer_px:
                return 0

    return 1 if cross >= 0 else -1


def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Return the center of a bounding box (x1, y1, x2, y2).

    Per slide 9: Center = midpoint of the detection bounding box.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
    polygon_np: np.ndarray | None = None,
) -> bool:
    """Return True if *point* is inside (or on the edge of) *polygon*.

    Args:
        polygon_np: Pre-computed numpy array of polygon points (avoids re-creation).
    """
    import cv2

    if polygon_np is None:
        polygon_np = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False)
    return result >= 0
