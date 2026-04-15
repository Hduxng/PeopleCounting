"""
Shared fixtures and synthetic scene simulator for people counting tests.

SceneSimulator generates frames with colored rectangles as "people",
providing ground-truth bounding boxes that bypass YOLO detection.
This lets us test tracker + counter logic in isolation.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Scene simulator — generates frames with moving "people" (colored patches)
# ---------------------------------------------------------------------------

class Person:
    """A simulated person moving through a scene."""

    __slots__ = ("pid", "x", "y", "w", "h", "vx", "vy", "color", "visible")

    def __init__(self, pid: int, x: float, y: float, w: int = 50, h: int = 120,
                 vx: float = 0, vy: float = 0, color: tuple = None):
        self.pid = pid
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.vx, self.vy = vx, vy
        self.color = color or _random_color(pid)
        self.visible = True

    def step(self):
        self.x += self.vx
        self.y += self.vy

    @property
    def bbox_xyxy(self) -> tuple:
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        return (x1, y1, x1 + self.w, y1 + self.h)

    @property
    def center(self) -> tuple:
        return (self.x, self.y)


def _random_color(seed: int) -> tuple:
    rng = np.random.RandomState(seed + 42)
    return tuple(int(c) for c in rng.randint(80, 255, 3))


class SceneSimulator:
    """
    Generates synthetic video frames with controllable "people".

    Usage:
        sim = SceneSimulator(1280, 720)
        sim.add_person(Person(1, 100, 400, vx=5, vy=0))
        for frame, gt_dets in sim.run(100):
            # frame: np.ndarray (H, W, 3)
            # gt_dets: list of (pid, x1, y1, x2, y2, conf)
            ...
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.people: list[Person] = []
        self._frame_idx = 0

    def add_person(self, person: Person):
        self.people.append(person)

    def run(self, num_frames: int):
        """Yields (frame, ground_truth_dets) for each step."""
        for _ in range(num_frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = (40, 40, 40)  # dark gray background

            dets = []
            for p in self.people:
                if p.visible:
                    x1, y1, x2, y2 = p.bbox_xyxy
                    # Clamp to frame
                    x1c = max(0, int(x1))
                    y1c = max(0, int(y1))
                    x2c = min(self.width, int(x2))
                    y2c = min(self.height, int(y2))
                    if x2c > x1c and y2c > y1c:
                        # Draw person as filled rectangle with head circle
                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), p.color, -1)
                        head_r = min(p.w, p.h) // 4
                        cv2.circle(frame, (int(p.x), y1c + head_r), head_r, p.color, -1)
                        dets.append((p.pid, x1, y1, x2, y2, 0.95))

                p.step()

            timestamp = self._frame_idx / self.fps
            self._frame_idx += 1
            yield frame, dets, timestamp

    def reset(self):
        self._frame_idx = 0
        self.people.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scene():
    """Fresh SceneSimulator for each test."""
    return SceneSimulator(1280, 720, fps=30.0)


@pytest.fixture
def crossline_cfg():
    """Horizontal crossline at y=360 (middle of 720p frame)."""
    return {
        "id": "test_line",
        "name": "Test Line",
        "points": [[0, 360], [1280, 360]],
        "enter_direction": "positive",
        "buffer_px": 20,
    }


@pytest.fixture
def zone_cfg():
    """Rectangular zone in the center of the frame."""
    return {
        "id": "test_zone",
        "name": "Test Zone",
        "points": [[300, 200], [900, 200], [900, 500], [300, 500]],
        "min_dwell_seconds": 2.0,
    }


@pytest.fixture
def fast_zone_cfg():
    """Zone with short dwell time for faster tests."""
    return {
        "id": "test_zone",
        "name": "Test Zone",
        "points": [[300, 200], [900, 200], [900, 500], [300, 500]],
        "min_dwell_seconds": 0.5,
    }
