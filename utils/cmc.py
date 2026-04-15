"""
Simple Camera Motion Compensation (CMC) using dense optical flow.

Estimates global camera motion between consecutive frames via
cv2.calcOpticalFlowFarneback, then computes the median flow vector.
This estimate can be subtracted from track velocity predictions
to compensate for camera pan/tilt/vibration.

Only useful for non-static cameras. Default: disabled.
"""

import cv2
import numpy as np


class SimpleMotionCompensation:
    """
    Args:
        pyr_scale: Farneback pyramid scale (0.5 = classic pyramid).
        levels:    Number of pyramid levels.
        winsize:   Averaging window size.
        downsample: Downsample factor for speed (2 = half resolution).
    """

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        downsample: int = 2,
    ):
        self._pyr_scale = pyr_scale
        self._levels = levels
        self._winsize = winsize
        self._downsample = max(1, downsample)
        self._prev_gray: np.ndarray | None = None

    def estimate(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Estimate global camera motion from the current frame.

        Args:
            frame: BGR frame (numpy array).

        Returns:
            (dx, dy): estimated camera motion in pixels (at original resolution).
                      Subtract this from track velocities to compensate.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downsample for speed
        if self._downsample > 1:
            gray_small = cv2.resize(
                gray,
                (gray.shape[1] // self._downsample, gray.shape[0] // self._downsample),
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray_small = gray

        if self._prev_gray is None:
            self._prev_gray = gray_small
            return (0.0, 0.0)

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray_small, None,
                self._pyr_scale, self._levels, self._winsize,
                3, 5, 1.2, 0,
            )
            # Median flow = robust estimate of global camera motion
            dx = float(np.median(flow[..., 0])) * self._downsample
            dy = float(np.median(flow[..., 1])) * self._downsample
        except cv2.error:
            dx, dy = 0.0, 0.0

        self._prev_gray = gray_small
        return (dx, dy)

    def reset(self) -> None:
        """Reset state (e.g. on scene change)."""
        self._prev_gray = None
