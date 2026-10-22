"""Shared frame/detection/pose types and stage protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Frame:
    """
    One synchronized RGB-D observation in the color camera frame.

    ``rgb`` is HxWx3 uint8, ``depth_m`` is HxW float32 metres (0 = invalid),
    ``K`` is 3x3 intrinsics, ``stamp`` is capture time in seconds.
    """

    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    K: NDArray[np.float64]
    stamp: float


@dataclass(frozen=True)
class Detection:
    """
    A segmented object proposal on a frame.

    ``slot`` is a known table index or None, ``label`` is the class name,
    ``score`` is confidence in [0, 1], ``mask`` is an HxW bool array.
    """

    slot: int | None
    label: str
    score: float
    mask: NDArray[np.bool_]


@dataclass(frozen=True)
class Pose:
    """
    6-DoF object pose in the camera frame.

    ``position`` is (x, y, z) metres, ``quat`` is (x, y, z, w),
    ``score`` is confidence in [0, 1].
    """

    position: NDArray[np.float64]
    quat: NDArray[np.float64]
    score: float


class Segmenter(Protocol):
    """Produce object detections from a single frame."""

    def segment(self, frame: Frame) -> list[Detection]:
        """Segment objects in ``frame`` and return detections."""
        ...


class PoseEstimator(Protocol):
    """Estimate pose for a single detection."""

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        """Estimate pose of ``det`` in ``frame``, or ``None`` if impossible."""
        ...
