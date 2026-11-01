"""Shared frame/detection/pose types and stage protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

LABEL_RED = 'red'
LABEL_GREEN = 'green'
LABEL_EMPTY = 'empty'
LABEL_UNCERTAIN = 'uncertain'

CANONICAL_LABELS = (LABEL_RED, LABEL_GREEN, LABEL_EMPTY, LABEL_UNCERTAIN)
COLOR_LABELS = (LABEL_RED, LABEL_GREEN)


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

    ``slot`` is a known table index or None, ``label`` is the raw class name
    as produced by the segmenter (mapped to a canonical label by the
    pipeline), ``score`` is confidence in [0, 1], ``mask`` is an HxW bool
    array.
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


@dataclass(frozen=True)
class SlotResult:
    """
    Pipeline output for one configured table slot.

    ``label`` is one of :data:`CANONICAL_LABELS`. ``score`` is the
    confidence of the chosen detection (0 when nothing was detected).
    ``detection`` is None for ``empty`` slots without a detection; ``pose``
    is set only for ``red`` / ``green`` when estimation succeeded.
    """

    slot: int
    label: str
    score: float
    detection: Detection | None
    pose: Pose | None


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
