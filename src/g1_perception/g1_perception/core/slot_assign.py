"""Assign place numbers 1..N to tracked cubes along a robot axis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from g1_perception.core.tracker import TrackedCube
from g1_perception.core.voting import VotedSlot

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


@dataclass(frozen=True)
class AssignmentResult:
    """Place-assigned cubes plus whether geometric validation passed."""

    slots: list[VotedSlot]
    valid: bool


def assign_slots(
    tracked: Sequence[TrackedCube],
    *,
    axis: str = 'y',
    expected_count: int = 4,
    max_line_deviation_m: float = 0.03,
    min_spacing_m: float = 0.04,
    max_spacing_m: float = 0.20,
) -> AssignmentResult:
    """
    Sort stable cubes along ``axis`` (ascending) and number them ``1..N``.

    Validation (all must hold, else every cube keeps its voted colour and
    pose, ``id`` stays the track id and ``assignment_ok`` is False):

    * ``len(tracked) == expected_count``
    * each cube has a pose
    * max distance to the least-squares line ≤ ``max_line_deviation_m``
    * consecutive gaps measured **along that line** in
      ``[min_spacing_m, max_spacing_m]`` (so a row at an angle to ``axis``
      keeps its true spacing; ``axis`` only sets the numbering order)
    """
    axis_key = str(axis).strip().lower()
    if axis_key not in _AXIS_INDEX:
        raise ValueError(f'axis must be x/y/z, got {axis!r}')
    if expected_count < 1:
        raise ValueError(f'expected_count must be >= 1, got {expected_count}')

    with_pose = [t for t in tracked if t.pose is not None]
    idx = _AXIS_INDEX[axis_key]
    ordered = sorted(
        with_pose,
        key=lambda t: float(t.pose.position[idx]),  # type: ignore[index]
    )

    valid = _validate(
        ordered,
        expected_count=expected_count,
        max_line_deviation_m=max_line_deviation_m,
        min_spacing_m=min_spacing_m,
        max_spacing_m=max_spacing_m,
    )

    if valid:
        slots = [
            t.as_voted_slot(slot=place)
            for place, t in enumerate(ordered, start=1)
        ]
        return AssignmentResult(slots=slots, valid=True)

    slots = [
        replace(t.as_voted_slot(slot=t.track_id), assignment_ok=False)
        for t in ordered
    ]
    return AssignmentResult(slots=slots, valid=False)


def _validate(
    ordered: Sequence[TrackedCube],
    *,
    expected_count: int,
    max_line_deviation_m: float,
    min_spacing_m: float,
    max_spacing_m: float,
) -> bool:
    if len(ordered) != expected_count:
        return False
    if any(t.pose is None for t in ordered):
        return False

    points = np.stack([t.pose.position for t in ordered])  # type: ignore[union-attr]
    if points.shape[0] < 2:
        return True
    centroid = points.mean(axis=0)
    centered = points - centroid
    direction = _line_direction(centered)
    if not _line_ok(centered, direction, max_line_deviation_m):
        return False

    gaps = np.abs(np.diff(centered @ direction))
    if np.any(gaps < min_spacing_m) or np.any(gaps > max_spacing_m):
        return False
    return True


def _line_direction(centered: np.ndarray) -> np.ndarray:
    """Return the unit first principal component of centred points."""
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    return direction / float(np.linalg.norm(direction))


def _line_ok(centered: np.ndarray, direction: np.ndarray, max_dev: float) -> bool:
    """Check that every point lies within ``max_dev`` of the best-fit 3D line."""
    # Distance to line: || (p - c) × d ||.
    dist = np.linalg.norm(np.cross(centered, direction), axis=1)
    return float(np.max(dist)) <= max_dev
