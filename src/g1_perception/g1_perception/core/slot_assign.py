"""Assign place numbers 1..N to tracked cubes along a robot axis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from g1_perception.core.tracker import TrackedCube
from g1_perception.core.types import LABEL_UNCERTAIN
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

    Validation (all must hold, else every cube becomes ``uncertain`` and
    ``id`` stays the track id):

    * ``len(tracked) == expected_count``
    * each cube has a pose
    * max distance to the least-squares line ≤ ``max_line_deviation_m``
    * consecutive gaps along ``axis`` in ``[min_spacing_m, max_spacing_m]``
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
        axis_index=idx,
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
        replace(
            t.as_voted_slot(slot=t.track_id),
            label=LABEL_UNCERTAIN,
            pose=t.pose,
            yaw_deg=t.yaw_deg,
            yaw_consistency=t.yaw_consistency,
            assignment_ok=False,
        )
        for t in ordered
    ]
    return AssignmentResult(slots=slots, valid=False)


def _validate(
    ordered: Sequence[TrackedCube],
    *,
    axis_index: int,
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
    if not _line_ok(points, max_line_deviation_m):
        return False

    coords = points[:, axis_index]
    gaps = np.diff(coords)
    if gaps.size and (
        np.any(gaps < min_spacing_m) or np.any(gaps > max_spacing_m)
    ):
        return False
    return True


def _line_ok(points: np.ndarray, max_dev: float) -> bool:
    """Check that every point lies within ``max_dev`` of the best-fit 3D line."""
    if points.shape[0] < 2:
        return True
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Direction = first principal component.
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return float(np.max(np.linalg.norm(centered, axis=1))) <= max_dev
    direction = direction / norm
    # Distance to line: || (p - c) × d ||.
    cross = np.cross(centered, direction)
    dist = np.linalg.norm(cross, axis=1)
    return float(np.max(dist)) <= max_dev
