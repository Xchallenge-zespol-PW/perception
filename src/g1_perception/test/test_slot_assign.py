"""Tests for place-number assignment along a robot axis."""

from __future__ import annotations

import numpy as np

from g1_perception.core.slot_assign import assign_slots
from g1_perception.core.tracker import TrackedCube
from g1_perception.core.types import LABEL_GREEN, LABEL_RED, LABEL_UNCERTAIN, Pose


def _cube(
    track_id: int,
    y: float,
    *,
    x: float = 0.5,
    z: float = 0.0,
    label: str = LABEL_RED,
) -> TrackedCube:
    return TrackedCube(
        track_id=track_id,
        label=label,
        score=1.0,
        n_frames=5,
        pose=Pose(
            position=np.array([x, y, z], dtype=np.float64),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            score=1.0,
        ),
        yaw_deg=10.0,
        yaw_consistency=1.0,
        detection=None,
        missed=0,
    )


def test_sorts_left_to_right_along_y_and_numbers_1_to_n():
    # Deliberately reverse track ids vs y order.
    tracked = [
        _cube(10, 0.30, label=LABEL_GREEN),
        _cube(3, 0.00, label=LABEL_RED),
        _cube(7, 0.20, label=LABEL_RED),
        _cube(5, 0.10, label=LABEL_GREEN),
    ]
    result = assign_slots(
        tracked,
        axis='y',
        expected_count=4,
        max_line_deviation_m=0.05,
        min_spacing_m=0.05,
        max_spacing_m=0.20,
    )
    assert result.valid
    assert [s.slot for s in result.slots] == [1, 2, 3, 4]
    assert [s.track_id for s in result.slots] == [3, 5, 7, 10]
    assert [round(float(s.pose.position[1]), 2) for s in result.slots] == [
        0.0, 0.1, 0.2, 0.3,
    ]


def test_wrong_count_marks_all_uncertain_with_track_ids():
    tracked = [_cube(1, 0.0), _cube(2, 0.1), _cube(3, 0.2)]
    result = assign_slots(tracked, expected_count=4, min_spacing_m=0.05)
    assert not result.valid
    assert all(s.label == LABEL_UNCERTAIN for s in result.slots)
    assert all(not s.assignment_ok for s in result.slots)
    assert {s.slot for s in result.slots} == {1, 2, 3}


def test_line_deviation_fails_validation():
    tracked = [
        _cube(1, 0.0, x=0.0),
        _cube(2, 0.1, x=0.0),
        _cube(3, 0.2, x=0.0),
        _cube(4, 0.3, x=0.20),  # far off the line
    ]
    result = assign_slots(
        tracked,
        expected_count=4,
        max_line_deviation_m=0.03,
        min_spacing_m=0.05,
        max_spacing_m=0.25,
    )
    assert not result.valid
    assert all(s.label == LABEL_UNCERTAIN for s in result.slots)


def test_spacing_out_of_range_fails():
    tracked = [
        _cube(1, 0.0),
        _cube(2, 0.02),  # too close
        _cube(3, 0.12),
        _cube(4, 0.22),
    ]
    result = assign_slots(
        tracked,
        expected_count=4,
        max_line_deviation_m=0.05,
        min_spacing_m=0.05,
        max_spacing_m=0.20,
    )
    assert not result.valid
