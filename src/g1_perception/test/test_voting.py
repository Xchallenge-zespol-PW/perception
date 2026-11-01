"""Tests for multi-frame slot voting on synthetic sequences."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.geometry import (
    quat_from_plane_yaw,
    quat_to_rotation_matrix,
    yaw_error_mod90_deg,
    yaw_on_plane_deg,
)
from g1_perception.core.types import (
    LABEL_EMPTY,
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Pose,
    SlotResult,
)
from g1_perception.core.voting import SlotVoter, circular_median_mod90

TABLE_NORMAL = np.array([0.0, -0.6, -0.8])  # camera looking down at table


def _result(
    label: str,
    *,
    slot: int = 0,
    yaw_deg: float = 10.0,
    position: tuple[float, float, float] = (0.1, 0.2, 0.9),
    with_pose: bool = True,
    normal: np.ndarray = TABLE_NORMAL,
) -> SlotResult:
    pose = None
    if with_pose and label in (LABEL_RED, LABEL_GREEN):
        pose = Pose(
            position=np.asarray(position, dtype=np.float64),
            quat=quat_from_plane_yaw(normal, yaw_deg),
            score=0.9,
        )
    return SlotResult(
        slot=slot,
        label=label,
        score=0.9 if label != LABEL_EMPTY else 0.0,
        detection=None,
        pose=pose,
    )


def _feed(voter: SlotVoter, frames: list[list[SlotResult]]):
    out = None
    for frame in frames:
        out = voter.update(frame)
    return out


def test_quat_from_plane_yaw_roundtrip():
    for yaw in (0.0, 12.5, 44.9, 89.0):
        quat = quat_from_plane_yaw(TABLE_NORMAL, yaw)
        n = quat_to_rotation_matrix(quat)[:, 2]
        assert np.allclose(n, TABLE_NORMAL / np.linalg.norm(TABLE_NORMAL))
        assert yaw_error_mod90_deg(yaw_on_plane_deg(quat, n), yaw) < 1e-6


def test_majority_colour_above_min_ratio():
    voter = SlotVoter(window=10, min_ratio=0.7)
    frames = [[_result(LABEL_RED)]] * 8 + [[_result(LABEL_GREEN)]] * 2

    voted = _feed(voter, frames)[0]

    assert voted.label == LABEL_RED
    assert voted.score == pytest.approx(0.8)
    assert voted.n_frames == 10
    assert voted.pose is not None


def test_below_min_ratio_is_uncertain():
    voter = SlotVoter(window=10, min_ratio=0.7)
    frames = [[_result(LABEL_RED)]] * 6 + [[_result(LABEL_GREEN)]] * 4

    voted = _feed(voter, frames)[0]

    assert voted.label == LABEL_UNCERTAIN
    assert voted.score == pytest.approx(0.6)
    assert voted.pose is None


def test_tie_is_uncertain_even_with_low_min_ratio():
    voter = SlotVoter(window=4, min_ratio=0.5)
    frames = [[_result(LABEL_RED)]] * 2 + [[_result(LABEL_GREEN)]] * 2

    assert _feed(voter, frames)[0].label == LABEL_UNCERTAIN


def test_too_few_frames_is_uncertain():
    voter = SlotVoter(window=10, min_ratio=0.7)

    voted = _feed(voter, [[_result(LABEL_RED)]] * 3)[0]

    assert voted.label == LABEL_UNCERTAIN
    assert voted.n_frames == 3
    assert voted.score == 0.0


def test_min_frames_allows_partial_window():
    voter = SlotVoter(window=10, min_ratio=0.7, min_frames=3)

    assert _feed(voter, [[_result(LABEL_RED)]] * 3)[0].label == LABEL_RED


def test_window_slides_to_new_majority():
    voter = SlotVoter(window=5, min_ratio=0.8)
    frames = [[_result(LABEL_GREEN)]] * 5 + [[_result(LABEL_RED)]] * 5

    voted = _feed(voter, frames)[0]

    assert voted.label == LABEL_RED
    assert voted.score == pytest.approx(1.0)
    assert voted.n_frames == 5


def test_uncertain_frames_count_against_winner():
    voter = SlotVoter(window=10, min_ratio=0.7)
    frames = [[_result(LABEL_RED)]] * 6 + [[_result(LABEL_UNCERTAIN)]] * 4

    assert _feed(voter, frames)[0].label == LABEL_UNCERTAIN


def test_uncertain_majority_stays_uncertain():
    voter = SlotVoter(window=5, min_ratio=0.6)

    voted = _feed(voter, [[_result(LABEL_UNCERTAIN)]] * 5)[0]

    assert voted.label == LABEL_UNCERTAIN


def test_empty_majority_has_no_pose():
    voter = SlotVoter(window=5, min_ratio=0.8)
    frames = [[_result(LABEL_EMPTY)]] * 4 + [[_result(LABEL_RED)]]

    voted = _feed(voter, frames)[0]

    assert voted.label == LABEL_EMPTY
    assert voted.pose is None
    assert voted.yaw_deg is None


def test_colour_without_any_pose_keeps_label():
    voter = SlotVoter(window=3, min_ratio=0.6)

    voted = _feed(voter, [[_result(LABEL_GREEN, with_pose=False)]] * 3)[0]

    assert voted.label == LABEL_GREEN
    assert voted.pose is None


def test_position_is_median_and_ignores_outlier():
    voter = SlotVoter(window=5, min_ratio=0.6)
    positions = [
        (0.10, 0.20, 0.90),
        (0.11, 0.21, 0.91),
        (0.09, 0.19, 0.89),
        (0.10, 0.20, 0.90),
        (0.60, -0.40, 2.50),  # depth outlier
    ]
    frames = [[_result(LABEL_RED, position=p)] for p in positions]

    voted = _feed(voter, frames)[0]

    assert np.allclose(voted.pose.position, [0.10, 0.20, 0.90], atol=1e-9)


def test_position_ignores_frames_with_other_labels():
    voter = SlotVoter(window=5, min_ratio=0.6)
    frames = [[_result(LABEL_RED, position=(0.1, 0.1, 1.0))]] * 4 + [
        [_result(LABEL_GREEN, position=(5.0, 5.0, 5.0))]
    ]

    voted = _feed(voter, frames)[0]

    assert np.allclose(voted.pose.position, [0.1, 0.1, 1.0])


@pytest.mark.parametrize(
    ('yaws', 'expected'),
    [
        ([88.0, 89.0, 1.0, 2.0, 3.0], 1.0),
        ([87.0, 88.0, 89.0, 0.0, 1.0, 2.0, 3.0], 0.0),
        ([89.5, 0.5, 89.5, 0.5], 0.0),
        ([-2.0, 92.0, 1.0], 1.0),
        ([30.0, 31.0, 32.0], 31.0),
    ],
)
def test_circular_median_wraps_across_0_and_90(yaws, expected):
    yaw, consistency = circular_median_mod90(yaws)

    assert 0.0 <= yaw < 90.0
    assert yaw_error_mod90_deg(yaw, expected) < 1e-6
    assert consistency > 0.9


def test_circular_median_ignores_yaw_outlier():
    yaw, _ = circular_median_mod90([10.0, 10.0, 11.0, 12.0, 40.0])

    assert yaw == pytest.approx(11.0)


def test_circular_median_low_consistency_for_spread_yaws():
    _, consistency = circular_median_mod90([0.0, 22.5, 45.0, 67.5])

    assert consistency < 1e-9


def test_circular_median_requires_values():
    with pytest.raises(ValueError):
        circular_median_mod90([])


def test_voted_yaw_across_zero_and_pose_orientation():
    voter = SlotVoter(window=6, min_ratio=0.8)
    yaws = [88.0, 89.0, 0.0, 1.0, 2.0, 89.5]
    frames = [[_result(LABEL_RED, yaw_deg=y)] for y in yaws]

    voted = _feed(voter, frames)[0]

    assert voted.label == LABEL_RED
    assert yaw_error_mod90_deg(voted.yaw_deg, 89.75) < 1e-6
    assert voted.yaw_consistency > 0.95
    R = quat_to_rotation_matrix(voted.pose.quat)
    n_true = TABLE_NORMAL / np.linalg.norm(TABLE_NORMAL)
    assert np.allclose(R[:, 2], n_true, atol=1e-9)
    assert yaw_error_mod90_deg(
        yaw_on_plane_deg(voted.pose.quat, R[:, 2]), voted.yaw_deg
    ) < 1e-6
    assert voted.pose.score == pytest.approx(1.0)


def test_slots_are_voted_independently_and_sorted():
    voter = SlotVoter(window=3, min_ratio=0.6)
    frame = [
        _result(LABEL_GREEN, slot=2),
        _result(LABEL_RED, slot=0),
        _result(LABEL_EMPTY, slot=1),
        _result(LABEL_UNCERTAIN, slot=3),
    ]

    voted = _feed(voter, [frame] * 3)

    assert [v.slot for v in voted] == [0, 1, 2, 3]
    assert [v.label for v in voted] == [
        LABEL_RED,
        LABEL_EMPTY,
        LABEL_GREEN,
        LABEL_UNCERTAIN,
    ]


def test_reset_clears_history():
    voter = SlotVoter(window=2, min_ratio=0.5)
    _feed(voter, [[_result(LABEL_RED)]] * 2)

    voter.reset()

    assert voter.vote() == []


@pytest.mark.parametrize(
    'kwargs',
    [
        {'window': 0},
        {'min_ratio': 0.0},
        {'min_ratio': 1.5},
        {'window': 5, 'min_frames': 6},
        {'window': 5, 'min_frames': 0},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        SlotVoter(**kwargs)
