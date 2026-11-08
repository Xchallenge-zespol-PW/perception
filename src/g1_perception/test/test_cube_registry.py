"""Tests for freezing the start layout and naming the cubes during a task."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.cube_registry import CubeRegistry
from g1_perception.core.slot_assign import AssignmentResult, assign_slots
from g1_perception.core.tracker import TrackedCube
from g1_perception.core.types import (
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Pose,
)

# Exercise 1: a column along x, centres 6 cm apart, pushed along y.
START_X = (0.30, 0.36, 0.42, 0.48)
COLORS = (LABEL_RED, LABEL_GREEN, LABEL_GREEN, LABEL_RED)

# Frame period: exact in binary, so stamp differences compare exactly.
DT = 0.25


def _cube(
    track_id: int,
    x: float,
    *,
    y: float = 0.0,
    label: str = LABEL_RED,
    missed: int = 0,
) -> TrackedCube:
    return TrackedCube(
        track_id=track_id,
        label=label,
        score=1.0,
        n_frames=10,
        pose=Pose(
            position=np.array([x, y, -0.045], dtype=np.float64),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            score=1.0,
        ),
        yaw_deg=0.0,
        yaw_consistency=1.0,
        detection=None,
        missed=missed,
    )


def _start_tracks() -> list[TrackedCube]:
    return [
        _cube(10 + i, x, label=label)
        for i, (x, label) in enumerate(zip(START_X, COLORS))
    ]


def _assign(tracked: list[TrackedCube]) -> AssignmentResult:
    return assign_slots(
        tracked,
        axis='x',
        expected_count=4,
        max_line_deviation_m=0.03,
        min_spacing_m=0.04,
        max_spacing_m=0.10,
    )


class _Camera:
    """Feeds frames to a registry, one ``DT`` apart."""

    def __init__(self, registry: CubeRegistry) -> None:
        self.registry = registry
        self.stamp = 0.0

    def frame(self, tracked: list[TrackedCube]):
        cubes = self.registry.update(_assign(tracked), tracked, self.stamp)
        self.stamp += DT
        return cubes


def _latched(**kwargs) -> _Camera:
    camera = _Camera(CubeRegistry(axis='x', latch_s=2 * DT, **kwargs))
    for _ in range(3):
        camera.frame(_start_tracks())
    assert camera.registry.latched
    return camera


def _by_name(cubes):
    return {c.name: c for c in cubes}


def test_freezes_after_latch_s_with_names_in_place_order():
    camera = _Camera(CubeRegistry(axis='x', latch_s=2 * DT))

    assert camera.frame(_start_tracks()) == []
    assert camera.frame(_start_tracks()) == []
    assert not camera.registry.latched and camera.registry.start_id == 0

    cubes = camera.frame(_start_tracks())

    assert camera.registry.latched and camera.registry.start_id == 1
    assert [c.name for c in cubes] == ['cube_1', 'cube_2', 'cube_3', 'cube_4']
    assert [c.label for c in cubes] == list(COLORS)
    assert [c.track_id for c in cubes] == [10, 11, 12, 13]
    assert [float(c.start_pose.position[0]) for c in cubes] == list(START_X)
    assert all(c.pose is c.start_pose for c in cubes)


def test_latch_counts_time_not_frames():
    registry = CubeRegistry(axis='x', latch_s=1.0)
    tracked = _start_tracks()

    registry.update(_assign(tracked), tracked, 10.0)
    registry.update(_assign(tracked), tracked, 10.9)
    assert not registry.latched

    registry.update(_assign(tracked), tracked, 11.0)
    assert registry.latched


def test_name_prefix_is_used():
    camera = _Camera(CubeRegistry(axis='x', name_prefix='box', latch_s=0.0))

    cubes = camera.frame(_start_tracks())

    assert [c.name for c in cubes] == ['box1', 'box2', 'box3', 'box4']


def test_invalid_or_undecided_layout_restarts_the_clock():
    camera = _Camera(CubeRegistry(axis='x', latch_s=DT))
    undecided = _start_tracks()
    undecided[1] = _cube(11, START_X[1], label=LABEL_UNCERTAIN)
    three = _start_tracks()[:3]

    camera.frame(_start_tracks())
    camera.frame(undecided)
    camera.frame(_start_tracks())
    camera.frame(three)
    assert not camera.registry.latched

    camera.frame(_start_tracks())
    camera.frame(_start_tracks())
    assert camera.registry.latched


def test_changed_tracks_or_colours_restart_the_clock():
    camera = _Camera(CubeRegistry(axis='x', latch_s=DT))
    new_track = _start_tracks()
    new_track[0] = _cube(99, START_X[0], label=COLORS[0])
    recoloured = _start_tracks()
    recoloured[0] = _cube(10, START_X[0], label=LABEL_GREEN)

    camera.frame(_start_tracks())
    camera.frame(new_track)
    camera.frame(recoloured)
    assert not camera.registry.latched

    camera.frame(recoloured)
    assert camera.registry.latched


def test_older_stamp_before_the_freeze_restarts_the_clock():
    registry = CubeRegistry(axis='x', latch_s=0.5)
    tracked = _start_tracks()

    registry.update(_assign(tracked), tracked, 5.0)
    registry.update(_assign(tracked), tracked, 1.0)  # bag --loop
    registry.update(_assign(tracked), tracked, 1.25)
    assert not registry.latched

    registry.update(_assign(tracked), tracked, 1.5)
    assert registry.latched


def test_follows_a_pushed_cube_while_its_track_lives():
    camera = _latched()
    tracked = _start_tracks()
    tracked[0] = _cube(10, START_X[0], y=0.12, label=LABEL_UNCERTAIN)

    cube_1 = _by_name(camera.frame(tracked))['cube_1']

    assert cube_1.track_id == 10
    assert float(cube_1.pose.position[1]) == pytest.approx(0.12)
    assert float(cube_1.start_pose.position[1]) == pytest.approx(0.0)


def test_cube_not_seen_in_a_frame_has_no_pose():
    camera = _latched()
    tracked = _start_tracks()
    tracked[1] = _cube(11, START_X[1], label=COLORS[1], missed=2)

    cubes = _by_name(camera.frame(tracked[:3]))

    assert cubes['cube_2'].pose is None and cubes['cube_2'].track_id == 11
    assert cubes['cube_4'].pose is None
    assert cubes['cube_1'].pose is not None


def test_lost_cube_is_found_again_along_its_line():
    camera = _latched()
    others = _start_tracks()[1:]
    camera.frame(others)  # cube_1 occluded, its track dropped

    pushed = _cube(42, START_X[0] + 0.015, y=0.3, label=LABEL_UNCERTAIN)
    cube_1 = _by_name(camera.frame(others + [pushed]))['cube_1']

    assert cube_1.track_id == 42
    assert float(cube_1.pose.position[1]) == pytest.approx(0.3)
    assert cube_1.label == LABEL_RED


def test_new_track_takes_over_from_an_unseen_live_track():
    camera = _latched()
    tracked = _start_tracks()
    tracked[2] = _cube(12, START_X[2], label=COLORS[2], missed=1)
    moved = _cube(50, START_X[2], y=-0.1, label=LABEL_UNCERTAIN)

    cube_3 = _by_name(camera.frame(tracked + [moved]))['cube_3']

    assert cube_3.track_id == 50
    assert float(cube_3.pose.position[1]) == pytest.approx(-0.1)


def test_track_outside_the_gate_or_with_the_other_colour_is_rejected():
    camera = _latched()
    others = _start_tracks()[1:]
    far = _cube(60, START_X[0] + 0.03, y=0.3, label=LABEL_UNCERTAIN)
    wrong_colour = _cube(61, START_X[0], y=0.3, label=LABEL_GREEN)

    cube_1 = _by_name(camera.frame(others + [far, wrong_colour]))['cube_1']

    assert cube_1.pose is None and cube_1.track_id == 10


def test_bound_track_voting_the_other_colour_counts_as_unseen():
    camera = _latched()
    tracked = _start_tracks()
    tracked[3] = _cube(13, START_X[3], label=LABEL_GREEN)

    cube_4 = _by_name(camera.frame(tracked))['cube_4']

    assert cube_4.pose is None


def test_two_lost_cubes_take_the_nearest_tracks():
    camera = _latched()
    kept = [_start_tracks()[0], _start_tracks()[3]]
    camera.frame(kept)  # cube_2 and cube_3 (both green) lost

    near_3 = _cube(70, START_X[2] - 0.005, y=-0.3, label=LABEL_UNCERTAIN)
    near_2 = _cube(71, START_X[1] + 0.01, y=-0.3, label=LABEL_UNCERTAIN)
    cubes = _by_name(camera.frame(kept + [near_3, near_2]))

    assert cubes['cube_2'].track_id == 71
    assert cubes['cube_3'].track_id == 70


def test_one_track_names_only_one_cube():
    camera = _latched(reid_gate_m=0.05)
    kept = [_start_tracks()[0], _start_tracks()[3]]
    camera.frame(kept)

    between = _cube(80, (START_X[1] + START_X[2]) / 2, label=LABEL_UNCERTAIN)
    cubes = _by_name(camera.frame(kept + [between]))

    named = [c.name for c in cubes.values() if c.track_id == 80]
    assert len(named) == 1


def test_older_stamp_after_the_freeze_finds_cubes_by_position():
    camera = _latched()
    # The tracker restarted its ids: id 10 is now the cube at place 4.
    renumbered = [
        _cube(13, START_X[0], label=LABEL_UNCERTAIN),
        _cube(12, START_X[1], label=LABEL_UNCERTAIN),
        _cube(11, START_X[2], label=LABEL_UNCERTAIN),
        _cube(10, START_X[3], label=LABEL_UNCERTAIN),
    ]
    camera.stamp = 0.0

    cubes = _by_name(camera.frame(renumbered))

    assert cubes['cube_1'].track_id == 13
    assert float(cubes['cube_1'].pose.position[0]) == pytest.approx(START_X[0])
    assert cubes['cube_4'].track_id == 10


def test_reset_waits_for_a_new_layout_and_counts_starts():
    camera = _latched()

    camera.registry.reset()

    assert not camera.registry.latched and camera.registry.start_id == 1
    assert camera.frame(_start_tracks()[1:]) == []
    for _ in range(3):
        camera.frame(_start_tracks())
    assert camera.registry.latched and camera.registry.start_id == 2


@pytest.mark.parametrize(
    'kwargs',
    [
        {'axis': 'w'},
        {'name_prefix': ''},
        {'latch_s': -0.1},
        {'reid_gate_m': 0.0},
    ],
)
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        CubeRegistry(**kwargs)
