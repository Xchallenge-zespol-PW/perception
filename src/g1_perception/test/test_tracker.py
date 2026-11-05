"""Tests for 3D nearest-neighbour cube tracking."""

from __future__ import annotations

import numpy as np

from g1_perception.core.tracker import CubeTracker
from g1_perception.core.types import (
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Pose,
    SlotResult,
)


def _pose(x: float, y: float, z: float = 0.5) -> Pose:
    return Pose(
        position=np.array([x, y, z], dtype=np.float64),
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        score=0.9,
    )


def _obs(label: str, x: float, y: float) -> SlotResult:
    return SlotResult(
        slot=None,
        label=label,
        score=0.9,
        detection=None,
        pose=_pose(x, y),
    )


def test_stable_ids_with_position_noise():
    tracker = CubeTracker(
        match_dist_m=0.03, max_missed=3, vote_window=3, vote_min_frames=1
    )
    rng = np.random.default_rng(0)
    base = [(LABEL_RED, 0.0, 0.0), (LABEL_GREEN, 0.1, 0.0)]

    last_ids = None
    for _ in range(8):
        obs = [
            _obs(lab, x + float(rng.normal(0, 0.005)), y)
            for lab, x, y in base
        ]
        tracked = tracker.update(obs)
        ids = tuple(sorted(t.track_id for t in tracked))
        assert len(ids) == 2
        if last_ids is not None:
            assert ids == last_ids
        last_ids = ids


def test_brief_miss_keeps_track_then_drops():
    tracker = CubeTracker(
        match_dist_m=0.03, max_missed=2, vote_window=3, vote_min_frames=1
    )
    tracker.update([_obs(LABEL_RED, 0.0, 0.0)])
    assert len(tracker.update([])) == 1  # missed=1
    assert len(tracker.update([])) == 1  # missed=2
    assert tracker.update([]) == []  # missed=3 > max_missed


def test_two_close_cubes_keep_separate_ids():
    tracker = CubeTracker(
        match_dist_m=0.03, max_missed=2, vote_window=5, vote_min_frames=1
    )
    # 2.5 cm apart — under typical spacing but each stays nearer to itself
    a = (LABEL_RED, 0.0, 0.0)
    b = (LABEL_GREEN, 0.025, 0.0)
    ids_a = ids_b = None
    for _ in range(5):
        tracked = tracker.update([_obs(*a), _obs(*b)])
        # May be uncertain until vote settles; match by position instead.
        by_y = {
            round(float(t.pose.position[0]), 3): t.track_id
            for t in tracked
            if t.pose is not None
        }
        assert len(tracked) == 2
        if ids_a is None:
            ids_a = by_y.get(0.0) or by_y.get(0.025)
            ids_b = [i for i in by_y.values() if i != ids_a][0]
        else:
            assert set(by_y.values()) == {ids_a, ids_b}


def test_majority_colour_over_window():
    tracker = CubeTracker(
        match_dist_m=0.05, max_missed=2, vote_window=5, vote_min_ratio=0.6,
        vote_min_frames=3,
    )
    for lab in (LABEL_RED, LABEL_RED, LABEL_GREEN, LABEL_RED, LABEL_RED):
        tracked = tracker.update([_obs(lab, 0.0, 0.0)])
    assert len(tracked) == 1
    assert tracked[0].label == LABEL_RED
    assert tracked[0].label != LABEL_UNCERTAIN
