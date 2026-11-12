"""Tests for 3D nearest-neighbour cube tracking."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from g1_perception.core.tracker import CubeTracker, fresh_tracks
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


def test_time_based_expiry_independent_of_frame_rate():
    for fps in (1.0, 20.0):
        tracker = CubeTracker(
            match_dist_m=0.03, vote_window=100, vote_window_s=2.0,
            vote_min_votes=1, max_missed_s=1.5,
        )
        tracker.update([_obs(LABEL_RED, 0.0, 0.0)], 0.0)
        t = 0.0
        alive_at_1s = None
        while t < 3.0:
            t += 1.0 / fps
            alive = bool(tracker.update([], t))
            if abs(t - 1.0) < 1e-6:
                alive_at_1s = alive
        assert alive_at_1s is True
        assert tracker.update([], t) == []


def test_time_based_tracker_needs_stamp_and_resets_on_loop():
    tracker = CubeTracker(vote_window=100, vote_window_s=2.0, max_missed_s=1.0)
    with pytest.raises(ValueError, match='stamp'):
        tracker.update([_obs(LABEL_RED, 0.0, 0.0)])
    tracker.update([_obs(LABEL_RED, 0.0, 0.0)], 10.0)
    tracked = tracker.update([_obs(LABEL_GREEN, 0.5, 0.0)], 0.1)
    assert [t.track_id for t in tracked] == [1]  # ids restart after reset


def test_split_track_ghost_is_not_fresh():
    tracker = CubeTracker(
        vote_window=100, vote_window_s=2.0, vote_min_votes=1, max_missed_s=1.0
    )
    for i in range(3):
        tracker.update([_obs(LABEL_RED, 0.0, 0.0)], stamp=0.1 * i)
    # The cube jumps beyond match_dist_m: new track, old one still alive.
    for i in range(3, 8):
        tracked = tracker.update([_obs(LABEL_RED, 0.05, 0.0)], stamp=0.1 * i)

    assert len(tracked) == 2
    old, new = sorted(tracked, key=lambda t: t.track_id)
    assert old.last_seen == pytest.approx(0.2)
    assert new.last_seen == pytest.approx(0.7)
    fresh = fresh_tracks(tracked, 0.7, 0.3)
    assert [t.track_id for t in fresh] == [new.track_id]


def test_fresh_tracks_disabled_keeps_all():
    tracker = CubeTracker(vote_window=100, vote_window_s=2.0, max_missed_s=1.0)
    tracker.update([_obs(LABEL_RED, 0.0, 0.0)], stamp=0.0)
    tracked = tracker.update([], stamp=0.5)
    assert len(fresh_tracks(tracked, 0.5, 0.0)) == 1
    assert fresh_tracks(tracked, 0.5, 0.3) == []


def test_untrusted_pose_keeps_track_but_not_its_position():
    tracker = CubeTracker(
        match_dist_m=0.03, max_missed_s=0.5, vote_window=20,
        vote_window_s=2.0, vote_min_votes=1,
    )
    first = tracker.update([_obs(LABEL_RED, 0.0, 0.0)], stamp=0.0)[0].track_id
    # Camera turning for 1 s: poses 2 cm off, all untrusted.
    for i in range(1, 11):
        obs = replace(_obs(LABEL_RED, 0.02, 0.0), pose_trusted=False)
        tracked = tracker.update([obs], stamp=0.1 * i)

    assert len(tracked) == 1
    tr = tracked[0]
    assert tr.track_id == first and tr.missed == 0 and tr.label == LABEL_RED
    assert tr.pose.position[0] == pytest.approx(0.0)
    assert tr.n_frames == 11  # colour votes still counted


def test_untrusted_observation_does_not_start_a_track():
    tracker = CubeTracker(
        match_dist_m=0.03, max_missed=3, vote_window=3, vote_min_frames=1
    )
    first = tracker.update([_obs(LABEL_RED, 0.0, 0.0)])[0].track_id
    far = replace(_obs(LABEL_GREEN, 0.3, 0.0), pose_trusted=False)
    tracked = tracker.update([_obs(LABEL_RED, 0.0, 0.0), far])
    assert [t.track_id for t in tracked] == [first]
