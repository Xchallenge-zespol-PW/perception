"""Tests for the world-frame ball map."""

import numpy as np
import pytest

from g1_perception.core.ball_map import CONFIRM_FRAMES, TENTATIVE_TIMEOUT_S, BallMap

R = 0.033
A = np.array([2.0, 0.5, 0.6])
B = np.array([2.0, 0.8, 0.6])


def _feed(m, frames, t0=0.0, dt=1 / 15):
    out = []
    for k, pts in enumerate(frames):
        out.append(m.update(pts, [R] * len(pts), t0 + k * dt))
    return out


def test_ball_gets_an_id_after_confirm_frames():
    m = BallMap()
    out = _feed(m, [[A]] * CONFIRM_FRAMES)
    assert [ids for ids, _ in out[:-1]] == [[None]] * (CONFIRM_FRAMES - 1)
    ids, confirmed = out[-1]
    assert ids == [1] and [b.ball_id for b in confirmed] == [1]
    (ball,) = m.known()
    assert ball.position == pytest.approx(A)
    assert ball.radius_m == pytest.approx(R)


def test_ids_are_stable_while_out_of_view_and_noise_is_matched():
    rng = np.random.default_rng(0)
    m = BallMap()
    _feed(m, [[A, B]] * CONFIRM_FRAMES)
    ids_before = {tuple(b.position.round(3)): b.ball_id for b in m.known()}
    # 10 s looking away, then both again with 1 cm noise, in swapped order.
    frames = [[B + rng.normal(0, 0.01, 3), A + rng.normal(0, 0.01, 3)]
              for _ in range(20)]
    out = _feed(m, frames, t0=10.0)
    assert all(ids == [2, 1] for ids, _ in out)
    assert len(m.known()) == 2
    assert ids_before == {(2.0, 0.5, 0.6): 1, (2.0, 0.8, 0.6): 2}
    assert m.known()[0].position == pytest.approx(A, abs=0.01)


def test_one_frame_false_detection_never_gets_an_id_and_is_forgotten():
    m = BallMap()
    _feed(m, [[A]])
    _feed(m, [[]] * 3, t0=TENTATIVE_TIMEOUT_S + 0.1)
    _feed(m, [[A]] * (CONFIRM_FRAMES - 1), t0=TENTATIVE_TIMEOUT_S + 0.5)
    assert m.known() == []


def test_two_balls_in_one_frame_near_one_map_ball_make_two_balls():
    m = BallMap()
    _feed(m, [[A]] * CONFIRM_FRAMES)
    near = A + np.array([0.0, 2.5 * R, 0.0])  # touching neighbour + margin
    ids, _ = m.update([A, near], [R, R], 1.0)
    assert ids[0] == 1 and ids[1] is None


def test_jump_beyond_the_gate_starts_a_new_ball():
    m = BallMap()
    _feed(m, [[A]] * CONFIRM_FRAMES)
    far = A + np.array([0.0, 0.0, 3 * R])
    out = _feed(m, [[far]] * CONFIRM_FRAMES, t0=1.0)
    assert out[-1][0] == [2]
    assert m.nearest_known(far, exclude=2)[0].ball_id == 1


def test_older_stamp_resets_and_reset_restarts_ids():
    m = BallMap()
    _feed(m, [[A]] * CONFIRM_FRAMES, t0=5.0)
    assert len(m.known()) == 1
    m.update([A], [R], 1.0)
    assert m.known() == []
    _feed(m, [[B]] * CONFIRM_FRAMES, t0=2.0)
    assert [b.ball_id for b in m.known()] == [1]
    m.reset()
    assert m.known() == [] and m.nearest_known(A) is None
    with pytest.raises(ValueError):
        m.update([A], [], 3.0)
