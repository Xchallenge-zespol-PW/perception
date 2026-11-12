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


K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])
C = np.array([0.1, 0.05, 1.5])  # ball centre in the camera frame


def _depth(value):
    return np.full((480, 640), value, np.float32)


def _ball_depth():
    from test_ball import _render_sphere  # noqa: PLC0415
    _, depth = _render_sphere(C, R, background=3.0)
    return depth


def test_view_state():
    from g1_perception.core.ball_map import (
        VIEW_EMPTY, VIEW_OCCLUDED, VIEW_OUT, VIEW_PRESENT, VIEW_UNKNOWN,
        project, view_state,
    )
    assert view_state(C, R, _ball_depth(), K) == VIEW_PRESENT
    assert view_state(C, R, _depth(3.0), K) == VIEW_EMPTY
    assert view_state(C, R, _depth(1.0), K) == VIEW_OCCLUDED
    assert view_state(C, R, _depth(0.0), K) == VIEW_UNKNOWN
    assert view_state(np.array([2.0, 0.0, 1.5]), R, _depth(3.0), K) == VIEW_OUT
    assert view_state(np.array([0.0, 0.0, -1.0]), R, _depth(3.0), K) == VIEW_OUT
    (u, v), r_px = project(C, R, K)
    assert (u, v) == pytest.approx((361.0, 260.5)) and r_px == pytest.approx(615 * R / 1.5)


def _known_map():
    m = BallMap()
    _feed(m, [[A, B]] * CONFIRM_FRAMES)
    return m


def test_ball_leaves_the_map_after_its_place_is_seen_empty():
    from g1_perception.core.ball_map import EMPTY_FRAMES, VIEW_EMPTY
    m = _known_map()
    removed = []
    for k in range(20):
        t = 1.0 + k / 15
        ids, _ = m.update([B], [R], t)
        assert ids == [2]
        removed += m.check_views({1: VIEW_EMPTY}, t)
        if removed:
            break
    assert [b.ball_id for b in removed] == [1]
    assert k + 1 >= max(EMPTY_FRAMES, 15)
    assert [b.ball_id for b in m.known()] == [2]
    # Put back: a new name, ids are never reused.
    out = _feed(m, [[A, B]] * CONFIRM_FRAMES, t0=3.0)
    assert out[-1][0] == [3, 2]


def test_present_seen_or_a_long_gap_cancel_the_removal():
    from g1_perception.core.ball_map import (
        EMPTY_TIMEOUT_S, VIEW_EMPTY, VIEW_OCCLUDED, VIEW_OUT, VIEW_PRESENT,
    )
    m = _known_map()
    t = 1.0
    for view in [VIEW_EMPTY] * 10 + [VIEW_PRESENT] + [VIEW_EMPTY] * 10:
        assert m.check_views({1: view}, t) == []
        t += 1 / 15
    assert m.empty_for(1, t) == pytest.approx(10 / 15, abs=0.01)
    # Occluded / out of view frames neither count nor cancel...
    for view in [VIEW_OCCLUDED, VIEW_OUT] * 3:
        assert m.check_views({1: view}, t) == []
        t += 1 / 15
    assert [b.ball_id for b in m.check_views({1: VIEW_EMPTY}, t)] == [1]
    # ...but a gap longer than the timeout restarts the count.
    m2 = _known_map()
    m2.check_views({1: VIEW_EMPTY}, 1.0)
    m2.check_views({1: VIEW_EMPTY}, 1.0 + EMPTY_TIMEOUT_S + 0.5)
    assert m2.empty_for(1, 2.6) == pytest.approx(0.1)  # restarted at 2.5
    assert len(m2.known()) == 2


def test_residuals_of_known_balls():
    m = _known_map()
    m.update([A + np.array([0.01, 0.0, 0.0]), B], [R, R], 2.0)
    assert sorted(m.last_residuals) == pytest.approx([0.0, 0.01])
    m.update([A * 3], [R], 2.1)  # new tentative ball: no residual
    assert m.last_residuals == []
