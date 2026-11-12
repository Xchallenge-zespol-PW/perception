"""Tests for the ball centre from mask + depth."""

import numpy as np
import pytest

from g1_perception.core.ball import (
    ball_from_mask,
    draw_balls,
    find_balls,
    format_balls,
    is_ball,
)
from g1_perception.core.types import Detection

K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])
H, W = 480, 640


def _render_sphere(center, radius):
    """Return (mask, depth_m) of a sphere seen by a pinhole camera ``K``."""
    us, vs = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    d = np.stack([
        (us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us),
    ], axis=-1)
    c = np.asarray(center, float)
    # Ray p = t * d (z = t): |t d - c|^2 = r^2.
    a = (d * d).sum(-1)
    b = -2.0 * (d @ c)
    disc = b * b - 4.0 * a * (c @ c - radius ** 2)
    mask = disc >= 0.0
    t = np.where(mask, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a), 0.0)
    return mask, t.astype(np.float32)


def _det(mask, label='red ball', score=0.9):
    return Detection(slot=None, label=label, score=score, mask=mask)


@pytest.mark.parametrize('center', [
    (0.0, 0.0, 1.0),
    (0.15, -0.05, 1.5),
    (-0.3, 0.2, 2.5),
])
def test_centre_and_radius_of_a_sphere(center):
    mask, depth = _render_sphere(center, 0.035)
    est = ball_from_mask(mask, depth, K)
    assert est is not None
    position, radius, _, _, n_depth = est
    assert n_depth >= 10
    assert np.linalg.norm(position - np.asarray(center)) < 0.01
    assert radius == pytest.approx(0.035, rel=0.1)


def test_no_depth_gives_none():
    mask, depth = _render_sphere((0.0, 0.0, 1.0), 0.035)
    assert ball_from_mask(mask, np.zeros_like(depth), K) is None
    assert ball_from_mask(np.zeros((H, W), bool), depth, K) is None


def test_rim_depth_from_background_does_not_move_the_centre():
    mask, depth = _render_sphere((0.0, 0.0, 1.2), 0.035)
    # Mask a bit too large: its rim reads the wall 1 m behind the ball.
    grown = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            grown[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)] = True
    depth = np.where(mask, depth, np.float32(2.2))
    position, *_ = ball_from_mask(grown, depth, K)
    assert position[2] == pytest.approx(1.2, abs=0.01)


def test_find_balls_keeps_balls_only_nearest_first():
    near_mask, near_depth = _render_sphere((0.1, 0.0, 1.0), 0.035)
    far_mask, far_depth = _render_sphere((-0.2, 0.0, 2.0), 0.035)
    cube_mask, _ = _render_sphere((0.0, 0.15, 1.0), 0.03)
    depth = np.where(near_mask, near_depth, far_depth)
    balls = find_balls(
        [_det(far_mask), _det(cube_mask, 'red cube'), _det(near_mask, ' Red Ball ')],
        depth, K,
    )
    assert [round(b.position[2], 1) for b in balls] == [1.0, 2.0]
    assert format_balls(balls).startswith('1: 1.0')
    assert format_balls([]) == 'none'


def test_ball_at_the_border_is_cut():
    mask, depth = _render_sphere((0.0, 0.0, 1.0), 0.035)
    (ball,) = find_balls([_det(mask)], depth, K, border_px=3)
    assert not ball.cut
    left = -int(np.nonzero(mask)[1].min())
    (ball,) = find_balls(
        [_det(np.roll(mask, left, axis=1))], np.roll(depth, left, axis=1), K,
        border_px=3,
    )
    assert ball.cut


def test_is_ball_and_draw():
    assert is_ball('red ball') and not is_ball('red cube')
    assert is_ball('pilka', ['Pilka'])
    mask, depth = _render_sphere((0.0, 0.0, 1.0), 0.035)
    balls = find_balls([_det(mask)], depth, K)
    rgb = np.zeros((H, W, 3), np.uint8)
    out = draw_balls(rgb, balls, inference_ms=12.0, backend='engine:0')
    assert out.shape == rgb.shape and out.any() and not rgb.any()
