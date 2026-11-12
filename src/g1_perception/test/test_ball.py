"""Tests for the ball centre from mask + depth."""

import numpy as np
import pytest

from g1_perception.core.ball import (
    METHOD_AREA,
    METHOD_CIRCLE,
    draw_balls,
    find_balls,
    fit_ball,
    fit_circle,
    format_balls,
    is_ball,
)
from g1_perception.core.types import Detection

K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])
H, W = 480, 640
R = 0.035


def _render_sphere(center, radius=R, background=0.0):
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
    t = np.where(
        mask, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a), background
    )
    return mask, t.astype(np.float32)


def _post(mask, depth, u_from, u_to, z):
    """Cover columns [u_from, u_to) with a post at depth ``z`` in front."""
    mask = mask.copy()
    depth = depth.copy()
    mask[:, u_from:u_to] = False
    depth[:, u_from:u_to] = z
    return mask, depth


def _det(mask, label='red ball', score=0.9):
    return Detection(slot=None, label=label, score=score, mask=mask)


def test_fit_circle_on_an_arc():
    ang = np.linspace(0.0, np.pi, 40)
    pts = np.column_stack((100 + 20 * np.cos(ang), 50 + 20 * np.sin(ang)))
    u, v, r = fit_circle(pts)
    assert (u, v, r) == pytest.approx((100, 50, 20), abs=1e-6)
    assert fit_circle(pts[:2]) is None


@pytest.mark.parametrize('center', [
    (0.0, 0.0, 1.0),
    (0.15, -0.05, 1.5),
    (-0.3, 0.2, 2.5),
])
def test_centre_and_radius_of_a_whole_sphere(center):
    mask, depth = _render_sphere(center, background=3.5)
    fit = fit_ball(mask, depth, K)
    assert fit is not None and fit.method == METHOD_CIRCLE
    assert fit.arc_frac == 1.0
    assert np.linalg.norm(fit.position - np.asarray(center)) < 0.005
    assert fit.radius_m == pytest.approx(R, rel=0.05)
    assert fit.depth_radius_m == pytest.approx(R, rel=0.2)
    assert fit.shape_ok


def test_no_depth_gives_none():
    mask, depth = _render_sphere((0.0, 0.0, 1.0))
    assert fit_ball(mask, np.zeros_like(depth), K) is None
    assert fit_ball(np.zeros((H, W), bool), depth, K) is None


def test_rim_depth_from_background_does_not_move_the_centre():
    mask, depth = _render_sphere((0.0, 0.0, 1.2), background=2.2)
    # Mask a bit too large: its rim reads the wall 1 m behind the ball.
    grown = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            grown[np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)] = True
    fit = fit_ball(grown, depth, K)
    assert fit.position[2] == pytest.approx(1.2, abs=0.01)


def test_half_hidden_ball_keeps_its_centre_and_size():
    center = (0.05, 0.0, 1.0)
    mask, depth = _render_sphere(center, background=2.0)
    u_c = int(K[0, 2] + K[0, 0] * center[0])
    # A post 30 cm in front covers the right half of the ball.
    hidden, depth = _post(mask, depth, u_c, u_c + 40, 0.7)
    fit = fit_ball(hidden, depth, K)
    assert fit.method == METHOD_CIRCLE
    assert 0.4 <= fit.arc_frac < 0.75
    assert np.linalg.norm(fit.position - np.asarray(center)) < 0.006
    assert fit.radius_m == pytest.approx(R, rel=0.08)
    # The mask centroid alone would sit about 4R/(3π) ≈ 1.5 cm to the left.


def test_mostly_hidden_ball_falls_back_to_area_and_is_not_trusted():
    center = (0.0, 0.0, 1.0)
    mask, depth = _render_sphere(center, background=2.0)
    u_c = int(K[0, 2])
    r_px = int(K[0, 0] * R / center[2])
    hidden, depth = _post(mask, depth, u_c - r_px // 2, u_c + r_px + 2, 0.7)
    (ball,) = find_balls([_det(hidden)], depth, K)
    assert ball.fit.method == METHOD_AREA
    assert not ball.trusted and ball.reject_reason == 'hidden'


def test_ball_cut_by_the_image_border_is_fitted():
    center = (0.0, 0.0, 1.0)
    mask, depth = _render_sphere(center, background=2.0)
    shift = -int(np.nonzero(mask)[1].min()) - 10  # 10 px off the left edge
    mask = np.roll(mask, shift, axis=1)
    depth = np.roll(depth, shift, axis=1)
    mask[:, W + shift:] = False
    (ball,) = find_balls([_det(mask)], depth, K, border_px=3)
    assert ball.cut and ball.trusted
    true_u = K[0, 2] + shift
    assert ball.fit.pixel[0] == pytest.approx(true_u, abs=1.0)
    assert ball.radius_m == pytest.approx(R, rel=0.05)


def test_flat_red_disc_is_rejected_by_its_depth_profile():
    mask, _ = _render_sphere((0.0, 0.0, 1.5))
    depth = np.where(mask, 1.5, 3.0).astype(np.float32)  # a poster
    (ball,) = find_balls([_det(mask)], depth, K)
    assert ball.fit.depth_radius_m == pytest.approx(0.0, abs=1e-6)
    assert not ball.trusted and ball.reject_reason == 'shape'
    assert '(shape)' in format_balls([ball])


def test_noisy_far_ball_is_not_rejected_for_shape():
    rng = np.random.default_rng(0)
    center = (0.0, 0.0, 3.5)
    mask, depth = _render_sphere(center, background=5.0)
    # D435 at 3.5 m: several cm of noise per pixel.
    depth = (depth + rng.normal(0.0, 0.05, depth.shape)).astype(np.float32)
    (ball,) = find_balls([_det(mask)], depth, K)
    assert ball.trusted
    assert ball.position[2] == pytest.approx(3.5, abs=0.05)


def test_find_balls_keeps_balls_only_nearest_first():
    near_mask, near_depth = _render_sphere((0.1, 0.0, 1.0))
    far_mask, far_depth = _render_sphere((-0.2, 0.0, 2.0), background=4.0)
    cube_mask, _ = _render_sphere((0.0, 0.15, 1.0), 0.03)
    depth = np.where(near_mask, near_depth, far_depth)
    balls = find_balls(
        [_det(far_mask), _det(cube_mask, 'red cube'), _det(near_mask, ' Red Ball ')],
        depth, K,
    )
    assert [round(b.position[2], 1) for b in balls] == [1.0, 2.0]
    assert format_balls(balls).startswith('1: 1.0')
    assert format_balls([]) == 'none'


def test_is_ball_and_draw():
    assert is_ball('red ball') and not is_ball('red cube')
    assert is_ball('pilka', ['Pilka'])
    mask, depth = _render_sphere((0.0, 0.0, 1.0))
    balls = find_balls([_det(mask)], depth, K)
    rgb = np.zeros((H, W, 3), np.uint8)
    out = draw_balls(rgb, balls, inference_ms=12.0, backend='engine:0')
    assert out.shape == rgb.shape and out.any() and not rgb.any()


def test_second_mask_on_the_same_ball_is_a_duplicate():
    mask, depth = _render_sphere((0.0, 0.0, 1.0), background=2.0)
    shifted = np.roll(mask, 3, axis=1)  # a sloppier mask of the same ball
    balls = find_balls(
        [_det(shifted, score=0.35), _det(mask, score=0.9)], depth, K
    )
    assert sorted(b.reject_reason for b in balls) == ['', 'duplicate']
    assert [b.score for b in balls if b.trusted] == [0.9]


def test_two_touching_balls_are_both_kept():
    a_mask, a_depth = _render_sphere((-R, 0.0, 1.0), background=2.0)
    b_mask, b_depth = _render_sphere((R, 0.0, 1.0), background=2.0)
    depth = np.minimum(np.where(a_mask, a_depth, 9), np.where(b_mask, b_depth, 9))
    depth = np.where(a_mask | b_mask, depth, 2.0).astype(np.float32)
    a_mask &= ~b_mask | (a_depth <= b_depth)
    b_mask &= ~a_mask
    balls = find_balls([_det(a_mask), _det(b_mask)], depth, K)
    assert all(b.trusted for b in balls) and len(balls) == 2
