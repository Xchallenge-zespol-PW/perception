"""Synthetic tests for the cube top-face PnP (IPPE_SQUARE)."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.geometry import (
    quat_from_plane_yaw,
    quat_to_rotation_matrix,
    yaw_error_mod90_deg,
    yaw_on_plane_deg,
)
from g1_perception.core.pnp_solver import (
    TopFaceSolution,
    cube_center_from_top,
    normal_angle_deg,
    pose_on_table,
    project_points,
    reprojection_error_px,
    select_solution,
    solve_top_face,
    solve_top_face_candidates,
    top_face_model,
    yaw_on_table_deg,
)

SIZE = 0.05
# D435i colour 640x480.
K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


def _table_normal(tilt_deg: float) -> np.ndarray:
    """Table "up" in the camera frame for a camera pitched down by ``tilt``."""
    t = np.deg2rad(tilt_deg)
    return np.array([0.0, -np.cos(t), -np.sin(t)])


def _scene(tilt_deg: float, yaw_deg: float, dist_m: float, offset=(0.0, 0.0)):
    """Top-face centre, rotation and projected corners of one cube."""
    n = _table_normal(tilt_deg)
    top = np.array([offset[0], offset[1], dist_m])
    R = quat_to_rotation_matrix(quat_from_plane_yaw(n, yaw_deg))
    corners_3d = R @ top_face_model(SIZE).T + top[:, None]
    return n, top, R, project_points(corners_3d, K)


@pytest.mark.parametrize('tilt_deg', [30.0, 50.0, 70.0])
@pytest.mark.parametrize('yaw_deg', [0.0, 10.0, 30.0, 45.0, 80.0])
@pytest.mark.parametrize('dist_m', [0.6, 1.0])
def test_noise_free_recovers_pose_and_c0_mapping(tilt_deg, yaw_deg, dist_m):
    n, top, _R, corners = _scene(tilt_deg, yaw_deg, dist_m, offset=(0.05, -0.03))
    shuffled = corners[[2, 0, 3, 1]]

    ordered, candidates = solve_top_face_candidates(shuffled, K, SIZE)
    sol = select_solution(candidates, n)
    assert sol is not None

    assert normal_angle_deg(sol.normal, n) < 0.5
    assert np.linalg.norm(sol.translation - top) < 1e-3
    yaw = yaw_on_table_deg(sol.rotation, n)
    assert yaw_error_mod90_deg(yaw, yaw_deg) < 0.1
    # Model point i reprojects onto image corner c_i.
    assert reprojection_error_px(sol, ordered, K, SIZE) < 0.05


@pytest.mark.parametrize('dist_m', [0.6, 1.0])
def test_one_px_noise_yaw_error_below_two_degrees(dist_m):
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(200):
        yaw_true = float(rng.uniform(0.0, 90.0))
        n, _top, _R, corners = _scene(50.0, yaw_true, dist_m)
        noisy = corners + rng.normal(0.0, 1.0, corners.shape)
        sol = solve_top_face(noisy, K, SIZE, table_normal=n)
        assert sol is not None
        errors.append(yaw_error_mod90_deg(yaw_on_table_deg(sol.rotation, n), yaw_true))
    assert float(np.mean(errors)) < 2.0


def test_table_normal_picks_flat_solution_over_lower_error():
    n = np.array([0.0, -0.6, -0.8])
    flat = TopFaceSolution(_rot_with_z(n), np.zeros(3), reproj_px=0.8)
    tilted = TopFaceSolution(_rot_with_z(np.array([0.0, 0.3, -0.95])), np.zeros(3), 0.2)
    assert select_solution([tilted, flat], n) is flat
    # Without a table normal the lower reprojection error wins.
    assert select_solution([tilted, flat]) is tilted
    assert select_solution([], n) is None


def test_near_equal_normals_fall_back_to_reprojection_error():
    n = np.array([0.0, -0.6, -0.8])
    a = TopFaceSolution(_rot_with_z(n), np.zeros(3), reproj_px=0.9)
    nearly = n + np.array([0.02, 0.0, 0.0])
    b = TopFaceSolution(_rot_with_z(nearly / np.linalg.norm(nearly)), np.zeros(3), 0.3)
    assert select_solution([a, b], n) is b


def test_pose_on_table_uses_body_centre_and_table_yaw():
    n, top, _R, corners = _scene(45.0, 30.0, 0.8)
    sol = solve_top_face(corners, K, SIZE, table_normal=n)
    assert sol is not None
    pose = pose_on_table(sol, n, SIZE, score=0.7)
    np.testing.assert_allclose(pose.position, top - SIZE / 2 * n, atol=1e-3)
    assert yaw_error_mod90_deg(yaw_on_plane_deg(pose.quat, n), 30.0) < 0.1
    np.testing.assert_allclose(quat_to_rotation_matrix(pose.quat)[:, 2], n, atol=1e-9)
    assert pose.score == pytest.approx(0.7)

    depth_top = top + np.array([0.01, 0.0, 0.0])
    moved = pose_on_table(sol, n, SIZE, top_center=depth_top)
    np.testing.assert_allclose(moved.position, depth_top - SIZE / 2 * n)


def test_cube_center_from_top_moves_toward_table():
    n = np.array([0.0, 0.0, -2.0])  # not unit on purpose
    np.testing.assert_allclose(
        cube_center_from_top([0.0, 0.0, 1.0], n, 0.05), [0.0, 0.0, 1.025]
    )


def test_top_face_model_order_matches_ippe_square():
    np.testing.assert_allclose(
        top_face_model(0.05),
        [[-0.025, 0.025, 0], [0.025, 0.025, 0], [0.025, -0.025, 0], [-0.025, -0.025, 0]],
    )


def _rot_with_z(z: np.ndarray) -> np.ndarray:
    return quat_to_rotation_matrix(quat_from_plane_yaw(z, 0.0))
