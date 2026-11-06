"""Tests for the ``pnp`` pose estimator on ray-cast RGB-D scenes."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.geometry import (
    quat_from_plane_yaw,
    quat_to_rotation_matrix,
    yaw_error_mod90_deg,
    yaw_on_plane_deg,
)
from g1_perception.core.pnp import project_points, top_face_model
from g1_perception.core.types import Detection, Frame
from g1_perception.pipeline import clear_registry, get_registered, register
from g1_perception.pose.geometric import GeometricPoseEstimator
from g1_perception.pose.pnp import PnpPoseEstimator

SIZE = 0.05
H, W = 480, 640
K = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])


@pytest.fixture(autouse=True)
def _registry():
    clear_registry()
    register('geometric')(GeometricPoseEstimator)
    register('pnp')(PnpPoseEstimator)
    yield
    clear_registry()


def _rot(z: np.ndarray, yaw_deg: float) -> np.ndarray:
    return quat_to_rotation_matrix(quat_from_plane_yaw(z, yaw_deg))


def render(
    *,
    tilt_deg: float = 50.0,
    yaw_deg: float = 25.0,
    dist_m: float = 0.6,
    face_tilt_deg: float = 0.0,
):
    """
    Ray-cast a table and a cube (all faces) seen by a camera pitched down.

    Returns ``(frame, keypoints, center, table_normal)``. ``face_tilt_deg``
    tilts only the returned corners (a cube not lying flat), not the depth.
    """
    t = np.deg2rad(tilt_deg)
    n = np.array([0.0, -np.cos(t), -np.sin(t)])
    on_table = np.array([0.0, 0.0, dist_m])
    center = on_table + SIZE / 2.0 * n
    R = _rot(n, yaw_deg)

    vs, us = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    # Table: n·x - n·on_table = 0 → ray scale s = (n·on_table) / (n·ray).
    s_table = (n @ on_table) / (rays @ n)
    # Cube: slab test in the cube frame (axis-aligned box of half-size S/2).
    o = R.T @ (-center)
    d = rays @ R
    half = SIZE / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (-half - o) / d
        t1 = (half - o) / d
    t_near = np.nanmax(np.minimum(t0, t1), axis=-1)
    t_far = np.nanmin(np.maximum(t0, t1), axis=-1)
    hit = (t_near <= t_far) & (t_near > 0)
    s = np.where(hit, np.minimum(t_near, s_table), s_table)
    depth = (s * rays[..., 2]).astype(np.float32)

    top = center + SIZE / 2.0 * n
    face_R = R
    if face_tilt_deg:
        tilt = np.deg2rad(face_tilt_deg)
        c, sn = np.cos(tilt), np.sin(tilt)
        rx = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
        face_R = R @ rx
    corners = project_points(face_R @ top_face_model(SIZE).T + top[:, None], K)
    keypoints = np.column_stack([corners, np.full(4, 0.9)])

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    frame = Frame(rgb=rgb, depth_m=depth, K=K, stamp=0.0)
    return frame, keypoints, center, n, hit


def _det(keypoints, mask=None) -> Detection:
    if mask is None:
        mask = np.zeros((H, W), dtype=bool)
    return Detection(slot=None, label='red', score=0.9, mask=mask, keypoints=keypoints)


def test_registered_as_pnp():
    assert get_registered('pnp') is PnpPoseEstimator


@pytest.mark.parametrize('yaw_deg', [0.0, 25.0, 45.0, 70.0])
@pytest.mark.parametrize('dist_m', [0.6, 1.0])
def test_pnp_path_recovers_centre_and_yaw(yaw_deg, dist_m):
    frame, kpts, center, n, _ = render(yaw_deg=yaw_deg, dist_m=dist_m)
    pose = PnpPoseEstimator().estimate(frame, _det(kpts[[3, 1, 0, 2]]))
    assert pose is not None and pose.method == 'pnp'
    assert np.linalg.norm(pose.position - center) < 0.005
    assert yaw_error_mod90_deg(yaw_on_plane_deg(pose.quat, n), yaw_deg) < 1.0
    assert pose.score == pytest.approx(0.9)


def test_one_px_corner_noise_keeps_pnp_and_depth_position():
    rng = np.random.default_rng(3)
    frame, kpts, center, n, _ = render(yaw_deg=30.0, dist_m=0.6)
    noisy = kpts.copy()
    noisy[:, :2] += rng.normal(0.0, 1.0, (4, 2))
    pose = PnpPoseEstimator().estimate(frame, _det(noisy))
    assert pose is not None and pose.method == 'pnp'
    assert np.linalg.norm(pose.position - center) < 0.005
    assert yaw_error_mod90_deg(yaw_on_plane_deg(pose.quat, n), 30.0) < 3.0


def test_missing_keypoints_fall_back_to_geometric_on_mask():
    frame, _kpts, center, _n, hit = render()
    pose = PnpPoseEstimator().estimate(frame, _det(None, mask=hit))
    assert pose is not None and pose.method == 'fallback:no_keypoints'
    assert np.linalg.norm(pose.position - center) < 0.02


def test_low_corner_confidence_falls_back():
    frame, kpts, center, _n, _ = render()
    kpts[2, 2] = 0.3
    pose = PnpPoseEstimator(kpt_min_conf=0.5).estimate(frame, _det(kpts))
    assert pose is not None and pose.method == 'fallback:kpt_conf'
    assert np.linalg.norm(pose.position - center) < 0.01


def test_large_reprojection_error_falls_back():
    frame, kpts, _center, _n, _ = render()
    kpts[0, :2] += (8.0, -6.0)  # one corner off: no longer a square
    pose = PnpPoseEstimator(max_reproj_px=2.0).estimate(frame, _det(kpts))
    assert pose is not None and pose.method == 'fallback:reproj'


def test_tilted_face_falls_back():
    frame, kpts, _center, _n, _ = render(face_tilt_deg=35.0)
    pose = PnpPoseEstimator(max_reproj_px=50.0).estimate(frame, _det(kpts))
    assert pose is not None and pose.method == 'fallback:tilt'


def test_depth_disagreeing_with_pnp_falls_back():
    frame, kpts, _center, _n, _ = render(dist_m=0.6)
    far = Frame(rgb=frame.rgb, depth_m=frame.depth_m + np.float32(0.15), K=K, stamp=0.0)
    pose = PnpPoseEstimator().estimate(far, _det(kpts))
    assert pose is not None and pose.method == 'fallback:depth_diff'


def test_depth_threshold_grows_with_range():
    est = PnpPoseEstimator(max_depth_diff_m=0.03, depth_diff_ref_m=0.6)
    assert est.max_depth_diff(0.4) == pytest.approx(0.03)
    assert est.max_depth_diff(1.2) == pytest.approx(0.12)


def test_no_depth_returns_none():
    frame, kpts, _center, _n, _ = render()
    empty = Frame(rgb=frame.rgb, depth_m=np.zeros_like(frame.depth_m), K=K, stamp=0.0)
    assert PnpPoseEstimator().estimate(empty, _det(kpts)) is None


def test_fallback_none_returns_none():
    frame, kpts, _center, _n, _ = render()
    kpts[0, 2] = 0.1
    assert PnpPoseEstimator(fallback='none').estimate(frame, _det(kpts)) is None


def test_bad_fallback_name_raises():
    with pytest.raises(ValueError, match='fallback'):
        PnpPoseEstimator(fallback='centroid')
