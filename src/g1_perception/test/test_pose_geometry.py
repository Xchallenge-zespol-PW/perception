"""Synthetic depth tests for geometric cube pose estimation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from g1_perception.core.config import Roi
from g1_perception.core.geometry import (
    footprint_on_plane,
    points_above_plane,
    cube_pose_on_table,
    deproject,
    plane_basis,
    quad_mask,
    table_plane_around,
    yaw_error_mod90_deg,
    yaw_on_plane_deg,
)
from g1_perception.core.types import Detection, Frame
from g1_perception.pipeline import clear_registry, register
from g1_perception.pose.geometric import GeometricPoseEstimator


@pytest.fixture(autouse=True)
def _register_geometric():
    clear_registry()
    register('geometric')(GeometricPoseEstimator)
    yield
    clear_registry()


def _make_K(
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 160.0,
    cy: float = 120.0,
) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def render_table_and_cube(
    *,
    K: np.ndarray,
    height: int = 240,
    width: int = 320,
    z_table: float = 1.0,
    center_xy: tuple[float, float] = (0.0, 0.0),
    yaw_deg: float = 25.0,
    size: float = 0.05,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[Frame, Detection, Roi, np.ndarray, float, np.ndarray]:
    """
    Render a fronto-parallel table plus a cube top face.

    Returns ``(frame, detection, roi, plane_normal, true_yaw_deg, center)``.
    """
    n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    u, v = plane_basis(n)
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    half = size / 2.0

    center = np.array(
        [center_xy[0], center_xy[1], z_table - size / 2.0],
        dtype=np.float64,
    )
    top_center = center + (size / 2.0) * n

    corners_3d = []
    for lu, lv in (
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half),
    ):
        ru = c * lu - s * lv
        rv = s * lu + c * lv
        corners_3d.append(top_center + ru * u + rv * v)
    corners_3d = np.asarray(corners_3d, dtype=np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us = fx * corners_3d[:, 0] / corners_3d[:, 2] + cx
    vs = fy * corners_3d[:, 1] / corners_3d[:, 2] + cy
    pts2 = np.stack([us, vs], axis=1).astype(np.float32)

    depth = np.full((height, width), np.float32(z_table), dtype=np.float32)
    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask_u8, np.round(pts2).astype(np.int32), 1)
    mask = mask_u8.astype(bool)
    z_top = float(top_center[2])
    depth[mask] = np.float32(z_top)

    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        depth = depth + rng.normal(0.0, noise_std, size=depth.shape).astype(
            np.float32
        )
        depth = np.maximum(depth, np.float32(1e-3))

    pad = 25
    x0 = int(max(0, np.floor(pts2[:, 0].min()) - pad))
    y0 = int(max(0, np.floor(pts2[:, 1].min()) - pad))
    x1 = int(min(width, np.ceil(pts2[:, 0].max()) + pad))
    y1 = int(min(height, np.ceil(pts2[:, 1].max()) + pad))
    roi = Roi(slot=0, x0=x0, y0=y0, x1=x1, y1=y1)

    rgb = np.full((height, width, 3), 180, dtype=np.uint8)
    rgb[mask] = (0, 200, 0)
    frame = Frame(
        rgb=rgb,
        depth_m=depth,
        K=K,
        stamp=0.0,
    )
    det = Detection(
        slot=0,
        label='zielony',
        score=0.9,
        mask=mask,
    )
    return frame, det, roi, n, float(yaw_deg % 90.0), center


def test_deproject_skips_zeros_and_respects_range():
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 1] = 1.0
    depth[1, 2] = 2.0
    depth[2, 2] = 0.5
    mask = depth > 0
    K = _make_K(fx=100, fy=100, cx=0, cy=0)

    pts = deproject(mask, depth, K, z_min=0.6, z_max=1.5)
    assert pts.shape == (1, 3)
    assert np.isclose(pts[0, 2], 1.0)


def test_table_plane_and_cube_pose_direct():
    K = _make_K()
    frame, det, roi, n_true, yaw_true, center_true = render_table_and_cube(
        K=K, yaw_deg=25.0, center_xy=(0.05, -0.03)
    )

    n, d, n_inliers = table_plane_around(
        roi, det.mask, frame.depth_m, frame.K, margin=15, threshold=0.01
    )
    assert n_inliers > 100
    assert abs(np.dot(n, n_true)) > 0.99

    points = deproject(det.mask, frame.depth_m, frame.K)
    pose = cube_pose_on_table(points, (n, d), size=0.05)

    assert np.linalg.norm(pose.position - center_true) < 0.005
    yaw_est = yaw_on_plane_deg(pose.quat, n)
    assert yaw_error_mod90_deg(yaw_est, yaw_true) < 2.0


def test_geometric_pose_estimator_recovers_pose():
    K = _make_K()
    frame, det, roi, n_true, yaw_true, center_true = render_table_and_cube(
        K=K, yaw_deg=35.0, center_xy=(-0.04, 0.02)
    )
    est = GeometricPoseEstimator(rois=[roi], size=0.05, margin=15)

    pose = est.estimate(frame, det)

    assert pose is not None
    assert np.linalg.norm(pose.position - center_true) < 0.005
    n, _d, _ = table_plane_around(
        roi, det.mask, frame.depth_m, frame.K, margin=15, threshold=0.01
    )
    yaw_est = yaw_on_plane_deg(pose.quat, n)
    assert yaw_error_mod90_deg(yaw_est, yaw_true) < 2.0
    assert abs(np.dot(n, n_true)) > 0.99


def test_geometric_pose_with_gaussian_depth_noise():
    K = _make_K()
    frame, det, roi, _n_true, yaw_true, center_true = render_table_and_cube(
        K=K,
        yaw_deg=20.0,
        center_xy=(0.03, 0.01),
        noise_std=0.002,
        seed=7,
    )
    est = GeometricPoseEstimator(
        rois=[roi],
        size=0.05,
        margin=20,
        ransac_threshold=0.008,
    )

    pose = est.estimate(frame, det)

    assert pose is not None
    assert np.linalg.norm(pose.position - center_true) < 0.005
    n, _d, _ = table_plane_around(
        roi,
        det.mask,
        frame.depth_m,
        frame.K,
        margin=20,
        threshold=0.008,
    )
    yaw_est = yaw_on_plane_deg(pose.quat, n)
    assert yaw_error_mod90_deg(yaw_est, yaw_true) < 2.0


def test_quad_mask_fills_square_from_corners():
    corners = np.array([[10.0, 5.0], [19.0, 5.0], [19.0, 14.0], [10.0, 14.0]])
    mask = quad_mask(corners, (20, 30))
    assert mask.shape == (20, 30) and mask.dtype == np.bool_
    assert mask[5:15, 10:20].all()
    assert np.count_nonzero(mask) == 100


def test_quad_mask_accepts_keypoints_and_clips_to_image():
    keypoints = np.array([
        [-5.0, -5.0, 0.9], [8.0, -5.0, 0.9], [8.0, 4.0, 0.2], [-5.0, 4.0, 0.9],
    ])
    mask = quad_mask(keypoints, (10, 10))
    assert mask[:5, :9].all() and not mask[5:, :].any() and not mask[:, 9:].any()


def test_quad_mask_matches_roi_rectangle_of_tilted_quad():
    corners = np.array([[20.0, 2.0], [38.0, 20.0], [20.0, 38.0], [2.0, 20.0]])
    mask = quad_mask(corners, (40, 40))
    assert mask[20, 20] and not mask[3, 3]
    assert abs(np.count_nonzero(mask) - 2 * 18 * 18) < 80


def test_quad_mask_rejects_bad_shape():
    with pytest.raises(ValueError, match='corners'):
        quad_mask(np.zeros((2, 2)), (10, 10))


def test_points_above_plane_and_footprint():
    plane = (np.array([0.0, 0.0, 1.0]), 0.0)
    pts = np.array([
        [0.0, 0.0, 0.03], [0.05, 0.0, 0.05], [0.05, 0.05, 0.05],
        [0.0, 0.05, 0.05], [0.2, 0.0, 0.0], [0.0, 0.0, 0.2],
    ])
    kept, ratio = points_above_plane(pts, plane, 0.01, 0.075)
    assert kept.shape == (4, 3)
    assert np.isclose(ratio, 4 / 6)
    short, long = footprint_on_plane(kept, plane)
    assert np.isclose(short, 0.05, atol=1e-6)
    assert np.isclose(long, 0.05, atol=1e-6)


def test_geometric_rejects_flat_tape():
    K = _make_K()
    frame, det, roi, *_ = render_table_and_cube(K=K, yaw_deg=10.0)
    # Same mask, but lying flat on the table (tape).
    depth = np.full_like(frame.depth_m, 1.0)
    tape = Frame(rgb=frame.rgb, depth_m=depth, K=K, stamp=0.0)
    est = GeometricPoseEstimator(rois=[roi], size=0.05, margin=15)

    assert est.estimate(frame, det) is not None
    assert est.estimate(tape, det) is None
    off = GeometricPoseEstimator(
        rois=[roi], size=0.05, margin=15, min_height=0.0, footprint_min=0.0,
    )
    assert off.estimate(tape, det) is not None


def test_geometric_ignores_mask_bleed_onto_table():
    K = _make_K()
    frame, det, roi, _n, _yaw, center_true = render_table_and_cube(
        K=K, yaw_deg=0.0, center_xy=(0.0, 0.0)
    )
    # Mask spills 3 cm onto the table on one side (12 px at 1 m, fx 500).
    ys, xs = np.nonzero(det.mask)
    bled = det.mask.copy()
    bled[ys.min():ys.max() + 1, xs.max() + 1:xs.max() + 13] = True
    det_bled = Detection(slot=0, label=det.label, score=det.score, mask=bled)
    roi_wide = Roi(slot=0, x0=roi.x0, y0=roi.y0, x1=roi.x1 + 15, y1=roi.y1)
    est = GeometricPoseEstimator(rois=[roi_wide], size=0.05, margin=15)
    raw = GeometricPoseEstimator(
        rois=[roi_wide], size=0.05, margin=15, min_height=0.0,
        footprint_max=0.0,
    )

    pose = est.estimate(frame, det_bled)
    assert pose is not None
    assert np.linalg.norm(pose.position - center_true) < 0.005
    raw_pose = raw.estimate(frame, det_bled)
    assert raw_pose is not None
    assert np.linalg.norm(raw_pose.position - center_true) > 0.01


def test_geometric_rejects_oversized_footprint():
    K = _make_K()
    frame, det, roi, *_ = render_table_and_cube(K=K, size=0.10, yaw_deg=0.0)
    est = GeometricPoseEstimator(
        rois=[roi], size=0.10, margin=15, footprint_max=0.08,
    )
    assert est.estimate(frame, det) is None
