"""3D geometry helpers: deprojection, plane fit, cube pose on a table."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from g1_perception.core.config import Roi
from g1_perception.core.types import Pose

Plane = tuple[NDArray[np.float64], float]  # (unit normal, d) with n·x + d = 0


def deproject(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> NDArray[np.float64]:
    """
    Back-project masked depth pixels to 3D camera-frame points.

    Skips zero depth. Optional ``z_min`` / ``z_max`` further filter metres.
    Returns an ``(N, 3)`` array (possibly empty).
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z = depth_m[ys, xs].astype(np.float64)
    valid = z > 0.0
    if z_min is not None:
        valid &= z >= z_min
    if z_max is not None:
        valid &= z <= z_max
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    z_v = z[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (xs_v - cx) * z_v / fx
    y = (ys_v - cy) * z_v / fy
    return np.column_stack((x, y, z_v))


def fit_plane_ransac(
    points: NDArray[np.float64],
    *,
    threshold: float = 0.005,
    iterations: int = 300,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], float, int]:
    """
    Fit a plane ``n·x + d = 0`` with RANSAC, then SVD-refine inliers.

    The unit normal is oriented so that it points toward the camera origin
    (``d >= 0``). Returns ``(normal, d, n_inliers)``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f'points must be Nx3, got shape {pts.shape}')
    if pts.shape[0] < 3:
        raise ValueError('Need at least 3 points to fit a plane')

    gen = rng if rng is not None else np.random.default_rng(0)
    best_count = -1
    best_n = np.array([0.0, 0.0, 1.0])
    best_d = 0.0

    n_pts = pts.shape[0]
    for _ in range(iterations):
        idx = gen.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n = n / norm
        d = float(-np.dot(n, p0))
        dist = np.abs(pts @ n + d)
        count = int(np.count_nonzero(dist < threshold))
        if count > best_count:
            best_count = count
            best_n, best_d = n, d

    if best_count < 3:
        raise ValueError('RANSAC failed to find a plane with enough inliers')

    inliers = pts[np.abs(pts @ best_n + best_d) < threshold]
    n, d = _fit_plane_svd(inliers)
    n, d = _orient_normal_toward_camera(n, d)
    n_inliers = int(np.count_nonzero(np.abs(pts @ n + d) < threshold))
    return n, d, n_inliers


def roi_from_mask(
    mask: NDArray[np.bool_],
    *,
    slot: int = -1,
) -> Roi:
    """Return the axis-aligned bbox of ``mask`` as an :class:`Roi`."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError('Cannot build ROI from an empty mask')
    return Roi(
        slot=int(slot),
        x0=int(xs.min()),
        y0=int(ys.min()),
        x1=int(xs.max()) + 1,
        y1=int(ys.max()) + 1,
    )


def quad_mask(
    corners: NDArray[np.float64],
    shape: tuple[int, int],
) -> NDArray[np.bool_]:
    """
    Return an HxW bool mask of the filled polygon through ``corners``.

    ``corners`` is ``(N, 2)`` pixels ``(u, v)`` in polygon order (for the
    cube top face: ``c0..c3``); Kx3 keypoints are accepted and their
    confidence column is ignored. Parts outside the image are clipped.
    For ``yolo_pose`` this mask covers only the top face, not the sides.
    """
    pts = np.asarray(corners, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] not in (2, 3):
        raise ValueError(
            f'Expected (N >= 3, 2) or (N, 3) corners, got {pts.shape}'
        )
    height, width = int(shape[0]), int(shape[1])
    canvas = np.zeros((height, width), dtype=np.uint8)
    polygon = np.round(pts[:, :2]).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [polygon], 1)
    return canvas.astype(bool)


def table_plane_around(
    roi: Roi,
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
    margin: int,
    *,
    z_min: float | None = None,
    z_max: float | None = None,
    **ransac_kwargs: Any,
) -> tuple[NDArray[np.float64], float, int]:
    """
    Estimate the table plane from depth around an ROI, excluding ``mask``.

    Builds a rectangle expanded by ``margin`` pixels, removes object pixels,
    deprojects, and runs :func:`fit_plane_ransac`.
    """
    h, w = depth_m.shape[:2]
    region = np.zeros((h, w), dtype=bool)
    y0 = max(int(roi.y0) - int(margin), 0)
    x0 = max(int(roi.x0) - int(margin), 0)
    y1 = min(int(roi.y1) + int(margin), h)
    x1 = min(int(roi.x1) + int(margin), w)
    region[y0:y1, x0:x1] = True
    region &= ~mask

    points = deproject(region, depth_m, K, z_min=z_min, z_max=z_max)
    return fit_plane_ransac(points, **ransac_kwargs)


def cube_pose_on_table(
    points: NDArray[np.float64],
    plane: Plane,
    size: float = 0.05,
) -> Pose:
    """
    Estimate cube pose from object points sitting on a known table plane.

    Projects points onto the plane, fits ``cv2.minAreaRect`` in the plane
    basis, takes yaw modulo 90°, places the centre at the rect centre lifted
    by ``size/2`` along the plane normal (local +Z).
    """
    n, d = plane
    n = np.asarray(n, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        raise ValueError('Need at least 4 points for cube_pose_on_table')

    projected = project_points_to_plane(pts, n, d)
    origin = projected.mean(axis=0)
    u, v = plane_basis(n)

    coords = np.column_stack(
        ((projected - origin) @ u, (projected - origin) @ v)
    ).astype(np.float32)
    rect = cv2.minAreaRect(coords)
    (c2x, c2y), (_w, _h), angle_deg = rect

    yaw_deg = float(angle_deg) % 90.0

    center_on_plane = origin + c2x * u + c2y * v
    position = center_on_plane + (size / 2.0) * n
    quat = quat_from_plane_yaw(n, yaw_deg)

    # Score: fraction of points close to the plane (relative consistency).
    dist = np.abs(pts @ n + d)
    score = float(np.mean(dist < max(size * 0.6, 0.01)))

    return Pose(position=position, quat=quat, score=score)


def project_points_to_plane(
    points: NDArray[np.float64],
    n: NDArray[np.float64],
    d: float,
) -> NDArray[np.float64]:
    """Orthogonally project points onto plane ``n·x + d = 0``."""
    dist = points @ n + d
    return points - np.outer(dist, n)


def plane_basis(
    n: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return orthonormal ``(u, v)`` spanning the plane with normal ``n``.

    ``u, v, n`` form a right-handed frame.
    """
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(n[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(n, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def quat_from_plane_yaw(
    n: NDArray[np.float64],
    yaw_deg: float,
) -> NDArray[np.float64]:
    """
    Quaternion ``(x, y, z, w)`` with +Z along ``n`` and X at ``yaw_deg``.

    Yaw is measured in the :func:`plane_basis` of ``n`` (inverse of
    :func:`yaw_on_plane_deg`).
    """
    u, v = plane_basis(n)
    yaw = np.deg2rad(yaw_deg)
    z_axis = np.asarray(n, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cos(yaw) * u + np.sin(yaw) * v
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix_to_quat_xyzw(R)


def transform_pose(
    pose: Pose,
    translation: NDArray[np.float64],
    rotation_xyzw: NDArray[np.float64],
) -> Pose:
    """
    Apply a rigid transform ``(translation, rotation)`` to ``pose``.

    The transform maps source-frame coordinates to the target frame
    (``p_target = R p_source + t``), as returned by a TF lookup of
    ``target <- source``. ``score`` and ``method`` are kept.
    """
    R_t = quat_to_rotation_matrix(rotation_xyzw)
    position = R_t @ np.asarray(pose.position, dtype=np.float64)
    position = position + np.asarray(translation, dtype=np.float64)
    R = R_t @ quat_to_rotation_matrix(pose.quat)
    return Pose(
        position=position,
        quat=rotation_matrix_to_quat_xyzw(R),
        score=pose.score,
        method=pose.method,
    )


def rotation_matrix_to_quat_xyzw(
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a unit quaternion ``(x, y, z, w)``."""
    m = np.asarray(R, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quat_to_rotation_matrix(
    quat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert quaternion ``(x, y, z, w)`` to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def yaw_on_plane_deg(
    quat: NDArray[np.float64],
    n: NDArray[np.float64],
) -> float:
    """
    Extract cube yaw (degrees) in the plane basis, modulo 90°.

    Uses the X column of the rotation matrix projected onto ``plane_basis(n)``.
    """
    R = quat_to_rotation_matrix(quat)
    x_axis = R[:, 0]
    u, v = plane_basis(n)
    yaw = np.arctan2(float(x_axis @ v), float(x_axis @ u))
    return float(np.rad2deg(yaw) % 90.0)


def yaw_error_mod90_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two angles modulo 90°."""
    d = abs(a_deg - b_deg) % 90.0
    return float(min(d, 90.0 - d))


def _fit_plane_svd(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Least-squares plane through points via SVD."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    n = vh[-1, :]
    n = n / np.linalg.norm(n)
    d = float(-np.dot(n, centroid))
    return n, d


def _orient_normal_toward_camera(
    n: NDArray[np.float64],
    d: float,
) -> tuple[NDArray[np.float64], float]:
    """Flip plane so the unit normal points toward the camera origin."""
    if d < 0.0:
        return -n, -d
    return n, d
