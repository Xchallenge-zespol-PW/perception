"""Cube top-face pose from 4 image corners with OpenCV IPPE_SQUARE."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from g1_perception.core.geometry import (
    plane_basis,
    quat_from_plane_yaw,
)
from g1_perception.core.keypoints import N_CORNERS, order_top_corners
from g1_perception.core.types import Pose

# Two IPPE solutions whose normals agree with the table within this many
# degrees are both plausible; the lower reprojection error then decides.
NORMAL_TIE_DEG = 5.0
# +90° about model z: model corner j goes to corner j - 1 (mod 4).
_TURN_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class TopFaceSolution:
    """
    One PnP solution for the top face, in the camera frame.

    ``rotation`` maps model points (``top_face_model``) to the camera,
    ``translation`` is the top-face centre [m], ``normal`` is the model
    ``+z`` (out of the top face) and ``reproj_px`` is the RMS reprojection
    error of the 4 corners.
    """

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    reproj_px: float

    @property
    def normal(self) -> NDArray[np.float64]:
        """Model ``+z`` in the camera frame (top-face normal)."""
        return self.rotation[:, 2]


def top_face_model(size: float) -> NDArray[np.float64]:
    """
    Square of side ``size`` in the order required by ``SOLVEPNP_IPPE_SQUARE``.

    ``[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]`` with ``s = size / 2``;
    index ``i`` corresponds to image corner ``c_i`` (``docs/pose_pnp.md``).
    """
    s = float(size) / 2.0
    return np.array(
        [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]],
        dtype=np.float64,
    )


def solve_top_face_candidates(
    corners: ArrayLike,
    K: NDArray[np.float64],
    size: float,
    dist_coeffs: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], list[TopFaceSolution]]:
    """
    Run IPPE_SQUARE on 4 corners in any order.

    Returns the corners in ``c0..c3`` order and every solution whose
    top face points toward the camera (normal against the viewing ray).
    IPPE_SQUARE (OpenCV 4.5) occasionally misses the exact pose for one
    cyclic start of the corners, so all 4 starts are solved; each
    rotation is expressed for ``c0..c3`` (a start shift is a 90° turn of
    the square) and its error is recomputed against ``c0..c3``.
    """
    ordered = order_top_corners(corners)
    K = np.asarray(K, dtype=np.float64)
    dist = (
        np.zeros(5) if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64)
    )
    model = top_face_model(size)
    solutions: list[TopFaceSolution] = []
    for k in range(N_CORNERS):
        shifted = np.roll(ordered, -k, axis=0)
        n_sol, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            model,
            shifted.reshape(N_CORNERS, 1, 2),
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        # Model point j-k lands on c_j; _TURN_90 maps model j to j-1.
        turn = np.linalg.matrix_power(_TURN_90, k)
        for i in range(int(n_sol)):
            rotation, _ = cv2.Rodrigues(rvecs[i])
            rotation = rotation @ turn
            translation = np.asarray(tvecs[i], dtype=np.float64).reshape(3)
            if translation[2] <= 0.0 or rotation[:, 2] @ translation >= 0.0:
                continue  # behind the camera, or the face points away from it
            candidate = TopFaceSolution(rotation, translation, 0.0)
            err = reprojection_error_px(candidate, ordered, K, size)
            solutions.append(TopFaceSolution(rotation, translation, err))
    return ordered, solutions


def select_solution(
    solutions: Sequence[TopFaceSolution],
    table_normal: ArrayLike | None = None,
    tie_deg: float = NORMAL_TIE_DEG,
) -> TopFaceSolution | None:
    """
    Pick the IPPE solution that lies flat on the table.

    With ``table_normal`` (unit, toward the camera) the solution with the
    normal closest to it wins; solutions within ``tie_deg`` of the best
    are decided by reprojection error. Without it: lowest error.
    """
    if not solutions:
        return None
    if table_normal is None:
        return min(solutions, key=lambda s: s.reproj_px)
    n = np.asarray(table_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    angles = [normal_angle_deg(s.normal, n) for s in solutions]
    best = min(angles)
    close = [s for s, a in zip(solutions, angles) if a <= best + tie_deg]
    return min(close, key=lambda s: s.reproj_px)


def solve_top_face(
    corners: ArrayLike,
    K: NDArray[np.float64],
    size: float,
    table_normal: ArrayLike | None = None,
    dist_coeffs: ArrayLike | None = None,
) -> TopFaceSolution | None:
    """Solve and select the top-face pose; None if IPPE gives no valid pose."""
    _ordered, solutions = solve_top_face_candidates(corners, K, size, dist_coeffs)
    return select_solution(solutions, table_normal)


def yaw_on_table_deg(
    rotation: NDArray[np.float64],
    table_normal: ArrayLike,
) -> float:
    """
    Yaw of the model ``+x`` projected on the table, modulo 90°.

    Measured in ``plane_basis(table_normal)`` like ``yaw_on_plane_deg``,
    so ``pnp`` and ``geometric`` yaws are directly comparable.
    """
    n = np.asarray(table_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    x_axis = np.asarray(rotation, dtype=np.float64)[:, 0]
    x_axis = x_axis - (x_axis @ n) * n
    if np.linalg.norm(x_axis) < 1e-9:
        raise ValueError('Model x axis is parallel to the table normal')
    u, v = plane_basis(n)
    return float(np.rad2deg(np.arctan2(x_axis @ v, x_axis @ u)) % 90.0)


def cube_center_from_top(
    top_center: ArrayLike,
    table_normal: ArrayLike,
    size: float,
) -> NDArray[np.float64]:
    """Body centre: top-face centre moved ``size / 2`` toward the table."""
    n = np.asarray(table_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return np.asarray(top_center, dtype=np.float64) - (float(size) / 2.0) * n


def pose_on_table(
    solution: TopFaceSolution,
    table_normal: ArrayLike,
    size: float,
    *,
    top_center: ArrayLike | None = None,
    score: float = 1.0,
) -> Pose:
    """
    Cube :class:`Pose`: ``+Z`` = table normal, yaw from PnP, centre of body.

    ``top_center`` (e.g. from depth) replaces the PnP translation, which is
    noisy for small cubes (``docs/pose_pnp.md``).
    """
    n = np.asarray(table_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    yaw = yaw_on_table_deg(solution.rotation, n)
    top = solution.translation if top_center is None else top_center
    return Pose(
        position=cube_center_from_top(top, n, size),
        quat=quat_from_plane_yaw(n, yaw),
        score=float(score),
    )


def reprojection_error_px(
    solution: TopFaceSolution,
    corners: ArrayLike,
    K: NDArray[np.float64],
    size: float,
) -> float:
    """RMS pixel error of ``solution`` against corners in ``c0..c3`` order."""
    projected = project_points(
        solution.rotation @ top_face_model(size).T + solution.translation[:, None],
        K,
    )
    diff = projected - np.asarray(corners, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def project_points(
    points_cam: NDArray[np.float64],
    K: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pinhole projection of ``(3, N)`` camera points to ``(N, 2)`` pixels."""
    uvw = np.asarray(K, dtype=np.float64) @ np.asarray(points_cam, dtype=np.float64)
    return (uvw[:2] / uvw[2]).T


def normal_angle_deg(a: ArrayLike, b: ArrayLike) -> float:
    """Angle between two directions in degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))
