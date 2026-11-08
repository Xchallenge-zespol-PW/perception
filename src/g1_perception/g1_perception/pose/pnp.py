"""Cube pose from YOLO-pose corners (PnP orientation) and depth (position)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from g1_perception.core.config import Roi
from g1_perception.core.geometry import (
    cube_pose_on_table,
    deproject,
    quad_mask,
    table_plane_around,
)
from g1_perception.core.keypoints import N_CORNERS, order_top_corners
from g1_perception.core.pnp import (
    normal_angle_deg,
    pose_on_table,
    solve_top_face,
)
from g1_perception.core.types import Detection, Frame, Pose
from g1_perception.pipeline import register
from g1_perception.pose.geometric import GeometricPoseEstimator

FALLBACK_GEOMETRIC = 'geometric'
FALLBACK_NONE = 'none'
FALLBACKS = (FALLBACK_GEOMETRIC, FALLBACK_NONE)
METHOD_PNP = 'pnp'


class _Reject(Exception):
    """PnP path rejected; ``reason`` goes into ``Pose.method``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@register('pnp')
class PnpPoseEstimator:
    """
    Orientation from the 4 top-face corners (IPPE_SQUARE), position from depth.

    Steps (``docs/pose_pnp.md``): table plane from depth around the cube,
    corners ``c0..c3`` → PnP, solution closest to the table normal, yaw on
    the table; top-face centre = median depth point inside the eroded
    corner quad, moved ``size / 2`` toward the table. The PnP path is
    rejected when a corner is missing or below ``kpt_min_conf``, the
    reprojection error exceeds ``max_reproj_px``, the face normal is more
    than ``max_normal_deg`` off the table normal, or the PnP translation is
    farther than ``max_depth_diff_m · (z / depth_diff_ref_m)²`` from the
    depth centre (PnP distance noise grows with ``z²``). On rejection
    ``fallback='geometric'`` returns the minAreaRect pose (top-face points
    when corners exist, else the detection mask), ``'none'`` returns None.
    ``Pose.method`` is ``pnp`` or ``fallback:<reason>``.
    """

    def __init__(
        self,
        *,
        size: float = 0.05,
        margin: int = 20,
        z_min: float | None = None,
        z_max: float | None = None,
        ransac_threshold: float = 0.005,
        kpt_min_conf: float = 0.5,
        max_reproj_px: float = 2.0,
        max_depth_diff_m: float = 0.03,
        depth_diff_ref_m: float = 0.6,
        max_normal_deg: float = 20.0,
        erode_px: int = 2,
        min_depth_points: int = 10,
        fallback: str = FALLBACK_GEOMETRIC,
        rois: list[Roi] | None = None,
        rois_path: str | Path | None = None,
    ) -> None:
        if fallback not in FALLBACKS:
            raise ValueError(f'fallback must be one of {FALLBACKS}, got {fallback!r}')
        self.size = float(size)
        self.margin = int(margin)
        self.z_min = z_min
        self.z_max = z_max
        self.ransac_threshold = float(ransac_threshold)
        self.kpt_min_conf = float(kpt_min_conf)
        self.max_reproj_px = float(max_reproj_px)
        self.max_depth_diff_m = float(max_depth_diff_m)
        self.depth_diff_ref_m = float(depth_diff_ref_m)
        self.max_normal_deg = float(max_normal_deg)
        self.erode_px = int(erode_px)
        self.min_depth_points = int(min_depth_points)
        self.fallback = fallback
        self._geometric = GeometricPoseEstimator(
            rois,
            rois_path=rois_path,
            size=size,
            margin=margin,
            z_min=z_min,
            z_max=z_max,
            ransac_threshold=ransac_threshold,
        )

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        """Return the PnP pose, the fallback pose, or None."""
        corners: NDArray[np.float64] | None = None
        plane: tuple[NDArray[np.float64], float] | None = None
        try:
            corners = self._corners(det)
            plane = self._table_plane(frame, corners)
            return self._pnp_pose(frame, det, corners, plane)
        except _Reject as rej:
            return self._fallback(frame, det, corners, plane, rej.reason)

    def _corners(self, det: Detection) -> NDArray[np.float64]:
        """Corners ``c0..c3`` from ``det.keypoints`` (4x3)."""
        kpts = det.keypoints
        if kpts is None or np.asarray(kpts).shape != (N_CORNERS, 3):
            raise _Reject('no_keypoints')
        return order_top_corners(np.asarray(kpts, dtype=np.float64)[:, :2])

    def _table_plane(
        self,
        frame: Frame,
        corners: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """
        Table plane around the cube, excluding the whole cube.

        The corner quad covers only the top face; the sides lie within one
        edge length of it, so the quad's box grown by its longest edge is
        excluded before the ``margin`` ring is fitted.
        """
        roi = _exclusion_roi(corners, frame.depth_m.shape[:2])
        exclude = np.zeros(frame.depth_m.shape[:2], dtype=bool)
        exclude[roi.y0:roi.y1, roi.x0:roi.x1] = True
        try:
            n, d, _ = table_plane_around(
                roi,
                exclude,
                frame.depth_m,
                frame.K,
                self.margin,
                z_min=self.z_min,
                z_max=self.z_max,
                threshold=self.ransac_threshold,
            )
        except ValueError as exc:
            raise _Reject('no_table') from exc
        return n, d

    def _pnp_pose(
        self,
        frame: Frame,
        det: Detection,
        corners: NDArray[np.float64],
        plane: tuple[NDArray[np.float64], float],
    ) -> Pose:
        n, _d = plane
        score = float(np.asarray(det.keypoints, dtype=np.float64)[:, 2].min())
        if score < self.kpt_min_conf:
            raise _Reject('kpt_conf')
        top_center = self._depth_top_center(frame, corners)
        solution = solve_top_face(corners, frame.K, self.size, table_normal=n)
        if solution is None:
            raise _Reject('pnp_failed')
        if solution.reproj_px > self.max_reproj_px:
            raise _Reject('reproj')
        if normal_angle_deg(solution.normal, n) > self.max_normal_deg:
            raise _Reject('tilt')
        diff = float(np.linalg.norm(solution.translation - top_center))
        if diff > self.max_depth_diff(float(top_center[2])):
            raise _Reject('depth_diff')
        pose = pose_on_table(solution, n, self.size, top_center=top_center, score=score)
        return replace(pose, method=METHOD_PNP)

    def max_depth_diff(self, z: float) -> float:
        """Return the allowed PnP vs depth distance at range ``z`` [m]."""
        scale = max(z / self.depth_diff_ref_m, 1.0)
        return self.max_depth_diff_m * scale * scale

    def _depth_top_center(
        self,
        frame: Frame,
        corners: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Median 3D point of the eroded top-face quad."""
        points = self._face_points(frame, corners)
        if points.shape[0] < self.min_depth_points:
            raise _Reject('no_depth')
        return np.median(points, axis=0)

    def _face_points(
        self,
        frame: Frame,
        corners: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Depth points inside the corner quad eroded by ``erode_px``."""
        face = quad_mask(corners, frame.depth_m.shape[:2])
        if self.erode_px > 0:
            kernel = np.ones((2 * self.erode_px + 1,) * 2, dtype=np.uint8)
            eroded = cv2.erode(face.astype(np.uint8), kernel).astype(bool)
            if eroded.any():
                face = eroded
        return deproject(
            face, frame.depth_m, frame.K, z_min=self.z_min, z_max=self.z_max
        )

    def _fallback(
        self,
        frame: Frame,
        det: Detection,
        corners: NDArray[np.float64] | None,
        plane: tuple[NDArray[np.float64], float] | None,
        reason: str,
    ) -> Pose | None:
        if self.fallback == FALLBACK_NONE:
            return None
        if corners is None:
            pose = self._geometric.estimate(frame, det)
        elif plane is None:
            return None
        else:
            # Top-face points only: quad-edge pixels that hit the table
            # behind the cube would stretch the minAreaRect.
            n, d = plane
            points = self._face_points(frame, corners)
            height = points @ n + d
            points = points[np.abs(height - self.size) < self.size / 2.0]
            if points.shape[0] < 4:
                return None
            pose = cube_pose_on_table(points, plane, size=self.size)
        if pose is None:
            return None
        return replace(pose, method=f'fallback:{reason}')


def _exclusion_roi(corners: NDArray[np.float64], shape: tuple[int, int]) -> Roi:
    """Corner box grown by the longest quad edge, clipped to the image."""
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    grow = int(np.ceil(edges.max()))
    height, width = int(shape[0]), int(shape[1])
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - grow
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + grow + 1
    return Roi(
        slot=-1,
        x0=max(int(x0), 0),
        y0=max(int(y0), 0),
        x1=min(int(x1), width),
        y1=min(int(y1), height),
    )
