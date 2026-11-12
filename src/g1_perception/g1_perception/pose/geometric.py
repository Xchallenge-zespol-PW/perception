"""Geometric cube pose from table plane and min-area rectangle."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from g1_perception.core.config import Roi, load_rois
from g1_perception.core.geometry import (
    Plane,
    cube_pose_on_table,
    deproject,
    footprint_on_plane,
    points_above_plane,
    roi_from_mask,
    table_plane_around,
    table_plane_around_rois,
)
from g1_perception.core.types import Detection, Frame, Pose
from g1_perception.pipeline import register


@register('geometric')
class GeometricPoseEstimator:
    """
    Estimate cube pose with :func:`cube_pose_on_table`.

    Deprojects the detection mask, fits a table plane around the ROI
    (excluding the object), then recovers centre and yaw modulo 90°.
    When ``det.slot`` is unknown, the mask bounding box is used as the ROI
    (free-detection / ``rois_mode=none``).

    Shape check (rejects tape and other flat or oversized objects): only
    mask points ``min_height`` .. ``1.5 * size`` above the table are kept,
    at least ``min_above_ratio`` of them must be kept, and their rectangle
    on the table must be ``footprint_min`` .. ``footprint_max`` (m).
    ``min_height <= 0`` turns off the height part, ``0`` a footprint bound.

    :meth:`estimate_many` places all detections of a frame on one table
    plane, fitted around all of them with every mask cut out. The cube
    height comes from the plane alone (its centre is ``size / 2`` above
    it), and :meth:`estimate` leaves out only the cube's own mask: in a
    column of cubes, whose gaps the front cubes hide from the camera, the
    window of a middle cube is up to half its neighbours, whose tops lie
    in one plane ``size`` above the table, and RANSAC could take that
    plane, putting the cube up to ``size`` too high.
    """

    def __init__(
        self,
        rois: list[Roi] | None = None,
        *,
        rois_path: str | Path | None = None,
        size: float = 0.05,
        margin: int = 20,
        z_min: float | None = None,
        z_max: float | None = None,
        ransac_threshold: float = 0.005,
        min_height: float = 0.01,
        min_above_ratio: float = 0.5,
        footprint_min: float = 0.02,
        footprint_max: float = 0.13,
    ) -> None:
        if rois is None and rois_path is None:
            self.rois = []
        elif rois is None:
            self.rois = load_rois(Path(rois_path) if rois_path else None)
        else:
            self.rois = list(rois)
        self._roi_by_slot = {r.slot: r for r in self.rois}
        self.size = float(size)
        self.margin = int(margin)
        self.z_min = z_min
        self.z_max = z_max
        self.ransac_threshold = float(ransac_threshold)
        self.min_height = float(min_height)
        self.min_above_ratio = float(min_above_ratio)
        self.footprint_min = float(footprint_min)
        self.footprint_max = float(footprint_max)

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        """Return geometric cube pose, or None if estimation fails."""
        roi = self._roi(det)
        if roi is None:
            return None

        points = self._mask_points(frame, det)
        if points.shape[0] < 4:
            return None

        try:
            n, d, _n_inliers = table_plane_around(
                roi,
                det.mask,
                frame.depth_m,
                frame.K,
                self.margin,
                z_min=self.z_min,
                z_max=self.z_max,
                threshold=self.ransac_threshold,
            )
        except ValueError:
            return None
        return self._pose_on(points, (n, d))

    def estimate_many(
        self,
        frame: Frame,
        dets: Sequence[Detection],
    ) -> list[Pose | None]:
        """
        Return the pose of every detection of ``frame`` on one table plane.

        The plane is fitted around all detections (each ROI expanded by
        ``margin``) without the pixels of any of their masks, so neither a
        neighbouring cube nor the cube itself enters it, and a cube whose
        own surroundings have no depth still gets the table of the others.
        A pose is None where :meth:`estimate` would fail on that plane, and
        every pose is None when no plane can be fitted.
        """
        rois = [self._roi(det) for det in dets]
        known = [roi for roi in rois if roi is not None]
        if not known:
            return [None] * len(dets)

        exclude = np.zeros(frame.depth_m.shape[:2], dtype=bool)
        for det in dets:
            exclude |= det.mask
        try:
            n, d, _n_inliers = table_plane_around_rois(
                known,
                exclude,
                frame.depth_m,
                frame.K,
                self.margin,
                z_min=self.z_min,
                z_max=self.z_max,
                threshold=self.ransac_threshold,
            )
        except ValueError:
            return [None] * len(dets)

        poses: list[Pose | None] = []
        for det, roi in zip(dets, rois):
            points = self._mask_points(frame, det)
            if roi is None or points.shape[0] < 4:
                poses.append(None)
                continue
            poses.append(self._pose_on(points, (n, d)))
        return poses

    def _roi(self, det: Detection) -> Roi | None:
        """Return the ROI of the slot, or the mask's box; None if empty."""
        if det.slot is not None and det.slot in self._roi_by_slot:
            return self._roi_by_slot[det.slot]
        try:
            return roi_from_mask(det.mask)
        except ValueError:
            return None

    def _mask_points(self, frame: Frame, det: Detection) -> NDArray[np.float64]:
        """Return the 3D points of the detection mask with valid depth."""
        return deproject(
            det.mask,
            frame.depth_m,
            frame.K,
            z_min=self.z_min,
            z_max=self.z_max,
        )

    def _pose_on(
        self,
        points: NDArray[np.float64],
        plane: Plane,
    ) -> Pose | None:
        """Return the cube pose of mask points on the table plane, or None."""
        try:
            cube_points = self._cube_points(points, plane)
            if cube_points is None:
                return None
            return cube_pose_on_table(cube_points, plane, size=self.size)
        except ValueError:
            return None

    def _cube_points(
        self,
        points: NDArray[np.float64],
        plane: Plane,
    ) -> NDArray[np.float64] | None:
        """Return the points standing above the table, or None if not a cube."""
        if self.min_height > 0.0:
            points, above = points_above_plane(
                points, plane, self.min_height, 1.5 * self.size
            )
            if above < self.min_above_ratio or points.shape[0] < 4:
                return None
        short, long = footprint_on_plane(points, plane)
        if short < self.footprint_min:
            return None
        if self.footprint_max > 0.0 and long > self.footprint_max:
            return None
        return points
