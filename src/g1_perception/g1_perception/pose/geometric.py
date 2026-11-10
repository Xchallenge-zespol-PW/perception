"""Geometric cube pose from table plane and min-area rectangle."""

from __future__ import annotations

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
        if det.slot is not None and det.slot in self._roi_by_slot:
            roi = self._roi_by_slot[det.slot]
        else:
            try:
                roi = roi_from_mask(det.mask)
            except ValueError:
                return None

        points = deproject(
            det.mask,
            frame.depth_m,
            frame.K,
            z_min=self.z_min,
            z_max=self.z_max,
        )
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
            plane = (n, d)
            points = self._cube_points(points, plane)
            if points is None:
                return None
            return cube_pose_on_table(points, plane, size=self.size)
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
