"""Centroid pose estimator: median 3D point, orientation from table plane."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from g1_perception.core.config import Roi, load_rois
from g1_perception.core.geometry import (
    deproject,
    plane_basis,
    rotation_matrix_to_quat_xyzw,
    table_plane_around,
)
from g1_perception.core.types import Detection, Frame, Pose
from g1_perception.pipeline import register


@register('centroid')
class CentroidPoseEstimator:
    """
    Estimate pose as the median of deprojected mask points.

    Orientation uses the table-plane normal as +Z and an arbitrary in-plane
    X axis from :func:`plane_basis` (yaw not observed).
    """

    def __init__(
        self,
        rois: list[Roi] | None = None,
        *,
        rois_path: str | Path | None = None,
        margin: int = 20,
        z_min: float | None = None,
        z_max: float | None = None,
        ransac_threshold: float = 0.005,
    ) -> None:
        if rois is None:
            self.rois = load_rois(
                Path(rois_path) if rois_path is not None else None
            )
        else:
            self.rois = list(rois)
        self._roi_by_slot = {r.slot: r for r in self.rois}
        self.margin = int(margin)
        self.z_min = z_min
        self.z_max = z_max
        self.ransac_threshold = float(ransac_threshold)

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        """Return median position with plane-aligned orientation, or None."""
        if det.slot is None or det.slot not in self._roi_by_slot:
            return None
        roi = self._roi_by_slot[det.slot]

        points = deproject(
            det.mask,
            frame.depth_m,
            frame.K,
            z_min=self.z_min,
            z_max=self.z_max,
        )
        if points.shape[0] == 0:
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

        position = np.median(points, axis=0).astype(np.float64)
        u, v = plane_basis(n)
        R = np.column_stack((u, v, n))
        quat = rotation_matrix_to_quat_xyzw(R)

        return Pose(position=position, quat=quat, score=float(det.score))
