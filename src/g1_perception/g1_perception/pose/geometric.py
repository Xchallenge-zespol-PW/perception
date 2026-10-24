"""Geometric cube pose from table plane and min-area rectangle."""

from __future__ import annotations

from pathlib import Path

from g1_perception.core.config import Roi, load_rois
from g1_perception.core.geometry import (
    cube_pose_on_table,
    deproject,
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
    ) -> None:
        if rois is None:
            self.rois = load_rois(
                Path(rois_path) if rois_path is not None else None
            )
        else:
            self.rois = list(rois)
        self._roi_by_slot = {r.slot: r for r in self.rois}
        self.size = float(size)
        self.margin = int(margin)
        self.z_min = z_min
        self.z_max = z_max
        self.ransac_threshold = float(ransac_threshold)

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        """Return geometric cube pose, or None if estimation fails."""
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
            return cube_pose_on_table(points, (n, d), size=self.size)
        except ValueError:
            return None
