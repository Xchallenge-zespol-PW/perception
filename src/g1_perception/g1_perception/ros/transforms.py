"""TF lookups and camera-motion gating for the perception node."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
)

from g1_perception.core.geometry import transform_pose
from g1_perception.core.types import SlotResult
from g1_perception.core.visibility import angular_speed_deg_s
from g1_perception.core.voting import VotedSlot


class FrameTransforms:
    """Look up ``target <- source`` and transform poses into ``target_frame``."""

    def __init__(
        self,
        buffer: Buffer,
        logger: Any,
        *,
        target_frame: str,
        tf_timeout_s: float,
        max_rot_deg_s: float,
    ) -> None:
        self._buffer = buffer
        self._logger = logger
        self._target_frame = target_frame
        self._tf_timeout_s = tf_timeout_s
        self._max_rot_deg_s = max_rot_deg_s
        # (stamp, rotation_xyzw) of the camera in target_frame, last frame.
        self._prev_camera: tuple[float, np.ndarray] | None = None

    def lookup(
        self,
        target: str,
        source: str,
        header: Header,
        consequence: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(translation, rotation_xyzw)`` of ``target <- source``."""
        try:
            tf = self._buffer.lookup_transform(
                target,
                source,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=float(self._tf_timeout_s)),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as exc:
            self._logger.warning(
                f'No TF {source} -> {target} ({exc}); {consequence}',
                throttle_duration_sec=5.0,
            )
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])

    def lookup_target(
        self,
        camera_header: Header,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Return ``(translation, rotation_xyzw, target)`` or None."""
        target = self._target_frame
        source = camera_header.frame_id
        if not target or target == source:
            return None
        tf = self.lookup(
            target, source, camera_header, f'publishing in {source}'
        )
        if tf is None:
            return None
        return tf[0], tf[1], target

    def voted_to_target(
        self,
        voted: list[VotedSlot],
        camera_header: Header,
    ) -> tuple[list[VotedSlot], str]:
        """Transform voted poses to ``target_frame``; fall back to camera."""
        tf = self.lookup_target(camera_header)
        if tf is None:
            return voted, camera_header.frame_id
        translation, rotation, target = tf
        return [
            replace(v, pose=transform_pose(v.pose, translation, rotation))
            if v.pose is not None else v
            for v in voted
        ], target

    def results_to_target(
        self,
        results: list[SlotResult],
        camera_header: Header,
    ) -> tuple[list[SlotResult] | None, str, np.ndarray | None]:
        """
        Transform result poses to ``target_frame`` before tracking.

        Returns ``(results, frame, camera rotation in target_frame)``; the
        rotation is None without a ``target_frame``. Returns ``(None, camera
        frame, None)`` when ``target_frame`` is set but its TF is missing:
        mixing frames would split every track.
        """
        source = camera_header.frame_id
        target = self._target_frame
        if not target or target == source:
            return list(results), source, None
        tf = self.lookup(
            target, source, camera_header, 'frame not tracked'
        )
        if tf is None:
            return None, source, None
        translation, rotation = tf
        return [
            replace(r, pose=transform_pose(r.pose, translation, rotation))
            if r.pose is not None else r
            for r in results
        ], target, rotation

    def camera_turning(self, rotation: np.ndarray, stamp: float) -> bool:
        """Return True when the camera turns faster than max_camera_rot_deg_s."""
        prev, self._prev_camera = self._prev_camera, (stamp, rotation)
        if self._max_rot_deg_s <= 0.0 or prev is None:
            return False
        speed = angular_speed_deg_s(prev[1], prev[0], rotation, stamp)
        return speed > self._max_rot_deg_s
