"""Exercise-1 start layout, cube TF publication, and registry updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from g1_perception.core.cube_registry import CubeRegistry, NamedCube
from g1_perception.core.geometry import transform_pose
from g1_perception.core.slot_assign import AssignmentResult
from g1_perception.core.tracker import TrackedCube
from g1_perception.ros.conversions import cube_colors_msg, pose_to_transform

# The logic node of exercise 1 reads the start pose of cube <name> from the
# static TF frame <name>_start.
START_FRAME_SUFFIX = '_start'
SEEN_FRAME_SUFFIX = '_seen'

LookupFn = Callable[
    [str, str, Header, str],
    tuple[Any, Any] | None,
]


@dataclass(frozen=True)
class StartLayout:
    """Frozen start layout; the start poses are in ``header.frame_id``."""

    start_id: int
    header: Header
    cubes: list[NamedCube]


@dataclass
class NamedCubeFrame:
    """Cubes seen in one processed frame (for ``<cube>_seen`` TF)."""

    frame_id: str
    stamp: Any  # builtin_interfaces/Time
    cubes: list[NamedCube]


class Task1State:
    """Registry lifecycle: freeze start layout, track re-ids, reset."""

    def __init__(
        self,
        registry: CubeRegistry,
        start_frame: str,
        lookup: LookupFn,
        logger: Any,
    ) -> None:
        self.registry = registry
        self._start_frame = start_frame
        self._lookup = lookup
        self._logger = logger
        self.start: StartLayout | None = None
        self.track_ids: dict[str, int | None] = {}
        self.cube_frame: str | None = None

    def reset(self) -> None:
        self.registry.reset()
        self.start = None
        self.track_ids = {}
        self.cube_frame = None

    def update(
        self,
        assignment: AssignmentResult,
        tracked: list[TrackedCube],
        camera_header: Header,
        frame_id: str,
        stamp: float,
    ) -> list[NamedCube]:
        """Name the cubes of a frame and anchor a newly frozen start layout."""
        cubes = self.registry.update(assignment, tracked, stamp)
        if not cubes:
            return cubes
        self.cube_frame = frame_id

        if self.start is None or self.start.start_id != self.registry.start_id:
            self.start = self._anchor_start(cubes, camera_header, frame_id)
            if self.start is None:
                # Without the start frame TF the logic would wait forever.
                self.registry.reset()
                return []
            self.track_ids = {}
            layout = ', '.join(f'{c.name}={c.label}' for c in cubes)
            self._logger.info(
                f'start layout frozen: {layout} '
                f'(start frames in {self.start.header.frame_id})'
            )

        for cube in cubes:
            previous = self.track_ids.get(cube.name)
            if previous is not None and cube.track_id != previous:
                self._logger.info(
                    f'{cube.name} found again as track {cube.track_id}'
                )
            self.track_ids[cube.name] = cube.track_id
        return cubes

    def _anchor_start(
        self,
        cubes: list[NamedCube],
        camera_header: Header,
        frame_id: str,
    ) -> StartLayout | None:
        """Express the start poses in start_frame; None without its TF."""
        parent = self._start_frame or frame_id
        if parent != frame_id:
            tf = self._lookup(
                parent, frame_id, camera_header,
                'start layout not frozen',
            )
            if tf is None:
                return None
            translation, rotation = tf
            cubes = [
                replace(
                    cube,
                    start_pose=transform_pose(
                        cube.start_pose, translation, rotation
                    ),
                )
                for cube in cubes
            ]
        return StartLayout(
            start_id=self.registry.start_id,
            header=Header(stamp=camera_header.stamp, frame_id=parent),
            cubes=cubes,
        )


class Task1Publisher:
    """Publish start layout once, live ``<cube>`` TF, and ``<cube>_seen``."""

    def __init__(
        self,
        static_tf: StaticTransformBroadcaster,
        tf: TransformBroadcaster,
        pub_colors: Any,
    ) -> None:
        self._static_tf = static_tf
        self._tf = tf
        self._pub_colors = pub_colors
        self._published_start_id = 0
        self._last_cube_tf_ns = -1
        self._last_seen_tf_ns = -1

    def publish_outputs(
        self,
        start: StartLayout | None,
        seen: NamedCubeFrame | None,
        known: list[NamedCube],
        cube_frame: str | None,
        now: Any,
    ) -> None:
        """Publish a new start layout once, then known + seen cube TFs."""
        if start is not None and start.start_id != self._published_start_id:
            # Start frames before the colours: the logic node starts the task
            # with whatever <name>_start frames TF has when its colours came.
            # One message: StaticTransformBroadcaster keeps only the last one.
            self._static_tf.sendTransform([
                pose_to_transform(
                    cube.start_pose, start.header, cube.name + START_FRAME_SUFFIX
                )
                for cube in start.cubes
            ])
            self._pub_colors.publish(cube_colors_msg(start.cubes))
            self._published_start_id = start.start_id

        self._publish_known(known, cube_frame, now)
        if seen is not None:
            self._publish_seen(seen)

    def _publish_seen(self, seen: NamedCubeFrame) -> None:
        stamp = Time.from_msg(seen.stamp).nanoseconds
        cubes = [c for c in seen.cubes if c.pose is not None]
        if not cubes or not seen.frame_id or stamp <= self._last_seen_tf_ns:
            return
        self._last_seen_tf_ns = stamp
        header = Header(stamp=seen.stamp, frame_id=seen.frame_id)
        self._tf.sendTransform([
            pose_to_transform(c.pose, header, c.name + SEEN_FRAME_SUFFIX)
            for c in cubes
        ])

    def _publish_known(
        self,
        cubes: list[NamedCube],
        frame_id: str | None,
        now: Any,
    ) -> None:
        if not cubes or not frame_id:
            return
        # A paused simulation clock would repeat the stamp (TF_REPEATED_DATA).
        if now.nanoseconds <= self._last_cube_tf_ns:
            return
        self._last_cube_tf_ns = now.nanoseconds
        header = Header(stamp=now.to_msg(), frame_id=frame_id)
        self._tf.sendTransform([
            pose_to_transform(cube.last_pose, header, cube.name)
            for cube in cubes
            if cube.last_pose is not None
        ])
