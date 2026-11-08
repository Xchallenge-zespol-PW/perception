"""ROS 2 node: RGB-D → per-slot cube colour and pose (thin wrapper)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

# Ultralytics may auto-upgrade numpy to 2.x and break Humble cv_bridge.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
# Small per-cube numpy ops get slower when BLAS spreads them over all cores.
# Not OMP_NUM_THREADS: it would also pin torch (.pt on CPU) to one thread.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import message_filters  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge, CvBridgeError  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.executors import (  # noqa: E402
    ExternalShutdownException,
    MultiThreadedExecutor,
)
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    QoSProfile,
    qos_profile_sensor_data,
    ReliabilityPolicy,
)
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from tf2_ros import (  # noqa: E402
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    StaticTransformBroadcaster,
    TransformBroadcaster,
    TransformListener,
)
from vision_msgs.msg import Detection3DArray  # noqa: E402
from xchallenge_msgs.msg import CubeColors  # noqa: E402

from g1_perception.core.cube_registry import CubeRegistry, NamedCube  # noqa: E402
from g1_perception.core.debug_vis import draw_debug, union_mask  # noqa: E402
from g1_perception.core.frame_utils import (  # noqa: E402
    depth_to_m,
    intrinsics_from_k,
)
from g1_perception.core.geometry import transform_pose  # noqa: E402
from g1_perception.core.slot_assign import AssignmentResult, assign_slots  # noqa: E402
from g1_perception.core.tracker import CubeTracker, TrackedCube  # noqa: E402
from g1_perception.core.types import Frame, SlotResult  # noqa: E402
from g1_perception.core.voting import SlotVoter, VotedSlot  # noqa: E402
from g1_perception.node_config import (  # noqa: E402
    DEFAULT_PARAMS,
    build_pipeline_config,
    registry_kwargs,
    slot_assign_kwargs,
    tracker_kwargs,
    voter_kwargs,
)
from g1_perception.pipeline import ROIS_MODE_NONE, Pipeline  # noqa: E402
from g1_perception.ros_conversions import (  # noqa: E402
    cube_colors_msg,
    pose_to_transform,
    voted_to_detection_array,
)

# The logic node of exercise 1 reads the start pose of cube <name> from the
# static TF frame <name>_start.
START_FRAME_SUFFIX = '_start'


@dataclass(frozen=True)
class _StartLayout:
    """Frozen start layout; the start poses are in ``header.frame_id``."""

    start_id: int
    header: Header
    cubes: list[NamedCube]


@dataclass
class _LatestBundle:
    """One synchronized RGB-D + camera_info message set."""

    color: Image
    depth: Image
    info: CameraInfo


@dataclass
class _PublishBundle:
    """
    Last finished inference result ready for publishers.

    ``seq`` numbers the processed frames, so a bundle published twice by the
    timer sends its TF only once. ``cubes`` are the named cubes of this frame
    in ``frame_id`` (empty before the start layout is frozen), ``start`` the
    frozen start layout.
    """

    camera_header: Header
    results: list[SlotResult]
    voted: list[VotedSlot]
    rgb: np.ndarray
    inference_ms: float
    frame_id: str
    seq: int
    cubes: list[NamedCube]
    start: _StartLayout | None


class PerceptionNode(Node):
    """
    Synchronize color + aligned depth + camera_info and publish cube slots.

    The sync callback only keeps the **newest** message triple (no queue).
    A worker thread always processes that newest set, so older frames are
    dropped instead of building latency (matters most on the CPU fallback).
    With ``rois_mode=none``, detections are tracked in 3D and assigned place
    numbers after TF to ``target_frame``. Results publish at ``publish_rate_hz``.

    For exercise 1 (``rois_mode=none``) the start layout is frozen once by a
    :class:`CubeRegistry` and published once: the static TF
    ``<name>_start`` in ``start_frame`` and ``xchallenge_msgs/CubeColors``
    (transient local). The cubes keep their names while they are pushed, and
    each cube seen in a frame is published as the TF ``<name>`` in
    ``target_frame``. ``~/reset`` (std_srvs/Trigger) forgets the layout
    before the next attempt.
    """

    def __init__(self) -> None:
        super().__init__('perception_node')
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        self._params: dict[str, Any] = {
            name: self.get_parameter(name).value for name in DEFAULT_PARAMS
        }
        p = self._params

        self._pipeline = Pipeline.from_config(build_pipeline_config(p))
        # Load + warm up now: fail early, and keep TensorRT / CUDA init
        # out of the first camera frame.
        t0 = time.perf_counter()
        self._pipeline.load()
        self._backend = self._pipeline.backend_label
        if self._backend is not None:
            load_s = time.perf_counter() - t0
            log = self.get_logger()
            if self._backend.endswith('fallback'):
                log.warning(
                    f'Inference backend {self._backend} (primary '
                    f"{p['model_path']!r} unusable on device "
                    f"{p['device'] or 'auto'!r}); expect slower inference"
                )
            else:
                log.info(f'Inference backend {self._backend}')
            log.info(f'Model loaded and warmed up in {load_s:.1f} s')
        self._rois = self._pipeline.rois
        self._rois_mode = self._pipeline.rois_mode
        self._free = self._rois_mode == ROIS_MODE_NONE or not self._rois
        self._voter = SlotVoter(**voter_kwargs(p))
        self._tracker = CubeTracker(**tracker_kwargs(p)) if self._free else None
        self._assign_kw = slot_assign_kwargs(p)
        self._registry = (
            CubeRegistry(**registry_kwargs(p)) if self._free else None
        )
        self._target_frame = str(p['target_frame']).strip()
        self._start_frame = str(p['start_frame']).strip()
        self._bridge = CvBridge()

        # Tracker, voter, registry and the frozen start: worker and ~/reset.
        self._state_lock = threading.Lock()
        self._start: _StartLayout | None = None
        self._track_ids: dict[str, int | None] = {}
        self._seq = 0

        # Used only by the publish timer.
        self._published_seq = 0
        self._published_start_id = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)

        self._pub_cubes = self.create_publisher(
            Detection3DArray, p['output_topic'], 10
        )
        self._pub_debug = self.create_publisher(
            Image, p['debug_image_topic'], 1
        )
        self._pub_mask = self.create_publisher(Image, p['mask_topic'], 1)
        # Transient local: a logic node started later still gets the colours.
        self._pub_colors = self.create_publisher(
            CubeColors,
            p['cube_colors_topic'],
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._reset_service = self.create_service(
            Trigger, '~/reset', self._on_reset
        )

        self._latest_cv = threading.Condition()
        self._latest: _LatestBundle | None = None

        self._result_lock = threading.Lock()
        self._result: _PublishBundle | None = None

        self._stats_lock = threading.Lock()
        self._infer_ms_sum = 0.0
        self._infer_count = 0
        self._skipped_since_log = 0

        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name='perception_worker', daemon=True
        )

        subs = [
            message_filters.Subscriber(
                self, Image, p['color_topic'],
                qos_profile=qos_profile_sensor_data,
            ),
            message_filters.Subscriber(
                self, Image, p['depth_topic'],
                qos_profile=qos_profile_sensor_data,
            ),
            message_filters.Subscriber(
                self, CameraInfo, p['camera_info_topic'],
                qos_profile=qos_profile_sensor_data,
            ),
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            subs,
            queue_size=int(p['sync_queue_size']),
            slop=float(p['sync_slop_s']),
        )
        self._sync.registerCallback(self._on_synced)

        rate = float(p['publish_rate_hz'])
        if rate <= 0.0:
            raise ValueError(f'publish_rate_hz must be > 0, got {rate}')
        self._publish_timer = self.create_timer(
            1.0 / rate,
            self._on_publish_timer,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._stats_timer = self.create_timer(
            10.0,
            self._on_stats_timer,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self._worker.start()
        self.get_logger().info(
            f"segmenter={p['segmenter']} pose={p['pose_estimator']} "
            f'rois_mode={self._rois_mode} '
            f'slots={[r.slot for r in self._rois] or "(free)"} '
            f"target_frame={p['target_frame'] or '(camera)'} "
            f"publish_rate_hz={rate} -> {p['output_topic']}"
        )
        if self._registry is not None:
            self.get_logger().info(
                f"exercise 1: cubes {p['cube_name_prefix']}<place> along "
                f"{p['slot_axis']}, start frames in "
                f"{self._start_frame or '(target_frame)'}, colours -> "
                f"{p['cube_colors_topic']}"
            )
        else:
            self.get_logger().warning(
                'rois_mode is not none: no cube colours or cube TF for '
                'exercise 1 (they need the tracker)'
            )

    def destroy_node(self) -> bool:
        """Stop the worker, then destroy ROS entities."""
        self._stop.set()
        with self._latest_cv:
            self._latest_cv.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        return super().destroy_node()

    def _on_synced(self, color: Image, depth: Image, info: CameraInfo) -> None:
        """Keep only the newest synchronized triple (drop backlog)."""
        with self._latest_cv:
            if self._latest is not None:
                self._skipped_since_log += 1
            self._latest = _LatestBundle(color=color, depth=depth, info=info)
            self._latest_cv.notify()

    def _worker_loop(self) -> None:
        """Process the newest frame as soon as the previous one finishes."""
        while not self._stop.is_set() and self.context.ok():
            with self._latest_cv:
                while self._latest is None and not self._stop.is_set():
                    self._latest_cv.wait(timeout=0.2)
                if self._stop.is_set():
                    return
                bundle = self._latest
                self._latest = None

            assert bundle is not None
            self._process_bundle(bundle)

    def _process_bundle(self, bundle: _LatestBundle) -> None:
        """Convert, run pipeline + track/vote, store result for publish."""
        try:
            frame = self._to_frame(bundle.color, bundle.depth, bundle.info)
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warning(
                f'Bad input frame: {exc}', throttle_duration_sec=5.0
            )
            return

        t0 = time.perf_counter()
        try:
            results = self._pipeline.process(frame)
        except Exception as exc:  # keep the node alive on per-frame failures
            self.get_logger().error(
                f'Pipeline failed: {exc!r}', throttle_duration_sec=5.0
            )
            return
        inference_ms = (time.perf_counter() - t0) * 1000.0

        if self._stop.is_set() or not self.context.ok():
            return

        camera_header = Header(
            stamp=bundle.color.header.stamp,
            frame_id=bundle.color.header.frame_id,
        )

        cubes: list[NamedCube] = []
        start: _StartLayout | None = None
        if self._free:
            assert self._tracker is not None and self._registry is not None
            n_det = len(results)
            n_pose = sum(1 for r in results if r.pose is not None)
            with self._state_lock:
                tracked = self._tracker.update(results, frame.stamp)
                tracked_tf, frame_id = self._tracks_to_target(
                    tracked, camera_header
                )
                assignment = assign_slots(tracked_tf, **self._assign_kw)
                # Camera-frame poses (no TF) must not name or move the cubes.
                if not self._target_frame or frame_id == self._target_frame:
                    cubes = self._update_registry(
                        assignment, tracked_tf, camera_header, frame_id,
                        frame.stamp,
                    )
                latched = self._registry.latched
                start = self._start
            voted = assignment.slots
            debug_results = _debug_results_from_tracks(
                tracked_tf, voted
            )
            # Once the layout is frozen, pushed cubes leave the line on purpose.
            if not assignment.valid and not latched:
                self.get_logger().warning(
                    'slot assignment invalid '
                    f'(det={n_det} with_pose={n_pose} '
                    f'tracks={len(tracked_tf)}, expected='
                    f'{self._assign_kw["expected_count"]}); '
                    'publishing voted colours with track ids (t<N>)',
                    throttle_duration_sec=2.0,
                )
        else:
            with self._state_lock:
                voted_cam = self._voter.update(results, frame.stamp)
            voted, frame_id = self._voted_to_target(voted_cam, camera_header)
            debug_results = results

        self._seq += 1
        with self._result_lock:
            self._result = _PublishBundle(
                camera_header=camera_header,
                results=debug_results,
                voted=voted,
                rgb=frame.rgb,
                inference_ms=inference_ms,
                frame_id=frame_id,
                seq=self._seq,
                cubes=cubes,
                start=start,
            )
        with self._stats_lock:
            self._infer_ms_sum += inference_ms
            self._infer_count += 1

    def _on_publish_timer(self) -> None:
        """Publish the latest finished result at ``publish_rate_hz``."""
        with self._result_lock:
            bundle = self._result
        if bundle is None or not self.context.ok():
            return

        out_header = Header(
            stamp=bundle.camera_header.stamp, frame_id=bundle.frame_id
        )
        self._pub_cubes.publish(
            voted_to_detection_array(
                bundle.voted, out_header, float(self._params['cube_size_m'])
            )
        )
        self._publish_task_outputs(bundle)

        mask = union_mask(bundle.results, bundle.rgb.shape[:2])
        mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = bundle.camera_header
        self._pub_mask.publish(mask_msg)

        if self._pub_debug.get_subscription_count() > 0:
            debug = draw_debug(
                bundle.rgb,
                self._rois,
                bundle.results,
                bundle.voted,
                inference_ms=bundle.inference_ms,
                backend=self._backend,
            )
            debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding='rgb8')
            debug_msg.header = bundle.camera_header
            self._pub_debug.publish(debug_msg)

    def _publish_task_outputs(self, bundle: _PublishBundle) -> None:
        """Publish a new start layout once, and the cubes seen in a new frame."""
        start = bundle.start
        if start is not None and start.start_id != self._published_start_id:
            # Start frames before the colours: the logic node starts the task
            # with whatever <name>_start frames TF has when its colours came.
            # One message: StaticTransformBroadcaster keeps only the last one.
            self._static_tf_broadcaster.sendTransform([
                pose_to_transform(
                    cube.start_pose, start.header, cube.name + START_FRAME_SUFFIX
                )
                for cube in start.cubes
            ])
            self._pub_colors.publish(cube_colors_msg(start.cubes))
            self._published_start_id = start.start_id

        if bundle.seq == self._published_seq:
            return
        self._published_seq = bundle.seq
        header = Header(
            stamp=bundle.camera_header.stamp, frame_id=bundle.frame_id
        )
        transforms = [
            pose_to_transform(cube.pose, header, cube.name)
            for cube in bundle.cubes
            if cube.pose is not None
        ]
        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

    def _on_reset(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Forget the start layout and the tracks before the next attempt."""
        with self._state_lock:
            if self._tracker is not None:
                self._tracker.reset()
            if self._registry is not None:
                self._registry.reset()
            self._voter.reset()
            self._start = None
            self._track_ids = {}
        with self._result_lock:
            self._result = None

        response.success = True
        response.message = (
            'Waiting for a new start layout'
            if self._registry is not None else 'Votes cleared'
        )
        self.get_logger().info(f'reset: {response.message}')
        return response

    def _update_registry(
        self,
        assignment: AssignmentResult,
        tracked: list[TrackedCube],
        camera_header: Header,
        frame_id: str,
        stamp: float,
    ) -> list[NamedCube]:
        """Name the cubes of a frame and anchor a newly frozen start layout."""
        assert self._registry is not None
        cubes = self._registry.update(assignment, tracked, stamp)
        if not cubes:
            return cubes

        if self._start is None or self._start.start_id != self._registry.start_id:
            self._start = self._anchor_start(cubes, camera_header, frame_id)
            if self._start is None:
                # Without the start frame TF the logic would wait forever.
                self._registry.reset()
                return []
            self._track_ids = {}
            layout = ', '.join(f'{c.name}={c.label}' for c in cubes)
            self.get_logger().info(
                f'start layout frozen: {layout} '
                f'(start frames in {self._start.header.frame_id})'
            )

        for cube in cubes:
            previous = self._track_ids.get(cube.name)
            if previous is not None and cube.track_id != previous:
                self.get_logger().info(
                    f'{cube.name} found again as track {cube.track_id}'
                )
            self._track_ids[cube.name] = cube.track_id
        return cubes

    def _anchor_start(
        self,
        cubes: list[NamedCube],
        camera_header: Header,
        frame_id: str,
    ) -> _StartLayout | None:
        """Express the start poses in start_frame; None without its TF."""
        assert self._registry is not None
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
        return _StartLayout(
            start_id=self._registry.start_id,
            header=Header(stamp=camera_header.stamp, frame_id=parent),
            cubes=cubes,
        )

    def _on_stats_timer(self) -> None:
        """Log mean inference time and dropped frames every 10 s."""
        with self._stats_lock:
            count = self._infer_count
            total_ms = self._infer_ms_sum
            self._infer_count = 0
            self._infer_ms_sum = 0.0
        with self._latest_cv:
            skipped = self._skipped_since_log
            self._skipped_since_log = 0

        if count == 0:
            self.get_logger().info(
                f'perception stats (10 s): no inferences, '
                f'skipped_frames={skipped}'
            )
            return
        self.get_logger().info(
            f'perception stats (10 s): '
            f'infer_ms_avg={total_ms / count:.1f} '
            f'n={count} skipped_frames={skipped} '
            f'backend={self._backend or "-"}'
        )

    def _to_frame(self, color: Image, depth: Image, info: CameraInfo) -> Frame:
        """Convert ROS messages to a :class:`Frame` (RGB, depth in metres)."""
        rgb = self._bridge.imgmsg_to_cv2(color, desired_encoding='rgb8')
        depth_raw = self._bridge.imgmsg_to_cv2(
            depth, desired_encoding='passthrough'
        )
        depth_m = depth_to_m(depth_raw, depth.encoding)
        if depth_m.shape != rgb.shape[:2]:
            raise ValueError(
                f'depth {depth_m.shape} does not match color {rgb.shape[:2]}; '
                'use aligned_depth_to_color'
            )
        stamp = color.header.stamp.sec + color.header.stamp.nanosec * 1e-9
        return Frame(
            rgb=np.ascontiguousarray(rgb),
            depth_m=depth_m,
            K=intrinsics_from_k(info.k),
            stamp=float(stamp),
        )

    def _lookup_tf(
        self,
        camera_header: Header,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Return ``(translation, rotation_xyzw, target)`` or None."""
        target = self._target_frame
        source = camera_header.frame_id
        if not target or target == source:
            return None
        tf = self._lookup(
            target, source, camera_header, f'publishing in {source}'
        )
        if tf is None:
            return None
        return tf[0], tf[1], target

    def _lookup(
        self,
        target: str,
        source: str,
        header: Header,
        consequence: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(translation, rotation_xyzw)`` of ``target <- source``."""
        try:
            tf = self._tf_buffer.lookup_transform(
                target,
                source,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=float(self._params['tf_timeout_s'])),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as exc:
            self.get_logger().warning(
                f'No TF {source} -> {target} ({exc}); {consequence}',
                throttle_duration_sec=5.0,
            )
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])

    def _voted_to_target(
        self,
        voted: list[VotedSlot],
        camera_header: Header,
    ) -> tuple[list[VotedSlot], str]:
        """Transform voted poses to ``target_frame``; fall back to camera."""
        tf = self._lookup_tf(camera_header)
        if tf is None:
            return voted, camera_header.frame_id
        translation, rotation, target = tf
        return [
            replace(v, pose=transform_pose(v.pose, translation, rotation))
            if v.pose is not None else v
            for v in voted
        ], target

    def _tracks_to_target(
        self,
        tracked: list[TrackedCube],
        camera_header: Header,
    ) -> tuple[list[TrackedCube], str]:
        """Transform track poses to ``target_frame`` before slot assignment."""
        tf = self._lookup_tf(camera_header)
        if tf is None:
            return tracked, camera_header.frame_id
        translation, rotation, target = tf
        out: list[TrackedCube] = []
        for t in tracked:
            if t.pose is None:
                out.append(t)
            else:
                out.append(
                    replace(
                        t,
                        pose=transform_pose(t.pose, translation, rotation),
                    )
                )
        return out, target


def _debug_results_from_tracks(
    tracked: list[TrackedCube],
    voted: list[VotedSlot],
) -> list[SlotResult]:
    """Build SlotResults (with masks) keyed by published place / track id."""
    by_track = {t.track_id: t for t in tracked}
    out: list[SlotResult] = []
    for v in voted:
        tid = v.track_id if v.track_id is not None else v.slot
        tr = by_track.get(tid)
        out.append(
            SlotResult(
                slot=v.slot,
                label=v.label,
                score=v.score,
                detection=tr.detection if tr is not None else None,
                pose=v.pose,
            )
        )
    return out


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run g1_perception perception_node``."""
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    rclpy.init(args=args)
    node = PerceptionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
