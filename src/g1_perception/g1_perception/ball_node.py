"""ROS 2 node: RGB-D → red balls (exercise 2) as TF in the world frame."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

# Ultralytics may auto-upgrade numpy to 2.x and break Humble cv_bridge.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import cv2  # noqa: E402
import message_filters  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge, CvBridgeError  # noqa: E402
from geometry_msgs.msg import Pose as PoseMsg  # noqa: E402
from geometry_msgs.msg import PoseArray, TransformStamped  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.executors import (  # noqa: E402
    ExternalShutdownException,
    MultiThreadedExecutor,
)
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Header, String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from tf2_ros import (  # noqa: E402
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformBroadcaster,
    TransformListener,
)
from vision_msgs.msg import (  # noqa: E402
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from g1_perception.core.ball import (  # noqa: E402
    BallObservation,
    draw_balls,
    find_balls,
    format_balls,
)
from g1_perception.core.ball_map import (  # noqa: E402
    GATE_RADII,
    VIEW_OUT,
    VIEW_SEEN,
    BallMap,
    MappedBall,
    map_summary,
    project,
    view_state,
)
from g1_perception.core.frame_utils import (  # noqa: E402
    depth_to_m,
    intrinsics_from_k,
)
from g1_perception.core.geometry import (  # noqa: E402
    quat_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
)
from g1_perception.core.types import Frame  # noqa: E402
from g1_perception.core.visibility import angular_speed_deg_s  # noqa: E402
from g1_perception.segmenters.yolo_seg import YoloSegSegmenter  # noqa: E402

SEEN_FRAME_PREFIX = 'ball_seen_'
# A new ball this close (in its radii) to a known one may be the same ball
# whose position jumped (odom drift, depth bias): logged as a warning.
SPLIT_WARN_RADII = 4.0

DEFAULT_PARAMS: dict[str, Any] = {
    'color_topic': '/camera/d435/color/image_raw',
    'depth_topic': '/camera/d435/aligned_depth_to_color/image_raw',
    'camera_info_topic': '/camera/d435/color/camera_info',
    'balls_topic': '/perception/balls',
    'seen_topic': '/perception/balls/seen',
    'summary_topic': '/perception/balls/summary',
    'debug_image_topic': '/perception/balls/debug_image',
    'world_frame': 'odom',
    'ball_name_prefix': 'ball_',
    'publish_rate_hz': 5.0,
    'tf_timeout_s': 0.1,
    'model_path': '/ws/models/cubes_seg_v4_aug.engine',
    'device': '0',
    'fallback_model_path': '/ws/models/cubes_seg_v4_aug.onnx',
    'fallback_device': 'cpu',
    'warmup_runs': 2,
    'conf': 0.35,
    'ball_labels': ['red ball'],
    'border_px': 3,
    'max_camera_rot_deg_s': 30.0,
    'log_period_s': 1.0,
    'sync_queue_size': 10,
    'sync_slop_s': 0.05,
}


@dataclass
class _LatestBundle:
    """One synchronized RGB-D + camera_info message set."""

    color: Image
    depth: Image
    info: CameraInfo


class BallNode(Node):
    """
    Detect red balls and keep them in a map in ``world_frame`` (``odom``).

    Every processed frame takes the balls it sees whose outline was fitted
    and whose depth profile is a ball (:attr:`BallObservation.trusted`; a
    ball hidden too much, a flat red blob or a second mask on one ball is
    only logged and drawn) into a :class:`BallMap`. A ball seen in
    ``CONFIRM_FRAMES`` frames gets a stable name ``ball_<N>`` and stays in
    the map while out of view or hidden. A ball whose place is in view but
    seen empty (depth behind it, no detection) for ``EMPTY_TIMEOUT_S``
    leaves the map (taken); its name is not reused. While the camera turns
    faster than ``max_camera_rot_deg_s`` in ``world_frame`` (motion blur,
    image and TF out of step) a frame neither adds to the map nor removes.

    At ``publish_rate_hz`` every ball of the map is published as the TF
    ``ball_<N>`` in ``world_frame`` (stamped now, at its median position)
    and on ``balls_topic`` (vision_msgs/Detection3DArray, ``id`` = name),
    and as JSON on ``summary_topic`` (std_msgs/String).
    Each frame's balls are also published as TF ``ball_seen_<K>`` (``K`` =
    rank by distance, stamped with the frame) and a PoseArray on
    ``seen_topic``. Without the TF camera -> ``world_frame`` a frame is
    published as seen in the camera frame but not mapped; with
    ``world_frame: ""`` the map is in the camera frame (camera standing
    still). ``~/reset`` (std_srvs/Trigger) empties the map.

    Like perception_node, only the newest synchronized frame is processed.
    """

    def __init__(self) -> None:
        super().__init__('ball_node')
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        p: dict[str, Any] = {
            name: self.get_parameter(name).value for name in DEFAULT_PARAMS
        }
        self._params = p
        self._world_frame = str(p['world_frame']).strip()
        self._labels = [str(x) for x in p['ball_labels']]
        self._border_px = int(p['border_px'])
        self._prefix = str(p['ball_name_prefix'])
        self._map = BallMap()
        self._map_lock = threading.Lock()
        self._map_frame: str | None = None
        self._last_map_tf_ns = -1
        self._max_rot_deg_s = float(p['max_camera_rot_deg_s'])
        # (stamp, rotation_xyzw) of the camera in world_frame, last frame.
        self._prev_camera: tuple[float, np.ndarray] | None = None

        self._segmenter = YoloSegSegmenter(
            model_path=p['model_path'],
            device=str(p['device']) or None,
            fallback_model_path=str(p['fallback_model_path']) or None,
            fallback_device=str(p['fallback_device']) or None,
            warmup_runs=int(p['warmup_runs']),
            conf=float(p['conf']),
        )
        t0 = time.perf_counter()
        backend = self._segmenter.load()
        self._backend = backend.describe()
        log = self.get_logger()
        if self._backend.endswith('fallback'):
            log.warning(
                f'Inference backend {self._backend} (primary '
                f"{p['model_path']!r} unusable); expect slower inference"
            )
        else:
            log.info(f'Inference backend {self._backend}')
        log.info(f'Model loaded and warmed up in {time.perf_counter() - t0:.1f} s')

        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._pub_balls = self.create_publisher(
            Detection3DArray, p['balls_topic'], 10
        )
        self._pub_seen = self.create_publisher(PoseArray, p['seen_topic'], 10)
        self._pub_summary = self.create_publisher(String, p['summary_topic'], 10)
        self._reset_service = self.create_service(
            Trigger, '~/reset', self._on_reset
        )
        self._pub_debug = self.create_publisher(Image, p['debug_image_topic'], 1)

        self._latest_cv = threading.Condition()
        self._latest: _LatestBundle | None = None
        self._last_stamp_ns = -1
        self._last_log = 0.0

        self._stats_lock = threading.Lock()
        self._infer_ms_sum = 0.0
        self._infer_count = 0
        self._skipped = 0
        self._n_balls = 0
        self._n_rejected: dict[str, int] = {}
        self._fit_ms_sum = 0.0
        self._n_turning = 0
        self._rot_speeds: list[float] = []
        self._residuals: list[float] = []

        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop, name='ball_worker', daemon=True
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
        self._stats_timer = self.create_timer(
            10.0,
            self._on_stats_timer,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        rate = float(p['publish_rate_hz'])
        if rate <= 0.0:
            raise ValueError(f'publish_rate_hz must be > 0, got {rate}')
        self._publish_timer = self.create_timer(
            1.0 / rate,
            self._on_publish_timer,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._worker.start()
        log.info(
            f"balls {self._labels} from {p['color_topic']} -> TF "
            f"{self._prefix}<N> in {self._world_frame or '(camera)'} at "
            f"{rate} Hz, {p['balls_topic']}; per frame {SEEN_FRAME_PREFIX}<K>, "
            f"{p['seen_topic']}"
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
                self._skipped += 1
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
            try:
                self._process(bundle)
            except Exception as exc:  # keep the node alive on per-frame failures
                self.get_logger().error(
                    f'Ball frame failed: {exc!r}', throttle_duration_sec=5.0
                )

    def _process(self, bundle: _LatestBundle) -> None:
        """Detect balls on one frame and publish them."""
        try:
            frame = self._to_frame(bundle.color, bundle.depth, bundle.info)
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warning(
                f'Bad input frame: {exc}', throttle_duration_sec=5.0
            )
            return
        t0 = time.perf_counter()
        detections = self._segmenter.segment(frame)
        t1 = time.perf_counter()
        balls = find_balls(
            detections, frame.depth_m, frame.K,
            labels=self._labels, border_px=self._border_px,
        )
        if self._stop.is_set() or not self.context.ok():
            return

        header = Header(
            stamp=bundle.color.header.stamp,
            frame_id=bundle.color.header.frame_id,
        )
        trusted = [b for b in balls if b.trusted]
        centres, frame_id, mapped, world_tf = self._to_world(trusted, header)
        speed = self._camera_rot_speed(world_tf, frame.stamp)
        turning = 0.0 < self._max_rot_deg_s < (speed or 0.0)
        ids: list[int | None] = [None] * len(trusted)
        marks: list[tuple[tuple[float, float], float, str]] = []
        residuals: list[float] = []
        if mapped and not turning:
            ids = self._update_map(trusted, centres, frame_id, frame.stamp)
            with self._map_lock:
                residuals = list(self._map.last_residuals)
            marks = self._check_views(ids, world_tf, frame)
        t2 = time.perf_counter()
        inference_ms = (t2 - t0) * 1000.0
        self._publish_seen(centres, Header(stamp=header.stamp, frame_id=frame_id))

        names = self._names(balls, trusted, ids, 'turn' if turning else 'new')
        self._log_balls(balls, names)
        if self._pub_debug.get_subscription_count() > 0:
            debug = draw_balls(
                frame.rgb, balls, names=names, marks=marks,
                inference_ms=inference_ms, backend=self._backend,
            )
            msg = self._bridge.cv2_to_imgmsg(debug, encoding='rgb8')
            msg.header = header
            self._pub_debug.publish(msg)

        with self._stats_lock:
            self._infer_ms_sum += (t1 - t0) * 1000.0
            self._fit_ms_sum += (t2 - t1) * 1000.0
            self._infer_count += 1
            self._n_turning += int(turning)
            if speed is not None:
                self._rot_speeds.append(speed)
            self._residuals.extend(residuals)
            for b in balls:
                if b.trusted:
                    self._n_balls += 1
                else:
                    reason = b.reject_reason
                    self._n_rejected[reason] = self._n_rejected.get(reason, 0) + 1

    def _to_world(
        self,
        balls: list[BallObservation],
        camera: Header,
    ) -> tuple[
        list[np.ndarray], str, bool, tuple[np.ndarray, np.ndarray] | None
    ]:
        """
        Return ``(centres, frame, mapped, (R, t))`` of ``balls`` in world_frame.

        ``(R, t)`` maps camera to world (None = the map is in the camera
        frame). Without its TF the centres stay in the camera frame and
        ``mapped`` is False: the camera moves, so they must not enter the map.
        """
        centres = [b.position for b in balls]
        if not self._world_frame or self._world_frame == camera.frame_id:
            return centres, camera.frame_id, True, None
        tf = self._lookup_world(camera)
        if tf is None:
            return centres, camera.frame_id, False, None
        translation, rotation = tf
        R = quat_to_rotation_matrix(rotation)
        return (
            [R @ c + translation for c in centres], self._world_frame, True,
            (R, translation),
        )

    def _camera_rot_speed(
        self,
        world_tf: tuple[np.ndarray, np.ndarray] | None,
        stamp: float,
    ) -> float | None:
        """Camera rotation speed (deg/s) in world_frame since the last frame."""
        if world_tf is None:
            self._prev_camera = None
            return None
        quat = rotation_matrix_to_quat_xyzw(world_tf[0])
        prev, self._prev_camera = self._prev_camera, (stamp, quat)
        if prev is None or stamp <= prev[0]:
            return None
        return angular_speed_deg_s(prev[1], prev[0], quat, stamp)

    def _check_views(
        self,
        ids: list[int | None],
        world_tf: tuple[np.ndarray, np.ndarray] | None,
        frame: Frame,
    ) -> list[tuple[tuple[float, float], float, str]]:
        """
        Look where each known ball not detected on ``frame`` should be.

        Removes the balls whose place stays empty (taken) and returns debug
        marks ``((u, v), r_px, text)`` of those in view.
        """
        seen = {i for i in ids if i is not None}
        with self._map_lock:
            known = self._map.known()
        views: dict[int, str] = {}
        marks: list[tuple[tuple[float, float], float, str]] = []
        for ball in known:
            if ball.ball_id in seen:
                views[ball.ball_id] = VIEW_SEEN
                continue
            center = ball.position
            if world_tf is not None:
                R, t = world_tf
                center = R.T @ (center - t)
            view = view_state(center, ball.radius_m, frame.depth_m, frame.K)
            views[ball.ball_id] = view
            proj = project(center, ball.radius_m, frame.K)
            if view != VIEW_OUT and proj is not None:
                marks.append((proj[0], proj[1], f'{self._prefix}{ball.ball_id} {view}'))
        with self._map_lock:
            removed = self._map.check_views(views, frame.stamp)
        for ball in removed:
            x, y, z = ball.position
            self.get_logger().info(
                f'{self._prefix}{ball.ball_id} removed: its place '
                f'({x:.2f}, {y:.2f}, {z:.2f}) seen empty (ball taken)'
            )
        return marks

    def _update_map(
        self,
        balls: list[BallObservation],
        centres: list[np.ndarray],
        frame_id: str,
        stamp: float,
    ) -> list[int | None]:
        """Add one frame's balls to the map; log balls that got a name."""
        with self._map_lock:
            ids, confirmed = self._map.update(
                centres, [b.radius_m for b in balls], stamp
            )
            self._map_frame = frame_id
            near = [
                (ball, self._map.nearest_known(ball.position, exclude=ball.ball_id))
                for ball in confirmed
            ]
        log = self.get_logger()
        for ball, other in near:
            x, y, z = ball.position
            log.info(
                f'{self._prefix}{ball.ball_id} new at ({x:.2f}, {y:.2f}, {z:.2f}) '
                f'in {frame_id}, r={ball.radius_m * 100:.1f} cm'
            )
            if other is not None and other[1] < SPLIT_WARN_RADII * ball.radius_m:
                log.warning(
                    f'{self._prefix}{ball.ball_id} is {other[1] * 100:.1f} cm from '
                    f'{self._prefix}{other[0].ball_id} (beyond the '
                    f'{GATE_RADII:.0f}-radius gate): two balls, or one that '
                    'moved in the map (odom drift)?'
                )
        return ids

    def _names(
        self,
        balls: list[BallObservation],
        trusted: list[BallObservation],
        ids: list[int | None],
        unnamed: str = 'new',
    ) -> list[str]:
        """Label of each ball: map name, ``unnamed`` (no name yet) or ``-``."""
        by_obs = {
            id(b): (f'{self._prefix}{i}' if i is not None else unnamed)
            for b, i in zip(trusted, ids)
        }
        return [by_obs.get(id(b), '-') for b in balls]

    def _publish_seen(self, centres: list[np.ndarray], header: Header) -> None:
        """Publish TF ``ball_seen_<K>`` and the PoseArray of one frame."""
        stamp_ns = Time.from_msg(header.stamp).nanoseconds
        if stamp_ns <= self._last_stamp_ns:  # TF_REPEATED_DATA on bag --loop
            return
        self._last_stamp_ns = stamp_ns

        array = PoseArray(header=header)
        transforms: list[TransformStamped] = []
        for k, c in enumerate(centres, start=1):
            pose = PoseMsg()
            pose.position.x, pose.position.y, pose.position.z = (
                float(v) for v in c
            )
            pose.orientation.w = 1.0
            array.poses.append(pose)
            t = TransformStamped(header=header)
            t.child_frame_id = f'{SEEN_FRAME_PREFIX}{k}'
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z
            t.transform.rotation.w = 1.0
            transforms.append(t)
        self._pub_seen.publish(array)
        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

    def _on_publish_timer(self) -> None:
        """Publish every ball of the map as TF ``ball_<N>`` and Detection3D."""
        with self._map_lock:
            balls = self._map.known()
            frame_id = self._map_frame
        if not frame_id or not self.context.ok():
            return
        now = self.get_clock().now()
        # A paused simulation clock would repeat the stamp (TF_REPEATED_DATA).
        if now.nanoseconds <= self._last_map_tf_ns:
            return
        self._last_map_tf_ns = now.nanoseconds
        header = Header(stamp=now.to_msg(), frame_id=frame_id)
        self._pub_balls.publish(self._detection_array(balls, header))
        self._pub_summary.publish(
            String(data=json.dumps(map_summary(balls, self._prefix)))
        )
        if not balls:
            return
        transforms = []
        for ball in balls:
            t = TransformStamped(header=header)
            t.child_frame_id = f'{self._prefix}{ball.ball_id}'
            (
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z,
            ) = (float(v) for v in ball.position)
            t.transform.rotation.w = 1.0
            transforms.append(t)
        self._tf_broadcaster.sendTransform(transforms)

    def _detection_array(
        self,
        balls: list[MappedBall],
        header: Header,
    ) -> Detection3DArray:
        """
        Build one Detection3D per map ball.

        ``id`` is the ball name, ``results[0].hypothesis.class_id`` the first
        ball label, ``bbox`` the ball centre and diameter.
        """
        array = Detection3DArray(header=header)
        for ball in balls:
            pose = PoseMsg()
            pose.position.x, pose.position.y, pose.position.z = (
                float(v) for v in ball.position
            )
            pose.orientation.w = 1.0
            det = Detection3D(header=header)
            det.id = f'{self._prefix}{ball.ball_id}'
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self._labels[0] if self._labels else 'ball'
            hypothesis.hypothesis.score = 1.0
            hypothesis.pose.pose = pose
            det.results.append(hypothesis)
            det.bbox.center = pose
            det.bbox.size.x = det.bbox.size.y = det.bbox.size.z = 2.0 * ball.radius_m
            array.detections.append(det)
        return array

    def _on_reset(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Empty the ball map before the next attempt."""
        with self._map_lock:
            n = len(self._map.known())
            self._map.reset()
            self._map_frame = None
        response.success = True
        response.message = f'Ball map cleared ({n} balls)'
        self.get_logger().info(f'reset: {response.message}')
        return response

    def _lookup_world(
        self,
        camera: Header,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(translation, rotation_xyzw)`` of world <- camera, or None."""
        target = self._world_frame
        if not target or target == camera.frame_id:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                target,
                camera.frame_id,
                Time.from_msg(camera.stamp),
                timeout=Duration(seconds=float(self._params['tf_timeout_s'])),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as exc:
            self.get_logger().warning(
                f'No TF {camera.frame_id} -> {target} ({exc}); '
                f'balls of this frame not mapped (seen in {camera.frame_id})',
                throttle_duration_sec=5.0,
            )
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])

    def _log_balls(self, balls: list[BallObservation], names: list[str]) -> None:
        """Log the balls in camera distance, at most every ``log_period_s``."""
        period = float(self._params['log_period_s'])
        now = time.monotonic()
        if period <= 0.0 or now - self._last_log < period:
            return
        self._last_log = now
        with self._map_lock:
            n_map = len(self._map.known())
        self.get_logger().info(
            f'balls seen: {format_balls(balls, names)}; in map: {n_map}'
        )

    def _on_stats_timer(self) -> None:
        """
        Log timing, balls and map health every 10 s.

        ``seg_ms`` is YOLO, ``fit_ms`` ball fit + map + view checks;
        ``rot_deg_s`` the camera rotation speed in world_frame (p50/p95, to
        set ``max_camera_rot_deg_s``), ``match_mm`` how far the observations
        of known balls lie from their map position (p50/p95; grows with
        odom drift).
        """
        with self._stats_lock:
            count, seg_ms, fit_ms, n_balls = (
                self._infer_count, self._infer_ms_sum, self._fit_ms_sum,
                self._n_balls,
            )
            rejected, self._n_rejected = self._n_rejected, {}
            turning, speeds, residuals = (
                self._n_turning, self._rot_speeds, self._residuals
            )
            self._infer_count = self._n_balls = self._n_turning = 0
            self._infer_ms_sum = self._fit_ms_sum = 0.0
            self._rot_speeds, self._residuals = [], []
        with self._latest_cv:
            skipped, self._skipped = self._skipped, 0
        with self._map_lock:
            n_map = len(self._map.known())

        def avg(total: float) -> str:
            return f'{total / count:.1f}' if count else '-'

        def p50_95(values: list[float], scale: float = 1.0) -> str:
            if not values:
                return '-'
            a, b = np.percentile(np.asarray(values) * scale, [50, 95])
            return f'{a:.0f}/{b:.0f}'

        self.get_logger().info(
            f'ball stats (10 s): seg_ms_avg={avg(seg_ms)} fit_ms_avg={avg(fit_ms)} '
            f'n={count} skipped_frames={skipped} balls_seen={n_balls} '
            f"rejected={','.join(f'{k}:{v}' for k, v in rejected.items()) or 0} "
            f'map={n_map} match_mm={p50_95(residuals, 1000.0)} '
            f'rot_deg_s={p50_95(speeds)} turning_frames={turning} '
            f'backend={self._backend}'
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


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run g1_perception ball_node``."""
    cv2.setNumThreads(1)
    rclpy.init(args=args)
    node = BallNode()
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
