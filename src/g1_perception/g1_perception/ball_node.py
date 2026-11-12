"""ROS 2 node: RGB-D → red balls (exercise 2) as TF in the world frame."""

from __future__ import annotations

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
from std_msgs.msg import Header  # noqa: E402
from tf2_ros import (  # noqa: E402
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformBroadcaster,
    TransformListener,
)

from g1_perception.core.ball import (  # noqa: E402
    BallObservation,
    draw_balls,
    find_balls,
    format_balls,
)
from g1_perception.core.frame_utils import (  # noqa: E402
    depth_to_m,
    intrinsics_from_k,
)
from g1_perception.core.geometry import quat_to_rotation_matrix  # noqa: E402
from g1_perception.core.types import Frame  # noqa: E402
from g1_perception.segmenters.yolo_seg import YoloSegSegmenter  # noqa: E402

SEEN_FRAME_PREFIX = 'ball_seen_'

DEFAULT_PARAMS: dict[str, Any] = {
    'color_topic': '/camera/d435/color/image_raw',
    'depth_topic': '/camera/d435/aligned_depth_to_color/image_raw',
    'camera_info_topic': '/camera/d435/color/camera_info',
    'balls_topic': '/perception/balls',
    'debug_image_topic': '/perception/balls/debug_image',
    'world_frame': 'odom',
    'tf_timeout_s': 0.1,
    'model_path': '/ws/models/cubes_seg_v4_aug.engine',
    'device': '0',
    'fallback_model_path': '/ws/models/cubes_seg_v4_aug.onnx',
    'fallback_device': 'cpu',
    'warmup_runs': 2,
    'conf': 0.35,
    'ball_labels': ['red ball'],
    'border_px': 3,
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
    Detect red balls and publish each one seen as TF ``ball_seen_<K>``.

    No tracking or memory yet. Every processed frame publishes the balls
    it sees whose outline was fitted and whose depth profile is a ball
    (:attr:`BallObservation.trusted`; a ball hidden too much, a flat red
    blob or a second mask on one ball is only logged and drawn), nearest
    first, as ``ball_seen_1..N`` in ``world_frame`` stamped with that
    frame, and the same centres as a PoseArray on ``balls_topic``. ``K`` is the
    rank by distance in that frame, not a stable id; a ball out of view
    leaves its old frame in TF, whose stamp tells its age. Without the TF
    camera -> ``world_frame`` (or with ``world_frame: ""``) the balls are
    published in the camera frame.

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
        self._pub_balls = self.create_publisher(PoseArray, p['balls_topic'], 10)
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
        self._worker.start()
        log.info(
            f"balls {self._labels} from {p['color_topic']} -> TF "
            f"{SEEN_FRAME_PREFIX}<K> in {self._world_frame or '(camera)'}, "
            f"{p['balls_topic']}"
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
        balls = find_balls(
            detections, frame.depth_m, frame.K,
            labels=self._labels, border_px=self._border_px,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
        if self._stop.is_set() or not self.context.ok():
            return

        header = Header(
            stamp=bundle.color.header.stamp,
            frame_id=bundle.color.header.frame_id,
        )
        self._publish(balls, header)
        self._log_balls(balls)

        if self._pub_debug.get_subscription_count() > 0:
            debug = draw_balls(
                frame.rgb, balls,
                inference_ms=inference_ms, backend=self._backend,
            )
            msg = self._bridge.cv2_to_imgmsg(debug, encoding='rgb8')
            msg.header = header
            self._pub_debug.publish(msg)

        with self._stats_lock:
            self._infer_ms_sum += inference_ms
            self._infer_count += 1
            for b in balls:
                if b.trusted:
                    self._n_balls += 1
                else:
                    reason = b.reject_reason
                    self._n_rejected[reason] = self._n_rejected.get(reason, 0) + 1

    def _publish(self, balls: list[BallObservation], camera: Header) -> None:
        """Publish TF ``ball_seen_<K>`` and the PoseArray of whole balls."""
        stamp_ns = Time.from_msg(camera.stamp).nanoseconds
        if stamp_ns <= self._last_stamp_ns:  # TF_REPEATED_DATA on bag --loop
            return
        self._last_stamp_ns = stamp_ns
        centres = [b.position for b in balls if b.trusted]
        frame_id = camera.frame_id
        tf = self._lookup_world(camera)
        if tf is not None:
            translation, rotation = tf
            R = quat_to_rotation_matrix(rotation)
            centres = [R @ c + translation for c in centres]
            frame_id = self._world_frame
        header = Header(stamp=camera.stamp, frame_id=frame_id)

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
        self._pub_balls.publish(array)
        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

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
                f'publishing balls in {camera.frame_id}',
                throttle_duration_sec=5.0,
            )
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])

    def _log_balls(self, balls: list[BallObservation]) -> None:
        """Log the balls in camera distance, at most every ``log_period_s``."""
        period = float(self._params['log_period_s'])
        now = time.monotonic()
        if period <= 0.0 or now - self._last_log < period:
            return
        self._last_log = now
        self.get_logger().info(f'balls: {format_balls(balls)}')

    def _on_stats_timer(self) -> None:
        """Log mean inference time and dropped frames every 10 s."""
        with self._stats_lock:
            count, total_ms, n_balls = (
                self._infer_count, self._infer_ms_sum, self._n_balls
            )
            rejected, self._n_rejected = self._n_rejected, {}
            self._infer_count = self._n_balls = 0
            self._infer_ms_sum = 0.0
        with self._latest_cv:
            skipped, self._skipped = self._skipped, 0
        avg = f'{total_ms / count:.1f}' if count else '-'
        self.get_logger().info(
            f'ball stats (10 s): infer_ms_avg={avg} n={count} '
            f'skipped_frames={skipped} balls_seen={n_balls} '
            f"rejected={','.join(f'{k}:{v}' for k, v in rejected.items()) or 0} "
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
