"""ROS 2 node: RGB-D → per-slot cube colour and pose (thin wrapper)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Ultralytics may auto-upgrade numpy to 2.x and break Humble cv_bridge.
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

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
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from tf2_ros import (  # noqa: E402
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from vision_msgs.msg import Detection3DArray  # noqa: E402

from g1_perception.core.debug_vis import draw_debug, union_mask  # noqa: E402
from g1_perception.core.frame_utils import (  # noqa: E402
    depth_to_m,
    intrinsics_from_k,
)
from g1_perception.core.geometry import transform_pose  # noqa: E402
from g1_perception.core.types import Frame, SlotResult  # noqa: E402
from g1_perception.core.voting import SlotVoter, VotedSlot  # noqa: E402
from g1_perception.node_config import (  # noqa: E402
    DEFAULT_PARAMS,
    build_pipeline_config,
    voter_kwargs,
)
from g1_perception.pipeline import Pipeline  # noqa: E402
from g1_perception.ros_conversions import voted_to_detection_array  # noqa: E402


@dataclass
class _LatestBundle:
    """One synchronized RGB-D + camera_info message set."""

    color: Image
    depth: Image
    info: CameraInfo


@dataclass
class _PublishBundle:
    """Last finished inference result ready for publishers."""

    camera_header: Header
    results: list[SlotResult]
    voted: list[VotedSlot]
    rgb: np.ndarray
    inference_ms: float


class PerceptionNode(Node):
    """
    Synchronize color + aligned depth + camera_info and publish cube slots.

    The sync callback only keeps the **newest** message triple (no queue).
    A worker thread always processes that newest set (~125 ms / frame on the
    robot CPU), so older frames are dropped instead of building latency.
    Results are published at ``publish_rate_hz``. Spin with a multi-threaded
    executor so TF callbacks keep running during ``lookup_transform``.
    """

    def __init__(self) -> None:
        super().__init__('perception_node')
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        self._params: dict[str, Any] = {
            name: self.get_parameter(name).value for name in DEFAULT_PARAMS
        }
        p = self._params

        if p['segmenter'] == 'yolo_seg' and not Path(p['model_path']).is_file():
            raise FileNotFoundError(
                f"model_path not found: {p['model_path']!r} "
                f'(relative paths are resolved against cwd: {Path.cwd()})'
            )

        self._pipeline = Pipeline.from_config(build_pipeline_config(p))
        self._rois = self._pipeline.rois
        self._voter = SlotVoter(**voter_kwargs(p))
        self._bridge = CvBridge()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pub_cubes = self.create_publisher(
            Detection3DArray, p['output_topic'], 10
        )
        self._pub_debug = self.create_publisher(
            Image, p['debug_image_topic'], 1
        )
        self._pub_mask = self.create_publisher(Image, p['mask_topic'], 1)

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
            f'slots={[r.slot for r in self._rois]} '
            f"target_frame={p['target_frame'] or '(camera)'} "
            f"publish_rate_hz={rate} -> {p['output_topic']}"
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
        """Convert, run pipeline + voter, store result for the publish timer."""
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

        voted = self._voter.update(results)
        camera_header = Header(
            stamp=bundle.color.header.stamp,
            frame_id=bundle.color.header.frame_id,
        )
        with self._result_lock:
            self._result = _PublishBundle(
                camera_header=camera_header,
                results=results,
                voted=voted,
                rgb=frame.rgb,
                inference_ms=inference_ms,
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

        out_voted, frame_id = self._to_target_frame(
            bundle.voted, bundle.camera_header
        )
        out_header = Header(
            stamp=bundle.camera_header.stamp, frame_id=frame_id
        )
        self._pub_cubes.publish(
            voted_to_detection_array(
                out_voted, out_header, float(self._params['cube_size_m'])
            )
        )

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
            )
            debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding='rgb8')
            debug_msg.header = bundle.camera_header
            self._pub_debug.publish(debug_msg)

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
            f'n={count} skipped_frames={skipped}'
        )

    def _to_frame(self, color: Image, depth: Image, info: CameraInfo) -> Frame:
        """Convert ROS messages to a :class:`Frame` (RGB, depth in metres)."""
        # desired_encoding='rgb8' yields RGB for Frame; YOLO converts to BGR.
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

    def _to_target_frame(
        self,
        voted: list[VotedSlot],
        camera_header: Header,
    ) -> tuple[list[VotedSlot], str]:
        """Transform poses to ``target_frame``; fall back to camera frame."""
        target = str(self._params['target_frame']).strip()
        source = camera_header.frame_id
        if not target or target == source:
            return voted, source

        try:
            tf = self._tf_buffer.lookup_transform(
                target,
                source,
                Time.from_msg(camera_header.stamp),
                timeout=Duration(seconds=float(self._params['tf_timeout_s'])),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as exc:
            self.get_logger().warning(
                f'No TF {source} -> {target} ({exc}); publishing in {source}',
                throttle_duration_sec=5.0,
            )
            return voted, source

        t = tf.transform.translation
        q = tf.transform.rotation
        translation = np.array([t.x, t.y, t.z])
        rotation = np.array([q.x, q.y, q.z, q.w])
        return [
            replace(v, pose=transform_pose(v.pose, translation, rotation))
            if v.pose is not None else v
            for v in voted
        ], target


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
