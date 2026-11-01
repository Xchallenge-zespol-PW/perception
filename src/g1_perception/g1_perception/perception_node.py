"""ROS 2 node: RGB-D → per-slot cube colour and pose (thin wrapper)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from vision_msgs.msg import Detection3DArray

from g1_perception.core.debug_vis import draw_debug, union_mask
from g1_perception.core.frame_utils import depth_to_m, intrinsics_from_k
from g1_perception.core.geometry import transform_pose
from g1_perception.core.types import Frame
from g1_perception.core.voting import SlotVoter, VotedSlot
from g1_perception.node_config import (
    DEFAULT_PARAMS,
    build_pipeline_config,
    voter_kwargs,
)
from g1_perception.pipeline import Pipeline
from g1_perception.ros_conversions import voted_to_detection_array


class PerceptionNode(Node):
    """
    Synchronize color + aligned depth + camera_info and publish cube slots.

    Synced messages are buffered; a timer at ``publish_rate_hz`` processes
    the newest unprocessed set, updates the voter and publishes
    ``Detection3DArray``, the debug image and the mono8 mask. Spin with a
    multi-threaded executor: the timer has its own callback group so TF
    keeps arriving while it waits in ``lookup_transform``.
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
        self._latest: tuple[Image, Image, CameraInfo] | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pub_cubes = self.create_publisher(
            Detection3DArray, p['output_topic'], 10
        )
        self._pub_debug = self.create_publisher(
            Image, p['debug_image_topic'], 1
        )
        self._pub_mask = self.create_publisher(Image, p['mask_topic'], 1)

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

        self._timer = self.create_timer(
            1.0 / float(p['publish_rate_hz']),
            self._on_timer,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f"segmenter={p['segmenter']} pose={p['pose_estimator']} "
            f'slots={[r.slot for r in self._rois]} '
            f"target_frame={p['target_frame'] or '(camera)'} "
            f"-> {p['output_topic']}"
        )

    def _on_synced(self, color: Image, depth: Image, info: CameraInfo) -> None:
        self._latest = (color, depth, info)

    def _on_timer(self) -> None:
        if self._latest is None:
            return
        color, depth, info = self._latest
        self._latest = None

        try:
            frame = self._to_frame(color, depth, info)
        except (CvBridgeError, ValueError) as exc:
            self.get_logger().warning(
                f'Bad input frame: {exc}', throttle_duration_sec=5.0
            )
            return

        try:
            results = self._pipeline.process(frame)
        except Exception as exc:  # keep the node alive on per-frame failures
            self.get_logger().error(
                f'Pipeline failed: {exc!r}', throttle_duration_sec=5.0
            )
            return
        if not self.context.ok():  # shutdown arrived during processing
            return
        voted = self._voter.update(results)

        camera_header = Header(
            stamp=color.header.stamp, frame_id=color.header.frame_id
        )
        out_voted, frame_id = self._to_target_frame(voted, camera_header)
        out_header = Header(stamp=color.header.stamp, frame_id=frame_id)
        self._pub_cubes.publish(
            voted_to_detection_array(
                out_voted, out_header, float(self._params['cube_size_m'])
            )
        )

        mask = union_mask(results, frame.depth_m.shape)
        mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = camera_header
        self._pub_mask.publish(mask_msg)

        if self._pub_debug.get_subscription_count() > 0:
            debug = draw_debug(frame.rgb, self._rois, results, voted)
            debug_msg = self._bridge.cv2_to_imgmsg(debug, encoding='rgb8')
            debug_msg.header = camera_header
            self._pub_debug.publish(debug_msg)

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
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
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
