"""ROS 2 node: colour image → AprilTag poses (TF, detections, debug image)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose as PoseMsg, TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
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
    TransformBroadcaster,
    TransformListener,
)
from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from g1_perception.core.apriltag import (
    AprilTagDetector,
    parse_tag_ids,
    TagDetection,
)
from g1_perception.core.frame_utils import intrinsics_from_k
from g1_perception.core.geometry import (
    quat_to_rotation_matrix,
    transform_pose,
)
from g1_perception.core.types import Pose

DEFAULT_PARAMS = {
    'image_topic': '/camera/camera/color/image_raw',
    'camera_info_topic': '/camera/camera/color/camera_info',
    'output_topic': '/apriltag/detections',
    'debug_image_topic': '/apriltag/debug_image',
    'family': '36h11',
    'tag_size_m': 0.10,
    'tag_ids': '',
    'max_reproj_px': 2.0,
    'target_frame': '',
    'tf_prefix': 'tag_',
    'publish_tf': True,
    'tf_timeout_s': 0.05,
    'stats_period_s': 10.0,
}


class AprilTagNode(Node):
    """
    Detect AprilTags in every colour frame.

    Each tag with a pose becomes the TF ``<tf_prefix><id>`` under the camera
    frame (stamp of the image) and one ``Detection3D`` (``id`` = tag id,
    ``class_id`` = ``tag<family>``, ``score`` = 1 / (1 + reprojection px))
    on ``output_topic``, in ``target_frame`` when its TF is known, else in
    the camera frame.
    """

    def __init__(self) -> None:
        super().__init__('apriltag_node')
        for name, default in DEFAULT_PARAMS.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in DEFAULT_PARAMS}
        self._p = p

        self._detector = AprilTagDetector(
            family=str(p['family']),
            tag_size=float(p['tag_size_m']),
            tag_ids=parse_tag_ids(p['tag_ids']),
            max_reproj_px=float(p['max_reproj_px']),
        )
        self._class_id = f'tag{self._detector.family}'
        self._bridge = CvBridge()
        self._K: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._target = str(p['target_frame']).strip()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_pub = TransformBroadcaster(self) if p['publish_tf'] else None
        self._pub = self.create_publisher(
            Detection3DArray, str(p['output_topic']), 10
        )
        self._pub_debug = self.create_publisher(
            Image, str(p['debug_image_topic']), 1
        )
        self.create_subscription(
            CameraInfo, str(p['camera_info_topic']), self._on_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, str(p['image_topic']), self._on_image,
            qos_profile_sensor_data,
        )

        self._n_frames = 0
        self._n_with_tags = 0
        self._ms_sum = 0.0
        self._ids_seen: set[int] = set()
        self.create_timer(float(p['stats_period_s']), self._log_stats)
        ids = sorted(self._detector.tag_ids) or 'all'
        self.get_logger().info(
            f'{self._class_id} size={self._detector.tag_size} m ids={ids} '
            f'{p["image_topic"]} -> {p["output_topic"]}'
            + (f' in {self._target}' if self._target else '')
        )

    def _on_info(self, msg: CameraInfo) -> None:
        self._K = intrinsics_from_k(msg.k)
        self._D = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else None

    def _on_image(self, msg: Image) -> None:
        if self._K is None:
            self.get_logger().warning(
                'No camera_info yet', throttle_duration_sec=5.0
            )
            return
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as exc:
            self.get_logger().warning(
                f'Bad image: {exc}', throttle_duration_sec=5.0
            )
            return

        t0 = time.perf_counter()
        tags = self._detector.detect(rgb, self._K, self._D)
        self._ms_sum += (time.perf_counter() - t0) * 1000.0
        self._n_frames += 1
        self._n_with_tags += bool(tags)
        self._ids_seen.update(t.tag_id for t in tags)

        posed = [t for t in tags if t.pose is not None]
        if self._tf_pub is not None and posed:
            self._tf_pub.sendTransform([
                _transform(msg.header, f'{self._p["tf_prefix"]}{t.tag_id}',
                           t.pose)
                for t in posed
            ])
        self._pub.publish(self._detections(msg.header, posed))
        if self._pub_debug.get_subscription_count() > 0:
            self._publish_debug(rgb, tags, msg.header)

    def _detections(
        self,
        header: Header,
        tags: list[TagDetection],
    ) -> Detection3DArray:
        """Build the detection array, in target_frame when its TF is known."""
        out_header = header
        tf = self._lookup(header) if self._target and tags else None
        if tf is not None:
            out_header = Header(stamp=header.stamp, frame_id=self._target)
        array = Detection3DArray(header=out_header)
        size = self._detector.tag_size
        for tag in tags:
            pose = tag.pose
            assert pose is not None
            if tf is not None:
                pose = transform_pose(pose, *tf)
            det = Detection3D(header=out_header, id=str(tag.tag_id))
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self._class_id
            hypothesis.hypothesis.score = float(pose.score)
            hypothesis.pose.pose = _pose_msg(pose)
            det.results.append(hypothesis)
            det.bbox.center = _pose_msg(pose)
            det.bbox.size.x = det.bbox.size.y = size
            array.detections.append(det)
        return array

    def _lookup(self, header: Header) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(translation, rotation_xyzw)`` of ``target <- camera``."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._target,
                header.frame_id,
                Time.from_msg(header.stamp),
                timeout=Duration(seconds=float(self._p['tf_timeout_s'])),
            )
        except (
            LookupException,
            ConnectivityException,
            ExtrapolationException,
        ) as exc:
            self.get_logger().warning(
                f'No TF {header.frame_id} -> {self._target} ({exc}); '
                f'publishing in {header.frame_id}',
                throttle_duration_sec=5.0,
            )
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return np.array([t.x, t.y, t.z]), np.array([q.x, q.y, q.z, q.w])

    def _publish_debug(
        self,
        rgb: np.ndarray,
        tags: list[TagDetection],
        header: Header,
    ) -> None:
        """Draw tag outlines, ids and axes (x red, y green, z blue)."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if tags:
            cv2.aruco.drawDetectedMarkers(
                bgr,
                [t.corners.reshape(1, 4, 2).astype(np.float32) for t in tags],
                np.array([[t.tag_id] for t in tags], dtype=np.int32),
            )
        for tag in tags:
            if tag.pose is None:
                continue
            cv2.drawFrameAxes(
                bgr, self._K, self._D, _quat_to_rvec(tag.pose.quat),
                tag.pose.position,
                self._detector.tag_size / 2.0,
            )
        out = self._bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        out.header = header
        self._pub_debug.publish(out)

    def _log_stats(self) -> None:
        if self._n_frames == 0:
            return
        self.get_logger().info(
            f'apriltag stats: frames={self._n_frames} '
            f'with_tags={self._n_with_tags} '
            f'ids={sorted(self._ids_seen)} '
            f'detect_ms_avg={self._ms_sum / self._n_frames:.1f}'
        )
        self._n_frames = self._n_with_tags = 0
        self._ms_sum = 0.0
        self._ids_seen = set()


def _pose_msg(pose: Pose) -> PoseMsg:
    msg = PoseMsg()
    msg.position.x, msg.position.y, msg.position.z = (
        float(v) for v in pose.position
    )
    (
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
    ) = (float(v) for v in pose.quat)
    return msg


def _transform(header: Header, child: str, pose: Pose) -> TransformStamped:
    msg = TransformStamped(header=header, child_frame_id=child)
    p = _pose_msg(pose)
    msg.transform.translation.x = p.position.x
    msg.transform.translation.y = p.position.y
    msg.transform.translation.z = p.position.z
    msg.transform.rotation = p.orientation
    return msg


def _quat_to_rvec(quat: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(quat_to_rotation_matrix(quat))
    return rvec


def main(args: list[str] | None = None) -> None:
    """Entry point for ``ros2 run g1_perception apriltag_node``."""
    rclpy.init(args=args)
    node = AprilTagNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
