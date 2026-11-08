"""Convert voted slot results to ROS messages (no rclpy node logic)."""

from __future__ import annotations

from collections.abc import Sequence

from geometry_msgs.msg import Pose as PoseMsg
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from g1_perception.core.types import Pose
from g1_perception.core.voting import VotedSlot


def pose_to_msg(pose: Pose | None) -> PoseMsg:
    """Convert a :class:`Pose` to ``geometry_msgs/Pose`` (identity if None)."""
    msg = PoseMsg()
    if pose is None:
        return msg
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


def voted_to_detection_array(
    voted: Sequence[VotedSlot],
    header: Header,
    cube_size: float,
) -> Detection3DArray:
    """
    Build one ``Detection3D`` per slot.

    ``id`` is the slot number, or ``t<track_id>`` when place assignment
    failed (``assignment_ok`` False), ``results[0].hypothesis`` the canonical
    label and voted score, ``results[0].pose.pose`` and ``bbox.center`` the
    pose. ``bbox.size`` is ``cube_size`` on each axis when a pose exists and
    zero otherwise (pose fields are then identity and must be ignored).
    """
    array = Detection3DArray()
    array.header = header
    for slot in voted:
        det = Detection3D()
        det.header = header
        det.id = str(slot.slot) if slot.assignment_ok else f't{slot.slot}'

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = slot.label
        hypothesis.hypothesis.score = float(slot.score)
        hypothesis.pose.pose = pose_to_msg(slot.pose)
        det.results.append(hypothesis)

        det.bbox.center = pose_to_msg(slot.pose)
        size = float(cube_size) if slot.pose is not None else 0.0
        det.bbox.size.x = det.bbox.size.y = det.bbox.size.z = size

        array.detections.append(det)
    return array
