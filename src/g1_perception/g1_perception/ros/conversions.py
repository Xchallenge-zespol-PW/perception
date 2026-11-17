"""Convert perception results to ROS messages (no rclpy node logic)."""

from __future__ import annotations

from collections.abc import Sequence

from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from xchallenge_msgs.msg import CubeColors

from g1_perception.core.cube_registry import NamedCube
from g1_perception.core.types import LABEL_GREEN, LABEL_RED, Pose
from g1_perception.core.voting import VotedSlot

_CUBE_COLORS = {LABEL_RED: CubeColors.RED, LABEL_GREEN: CubeColors.GREEN}


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


def pose_to_transform(
    pose: Pose,
    header: Header,
    child_frame_id: str,
) -> TransformStamped:
    """Build the TF ``header.frame_id -> child_frame_id`` placed at ``pose``."""
    msg = TransformStamped()
    msg.header = header
    msg.child_frame_id = child_frame_id
    (
        msg.transform.translation.x,
        msg.transform.translation.y,
        msg.transform.translation.z,
    ) = (float(v) for v in pose.position)
    (
        msg.transform.rotation.x,
        msg.transform.rotation.y,
        msg.transform.rotation.z,
        msg.transform.rotation.w,
    ) = (float(v) for v in pose.quat)
    return msg


def cube_colors_msg(cubes: Sequence[NamedCube]) -> CubeColors:
    """
    Build ``xchallenge_msgs/CubeColors``: ``cube_names[i]`` has ``colors[i]``.

    Raises ValueError for a label other than red / green, which the logic
    node of exercise 1 would reject.
    """
    msg = CubeColors()
    for cube in cubes:
        if cube.label not in _CUBE_COLORS:
            raise ValueError(f'{cube.name}: no CubeColors colour for {cube.label!r}')
        msg.cube_names.append(cube.name)
        msg.colors.append(_CUBE_COLORS[cube.label])
    return msg
