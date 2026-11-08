"""Tests for VotedSlot → vision_msgs/Detection3DArray conversion."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('vision_msgs.msg')

from std_msgs.msg import Header  # noqa: E402

from g1_perception.core.types import (  # noqa: E402
    LABEL_EMPTY,
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Pose,
)
from g1_perception.core.voting import VotedSlot  # noqa: E402
from g1_perception.ros_conversions import voted_to_detection_array  # noqa: E402


def _pose(x: float) -> Pose:
    return Pose(
        position=np.array([x, 0.2, 0.9]),
        quat=np.array([0.0, 0.0, 0.38268343, 0.92387953]),
        score=0.8,
    )


def test_one_detection_per_slot_with_contract_fields():
    header = Header(frame_id='camera_link')
    header.stamp.sec = 12
    voted = [
        VotedSlot(0, LABEL_RED, 0.9, 10, _pose(0.1), 45.0, 1.0),
        VotedSlot(1, LABEL_EMPTY, 1.0, 10, None, None, None),
        VotedSlot(2, LABEL_GREEN, 0.8, 10, _pose(0.3), 45.0, 1.0),
        VotedSlot(3, LABEL_UNCERTAIN, 0.5, 10, None, None, None),
    ]

    array = voted_to_detection_array(voted, header, 0.05)

    assert array.header.frame_id == 'camera_link'
    assert array.header.stamp.sec == 12
    assert [d.id for d in array.detections] == ['0', '1', '2', '3']
    assert [d.results[0].hypothesis.class_id for d in array.detections] == [
        LABEL_RED, LABEL_EMPTY, LABEL_GREEN, LABEL_UNCERTAIN,
    ]

    red = array.detections[0]
    assert len(red.results) == 1
    assert red.header.frame_id == 'camera_link'
    assert red.results[0].hypothesis.score == pytest.approx(0.9)
    p = red.results[0].pose.pose
    assert (p.position.x, p.position.y, p.position.z) == pytest.approx(
        (0.1, 0.2, 0.9)
    )
    assert (p.orientation.z, p.orientation.w) == pytest.approx(
        (0.38268343, 0.92387953)
    )
    assert red.bbox.center == p
    assert (red.bbox.size.x, red.bbox.size.y, red.bbox.size.z) == (
        0.05, 0.05, 0.05
    )

    empty = array.detections[1]
    assert empty.bbox.size.x == 0.0
    assert empty.results[0].pose.pose.orientation.w == 1.0


def test_failed_assignment_uses_track_id_prefix_and_keeps_colour():
    header = Header(frame_id='camera_link')
    voted = [
        VotedSlot(22, LABEL_RED, 0.9, 10, _pose(0.1), 18.0, 1.0,
                  track_id=22, assignment_ok=False),
        VotedSlot(2, LABEL_GREEN, 0.8, 10, _pose(0.3), 0.0, 1.0,
                  track_id=7, assignment_ok=True),
    ]

    array = voted_to_detection_array(voted, header, 0.05)

    assert [d.id for d in array.detections] == ['t22', '2']
    assert array.detections[0].results[0].hypothesis.class_id == LABEL_RED
    assert array.detections[0].bbox.size.x == pytest.approx(0.05)
