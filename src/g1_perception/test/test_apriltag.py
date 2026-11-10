"""Tests for AprilTag detection and pose (synthetic warped tag images)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from g1_perception.core.apriltag import (
    AprilTagDetector,
    parse_tag_ids,
    tag_object_points,
)
from g1_perception.core.geometry import quat_to_rotation_matrix

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = 0.10


def _render(tags, shape=(480, 640)):
    """Draw tags ``(id, rvec, tvec)`` of side SIZE into one gray image."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side, border = 200, 50
    img = np.full(shape, 128, dtype=np.uint8)
    for tag_id, rvec, tvec in tags:
        canvas = np.full((side + 2 * border,) * 2, 255, dtype=np.uint8)
        draw = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
        canvas[border:-border, border:-border] = draw(dictionary, tag_id, side)
        src = np.array(
            [[border, border], [border + side, border],
             [border + side, border + side], [border, border + side]],
            dtype=np.float32,
        )
        dst, _ = cv2.projectPoints(tag_object_points(SIZE), rvec, tvec, K, None)
        H = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(canvas, H, shape[::-1], borderValue=0)
        inside = cv2.warpPerspective(
            np.ones_like(canvas), H, shape[::-1], borderValue=0
        ).astype(bool)
        img[inside] = warped[inside]
    return img


def test_parse_tag_ids():
    assert parse_tag_ids('') == ()
    assert parse_tag_ids('0, 3,7,') == (0, 3, 7)


def test_bad_family_and_size_raise():
    with pytest.raises(ValueError):
        AprilTagDetector(family='36h12')
    with pytest.raises(ValueError):
        AprilTagDetector(tag_size=0.0)
    assert AprilTagDetector(family='tag36h11').family == '36h11'


def test_tilted_tag_pose_is_recovered():
    rvec = np.array([np.deg2rad(140.0), 0.0, 0.2])
    tvec = np.array([0.05, 0.1, 0.8])
    tags = AprilTagDetector(tag_size=SIZE).detect(_render([(7, rvec, tvec)]), K)

    assert [t.tag_id for t in tags] == [7]
    tag = tags[0]
    assert tag.pose is not None
    assert np.linalg.norm(tag.pose.position - tvec) < 0.01
    R_true, _ = cv2.Rodrigues(rvec)
    R = quat_to_rotation_matrix(tag.pose.quat)
    # z of the tag frame points out of the printed face.
    assert float(R[:, 2] @ R_true[:, 2]) > np.cos(np.deg2rad(3.0))
    assert tag.reproj_px < 1.0


def test_ids_filter_and_sorting():
    img = _render([
        (5, np.array([np.pi, 0.0, 0.0]), np.array([0.15, 0.0, 0.9])),
        (2, np.array([np.pi, 0.0, 0.0]), np.array([-0.15, 0.0, 0.9])),
    ])
    assert [t.tag_id for t in AprilTagDetector().detect(img, K)] == [2, 5]
    only = AprilTagDetector(tag_ids=(5,)).detect(img, K)
    assert [t.tag_id for t in only] == [5]


def test_pose_dropped_above_reprojection_limit():
    img = _render([(1, np.array([np.pi, 0.0, 0.0]), np.array([0.0, 0.0, 0.7]))])
    tags = AprilTagDetector(max_reproj_px=0.0).detect(img, K)
    assert len(tags) == 1 and tags[0].pose is None
