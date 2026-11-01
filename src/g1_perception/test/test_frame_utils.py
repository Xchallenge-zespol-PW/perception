"""Tests for depth / intrinsics conversion and pose frame transforms."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.frame_utils import depth_to_m, intrinsics_from_k
from g1_perception.core.geometry import (
    quat_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
    transform_pose,
)
from g1_perception.core.types import Pose


def test_depth_16uc1_millimetres_to_metres():
    depth = np.array([[0, 1000], [1234, 65535]], dtype=np.uint16)

    depth_m = depth_to_m(depth, '16UC1')

    assert depth_m.dtype == np.float32
    assert np.allclose(depth_m, [[0.0, 1.0], [1.234, 65.535]])


def test_depth_32fc1_keeps_metres_and_zeroes_invalid():
    depth = np.array([[0.5, np.nan], [np.inf, 1.25]], dtype=np.float32)

    depth_m = depth_to_m(depth, '32FC1')

    assert np.allclose(depth_m, [[0.5, 0.0], [0.0, 1.25]])
    assert np.isnan(depth[0, 1])  # input untouched


def test_depth_unknown_encoding_raises():
    with pytest.raises(ValueError, match='Unsupported'):
        depth_to_m(np.zeros((2, 2), dtype=np.uint8), 'mono8')


def test_intrinsics_from_k():
    K = intrinsics_from_k([500.0, 0, 320, 0, 501.0, 240, 0, 0, 1])

    assert K.shape == (3, 3)
    assert K[0, 0] == 500.0 and K[1, 1] == 501.0
    assert K[0, 2] == 320 and K[1, 2] == 240

    with pytest.raises(ValueError):
        intrinsics_from_k([1.0, 2.0])


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1]]
    )


def test_transform_pose_applies_rotation_then_translation():
    pose = Pose(
        position=np.array([1.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        score=0.7,
    )
    rotation = rotation_matrix_to_quat_xyzw(_rot_z(90.0))

    out = transform_pose(pose, np.array([0.0, 0.0, 2.0]), rotation)

    assert np.allclose(out.position, [0.0, 1.0, 2.0])
    assert np.allclose(quat_to_rotation_matrix(out.quat), _rot_z(90.0))
    assert out.score == 0.7


def test_transform_pose_composes_orientations():
    pose = Pose(
        position=np.zeros(3),
        quat=rotation_matrix_to_quat_xyzw(_rot_z(30.0)),
        score=1.0,
    )
    rotation = rotation_matrix_to_quat_xyzw(_rot_z(45.0))

    out = transform_pose(pose, np.zeros(3), rotation)

    assert np.allclose(quat_to_rotation_matrix(out.quat), _rot_z(75.0))
