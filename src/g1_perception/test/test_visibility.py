"""Tests for the pose update gates (cut-off / hidden cube, camera turning)."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.geometry import rotation_matrix_to_quat_xyzw
from g1_perception.core.types import Detection, Frame, Pose, SlotResult
from g1_perception.core.visibility import (
    angular_speed_deg_s,
    mark_hidden,
    mask_touches_border,
    visible_ratio,
)

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
SIZE = 0.05
DEPTH = 0.6
SIDE_PX = int(round(600.0 * SIZE / DEPTH))  # one cube face at 0.6 m = 50 px


def _mask(x0: int, y0: int, w: int, h: int) -> np.ndarray:
    mask = np.zeros((480, 640), dtype=bool)
    mask[y0:y0 + h, x0:x0 + w] = True
    return mask


def _frame() -> Frame:
    depth = np.full((480, 640), DEPTH, dtype=np.float32)
    return Frame(
        rgb=np.zeros((480, 640, 3), np.uint8), depth_m=depth, K=K, stamp=0.0
    )


def _result(mask: np.ndarray) -> SlotResult:
    pose = Pose(position=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]), score=1.0)
    return SlotResult(
        slot=None, label='red', score=0.9,
        detection=Detection(slot=None, label='red', score=0.9, mask=mask),
        pose=pose,
    )


def _quat_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return rotation_matrix_to_quat_xyzw(R)


def test_mask_touches_border():
    assert mask_touches_border(_mask(0, 100, 50, 50), 3)
    assert mask_touches_border(_mask(589, 100, 50, 50), 3)
    assert not mask_touches_border(_mask(100, 100, 50, 50), 3)
    assert not mask_touches_border(_mask(0, 100, 50, 50), 0)


def test_visible_ratio_whole_and_half_face():
    frame = _frame()
    whole = visible_ratio(_mask(100, 100, SIDE_PX, SIDE_PX), frame.depth_m, K, SIZE)
    half = visible_ratio(_mask(100, 100, SIDE_PX // 2, SIDE_PX), frame.depth_m, K, SIZE)
    assert whole == pytest.approx(1.0, abs=0.01)
    assert half == pytest.approx(0.5, abs=0.02)
    no_depth = np.zeros_like(frame.depth_m)
    assert np.isnan(visible_ratio(_mask(100, 100, 10, 10), no_depth, K, SIZE))


def test_mark_hidden_flags_cut_off_and_small_masks():
    results = [
        _result(_mask(100, 100, SIDE_PX, SIDE_PX)),       # whole
        _result(_mask(100, 100, SIDE_PX // 3, SIDE_PX)),  # mostly hidden
        _result(_mask(0, 200, SIDE_PX, SIDE_PX)),         # cut by the edge
        SlotResult(slot=None, label='uncertain', score=0.5, detection=None, pose=None),
    ]
    out, marked = mark_hidden(
        results, _frame(), size=SIZE, border_px=3, min_visible_ratio=0.45
    )
    assert [r.pose_trusted for r in out] == [True, False, False, True]
    assert marked == 2
    off, n = mark_hidden(results, _frame(), size=SIZE)
    assert n == 0 and all(r.pose_trusted for r in off)


def test_angular_speed():
    assert angular_speed_deg_s(_quat_z(0.0), 0.0, _quat_z(5.0), 0.1) == pytest.approx(50.0)
    assert angular_speed_deg_s(_quat_z(0.0), 0.0, _quat_z(-1.0), 0.1) == pytest.approx(10.0)
    assert angular_speed_deg_s(_quat_z(0.0), 1.0, _quat_z(30.0), 1.0) == 0.0
