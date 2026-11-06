"""Tests for torch-free YOLO pre/post-processing (letterbox, NMS, decode)."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.segmenters.yolo_decode import decode, letterbox, nms

NAMES = {0: 'red', 1: 'green'}


def _raw(rows: list[list[float]]) -> np.ndarray:
    """Candidates as rows ``[cx, cy, w, h, s0, s1, extra...]`` → (1, C, N)."""
    return np.asarray(rows, dtype=np.float32).T[None]


def test_letterbox_640x480_pads_top_and_bottom():
    img = np.full((480, 640, 3), 50, dtype=np.uint8)
    blob, info = letterbox(img, (640, 640))
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert info.gain == pytest.approx(1.0)
    assert (info.pad_x, info.pad_y) == (0.0, 80.0)
    assert blob[0, 0, 0, 0] == pytest.approx(114 / 255)
    assert blob[0, 0, 320, 320] == pytest.approx(50 / 255)
    np.testing.assert_allclose(info.to_image(np.array([[10.0, 90.0]])), [[10.0, 10.0]])


def test_letterbox_scales_large_image():
    _blob, info = letterbox(np.zeros((960, 1280, 3), np.uint8), (640, 640))
    assert info.gain == pytest.approx(0.5)
    np.testing.assert_allclose(info.to_image(np.array([[320.0, 320.0]])), [[640.0, 480.0]])


def test_nms_suppresses_overlaps_by_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    keep = nms(boxes, np.array([0.5, 0.9, 0.7]), iou_thres=0.5)
    assert list(keep) == [1, 2]


def test_decode_pose_filters_nms_per_class_and_maps_keypoints():
    kpts = [100.0, 90.0, 0.9, 120.0, 90.0, 0.8, 120.0, 110.0, 0.7, 100.0, 110.0, 0.6]
    rows = [
        [110, 100, 20, 20, 0.9, 0.1, *kpts],
        [111, 101, 20, 20, 0.8, 0.1, *kpts],  # same class, overlaps → removed
        [111, 101, 20, 20, 0.1, 0.85, *kpts],  # other class → kept
        [300, 300, 20, 20, 0.1, 0.1, *kpts],  # below conf → dropped
    ]
    _blob, info = letterbox(np.zeros((480, 640, 3), np.uint8))
    res = decode([_raw(rows)], info, task='pose', names=NAMES, kpt_shape=(4, 3))
    assert len(res.boxes) == 2
    assert sorted(res.boxes.cls.tolist()) == [0.0, 1.0]
    first = int(np.argmax(res.boxes.conf))
    np.testing.assert_allclose(res.boxes.xyxy[first], [100, 10, 120, 30], atol=1e-4)
    np.testing.assert_allclose(res.keypoints.data[first, 0], [100, 10, 0.9], atol=1e-4)
    assert res.masks is None and res.names == NAMES


def test_decode_segment_builds_masks_inside_boxes():
    rows = [[320, 320, 64, 64, 0.95, 0.0, 1.0, 0.0]]  # 2 mask coefficients
    proto = np.zeros((1, 2, 160, 160), dtype=np.float32)
    proto[0, 0] = 1.0  # positive logit everywhere → mask = box
    _blob, info = letterbox(np.zeros((480, 640, 3), np.uint8))
    res = decode([_raw(rows), proto], info, task='segment', names=NAMES)
    mask = res.masks.data[0]
    assert mask.shape == (480, 640)
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (288, 351, 208, 271)


def test_decode_empty_and_bad_task():
    _blob, info = letterbox(np.zeros((480, 640, 3), np.uint8))
    res = decode([_raw([[1, 1, 1, 1, 0.0, 0.0]])], info, task='detect', names=NAMES)
    assert len(res.boxes) == 0
    with pytest.raises(ValueError, match='task'):
        decode([_raw([[1, 1, 1, 1, 0.0, 0.0]])], info, task='obb', names=NAMES)
