"""Tests for shared pipeline types."""

from __future__ import annotations

import numpy as np

from g1_perception.core.types import Detection


def test_detection_keypoints_default_to_none():
    det = Detection(0, 'red', 0.9, np.zeros((4, 4), dtype=bool))
    assert det.keypoints is None


def test_detection_keeps_keypoints():
    kpts = np.array([[1.0, 2.0, 0.9]] * 4)
    det = Detection(None, 'red', 0.9, np.zeros((4, 4), dtype=bool), keypoints=kpts)
    assert det.keypoints is not None and det.keypoints.shape == (4, 3)
