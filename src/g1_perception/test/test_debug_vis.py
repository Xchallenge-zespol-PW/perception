"""Tests for debug image rendering and mask union."""

from __future__ import annotations

import numpy as np

from g1_perception.core.config import Roi
from g1_perception.core.debug_vis import (
    LABEL_COLORS_RGB,
    draw_debug,
    union_mask,
)
from g1_perception.core.types import (
    LABEL_EMPTY,
    LABEL_RED,
    Detection,
    SlotResult,
)
from g1_perception.core.voting import VotedSlot


def _results(h: int = 60, w: int = 80) -> list[SlotResult]:
    mask = np.zeros((h, w), dtype=bool)
    mask[20:30, 10:20] = True
    det = Detection(slot=0, label='red cube', score=0.9, mask=mask)
    return [
        SlotResult(slot=0, label=LABEL_RED, score=0.9, detection=det, pose=None),
        SlotResult(slot=1, label=LABEL_EMPTY, score=0.0, detection=None, pose=None),
    ]


def test_union_mask_is_mono8_of_all_detection_masks():
    results = _results()

    mask = union_mask(results, (60, 80))

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    assert int(np.count_nonzero(mask)) == 100
    assert mask[25, 15] == 255


def test_union_mask_empty_when_no_detections():
    assert not union_mask([], (4, 5)).any()


def test_draw_debug_tints_masks_and_draws_voted_roi():
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    rois = [
        Roi(slot=0, x0=5, y0=15, x1=25, y1=35),
        Roi(slot=1, x0=40, y0=15, x1=60, y1=35),
    ]
    voted = [
        VotedSlot(0, LABEL_RED, 1.0, 10, None, 12.0, 1.0),
        VotedSlot(1, LABEL_EMPTY, 1.0, 10, None, None, None),
    ]

    results = _results()
    out = draw_debug(rgb, rois, results, voted, inference_ms=124.0)
    base = draw_debug(rgb, rois, results, voted)

    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert not rgb.any()  # input untouched
    assert out[25, 15, 0] > 0 and out[25, 15, 1] < out[25, 15, 0]  # red tint
    assert tuple(out[25, 5]) == LABEL_COLORS_RGB[LABEL_RED]  # ROI 0 border
    assert tuple(out[25, 40]) == LABEL_COLORS_RGB[LABEL_EMPTY]  # ROI 1 border
    assert np.count_nonzero(out != base) > 0  # inference_ms text drawn


def test_draw_debug_draws_backend_label():
    rgb = np.zeros((60, 200, 3), dtype=np.uint8)
    plain = draw_debug(rgb, [], [], [], inference_ms=30.0)
    labelled = draw_debug(
        rgb, [], [], [], inference_ms=30.0, backend='onnx:cpu fallback'
    )
    assert np.count_nonzero(labelled != plain) > 0
