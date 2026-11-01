"""Tests for the HSV ROI segmenter on synthetic images."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.config import HsvConfig, HsvRange, Roi
from g1_perception.core.types import (
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Frame,
)
from g1_perception.pipeline import clear_registry, get_registered
from g1_perception.segmenters.hsv import HsvSegmenter


@pytest.fixture(autouse=True)
def _ensure_hsv_registered():
    """Re-register after other tests that call ``clear_registry``."""
    clear_registry()
    # Re-apply decorator side effect by constructing registration explicitly.
    from g1_perception.pipeline import register

    register('hsv')(HsvSegmenter)
    yield
    clear_registry()


def _loose_hsv() -> HsvConfig:
    """Wide thresholds that match pure RGB red/green test squares."""
    return HsvConfig(
        red_low=HsvRange(0, 10, 50, 255, 50, 255),
        red_high=HsvRange(170, 180, 50, 255, 50, 255),
        green=HsvRange(35, 95, 50, 255, 50, 255),
    )


def _frame_with_square(
    color_rgb: tuple[int, int, int],
    *,
    square: tuple[int, int, int, int],
    size: tuple[int, int] = (80, 100),
) -> Frame:
    """Build a gray Frame with one solid RGB square ``(y0,x0,y1,x1)``."""
    h, w = size
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    y0, x0, y1, x1 = square
    rgb[y0:y1, x0:x1] = color_rgb
    return Frame(
        rgb=rgb,
        depth_m=np.zeros((h, w), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=0.0,
    )


def test_register_hsv_name():
    assert get_registered('hsv') is HsvSegmenter


def test_red_square_in_roi():
    roi = Roi(slot=0, x0=20, y0=20, x1=50, y1=50)
    frame = _frame_with_square((255, 0, 0), square=(25, 25, 45, 45))
    seg = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.1, erode=0
    )

    dets = seg.segment(frame)

    assert len(dets) == 1
    det = dets[0]
    assert det.slot == 0
    assert det.label == LABEL_RED
    assert det.score >= 0.1
    assert det.mask.shape == frame.rgb.shape[:2]
    assert det.mask.any()
    # Mask stays inside ROI.
    ys, xs = np.where(det.mask)
    assert ys.min() >= roi.y0 and ys.max() < roi.y1
    assert xs.min() >= roi.x0 and xs.max() < roi.x1


def test_green_square_in_roi():
    roi = Roi(slot=2, x0=10, y0=10, x1=40, y1=40)
    frame = _frame_with_square((0, 255, 0), square=(15, 15, 35, 35))
    seg = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.1, erode=0
    )

    det = seg.segment(frame)[0]

    assert det.slot == 2
    assert det.label == LABEL_GREEN
    assert det.score >= 0.1
    assert det.mask.any()


def test_empty_roi_yields_no_detection():
    roi = Roi(slot=1, x0=10, y0=10, x1=40, y1=40)
    # Gray image only — no saturated colour inside ROI.
    frame = _frame_with_square((128, 128, 128), square=(0, 0, 0, 0))
    seg = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.1, erode=0
    )

    assert seg.segment(frame) == []


def test_below_min_frac_yields_no_detection():
    roi = Roi(slot=0, x0=0, y0=0, x1=50, y1=50)
    # Tiny red speck inside a large ROI → fraction below threshold.
    frame = _frame_with_square((255, 0, 0), square=(20, 20, 22, 22))
    seg = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.5, erode=0
    )

    assert seg.segment(frame) == []


def test_roi_outside_image_is_uncertain():
    roi = Roi(slot=0, x0=500, y0=500, x1=520, y1=520)
    frame = _frame_with_square((255, 0, 0), square=(0, 0, 10, 10))
    seg = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.1, erode=0
    )

    det = seg.segment(frame)[0]

    assert det.label == LABEL_UNCERTAIN
    assert det.score == 0.0
    assert not det.mask.any()


def test_erode_shrinks_roi_area():
    roi = Roi(slot=0, x0=10, y0=10, x1=30, y1=30)
    # Red fills the entire ROI; after erode the counted area is smaller
    # but still classified as red when erode is mild.
    frame = _frame_with_square((255, 0, 0), square=(10, 10, 30, 30))
    seg0 = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.1, erode=0
    )
    seg1 = HsvSegmenter(
        rois=[roi], hsv=_loose_hsv(), min_frac=0.1, erode=2
    )

    d0 = seg0.segment(frame)[0]
    d1 = seg1.segment(frame)[0]

    assert d0.label == LABEL_RED
    assert d1.label == LABEL_RED
    assert int(d1.mask.sum()) <= int(d0.mask.sum())


def test_multiple_rois():
    rois = [
        Roi(slot=0, x0=5, y0=5, x1=25, y1=25),
        Roi(slot=1, x0=40, y0=5, x1=60, y1=25),
        Roi(slot=2, x0=70, y0=5, x1=90, y1=25),
    ]
    h, w = 40, 100
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    rgb[8:22, 8:22] = (255, 0, 0)      # red in slot 0
    rgb[8:22, 43:57] = (0, 255, 0)     # green in slot 1
    # slot 2 left gray
    frame = Frame(
        rgb=rgb,
        depth_m=np.zeros((h, w), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=0.0,
    )
    seg = HsvSegmenter(
        rois=rois, hsv=_loose_hsv(), min_frac=0.1, erode=0
    )

    dets = seg.segment(frame)

    assert [d.label for d in dets] == [LABEL_RED, LABEL_GREEN]
    assert [d.slot for d in dets] == [0, 1]
