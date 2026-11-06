"""Tests for YOLO-seg slot assignment and HSV verification (no ultralytics)."""

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
from g1_perception.pipeline import clear_registry, get_registered, register
from g1_perception.segmenters import yolo_backend
from g1_perception.segmenters.yolo_backend import YoloBackend
from g1_perception.segmenters.yolo_seg import (
    YoloSegSegmenter,
    best_slot_for_mask,
    hsv_color_in_mask,
)

LABEL_MAP = {'czerwona': LABEL_RED, 'zielona': LABEL_GREEN}


@pytest.fixture(autouse=True)
def _register_yolo_seg():
    clear_registry()
    register('yolo_seg')(YoloSegSegmenter)
    yield
    clear_registry()


def _loose_hsv() -> HsvConfig:
    return HsvConfig(
        red_low=HsvRange(0, 10, 50, 255, 50, 255),
        red_high=HsvRange(170, 180, 50, 255, 50, 255),
        green=HsvRange(35, 95, 50, 255, 50, 255),
    )


def test_register_yolo_seg_name():
    assert get_registered('yolo_seg') is YoloSegSegmenter


def test_best_slot_requires_over_50_percent_overlap():
    rois = [
        Roi(slot=0, x0=0, y0=0, x1=10, y1=10),
        Roi(slot=1, x0=20, y0=0, x1=40, y1=20),
    ]
    mask = np.zeros((30, 50), dtype=bool)
    # All mass inside slot 1.
    mask[0:10, 20:26] = True
    assert best_slot_for_mask(mask, rois, min_overlap=0.5) == 1

    # 40% in slot 0, 60% outside → below 50% threshold.
    mask[:, :] = False
    mask[0:10, 0:4] = True       # 40 px in slot 0
    mask[0:10, 40:46] = True     # 60 px outside ROIs
    assert best_slot_for_mask(mask, rois, min_overlap=0.5) is None

    mask[:, :] = False
    mask[25:30, 0:20] = True  # entirely outside both ROIs
    assert best_slot_for_mask(mask, rois, min_overlap=0.5) is None


def test_hsv_color_in_mask_red():
    rgb = np.full((40, 40, 3), 128, dtype=np.uint8)
    rgb[10:30, 10:30] = (255, 0, 0)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    assert hsv_color_in_mask(rgb, mask, _loose_hsv()) == LABEL_RED


def test_verify_hsv_downgrades_mismatched_label(tmp_path):
    """Segment with a fake YOLO model: green class on a red square → uncertain."""
    rois = [Roi(slot=0, x0=5, y0=5, x1=35, y1=35)]
    h, w = 40, 40
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    rgb[10:30, 10:30] = (255, 0, 0)
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 10:30] = True

    class _Boxes:
        conf = np.array([0.9])
        cls = np.array([1.0])  # zielona

    class _Masks:
        data = mask[None, :, :].astype(np.float32)

    class _Result:
        masks = _Masks()
        boxes = _Boxes()
        names = {0: 'czerwona', 1: 'zielona'}

    class _FakeModel:
        names = {0: 'czerwona', 1: 'zielona'}

        def predict(self, _img, **_kwargs):
            return [_Result()]

    weights = tmp_path / 'dummy.pt'
    weights.write_bytes(b'x')
    seg = YoloSegSegmenter(
        model_path=weights,
        rois=rois,
        verify_hsv=True,
        hsv=_loose_hsv(),
        label_map=LABEL_MAP,
        min_overlap=0.5,
    )
    seg._backend = YoloBackend(_FakeModel(), weights, device=None)

    frame = Frame(
        rgb=rgb,
        depth_m=np.zeros((h, w), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=0.0,
    )
    dets = seg.segment(frame)
    assert len(dets) == 1
    assert dets[0].slot == 0
    assert dets[0].label == LABEL_UNCERTAIN
    assert dets[0].score == pytest.approx(0.9)


def test_segment_keeps_higher_confidence_per_slot(tmp_path):
    rois = [Roi(slot=0, x0=0, y0=0, x1=40, y1=40)]
    h, w = 40, 40
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[5:20, 5:20] = True
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[10:25, 10:25] = True

    class _Boxes:
        conf = np.array([0.4, 0.95])
        cls = np.array([0.0, 0.0])

    class _Masks:
        data = np.stack(
            [mask_a.astype(np.float32), mask_b.astype(np.float32)], axis=0
        )

    class _Result:
        masks = _Masks()
        boxes = _Boxes()
        names = {0: 'czerwona'}

    class _FakeModel:
        def predict(self, _img, **_kwargs):
            return [_Result()]

    weights = tmp_path / 'dummy.pt'
    weights.write_bytes(b'x')
    seg = YoloSegSegmenter(
        model_path=weights, rois=rois, verify_hsv=False, min_overlap=0.5
    )
    seg._backend = YoloBackend(_FakeModel(), weights, device=None)

    frame = Frame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=np.zeros((h, w), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=0.0,
    )
    dets = seg.segment(frame)
    assert len(dets) == 1
    assert dets[0].score == pytest.approx(0.95)
    assert np.array_equal(dets[0].mask, mask_b)


def test_segment_uses_fallback_device(tmp_path, monkeypatch):
    """Primary engine without CUDA → ONNX fallback; predict runs on CPU."""
    calls: list[dict] = []

    class _FakeYolo:
        def __init__(self, path: str) -> None:
            self.path = path

        def predict(self, _img, **kwargs):
            calls.append(kwargs)
            return []

    monkeypatch.setattr(yolo_backend, '_import_yolo', lambda: _FakeYolo)
    monkeypatch.setattr(yolo_backend, 'cuda_available', lambda: False)
    monkeypatch.setattr(yolo_backend, 'engine_backend_available', lambda: False)
    engine = tmp_path / 'cubes.engine'
    onnx = tmp_path / 'cubes.onnx'
    engine.write_bytes(b'x')
    onnx.write_bytes(b'x')
    seg = YoloSegSegmenter(
        model_path=engine,
        device='0',
        fallback_model_path=onnx,
        warmup_runs=1,
    )

    backend = seg.load()
    assert backend.fallback and backend.model_path == onnx
    assert len(calls) == 1 and calls[0]['retina_masks'] is True

    frame = Frame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.zeros((8, 8), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=0.0,
    )
    assert seg.segment(frame) == []
    assert [c['device'] for c in calls] == ['cpu', 'cpu']
