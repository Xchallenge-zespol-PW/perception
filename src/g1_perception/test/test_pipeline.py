"""Tests for pipeline registry, config construction, and per-slot output."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.config import Roi
from g1_perception.core.types import (
    LABEL_EMPTY,
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Detection,
    Frame,
    Pose,
    SlotResult,
)
from g1_perception.pipeline import (
    Pipeline,
    clear_registry,
    get_registered,
    register,
)
from g1_perception.segmenters.yolo_seg import YoloSegSegmenter

LABEL_MAP = {
    'red cube': LABEL_RED,
    'green cube': LABEL_GREEN,
    'czerwona': LABEL_RED,
    'zielony': LABEL_GREEN,
}

ROIS_4 = [
    Roi(slot=0, x0=0, y0=0, x1=10, y1=10),
    Roi(slot=1, x0=10, y0=0, x1=20, y1=10),
    Roi(slot=2, x0=20, y0=0, x1=30, y1=10),
    Roi(slot=3, x0=30, y0=0, x1=40, y1=10),
]


class FixedSegmenter:
    """Returns ``(slot, label, score)`` triples as detections."""

    def __init__(self, items: list[tuple[int | None, str, float]]) -> None:
        self.items = items

    def segment(self, frame: Frame) -> list[Detection]:
        h, w = frame.rgb.shape[:2]
        return [
            Detection(
                slot=slot,
                label=label,
                score=score,
                mask=np.zeros((h, w), dtype=bool),
            )
            for slot, label, score in self.items
        ]


class RecordingPoseEstimator:
    """Returns a pose encoding the slot, or None when ``fail`` is true."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[int | None] = []

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        self.calls.append(det.slot)
        if self.fail:
            return None
        return Pose(
            position=np.array([0.0, 0.0, float(det.slot)], dtype=np.float64),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            score=0.8,
        )


class RoiAwareSegmenter:
    """Accepts ``rois`` / ``label_map`` so the pipeline can inject them."""

    def __init__(
        self,
        rois: list[Roi] | None = None,
        rois_path: str | None = None,
        label_map: dict[str, str] | None = None,
    ) -> None:
        self.rois = rois
        self.rois_path = rois_path
        self.label_map = label_map

    def segment(self, frame: Frame) -> list[Detection]:
        return []


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear_registry()
    register('fixed')(FixedSegmenter)
    register('recording_pose')(RecordingPoseEstimator)
    register('roi_aware')(RoiAwareSegmenter)
    register('yolo_seg')(YoloSegSegmenter)
    yield
    clear_registry()


def _make_frame(h: int = 10, w: int = 40) -> Frame:
    return Frame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=np.zeros((h, w), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=1.5,
    )


def _pipeline(items, *, rois=ROIS_4, fail=False) -> Pipeline:
    return Pipeline.from_config(
        {
            'segmenter': {'name': 'fixed', 'params': {'items': items}},
            'pose_estimator': {
                'name': 'recording_pose',
                'params': {'fail': fail},
            },
        },
        rois=rois,
        label_map=LABEL_MAP,
    )


def test_register_and_lookup():
    assert get_registered('fixed') is FixedSegmenter
    assert get_registered('recording_pose') is RecordingPoseEstimator


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match='already registered'):
        @register('fixed')
        class _Other:
            pass


def test_four_rois_two_yolo_detections_give_two_colours_two_empty(tmp_path):
    h, w = 10, 40
    mask_red = np.zeros((h, w), dtype=bool)
    mask_red[2:8, 2:8] = True          # slot 0
    mask_green = np.zeros((h, w), dtype=bool)
    mask_green[2:8, 22:28] = True      # slot 2

    class _Boxes:
        conf = np.array([0.9, 0.8])
        cls = np.array([1.0, 0.0])

    class _Masks:
        data = np.stack(
            [mask_red.astype(np.float32), mask_green.astype(np.float32)]
        )

    class _Result:
        masks = _Masks()
        boxes = _Boxes()
        names = {0: 'green cube', 1: 'red cube'}

    class _FakeModel:
        def predict(self, _img, **_kwargs):
            return [_Result()]

    weights = tmp_path / 'dummy.pt'
    weights.write_bytes(b'x')
    pipeline = Pipeline.from_config(
        {
            'segmenter': {
                'name': 'yolo_seg',
                'params': {'model_path': str(weights)},
            },
            'pose_estimator': {'name': 'recording_pose'},
        },
        rois=ROIS_4,
        label_map=LABEL_MAP,
    )
    pipeline._segmenter._model = _FakeModel()

    results = pipeline.process(_make_frame(h, w))

    assert all(isinstance(r, SlotResult) for r in results)
    assert [r.slot for r in results] == [0, 1, 2, 3]
    assert [r.label for r in results] == [
        LABEL_RED,
        LABEL_EMPTY,
        LABEL_GREEN,
        LABEL_EMPTY,
    ]
    assert results[0].score == pytest.approx(0.9)
    assert results[2].score == pytest.approx(0.8)
    for r in (results[0], results[2]):
        assert r.detection is not None
        assert r.pose is not None
        assert r.pose.position[2] == pytest.approx(float(r.slot))
    for r in (results[1], results[3]):
        assert r.detection is None
        assert r.pose is None
        assert r.score == 0.0


def test_unmapped_label_is_uncertain_without_pose():
    pipeline = _pipeline([(1, 'blue cube', 0.7)])

    results = pipeline.process(_make_frame())

    assert [r.label for r in results] == [
        LABEL_EMPTY,
        LABEL_UNCERTAIN,
        LABEL_EMPTY,
        LABEL_EMPTY,
    ]
    assert results[1].detection is not None
    assert results[1].pose is None
    assert results[1].score == pytest.approx(0.7)
    assert pipeline._pose_estimator.calls == []


def test_detections_outside_rois_are_ignored():
    pipeline = _pipeline([(None, 'red cube', 0.9), (7, 'czerwona', 0.9)])

    results = pipeline.process(_make_frame())

    assert [r.label for r in results] == [LABEL_EMPTY] * 4


def test_highest_score_detection_wins_per_slot():
    pipeline = _pipeline([(0, 'red cube', 0.4), (0, 'zielony', 0.95)])

    result = pipeline.process(_make_frame())[0]

    assert result.label == LABEL_GREEN
    assert result.score == pytest.approx(0.95)


def test_failed_pose_keeps_colour_label():
    pipeline = _pipeline([(3, 'czerwona', 0.6)], fail=True)

    result = pipeline.process(_make_frame())[3]

    assert result.label == LABEL_RED
    assert result.pose is None


def test_results_sorted_by_slot_regardless_of_roi_order():
    pipeline = _pipeline([], rois=list(reversed(ROIS_4)))

    assert [r.slot for r in pipeline.process(_make_frame())] == [0, 1, 2, 3]


def test_from_config_injects_rois_and_label_map():
    pipeline = Pipeline.from_config(
        {
            'segmenter': {'name': 'roi_aware'},
            'pose_estimator': {'name': 'recording_pose'},
        },
        rois=ROIS_4,
        label_map=LABEL_MAP,
    )

    seg = pipeline._segmenter
    assert seg.rois == ROIS_4
    assert seg.label_map == LABEL_MAP


def test_from_config_does_not_override_explicit_path_param():
    pipeline = Pipeline.from_config(
        {
            'segmenter': {
                'name': 'roi_aware',
                'params': {'rois_path': 'custom.yaml'},
            },
            'pose_estimator': {'name': 'recording_pose'},
        },
        rois=ROIS_4,
        label_map=LABEL_MAP,
    )

    assert pipeline._segmenter.rois is None
    assert pipeline._segmenter.rois_path == 'custom.yaml'


def test_from_config_reads_rois_and_labels_paths(tmp_path):
    rois_path = tmp_path / 'rois.yaml'
    rois_path.write_text(
        'rois:\n- {slot: 0, x0: 0, y0: 0, x1: 5, y1: 5}\n',
        encoding='utf-8',
    )
    labels_path = tmp_path / 'labels.yaml'
    labels_path.write_text('label_map:\n  kostka: red\n', encoding='utf-8')

    pipeline = Pipeline.from_config(
        {
            'rois_path': str(rois_path),
            'labels_path': str(labels_path),
            'segmenter': {
                'name': 'fixed',
                'params': {'items': [(0, 'Kostka', 0.5)]},
            },
            'pose_estimator': {'name': 'recording_pose'},
        }
    )

    results = pipeline.process(_make_frame())

    assert [(r.slot, r.label) for r in results] == [(0, LABEL_RED)]


def test_from_config_unknown_name_raises():
    with pytest.raises(KeyError, match='Unknown implementation'):
        Pipeline.from_config(
            {
                'segmenter': {'name': 'missing'},
                'pose_estimator': {'name': 'recording_pose'},
            },
            rois=ROIS_4,
            label_map=LABEL_MAP,
        )


def test_from_config_missing_keys_raise():
    with pytest.raises(KeyError, match='segmenter'):
        Pipeline.from_config(
            {'pose_estimator': {'name': 'recording_pose'}},
            rois=ROIS_4,
            label_map=LABEL_MAP,
        )

    with pytest.raises(KeyError, match=r"pose_estimator\.name"):
        Pipeline.from_config(
            {
                'segmenter': {'name': 'fixed', 'params': {'items': []}},
                'pose_estimator': {},
            },
            rois=ROIS_4,
            label_map=LABEL_MAP,
        )
