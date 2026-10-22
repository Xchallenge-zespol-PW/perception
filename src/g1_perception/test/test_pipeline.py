"""Tests for pipeline registry, config construction, and data flow."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.types import Detection, Frame, Pose
from g1_perception.pipeline import (
    Pipeline,
    clear_registry,
    get_registered,
    register,
)


class DummySegmenter:
    """Returns a fixed number of synthetic detections."""

    def __init__(self, label: str = 'red', count: int = 1) -> None:
        self.label = label
        self.count = count

    def segment(self, frame: Frame) -> list[Detection]:
        h, w = frame.rgb.shape[:2]
        return [
            Detection(
                slot=i,
                label=self.label,
                score=0.9,
                mask=np.zeros((h, w), dtype=bool),
            )
            for i in range(self.count)
        ]


class DummyPoseEstimator:
    """Returns a fixed pose, or ``None`` when ``fail`` is true."""

    def __init__(self, fail: bool = False, score: float = 0.8) -> None:
        self.fail = fail
        self.score = score

    def estimate(self, frame: Frame, det: Detection) -> Pose | None:
        if self.fail:
            return None
        return Pose(
            position=np.array([0.0, 0.0, float(det.slot or 0)], dtype=np.float64),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            score=self.score,
        )


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Register dummies in a clean registry for each test."""
    clear_registry()
    register('dummy_segmenter')(DummySegmenter)
    register('dummy_pose')(DummyPoseEstimator)
    yield
    clear_registry()


def _make_frame(h: int = 4, w: int = 4) -> Frame:
    return Frame(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=np.zeros((h, w), dtype=np.float32),
        K=np.eye(3, dtype=np.float64),
        stamp=1.5,
    )


def test_register_and_lookup():
    assert get_registered('dummy_segmenter') is DummySegmenter
    assert get_registered('dummy_pose') is DummyPoseEstimator


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match='already registered'):
        @register('dummy_segmenter')
        class _Other:
            pass


def test_from_config_builds_pipeline_with_params():
    pipeline = Pipeline.from_config(
        {
            'segmenter': {
                'name': 'dummy_segmenter',
                'params': {'label': 'green', 'count': 2},
            },
            'pose_estimator': {
                'name': 'dummy_pose',
                'params': {'score': 0.5},
            },
        }
    )

    results = pipeline.process(_make_frame())

    assert len(results) == 2
    for i, (det, pose) in enumerate(results):
        assert det.slot == i
        assert det.label == 'green'
        assert pose is not None
        assert pose.score == 0.5
        assert np.allclose(pose.position, [0.0, 0.0, float(i)])
        assert np.allclose(pose.quat, [0.0, 0.0, 0.0, 1.0])


def test_process_passes_frame_and_allows_none_pose():
    pipeline = Pipeline.from_config(
        {
            'segmenter': {'name': 'dummy_segmenter', 'params': {'count': 1}},
            'pose_estimator': {'name': 'dummy_pose', 'params': {'fail': True}},
        }
    )

    results = pipeline.process(_make_frame())

    assert len(results) == 1
    det, pose = results[0]
    assert isinstance(det, Detection)
    assert pose is None


def test_from_config_unknown_name_raises():
    with pytest.raises(KeyError, match='Unknown implementation'):
        Pipeline.from_config(
            {
                'segmenter': {'name': 'missing'},
                'pose_estimator': {'name': 'dummy_pose'},
            }
        )


def test_from_config_missing_keys_raise():
    with pytest.raises(KeyError, match='segmenter'):
        Pipeline.from_config({'pose_estimator': {'name': 'dummy_pose'}})

    with pytest.raises(KeyError, match=r"pose_estimator\.name"):
        Pipeline.from_config(
            {
                'segmenter': {'name': 'dummy_segmenter'},
                'pose_estimator': {},
            }
        )
