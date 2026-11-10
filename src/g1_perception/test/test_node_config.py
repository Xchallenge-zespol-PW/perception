"""Tests for flat node parameters → pipeline / voter configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1_perception.core.config import Roi
from g1_perception.core.types import LABEL_GREEN, LABEL_RED, Frame
from g1_perception.core.voting import SlotVoter
from g1_perception.core.cube_registry import CubeRegistry
from g1_perception.core.tracker import CubeTracker
from g1_perception.node_config import (
    DEFAULT_PARAMS,
    build_pipeline_config,
    registry_kwargs,
    tracker_kwargs,
    voter_kwargs,
)
from g1_perception.pipeline import Pipeline, clear_registry, register
from g1_perception.pose.centroid import CentroidPoseEstimator
from g1_perception.pose.geometric import GeometricPoseEstimator
from g1_perception.pose.pnp import PnpPoseEstimator
from g1_perception.segmenters.hsv import HsvSegmenter
from g1_perception.segmenters.yolo_seg import YoloSegSegmenter


@pytest.fixture(autouse=True)
def _registry():
    clear_registry()
    register('hsv')(HsvSegmenter)
    register('yolo_seg')(YoloSegSegmenter)
    register('centroid')(CentroidPoseEstimator)
    register('geometric')(GeometricPoseEstimator)
    register('pnp')(PnpPoseEstimator)
    yield
    clear_registry()


def _params(**overrides):
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    return params


def test_yolo_geometric_config_from_defaults():
    config = build_pipeline_config(_params(device='0', verify_hsv=True))

    assert config['segmenter'] == {
        'name': 'yolo_seg',
        'params': {
            'model_path': DEFAULT_PARAMS['model_path'],
            'device': '0',
            'fallback_device': 'cpu',
            'warmup_runs': 2,
            'conf': 0.25,
            'verify_hsv': True,
        },
    }
    assert config['pose_estimator'] == {
        'name': 'geometric',
        'params': {
            'size': 0.05, 'margin': 20, 'z_min': 0.1, 'z_max': 2.0,
            'min_height': 0.01, 'min_above_ratio': 0.5,
            'footprint_min': 0.02, 'footprint_max': 0.13,
        },
    }
    assert 'rois_path' not in config and 'labels_path' not in config


def test_only_params_accepted_by_stage_are_passed():
    config = build_pipeline_config(
        _params(segmenter='hsv', pose_estimator='centroid', hsv_path='h.yaml')
    )

    assert config['segmenter']['params'] == {'hsv_path': 'h.yaml'}
    assert 'size' not in config['pose_estimator']['params']


def test_non_positive_depth_limits_are_disabled():
    params = build_pipeline_config(_params(z_min_m=0.0, z_max_m=-1.0))[
        'pose_estimator'
    ]['params']

    assert 'z_min' not in params and 'z_max' not in params


def test_paths_forwarded_when_set():
    config = build_pipeline_config(
        _params(rois_path=' r.yaml ', labels_path='l.yaml')
    )

    assert config['rois_path'] == 'r.yaml'
    assert config['labels_path'] == 'l.yaml'


def test_fallback_model_forwarded_when_set():
    config = build_pipeline_config(_params(
        model_path='/ws/models/cubes_seg.engine',
        device='0',
        fallback_model_path='/ws/models/cubes_seg.onnx',
        warmup_runs=0,
    ))
    params = config['segmenter']['params']
    assert params['fallback_model_path'] == '/ws/models/cubes_seg.onnx'
    assert params['fallback_device'] == 'cpu'
    assert params['warmup_runs'] == 0


def test_pnp_params_forwarded_to_pnp_only():
    config = build_pipeline_config(_params(
        pose_estimator='pnp', pnp_fallback='none', pnp_max_reproj_px=3.0
    ))
    params = config['pose_estimator']['params']
    assert params['fallback'] == 'none'
    assert params['max_reproj_px'] == 3.0
    assert params['kpt_min_conf'] == 0.5
    assert params['depth_diff_ref_m'] == 0.6 and params['size'] == 0.05


def test_voter_kwargs_min_frames_zero_means_window():
    assert voter_kwargs(_params(vote_window=7, vote_min_frames=0)) == {
        'window': 7,
        'min_ratio': 0.7,
        'min_frames': None,
        'window_s': None,
        'min_votes': 1,
    }
    assert voter_kwargs(_params(vote_min_frames=4))['min_frames'] == 4
    SlotVoter(**voter_kwargs(_params()))


def test_time_window_params_reach_voter_and_tracker():
    params = _params(
        vote_window_s=2.0, vote_min_votes=5, vote_window=100, max_missed_s=1.0
    )
    vk = voter_kwargs(params)
    assert vk['window_s'] == 2.0 and vk['min_votes'] == 5
    SlotVoter(**vk)
    tk = tracker_kwargs(params)
    assert tk['vote_window_s'] == 2.0 and tk['max_missed_s'] == 1.0
    CubeTracker(**tk)


def test_registry_kwargs_use_the_slot_axis():
    kwargs = registry_kwargs(
        _params(slot_axis='x', cube_name_prefix='cube_', latch_s=0.5)
    )

    assert kwargs == {
        'axis': 'x',
        'name_prefix': 'cube_',
        'latch_s': 0.5,
        'reid_gate_m': 0.025,
    }
    CubeRegistry(**kwargs)


def test_built_config_runs_hsv_pipeline(tmp_path: Path):
    rois_path = tmp_path / 'rois.yaml'
    rois_path.write_text(
        'rois:\n'
        '- {slot: 0, x0: 0, y0: 0, x1: 10, y1: 10}\n'
        '- {slot: 1, x0: 10, y0: 0, x1: 20, y1: 10}\n'
        '- {slot: 2, x0: 20, y0: 0, x1: 30, y1: 10}\n',
        encoding='utf-8',
    )
    hsv_path = tmp_path / 'hsv.yaml'
    hsv_path.write_text(
        'red_low:  {h_min: 0, h_max: 10, s_min: 50, s_max: 255, v_min: 50, v_max: 255}\n'
        'red_high: {h_min: 170, h_max: 180, s_min: 50, s_max: 255, v_min: 50, v_max: 255}\n'
        'green:    {h_min: 35, h_max: 95, s_min: 50, s_max: 255, v_min: 50, v_max: 255}\n',
        encoding='utf-8',
    )
    config = build_pipeline_config(
        _params(
            segmenter='hsv',
            pose_estimator='centroid',
            rois_path=str(rois_path),
            hsv_path=str(hsv_path),
        )
    )
    pipeline = Pipeline.from_config(config)
    rgb = np.full((10, 30, 3), 128, dtype=np.uint8)
    rgb[:, 0:10] = (255, 0, 0)
    rgb[:, 20:30] = (0, 255, 0)
    frame = Frame(
        rgb=rgb,
        depth_m=np.zeros((10, 30), dtype=np.float32),
        K=np.eye(3),
        stamp=0.0,
    )

    labels = [r.label for r in pipeline.process(frame)]

    assert pipeline.rois == [
        Roi(0, 0, 0, 10, 10), Roi(1, 10, 0, 20, 10), Roi(2, 20, 0, 30, 10)
    ]
    assert labels == [LABEL_RED, 'empty', LABEL_GREEN]
