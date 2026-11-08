"""Translate flat node parameters into pipeline / voter configuration."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import g1_perception.pose  # noqa: F401  (registers pose estimators)
import g1_perception.segmenters  # noqa: F401  (registers segmenters)
from g1_perception.pipeline import ROIS_MODE_IMAGE, get_registered

DEFAULT_PARAMS: dict[str, Any] = {
    'color_topic': '/camera/camera/color/image_raw',
    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
    'camera_info_topic': '/camera/camera/color/camera_info',
    'output_topic': '/perception/cubes',
    'debug_image_topic': '/perception/debug_image',
    'mask_topic': '/perception/mask',
    'target_frame': 'camera_link',
    'tf_timeout_s': 0.1,
    'segmenter': 'yolo_seg',
    'pose_estimator': 'geometric',
    'model_path': 'runs/segment/runs/seg/cubes-2/weights/best.pt',
    'device': '',
    'fallback_model_path': '',
    'fallback_device': 'cpu',
    'warmup_runs': 2,
    'conf': 0.25,
    'verify_hsv': False,
    'rois_mode': ROIS_MODE_IMAGE,
    'rois_path': '',
    'hsv_path': '',
    'labels_path': '',
    'cube_size_m': 0.05,
    'margin_px': 20,
    'z_min_m': 0.1,
    'z_max_m': 2.0,
    'kpt_min_conf': 0.5,
    'pnp_max_reproj_px': 2.0,
    'pnp_max_depth_diff_m': 0.03,
    'pnp_depth_diff_ref_m': 0.6,
    'pnp_max_normal_deg': 20.0,
    'pnp_fallback': 'geometric',
    'vote_window': 10,
    'vote_min_ratio': 0.7,
    'vote_min_frames': 0,
    'vote_window_s': 0.0,
    'vote_min_votes': 1,
    'max_missed_s': 0.0,
    'match_dist_m': 0.03,
    'max_missed': 5,
    'slot_axis': 'y',
    'expected_count': 4,
    'max_line_deviation_m': 0.03,
    'min_spacing_m': 0.04,
    'max_spacing_m': 0.20,
    'publish_rate_hz': 5.0,
    'sync_queue_size': 10,
    'sync_slop_s': 0.05,
}


def build_pipeline_config(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build a :meth:`Pipeline.from_config` dict from flat node parameters.

    Empty strings mean "use the default" and are omitted; non-positive
    ``z_min_m`` / ``z_max_m`` disable that depth limit. Only parameters the
    selected stage constructor accepts are passed to it.
    """
    seg_name = str(params['segmenter'])
    pose_name = str(params['pose_estimator'])

    seg_candidates = {
        'model_path': _opt_str(params.get('model_path')),
        'device': _opt_str(params.get('device')),
        'fallback_model_path': _opt_str(params.get('fallback_model_path')),
        'fallback_device': _opt_str(params.get('fallback_device')),
        'warmup_runs': int(params.get('warmup_runs', 0)),
        'conf': float(params['conf']),
        'verify_hsv': bool(params['verify_hsv']),
        'hsv_path': _opt_str(params.get('hsv_path')),
    }
    pose_candidates = {
        'size': float(params['cube_size_m']),
        'margin': int(params['margin_px']),
        'z_min': _opt_positive(params.get('z_min_m')),
        'z_max': _opt_positive(params.get('z_max_m')),
        'kpt_min_conf': float(params['kpt_min_conf']),
        'max_reproj_px': float(params['pnp_max_reproj_px']),
        'max_depth_diff_m': float(params['pnp_max_depth_diff_m']),
        'depth_diff_ref_m': float(params['pnp_depth_diff_ref_m']),
        'max_normal_deg': float(params['pnp_max_normal_deg']),
        'fallback': str(params['pnp_fallback']),
    }

    config: dict[str, Any] = {
        'rois_mode': str(params.get('rois_mode', ROIS_MODE_IMAGE)),
        'segmenter': {
            'name': seg_name,
            'params': stage_params(get_registered(seg_name), seg_candidates),
        },
        'pose_estimator': {
            'name': pose_name,
            'params': stage_params(get_registered(pose_name), pose_candidates),
        },
    }
    for key in ('rois_path', 'labels_path'):
        value = _opt_str(params.get(key))
        if value is not None:
            config[key] = value
    return config


def voter_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return :class:`SlotVoter` kwargs.

    ``vote_window_s > 0`` selects the time window (``vote_min_votes`` votes,
    ``vote_window`` caps frames); otherwise the frame window with
    ``vote_min_frames`` (``<= 0`` = whole window).
    """
    min_frames = int(params['vote_min_frames'])
    window_s = float(params.get('vote_window_s', 0.0))
    return {
        'window': int(params['vote_window']),
        'min_ratio': float(params['vote_min_ratio']),
        'min_frames': min_frames if min_frames > 0 else None,
        'window_s': window_s if window_s > 0.0 else None,
        'min_votes': int(params.get('vote_min_votes', 1)),
    }


def tracker_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return :class:`CubeTracker` kwargs from flat node parameters."""
    vk = voter_kwargs(params)
    max_missed_s = float(params.get('max_missed_s', 0.0))
    return {
        'match_dist_m': float(params['match_dist_m']),
        'max_missed': int(params['max_missed']),
        'max_missed_s': max_missed_s if max_missed_s > 0.0 else None,
        'vote_window': vk['window'],
        'vote_min_ratio': vk['min_ratio'],
        'vote_min_frames': vk['min_frames'],
        'vote_window_s': vk['window_s'],
        'vote_min_votes': vk['min_votes'],
    }


def slot_assign_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return :func:`assign_slots` kwargs from flat node parameters."""
    return {
        'axis': str(params['slot_axis']),
        'expected_count': int(params['expected_count']),
        'max_line_deviation_m': float(params['max_line_deviation_m']),
        'min_spacing_m': float(params['min_spacing_m']),
        'max_spacing_m': float(params['max_spacing_m']),
    }


def stage_params(cls: type, candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Keep candidates accepted by ``cls.__init__`` whose value is not None."""
    accepted = inspect.signature(cls).parameters
    return {
        k: v for k, v in candidates.items()
        if k in accepted and v is not None
    }


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_positive(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if number > 0.0 else None
