"""Core data types and stage protocols for the perception pipeline."""

from g1_perception.core.config import (
    HsvConfig,
    HsvRange,
    Roi,
    load_hsv,
    load_rois,
)
from g1_perception.core.dataset import load_frame
from g1_perception.core.types import (
    Detection,
    Frame,
    Pose,
    PoseEstimator,
    Segmenter,
)

__all__ = [
    'Detection',
    'Frame',
    'HsvConfig',
    'HsvRange',
    'Pose',
    'PoseEstimator',
    'Roi',
    'Segmenter',
    'load_frame',
    'load_hsv',
    'load_rois',
]
