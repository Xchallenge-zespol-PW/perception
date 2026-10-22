"""Core data types and stage protocols for the perception pipeline."""

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
    'Pose',
    'PoseEstimator',
    'Segmenter',
    'load_frame',
]
