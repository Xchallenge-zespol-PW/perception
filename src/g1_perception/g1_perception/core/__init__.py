"""Core data types and stage protocols for the perception pipeline."""

from g1_perception.core.config import (
    HsvConfig,
    HsvRange,
    Roi,
    load_hsv,
    load_label_map,
    load_rois,
)
from g1_perception.core.dataset import load_frame
from g1_perception.core.labels import canonical_label
from g1_perception.core.types import (
    CANONICAL_LABELS,
    COLOR_LABELS,
    LABEL_EMPTY,
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Detection,
    Frame,
    Pose,
    PoseEstimator,
    Segmenter,
    SlotResult,
)

__all__ = [
    'CANONICAL_LABELS',
    'COLOR_LABELS',
    'Detection',
    'Frame',
    'HsvConfig',
    'HsvRange',
    'LABEL_EMPTY',
    'LABEL_GREEN',
    'LABEL_RED',
    'LABEL_UNCERTAIN',
    'Pose',
    'PoseEstimator',
    'Roi',
    'Segmenter',
    'SlotResult',
    'canonical_label',
    'load_frame',
    'load_hsv',
    'load_label_map',
    'load_rois',
]
