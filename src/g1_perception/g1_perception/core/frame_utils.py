"""Convert raw camera arrays into :class:`Frame` inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

DEPTH_MM_ENCODINGS = ('16UC1', 'mono16')
DEPTH_M_ENCODINGS = ('32FC1',)


def depth_to_m(depth: np.ndarray, encoding: str) -> NDArray[np.float32]:
    """
    Convert a depth image to float32 metres (0 = invalid).

    ``16UC1`` / ``mono16`` are millimetres (RealSense aligned depth);
    ``32FC1`` is already metres, with NaN / inf mapped to 0.
    """
    if encoding in DEPTH_MM_ENCODINGS:
        return np.asarray(depth, dtype=np.float32) / np.float32(1000.0)
    if encoding in DEPTH_M_ENCODINGS:
        depth_m = np.asarray(depth, dtype=np.float32).copy()
        depth_m[~np.isfinite(depth_m)] = 0.0
        return depth_m
    raise ValueError(f'Unsupported depth encoding: {encoding!r}')


def intrinsics_from_k(k: Sequence[float]) -> NDArray[np.float64]:
    """Return the 3x3 intrinsics matrix from a row-major 9-element ``K``."""
    K = np.asarray(k, dtype=np.float64)
    if K.size != 9:
        raise ValueError(f'K must have 9 elements, got {K.size}')
    return K.reshape(3, 3)
