"""Canonical order of the 4 top-face cube corners on the image."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Two corners closer than this in ``v`` count as a tie for ``c0``.
C0_TIE_PX = 3.0
N_CORNERS = 4


def order_top_corners(
    corners: ArrayLike,
    tie_px: float = C0_TIE_PX,
) -> NDArray[np.float64]:
    """
    Return the 4 corners as ``c0..c3`` (clockwise, ``c0`` = lowest; see ``docs/training.md``).

    The corners are sorted visually clockwise on the image (``v`` down),
    starting at ``c0`` = the corner with the largest ``v``. When another
    corner is within ``tie_px`` of that ``v``, the one with the larger
    ``u`` is ``c0``. Input order does not matter; output is ``(4, 2)``.
    """
    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (N_CORNERS, 2):
        raise ValueError(f'Expected corners of shape (4, 2), got {pts.shape}')

    center = pts.mean(axis=0)
    # With v pointing down, increasing atan2(dv, du) is visually clockwise.
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ring = pts[np.argsort(angles, kind='stable')]

    v_max = float(ring[:, 1].max())
    near_bottom = np.flatnonzero(ring[:, 1] >= v_max - float(tie_px))
    start = int(near_bottom[np.argmax(ring[near_bottom, 0])])
    return np.roll(ring, -start, axis=0)


def is_clockwise(corners: ArrayLike) -> bool:
    """Return True when ``corners`` run visually clockwise (``v`` down)."""
    pts = np.asarray(corners, dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    area2 = float(np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]))
    return area2 > 0.0
