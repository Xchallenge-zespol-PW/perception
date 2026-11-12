"""Is a cube seen well enough to update its pose? (no ROS)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from g1_perception.core.geometry import quat_to_rotation_matrix
from g1_perception.core.types import Frame, SlotResult


def mask_touches_border(mask: NDArray[np.bool_], margin_px: int) -> bool:
    """Return True when ``mask`` reaches within ``margin_px`` of the image edge."""
    if margin_px <= 0:
        return False
    m = int(margin_px)
    return bool(
        mask[:m].any() or mask[-m:].any()
        or mask[:, :m].any() or mask[:, -m:].any()
    )


def visible_ratio(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
    size: float,
) -> float:
    """
    Return the mask area over the smallest silhouette of a whole cube.

    The mask's pixel count is scaled to m² at its median valid depth and
    divided by ``size²``: a cube seen from any side covers at least one
    face, so a whole cube gives about 1 or more, a half hidden one about
    0.5. NaN when the mask has no valid depth.
    """
    n_px = int(np.count_nonzero(mask))
    z = depth_m[mask]
    z = z[z > 0.0]
    if n_px == 0 or z.size == 0:
        return float('nan')
    z_med = float(np.median(z))
    area_m2 = n_px * z_med * z_med / (float(K[0, 0]) * float(K[1, 1]))
    return area_m2 / (float(size) ** 2)


def angular_speed_deg_s(
    quat_a: NDArray[np.float64],
    stamp_a: float,
    quat_b: NDArray[np.float64],
    stamp_b: float,
) -> float:
    """Return the rotation speed (deg/s) between two orientations; 0 if dt <= 0."""
    dt = float(stamp_b) - float(stamp_a)
    if dt <= 0.0:
        return 0.0
    R = quat_to_rotation_matrix(np.asarray(quat_a, float)).T @ quat_to_rotation_matrix(
        np.asarray(quat_b, float)
    )
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos))) / dt


def mark_hidden(
    results: list[SlotResult],
    frame: Frame,
    *,
    size: float,
    border_px: int = 0,
    min_visible_ratio: float = 0.0,
) -> tuple[list[SlotResult], int]:
    """
    Mark posed results of cut-off or hidden cubes as ``pose_trusted=False``.

    A result is untrusted when its mask touches the image border
    (``border_px``, 0 = off) or its :func:`visible_ratio` is below
    ``min_visible_ratio`` (0 = off). Returns ``(results, number marked)``.
    """
    out: list[SlotResult] = []
    marked = 0
    for r in results:
        if r.pose is not None and r.detection is not None and r.pose_trusted:
            mask = r.detection.mask
            hidden = mask_touches_border(mask, border_px) or (
                min_visible_ratio > 0.0
                and not visible_ratio(mask, frame.depth_m, frame.K, size)
                >= min_visible_ratio
            )
            if hidden:
                r = replace(r, pose_trusted=False)
                marked += 1
        out.append(r)
    return out, marked
