"""Debug rendering: ROI boxes, masks, labels and yaw on the color image."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from g1_perception.core.config import Roi
from g1_perception.core.types import (
    LABEL_EMPTY,
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    SlotResult,
)
from g1_perception.core.voting import VotedSlot

LABEL_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    LABEL_RED: (255, 40, 40),
    LABEL_GREEN: (40, 220, 40),
    LABEL_EMPTY: (160, 160, 160),
    LABEL_UNCERTAIN: (255, 200, 0),
}
MASK_ALPHA = 0.45


def union_mask(
    results: Sequence[SlotResult],
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    """Return a mono8 (0 / 255) union of all detection masks in ``results``."""
    out = np.zeros(shape, dtype=bool)
    for result in results:
        if result.detection is not None:
            out |= result.detection.mask
    return out.astype(np.uint8) * 255


def draw_debug(
    rgb: NDArray[np.uint8],
    rois: Sequence[Roi],
    results: Sequence[SlotResult],
    voted: Sequence[VotedSlot],
) -> NDArray[np.uint8]:
    """
    Render a debug RGB image.

    Current-frame masks are tinted by their per-frame label; ROI boxes and
    text use the voted label (falling back to the per-frame result).
    """
    out = rgb.copy()
    for result in results:
        if result.detection is None or not result.detection.mask.any():
            continue
        color = np.array(_color(result.label), dtype=np.float32)
        mask = result.detection.mask
        out[mask] = (
            (1.0 - MASK_ALPHA) * out[mask] + MASK_ALPHA * color
        ).astype(np.uint8)

    by_slot_voted = {v.slot: v for v in voted}
    by_slot_result = {r.slot: r for r in results}
    for roi in rois:
        v = by_slot_voted.get(roi.slot)
        r = by_slot_result.get(roi.slot)
        label = v.label if v is not None else (r.label if r else LABEL_EMPTY)
        score = v.score if v is not None else (r.score if r else 0.0)
        color = _color(label)
        cv2.rectangle(out, (roi.x0, roi.y0), (roi.x1 - 1, roi.y1 - 1), color, 2)
        text = f'{roi.slot}:{label} {score:.2f}'
        if v is not None and v.yaw_deg is not None:
            text += f' yaw {v.yaw_deg:.0f}'
        cv2.putText(
            out,
            text,
            (roi.x0, max(roi.y0 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS_RGB.get(label, LABEL_COLORS_RGB[LABEL_UNCERTAIN])
