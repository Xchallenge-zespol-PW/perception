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
    *,
    inference_ms: float | None = None,
) -> NDArray[np.uint8]:
    """
    Render a debug RGB image.

    Current-frame masks are tinted by their per-frame label. With ROIs,
    boxes and text use the voted label. Without ROIs, each result with a
    mask shows ``id:colour status`` (and yaw) at the mask centroid —
    pass ``results`` whose ``slot`` is already the place / track id.
    Optional ``inference_ms`` is drawn in the top-left corner.
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

    if rois:
        _draw_roi_labels(out, rois, results, voted)
    else:
        _draw_free_labels(out, results, voted)

    if inference_ms is not None:
        cv2.putText(
            out,
            f'{inference_ms:.0f} ms',
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_roi_labels(
    out: NDArray[np.uint8],
    rois: Sequence[Roi],
    results: Sequence[SlotResult],
    voted: Sequence[VotedSlot],
) -> None:
    by_slot_voted = {v.slot: v for v in voted}
    by_slot_result = {r.slot: r for r in results if r.slot is not None}
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


def _draw_free_labels(
    out: NDArray[np.uint8],
    results: Sequence[SlotResult],
    voted: Sequence[VotedSlot],
) -> None:
    by_id = {v.slot: v for v in voted}
    for result in results:
        if result.detection is None or not result.detection.mask.any():
            continue
        v = by_id.get(result.slot) if result.slot is not None else None
        label = v.label if v is not None else result.label
        status = (
            'ok' if v is not None and v.assignment_ok
            else ('bad' if v is not None else 'raw')
        )
        cube_id = (
            str(result.slot) if result.slot is not None
            else (str(v.track_id) if v and v.track_id is not None else '?')
        )
        color = _color(label)
        ys, xs = np.where(result.detection.mask)
        cx = int(xs.mean())
        cy = int(ys.mean())
        text = f'{cube_id}:{label} {status}'
        if v is not None and v.yaw_deg is not None:
            text += f' yaw {v.yaw_deg:.0f}'
        cv2.putText(
            out,
            text,
            (max(cx - 40, 2), max(cy - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS_RGB.get(label, LABEL_COLORS_RGB[LABEL_UNCERTAIN])
