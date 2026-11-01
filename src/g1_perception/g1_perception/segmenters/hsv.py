"""HSV color segmenter over calibrated table-slot ROIs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from g1_perception.core.config import (
    HsvConfig,
    HsvRange,
    Roi,
    load_hsv,
    load_rois,
)
from g1_perception.core.types import (
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Detection,
    Frame,
)
from g1_perception.pipeline import register


def _in_range(hsv: np.ndarray, r: HsvRange) -> np.ndarray:
    """Return uint8 mask for one HSV range via ``cv2.inRange``."""
    lower = np.array([r.h_min, r.s_min, r.v_min], dtype=np.uint8)
    upper = np.array([r.h_max, r.s_max, r.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


@register('hsv')
class HsvSegmenter:
    """
    Classify each ROI as red or green from HSV pixel fractions.

    For every ROI builds red (two hue wraps) and green masks, optionally
    erodes the ROI border, then picks the colour with the larger fraction
    if it is at least ``min_frac``; otherwise the ROI yields no detection
    (reported as ``empty`` by the pipeline). A ROI with no pixels on the
    image yields ``uncertain``.
    """

    def __init__(
        self,
        rois: list[Roi] | None = None,
        hsv: HsvConfig | None = None,
        *,
        rois_path: str | Path | None = None,
        hsv_path: str | Path | None = None,
        min_frac: float = 0.05,
        erode: int = 1,
    ) -> None:
        if rois is None:
            self.rois = load_rois(
                Path(rois_path) if rois_path is not None else None
            )
        else:
            self.rois = list(rois)

        if hsv is None:
            self.hsv = load_hsv(
                Path(hsv_path) if hsv_path is not None else None
            )
        else:
            self.hsv = hsv

        if not 0.0 <= min_frac <= 1.0:
            raise ValueError(f'min_frac must be in [0, 1], got {min_frac}')
        if erode < 0:
            raise ValueError(f'erode must be >= 0, got {erode}')

        self.min_frac = float(min_frac)
        self.erode = int(erode)

    def segment(self, frame: Frame) -> list[Detection]:
        """Return a :class:`Detection` for every ROI with a colour found."""
        h, w = frame.rgb.shape[:2]
        hsv_img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)

        red = _in_range(hsv_img, self.hsv.red_low) | _in_range(
            hsv_img, self.hsv.red_high
        )
        green = _in_range(hsv_img, self.hsv.green)

        detections = [
            self._classify_roi(roi, red, green, h, w) for roi in self.rois
        ]
        return [det for det in detections if det is not None]

    def _classify_roi(
        self,
        roi: Roi,
        red: np.ndarray,
        green: np.ndarray,
        h: int,
        w: int,
    ) -> Detection | None:
        """Classify a single ROI; None when no colour reaches ``min_frac``."""
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        y0 = max(roi.y0, 0)
        x0 = max(roi.x0, 0)
        y1 = min(roi.y1, h)
        x1 = min(roi.x1, w)
        if y1 > y0 and x1 > x0:
            roi_mask[y0:y1, x0:x1] = 255

        if self.erode > 0:
            kernel = np.ones((3, 3), np.uint8)
            roi_mask = cv2.erode(
                roi_mask, kernel, iterations=self.erode
            )

        area = int(np.count_nonzero(roi_mask))
        if area == 0:
            return Detection(
                slot=roi.slot,
                label=LABEL_UNCERTAIN,
                score=0.0,
                mask=np.zeros((h, w), dtype=bool),
            )

        inside = roi_mask > 0
        red_frac = float(np.count_nonzero(inside & (red > 0))) / area
        green_frac = float(np.count_nonzero(inside & (green > 0))) / area

        if red_frac >= green_frac:
            best_label = LABEL_RED
            best_frac = red_frac
            best_pixels = inside & (red > 0)
        else:
            best_label = LABEL_GREEN
            best_frac = green_frac
            best_pixels = inside & (green > 0)

        if best_frac < self.min_frac:
            return None

        return Detection(
            slot=roi.slot,
            label=best_label,
            score=best_frac,
            mask=best_pixels,
        )
