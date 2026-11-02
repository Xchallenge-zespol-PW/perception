"""YOLO instance-segmentation segmenter assigned to table-slot ROIs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from g1_perception.core.config import (
    HsvConfig,
    Roi,
    load_hsv,
    load_label_map,
    load_rois,
)
from g1_perception.core.labels import canonical_label
from g1_perception.core.types import (
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
    Detection,
    Frame,
)
from g1_perception.pipeline import register
from g1_perception.segmenters.hsv import _in_range

DEFAULT_MIN_OVERLAP = 0.5


def best_slot_for_mask(
    mask: np.ndarray,
    rois: list[Roi],
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> int | None:
    """
    Assign a mask to the ROI with the largest coverage fraction.

    Coverage is ``|mask ∩ ROI| / |mask|``. Returns ``None`` if the best
    score is below ``min_overlap`` (default 50%).
    """
    mask_area = float(np.count_nonzero(mask))
    if mask_area <= 0:
        return None

    best_slot: int | None = None
    best_score = 0.0
    h, w = mask.shape[:2]
    for roi in rois:
        y0, y1 = max(roi.y0, 0), min(roi.y1, h)
        x0, x1 = max(roi.x0, 0), min(roi.x1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        inter = float(np.count_nonzero(mask[y0:y1, x0:x1]))
        score = inter / mask_area
        if score > best_score:
            best_score = score
            best_slot = roi.slot

    if best_slot is None or best_score < min_overlap:
        return None
    return best_slot


def hsv_color_in_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    hsv: HsvConfig,
    min_frac: float = 0.05,
) -> str:
    """Return majority HSV colour in ``mask`` (red / green / uncertain)."""
    if not np.any(mask):
        return LABEL_UNCERTAIN

    hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = _in_range(hsv_img, hsv.red_low) | _in_range(hsv_img, hsv.red_high)
    green = _in_range(hsv_img, hsv.green)

    area = float(np.count_nonzero(mask))
    red_frac = float(np.count_nonzero(mask & (red > 0))) / area
    green_frac = float(np.count_nonzero(mask & (green > 0))) / area

    if red_frac >= green_frac and red_frac >= min_frac:
        return LABEL_RED
    if green_frac > red_frac and green_frac >= min_frac:
        return LABEL_GREEN
    return LABEL_UNCERTAIN


@register('yolo_seg')
class YoloSegSegmenter:
    """
    Run a YOLO-seg model and map instance masks onto ROI slots.

    ``ultralytics`` is imported lazily when the model is first loaded so the
    rest of the package stays usable without that dependency. Detections
    keep the model's class name; with ``verify_hsv`` a class whose canonical
    label (via ``label_map``) disagrees with the HSV colour becomes
    ``uncertain``.
    """

    def __init__(
        self,
        model_path: str | Path,
        rois: list[Roi] | None = None,
        *,
        rois_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.7,
        min_overlap: float = DEFAULT_MIN_OVERLAP,
        verify_hsv: bool = False,
        hsv: HsvConfig | None = None,
        hsv_path: str | Path | None = None,
        hsv_min_frac: float = 0.05,
        label_map: Mapping[str, str] | None = None,
        labels_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if rois is None and rois_path is None:
            self.rois = []
        elif rois is None:
            self.rois = load_rois(
                Path(rois_path) if rois_path is not None else None
            )
        else:
            self.rois = list(rois)

        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError(
                f'min_overlap must be in [0, 1], got {min_overlap}'
            )

        self.conf = float(conf)
        self.iou = float(iou)
        self.min_overlap = float(min_overlap)
        self.verify_hsv = bool(verify_hsv)
        self.hsv_min_frac = float(hsv_min_frac)
        self.device = device
        self._model = None

        self.hsv: HsvConfig | None
        if verify_hsv:
            if hsv is not None:
                self.hsv = hsv
            else:
                self.hsv = load_hsv(
                    Path(hsv_path) if hsv_path is not None else None
                )
        else:
            self.hsv = hsv

        self.label_map: dict[str, str] | None
        if verify_hsv and label_map is None:
            self.label_map = load_label_map(
                Path(labels_path) if labels_path is not None else None
            )
        else:
            self.label_map = dict(label_map) if label_map is not None else None

    def _ensure_model(self):
        """Lazy-load Ultralytics YOLO (``.pt`` or ``.engine``)."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                'ultralytics is required for YoloSegSegmenter. '
                'Install with: pip install ultralytics'
            ) from exc
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Model weights not found: {self.model_path}')
        self._model = YOLO(str(self.model_path))
        return self._model

    def segment(self, frame: Frame) -> list[Detection]:
        """
        Predict instance masks and return one Detection per filled slot.

        When several instances map to the same slot, the highest-confidence
        one is kept. With ``verify_hsv``, mismatched colour → ``uncertain``.
        """
        model = self._ensure_model()
        h, w = frame.rgb.shape[:2]
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

        predict_kwargs = {
            'retina_masks': True,
            'conf': self.conf,
            'iou': self.iou,
            'verbose': False,
        }
        if self.device is not None:
            predict_kwargs['device'] = self.device

        results = model.predict(bgr, **predict_kwargs)
        if not results:
            return []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        names = result.names or getattr(model, 'names', {}) or {}
        masks_t = result.masks.data
        masks = (
            masks_t.cpu().numpy()
            if hasattr(masks_t, 'cpu')
            else np.asarray(masks_t)
        )
        confs = result.boxes.conf
        clss = result.boxes.cls
        confs_np = (
            confs.cpu().numpy() if hasattr(confs, 'cpu') else np.asarray(confs)
        )
        clss_np = (
            clss.cpu().numpy() if hasattr(clss, 'cpu') else np.asarray(clss)
        )
        if clss_np.size == 0:
            return []

        free = not self.rois
        # Best detection per slot (ROI mode), or every instance (free mode).
        best: dict[int, tuple[float, str, np.ndarray]] = {}
        free_dets: list[Detection] = []

        for i in range(len(clss_np)):
            mask = self._mask_to_full(masks[i], h, w)
            if not np.any(mask):
                continue

            class_id = int(clss_np[i])
            label = str(names.get(class_id, class_id))
            score = float(confs_np[i])

            if self.verify_hsv and self.hsv is not None:
                hsv_label = hsv_color_in_mask(
                    frame.rgb, mask, self.hsv, self.hsv_min_frac
                )
                model_label = canonical_label(label, self.label_map or {})
                if model_label != hsv_label:
                    label = LABEL_UNCERTAIN

            if free:
                free_dets.append(
                    Detection(
                        slot=None, label=label, score=score, mask=mask
                    )
                )
                continue

            slot = best_slot_for_mask(mask, self.rois, self.min_overlap)
            if slot is None:
                continue
            prev = best.get(slot)
            if prev is None or score > prev[0]:
                best[slot] = (score, label, mask)

        if free:
            return free_dets
        return [
            Detection(
                slot=slot,
                label=label,
                score=score,
                mask=mask,
            )
            for slot, (score, label, mask) in sorted(best.items())
        ]

    @staticmethod
    def _mask_to_full(
        mask: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Resize / threshold a model mask to full ``(H, W)`` bool."""
        m = np.asarray(mask)
        if m.dtype != np.bool_:
            m_bool = m > 0.5
        else:
            m_bool = m
        if m_bool.shape[0] != height or m_bool.shape[1] != width:
            resized = cv2.resize(
                m_bool.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            return resized.astype(bool)
        return m_bool.astype(bool)
