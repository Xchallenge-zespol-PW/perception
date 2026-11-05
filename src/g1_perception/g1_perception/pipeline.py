"""Configurable perception pipeline: segment then estimate pose per slot."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from g1_perception.core.config import Roi, load_label_map, load_rois
from g1_perception.core.labels import canonical_label
from g1_perception.core.types import (
    COLOR_LABELS,
    LABEL_EMPTY,
    Detection,
    Frame,
    PoseEstimator,
    Segmenter,
    SlotResult,
)

_REGISTRY: dict[str, type] = {}

ROIS_MODE_NONE = 'none'
ROIS_MODE_IMAGE = 'image_roi'
ROIS_MODES = (ROIS_MODE_NONE, ROIS_MODE_IMAGE)


def register(name: str):
    """
    Decorate a class to register it under ``name``.

    Names must be unique across segmenters and pose estimators.
    """

    def decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"Implementation already registered: {name!r}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_registered(name: str) -> type:
    """Return a registered implementation class by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ', '.join(sorted(_REGISTRY)) or '(none)'
        raise KeyError(
            f"Unknown implementation {name!r}. Registered: {known}"
        ) from exc


def clear_registry() -> None:
    """Remove all registered implementations (intended for tests)."""
    _REGISTRY.clear()


class Pipeline:
    """
    Run segmentation, then pose estimation.

    With ROIs (``rois_mode=image_roi``): one :class:`SlotResult` per ROI.
    Without ROIs (``rois_mode=none``): one result per YOLO detection
    (``slot=None``) with geometric pose when the label is a colour.
    """

    def __init__(
        self,
        segmenter: Segmenter,
        pose_estimator: PoseEstimator,
        rois: Sequence[Roi],
        label_map: Mapping[str, str],
        *,
        rois_mode: str = ROIS_MODE_IMAGE,
    ) -> None:
        mode = str(rois_mode).strip().lower()
        if mode not in ROIS_MODES:
            raise ValueError(
                f'rois_mode must be one of {ROIS_MODES}, got {rois_mode!r}'
            )
        self._segmenter = segmenter
        self._pose_estimator = pose_estimator
        self._rois = sorted(rois, key=lambda r: r.slot)
        self._label_map = dict(label_map)
        self._rois_mode = mode

    @property
    def rois(self) -> list[Roi]:
        """Return configured ROIs (empty when ``rois_mode=none``)."""
        return list(self._rois)

    @property
    def rois_mode(self) -> str:
        """Return ``none`` or ``image_roi``."""
        return self._rois_mode

    @property
    def backend_label(self) -> str | None:
        """Segmenter inference backend (e.g. ``engine:0``), if it has one."""
        backend = getattr(self._segmenter, 'backend', None)
        return backend.describe() if backend is not None else None

    def load(self) -> None:
        """Load segmenter models eagerly instead of on the first frame."""
        load = getattr(self._segmenter, 'load', None)
        if callable(load):
            load()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        rois: Sequence[Roi] | None = None,
        label_map: Mapping[str, str] | None = None,
    ) -> Pipeline:
        """
        Build a pipeline from a configuration dictionary.

        Expected shape::

            {
                "rois_mode": "none" | "image_roi",  # default image_roi
                "rois_path": "...",      # optional; ignored when mode=none
                "labels_path": "...",    # optional, default config/labels.yaml
                "segmenter": {"name": "...", "params": {...}},
                "pose_estimator": {"name": "...", "params": {...}},
            }

        In ``none`` mode missing ``rois.yaml`` is not an error (rois=[]).
        """
        rois_mode = str(
            config.get('rois_mode', ROIS_MODE_IMAGE)
        ).strip().lower()
        if rois_mode not in ROIS_MODES:
            raise ValueError(
                f'rois_mode must be one of {ROIS_MODES}, got {rois_mode!r}'
            )

        if rois is None:
            if rois_mode == ROIS_MODE_NONE:
                rois = []
            else:
                rois_path = _optional_path(config.get('rois_path'))
                if rois_path is None:
                    # empty path → package default; still required in image_roi
                    rois = load_rois(None)
                else:
                    rois = load_rois(rois_path)
        if label_map is None:
            label_map = load_label_map(
                _optional_path(config.get('labels_path'))
            )

        shared = {
            'rois': ('rois_path', list(rois)),
            'label_map': ('labels_path', dict(label_map)),
        }
        segmenter = _build_stage(config, 'segmenter', shared)
        pose_estimator = _build_stage(config, 'pose_estimator', shared)
        return cls(
            segmenter=segmenter,
            pose_estimator=pose_estimator,
            rois=rois,
            label_map=label_map,
            rois_mode=rois_mode,
        )

    def process(self, frame: Frame) -> list[SlotResult]:
        """Segment ``frame`` and return slot or free-detection results."""
        detections = self._segmenter.segment(frame)
        if self._rois_mode == ROIS_MODE_NONE or not self._rois:
            return self._resolve_free(frame, detections)
        best = _best_detection_per_slot(detections)
        return [
            self._resolve_slot(frame, roi.slot, best.get(roi.slot))
            for roi in self._rois
        ]

    def _resolve_free(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> list[SlotResult]:
        """One result per detection; ``slot=None``; pose for colour labels."""
        results: list[SlotResult] = []
        for det in detections:
            free = Detection(
                slot=None,
                label=det.label,
                score=det.score,
                mask=det.mask,
            )
            results.append(self._resolve_slot(frame, None, free))
        return results

    def _resolve_slot(
        self,
        frame: Frame,
        slot: int | None,
        det: Detection | None,
    ) -> SlotResult:
        """Map a detection to a canonical label and pose."""
        if det is None:
            return SlotResult(
                slot=slot,
                label=LABEL_EMPTY,
                score=0.0,
                detection=None,
                pose=None,
            )

        label = canonical_label(det.label, self._label_map)
        pose = (
            self._pose_estimator.estimate(frame, det)
            if label in COLOR_LABELS
            else None
        )
        return SlotResult(
            slot=slot,
            label=label,
            score=float(det.score),
            detection=det,
            pose=pose,
        )


def _best_detection_per_slot(
    detections: Iterable[Detection],
) -> dict[int, Detection]:
    """Keep the highest-score detection per slot; drop unassigned ones."""
    best: dict[int, Detection] = {}
    for det in detections:
        if det.slot is None:
            continue
        prev = best.get(det.slot)
        if prev is None or det.score > prev.score:
            best[det.slot] = det
    return best


def _optional_path(value: Any) -> Path | None:
    """Convert an optional config value to ``Path``."""
    return None if value is None else Path(value)


def _build_stage(
    config: dict[str, Any],
    key: str,
    shared: Mapping[str, tuple[str, Any]],
) -> Any:
    """
    Instantiate a registered stage from ``config[key]``.

    ``shared`` maps a constructor argument name to ``(path_param, value)``;
    the value is injected when the constructor accepts the argument and
    neither it nor ``path_param`` is given in ``params``.
    """
    try:
        stage_cfg = config[key]
    except KeyError as exc:
        raise KeyError(f"Missing config key: {key!r}") from exc

    try:
        name = stage_cfg['name']
    except KeyError as exc:
        raise KeyError(f"Missing '{key}.name' in config") from exc

    params = dict(stage_cfg.get('params') or {})
    impl_cls = get_registered(name)
    accepted = inspect.signature(impl_cls).parameters
    for arg, (path_param, value) in shared.items():
        if arg in accepted and arg not in params and path_param not in params:
            params[arg] = value
    return impl_cls(**params)
