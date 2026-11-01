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
    Run segmentation, then pose estimation, and report every ROI slot.

    :meth:`process` returns exactly one :class:`SlotResult` per ROI, sorted
    by slot: no detection → ``empty``, unmapped label → ``uncertain``.
    """

    def __init__(
        self,
        segmenter: Segmenter,
        pose_estimator: PoseEstimator,
        rois: Sequence[Roi],
        label_map: Mapping[str, str],
    ) -> None:
        self._segmenter = segmenter
        self._pose_estimator = pose_estimator
        self._rois = sorted(rois, key=lambda r: r.slot)
        self._label_map = dict(label_map)

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
                "rois_path": "...",      # optional, default config/rois.yaml
                "labels_path": "...",    # optional, default config/labels.yaml
                "segmenter": {"name": "...", "params": {...}},
                "pose_estimator": {"name": "...", "params": {...}},
            }

        ``params`` is optional and defaults to an empty dict. Each ``name``
        must refer to a class previously registered with :func:`register`.
        ``rois`` / ``label_map`` arguments override the YAML files. Stages
        whose constructor accepts ``rois`` / ``label_map`` receive the
        pipeline's values unless their params set them (or the matching
        ``rois_path`` / ``labels_path``) explicitly.
        """
        if rois is None:
            rois = load_rois(_optional_path(config.get('rois_path')))
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
        )

    def process(self, frame: Frame) -> list[SlotResult]:
        """Segment ``frame`` and return one result per configured ROI."""
        best = _best_detection_per_slot(self._segmenter.segment(frame))
        return [
            self._resolve_slot(frame, roi.slot, best.get(roi.slot))
            for roi in self._rois
        ]

    def _resolve_slot(
        self,
        frame: Frame,
        slot: int,
        det: Detection | None,
    ) -> SlotResult:
        """Map the slot's detection to a canonical label and pose."""
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
