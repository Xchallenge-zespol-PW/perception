"""Configurable perception pipeline: segment then estimate pose."""

from __future__ import annotations

from typing import Any

from g1_perception.core.types import Detection, Frame, Pose, PoseEstimator, Segmenter

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
    """Runs segmentation, then pose estimation for each detection."""

    def __init__(
        self,
        segmenter: Segmenter,
        pose_estimator: PoseEstimator,
    ) -> None:
        self._segmenter = segmenter
        self._pose_estimator = pose_estimator

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Pipeline:
        """
        Build a pipeline from a configuration dictionary.

        Expected shape::

            {
                "segmenter": {"name": "...", "params": {...}},
                "pose_estimator": {"name": "...", "params": {...}},
            }

        ``params`` is optional and defaults to an empty dict. Each ``name``
        must refer to a class previously registered with :func:`register`.
        """
        segmenter = _build_stage(config, 'segmenter')
        pose_estimator = _build_stage(config, 'pose_estimator')
        return cls(segmenter=segmenter, pose_estimator=pose_estimator)

    def process(
        self, frame: Frame
    ) -> list[tuple[Detection, Pose | None]]:
        """Segment ``frame`` and estimate a pose for every detection."""
        detections = self._segmenter.segment(frame)
        return [
            (det, self._pose_estimator.estimate(frame, det))
            for det in detections
        ]


def _build_stage(config: dict[str, Any], key: str) -> Any:
    """Instantiate a registered stage from ``config[key]``."""
    try:
        stage_cfg = config[key]
    except KeyError as exc:
        raise KeyError(f"Missing config key: {key!r}") from exc

    try:
        name = stage_cfg['name']
    except KeyError as exc:
        raise KeyError(f"Missing '{key}.name' in config") from exc

    params = stage_cfg.get('params', {})
    impl_cls = get_registered(name)
    return impl_cls(**params)
