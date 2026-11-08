"""Load Ultralytics YOLO weights with an optional CPU fallback model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)

MODEL_SUFFIXES = ('.pt', '.onnx', '.engine')
# Camera profile on laptop and robot (640x480); YOLO letterboxes anyway.
WARMUP_SHAPE = (480, 640)


@dataclass(frozen=True)
class YoloBackend:
    """
    A loaded YOLO model and the device it runs on.

    ``device`` is passed to ``predict`` (None = Ultralytics default).
    ``fallback`` is True when the primary model could not be used;
    ``reason`` then says why.
    """

    model: Any
    model_path: Path
    device: str | None
    fallback: bool = False
    reason: str | None = None

    def describe(self) -> str:
        """Short label like ``engine:0`` or ``onnx:cpu fallback``."""
        fmt = self.model_path.suffix.lstrip('.') or '?'
        text = f'{fmt}:{self.device or "auto"}'
        return f'{text} fallback' if self.fallback else text


def needs_cuda(model_path: Path, device: str | None) -> bool:
    """Return True when ``model_path`` on ``device`` requires CUDA."""
    if model_path.suffix == '.engine':
        return True
    return device not in (None, '', 'cpu')


def cuda_available() -> bool:
    """Return True when PyTorch sees a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def engine_backend_available() -> bool:
    """Check that ``.engine`` files can run without torch (TensorRT + GPU)."""
    from g1_perception.segmenters.trt_runtime import tensorrt_available

    return tensorrt_available()


def _engine_model_cls() -> type:
    from g1_perception.segmenters.trt_runtime import TrtYoloModel

    return TrtYoloModel


def load_yolo(
    model_path: str | Path,
    device: str | None = None,
    *,
    fallback_model_path: str | Path | None = None,
    fallback_device: str | None = 'cpu',
    warmup_runs: int = 0,
    predict_kwargs: Mapping[str, Any] | None = None,
) -> YoloBackend:
    """
    Load ``model_path`` on ``device``, else ``fallback_model_path``.

    Ultralytics opens exported models (``.onnx`` / ``.engine``) on the
    first ``predict``, so ``warmup_runs`` blank predictions are part of
    loading: a TensorRT / CUDA failure there also triggers the fallback.
    The primary model is skipped when its file is missing, it needs CUDA
    and none is available, or loading / warm-up fails. The fallback runs
    on ``fallback_device`` with a warning. Raises ``FileNotFoundError`` /
    ``RuntimeError`` when neither model loads.
    """
    kwargs = dict(predict_kwargs or {})
    primary = Path(model_path)
    try:
        model = _load(primary, device, warmup_runs, kwargs)
        return YoloBackend(model=model, model_path=primary, device=device)
    except (FileNotFoundError, RuntimeError) as exc:
        if fallback_model_path is None:
            raise
        reason = str(exc)

    fallback = Path(fallback_model_path)
    _LOG.warning(
        'Primary YOLO model unusable (%s); falling back to %s on %s',
        reason, fallback, fallback_device or 'auto',
    )
    try:
        model = _load(fallback, fallback_device, warmup_runs, kwargs)
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            f'Neither YOLO model loads: primary: {reason}; fallback: {exc}'
        ) from exc
    return YoloBackend(
        model=model,
        model_path=fallback,
        device=fallback_device,
        fallback=True,
        reason=reason,
    )


def _load(
    model_path: Path,
    device: str | None,
    warmup_runs: int,
    predict_kwargs: Mapping[str, Any],
) -> Any:
    """
    Construct the model and warm it up on ``device``.

    ``.engine`` runs on :class:`TrtYoloModel` (TensorRT, no torch) when the
    ``tensorrt`` bindings and a GPU are present; everything else, and
    ``.engine`` without those bindings, goes through Ultralytics ``YOLO``.
    """
    if model_path.suffix not in MODEL_SUFFIXES:
        raise RuntimeError(
            f'Unsupported model format {model_path.suffix!r} '
            f'(expected one of {MODEL_SUFFIXES}): {model_path}'
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f'Model weights not found: {model_path} '
            f'(relative paths are resolved against cwd: {Path.cwd()})'
        )
    if model_path.suffix == '.engine' and engine_backend_available():
        model_cls = _engine_model_cls()
    elif needs_cuda(model_path, device) and not cuda_available():
        raise RuntimeError(
            f'{model_path.name} on device {device or "auto"!r} needs CUDA: '
            'no TensorRT bindings with a GPU and torch.cuda.is_available() '
            'is False'
        )
    else:
        model_cls = _import_yolo()
    kwargs = {'verbose': False, **predict_kwargs}
    if device is not None:
        kwargs['device'] = device
    blank = np.zeros((*WARMUP_SHAPE, 3), dtype=np.uint8)
    try:
        model = model_cls(str(model_path))
        for _ in range(max(int(warmup_runs), 0)):
            model.predict(blank, **kwargs)
    except Exception as exc:  # Ultralytics / TensorRT raise many types
        raise RuntimeError(
            f'Failed to load {model_path}: {exc!r}'
        ) from exc
    return model


def _import_yolo() -> type:
    """Import ``ultralytics.YOLO`` lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            'ultralytics is required for YOLO segmenters. '
            'Install with: pip install ultralytics'
        ) from exc
    return YOLO
