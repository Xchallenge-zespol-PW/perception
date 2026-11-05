"""Tests for YOLO model loading with CPU fallback (no ultralytics / torch)."""

from __future__ import annotations

from pathlib import Path

import pytest

from g1_perception.segmenters import yolo_backend
from g1_perception.segmenters.yolo_backend import (
    YoloBackend,
    load_yolo,
    needs_cuda,
)


class _FakeYolo:
    """Stand-in for ``ultralytics.YOLO`` recording predict calls."""

    instances: list[_FakeYolo] = []
    fail_suffixes: tuple[str, ...] = ()

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: list[dict] = []
        _FakeYolo.instances.append(self)

    def predict(self, _img, **kwargs):
        if Path(self.path).suffix in _FakeYolo.fail_suffixes:
            raise RuntimeError('TensorRT engine deserialization failed')
        self.calls.append(kwargs)
        return []


@pytest.fixture(autouse=True)
def _fake_ultralytics(monkeypatch):
    _FakeYolo.instances = []
    _FakeYolo.fail_suffixes = ()
    monkeypatch.setattr(yolo_backend, '_import_yolo', lambda: _FakeYolo)
    monkeypatch.setattr(yolo_backend, 'cuda_available', lambda: True)


def _touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b'x')
    return path


def test_needs_cuda():
    assert needs_cuda(Path('m.engine'), 'cpu')
    assert needs_cuda(Path('m.pt'), '0')
    assert not needs_cuda(Path('m.onnx'), 'cpu')
    assert not needs_cuda(Path('m.onnx'), None)
    assert not needs_cuda(Path('m.pt'), '')


def test_primary_loads_and_warms_up(tmp_path):
    engine = _touch(tmp_path, 'cubes.engine')
    backend = load_yolo(
        engine, '0', warmup_runs=2, predict_kwargs={'conf': 0.3}
    )
    assert backend.model_path == engine
    assert backend.device == '0'
    assert not backend.fallback
    assert backend.describe() == 'engine:0'
    calls = backend.model.calls
    assert len(calls) == 2
    assert calls[0]['device'] == '0' and calls[0]['conf'] == 0.3


def test_no_cuda_falls_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_backend, 'cuda_available', lambda: False)
    engine = _touch(tmp_path, 'cubes.engine')
    onnx = _touch(tmp_path, 'cubes.onnx')
    backend = load_yolo(engine, '0', fallback_model_path=onnx, warmup_runs=1)
    assert backend.fallback
    assert backend.model_path == onnx
    assert backend.device == 'cpu'
    assert 'CUDA' in (backend.reason or '')
    assert backend.describe() == 'onnx:cpu fallback'
    assert backend.model.calls[0]['device'] == 'cpu'
    # The engine was never constructed without CUDA.
    assert [Path(m.path).suffix for m in _FakeYolo.instances] == ['.onnx']


def test_warmup_failure_falls_back(tmp_path):
    _FakeYolo.fail_suffixes = ('.engine',)
    engine = _touch(tmp_path, 'cubes.engine')
    onnx = _touch(tmp_path, 'cubes.onnx')
    backend = load_yolo(engine, '0', fallback_model_path=onnx, warmup_runs=1)
    assert backend.fallback
    assert 'deserialization' in (backend.reason or '')


def test_missing_primary_falls_back(tmp_path):
    onnx = _touch(tmp_path, 'cubes.onnx')
    backend = load_yolo(
        tmp_path / 'missing.engine', '0', fallback_model_path=onnx
    )
    assert backend.fallback
    assert backend.model_path == onnx


def test_missing_primary_without_fallback_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='not found'):
        load_yolo(tmp_path / 'missing.pt', 'cpu')


def test_both_models_unusable_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_backend, 'cuda_available', lambda: False)
    engine = _touch(tmp_path, 'cubes.engine')
    with pytest.raises(RuntimeError, match='Neither YOLO model loads'):
        load_yolo(
            engine, '0', fallback_model_path=tmp_path / 'missing.onnx'
        )


def test_unsupported_suffix_raises(tmp_path):
    weights = _touch(tmp_path, 'cubes.tflite')
    with pytest.raises(RuntimeError, match='Unsupported model format'):
        load_yolo(weights, 'cpu')


def test_describe_auto_device():
    backend = YoloBackend(model=None, model_path=Path('m.pt'), device=None)
    assert backend.describe() == 'pt:auto'
