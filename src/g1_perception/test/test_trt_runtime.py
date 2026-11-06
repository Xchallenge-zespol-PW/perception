"""Tests for the torch-free TensorRT backend (no TensorRT / GPU needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from g1_perception.segmenters.trt_runtime import (
    EngineMeta,
    TrtYoloModel,
    join_engine_file,
    split_engine_file,
)

META = EngineMeta(names={0: 'red', 1: 'green'}, task='pose', imgsz=(640, 640), kpt_shape=(4, 3))


def test_header_roundtrip_and_ultralytics_layout():
    data = join_engine_file(META, b'ENGINE')
    meta_len = int.from_bytes(data[:4], 'little')
    assert json.loads(data[4:4 + meta_len])['names'] == {'0': 'red', '1': 'green'}
    meta, engine = split_engine_file(data)
    assert engine == b'ENGINE'
    assert EngineMeta.from_dict(meta) == META


def test_split_without_header_returns_raw_bytes():
    raw = b'\x00\x10\x00\x00plain engine bytes'
    assert split_engine_file(raw) == (None, raw)


def test_meta_from_onnx_style_strings():
    meta = EngineMeta.from_dict({
        'names': "{0: 'red', 1: 'green'}", 'task': 'segment', 'imgsz': '[640, 640]',
    })
    assert meta.names == {0: 'red', 1: 'green'} and meta.imgsz == (640, 640)
    assert meta.kpt_shape is None
    assert EngineMeta.from_dict({'names': ['a'], 'imgsz': 320}).imgsz == (320, 320)


class _FakeRunner:
    def __init__(self, engine: bytes) -> None:
        assert engine == b'ENGINE'
        self.calls = 0

    def infer(self, blob: np.ndarray) -> list[np.ndarray]:
        assert blob.shape == (1, 3, 640, 640)
        self.calls += 1
        row = [320, 320, 40, 40, 0.9, 0.05] + [300, 300, 0.9, 340, 300, 0.9,
                                               340, 340, 0.9, 300, 340, 0.9]
        return [np.asarray([row], np.float32).T[None]]


def test_trt_model_predict_decodes_like_ultralytics(tmp_path):
    path = tmp_path / 'cubes_pose.engine'
    path.write_bytes(join_engine_file(META, b'ENGINE'))
    model = TrtYoloModel(path, runner_factory=_FakeRunner)
    assert model.names == {0: 'red', 1: 'green'} and model.task == 'pose'
    res = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.25, device='0', verbose=False)[0]
    np.testing.assert_allclose(res.boxes.xyxy[0], [300, 220, 340, 260], atol=1e-4)
    assert res.keypoints.data.shape == (1, 4, 3)
    assert res.boxes.conf[0] == pytest.approx(0.9)


def test_engine_without_metadata_is_rejected(tmp_path):
    path = tmp_path / 'raw.engine'
    path.write_bytes(b'\x00' * 32)
    with pytest.raises(RuntimeError, match='metadata'):
        TrtYoloModel(path, runner_factory=_FakeRunner)
