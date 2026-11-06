"""Tests for dataset / training / export tools (no ultralytics, no GPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

_TOOLS = Path(__file__).resolve().parents[3] / 'tools'

CVAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="bag_a/0_color.png" width="100" height="80">
    <polygon label="red" points="10,10;40,10;40,50;10,50" group_id="1"/>
    <points label="red" points="15,12;35,12;35,30;15,32" group_id="1"/>
    <polygon label="zielona" points="60,10;90,10;90,50;60,50"/>
    <points label="green" points="85,14;65,14;65,30;85,30"/>
  </image>
  <image id="1" name="bag_b/0_color.png" width="100" height="80">
    <polygon label="green" points="20,20;50,20;50,60;20,60"/>
    <skeleton label="corners">
      <points label="1" points="25,25" outside="0" occluded="0"/>
      <points label="2" points="45,25" outside="0" occluded="1"/>
      <points label="3" points="45,40" outside="0" occluded="0"/>
      <points label="4" points="25,40" outside="0" occluded="0"/>
    </skeleton>
    <points label="red" points="1,1;2,2;3,3;4,4;5,5"/>
  </image>
</annotations>
"""


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f'{name}_tool', _TOOLS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'{name}_tool'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def cvat():
    return _load_tool('cvat_to_yolo')


@pytest.fixture(scope='module')
def preview():
    return _load_tool('preview_pose_labels')


@pytest.fixture(scope='module')
def train():
    return _load_tool('train_yolo')


@pytest.fixture(scope='module')
def export():
    return _load_tool('export_model')


@pytest.fixture(scope='module')
def bench():
    return _load_tool('bench_models')


@pytest.fixture(scope='module')
def compare():
    return _load_tool('compare_backends')


def _write_cvat(tmp_path: Path) -> tuple[Path, Path]:
    xml = tmp_path / 'annotations.xml'
    xml.write_text(CVAT_XML, encoding='utf-8')
    frames = tmp_path / 'frames'
    for bag in ('bag_a', 'bag_b'):
        (frames / bag).mkdir(parents=True)
        cv2.imwrite(str(frames / bag / '0_color.png'), np.zeros((80, 100, 3), np.uint8))
    return xml, frames


def test_parse_pairs_corners_by_group_and_containment(cvat, tmp_path):
    xml, _ = _write_cvat(tmp_path)
    images, warnings = cvat.parse_cvat_xml(xml)
    first, second = images
    assert [c.corners is not None for c in first.cubes] == [True, True]
    # Second cube has no group: paired by containment despite label mismatch.
    assert any('label' in w for w in warnings)
    # Skeleton corners keep per-point visibility (occluded → 1).
    assert second.cubes[0].corners.visibility == [2, 1, 2, 2]
    # A 5-point shape is not a corner set.
    assert any('points shape' in w for w in warnings)


def test_pose_line_orders_corners_and_keeps_visibility(cvat, tmp_path):
    xml, _ = _write_cvat(tmp_path)
    images, _ = cvat.parse_cvat_xml(xml)
    cube = images[1].cubes[0]
    values = [float(v) for v in cvat.pose_line(cube, 1, 100, 80).split()]
    assert values[0] == 1
    np.testing.assert_allclose(values[1:5], [0.35, 0.5, 0.3, 0.5])
    kpts = np.array(values[5:]).reshape(4, 3)
    # c0 = bottom-right (tie in v within 3 px → larger u), then clockwise.
    np.testing.assert_allclose(kpts[:, :2] * [100, 80], [[45, 40], [25, 40], [25, 25], [45, 25]])
    assert list(kpts[:, 2]) == [2, 2, 2, 1]


def test_pose_line_without_corners_is_zero_keypoints(cvat):
    cube = cvat.Cube('red', np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float))
    values = cvat.pose_line(cube, 0, 20, 20).split()
    assert len(values) == 17 and values[5:] == ['0'] * 12


def test_split_groups_explicit_and_ratio(cvat):
    train, val = cvat.split_groups(['a', 'b', 'c'], 0.2, 0, ['b'])
    assert val == {'b'} and train == {'a', 'c'}
    train, val = cvat.split_groups(['a', 'b', 'c', 'd', 'e'], 0.2, 0)
    assert len(val) == 1 and len(train) == 4
    assert cvat.split_groups(['a'], 0.5, 0) == ({'a'}, set())
    with pytest.raises(ValueError, match='Unknown'):
        cvat.split_groups(['a'], 0.2, 0, ['zz'])


def test_main_writes_both_datasets_that_pass_preview(cvat, preview, tmp_path, capsys):
    xml, frames = _write_cvat(tmp_path)
    out = tmp_path / 'datasets'
    rc = cvat.main([
        str(xml), '--images-root', str(frames), '--out', str(out),
        '--val-groups', 'bag_b',
    ])
    assert rc == 0
    seg = yaml.safe_load((out / 'cubes_seg' / 'data.yaml').read_text())
    pose = yaml.safe_load((out / 'cubes_pose' / 'data.yaml').read_text())
    assert 'kpt_shape' not in seg
    assert pose['kpt_shape'] == [4, 3] and pose['flip_idx'] == [0, 3, 2, 1]
    assert pose['names'] == {0: 'red', 1: 'green'}

    seg_lbl = out / 'cubes_seg' / 'labels' / 'train' / 'bag_a__0_color.txt'
    pose_lbl = out / 'cubes_pose' / 'labels' / 'val' / 'bag_b__0_color.txt'
    assert [line.split()[0] for line in seg_lbl.read_text().splitlines()] == ['0', '1']
    assert len(pose_lbl.read_text().splitlines()) == 1
    assert (out / 'cubes_pose' / 'images' / 'val' / 'bag_b__0_color.png').is_file()

    for split in ('train', 'val'):
        assert preview.main([str(out / 'cubes_pose'), '--split', split]) == 0
    # A second run refuses to overwrite.
    assert cvat.main([str(xml), '--images-root', str(frames), '--out', str(out)]) == 1


def test_preview_flags_non_canonical_order(preview):
    corners = np.array([[25, 25], [45, 25], [45, 40], [25, 40]], float)
    assert preview.order_problem(corners, np.full(4, 2.0)) is not None
    assert preview.order_problem(corners, np.zeros(4)) is None


def test_resolve_data_yaml_roboflow_and_path_style(train, tmp_path):
    rf = tmp_path / 'rf'
    for split in ('train', 'valid'):
        (rf / split / 'images').mkdir(parents=True)
    (rf / 'data.yaml').write_text(
        'train: ../train/images\nval: ../valid/images\ntest: ../test/images\n'
        "names: ['green cube', 'red cube']\n"
    )
    data = train.resolve_data_yaml(rf / 'data.yaml')
    assert data['train'] == str((rf / 'train' / 'images').resolve())
    assert data['val'] == str((rf / 'valid' / 'images').resolve())
    assert 'test' not in data

    ds = tmp_path / 'ds'
    (ds / 'images' / 'train').mkdir(parents=True)
    (ds / 'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\n')
    with pytest.raises(FileNotFoundError, match='val'):
        train.resolve_data_yaml(ds / 'data.yaml')


def test_task_defaults_and_data_checks(train):
    args = train.parse_args(['--task', 'pose'])
    assert args.model == 'yolo26n-pose.pt' and args.fliplr == 0.0 and args.hsv_h == 0.0
    assert args.data == Path('datasets/cubes_pose/data.yaml')
    assert train.parse_args([]).fliplr == 0.5
    train.check_task_data('pose', {'kpt_shape': [4, 3]})
    with pytest.raises(ValueError, match='kpt_shape'):
        train.check_task_data('pose', {})
    with pytest.raises(ValueError, match='pose dataset'):
        train.check_task_data('seg', {'kpt_shape': [4, 3]})


def test_export_formats_and_targets(export, tmp_path):
    assert export.parse_formats('onnx, engine') == ['onnx', 'engine']
    with pytest.raises(ValueError):
        export.parse_formats('tflite')
    assert export.target_path(tmp_path, 'cubes_pose', 'engine') == tmp_path / 'cubes_pose.engine'
    assert export.target_path(tmp_path, 'cubes_pose', 'pt') == tmp_path / 'cubes_pose.pt'


def test_bench_summary_and_device(bench):
    stats = bench.summarize([10.0, 20.0, 30.0, 40.0])
    assert stats['mean_ms'] == pytest.approx(25.0)
    assert stats['fps'] == pytest.approx(40.0)
    assert bench.device_for(Path('m.onnx'), '0') == 'cpu'
    assert bench.device_for(Path('m.engine'), '0') == '0'


def test_compare_results_matches_by_iou(compare):
    class _B:
        def __init__(self, xyxy, conf):
            self.xyxy = np.asarray(xyxy, float)
            self.conf = np.asarray(conf, float)

    class _R:
        def __init__(self, xyxy, conf, kpts=None):
            self.boxes = _B(xyxy, conf)
            self.keypoints = None if kpts is None else type('K', (), {'data': np.asarray(kpts)})
            self.masks = None

    ref = _R([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.8], [[[1, 1, 1]], [[55, 55, 1]]])
    test = _R([[50.5, 50, 60, 60], [0, 0, 10, 11]], [0.79, 0.91], [[[55, 56, 1]], [[1, 1, 1]]])
    out = compare.compare_results(ref, test)
    assert out['count_diff'] == 0
    assert out['box_px'] == pytest.approx(1.0)
    assert out['conf'] == pytest.approx(0.01)
    assert out['kpt_px'] == pytest.approx(1.0)
    assert out['mask_iou'] == 1.0
