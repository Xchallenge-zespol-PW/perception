"""Tests for bag recording / frame extraction tools (no camera, no ``ros2``)."""

from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[3] / 'tools'
_RECORD_BALLS = _TOOLS / 'record_balls.sh'

# Fake ``ros2``: logs every call, ``bag record -o DIR`` creates DIR like a real bag.
_FAKE_ROS2 = """#!/usr/bin/env bash
echo "$@" >> "$FAKE_ROS2_LOG"
if [ "$1 $2" = "bag record" ]; then
  while [ $# -gt 0 ]; do
    if [ "$1" = "-o" ]; then mkdir -p "$2"; touch "$2/metadata.yaml"; fi
    shift
  done
elif [ "$1 $2" = "bag info" ]; then
  echo "Duration: 1.0s"
  echo "Topic: /camera/camera/color/image_raw"
fi
"""


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f'{name}_tool', _TOOLS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'{name}_tool'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def labels_csv():
    return _load_tool('labels_csv')


@pytest.fixture(scope='module')
def extract():
    return _load_tool('extract_frames')


@pytest.fixture
def fake_ros2(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    ros2 = bin_dir / 'ros2'
    ros2.write_text(_FAKE_ROS2, encoding='utf-8')
    ros2.chmod(0o755)
    log = tmp_path / 'ros2.log'
    bag_dir = tmp_path / 'bags'
    env = dict(os.environ)
    env['PATH'] = f'{bin_dir}{os.pathsep}{env.get("PATH", "")}'
    env['FAKE_ROS2_LOG'] = str(log)
    env['BAG_DIR'] = str(bag_dir)
    return env, log, bag_dir


def _run_balls(env, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['bash', str(_RECORD_BALLS), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


@pytest.mark.parametrize(
    'args',
    [
        (),
        ('3',),
        ('3', 'h60'),
        ('5', 'h60', 'lampa'),
        ('-1', 'h60', 'lampa'),
        ('x', 'h60', 'lampa'),
        ('34', 'h60', 'lampa'),
        ('3', 'h80', 'lampa'),
        ('3', 'H60', 'lampa'),
        ('3', 'polka', 'lampa'),
        ('3', 'h60', 'lampa boczna'),
        ('3', 'h60', 'lampa-bok'),
        ('3', 'h60', 'lampa', '0'),
        ('3', 'h60', 'lampa', '1.5'),
        ('3', 'h60', 'lampa', 'abc'),
    ],
)
def test_record_balls_rejects_bad_args_without_recording(fake_ros2, args):
    env, log, bag_dir = fake_ros2
    res = _run_balls(env, *args)
    assert res.returncode != 0
    assert not log.exists(), 'ros2 must not be called for invalid arguments'
    assert not (bag_dir / 'labels_pilki.csv').exists()
    assert not bag_dir.exists() or not any(bag_dir.iterdir())


@pytest.mark.parametrize('shelf', ['h60', 'h100', 'h60_100', 'stol'])
def test_record_balls_records_and_appends_label(fake_ros2, labels_csv, shelf):
    env, log, bag_dir = fake_ros2
    res = _run_balls(env, '4', shelf, 'sufit_lampa', '1', 'pilka przy krawedzi, lekki odblask')
    assert res.returncode == 0, res.stdout + res.stderr

    bags = [p for p in bag_dir.iterdir() if p.is_dir()]
    assert len(bags) == 1
    name = bags[0].name
    assert name.startswith(f'pilki_4_{shelf}_sufit_lampa_')

    record_call = log.read_text(encoding='utf-8').splitlines()[0]
    assert record_call.startswith('bag record -s mcap -o ')
    assert '/camera/camera/aligned_depth_to_color/image_raw' in record_call

    header, rows = _read_csv(bag_dir / 'labels_pilki.csv')
    assert header == labels_csv.BALL_FIELDNAMES
    assert rows == [
        {
            'bag': name,
            'liczba_pilek': '4',
            'polka': shelf,
            'swiatlo': 'sufit_lampa',
            'czas_s': '1',
            'uwagi': 'pilka przy krawedzi, lekki odblask',
        }
    ]
    assert not (bag_dir / 'labels.csv').exists(), 'cube labels must stay untouched'


def test_record_balls_defaults_and_appends(fake_ros2):
    env, _log, bag_dir = fake_ros2
    assert _run_balls(env, '0', 'stol', 'dzienne').returncode == 0
    # Note starting with '-' must not be parsed as a flag.
    assert _run_balls(env, '2', 'h100', 'dzienne', '3', '-brak-').returncode == 0

    _, rows = _read_csv(bag_dir / 'labels_pilki.csv')
    assert [(r['liczba_pilek'], r['polka'], r['czas_s'], r['uwagi']) for r in rows] == [
        ('0', 'stol', '10', ''),
        ('2', 'h100', '3', '-brak-'),
    ]


def test_record_balls_no_label_when_bag_missing(tmp_path, fake_ros2):
    env, _log, bag_dir = fake_ros2
    # ros2 that "fails" without creating the bag directory
    (tmp_path / 'bin' / 'ros2').write_text('#!/usr/bin/env bash\nexit 1\n', encoding='utf-8')
    res = _run_balls(env, '1', 'h60', 'lampa', '1')
    assert res.returncode != 0
    assert not (bag_dir / 'labels_pilki.csv').exists()


def test_ball_labels_csv_ensure_keeps_existing_rows(labels_csv, tmp_path):
    path = tmp_path / 'labels_pilki.csv'
    labels_csv.main(['ensure-balls', str(path)])
    labels_csv.main([
        'append-balls', str(path), '--bag', 'pilki_1_h60_lampa_x',
        '--liczba-pilek', '1', '--polka', 'h60', '--swiatlo', 'lampa',
        '--czas-s', '10',
    ])
    labels_csv.main(['ensure-balls', str(path)])
    header, rows = _read_csv(path)
    assert header == labels_csv.BALL_FIELDNAMES
    assert [r['bag'] for r in rows] == ['pilki_1_h60_lampa_x']
    assert rows[0]['uwagi'] == ''


def test_cube_labels_csv_unchanged(labels_csv, tmp_path):
    path = tmp_path / 'labels.csv'
    labels_csv.main(['ensure', str(path)])
    header, _ = _read_csv(path)
    assert header == ['bag', 'ulozenie', 'swiatlo', 'czas_s', 'uwagi', 'yaw_deg']


def _fake_bag(bags_dir: Path, name: str) -> None:
    (bags_dir / name).mkdir(parents=True)
    (bags_dir / name / 'metadata.yaml').write_text('', encoding='utf-8')


@pytest.fixture
def fake_read_bag(extract, monkeypatch):
    def read_bag(_bag_dir):
        colors = [(i * 33_000_000, np.full((4, 6, 3), i, np.uint8)) for i in range(5)]
        depths = [(t + 1_000_000, np.ones((4, 6), np.float32)) for t, _ in colors]
        info = {'width': 6, 'height': 4, 'K': [[1, 0, 3], [0, 1, 2], [0, 0, 1]]}
        return colors, depths, info

    monkeypatch.setattr(extract, 'read_bag', read_bag)


def _write_labels(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_extract_frames_ball_bags_named_after_bag(extract, labels_csv, fake_read_bag, tmp_path):
    bags_dir = tmp_path / 'bags'
    out_dir = tmp_path / 'data'
    names = ['pilki_3_h60_lampa_20261110_120000', 'pilki_0_stol_dzienne_20261110_120100']
    for name in names:
        _fake_bag(bags_dir, name)
    labels = tmp_path / 'labels_pilki.csv'
    _write_labels(labels, labels_csv.BALL_FIELDNAMES, [
        {'bag': names[0], 'liczba_pilek': '3', 'polka': 'h60', 'swiatlo': 'lampa',
         'czas_s': '10', 'uwagi': ''},
        {'bag': names[1], 'liczba_pilek': '0', 'polka': 'stol', 'swiatlo': 'dzienne',
         'czas_s': '10', 'uwagi': ''},
    ])

    assert extract.main([
        '--labels', str(labels), '--bags-dir', str(bags_dir),
        '--out-dir', str(out_dir), '--every', '2',
    ]) == 0

    for name in names:
        files = sorted(p.name for p in (out_dir / 'frames' / name).iterdir())
        assert files == sorted(
            ['camera_info.json']
            + [f'{name}_{i}_color.png' for i in (0, 2, 4)]
            + [f'{name}_{i}_depth.npy' for i in (0, 2, 4)]
        )

    header, rows = _read_csv(out_dir / 'index.csv')
    assert header == extract.INDEX_FIELDNAMES + ['liczba_pilek', 'polka']
    assert len(rows) == 6
    first = rows[0]
    assert first['bag'] == names[0]
    assert Path(first['color_path']).name == f'{names[0]}_0_color.png'
    assert Path(first['color_path']).is_file()
    assert (first['liczba_pilek'], first['polka'], first['swiatlo']) == ('3', 'h60', 'lampa')
    assert first['ulozenie'] == ''


def test_extract_frames_cube_index_unchanged(extract, labels_csv, fake_read_bag, tmp_path):
    bags_dir = tmp_path / 'bags'
    out_dir = tmp_path / 'data'
    name = 'kostki_RGGR_lampa_20261107_231112'
    _fake_bag(bags_dir, name)
    labels = tmp_path / 'labels.csv'
    _write_labels(labels, labels_csv.FIELDNAMES, [
        {'bag': name, 'ulozenie': 'RGGR', 'swiatlo': 'lampa', 'czas_s': '15',
         'uwagi': '', 'yaw_deg': '0,15,45,30'},
    ])

    assert extract.main([
        '--labels', str(labels), '--bags-dir', str(bags_dir),
        '--out-dir', str(out_dir), '--every', '5',
    ]) == 0

    header, rows = _read_csv(out_dir / 'index.csv')
    assert header == ['bag', 'frame', 'color_path', 'depth_path',
                      'ulozenie', 'swiatlo', 'yaw_deg']
    assert len(rows) == 1
    assert rows[0]['ulozenie'] == 'RGGR'
    assert rows[0]['yaw_deg'] == '0,15,45,30'
    assert Path(rows[0]['color_path']).name == f'{name}_0_color.png'
    assert Path(rows[0]['depth_path']).name == f'{name}_0_depth.npy'
