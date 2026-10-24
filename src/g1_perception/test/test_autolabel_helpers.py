"""Unit tests for autolabel helpers (no Grounded SAM / GPU required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from g1_perception.core.config import Roi

_TOOLS = Path(__file__).resolve().parents[3] / 'tools' / 'autolabel.py'


@pytest.fixture(scope='module')
def al():
    """Load ``tools/autolabel.py`` as a module without running ``main``."""
    spec = importlib.util.spec_from_file_location('autolabel_tool', _TOOLS)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules['autolabel_tool'] = mod
    spec.loader.exec_module(mod)
    return mod


def test_split_bags_by_bag_not_frame(al):
    train, val = al.split_bags(
        ['bag_a', 'bag_b', 'bag_c', 'bag_d'], val_ratio=0.25, seed=0
    )
    assert train.isdisjoint(val)
    assert train | val == {'bag_a', 'bag_b', 'bag_c', 'bag_d'}
    assert len(val) == 1
    assert len(train) == 3


def test_split_bags_single_goes_to_train(al):
    train, val = al.split_bags(['only'], val_ratio=0.5, seed=0)
    assert train == {'only'}
    assert val == set()


def test_class_from_ulozenie(al):
    assert al.class_from_ulozenie('RGGR', 0) == al.CLASS_RED
    assert al.class_from_ulozenie('RGGR', 1) == al.CLASS_GREEN
    assert al.class_from_ulozenie('RXGR', 1) is None
    assert al.class_from_ulozenie('RG', 5) is None


def test_best_slot_for_mask(al):
    rois = [
        Roi(slot=0, x0=0, y0=0, x1=10, y1=10),
        Roi(slot=1, x0=20, y0=0, x1=30, y1=10),
    ]
    mask = np.zeros((20, 40), dtype=bool)
    mask[2:8, 22:28] = True
    assert al.best_slot_for_mask(mask, rois, min_overlap=0.1) == 1


def test_mask_to_yolo_polygon(al):
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:50] = True
    line = al.mask_to_yolo_polygon(mask, class_id=0)
    assert line is not None
    parts = line.split()
    assert parts[0] == '0'
    assert len(parts) >= 1 + 6  # class + ≥3 points
    coords = [float(x) for x in parts[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)
