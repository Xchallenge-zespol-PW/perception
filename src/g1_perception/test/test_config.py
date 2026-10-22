"""Tests for ROI and HSV YAML config loaders."""

from __future__ import annotations

from pathlib import Path

from g1_perception.core.config import (
    HsvConfig,
    HsvRange,
    Roi,
    load_hsv,
    load_rois,
    save_hsv,
    save_rois,
)


def test_save_and_load_rois_roundtrip(tmp_path: Path):
    path = tmp_path / 'rois.yaml'
    rois = [
        Roi(slot=1, x0=10, y0=20, x1=30, y1=40),
        Roi(slot=0, x0=1, y0=2, x1=3, y1=4),
    ]

    save_rois(rois, path)
    loaded = load_rois(path)

    assert [r.slot for r in loaded] == [0, 1]
    assert loaded[0] == Roi(slot=0, x0=1, y0=2, x1=3, y1=4)
    assert loaded[1] == rois[0]


def test_save_and_load_hsv_roundtrip(tmp_path: Path):
    path = tmp_path / 'hsv.yaml'
    cfg = HsvConfig(
        red_low=HsvRange(0, 10, 50, 255, 40, 255),
        red_high=HsvRange(170, 180, 50, 255, 40, 255),
        green=HsvRange(40, 90, 40, 255, 40, 255),
    )

    save_hsv(cfg, path)
    loaded = load_hsv(path)

    assert loaded == cfg
