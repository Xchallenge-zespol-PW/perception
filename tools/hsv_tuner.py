#!/usr/bin/env python3
"""Tune red/green HSV thresholds on extracted frames with ROI overlay."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from g1_perception.core.config import (
    HsvConfig,
    HsvRange,
    Roi,
    default_config_dir,
    load_hsv,
    load_rois,
    save_hsv,
)

WIN = 'hsv_tuner'
CTRL = 'hsv_controls'

DEFAULT_HSV = HsvConfig(
    red_low=HsvRange(0, 10, 80, 255, 50, 255),
    red_high=HsvRange(170, 180, 80, 255, 50, 255),
    green=HsvRange(40, 90, 40, 255, 40, 255),
)

# Trackbar id -> (group, field)
_TRACKBARS: list[tuple[str, str, str]] = [
    ('rL_hmin', 'red_low', 'h_min'),
    ('rL_hmax', 'red_low', 'h_max'),
    ('rL_smin', 'red_low', 's_min'),
    ('rL_smax', 'red_low', 's_max'),
    ('rL_vmin', 'red_low', 'v_min'),
    ('rL_vmax', 'red_low', 'v_max'),
    ('rH_hmin', 'red_high', 'h_min'),
    ('rH_hmax', 'red_high', 'h_max'),
    ('rH_smin', 'red_high', 's_min'),
    ('rH_smax', 'red_high', 's_max'),
    ('rH_vmin', 'red_high', 'v_min'),
    ('rH_vmax', 'red_high', 'v_max'),
    ('g_hmin', 'green', 'h_min'),
    ('g_hmax', 'green', 'h_max'),
    ('g_smin', 'green', 's_min'),
    ('g_smax', 'green', 's_max'),
    ('g_vmin', 'green', 'v_min'),
    ('g_vmax', 'green', 'v_max'),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Browse data/index.csv frames (n/p), tune HSV with trackbars, '
            'preview mask + ROI boxes, save with s.'
        )
    )
    parser.add_argument(
        '--index',
        type=Path,
        default=Path('data/index.csv'),
        help='Frame index CSV (default: data/index.csv)',
    )
    parser.add_argument(
        '--rois',
        type=Path,
        default=None,
        help=f'ROI YAML (default: {default_config_dir() / "rois.yaml"})',
    )
    parser.add_argument(
        '--hsv',
        type=Path,
        default=None,
        help=f'HSV YAML to load/save (default: {default_config_dir() / "hsv.yaml"})',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Base directory for paths in index.csv (default: cwd)',
    )
    return parser.parse_args(argv)


def load_index(path: Path) -> list[dict[str, str]]:
    """Read index.csv rows that have a color_path."""
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f'No rows in {path}')
    return rows


def try_load_hsv(path: Path | None) -> HsvConfig:
    """Load HSV config if the file exists, otherwise return defaults."""
    cfg_path = path if path is not None else default_config_dir() / 'hsv.yaml'
    if cfg_path.is_file():
        return load_hsv(cfg_path)
    return DEFAULT_HSV


def try_load_rois(path: Path | None) -> list[Roi]:
    """Load ROIs if the file exists, otherwise return an empty list."""
    cfg_path = path if path is not None else default_config_dir() / 'rois.yaml'
    if not cfg_path.is_file():
        return []
    return load_rois(cfg_path)


def _field_max(field: str) -> int:
    return 180 if field.startswith('h_') else 255


def create_trackbars(cfg: HsvConfig) -> None:
    """Create HSV trackbars on the controls window."""
    cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
    groups = {
        'red_low': cfg.red_low,
        'red_high': cfg.red_high,
        'green': cfg.green,
    }
    for name, group, field in _TRACKBARS:
        value = getattr(groups[group], field)
        cv2.createTrackbar(name, CTRL, int(value), _field_max(field), lambda _v: None)


def read_hsv_from_trackbars() -> HsvConfig:
    """Read current trackbar positions into an :class:`HsvConfig`."""

    def _range(prefix_group: str) -> HsvRange:
        vals: dict[str, int] = {}
        for name, group, field in _TRACKBARS:
            if group != prefix_group:
                continue
            vals[field] = cv2.getTrackbarPos(name, CTRL)
        return HsvRange(**vals)

    return HsvConfig(
        red_low=_range('red_low'),
        red_high=_range('red_high'),
        green=_range('green'),
    )


def hsv_mask(hsv: np.ndarray, r: HsvRange) -> np.ndarray:
    """Binary mask for one HSV range."""
    lower = np.array([r.h_min, r.s_min, r.v_min], dtype=np.uint8)
    upper = np.array([r.h_max, r.s_max, r.v_max], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def render_preview(
    bgr: np.ndarray,
    cfg: HsvConfig,
    rois: list[Roi],
    title: str,
) -> np.ndarray:
    """Overlay red/green masks and draw ROI rectangles."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = hsv_mask(hsv, cfg.red_low) | hsv_mask(hsv, cfg.red_high)
    green = hsv_mask(hsv, cfg.green)

    overlay = bgr.copy()
    overlay[red > 0] = (0, 0, 255)
    overlay[green > 0] = (0, 255, 0)
    vis = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)

    for roi in rois:
        cv2.rectangle(
            vis,
            (roi.x0, roi.y0),
            (roi.x1, roi.y1),
            (255, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f'slot {roi.slot}',
            (roi.x0, max(roi.y0 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        'n/p: next/prev   s: save   q/ESC: quit',
        (8, vis.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return vis


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_index(args.index)
    except OSError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    rois = try_load_rois(args.rois)
    hsv_cfg = try_load_hsv(args.hsv)
    hsv_out = args.hsv if args.hsv is not None else default_config_dir() / 'hsv.yaml'

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    create_trackbars(hsv_cfg)

    idx = 0
    print(f'{len(rows)} frames  |  ROIs: {len(rois)}  |  save -> {hsv_out}')
    print('Keys: n next, p prev, s save, q/ESC quit')

    while True:
        row = rows[idx]
        color_path = args.root / row['color_path']
        bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'WARNING: cannot read {color_path}', file=sys.stderr)
            idx = (idx + 1) % len(rows)
            continue

        cfg = read_hsv_from_trackbars()
        title = (
            f'[{idx + 1}/{len(rows)}] {row.get("bag", "")} '
            f'frame={row.get("frame", "")}'
        )
        vis = render_preview(bgr, cfg, rois, title)
        cv2.imshow(WIN, vis)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('n'):
            idx = (idx + 1) % len(rows)
        elif key == ord('p'):
            idx = (idx - 1) % len(rows)
        elif key == ord('s'):
            out = save_hsv(cfg, hsv_out)
            print(f'Saved HSV -> {out}')

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
