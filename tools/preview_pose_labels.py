#!/usr/bin/env python3
"""
Check and draw YOLO-pose cube labels (box + numbered corners c0..c3).

Reports labels whose corners are not in the canonical order from
``docs/training.md`` (visually clockwise, ``c0`` = lowest corner) and
writes preview images for a manual look.

Example::

    python3 tools/preview_pose_labels.py datasets/cubes_pose --split val --limit 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from g1_perception.core.keypoints import N_CORNERS, order_top_corners

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp')
CORNER_COLORS_BGR = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 128, 0)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Check corner order and draw YOLO-pose labels.'
    )
    parser.add_argument('dataset', type=Path, help='Dataset dir with data.yaml')
    parser.add_argument('--split', default='train', help='train | val (default: train)')
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Preview dir (default: <dataset>/preview_<split>)',
    )
    parser.add_argument(
        '--limit', type=int, default=30, help='Max preview images (default: 30)'
    )
    return parser.parse_args(argv)


def parse_pose_line(
    line: str, width: int, height: int
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cls, box_xyxy, corners_px (4, 2), visibility (4,))``."""
    values = [float(v) for v in line.split()]
    expected = 5 + 3 * N_CORNERS
    if len(values) != expected:
        raise ValueError(f'expected {expected} values, got {len(values)}')
    cls = int(values[0])
    cx, cy, w, h = values[1:5]
    box = np.array([
        (cx - w / 2) * width, (cy - h / 2) * height,
        (cx + w / 2) * width, (cy + h / 2) * height,
    ])
    kpts = np.asarray(values[5:], dtype=np.float64).reshape(N_CORNERS, 3)
    corners = kpts[:, :2] * np.array([width, height])
    return cls, box, corners, kpts[:, 2]


def order_problem(corners: np.ndarray, visibility: np.ndarray) -> str | None:
    """Describe why labelled corners are not ``c0..c3``, or None if fine."""
    if not np.any(visibility > 0):
        return None
    if np.any(visibility <= 0):
        return 'some corners have visibility 0'
    if not np.allclose(order_top_corners(corners), corners, atol=1e-3):
        return 'corners not in c0..c3 order'
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_yaml = args.dataset / 'data.yaml'
    names = {}
    if data_yaml.is_file():
        names = yaml.safe_load(data_yaml.read_text(encoding='utf-8')).get('names', {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    img_dir = args.dataset / 'images' / args.split
    lbl_dir = args.dataset / 'labels' / args.split
    out_dir = args.out or args.dataset / f'preview_{args.split}'
    out_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_cubes = n_drawn = 0
    problems: list[str] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl_path = lbl_dir / f'{img_path.stem}.txt'
        if not lbl_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            problems.append(f'{img_path.name}: unreadable image')
            continue
        height, width = img.shape[:2]
        n_images += 1
        for i, line in enumerate(lbl_path.read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            try:
                cls, box, corners, vis = parse_pose_line(line, width, height)
            except ValueError as exc:
                problems.append(f'{lbl_path.name}:{i + 1}: {exc}')
                continue
            n_cubes += 1
            problem = order_problem(corners, vis)
            if problem:
                problems.append(f'{lbl_path.name}:{i + 1}: {problem}')
            _draw(img, box, corners, vis, str(names.get(cls, cls)), problem)
        if n_drawn < args.limit:
            cv2.imwrite(str(out_dir / f'{img_path.stem}.jpg'), img)
            n_drawn += 1

    for p in problems:
        print(f'WARN: {p}', file=sys.stderr)
    print(
        f'{n_images} images, {n_cubes} cubes, {len(problems)} problems; '
        f'{n_drawn} previews in {out_dir}'
    )
    return 1 if problems else 0


def _draw(
    img: np.ndarray,
    box: np.ndarray,
    corners: np.ndarray,
    vis: np.ndarray,
    name: str,
    problem: str | None,
) -> None:
    color = (0, 0, 255) if problem else (255, 255, 255)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    cv2.putText(
        img, name, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
    )
    if not np.any(vis > 0):
        return
    pts = corners.round().astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (200, 200, 200), 1)
    for k, (p, v) in enumerate(zip(pts, vis)):
        if v <= 0:
            continue
        cv2.circle(img, tuple(int(c) for c in p), 3, CORNER_COLORS_BGR[k], -1)
        cv2.putText(
            img, f'c{k}', (int(p[0]) + 4, int(p[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, CORNER_COLORS_BGR[k], 1,
        )


if __name__ == '__main__':
    raise SystemExit(main())
