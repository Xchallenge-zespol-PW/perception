#!/usr/bin/env python3
"""
Turn YOLO detection boxes into segmentation polygons.

Label files that mix a polygon (``class x1 y1 ...``) with a box
(``class xc yc w h``) are rejected by Ultralytics segment training.
This script leaves existing polygon rows untouched and replaces only
the 5-number box rows. SAM2 is prompted with that box; the class id
stays the one already in the label. If SAM returns no mask, the four
corners of the box are written instead.

Each rewritten label is copied to ``<file>.txt.bak`` first. Files with
no box rows are not opened for writing.

Example::

    python3 tools/box2poly.py datasets/cubes_seg
    python3 tools/box2poly.py datasets/cubes_seg --device 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Replace YOLO detection boxes with SAM polygons. Polygon rows are kept.'
    )
    parser.add_argument(
        'dataset',
        type=Path,
        help='Dataset root with train/ and valid/ (images/ + labels/)',
    )
    parser.add_argument(
        '--model',
        default='sam2.1_b.pt',
        help='Ultralytics SAM weights (default: sam2.1_b.pt)',
    )
    parser.add_argument(
        '--device',
        default='',
        help="Device, e.g. '' (auto) or '0'",
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid'],
        help='Split directory names (default: train valid)',
    )
    return parser.parse_args(argv)


def box_corners(xc: float, yc: float, bw: float, bh: float) -> np.ndarray:
    """Four normalized corners of a YOLO xywh box, shape (4, 2)."""
    x0, y0 = xc - bw / 2, yc - bh / 2
    x1, y1 = xc + bw / 2, yc + bh / 2
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def sam_polygon(result, fallback: np.ndarray) -> np.ndarray:
    """Largest-point SAM contour, or the box corners when SAM finds nothing."""
    masks = result.masks
    polys = []
    if masks is not None and masks.xyn is not None:
        polys = [np.asarray(p) for p in masks.xyn if len(p) >= 3]
    if not polys:
        return fallback
    return max(polys, key=len)


def format_row(class_id: str, poly: np.ndarray) -> str:
    coords = ' '.join(f'{v:.6f}' for v in np.asarray(poly, dtype=np.float64).ravel())
    return f'{class_id} {coords}'


def find_image(images_dir: Path, stem: str) -> Path | None:
    matches = sorted(images_dir.glob(f'{stem}.*'))
    return matches[0] if matches else None


def convert_label(label_path: Path, images_dir: Path, sam, device: str) -> str:
    """
    Rewrite one label file.

    Returns ``skipped`` (no box rows), ``missing-image``, or ``fixed``.
    """
    rows = [line.split() for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not any(len(row) == 5 for row in rows):
        return 'skipped'

    image_path = find_image(images_dir, label_path.stem)
    if image_path is None:
        print(f'WARNING: no image for {label_path}', file=sys.stderr)
        return 'missing-image'
    image = cv2.imread(str(image_path))
    if image is None:
        print(f'WARNING: unreadable image {image_path}', file=sys.stderr)
        return 'missing-image'
    height, width = image.shape[:2]

    out: list[str] = []
    for row in rows:
        if len(row) != 5:
            out.append(' '.join(row))
            continue
        class_id = row[0]
        xc, yc, bw, bh = (float(value) for value in row[1:])
        corners = box_corners(xc, yc, bw, bh)
        x0, y0 = corners[0]
        x1, y1 = corners[2]
        box = [float(x0 * width), float(y0 * height), float(x1 * width), float(y1 * height)]
        predict_kwargs = {'bboxes': [box], 'verbose': False}
        if device != '':
            predict_kwargs['device'] = device
        result = sam(image, **predict_kwargs)[0]
        out.append(format_row(class_id, sam_polygon(result, corners)))

    backup = label_path.with_suffix(label_path.suffix + '.bak')
    shutil.copy2(label_path, backup)
    label_path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return 'fixed'


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        print(f'ERROR: dataset not found: {root}', file=sys.stderr)
        return 1

    try:
        from ultralytics import SAM
    except ImportError as exc:
        print(
            'Missing ultralytics. Use the training venv:\n'
            '  /home/dev/venvs/yolo/bin/python tools/box2poly.py ...\n'
            f'Original error: {exc}',
            file=sys.stderr,
        )
        return 1

    sam = SAM(args.model)

    fixed = 0
    for split in args.splits:
        labels_dir = root / split / 'labels'
        images_dir = root / split / 'images'
        if not labels_dir.is_dir():
            print(f'WARNING: missing {labels_dir}', file=sys.stderr)
            continue
        for label_path in sorted(labels_dir.glob('*.txt')):
            status = convert_label(label_path, images_dir, sam, args.device)
            if status == 'fixed':
                fixed += 1
                if fixed == 1 or fixed % 10 == 0:
                    print(f'fixed {fixed}: {label_path.relative_to(root)}')
    print(f'poprawiono plików: {fixed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
