#!/usr/bin/env python3
"""Select cube-slot ROIs on a color frame and save them to rois.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from g1_perception.core.config import Roi, default_config_dir, save_rois


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Open a PNG frame, select one rectangle per cube slot with the '
            'mouse (cv2.selectROIs), and write rois.yaml.'
        )
    )
    parser.add_argument(
        'image',
        type=Path,
        help='Color PNG path (e.g. data/frames/<bag>/<bag>_0_color.png)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=f'Output YAML (default: {default_config_dir() / "rois.yaml"})',
    )
    return parser.parse_args(argv)


def rois_from_select(image_bgr) -> list[Roi]:
    """
    Run ``cv2.selectROIs`` and convert (x, y, w, h) to :class:`Roi` list.

    Selection order becomes ``slot`` 0, 1, 2, ...
    """
    print(
        'Zaznacz ROI dla kolejnych miejsc (slot 0, 1, ...).\n'
        'ENTER / SPACE = kolejny prostokat, ESC = koniec.'
    )
    boxes = cv2.selectROIs(
        'roi_picker',
        image_bgr,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow('roi_picker')

    rois: list[Roi] = []
    for slot, box in enumerate(boxes):
        x, y, w, h = (int(v) for v in box)
        if w <= 0 or h <= 0:
            continue
        rois.append(Roi(slot=slot, x0=x, y0=y, x1=x + w, y1=y + h))
    return rois


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f'ERROR: cannot read image: {args.image}', file=sys.stderr)
        return 1

    rois = rois_from_select(image)
    if not rois:
        print('ERROR: no ROIs selected', file=sys.stderr)
        return 1

    out = save_rois(rois, args.output)
    print(f'Saved {len(rois)} ROIs -> {out}')
    for r in rois:
        print(f'  slot={r.slot}  ({r.x0},{r.y0})-({r.x1},{r.y1})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
