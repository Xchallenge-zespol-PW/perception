#!/usr/bin/env python3
"""
Convert a CVAT export (polygon + 4 corners per cube) to YOLO seg + pose.

Input: "CVAT for images 1.1" XML. Per cube, label one **polygon** around
the whole visible cube and the 4 **top-face corners** as a ``points``
shape (4 points) or a 4-point ``skeleton``, in any order. Corners are
paired with the polygon by ``group_id`` or, without it, by the smallest
polygon containing their centre. Corners are re-ordered to ``c0..c3``
(``docs/pose_pnp.md``), so the labelling order does not matter.

Output (split by recording = first directory of the image name)::

    <out>/cubes_seg/{images,labels}/{train,val}/  + data.yaml
    <out>/cubes_pose/{images,labels}/{train,val}/ + data.yaml (kpt_shape)

Example::

    python3 tools/cvat_to_yolo.py annotations.xml --images-root data/frames
    python3 tools/cvat_to_yolo.py annotations.xml --images-root data/frames \\
        --val-groups kostki_RGGR_niskie_20261015_184332 --overwrite
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from g1_perception.core.config import load_label_map
from g1_perception.core.keypoints import N_CORNERS, order_top_corners
from g1_perception.core.labels import canonical_label
from g1_perception.core.types import COLOR_LABELS

CLASS_NAMES = list(COLOR_LABELS)  # ['red', 'green']
# Ultralytics pose: index i becomes flip_idx[i] after a horizontal flip.
FLIP_IDX = [0, 3, 2, 1]
VIS_VISIBLE = 2
VIS_OCCLUDED = 1


@dataclass
class Corners:
    """4 top-face corners (any order) and their per-point visibility."""

    points: np.ndarray  # (4, 2) pixels
    visibility: list[int]
    label: str
    group_id: str | None = None


@dataclass
class Cube:
    """One labelled cube: whole-cube polygon and optional top-face corners."""

    label: str
    polygon: np.ndarray  # (N, 2) pixels
    group_id: str | None = None
    corners: Corners | None = None


@dataclass
class LabeledImage:
    """All cubes on one CVAT image."""

    name: str
    width: int
    height: int
    cubes: list[Cube] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CVAT XML (polygon + 4 corners) → YOLO seg + pose datasets.'
    )
    parser.add_argument('xml', type=Path, help='CVAT for images 1.1 XML')
    parser.add_argument(
        '--images-root',
        type=Path,
        default=Path('data/frames'),
        help='Directory the CVAT image names are relative to (default: data/frames)',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('datasets'),
        help='Output root (default: datasets)',
    )
    parser.add_argument('--seg-name', default='cubes_seg', help='Seg dataset dir name')
    parser.add_argument('--pose-name', default='cubes_pose', help='Pose dataset dir name')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Fraction of recordings for val (default: 0.2); ignored with --val-groups',
    )
    parser.add_argument(
        '--val-groups',
        nargs='*',
        default=None,
        help='Recording names (first path component) that go to val',
    )
    parser.add_argument('--seed', type=int, default=0, help='Split seed (default: 0)')
    parser.add_argument(
        '--labels-path',
        type=Path,
        default=None,
        help='labels.yaml mapping CVAT label names (default: package config)',
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Symlink images instead of copying (not for Colab uploads)',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace existing output dataset directories',
    )
    return parser.parse_args(argv)


def parse_points(text: str) -> np.ndarray:
    """Parse CVAT ``"x1,y1;x2,y2;..."`` into an ``(N, 2)`` float array."""
    pts = [
        [float(v) for v in pair.split(',')]
        for pair in text.strip().split(';')
        if pair.strip()
    ]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def parse_cvat_xml(path: Path) -> tuple[list[LabeledImage], list[str]]:
    """Parse a CVAT XML export; return images and human-readable warnings."""
    root = ET.parse(path).getroot()
    images: list[LabeledImage] = []
    warnings: list[str] = []
    for el in root.iter('image'):
        name = el.get('name', '')
        polygons: list[Cube] = []
        corner_sets: list[Corners] = []
        for shape in el:
            group = shape.get('group_id')
            label = shape.get('label', '')
            if shape.tag == 'polygon':
                polygons.append(
                    Cube(label, parse_points(shape.get('points', '')), group)
                )
            elif shape.tag == 'points':
                corners = _corners_from_points(shape, label, group)
                if corners is None:
                    warnings.append(
                        f'{name}: points shape with != {N_CORNERS} points skipped'
                    )
                else:
                    corner_sets.append(corners)
            elif shape.tag == 'skeleton':
                corners = _corners_from_skeleton(shape, label, group)
                if corners is None:
                    warnings.append(
                        f'{name}: skeleton without {N_CORNERS} points skipped'
                    )
                else:
                    corner_sets.append(corners)
        cubes, pair_warnings = pair_corners(polygons, corner_sets)
        warnings.extend(f'{name}: {w}' for w in pair_warnings)
        images.append(
            LabeledImage(
                name=name,
                width=int(float(el.get('width', 0))),
                height=int(float(el.get('height', 0))),
                cubes=cubes,
            )
        )
    return images, warnings


def pair_corners(
    cubes: list[Cube],
    corner_sets: list[Corners],
) -> tuple[list[Cube], list[str]]:
    """
    Attach each corner set to one cube polygon.

    Same ``group_id`` wins; otherwise the smallest polygon containing the
    corners' centre. Corners without a polygon are dropped with a warning
    (the pose box must cover the whole cube).
    """
    warnings: list[str] = []
    for corners in corner_sets:
        target = None
        if corners.group_id is not None:
            target = next(
                (c for c in cubes if c.group_id == corners.group_id), None
            )
        if target is None:
            center = corners.points.mean(axis=0)
            inside = [
                c for c in cubes
                if c.corners is None and _contains(c.polygon, center)
            ]
            if inside:
                target = min(inside, key=lambda c: _area(c.polygon))
        if target is None:
            warnings.append('corners outside every cube polygon skipped')
            continue
        if target.corners is not None:
            warnings.append('second corner set for one cube skipped')
            continue
        if corners.label and corners.label != target.label:
            warnings.append(
                f'corner label {corners.label!r} != polygon label '
                f'{target.label!r}; using the polygon label'
            )
        target.corners = corners
    return cubes, warnings


def seg_line(cube: Cube, cls: int, width: int, height: int) -> str:
    """YOLO-seg line: class + normalised polygon."""
    norm = cube.polygon / np.array([width, height], dtype=np.float64)
    coords = ' '.join(f'{v:.6f}' for v in np.clip(norm, 0.0, 1.0).ravel())
    return f'{cls} {coords}'


def pose_line(cube: Cube, cls: int, width: int, height: int) -> str:
    """
    YOLO-pose line: class, polygon box, 4 corners ``c0..c3`` with visibility.

    Without corners (top face not visible) all keypoints are ``0 0 0``.
    """
    x0, y0 = cube.polygon.min(axis=0)
    x1, y1 = cube.polygon.max(axis=0)
    box = [
        (x0 + x1) / 2.0 / width,
        (y0 + y1) / 2.0 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]
    parts = [str(cls)] + [f'{v:.6f}' for v in np.clip(box, 0.0, 1.0)]
    if cube.corners is None:
        parts += ['0'] * (3 * N_CORNERS)
    else:
        vis_by_point = {
            tuple(p): v
            for p, v in zip(cube.corners.points, cube.corners.visibility)
        }
        for p in order_top_corners(cube.corners.points):
            parts += [
                f'{np.clip(p[0] / width, 0.0, 1.0):.6f}',
                f'{np.clip(p[1] / height, 0.0, 1.0):.6f}',
                str(vis_by_point[tuple(p)]),
            ]
    return ' '.join(parts)


def group_of(name: str) -> str:
    """Recording of an image: first path component (or ``'default'``)."""
    parts = Path(name).parts
    return parts[0] if len(parts) > 1 else 'default'


def split_groups(
    groups: list[str],
    val_ratio: float,
    seed: int,
    val_groups: list[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Split recordings into train / val (explicit list or seeded ratio)."""
    unique = sorted(set(groups))
    if val_groups is not None:
        unknown = set(val_groups) - set(unique)
        if unknown:
            raise ValueError(f'Unknown --val-groups: {sorted(unknown)}')
        val = set(val_groups)
        return set(unique) - val, val
    if len(unique) < 2 or val_ratio <= 0.0:
        return set(unique), set()
    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    n_val = min(max(1, round(len(shuffled) * val_ratio)), len(shuffled) - 1)
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def flat_name(name: str) -> str:
    """Unique flat file name for an image path (``a/b.png`` → ``a__b.png``)."""
    return '__'.join(Path(name).parts)


def write_data_yaml(out_dir: Path, *, pose: bool) -> None:
    """Write Ultralytics ``data.yaml`` (pose adds ``kpt_shape`` / ``flip_idx``)."""
    payload: dict = {
        'path': str(out_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    if pose:
        payload['kpt_shape'] = [N_CORNERS, 3]
        payload['flip_idx'] = FLIP_IDX
    with (out_dir / 'data.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images, warnings = parse_cvat_xml(args.xml)
    label_map = load_label_map(args.labels_path)

    seg_dir = args.out / args.seg_name
    pose_dir = args.out / args.pose_name
    for out_dir in (seg_dir, pose_dir):
        if out_dir.exists() and any(out_dir.iterdir()):
            if not args.overwrite:
                print(
                    f'ERROR: {out_dir} exists and is not empty '
                    '(use --overwrite or another --seg-name / --pose-name)',
                    file=sys.stderr,
                )
                return 1
            shutil.rmtree(out_dir)

    train, val = split_groups(
        [group_of(im.name) for im in images],
        args.val_ratio,
        args.seed,
        args.val_groups,
    )
    counts = {'images': 0, 'cubes': 0, 'with_corners': 0, 'missing_image': 0}
    for im in images:
        src = args.images_root / im.name
        if not src.is_file():
            src = args.images_root / Path(im.name).name
        if not src.is_file():
            warnings.append(f'{im.name}: image not found under {args.images_root}')
            counts['missing_image'] += 1
            continue
        width, height = im.width, im.height
        if width <= 0 or height <= 0:
            img = cv2.imread(str(src))
            height, width = img.shape[:2]

        seg_lines: list[str] = []
        pose_lines: list[str] = []
        for cube in im.cubes:
            canonical = canonical_label(cube.label, label_map)
            if canonical not in CLASS_NAMES:
                warnings.append(f'{im.name}: label {cube.label!r} skipped')
                continue
            cls = CLASS_NAMES.index(canonical)
            seg_lines.append(seg_line(cube, cls, width, height))
            pose_lines.append(pose_line(cube, cls, width, height))
            counts['cubes'] += 1
            counts['with_corners'] += cube.corners is not None

        split = 'val' if group_of(im.name) in val else 'train'
        stem = Path(flat_name(im.name)).stem
        for out_dir, lines in ((seg_dir, seg_lines), (pose_dir, pose_lines)):
            img_dst = out_dir / 'images' / split / flat_name(im.name)
            lbl_dst = out_dir / 'labels' / split / f'{stem}.txt'
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            if args.symlink:
                img_dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, img_dst)
            lbl_dst.write_text(
                '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8'
            )
        counts['images'] += 1

    for out_dir, pose in ((seg_dir, False), (pose_dir, True)):
        for split in ('train', 'val'):
            (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        write_data_yaml(out_dir, pose=pose)
    if not val:
        warnings.append('val split is empty (need >= 2 recordings or --val-groups)')

    for w in warnings:
        print(f'WARN: {w}', file=sys.stderr)
    print(
        f"Done: {counts['images']} images, {counts['cubes']} cubes "
        f"({counts['with_corners']} with corners), "
        f"{counts['missing_image']} missing images, {len(warnings)} warnings\n"
        f'  train recordings: {sorted(train)}\n'
        f'  val recordings:   {sorted(val)}\n'
        f'  {seg_dir}/data.yaml\n  {pose_dir}/data.yaml'
    )
    return 0


def _corners_from_points(
    shape: ET.Element, label: str, group: str | None
) -> Corners | None:
    pts = parse_points(shape.get('points', ''))
    if pts.shape[0] != N_CORNERS:
        return None
    vis = VIS_OCCLUDED if shape.get('occluded') == '1' else VIS_VISIBLE
    return Corners(pts, [vis] * N_CORNERS, label, group)


def _corners_from_skeleton(
    shape: ET.Element, label: str, group: str | None
) -> Corners | None:
    """Skeleton children: one point each; ``outside=1`` = not labelled."""
    pts: list[np.ndarray] = []
    vis: list[int] = []
    for child in shape.iter('points'):
        if child.get('outside') == '1':
            continue
        pts.append(parse_points(child.get('points', ''))[0])
        vis.append(VIS_OCCLUDED if child.get('occluded') == '1' else VIS_VISIBLE)
    if len(pts) != N_CORNERS:
        return None
    return Corners(np.stack(pts), vis, label, group)


def _contains(polygon: np.ndarray, point: np.ndarray) -> bool:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def _area(polygon: np.ndarray) -> float:
    return float(abs(cv2.contourArea(polygon.astype(np.float32))))


if __name__ == '__main__':
    raise SystemExit(main())
