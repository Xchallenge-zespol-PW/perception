#!/usr/bin/env python3
"""
Auto-label cube frames with Grounded SAM → YOLO-seg dataset.

Uses autodistill + autodistill-grounded-sam with ontology ``{"cube": "kostka"}``,
then assigns colour class from ``rois.yaml`` slots and the ``ulozenie`` column
(R→czerwona=0, G→zielona=1, X→skip).

Train/val split is by **bag**, not by frame.

Colab (GPU runtime)::

    !pip install -q autodistill autodistill-grounded-sam supervision \\
        opencv-python-headless pyyaml torch torchvision
    # clone / mount this repo, then:
    import sys
    sys.path.insert(0, '/content/perception/src/g1_perception')  # adjust path
    !python tools/autolabel.py --index data/index.csv --visualize

Local (after ``source install/setup.bash``)::

    python3 tools/autolabel.py --visualize
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from g1_perception.core.config import Roi, load_rois

# YOLO-seg class ids
CLASS_RED = 0
CLASS_GREEN = 1
CLASS_NAMES = ['czerwona', 'zielona']
LETTER_TO_CLASS = {'R': CLASS_RED, 'G': CLASS_GREEN}

ONTOLOGY = {'cube': 'kostka'}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Auto-label cubes with Grounded SAM and build a YOLO-seg dataset '
            'coloured by ROI slot + ulozenie.'
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
        help='rois.yaml (default: package config/rois.yaml)',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('datasets/cubes_seg'),
        help='Output YOLO-seg root (default: datasets/cubes_seg)',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=Path.cwd(),
        help='Base directory for paths in index.csv (default: cwd)',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.25,
        help='Fraction of bags for validation (default: 0.25)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed for bag split (default: 0)',
    )
    parser.add_argument(
        '--box-threshold',
        type=float,
        default=0.25,
        help='Grounding DINO box threshold (default: 0.25)',
    )
    parser.add_argument(
        '--text-threshold',
        type=float,
        default=0.25,
        help='Grounding DINO text threshold (default: 0.25)',
    )
    parser.add_argument(
        '--min-overlap',
        type=float,
        default=0.15,
        help='Min mask∩ROI / mask area to assign a slot (default: 0.15)',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Write preview images under <out-dir>/visualizations/',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Optional cap on number of frames (debug / Colab smoke test)',
    )
    return parser.parse_args(argv)


def load_index(path: Path) -> list[dict[str, str]]:
    """Load ``data/index.csv`` rows."""
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f'No rows in {path}')
    return rows


def split_bags(
    bags: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """
    Split unique bag names into train/val sets.

    With a single bag everything goes to train. Otherwise at least one bag
    is placed in val when ``val_ratio > 0`` and there are ≥2 bags.
    """
    bags = sorted(bags)
    if not bags:
        return set(), set()
    if len(bags) == 1 or val_ratio <= 0.0:
        return set(bags), set()

    rng = random.Random(seed)
    shuffled = bags[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)
    val = set(shuffled[:n_val])
    train = set(shuffled[n_val:])
    return train, val


def build_model(box_threshold: float, text_threshold: float):
    """Construct GroundedSAM (lazy import for Colab / optional dependency)."""
    try:
        import torch
        from autodistill.detection import CaptionOntology
        from autodistill_grounded_sam import GroundedSAM
    except ImportError as exc:
        raise SystemExit(
            'Missing dependency. Install with:\n'
            '  pip install autodistill autodistill-grounded-sam supervision\n'
            f'Original error: {exc}'
        ) from exc

    if torch.cuda.is_available():
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print(
            'WARNING: CUDA not available — Grounded SAM will be slow on CPU.',
            file=sys.stderr,
        )

    ontology = CaptionOntology(ONTOLOGY)
    try:
        return GroundedSAM(
            ontology=ontology,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    except TypeError:
        # Older autodistill-grounded-sam may not expose thresholds.
        return GroundedSAM(ontology=ontology)


def best_slot_for_mask(
    mask: np.ndarray,
    rois: list[Roi],
    min_overlap: float,
) -> int | None:
    """Return ROI slot with highest mask overlap, or None if below threshold."""
    mask_area = float(np.count_nonzero(mask))
    if mask_area <= 0:
        return None

    best_slot: int | None = None
    best_score = 0.0
    h, w = mask.shape[:2]
    for roi in rois:
        y0, y1 = max(roi.y0, 0), min(roi.y1, h)
        x0, x1 = max(roi.x0, 0), min(roi.x1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        inter = float(np.count_nonzero(mask[y0:y1, x0:x1]))
        score = inter / mask_area
        if score > best_score:
            best_score = score
            best_slot = roi.slot

    if best_slot is None or best_score < min_overlap:
        return None
    return best_slot


def class_from_ulozenie(ulozenie: str, slot: int) -> int | None:
    """Map ``ulozenie[slot]`` letter to YOLO class id (skip X / unknown)."""
    if slot < 0 or slot >= len(ulozenie):
        return None
    letter = ulozenie[slot].upper()
    return LETTER_TO_CLASS.get(letter)


def mask_to_yolo_polygon(
    mask: np.ndarray,
    class_id: int,
) -> str | None:
    """
    Convert a boolean mask to one YOLO-seg line.

    Format: ``class x1 y1 x2 y2 ...`` with normalized coordinates.
    """
    h, w = mask.shape[:2]
    u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None

    # Simplify slightly to keep label files small.
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx.shape[0] < 3:
        approx = contour

    coords: list[str] = []
    for point in approx.reshape(-1, 2):
        x = float(point[0]) / w
        y = float(point[1]) / h
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        coords.append(f'{x:.6f}')
        coords.append(f'{y:.6f}')
    return f'{class_id} ' + ' '.join(coords)


def detections_to_labels(
    detections,
    rois: list[Roi],
    ulozenie: str,
    min_overlap: float,
) -> list[tuple[int, np.ndarray, str]]:
    """
    Assign each SAM mask to a colour class via ROI + ulozenie.

    Returns list of ``(class_id, mask, yolo_line)``.
    """
    if detections.mask is None or len(detections) == 0:
        return []

    out: list[tuple[int, np.ndarray, str]] = []
    for i in range(len(detections)):
        mask = detections.mask[i].astype(bool)
        slot = best_slot_for_mask(mask, rois, min_overlap)
        if slot is None:
            continue
        class_id = class_from_ulozenie(ulozenie, slot)
        if class_id is None:
            continue
        line = mask_to_yolo_polygon(mask, class_id)
        if line is None:
            continue
        out.append((class_id, mask, line))
    return out


def write_data_yaml(out_dir: Path) -> None:
    """Write Ultralytics-compatible ``data.yaml`` for YOLO-seg."""
    payload = {
        'path': str(out_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)},
        'nc': len(CLASS_NAMES),
    }
    with (out_dir / 'data.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def visualize_frame(
    bgr: np.ndarray,
    items: list[tuple[int, np.ndarray, str]],
    rois: list[Roi],
    path: Path,
) -> None:
    """Overlay masks, class names and ROI boxes for manual review."""
    vis = bgr.copy()
    colors = {
        CLASS_RED: (0, 0, 255),
        CLASS_GREEN: (0, 200, 0),
    }
    for class_id, mask, _line in items:
        color = colors.get(class_id, (255, 255, 0))
        overlay = vis.copy()
        overlay[mask] = color
        vis = cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)
        ys, xs = np.where(mask)
        if ys.size:
            cv2.putText(
                vis,
                CLASS_NAMES[class_id],
                (int(xs.min()), max(int(ys.min()) - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    for roi in rois:
        cv2.rectangle(
            vis, (roi.x0, roi.y0), (roi.x1, roi.y1), (255, 255, 0), 1
        )
        cv2.putText(
            vis,
            f's{roi.slot}',
            (roi.x0, max(roi.y0 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def prepare_dirs(out_dir: Path) -> None:
    """Create YOLO-seg folder layout."""
    for split in ('train', 'val'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.val_ratio < 1.0:
        print('ERROR: --val-ratio must be in [0, 1)', file=sys.stderr)
        return 2

    rows = load_index(args.index)
    if args.max_frames is not None:
        rows = rows[: args.max_frames]

    try:
        rois = load_rois(args.rois)
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    if not rois:
        print(
            'ERROR: rois.yaml is empty — run tools/roi_picker.py first',
            file=sys.stderr,
        )
        return 1

    bags = sorted({row['bag'] for row in rows})
    train_bags, val_bags = split_bags(bags, args.val_ratio, args.seed)
    print(
        f'Bags: {len(bags)} total | train={sorted(train_bags)} | '
        f'val={sorted(val_bags)}'
    )

    out_dir: Path = args.out_dir
    prepare_dirs(out_dir)
    write_data_yaml(out_dir)

    model = build_model(args.box_threshold, args.text_threshold)

    by_bag: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bag[row['bag']].append(row)

    n_ok = 0
    n_empty = 0
    for bag, bag_rows in sorted(by_bag.items()):
        if bag in train_bags:
            split = 'train'
        elif bag in val_bags:
            split = 'val'
        else:
            continue

        for row in bag_rows:
            color_path = args.root / row['color_path']
            if not color_path.is_file():
                print(f'WARNING: missing {color_path}', file=sys.stderr)
                continue

            stem = f'{row["bag"]}_{row["frame"]}'
            try:
                detections = model.predict(str(color_path))
            except Exception as exc:  # noqa: BLE001 — keep batch going on Colab
                print(f'WARNING: predict failed for {color_path}: {exc}', file=sys.stderr)
                continue

            ulozenie = row.get('ulozenie', '')
            items = detections_to_labels(
                detections, rois, ulozenie, args.min_overlap
            )

            img_out = out_dir / 'images' / split / f'{stem}.png'
            lbl_out = out_dir / 'labels' / split / f'{stem}.txt'
            shutil.copy2(color_path, img_out)
            lbl_out.write_text(
                '\n'.join(line for _cid, _m, line in items) + ('\n' if items else ''),
                encoding='utf-8',
            )

            if items:
                n_ok += 1
            else:
                n_empty += 1

            if args.visualize:
                bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    visualize_frame(
                        bgr,
                        items,
                        rois,
                        out_dir / 'visualizations' / split / f'{stem}.jpg',
                    )

            print(
                f'[{split}] {stem}: {len(items)} cube(s)  ulozenie={ulozenie}'
            )

    print(
        f'Done → {out_dir}  '
        f'(labelled={n_ok}, empty={n_empty}, data.yaml written)'
    )
    if args.visualize:
        print(f'Previews → {out_dir / "visualizations"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
