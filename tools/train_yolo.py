#!/usr/bin/env python3
"""
Train a YOLO instance-segmentation model on ``datasets/cubes_seg``.

Uses Ultralytics (default weights: ``yolo11n-seg.pt``).

Example::

    python3 tools/train_yolo.py
    python3 tools/train_yolo.py --epochs 100 --device 0 --batch 8
    python3 tools/train_yolo.py --data datasets/cubes_seg/data.yaml \\
        --model yolo11s-seg.pt --project runs/seg --name cubes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train YOLO-seg on the cubes_seg dataset (Ultralytics).'
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('datasets/cubes_seg/data.yaml'),
        help='Dataset YAML (default: datasets/cubes_seg/data.yaml)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n-seg.pt',
        help='Base weights or checkpoint (default: yolo11n-seg.pt)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs (default: 50)',
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size (default: 640)',
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=8,
        help='Batch size (default: 8)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help="Device, e.g. '' (auto), 'cpu', '0', '0,1'",
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Dataloader workers (default: 4)',
    )
    parser.add_argument(
        '--project',
        type=Path,
        default=Path('runs/seg'),
        help='Ultralytics project directory (default: runs/seg)',
    )
    parser.add_argument(
        '--name',
        type=str,
        default='cubes',
        help='Run name under project (default: cubes)',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early-stopping patience (default: 20)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed (default: 0)',
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Overwrite an existing run directory',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        print(
            f'ERROR: dataset YAML not found: {args.data}\n'
            f'  Run tools/autolabel.py first.',
            file=sys.stderr,
        )
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            'Missing ultralytics. Install with:\n'
            '  pip install ultralytics\n'
            f'Original error: {exc}'
        ) from exc

    train_kwargs = {
        'data': str(args.data),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'project': str(args.project),
        'name': args.name,
        'patience': args.patience,
        'seed': args.seed,
        'exist_ok': args.exist_ok,
    }
    if args.device != '':
        train_kwargs['device'] = args.device

    print(f'Training {args.model} on {args.data}')
    model = YOLO(args.model)
    results = model.train(**train_kwargs)
    save_dir = getattr(results, 'save_dir', args.project / args.name)
    print(f'Done. Weights under: {save_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
