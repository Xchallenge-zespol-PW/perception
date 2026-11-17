#!/usr/bin/env python3
"""
Train a YOLO seg or pose model on the cube datasets (Ultralytics).

``--task seg`` (default) trains on ``datasets/cubes_seg``, ``--task pose``
on ``datasets/cubes_pose`` (4 top-face corners ``c0..c3``; see ``docs/training.md``).
Colour is the class, so hue augmentation is off (``hsv_h=0``); pose also
trains without horizontal flips (``fliplr=0``) so corner order stays
canonical. Dataset paths are resolved to absolute ones before training
(``<project>/<name>_data.yaml``), so the run does not depend on cwd or
Ultralytics' ``datasets_dir``.

Example::

    python3 tools/train_yolo.py
    python3 tools/train_yolo.py --task pose --model yolo26s-pose.pt --device 0
    python3 tools/train_yolo.py --data datasets/cubes_seg/data.yaml \\
        --model yolo26n-seg.pt --project runs/seg --name cubes --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    'seg': {
        'data': Path('datasets/cubes_seg/data.yaml'),
        'model': 'yolo26n-seg.pt',
        'project': Path('runs/seg'),
        'name': 'cubes',
        'fliplr': 0.5,
    },
    'pose': {
        'data': Path('datasets/cubes_pose/data.yaml'),
        'model': 'yolo26n-pose.pt',
        'project': Path('runs/pose'),
        'name': 'cubes_pose',
        'fliplr': 0.0,
    },
}
POSE_KPT_SHAPE = [4, 3]
SPLIT_KEYS = ('train', 'val', 'test')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train YOLO seg / pose on the cube datasets (Ultralytics).'
    )
    parser.add_argument(
        '--task',
        choices=sorted(TASK_DEFAULTS),
        default='seg',
        help='seg (default) or pose; sets the defaults below',
    )
    parser.add_argument('--data', type=Path, default=None, help='Dataset YAML')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Base weights (default: yolo26n-seg.pt / yolo26n-pose.pt)',
    )
    parser.add_argument('--epochs', type=int, default=100, help='Epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help="Device, e.g. '' (auto), 'cpu', '0', '0,1'",
    )
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--project', type=Path, default=None, help='Ultralytics project dir')
    parser.add_argument('--name', type=str, default=None, help='Run name under project')
    parser.add_argument('--patience', type=int, default=30, help='Early-stopping patience')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default: 0)')
    parser.add_argument(
        '--fliplr',
        type=float,
        default=None,
        help='Horizontal flip probability (default: seg 0.5, pose 0.0)',
    )
    parser.add_argument(
        '--hsv-h',
        type=float,
        default=0.0,
        help='Hue augmentation (default: 0.0 — colour is the class)',
    )
    parser.add_argument('--hsv-s', type=float, default=0.7, help='Saturation gain (default: 0.7)')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='Value gain (default: 0.4)')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation degrees (default: 0)')
    parser.add_argument('--translate', type=float, default=0.1, help='Translation fraction (default: 0.1)')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale gain (default: 0.5)')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear degrees (default: 0)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspective fraction (default: 0)')
    parser.add_argument('--flipud', type=float, default=0.0, help='Vertical flip probability (default: 0)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic probability (default: 1)')
    parser.add_argument('--close-mosaic', type=int, default=10, help='Epochs without mosaic at the end (default: 10)')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup probability (default: 0)')
    parser.add_argument('--copy-paste', type=float, default=0.0, help='Copy-paste probability (default: 0)')
    parser.add_argument('--exist-ok', action='store_true', help='Reuse the run dir')
    args = parser.parse_args(argv)

    defaults = TASK_DEFAULTS[args.task]
    for key in ('data', 'model', 'project', 'name', 'fliplr'):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    return args


def resolve_data_yaml(data_path: Path) -> dict[str, Any]:
    """
    Load a dataset YAML with absolute ``path`` / split directories.

    Relative ``path`` is taken against the YAML's directory. A split like
    ``../train/images`` (Roboflow export) falls back to ``train/images``
    when only that exists, matching Ultralytics. Missing ``test`` is
    dropped; missing ``train`` / ``val`` raise ``FileNotFoundError``.
    """
    data = yaml.safe_load(data_path.read_text(encoding='utf-8')) or {}
    root = Path(data.get('path') or '.')
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    data['path'] = str(root)

    for key in SPLIT_KEYS:
        value = data.get(key)
        if value is None:
            continue
        split = Path(value)
        if not split.is_absolute():
            candidate = (root / split).resolve()
            if not candidate.exists() and value.startswith('../'):
                candidate = (root / value[3:]).resolve()
            split = candidate
        if not split.exists():
            if key == 'test':
                del data[key]
                continue
            raise FileNotFoundError(f'{key} split not found: {split} (from {data_path})')
        data[key] = str(split)
    for key in ('train', 'val'):
        if key not in data:
            raise FileNotFoundError(f'{data_path} has no {key!r} split')
    return data


def check_task_data(task: str, data: dict[str, Any]) -> None:
    """Fail early when a pose run gets a dataset without 4 corners."""
    if task == 'pose' and list(data.get('kpt_shape') or []) != POSE_KPT_SHAPE:
        raise ValueError(
            f"pose needs kpt_shape {POSE_KPT_SHAPE}, got {data.get('kpt_shape')!r}"
        )
    if task == 'seg' and 'kpt_shape' in data:
        raise ValueError('seg run got a pose dataset (kpt_shape set); use --task pose')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        print(
            f'ERROR: dataset YAML not found: {args.data}\n'
            '  Build it with tools/cvat_to_yolo.py (or tools/autolabel.py for seg).',
            file=sys.stderr,
        )
        return 1
    try:
        data = resolve_data_yaml(args.data)
        check_task_data(args.task, data)
    except (FileNotFoundError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    if args.task == 'pose' and args.fliplr > 0.0:
        print(
            'WARN: fliplr > 0 breaks the c0 rule after flips (docs/training.md)',
            file=sys.stderr,
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            'Missing ultralytics. Install with:\n'
            '  pip install -c deploy/constraints.txt ultralytics\n'
            f'Original error: {exc}'
        ) from exc

    args.project.mkdir(parents=True, exist_ok=True)
    resolved = (args.project / f'{args.name}_data.yaml').resolve()
    with resolved.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    train_kwargs = {
        'data': str(resolved),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'project': str(args.project.resolve()),
        'name': args.name,
        'patience': args.patience,
        'seed': args.seed,
        'exist_ok': args.exist_ok,
        'fliplr': args.fliplr,
        'flipud': args.flipud,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'mosaic': args.mosaic,
        'close_mosaic': args.close_mosaic,
        'mixup': args.mixup,
        'copy_paste': args.copy_paste,
    }
    if args.device != '':
        train_kwargs['device'] = args.device

    print(f'Training {args.model} ({args.task}) on {resolved}')
    model = YOLO(args.model)
    results = model.train(**train_kwargs)
    save_dir = getattr(results, 'save_dir', args.project / args.name)
    print(
        f'Done. Weights under: {save_dir}\n'
        f'  Export: python3 tools/export_model.py {save_dir}/weights/best.pt '
        f'--name cubes_{args.task}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
