#!/usr/bin/env python3
"""
Export trained YOLO weights for the node: ``.onnx`` (CPU) and ``.engine`` (GPU).

Copies the results under one name to ``--out`` (default ``models/``), as
``params_robot.yaml`` expects: ``cubes_seg.{pt,onnx,engine}`` or
``cubes_pose.{pt,onnx,engine}``. The ``.pt`` is copied too — the TensorRT
engine is built from it **on the Jetson** (an engine from another machine
or TensorRT version does not load there).

Example::

    # laptop / Colab: ONNX for the CPU fallback + .pt for the robot
    python3 tools/export_model.py runs/pose/cubes_pose/weights/best.pt --name cubes_pose
    # on the Jetson (inside the container), next to the deployed .pt:
    python3 /ws/src/perception/tools/export_model.py /ws/models/cubes_pose.pt \\
        --formats engine --out /ws/models
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

FORMAT_SUFFIX = {'onnx': '.onnx', 'engine': '.engine'}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export YOLO .pt to .onnx / .engine under a fixed name.'
    )
    parser.add_argument('weights', type=Path, help='Trained .pt (e.g. best.pt)')
    parser.add_argument(
        '--formats',
        default='onnx',
        help='Comma list of onnx, engine (default: onnx)',
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Output base name (default: weights stem), e.g. cubes_pose',
    )
    parser.add_argument('--out', type=Path, default=Path('models'), help='Output dir')
    parser.add_argument('--imgsz', type=int, default=640, help='Export size (default: 640)')
    parser.add_argument(
        '--device',
        default=None,
        help="Export device (default: '0' for engine, 'cpu' for onnx)",
    )
    parser.add_argument(
        '--no-half',
        action='store_true',
        help='Build the engine in FP32 instead of FP16',
    )
    return parser.parse_args(argv)


def parse_formats(text: str) -> list[str]:
    """Validate a comma list of export formats."""
    formats = [f.strip() for f in text.split(',') if f.strip()]
    unknown = [f for f in formats if f not in FORMAT_SUFFIX]
    if unknown or not formats:
        raise ValueError(
            f'Unknown formats {unknown or text!r}; use {sorted(FORMAT_SUFFIX)}'
        )
    return formats


def target_path(out_dir: Path, name: str, fmt: str) -> Path:
    """``<out>/<name>.<ext>`` for an export format or ``pt``."""
    suffix = '.pt' if fmt == 'pt' else FORMAT_SUFFIX[fmt]
    return out_dir / f'{name}{suffix}'


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.weights.suffix != '.pt' or not args.weights.is_file():
        print(f'ERROR: expected an existing .pt file: {args.weights}', file=sys.stderr)
        return 1
    try:
        formats = parse_formats(args.formats)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    if 'engine' in formats and platform.machine() != 'aarch64':
        print(
            'WARN: building .engine off the Jetson — it will not load on the robot',
            file=sys.stderr,
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f'Missing ultralytics: {exc}') from exc

    name = args.name or args.weights.stem
    args.out.mkdir(parents=True, exist_ok=True)
    pt_target = target_path(args.out, name, 'pt')
    if pt_target.resolve() != args.weights.resolve():
        shutil.copy2(args.weights, pt_target)
        print(f'Copied {args.weights} -> {pt_target}')

    onnx_target = target_path(args.out, name, 'onnx')
    for fmt in formats:
        kwargs = {'format': fmt, 'imgsz': args.imgsz}
        backup = None
        if fmt == 'engine':
            kwargs['half'] = not args.no_half
            kwargs['device'] = args.device or '0'
            # The engine export writes an intermediate <name>.onnx next to
            # the .pt; keep the deployed CPU-fallback ONNX intact.
            if onnx_target.is_file():
                backup = onnx_target.with_suffix('.onnx.bak')
                shutil.move(str(onnx_target), backup)
        else:
            kwargs['device'] = args.device or 'cpu'
        try:
            exported = Path(YOLO(str(pt_target)).export(**kwargs))
        finally:
            if backup is not None:
                shutil.move(str(backup), onnx_target)
        target = target_path(args.out, name, fmt)
        if exported.resolve() != target.resolve():
            shutil.move(str(exported), target)
        print(f'Exported {fmt}: {target}')

    print(
        f'Deploy: ./tools/deploy_robot.sh {pt_target} '
        f"{target_path(args.out, name, 'onnx')}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
