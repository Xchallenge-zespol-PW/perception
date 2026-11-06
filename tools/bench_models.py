#!/usr/bin/env python3
"""
Benchmark YOLO models (``.pt`` / ``.onnx`` / ``.engine``) as the node runs them.

Loads each model through the node's loader (no fallback: a model that
cannot run on ``--device`` is reported as failed), warms it up, then
times ``predict`` on real frames (``--frames``) or a blank 640x480 image.
Use it to pick the model size: n / s / m, GPU on the robot vs CPU.

Example::

    python3 tools/bench_models.py models/cubes_pose.onnx --device cpu
    python3 tools/bench_models.py /ws/models/cubes_pose.engine \\
        /ws/models/cubes_pose.onnx --device 0 --frames /ws/data/frames/kostki_x \\
        --csv /ws/results/bench_robot.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from g1_perception.segmenters.yolo_backend import (
    WARMUP_SHAPE,
    load_yolo,
    needs_cuda,
)

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Time YOLO predict per model.')
    parser.add_argument('models', type=Path, nargs='+', help='.pt / .onnx / .engine')
    parser.add_argument(
        '--device',
        default='0',
        help="Device for .pt / .engine (default: '0'); .onnx always runs on cpu",
    )
    parser.add_argument('--frames', type=Path, default=None, help='Dir with color frames')
    parser.add_argument('--runs', type=int, default=50, help='Timed runs (default: 50)')
    parser.add_argument('--warmup', type=int, default=5, help='Warm-up runs (default: 5)')
    parser.add_argument('--imgsz', type=int, default=640, help='predict imgsz (default: 640)')
    parser.add_argument('--csv', type=Path, default=None, help='Append results to CSV')
    return parser.parse_args(argv)


def load_frames(frames_dir: Path | None, limit: int = 50) -> list[np.ndarray]:
    """BGR frames from ``*_color.png`` / images in a dir, or one blank frame."""
    if frames_dir is None:
        return [np.zeros((*WARMUP_SHAPE, 3), dtype=np.uint8)]
    paths = sorted(
        p for p in frames_dir.rglob('*')
        if p.suffix.lower() in IMAGE_SUFFIXES and 'depth' not in p.name
    )[:limit]
    frames = [img for img in (cv2.imread(str(p)) for p in paths) if img is not None]
    if not frames:
        raise FileNotFoundError(f'No images under {frames_dir}')
    return frames


def summarize(times_ms: list[float]) -> dict[str, float]:
    """Mean / p50 / p95 latency and throughput of timed runs."""
    arr = np.asarray(times_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        'mean_ms': mean,
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'fps': 1000.0 / mean if mean > 0 else 0.0,
    }


def device_for(model: Path, device: str) -> str:
    """ONNX is benchmarked on CPU (the fallback path); others on ``device``."""
    return 'cpu' if model.suffix == '.onnx' else device


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = load_frames(args.frames)
    rows: list[dict[str, object]] = []
    failed = 0
    for model_path in args.models:
        device = device_for(model_path, args.device)
        try:
            backend = load_yolo(
                model_path,
                device,
                warmup_runs=args.warmup,
                predict_kwargs={'imgsz': args.imgsz},
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f'FAIL {model_path}: {exc}', file=sys.stderr)
            failed += 1
            continue
        kwargs = {'imgsz': args.imgsz, 'verbose': False, 'device': device}
        times: list[float] = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            backend.model.predict(frames[i % len(frames)], **kwargs)
            times.append((time.perf_counter() - t0) * 1000.0)
        stats = summarize(times)
        rows.append({
            'model': str(model_path),
            'backend': backend.describe(),
            'cuda': needs_cuda(model_path, device),
            **{k: round(v, 2) for k, v in stats.items()},
        })

    if rows:
        print(f"{'model':40s} {'backend':12s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'fps':>6s}")
        for r in rows:
            print(
                f"{str(r['model'])[-40:]:40s} {r['backend']:12s} "
                f"{r['mean_ms']:7.1f} {r['p50_ms']:7.1f} {r['p95_ms']:7.1f} "
                f"{r['fps']:6.1f}"
            )
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.csv.exists()
        with args.csv.open('a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
        print(f'Appended {len(rows)} rows to {args.csv}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
