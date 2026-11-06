#!/usr/bin/env python3
"""
Check the robot container for GPU inference (run inside the humble container).

Checks, each PASS / FAIL: NumPy 1.x, ``cv_bridge`` import, ``ultralytics``,
``onnxruntime`` providers, the CUDA driver (``libcuda``), TensorRT on the
GPU. PyTorch with CUDA is reported as INFO only: the node runs ``.engine``
without torch (``trt_runtime``). With ``--export``, also builds a TensorRT
engine from stock weights (e.g. ``yolo26n-pose.pt``, downloaded by
Ultralytics) without torch and times it — the go / no-go for YOLO26 on
this JetPack before training (fallback: ``yolo11n-pose.pt``).

Example::

    YOLO_AUTOINSTALL=false python3 tools/jetson_gpu_check.py
    YOLO_AUTOINSTALL=false python3 tools/jetson_gpu_check.py \\
        --export yolo26n-pose.pt --workdir /ws/models/check
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check GPU inference readiness.')
    parser.add_argument(
        '--export',
        default=None,
        help='Stock weights to export to .engine and time (e.g. yolo26n-pose.pt)',
    )
    parser.add_argument(
        '--workdir',
        type=Path,
        default=Path('check_engine'),
        help='Where the test engine is written (default: ./check_engine)',
    )
    parser.add_argument('--runs', type=int, default=30, help='Timed runs (default: 30)')
    return parser.parse_args(argv)


def check_numpy() -> str:
    import numpy
    if not numpy.__version__.startswith('1.'):
        raise RuntimeError(f'numpy {numpy.__version__} (cv_bridge needs 1.x)')
    return numpy.__version__


def check_cv_bridge() -> str:
    from cv_bridge import CvBridge
    CvBridge()
    return 'ok'


def check_ultralytics() -> str:
    import ultralytics
    return ultralytics.__version__


def check_onnxruntime() -> str:
    import onnxruntime
    return f'{onnxruntime.__version__} {onnxruntime.get_available_providers()}'


def check_torch_cuda() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f'torch {torch.__version__}: cuda.is_available() is False')
    return f'torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}'


def check_cuda_driver() -> str:
    from g1_perception.segmenters.trt_runtime import cuda_driver_ok
    if not cuda_driver_ok():
        raise RuntimeError('libcuda cuInit / cuDeviceGetCount failed (runtime nvidia, video group?)')
    return 'ok'


def check_tensorrt() -> str:
    import tensorrt
    builder = tensorrt.Builder(tensorrt.Logger())
    return f'{tensorrt.__version__}, fast FP16: {builder.platform_has_fast_fp16}'


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ('numpy<2', check_numpy),
    ('cv_bridge', check_cv_bridge),
    ('ultralytics', check_ultralytics),
    ('onnxruntime', check_onnxruntime),
    ('CUDA driver', check_cuda_driver),
    ('tensorrt (GPU)', check_tensorrt),
]
# Informational: the node does not need CUDA torch for .engine.
INFO_CHECKS: list[tuple[str, Callable[[], str]]] = [
    ('torch + CUDA (optional)', check_torch_cuda),
]


def run_checks(
    checks: list[tuple[str, Callable[[], str]]],
) -> list[tuple[str, bool, str]]:
    """Run each check; return ``(name, ok, detail)`` without raising."""
    results = []
    for name, check in checks:
        try:
            results.append((name, True, check()))
        except Exception as exc:  # any import / runtime error is a FAIL
            results.append((name, False, f'{type(exc).__name__}: {exc}'))
    return results


def export_and_time(weights: str, workdir: Path, runs: int) -> str:
    """Build a FP16 engine from stock weights without torch and time it."""
    import numpy as np
    from ultralytics import YOLO

    from g1_perception.segmenters.trt_runtime import TrtYoloModel, build_engine

    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)  # stock weights download here; the engine lands next to them
    onnx_path = Path(YOLO(weights).export(format='onnx', imgsz=640, device='cpu'))
    engine = build_engine(onnx_path, onnx_path.with_suffix('.engine'), half=True)
    model = TrtYoloModel(engine)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(blank)
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(blank)
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    return f'{engine} — {ms:.1f} ms / frame (incl. pre/post-processing)'


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f'Python {platform.python_version()} on {platform.machine()}')
    results = run_checks(CHECKS)
    if args.export:
        results += run_checks([
            (f'engine export {args.export}',
             lambda: export_and_time(args.export, args.workdir, args.runs)),
        ])
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {detail}")
    for name, _ok, detail in run_checks(INFO_CHECKS):
        print(f"INFO  {name:28s} {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == '__main__':
    sys.exit(main())
