#!/usr/bin/env python3
"""
Check a TensorRT ``.engine`` against its ``.onnx`` on the same frames.

The engine runs on the node's torch-free backend (``TrtYoloModel``, GPU),
the ONNX through Ultralytics on CPU as the reference. Detections are
matched by IoU; the report gives count mismatches, the largest box /
keypoint difference [px], confidence difference and the smallest mask IoU,
plus mean latency of both. Exit code 1 when a tolerance is exceeded (FP16
engines differ slightly from FP32 ONNX).

Example (robot, inside the container)::

    python3 /ws/src/perception/tools/compare_backends.py \\
        /ws/models/cubes_pose.engine /ws/models/cubes_pose.onnx \\
        --frames /ws/data/frames
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare .engine vs .onnx outputs.')
    parser.add_argument('engine', type=Path, help='TensorRT engine')
    parser.add_argument('onnx', type=Path, help='Reference ONNX of the same model')
    parser.add_argument('--frames', type=Path, default=None, help='Dir with images')
    parser.add_argument('--limit', type=int, default=20, help='Max frames (default: 20)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--max-box-px', type=float, default=3.0, help='Box tolerance')
    parser.add_argument('--max-kpt-px', type=float, default=3.0, help='Keypoint tolerance')
    parser.add_argument('--min-mask-iou', type=float, default=0.9, help='Mask tolerance')
    return parser.parse_args(argv)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _np(x: Any) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def compare_results(ref: Any, test: Any) -> dict[str, float]:
    """
    Compare two Ultralytics-like results for one image.

    Each reference detection is matched to the test detection with the
    highest box IoU. Returns ``count_diff``, ``box_px``, ``conf``,
    ``kpt_px`` (pose) and ``mask_iou`` (segment; 1.0 when absent).
    """
    rb, tb = _np(ref.boxes.xyxy), _np(test.boxes.xyxy)
    out = {
        'count_diff': float(abs(len(rb) - len(tb))),
        'box_px': 0.0, 'conf': 0.0, 'kpt_px': 0.0, 'mask_iou': 1.0,
    }
    if not len(rb) or not len(tb):
        return out
    rconf, tconf = _np(ref.boxes.conf), _np(test.boxes.conf)
    for i in range(len(rb)):
        j = max(range(len(tb)), key=lambda k: box_iou(rb[i], tb[k]))
        out['box_px'] = max(out['box_px'], float(np.abs(rb[i] - tb[j]).max()))
        out['conf'] = max(out['conf'], abs(float(rconf[i]) - float(tconf[j])))
        if getattr(ref, 'keypoints', None) is not None and test.keypoints is not None:
            rk = _np(ref.keypoints.data)[i][..., :2]
            tk = _np(test.keypoints.data)[j][..., :2]
            out['kpt_px'] = max(out['kpt_px'], float(np.abs(rk - tk).max()))
        if getattr(ref, 'masks', None) is not None and test.masks is not None:
            rm = _np(ref.masks.data)[i] > 0.5
            tm = _np(test.masks.data)[j] > 0.5
            union = float((rm | tm).sum())
            iou = float((rm & tm).sum()) / union if union else 1.0
            out['mask_iou'] = min(out['mask_iou'], iou)
    return out


def load_images(frames: Path | None, limit: int) -> list[np.ndarray]:
    """Color frames from a dir (recursive), or Ultralytics' sample images."""
    if frames is None:
        from ultralytics.utils import ASSETS

        paths = [ASSETS / 'bus.jpg', ASSETS / 'zidane.jpg']
    else:
        paths = sorted(
            p for p in frames.rglob('*')
            if p.suffix.lower() in IMAGE_SUFFIXES and 'depth' not in p.name
        )[:limit]
    images = [img for img in (cv2.imread(str(p)) for p in paths) if img is not None]
    if not images:
        raise FileNotFoundError(f'No images found ({frames})')
    return images


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO

    from g1_perception.segmenters.trt_runtime import TrtYoloModel

    engine = TrtYoloModel(args.engine)
    ref_model = YOLO(str(args.onnx), task=engine.task)
    images = load_images(args.frames, args.limit)

    worst = {'count_diff': 0.0, 'box_px': 0.0, 'conf': 0.0, 'kpt_px': 0.0, 'mask_iou': 1.0}
    t_engine: list[float] = []
    t_ref: list[float] = []
    for img in images:
        t0 = time.perf_counter()
        ref = ref_model.predict(
            img, conf=args.conf, retina_masks=True, verbose=False, device='cpu'
        )[0]
        t_ref.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        test = engine.predict(img, conf=args.conf)[0]
        t_engine.append((time.perf_counter() - t0) * 1000.0)
        for key, value in compare_results(ref, test).items():
            worst[key] = min(worst[key], value) if key == 'mask_iou' else max(worst[key], value)

    print(f'{len(images)} images, task={engine.task}')
    print(f'  engine {np.mean(t_engine[1:] or t_engine):.1f} ms, onnx cpu {np.mean(t_ref):.1f} ms')
    print('  ' + ', '.join(f'{k}={v:.3f}' for k, v in worst.items()))
    ok = (
        worst['count_diff'] == 0 and worst['box_px'] <= args.max_box_px
        and worst['kpt_px'] <= args.max_kpt_px and worst['mask_iou'] >= args.min_mask_iou
    )
    print('OK' if ok else 'MISMATCH (check tolerances / FP16 near the conf threshold)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
