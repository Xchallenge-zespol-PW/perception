"""Offline dataset helpers for extracted RGB-D frames."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from g1_perception.core.types import Frame


def load_frame(row: Mapping[str, str], *, root: Path | None = None) -> Frame:
    """
    Load a :class:`Frame` from one ``data/index.csv`` row.

    Expects ``color_path`` and ``depth_path`` relative to ``root`` (default:
    current working directory). Intrinsics come from ``camera_info.json``
    next to the color image. Optional ``frame`` becomes ``Frame.stamp``.
    """
    base = Path.cwd() if root is None else Path(root)
    color_path = base / row['color_path']
    depth_path = base / row['depth_path']
    info_path = color_path.parent / 'camera_info.json'

    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Cannot read color image: {color_path}')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_m = np.load(str(depth_path))
    if depth_m.dtype != np.float32:
        depth_m = depth_m.astype(np.float32)

    with info_path.open(encoding='utf-8') as f:
        info = json.load(f)
    K = np.asarray(info['K'], dtype=np.float64)

    stamp_raw = row.get('frame', 0)
    stamp = float(stamp_raw) if stamp_raw != '' else 0.0

    return Frame(rgb=rgb, depth_m=depth_m, K=K, stamp=stamp)
