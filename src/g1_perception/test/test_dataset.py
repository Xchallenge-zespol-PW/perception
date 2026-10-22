"""Tests for offline frame loading from extracted dataset rows."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from g1_perception.core.dataset import load_frame


def test_load_frame_reads_rgb_depth_and_intrinsics(tmp_path: Path):
    bag_dir = tmp_path / 'data' / 'frames' / 'demo_bag'
    bag_dir.mkdir(parents=True)

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[1, 2] = (10, 20, 30)
    depth_m = np.zeros((4, 6), dtype=np.float32)
    depth_m[1, 2] = 1.25
    K = [[100.0, 0.0, 3.0], [0.0, 110.0, 2.0], [0.0, 0.0, 1.0]]

    cv2.imwrite(
        str(bag_dir / '10_color.png'),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    np.save(str(bag_dir / '10_depth.npy'), depth_m)
    (bag_dir / 'camera_info.json').write_text(
        json.dumps({'width': 6, 'height': 4, 'K': K}) + '\n',
        encoding='utf-8',
    )

    row = {
        'bag': 'demo_bag',
        'frame': '10',
        'color_path': 'data/frames/demo_bag/10_color.png',
        'depth_path': 'data/frames/demo_bag/10_depth.npy',
        'ulozenie': 'RGGR',
        'swiatlo': 'dzienne',
    }

    frame = load_frame(row, root=tmp_path)

    assert frame.rgb.shape == (4, 6, 3)
    assert frame.rgb.dtype == np.uint8
    assert tuple(frame.rgb[1, 2]) == (10, 20, 30)
    assert frame.depth_m.dtype == np.float32
    assert frame.depth_m[1, 2] == np.float32(1.25)
    assert np.allclose(frame.K, K)
    assert frame.stamp == 10.0
