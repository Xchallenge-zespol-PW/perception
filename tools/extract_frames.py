#!/usr/bin/env python3
"""
Extract synchronized RGB-D frames from MCAP bags listed in labels.csv.

Works for cube bags (``/bags/labels.csv``) and ball bags
(``--labels /bags/labels_pilki.csv``). Frame files are named
``<bag>_<frame>_color.png`` / ``<bag>_<frame>_depth.npy`` so a flat upload
(e.g. Roboflow) can still be split by recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'

DEFAULT_BAGS_DIR = Path('/bags')
DEFAULT_LABELS = DEFAULT_BAGS_DIR / 'labels.csv'
DEFAULT_OUT = Path('data')
MAX_DT_NS_DEFAULT = 20_000_000  # 20 ms

INDEX_FIELDNAMES = [
    'bag',
    'frame',
    'color_path',
    'depth_path',
    'ulozenie',
    'swiatlo',
    'yaw_deg',
]
# labels_pilki.csv columns copied to index.csv when present in the labels file
BALL_INDEX_FIELDNAMES = ['liczba_pilek', 'polka']


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Extract every N-th color frame (with nearest depth) from rosbag '
            'MCAPs listed in labels.csv.'
        )
    )
    parser.add_argument(
        '--bags-dir',
        type=Path,
        default=DEFAULT_BAGS_DIR,
        help='Directory containing bag folders (default: /bags)',
    )
    parser.add_argument(
        '--labels',
        type=Path,
        default=DEFAULT_LABELS,
        help=(
            'CSV with columns bag,swiatlo,... (default: /bags/labels.csv; '
            'balls: /bags/labels_pilki.csv)'
        ),
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=DEFAULT_OUT,
        help='Output root for frames/ and index.csv (default: data)',
    )
    parser.add_argument(
        '--every',
        type=int,
        default=10,
        help='Keep every N-th color frame (default: 10)',
    )
    parser.add_argument(
        '--max-dt-ms',
        type=float,
        default=20.0,
        help='Max |color-depth| time difference in ms (default: 20)',
    )
    return parser.parse_args(argv)


def image_to_rgb(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image (rgb8) to HxWx3 uint8."""
    if msg.encoding != 'rgb8':
        raise ValueError(f'Unsupported color encoding: {msg.encoding!r}')
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, 3
    ).copy()


def image_to_depth_m(msg) -> np.ndarray:
    """Convert aligned depth Image (16UC1, millimetres) to float32 metres."""
    if msg.encoding not in ('16UC1', 'mono16'):
        raise ValueError(f'Unsupported depth encoding: {msg.encoding!r}')
    depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(
        msg.height, msg.width
    )
    return depth_mm.astype(np.float32) / 1000.0


def nearest_depth(
    t_color_ns: int,
    depth_times: np.ndarray,
    depth_images: list[np.ndarray],
    max_dt_ns: int,
) -> np.ndarray | None:
    """Return depth image nearest in time, or None if outside ``max_dt_ns``."""
    if depth_times.size == 0:
        return None
    idx = int(np.argmin(np.abs(depth_times - t_color_ns)))
    if abs(int(depth_times[idx]) - t_color_ns) > max_dt_ns:
        return None
    return depth_images[idx]


def read_bag(
    bag_dir: Path,
) -> tuple[
    list[tuple[int, np.ndarray]],
    list[tuple[int, np.ndarray]],
    dict | None,
]:
    """Load color frames, depth frames, and one camera_info from a bag."""
    colors: list[tuple[int, np.ndarray]] = []
    depths: list[tuple[int, np.ndarray]] = []
    camera_info: dict | None = None

    with AnyReader([bag_dir]) as reader:
        for conn, t_ns, raw in reader.messages():
            if conn.topic == COLOR_TOPIC:
                msg = reader.deserialize(raw, conn.msgtype)
                colors.append((t_ns, image_to_rgb(msg)))
            elif conn.topic == DEPTH_TOPIC:
                msg = reader.deserialize(raw, conn.msgtype)
                depths.append((t_ns, image_to_depth_m(msg)))
            elif conn.topic == INFO_TOPIC and camera_info is None:
                msg = reader.deserialize(raw, conn.msgtype)
                k = list(msg.k)
                camera_info = {
                    'width': int(msg.width),
                    'height': int(msg.height),
                    'K': [k[0:3], k[3:6], k[6:9]],
                }

    return colors, depths, camera_info


def frame_stem(bag_name: str, frame_idx: int) -> str:
    """File name stem for one extracted frame, prefixed with the bag name."""
    return f'{bag_name}_{frame_idx}'


def extract_bag(
    bag_name: str,
    bags_dir: Path,
    out_dir: Path,
    every: int,
    max_dt_ns: int,
    ulozenie: str,
    swiatlo: str,
    yaw_deg: str = '',
    extra: dict[str, str] | None = None,
) -> list[dict[str, str | int]]:
    """Extract frames for one bag; return index rows (may be empty)."""
    bag_dir = bags_dir / bag_name
    meta = bag_dir / 'metadata.yaml'
    if not meta.is_file():
        print(
            f'WARNING: skipping {bag_name}: missing metadata.yaml',
            file=sys.stderr,
        )
        return []

    colors, depths, camera_info = read_bag(bag_dir)
    if camera_info is None:
        print(
            f'WARNING: skipping {bag_name}: no {INFO_TOPIC}',
            file=sys.stderr,
        )
        return []

    bag_out = out_dir / 'frames' / bag_name
    bag_out.mkdir(parents=True, exist_ok=True)
    info_path = bag_out / 'camera_info.json'
    info_path.write_text(json.dumps(camera_info, indent=2) + '\n', encoding='utf-8')

    depth_times = np.array([t for t, _ in depths], dtype=np.int64)
    depth_images = [img for _, img in depths]

    rows: list[dict[str, str | int]] = []
    for frame_idx, (t_ns, rgb) in enumerate(colors):
        if frame_idx % every != 0:
            continue
        depth_m = nearest_depth(t_ns, depth_times, depth_images, max_dt_ns)
        if depth_m is None:
            continue

        stem = frame_stem(bag_name, frame_idx)
        color_rel = out_dir / 'frames' / bag_name / f'{stem}_color.png'
        depth_rel = out_dir / 'frames' / bag_name / f'{stem}_depth.npy'

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(color_rel), bgr)
        np.save(str(depth_rel), depth_m)

        rows.append(
            {
                'bag': bag_name,
                'frame': frame_idx,
                'color_path': color_rel.as_posix(),
                'depth_path': depth_rel.as_posix(),
                'ulozenie': ulozenie,
                'swiatlo': swiatlo,
                'yaw_deg': yaw_deg,
                **(extra or {}),
            }
        )

    print(
        f'{bag_name}: saved {len(rows)} frames '
        f'(color={len(colors)}, depth={len(depths)}, every={every})'
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.every < 1:
        print('ERROR: --every must be >= 1', file=sys.stderr)
        return 2

    max_dt_ns = int(args.max_dt_ms * 1_000_000)
    out_dir: Path = args.out_dir
    (out_dir / 'frames').mkdir(parents=True, exist_ok=True)

    with args.labels.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        labels = list(reader)
        label_columns = reader.fieldnames or []
    extra_fields = [c for c in BALL_INDEX_FIELDNAMES if c in label_columns]

    all_rows: list[dict[str, str | int]] = []
    for row in labels:
        bag_name = row['bag']
        all_rows.extend(
            extract_bag(
                bag_name=bag_name,
                bags_dir=args.bags_dir,
                out_dir=out_dir,
                every=args.every,
                max_dt_ns=max_dt_ns,
                ulozenie=row.get('ulozenie', ''),
                swiatlo=row.get('swiatlo', ''),
                yaw_deg=row.get('yaw_deg', '') or '',
                extra={c: row.get(c, '') or '' for c in extra_fields},
            )
        )

    index_path = out_dir / 'index.csv'
    fieldnames = INDEX_FIELDNAMES + extra_fields
    with index_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'Wrote {len(all_rows)} rows to {index_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
