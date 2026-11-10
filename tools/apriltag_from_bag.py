#!/usr/bin/env python3
"""Detect AprilTags in recorded bags and write their poses to one CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores

from g1_perception.core.apriltag import AprilTagDetector, parse_tag_ids
from g1_perception.core.frame_utils import intrinsics_from_k
from g1_perception.core.geometry import transform_pose
from g1_perception.core.tf_chain import TfChain

IMAGE_TOPIC = '/camera/camera/color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'
TF_TOPICS = ('/tf', '/tf_static')
COLUMNS = [
    'bag', 'stamp_s', 'tag_id', 'frame', 'x', 'y', 'z',
    'qx', 'qy', 'qz', 'qw', 'distance_m', 'reproj_px', 'ambiguity',
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Detect AprilTags in every N-th colour frame of the bags and write '
            'one CSV row per tag, frame and output TF frame.'
        )
    )
    parser.add_argument('bags', nargs='+', type=Path, help='Bag directories')
    parser.add_argument('--out', type=Path, default=Path('data/apriltags.csv'))
    parser.add_argument('--tag-size', type=float, default=0.10,
                        help='Side of the black square in metres (default 0.10)')
    parser.add_argument('--family', default='36h11')
    parser.add_argument('--ids', default='', help='Keep only these ids, e.g. 0,1,2')
    parser.add_argument('--frames', default='',
                        help='Also write poses in these TF frames, e.g. '
                             'pelvis,left_ankle_roll_link (needs /tf in the bag)')
    parser.add_argument('--every', type=int, default=1,
                        help='Use every N-th colour frame (default 1)')
    parser.add_argument('--max-reproj-px', type=float, default=2.0)
    parser.add_argument('--max-tf-dt-ms', type=float, default=50.0,
                        help='Nearest TF sample must be this close (default 50)')
    parser.add_argument('--image-topic', default=IMAGE_TOPIC)
    parser.add_argument('--info-topic', default=INFO_TOPIC)
    return parser.parse_args(argv)


def image_to_rgb(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image (rgb8 / bgr8 / mono8) to uint8 RGB or gray."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == 'mono8':
        return data.reshape(msg.height, msg.width).copy()
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Unsupported color encoding: {msg.encoding!r}')
    img = data.reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if msg.encoding == 'bgr8' else img.copy()


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def read_tf_and_info(reader: AnyReader, info_topic: str, max_dt_s: float):
    """First pass: the TF tree and one camera_info (K, D)."""
    tf = TfChain(max_dt_s=max_dt_s)
    K = D = None
    conns = [c for c in reader.connections if c.topic in (*TF_TOPICS, info_topic)]
    for conn, _t, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        if conn.topic == info_topic:
            if K is None:
                K = intrinsics_from_k(msg.k)
                D = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else None
            continue
        for t in msg.transforms:
            tr, q = t.transform.translation, t.transform.rotation
            tf.add(
                t.header.frame_id, t.child_frame_id, stamp_s(t.header),
                (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w),
                static=conn.topic == '/tf_static',
            )
    return tf, K, D


def process_bag(bag: Path, args, detector, frames, writer) -> tuple[int, int]:
    """Write the rows of one bag; return ``(frames used, tags seen)``."""
    n_frames = n_tags = 0
    # sqlite3 bags carry no message definitions; MCAP ones do.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag], default_typestore=typestore) as reader:
        tf, K, D = read_tf_and_info(reader, args.info_topic, args.max_tf_dt_ms / 1000)
        if K is None:
            print(f'WARNING: skipping {bag.name}: no {args.info_topic}', file=sys.stderr)
            return 0, 0
        conns = [c for c in reader.connections if c.topic == args.image_topic]
        for i, (conn, _t, raw) in enumerate(reader.messages(connections=conns)):
            if i % args.every:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = stamp_s(msg.header)
            n_frames += 1
            for tag in detector.detect(image_to_rgb(msg), K, D):
                if tag.pose is None:
                    continue
                n_tags += 1
                poses = [(msg.header.frame_id, tag.pose)]
                for frame in frames:
                    try:
                        poses.append((frame, transform_pose(
                            tag.pose, *tf.lookup(frame, msg.header.frame_id, stamp)
                        )))
                    except (LookupError, KeyError) as exc:
                        print(f'WARNING: {bag.name} {stamp:.3f}: {exc}',
                              file=sys.stderr)
                for frame, pose in poses:
                    writer.writerow([
                        bag.name, f'{stamp:.6f}', tag.tag_id, frame,
                        *(f'{v:.5f}' for v in pose.position),
                        *(f'{v:.6f}' for v in pose.quat),
                        f'{float(np.linalg.norm(tag.pose.position)):.4f}',
                        f'{tag.reproj_px:.3f}', f'{tag.ambiguity:.3f}',
                    ])
    return n_frames, n_tags


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    detector = AprilTagDetector(
        family=args.family,
        tag_size=args.tag_size,
        tag_ids=parse_tag_ids(args.ids),
        max_reproj_px=args.max_reproj_px,
    )
    frames = [f.strip() for f in args.frames.split(',') if f.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for bag in args.bags:
            n_frames, n_tags = process_bag(bag, args, detector, frames, writer)
            print(f'{bag.name}: frames={n_frames} tags={n_tags}')
    print(f'Wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
