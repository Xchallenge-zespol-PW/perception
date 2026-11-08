"""
Backend-independent YOLO pre/post-processing (numpy + OpenCV, no torch).

Mirrors Ultralytics for exported (non-end2end) detect / segment / pose
models: static letterbox, confidence filter, per-class NMS, box / keypoint
rescaling and prototype masks at full image size (``retina_masks``).
Results expose the attributes the segmenters read from Ultralytics
results (``boxes.conf``, ``boxes.cls``, ``masks.data``, ``keypoints.data``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

PAD_VALUE = 114
TASKS = ('detect', 'segment', 'pose')


@dataclass(frozen=True)
class Letterbox:
    """Scale and padding that map input-tensor pixels back to the image."""

    gain: float
    pad_x: float
    pad_y: float
    shape: tuple[int, int]  # original (H, W)
    size: tuple[int, int]  # network input (H, W)

    def to_image(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(..., 2)`` input-tensor coordinates to image pixels."""
        out = np.array(xy, dtype=np.float64, copy=True)
        out[..., 0] = (out[..., 0] - self.pad_x) / self.gain
        out[..., 1] = (out[..., 1] - self.pad_y) / self.gain
        return out


def letterbox(
    bgr: NDArray[np.uint8],
    size: tuple[int, int] = (640, 640),
) -> tuple[NDArray[np.float32], Letterbox]:
    """
    Resize keeping aspect ratio, pad to ``size``, return a 1x3xHxW RGB blob.

    Same as Ultralytics ``LetterBox(auto=False, center=True)`` with pad 114
    and ``INTER_LINEAR``; values in [0, 1].
    """
    h0, w0 = bgr.shape[:2]
    h, w = size
    gain = min(h / h0, w / w0)
    new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))
    dw, dh = (w - new_w) / 2.0, (h - new_h) / 2.0
    img = bgr
    if (w0, h0) != (new_w, new_h):
        img = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    info = Letterbox(
        gain=float(gain), pad_x=float(left), pad_y=float(top),
        shape=(h0, w0), size=(h, w),
    )
    return np.ascontiguousarray(blob), info


def nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    iou_thres: float,
) -> NDArray[np.int64]:
    """Greedy NMS on ``(N, 4)`` xyxy boxes; return kept indices by score."""
    order = np.argsort(-scores, kind='stable')
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


class _Boxes:
    def __init__(self, xyxy: NDArray, conf: NDArray, cls: NDArray) -> None:
        self.xyxy = xyxy.astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    def __len__(self) -> int:
        return int(self.conf.shape[0])


class _Masks:
    def __init__(self, data: NDArray) -> None:
        self.data = data.astype(np.float32)


class _Keypoints:
    def __init__(self, data: NDArray) -> None:
        self.data = data.astype(np.float32)
        self.xy = self.data[..., :2]
        self.conf = self.data[..., 2] if self.data.shape[-1] == 3 else None


@dataclass
class YoloResult:
    """Ultralytics-like result for one image (numpy arrays)."""

    names: dict[int, str]
    boxes: _Boxes
    masks: _Masks | None = None
    keypoints: _Keypoints | None = None


def decode(
    outputs: Sequence[NDArray],
    info: Letterbox,
    *,
    task: str,
    names: Mapping[int, str],
    kpt_shape: Sequence[int] | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
) -> YoloResult:
    """
    Decode raw exported YOLO outputs for one image.

    ``outputs[0]`` is ``(1, 4 + nc + extra, N)``: ``cx, cy, w, h`` in input
    pixels, ``nc`` class scores (already sigmoid), then ``kx, ky, kconf``
    per keypoint (pose) or mask coefficients (segment). Segment also needs
    ``outputs[1]`` prototypes ``(1, nm, mh, mw)``. NMS is per class.
    """
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}, got {task!r}')
    names = {int(k): str(v) for k, v in names.items()}
    nc = len(names)
    pred = np.asarray(outputs[0], dtype=np.float32)[0].T  # (N, C)
    scores_all = pred[:, 4:4 + nc]
    cls = scores_all.argmax(axis=1)
    score = scores_all[np.arange(pred.shape[0]), cls]
    sel = score > conf
    pred, cls, score = pred[sel], cls[sel], score[sel]

    cxcywh = pred[:, :4].astype(np.float64)
    xyxy = np.column_stack((
        cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
        cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2,
    ))
    # Per-class NMS via a large per-class offset (as Ultralytics does).
    offset = cls[:, None].astype(np.float64) * 7680.0
    keep = nms(xyxy + offset, score.astype(np.float64), iou)[:max_det]
    pred, cls, score, xyxy = pred[keep], cls[keep], score[keep], xyxy[keep]

    h0, w0 = info.shape
    boxes_img = info.to_image(xyxy.reshape(-1, 2, 2)).reshape(-1, 4)
    boxes_img[:, [0, 2]] = boxes_img[:, [0, 2]].clip(0, w0)
    boxes_img[:, [1, 3]] = boxes_img[:, [1, 3]].clip(0, h0)
    result = YoloResult(names=names, boxes=_Boxes(boxes_img, score, cls))
    extra = pred[:, 4 + nc:]

    if task == 'pose':
        k, dims = (int(kpt_shape[0]), int(kpt_shape[1])) if kpt_shape else (0, 3)
        kpts = extra[:, :k * dims].reshape(-1, k, dims).astype(np.float64)
        kpts[..., :2] = info.to_image(kpts[..., :2])
        kpts[..., 0] = kpts[..., 0].clip(0, w0)
        kpts[..., 1] = kpts[..., 1].clip(0, h0)
        result.keypoints = _Keypoints(kpts)
    elif task == 'segment':
        proto = np.asarray(outputs[1], dtype=np.float32)[0]
        result.masks = _Masks(_masks(extra, proto, boxes_img, info))
    return result


def _masks(
    coefs: NDArray[np.float32],
    proto: NDArray[np.float32],
    boxes_img: NDArray[np.float64],
    info: Letterbox,
) -> NDArray[np.float32]:
    """Full-resolution (n, H, W) 0/1 masks, cropped to their boxes."""
    h0, w0 = info.shape
    nm, mh, mw = proto.shape
    if coefs.shape[0] == 0:
        return np.zeros((0, h0, w0), dtype=np.float32)
    logits = (coefs[:, :nm] @ proto.reshape(nm, -1)).reshape(-1, mh, mw)
    # Drop the letterbox padding in prototype scale, then resize to image.
    sy, sx = mh / info.size[0], mw / info.size[1]
    top, left = int(round(info.pad_y * sy - 0.1)), int(round(info.pad_x * sx - 0.1))
    bottom = int(round(mh - info.pad_y * sy + 0.1))
    right = int(round(mw - info.pad_x * sx + 0.1))
    out = np.zeros((logits.shape[0], h0, w0), dtype=np.float32)
    ys = np.arange(h0)[:, None]
    xs = np.arange(w0)[None, :]
    for i, m in enumerate(logits):
        full = cv2.resize(
            m[top:bottom, left:right], (w0, h0), interpolation=cv2.INTER_LINEAR
        )
        x1, y1, x2, y2 = boxes_img[i]
        inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        out[i] = ((full > 0.0) & inside).astype(np.float32)
    return out
