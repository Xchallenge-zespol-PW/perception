"""Ball centre from a YOLO mask and aligned depth (no ROS)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from g1_perception.core.types import Detection
from g1_perception.core.visibility import mask_touches_border

DEFAULT_BALL_LABELS = ('red ball',)
# Depth is read in the inner disk of the mask (this fraction of its radius):
# the rim mixes the ball with the shelf or the background behind it.
INNER_RADIUS_FRAC = 0.5
MIN_DEPTH_PX = 10


@dataclass(frozen=True)
class BallObservation:
    """
    One ball seen on a frame, in the camera optical frame.

    ``position`` is the ball centre (x, y, z) in metres, ``radius_m`` its
    radius measured from the mask size and depth (no fixed ball size).
    ``pixel`` / ``radius_px`` are the mask centroid and equal-area radius.
    ``cut`` is True when the mask touches the image border: centre and
    radius are then wrong, the ball is shown but not published.
    """

    position: NDArray[np.float64]
    radius_m: float
    score: float
    pixel: tuple[float, float]
    radius_px: float
    n_depth: int
    cut: bool
    detection: Detection

    @property
    def distance_m(self) -> float:
        """Distance from the camera to the ball centre."""
        return float(np.linalg.norm(self.position))


def is_ball(label: str, names: Sequence[str] = DEFAULT_BALL_LABELS) -> bool:
    """Return True when the raw class ``label`` is one of ``names``."""
    key = label.strip().lower()
    return any(key == n.strip().lower() for n in names)


def ball_from_mask(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, tuple[float, float], float, int] | None:
    """
    Estimate a ball centre from its mask; None without enough depth.

    The mask centroid gives the centre ray, its equal-area radius ``r_px``
    the size. The median depth of the inner disk is the front surface
    ``z_s``; the centre lies ``r`` behind it and ``r = r_px * z / f``, so
    ``r = r_px * z_s / (f - r_px)``. Returns ``(position, radius_m,
    (u, v), r_px, n_depth)``.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    u = float(xs.mean())
    v = float(ys.mean())
    r_px = float(np.sqrt(ys.size / np.pi))

    inner = (xs - u) ** 2 + (ys - v) ** 2 <= (INNER_RADIUS_FRAC * r_px) ** 2
    z = depth_m[ys[inner], xs[inner]]
    z = z[z > 0.0]
    if z.size < MIN_DEPTH_PX:
        z = depth_m[ys, xs]
        z = z[z > 0.0]
    if z.size < MIN_DEPTH_PX:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    f = 0.5 * (fx + fy)
    if r_px >= f:
        return None
    z_s = float(np.median(z))
    radius = r_px * z_s / (f - r_px)
    z_c = z_s + radius
    position = np.array([
        (u - float(K[0, 2])) * z_c / fx,
        (v - float(K[1, 2])) * z_c / fy,
        z_c,
    ])
    return position, radius, (u, v), r_px, int(z.size)


def find_balls(
    detections: Sequence[Detection],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
    *,
    labels: Sequence[str] = DEFAULT_BALL_LABELS,
    border_px: int = 0,
) -> list[BallObservation]:
    """Return balls among ``detections``, nearest first."""
    out: list[BallObservation] = []
    for det in detections:
        if not is_ball(det.label, labels):
            continue
        est = ball_from_mask(det.mask, depth_m, K)
        if est is None:
            continue
        position, radius, pixel, r_px, n_depth = est
        out.append(
            BallObservation(
                position=position,
                radius_m=radius,
                score=float(det.score),
                pixel=pixel,
                radius_px=r_px,
                n_depth=n_depth,
                cut=mask_touches_border(det.mask, border_px),
                detection=det,
            )
        )
    out.sort(key=lambda b: b.distance_m)
    return out


def format_balls(balls: Sequence[BallObservation]) -> str:
    """Return ``"1: 1.23 m r=3.4 cm, 2: ... (cut)"`` for logs."""
    return ', '.join(
        f'{i}: {b.distance_m:.2f} m r={b.radius_m * 100:.1f} cm'
        + (' (cut)' if b.cut else '')
        for i, b in enumerate(balls, start=1)
    ) or 'none'


def draw_balls(
    rgb: NDArray[np.uint8],
    balls: Sequence[BallObservation],
    *,
    inference_ms: float | None = None,
    backend: str | None = None,
) -> NDArray[np.uint8]:
    """Render masks, centre circles and ``K d=.. r=..`` labels on ``rgb``."""
    out = rgb.copy()
    for i, b in enumerate(balls, start=1):
        color = (255, 200, 0) if b.cut else (255, 40, 40)
        mask = b.detection.mask
        out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
        center = (int(round(b.pixel[0])), int(round(b.pixel[1])))
        cv2.circle(out, center, max(1, int(round(b.radius_px))), color, 1)
        cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 8, 1)
        text = f'{i} {b.distance_m:.2f}m r={b.radius_m * 100:.1f}cm'
        org = (center[0] + int(b.radius_px) + 4, center[1])
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    status = ' '.join(
        part for part in (
            f'{inference_ms:.0f} ms' if inference_ms is not None else None,
            backend,
        ) if part
    )
    if status:
        cv2.putText(out, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out
