"""Ball centre from a YOLO mask and aligned depth (no ROS)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from g1_perception.core.types import Detection
from g1_perception.core.visibility import mask_touches_border

DEFAULT_BALL_LABELS = ('red ball',)
METHOD_CIRCLE = 'circle'
METHOD_AREA = 'area'
# Depth is read inside this fraction of the silhouette radius: the rim mixes
# the ball with the shelf or the background behind it.
INNER_RADIUS_FRAC = 0.85
MIN_DEPTH_PX = 10
# Depth outside the mask is sampled this far (px) beyond each contour point.
EDGE_PROBE_PX = (3, 5, 7)
# A contour point is an occlusion edge when the depth just outside it is in
# front of the ball's front surface by more than this fraction of its radius.
OCCLUDER_FRAC = 0.25
MIN_EDGE_PX = 8
ARC_BINS = 36
# The circle is fitted only to an outline spanning this much of the turn
# (0.4 = 144°); a ball hidden more is estimated from its mask area.
MIN_ARC_FRAC = 0.4
# Ball radius from the depth profile vs from the silhouette: outside these
# ratios (beyond 2 standard errors) the blob is flat (poster) or no ball.
MIN_DEPTH_RADIUS_RATIO = 0.4
MAX_DEPTH_RADIUS_RATIO = 2.5
# The D435 smooths the depth of a small ball: below this silhouette radius
# (px; about 1.2 m on the robot) whole balls looked flat or too deep in
# the bags of 2026-11-11, so the shape is not judged there.
MIN_SHAPE_RADIUS_PX = 10.0


@dataclass(frozen=True)
class BallFit:
    """
    Geometry of one ball in the camera optical frame.

    ``position`` is the centre (x, y, z) in metres, ``radius_m`` the radius
    from the silhouette radius ``radius_px`` and the depth (no fixed ball
    size). ``pixel`` is the silhouette centre (may lie outside the mask of
    a half hidden ball). ``method`` is ``circle`` (circle fitted to the true
    outline, occlusion edges dropped) or ``area`` (mask centroid and
    equal-area radius, biased when hidden). ``arc_frac`` is the share of the
    turn covered by the true outline. ``depth_radius_m`` is the radius seen
    in the depth profile, ``depth_radius_se`` its standard error.
    """

    position: NDArray[np.float64]
    radius_m: float
    pixel: tuple[float, float]
    radius_px: float
    n_depth: int
    method: str
    arc_frac: float
    depth_radius_m: float
    depth_radius_se: float

    @property
    def shape_ok(self) -> bool:
        """Return False when the depth profile is surely not a ball of this size."""
        r, rd, se = self.radius_m, self.depth_radius_m, self.depth_radius_se
        if self.radius_px < MIN_SHAPE_RADIUS_PX:
            return True
        if not np.isfinite(rd) or not np.isfinite(se):
            return True
        return (
            rd + 2.0 * se >= MIN_DEPTH_RADIUS_RATIO * r
            and rd - 2.0 * se <= MAX_DEPTH_RADIUS_RATIO * r
        )


@dataclass(frozen=True)
class BallObservation:
    """
    One ball seen on a frame: its :class:`BallFit` and detection.

    ``cut`` is True when the mask touches the image border, ``duplicate``
    when a better ball has its centre within one radius (a second mask on
    the same ball). ``trusted`` balls (outline fitted, depth profile of a
    ball, not a duplicate) are published.
    """

    fit: BallFit
    score: float
    cut: bool
    detection: Detection
    duplicate: bool = False

    @property
    def position(self) -> NDArray[np.float64]:
        """Ball centre in the camera optical frame (m)."""
        return self.fit.position

    @property
    def radius_m(self) -> float:
        """Ball radius measured on this frame (m)."""
        return self.fit.radius_m

    @property
    def distance_m(self) -> float:
        """Distance from the camera to the ball centre."""
        return float(np.linalg.norm(self.fit.position))

    @property
    def reject_reason(self) -> str:
        """Empty for a trusted ball, else ``duplicate``, ``hidden`` or ``shape``."""
        if self.duplicate:
            return 'duplicate'
        if self.fit.method != METHOD_CIRCLE:
            return 'hidden'
        if not self.fit.shape_ok:
            return 'shape'
        return ''

    @property
    def trusted(self) -> bool:
        """Return True when the pose may be published."""
        return not self.reject_reason


def is_ball(label: str, names: Sequence[str] = DEFAULT_BALL_LABELS) -> bool:
    """Return True when the raw class ``label`` is one of ``names``."""
    key = label.strip().lower()
    return any(key == n.strip().lower() for n in names)


def fit_circle(
    points: NDArray[np.float64],
) -> tuple[float, float, float] | None:
    """
    Fit a circle to 2D ``points`` (Kåsa least squares, one outlier pass).

    Returns ``(u, v, radius)`` or None for fewer than 3 points / a line.
    The radius is that of the points; a contour of pixel centres lies half
    a pixel inside the mask boundary (see :func:`fit_ball`).
    """
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(2):
        if len(pts) < 3:
            return None
        A = np.column_stack((2.0 * pts, np.ones(len(pts))))
        b = (pts ** 2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        u, v, c = sol
        r2 = c + u * u + v * v
        if not np.isfinite(r2) or r2 <= 0.0:
            return None
        r = float(np.sqrt(r2))
        res = np.abs(np.hypot(pts[:, 0] - u, pts[:, 1] - v) - r)
        mad = float(np.median(res))
        keep = res <= 3.0 * 1.4826 * mad + 0.5
        if keep.all():
            break
        pts = pts[keep]
    return float(u), float(v), r


def true_outline(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    *,
    center: tuple[float, float],
    z_front: float,
    radius_m: float,
    border_px: int = 0,
    image_box: tuple[int, int, int, int] | None = None,
) -> NDArray[np.float64]:
    """
    Return the contour points (u, v) where the ball's silhouette is real.

    Dropped: points at the image border, and occlusion edges, where the
    depth just outside the mask is in front of the ball (a rack post or a
    hand covers it). No depth outside counts as background. For a crop of
    the image, ``image_box`` is ``(left, top, right, bottom)`` of the whole
    image in crop pixels (default: the array itself).
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.zeros((0, 2))
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    h, w = mask.shape
    left, top, right, bottom = image_box or (0, 0, w, h)
    m = max(int(border_px), 1)
    on_image = (
        (pts[:, 0] >= left + m) & (pts[:, 0] < right - m)
        & (pts[:, 1] >= top + m) & (pts[:, 1] < bottom - m)
    )
    d = pts - np.asarray(center, dtype=np.float64)
    norm = np.linalg.norm(d, axis=1)
    norm[norm == 0.0] = 1.0
    d /= norm[:, None]

    outside = np.zeros((len(pts), len(EDGE_PROBE_PX)), dtype=np.float32)
    for j, step in enumerate(EDGE_PROBE_PX):
        q = np.rint(pts + step * d).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, w - 1)
        q[:, 1] = np.clip(q[:, 1], 0, h - 1)
        z = depth_m[q[:, 1], q[:, 0]]
        outside[:, j] = np.where(mask[q[:, 1], q[:, 0]], 0.0, z)
    valid = outside > 0.0
    z_near = np.where(valid, outside, np.inf).min(axis=1)
    occluded = z_near < z_front - OCCLUDER_FRAC * radius_m
    return pts[on_image & ~occluded]


def _arc_frac(points: NDArray[np.float64], u: float, v: float) -> float:
    """Share of ``ARC_BINS`` angular bins around (u, v) holding a point."""
    ang = np.arctan2(points[:, 1] - v, points[:, 0] - u)
    bins = np.floor((ang + np.pi) / (2.0 * np.pi) * ARC_BINS).astype(int)
    return len(np.unique(np.clip(bins, 0, ARC_BINS - 1))) / ARC_BINS


def _depth_profile(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    u: float,
    v: float,
    r_px: float,
    f: float,
) -> tuple[float, int, float, float] | None:
    """
    Centre depth from the visible surface of a sphere of silhouette (u, v, r_px).

    A pixel at ``rho`` (silhouette radii) from the centre sees the surface
    ``r * sqrt(1 - rho²)`` in front of the centre, and ``r = r_px * z_c / f``,
    so each pixel gives ``z_c = z / (1 - r_px * s / f)``; the median is
    kept. A least-squares fit ``z = z_c - r_d * s`` gives the radius seen
    in depth ``r_d`` and its standard error (NaN when ill-posed). Returns
    ``(z_c, n_pixels, r_d, se)`` or None without enough depth.
    """
    ys, xs = np.nonzero(mask)
    rho = np.hypot(xs - u, ys - v) / r_px
    z = depth_m[ys, xs].astype(np.float64)
    use = (rho < INNER_RADIUS_FRAC) & (z > 0.0)
    if np.count_nonzero(use) < MIN_DEPTH_PX:
        return None
    z = z[use]
    s = np.sqrt(1.0 - rho[use] ** 2)
    z_c = float(np.median(z / (1.0 - r_px * s / f)))

    # Depth outliers (flying pixels, background seen through the mask) would
    # dominate the fit: keep pixels near the sphere of the median centre.
    res = z - (z_c - r_px * z_c / f * s)
    mad = float(np.median(np.abs(res - np.median(res))))
    inl = np.abs(res - np.median(res)) <= 3.0 * 1.4826 * mad + 1e-3
    zi, si = z[inl], s[inl]
    A = np.column_stack((np.ones_like(si), -si))
    r_d = se = float('nan')
    if len(zi) > 2:
        sol, _, rank, _ = np.linalg.lstsq(A, zi, rcond=None)
        if rank == 2:
            res = zi - A @ sol
            sigma2 = float(res @ res) / (len(zi) - 2)
            cov = sigma2 * np.linalg.inv(A.T @ A)
            r_d, se = float(sol[1]), float(np.sqrt(max(cov[1, 1], 0.0)))
    return z_c, len(z), r_d, se


def fit_ball(
    mask: NDArray[np.bool_],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
    *,
    border_px: int = 0,
) -> BallFit | None:
    """
    Estimate a ball from its mask and aligned depth; None without depth.

    1. Mask centroid, equal-area radius and the median depth of its inner
       disk give a first guess (exact for a whole ball).
    2. A circle is fitted to the true outline (:func:`true_outline`), so a
       ball half hidden by a post or cut by the image border keeps its
       centre and size; with too little outline the first guess is kept
       (``method=area``).
    3. :func:`_depth_profile` puts the centre behind the visible surface.

    Works on a crop around the mask: a few ms per ball on the whole image.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    pad = max(EDGE_PROBE_PX) + 1
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    mask = mask[y0:y1, x0:x1]
    depth_m = depth_m[y0:y1, x0:x1]
    xs = xs - x0
    ys = ys - y0

    fx, fy = float(K[0, 0]), float(K[1, 1])
    f = 0.5 * (fx + fy)
    u0, v0 = float(xs.mean()), float(ys.mean())
    r0 = float(np.sqrt(ys.size / np.pi))
    if r0 >= f:
        return None
    first = _depth_profile(mask, depth_m, u0, v0, r0, f)
    if first is None:
        return None
    z0 = first[0]
    radius0 = r0 * z0 / f

    u, v, r_px, method, arc = u0, v0, r0, METHOD_AREA, 0.0
    edge = true_outline(
        mask, depth_m, center=(u0, v0), z_front=z0 - radius0,
        radius_m=radius0, border_px=border_px,
        image_box=(-x0, -y0, w - x0, h - y0),
    )
    if len(edge) >= MIN_EDGE_PX:
        circle = fit_circle(edge)
        if circle is not None:
            cu, cv_, cr = circle
            cr += 0.5  # contour pixel centres -> mask boundary
            arc = _arc_frac(edge, cu, cv_)
            if arc >= MIN_ARC_FRAC and 0.5 * r0 <= cr <= 3.0 * r0 and cr < f:
                u, v, r_px, method = cu, cv_, cr, METHOD_CIRCLE

    prof = _depth_profile(mask, depth_m, u, v, r_px, f)
    if prof is None:
        return None
    z_c, n_depth, r_d, se = prof
    u += x0
    v += y0
    position = np.array([
        (u - float(K[0, 2])) * z_c / fx,
        (v - float(K[1, 2])) * z_c / fy,
        z_c,
    ])
    return BallFit(
        position=position,
        radius_m=r_px * z_c / f,
        pixel=(u, v),
        radius_px=r_px,
        n_depth=n_depth,
        method=method,
        arc_frac=arc,
        depth_radius_m=r_d,
        depth_radius_se=se,
    )


def find_balls(
    detections: Sequence[Detection],
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
    *,
    labels: Sequence[str] = DEFAULT_BALL_LABELS,
    border_px: int = 0,
) -> list[BallObservation]:
    """
    Return balls among ``detections``, nearest first.

    Balls cannot overlap: of two centres closer than the larger radius, the
    one trusted on its own (then the higher score) stays, the other is
    marked ``duplicate``.
    """
    out: list[BallObservation] = []
    for det in detections:
        if not is_ball(det.label, labels):
            continue
        fit = fit_ball(det.mask, depth_m, K, border_px=border_px)
        if fit is None:
            continue
        out.append(
            BallObservation(
                fit=fit,
                score=float(det.score),
                cut=mask_touches_border(det.mask, max(int(border_px), 1)),
                detection=det,
            )
        )
    out.sort(key=lambda b: (not b.trusted, -b.score))
    kept: list[BallObservation] = []
    for i, b in enumerate(out):
        if any(
            np.linalg.norm(b.position - k.position) < max(b.radius_m, k.radius_m)
            for k in kept
        ):
            out[i] = replace(b, duplicate=True)
        else:
            kept.append(b)
    out.sort(key=lambda b: b.distance_m)
    return out


def _label(k: int, names: Sequence[str] | None) -> str:
    """Name of ball ``k`` (0-based): ``names[k]`` unless ``-``, else ``k + 1``."""
    if names is not None and k < len(names) and names[k] != '-':
        return names[k]
    return str(k + 1)


def format_balls(
    balls: Sequence[BallObservation],
    names: Sequence[str] | None = None,
) -> str:
    """
    Return ``"1: 1.23 m r=3.4 cm, 2: ... (hidden)"`` for logs.

    With ``names`` (e.g. map names, ``-`` = none) they replace the numbers.
    """
    return ', '.join(
        f'{_label(k, names)}: {b.distance_m:.2f} m r={b.radius_m * 100:.1f} cm'
        + (f' ({b.reject_reason})' if b.reject_reason else '')
        for k, b in enumerate(balls)
    ) or 'none'


def draw_balls(
    rgb: NDArray[np.uint8],
    balls: Sequence[BallObservation],
    *,
    names: Sequence[str] | None = None,
    inference_ms: float | None = None,
    backend: str | None = None,
) -> NDArray[np.uint8]:
    """
    Render masks, fitted circles and ``name d=.. r=..`` labels on ``rgb``.

    Red = published, yellow = rejected (reason in the label). Without
    ``names`` the balls are numbered nearest first.
    """
    out = rgb.copy()
    for k, b in enumerate(balls):
        color = (255, 40, 40) if b.trusted else (255, 200, 0)
        mask = b.detection.mask
        out[mask] = (0.55 * out[mask] + 0.45 * np.array(color)).astype(np.uint8)
        center = (int(round(b.fit.pixel[0])), int(round(b.fit.pixel[1])))
        cv2.circle(out, center, max(1, int(round(b.fit.radius_px))), color, 1)
        cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 8, 1)
        text = f'{_label(k, names)} {b.distance_m:.2f}m r={b.radius_m * 100:.1f}cm'
        if b.reject_reason:
            text += f' {b.reject_reason}'
        org = (center[0] + int(b.fit.radius_px) + 4, center[1])
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
