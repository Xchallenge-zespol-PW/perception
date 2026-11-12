"""Map of balls in a world-fixed frame with stable ids (no ROS)."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# A ball enters the map after this many frames with an observation
# (about 0.3 s at 15 fps): one-frame false detections never get an id.
CONFIRM_FRAMES = 5
# A ball not yet in the map is forgotten after this long without one.
TENTATIVE_TIMEOUT_S = 1.0
# The position is the median of this many latest observations.
MAX_SAMPLES = 30
# An observation belongs to a ball within this many ball radii of it: two
# balls are at least 2 radii apart (centre to centre), so a ball is never
# matched to its neighbour unless they touch.
GATE_RADII = 2.0

# What the camera sees where a ball of the map should be (view_state).
VIEW_SEEN = 'seen'            # detected on this frame
VIEW_PRESENT = 'present'      # not detected, but something at its depth
VIEW_EMPTY = 'empty'          # the depth there is behind the ball: it is gone
VIEW_OCCLUDED = 'occluded'    # something in front of it
VIEW_OUT = 'out_of_view'      # outside the image (whole ball not in frame)
VIEW_UNKNOWN = 'unknown'      # too little depth there
# Depth is read in the disk of this fraction of the ball's image radius.
VIEW_DISK_FRAC = 0.5
MIN_VALID_FRAC = 0.3
MIN_VIEW_DEPTH_M = 0.2
# A ball leaves the map after this many frames and this long seen empty
# with no frame in between showing it (seen or present); a longer gap
# between empty frames starts the count again.
EMPTY_FRAMES = 5
EMPTY_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class MappedBall:
    """
    A ball of the map.

    ``ball_id`` is stable for the run (1, 2, ... in order of entry), or
    None while the ball is tentative. ``position`` is the median of the
    latest observations in the map frame, ``radius_m`` their median radius.
    """

    ball_id: int | None
    position: NDArray[np.float64]
    radius_m: float
    n_obs: int
    first_seen: float
    last_seen: float


@dataclass
class _Ball:
    """Observations of one ball (tentative while ``ball_id`` is None)."""

    first_seen: float
    last_seen: float
    ball_id: int | None = None
    n_obs: int = 0
    positions: deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    radii: deque = field(default_factory=lambda: deque(maxlen=MAX_SAMPLES))
    empty_since: float | None = None
    last_empty: float | None = None
    n_empty: int = 0

    def add(self, position: NDArray[np.float64], radius: float, stamp: float) -> None:
        """Add one observation."""
        self.positions.append(np.asarray(position, dtype=np.float64))
        self.radii.append(float(radius))
        self.n_obs += 1
        self.last_seen = stamp
        self.clear_empty()

    def clear_empty(self) -> None:
        """Forget the frames that saw this ball's place empty."""
        self.empty_since = self.last_empty = None
        self.n_empty = 0

    def see_empty(self, stamp: float) -> bool:
        """Count a frame that saw the place empty; True when the ball is gone."""
        if self.last_empty is None or stamp - self.last_empty > EMPTY_TIMEOUT_S:
            self.empty_since, self.n_empty = stamp, 0
        self.last_empty = stamp
        self.n_empty += 1
        return (
            self.n_empty >= EMPTY_FRAMES
            and stamp - self.empty_since >= EMPTY_TIMEOUT_S
        )

    @property
    def position(self) -> NDArray[np.float64]:
        """Median of the latest positions."""
        return np.median(np.asarray(self.positions), axis=0)

    @property
    def radius(self) -> float:
        """Median of the latest radii."""
        return float(np.median(self.radii))

    def snapshot(self) -> MappedBall:
        """Return an immutable copy."""
        return MappedBall(
            ball_id=self.ball_id,
            position=self.position,
            radius_m=self.radius,
            n_obs=self.n_obs,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


class BallMap:
    """
    Keep every ball seen in a world-fixed frame (``odom``) under a stable id.

    Each frame's balls are matched one to one, nearest first, to the balls
    of the map within ``GATE_RADII`` radii; an unmatched ball starts a
    tentative one, which gets the next id after ``CONFIRM_FRAMES`` frames.
    A ball keeps its id while out of view or hidden; it leaves the map only
    when its place is seen empty (:meth:`check_views`), and its id is never
    reused.
    """

    def __init__(self) -> None:
        self._balls: list[_Ball] = []
        self._next_id = 1
        self._latest: float | None = None
        # Distance (m) of each matched observation of the last update from
        # its ball: grows when the map frame drifts (odom while walking).
        self.last_residuals: list[float] = []

    def reset(self) -> None:
        """Forget all balls; ids start again at 1."""
        self._balls.clear()
        self._next_id = 1
        self._latest = None

    def update(
        self,
        positions: Sequence[NDArray[np.float64]],
        radii: Sequence[float],
        stamp: float,
    ) -> tuple[list[int | None], list[MappedBall]]:
        """
        Add one frame's balls (map frame) and return ``(ids, confirmed)``.

        ``ids[i]`` is the map id of ``positions[i]`` (None while tentative),
        ``confirmed`` the balls that got an id on this frame. A stamp older
        than the previous one resets the map (bag ``--loop``).
        """
        if len(positions) != len(radii):
            raise ValueError('positions and radii differ in length')
        stamp = float(stamp)
        if self._latest is not None and stamp < self._latest:
            self.reset()
        self._latest = stamp
        self._balls = [
            b for b in self._balls
            if b.ball_id is not None or stamp - b.last_seen <= TENTATIVE_TIMEOUT_S
        ]

        pts = [np.asarray(p, dtype=np.float64) for p in positions]
        candidates: list[tuple[float, int, int]] = []
        for i, (p, r) in enumerate(zip(pts, radii)):
            for j, ball in enumerate(self._balls):
                dist = float(np.linalg.norm(p - ball.position))
                if dist <= GATE_RADII * max(float(r), ball.radius):
                    candidates.append((dist, i, j))
        candidates.sort()

        owner: dict[int, _Ball] = {}
        used: set[int] = set()
        self.last_residuals = []
        for dist, i, j in candidates:
            if i in owner or j in used:
                continue
            owner[i] = self._balls[j]
            used.add(j)
            if self._balls[j].ball_id is not None:
                self.last_residuals.append(dist)

        ids: list[int | None] = []
        confirmed: list[MappedBall] = []
        for i, (p, r) in enumerate(zip(pts, radii)):
            ball = owner.get(i)
            if ball is None:
                ball = _Ball(first_seen=stamp, last_seen=stamp)
                self._balls.append(ball)
            ball.add(p, r, stamp)
            if ball.ball_id is None and ball.n_obs >= CONFIRM_FRAMES:
                ball.ball_id = self._next_id
                self._next_id += 1
                confirmed.append(ball.snapshot())
            ids.append(ball.ball_id)
        return ids, confirmed

    def check_views(
        self,
        views: Mapping[int, str],
        stamp: float,
    ) -> list[MappedBall]:
        """
        Apply what a frame saw at the places of known balls (by id).

        ``VIEW_EMPTY`` counts towards removal (``EMPTY_FRAMES`` frames over
        ``EMPTY_TIMEOUT_S``), ``VIEW_SEEN`` / ``VIEW_PRESENT`` cancel it,
        the other views change nothing. Returns the balls removed.
        """
        stamp = float(stamp)
        removed: list[_Ball] = []
        for ball in self._balls:
            view = views.get(ball.ball_id) if ball.ball_id is not None else None
            if view == VIEW_EMPTY:
                if ball.see_empty(stamp):
                    removed.append(ball)
            elif view in (VIEW_SEEN, VIEW_PRESENT):
                ball.clear_empty()
        if removed:
            self._balls = [b for b in self._balls if b not in removed]
        return [b.snapshot() for b in removed]

    def empty_for(self, ball_id: int, stamp: float) -> float | None:
        """Seconds the place of ``ball_id`` has been seen empty, or None."""
        for ball in self._balls:
            if ball.ball_id == ball_id and ball.empty_since is not None:
                return float(stamp) - ball.empty_since
        return None

    def known(self) -> list[MappedBall]:
        """Return the balls with an id, sorted by id."""
        return sorted(
            (b.snapshot() for b in self._balls if b.ball_id is not None),
            key=lambda b: b.ball_id,
        )

    def nearest_known(
        self,
        position: NDArray[np.float64],
        exclude: int | None = None,
    ) -> tuple[MappedBall, float] | None:
        """Return the known ball nearest to ``position`` and its distance."""
        best: tuple[MappedBall, float] | None = None
        for b in self.known():
            if b.ball_id == exclude:
                continue
            dist = float(np.linalg.norm(b.position - np.asarray(position)))
            if best is None or dist < best[1]:
                best = (b, dist)
        return best


def map_summary(
    balls: Sequence[MappedBall],
    prefix: str = 'ball_',
) -> list[dict[str, float | str]]:
    """Return ``[{'name': 'ball_1', 'x': .., 'y': .., 'z': .., 'diameter_m': ..}]``."""
    return [
        {
            'name': f'{prefix}{b.ball_id}',
            'x': round(float(b.position[0]), 3),
            'y': round(float(b.position[1]), 3),
            'z': round(float(b.position[2]), 3),
            'diameter_m': round(2.0 * b.radius_m, 3),
        }
        for b in balls
    ]


def project(
    center: NDArray[np.float64],
    radius: float,
    K: NDArray[np.float64],
) -> tuple[tuple[float, float], float] | None:
    """Return the pixel and image radius of a ball (camera optical frame)."""
    x, y, z = (float(v) for v in center)
    if z <= MIN_VIEW_DEPTH_M:
        return None
    u = float(K[0, 0]) * x / z + float(K[0, 2])
    v = float(K[1, 1]) * y / z + float(K[1, 2])
    r_px = 0.5 * (float(K[0, 0]) + float(K[1, 1])) * float(radius) / z
    return (u, v), r_px


def view_state(
    center: NDArray[np.float64],
    radius: float,
    depth_m: NDArray[np.float32],
    K: NDArray[np.float64],
) -> str:
    """
    Classify what the camera sees where a ball should be (not detected).

    ``center`` is the ball centre in the camera optical frame. The median
    depth of the inner disk of its image tells: beyond the ball's back
    (``z + r``) = ``empty``, more than a radius in front of its front
    (``z - 2r``) = ``occluded``, otherwise ``present``. A ball not wholly
    in the image is ``out_of_view``, a disk with too little depth
    ``unknown``.
    """
    proj = project(center, radius, K)
    if proj is None:
        return VIEW_OUT
    (u, v), r_px = proj
    h, w = depth_m.shape
    if u - r_px < 0 or v - r_px < 0 or u + r_px >= w or v + r_px >= h:
        return VIEW_OUT
    rd = max(VIEW_DISK_FRAC * r_px, 1.0)
    x0, x1 = int(np.floor(u - rd)), int(np.ceil(u + rd)) + 1
    y0, y1 = int(np.floor(v - rd)), int(np.ceil(v + rd)) + 1
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disk = (xs - u) ** 2 + (ys - v) ** 2 <= rd * rd
    z = depth_m[y0:y1, x0:x1][disk]
    valid = z[z > 0.0]
    if valid.size < max(1, MIN_VALID_FRAC * z.size):
        return VIEW_UNKNOWN
    z_med = float(np.median(valid))
    z_c = float(center[2])
    if z_med > z_c + radius:
        return VIEW_EMPTY
    if z_med < z_c - 2.0 * radius:
        return VIEW_OCCLUDED
    return VIEW_PRESENT
