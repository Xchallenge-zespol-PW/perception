"""Map of balls in a world-fixed frame with stable ids (no ROS)."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
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

    def add(self, position: NDArray[np.float64], radius: float, stamp: float) -> None:
        """Add one observation."""
        self.positions.append(np.asarray(position, dtype=np.float64))
        self.radii.append(float(radius))
        self.n_obs += 1
        self.last_seen = stamp

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
    A ball keeps its id while out of view; no ball leaves the map yet.
    """

    def __init__(self) -> None:
        self._balls: list[_Ball] = []
        self._next_id = 1
        self._latest: float | None = None

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
        for _, i, j in candidates:
            if i in owner or j in used:
                continue
            owner[i] = self._balls[j]
            used.add(j)

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
