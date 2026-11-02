"""Multi-frame voting of per-slot pipeline results."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from g1_perception.core.geometry import (
    quat_from_plane_yaw,
    quat_to_rotation_matrix,
    yaw_on_plane_deg,
)
from g1_perception.core.types import (
    COLOR_LABELS,
    LABEL_UNCERTAIN,
    Pose,
    SlotResult,
)


@dataclass(frozen=True)
class VotedSlot:
    """
    Voted decision for one slot / track over the frame window.

    ``score`` is the fraction of window frames that agree with the winning
    label (0 when there are too few frames). ``n_frames`` is the number of
    frames currently in the slot's window. ``pose`` and ``yaw_deg`` are set
    only for ``red`` / ``green`` winners with at least one pose;
    ``yaw_consistency`` is the circular resultant length of ``4 * yaw`` in
    [0, 1] (1 = all yaws agree modulo 90°). ``track_id`` is set in free
    (no-ROI) mode; ``assignment_ok`` is False when place validation failed.
    """

    slot: int
    label: str
    score: float
    n_frames: int
    pose: Pose | None
    yaw_deg: float | None
    yaw_consistency: float | None
    track_id: int | None = None
    assignment_ok: bool = True


def circular_median_mod90(
    yaws_deg: Iterable[float],
) -> tuple[float, float]:
    """
    Robust yaw with 90° symmetry; returns ``(yaw_deg, consistency)``.

    The circular mean of ``4 * yaw`` gives a reference; each yaw is wrapped
    to within ±45° of it, and the median offset is added back. Result is in
    ``[0, 90)``. ``consistency`` is the resultant length of ``4 * yaw``.
    """
    yaws = np.asarray(list(yaws_deg), dtype=np.float64)
    if yaws.size == 0:
        raise ValueError('Need at least one yaw')

    theta = np.deg2rad(4.0 * yaws)
    s = float(np.mean(np.sin(theta)))
    c = float(np.mean(np.cos(theta)))
    consistency = float(np.hypot(s, c))
    ref = float(np.rad2deg(np.arctan2(s, c)) / 4.0)

    offsets = (yaws - ref + 45.0) % 90.0 - 45.0
    yaw = (ref + float(np.median(offsets))) % 90.0
    if yaw >= 90.0:  # tiny negative values round up to 90.0 under %
        yaw = 0.0
    return float(yaw), consistency


class SlotVoter:
    """
    Aggregate :class:`SlotResult` sequences over a sliding window per slot.

    Per slot: majority label over the last ``window`` frames; the winner is
    accepted only if its share is at least ``min_ratio``, it is not tied, and
    the window holds at least ``min_frames`` frames (default: ``window``);
    otherwise the slot is ``uncertain``. For colour winners the pose is the
    per-axis median position of agreeing frames, the yaw is
    :func:`circular_median_mod90`, and +Z is the mean plane normal.
    """

    def __init__(
        self,
        window: int = 10,
        min_ratio: float = 0.7,
        min_frames: int | None = None,
    ) -> None:
        if window < 1:
            raise ValueError(f'window must be >= 1, got {window}')
        if not 0.0 < min_ratio <= 1.0:
            raise ValueError(f'min_ratio must be in (0, 1], got {min_ratio}')
        if min_frames is None:
            min_frames = window
        if not 1 <= min_frames <= window:
            raise ValueError(
                f'min_frames must be in [1, window], got {min_frames}'
            )
        self.window = int(window)
        self.min_ratio = float(min_ratio)
        self.min_frames = int(min_frames)
        self._history: dict[int, deque[SlotResult]] = {}

    def update(self, results: Sequence[SlotResult]) -> list[VotedSlot]:
        """Add one frame of results and return the current vote."""
        for result in results:
            history = self._history.get(result.slot)
            if history is None:
                history = deque(maxlen=self.window)
                self._history[result.slot] = history
            history.append(result)
        return self.vote()

    def vote(self) -> list[VotedSlot]:
        """Return the voted decision for every slot seen, sorted by slot."""
        return [
            self._vote_slot(slot, list(self._history[slot]))
            for slot in sorted(self._history)
        ]

    def reset(self) -> None:
        """Forget all accumulated frames."""
        self._history.clear()

    def forget(self, slot: int) -> None:
        """Drop the window for one slot / track id."""
        self._history.pop(slot, None)

    def _vote_slot(self, slot: int, history: list[SlotResult]) -> VotedSlot:
        """Vote a single slot from its window history."""
        n = len(history)
        if n < self.min_frames:
            return _uncertain(slot, n)

        ranked = Counter(r.label for r in history).most_common()
        label, count = ranked[0]
        tied = len(ranked) > 1 and ranked[1][1] == count
        ratio = count / n
        if tied or ratio < self.min_ratio or label == LABEL_UNCERTAIN:
            return _uncertain(slot, n, score=ratio)

        poses = [
            r.pose for r in history
            if r.label == label and r.pose is not None
        ]
        if label not in COLOR_LABELS or not poses:
            return VotedSlot(slot, label, ratio, n, None, None, None)

        pose, yaw_deg, consistency = _vote_pose(poses, ratio)
        return VotedSlot(slot, label, ratio, n, pose, yaw_deg, consistency)


def _uncertain(slot: int, n_frames: int, score: float = 0.0) -> VotedSlot:
    return VotedSlot(slot, LABEL_UNCERTAIN, score, n_frames, None, None, None)


def _vote_pose(
    poses: Sequence[Pose],
    score: float,
) -> tuple[Pose, float, float]:
    """Median position, circular-median yaw, and mean normal of ``poses``."""
    positions = np.stack([p.position for p in poses])
    position = np.median(positions, axis=0).astype(np.float64)

    normals = [_z_axis(p.quat) for p in poses]
    normal = np.mean(normals, axis=0)
    normal = normal / np.linalg.norm(normal)

    yaws = [yaw_on_plane_deg(p.quat, n) for p, n in zip(poses, normals)]
    yaw_deg, consistency = circular_median_mod90(yaws)

    quat = quat_from_plane_yaw(normal, yaw_deg)
    return Pose(position=position, quat=quat, score=score), yaw_deg, consistency


def _z_axis(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the +Z column of the rotation for ``quat`` (xyzw)."""
    return quat_to_rotation_matrix(quat)[:, 2]
