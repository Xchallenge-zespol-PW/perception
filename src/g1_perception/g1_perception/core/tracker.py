"""Multi-frame 3D nearest-neighbour tracking of free cube detections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from g1_perception.core.types import Detection, Pose, SlotResult
from g1_perception.core.voting import SlotVoter, VotedSlot


@dataclass
class _Track:
    track_id: int
    missed: int = 0
    last_detection: Detection | None = None
    last_position: np.ndarray | None = None


@dataclass(frozen=True)
class TrackedCube:
    """Voted cube with a stable track id (before place assignment)."""

    track_id: int
    label: str
    score: float
    n_frames: int
    pose: Pose | None
    yaw_deg: float | None
    yaw_consistency: float | None
    detection: Detection | None
    missed: int

    def as_voted_slot(self, slot: int | None = None) -> VotedSlot:
        """Convert to :class:`VotedSlot` (``slot`` defaults to ``track_id``)."""
        return VotedSlot(
            slot=self.track_id if slot is None else int(slot),
            label=self.label,
            score=self.score,
            n_frames=self.n_frames,
            pose=self.pose,
            yaw_deg=self.yaw_deg,
            yaw_consistency=self.yaw_consistency,
            track_id=self.track_id,
            assignment_ok=True,
        )


class CubeTracker:
    """
    Track cubes by 3D position with greedy nearest-neighbour matching.

    Each track owns a :class:`SlotVoter` window (keyed by ``track_id``):
    majority colour, median position, circular-median yaw modulo 90°.
    Tracks unseen for more than ``max_missed`` frames are dropped.
    """

    def __init__(
        self,
        *,
        match_dist_m: float = 0.03,
        max_missed: int = 5,
        vote_window: int = 10,
        vote_min_ratio: float = 0.7,
        vote_min_frames: int | None = None,
    ) -> None:
        if match_dist_m <= 0.0:
            raise ValueError(f'match_dist_m must be > 0, got {match_dist_m}')
        if max_missed < 0:
            raise ValueError(f'max_missed must be >= 0, got {max_missed}')
        self.match_dist_m = float(match_dist_m)
        self.max_missed = int(max_missed)
        self._voter = SlotVoter(
            window=vote_window,
            min_ratio=vote_min_ratio,
            min_frames=vote_min_frames,
        )
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Forget all tracks and voting history."""
        self._tracks.clear()
        self._voter.reset()
        self._next_id = 1

    def update(self, results: Sequence[SlotResult]) -> list[TrackedCube]:
        """
        Match framed detections to tracks and return active tracked cubes.

        Only observations with a pose participate in matching. Unmatched
        observations spawn new tracks. Returned list is sorted by track id.
        """
        observations = [r for r in results if r.pose is not None]
        track_ids = list(self._tracks)
        pairs = self._greedy_match(observations, track_ids)

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        for obs_i, tid in pairs:
            matched_obs.add(obs_i)
            matched_tracks.add(tid)
            obs = observations[obs_i]
            track = self._tracks[tid]
            track.missed = 0
            track.last_detection = obs.detection
            track.last_position = np.asarray(
                obs.pose.position, dtype=np.float64
            ).copy()
            self._voter.update([
                SlotResult(
                    slot=tid,
                    label=obs.label,
                    score=obs.score,
                    detection=obs.detection,
                    pose=obs.pose,
                )
            ])

        for obs_i, obs in enumerate(observations):
            if obs_i in matched_obs:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid,
                missed=0,
                last_detection=obs.detection,
                last_position=np.asarray(
                    obs.pose.position, dtype=np.float64
                ).copy(),
            )
            self._voter.update([
                SlotResult(
                    slot=tid,
                    label=obs.label,
                    score=obs.score,
                    detection=obs.detection,
                    pose=obs.pose,
                )
            ])

        for tid in track_ids:
            if tid in matched_tracks:
                continue
            self._tracks[tid].missed += 1

        dead = [
            tid for tid, tr in self._tracks.items()
            if tr.missed > self.max_missed
        ]
        for tid in dead:
            del self._tracks[tid]
            self._voter.forget(tid)

        voted = {v.slot: v for v in self._voter.vote()}
        out: list[TrackedCube] = []
        for tid in sorted(self._tracks):
            tr = self._tracks[tid]
            v = voted.get(tid)
            if v is None:
                continue
            out.append(
                TrackedCube(
                    track_id=tid,
                    label=v.label,
                    score=v.score,
                    n_frames=v.n_frames,
                    pose=v.pose,
                    yaw_deg=v.yaw_deg,
                    yaw_consistency=v.yaw_consistency,
                    detection=tr.last_detection,
                    missed=tr.missed,
                )
            )
        return out

    def _greedy_match(
        self,
        observations: Sequence[SlotResult],
        track_ids: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Return ``(obs_index, track_id)`` pairs under ``match_dist_m``."""
        if not observations or not track_ids:
            return []

        candidates: list[tuple[float, int, int]] = []
        for i, obs in enumerate(observations):
            assert obs.pose is not None
            p = np.asarray(obs.pose.position, dtype=np.float64)
            for tid in track_ids:
                tr = self._tracks[tid]
                if tr.last_position is None:
                    continue
                dist = float(np.linalg.norm(p - tr.last_position))
                if dist <= self.match_dist_m:
                    candidates.append((dist, i, tid))

        candidates.sort(key=lambda t: t[0])
        used_obs: set[int] = set()
        used_tr: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _dist, i, tid in candidates:
            if i in used_obs or tid in used_tr:
                continue
            used_obs.add(i)
            used_tr.add(tid)
            pairs.append((i, tid))
        return pairs
