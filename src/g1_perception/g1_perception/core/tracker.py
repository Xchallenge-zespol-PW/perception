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
    last_seen: float = 0.0
    last_detection: Detection | None = None
    last_position: np.ndarray | None = None
    last_pose: Pose | None = None


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
    last_seen: float = 0.0  # frame stamp (s) of the last matched detection

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


def fresh_tracks(
    tracked: Sequence[TrackedCube],
    stamp: float,
    max_age_s: float,
) -> list[TrackedCube]:
    """
    Keep tracks seen within ``max_age_s`` of ``stamp`` (all when <= 0).

    A cube whose track split (new id after a jump) leaves the old track
    alive until it expires; counting it would make one cube two.
    """
    if max_age_s <= 0.0:
        return list(tracked)
    return [t for t in tracked if stamp - t.last_seen <= max_age_s]


class CubeTracker:
    """
    Track cubes by 3D position with greedy nearest-neighbour matching.

    Each track owns a :class:`SlotVoter` window (keyed by ``track_id``):
    majority colour, median position, circular-median yaw modulo 90°.
    Tracks unseen for more than ``max_missed`` frames are dropped, or, in
    time mode (``vote_window_s`` / ``max_missed_s`` set), unseen for more
    than ``max_missed_s`` seconds of frame stamps.

    An observation matches a track within ``match_dist_m`` in 3D, or, with
    ``match_max_dz_m``, within ``match_dist_m`` in the plane (x, y; z is up
    in the tracking frame) and ``match_max_dz_m`` in height: a cube whose
    height came out wrong in one frame (a bad table plane) stays on its
    track, and the median of its window drops that height. An unmatched
    observation within the matching distance of a track matched in the same
    frame is a second mask of that cube and starts no track.
    """

    def __init__(
        self,
        *,
        match_dist_m: float = 0.03,
        match_max_dz_m: float | None = None,
        max_missed: int = 5,
        vote_window: int = 10,
        vote_min_ratio: float = 0.7,
        vote_min_frames: int | None = None,
        vote_window_s: float | None = None,
        vote_min_votes: int = 1,
        max_missed_s: float | None = None,
    ) -> None:
        if match_dist_m <= 0.0:
            raise ValueError(f'match_dist_m must be > 0, got {match_dist_m}')
        if match_max_dz_m is not None and match_max_dz_m <= 0.0:
            raise ValueError(
                f'match_max_dz_m must be > 0 or None, got {match_max_dz_m}'
            )
        if max_missed < 0:
            raise ValueError(f'max_missed must be >= 0, got {max_missed}')
        if max_missed_s is not None and max_missed_s < 0.0:
            raise ValueError(f'max_missed_s must be >= 0, got {max_missed_s}')
        self.match_dist_m = float(match_dist_m)
        self.match_max_dz_m = (
            None if match_max_dz_m is None else float(match_max_dz_m)
        )
        self.max_missed = int(max_missed)
        self.max_missed_s = None if max_missed_s is None else float(max_missed_s)
        self._voter = SlotVoter(
            window=vote_window,
            min_ratio=vote_min_ratio,
            min_frames=vote_min_frames,
            window_s=vote_window_s,
            min_votes=vote_min_votes,
        )
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._latest: float | None = None

    def reset(self) -> None:
        """Forget all tracks and voting history."""
        self._tracks.clear()
        self._voter.reset()
        self._next_id = 1
        self._latest = None

    def update(
        self,
        results: Sequence[SlotResult],
        stamp: float | None = None,
    ) -> list[TrackedCube]:
        """
        Match framed detections to tracks and return active tracked cubes.

        Only observations with a pose participate in matching. Unmatched
        trusted observations spawn new tracks, except one within the
        matching distance of a track matched or started in this frame (a
        second mask of that cube); an untrusted one (``pose_trusted``
        False) only refreshes the track it matches.
        Returned list is sorted by track id.
        ``stamp`` (frame time, s) is required in time mode; a stamp older
        than the previous one resets all tracks (bag ``--loop``).
        """
        timed = self.max_missed_s is not None or self._voter.window_s is not None
        if timed and stamp is None:
            raise ValueError('time-based tracking needs a frame stamp')
        if stamp is not None and self._latest is not None and stamp < self._latest:
            self.reset()
        if stamp is not None:
            self._latest = float(stamp)
        now = 0.0 if stamp is None else float(stamp)
        if self._voter.window_s is not None:
            # Advance the voter clock so windows of unmatched tracks age too.
            self._voter.update([], stamp)

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
            track.last_seen = now
            track.last_detection = obs.detection
            # An untrusted pose keeps the track alive and votes its colour,
            # but neither moves it nor enters its pose median.
            if obs.pose_trusted:
                track.last_pose = obs.pose
                track.last_position = np.asarray(
                    obs.pose.position, dtype=np.float64
                ).copy()
            self._voter.update([
                SlotResult(
                    slot=tid,
                    label=obs.label,
                    score=obs.score,
                    detection=obs.detection,
                    pose=obs.pose if obs.pose_trusted else None,
                )
            ], stamp)

        # From the highest score: a second mask of a cube that a track took
        # in this frame, or that the first mask just started, is no new cube.
        taken = set(matched_tracks)
        unmatched = sorted(
            (
                i for i, obs in enumerate(observations)
                if i not in matched_obs and obs.pose_trusted
            ),
            key=lambda i: -float(observations[i].score),
        )
        for obs_i in unmatched:
            obs = observations[obs_i]
            if self._second_mask(obs, taken):
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid,
                missed=0,
                last_seen=now,
                last_detection=obs.detection,
                last_pose=obs.pose,
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
            ], stamp)
            taken.add(tid)

        for tid in track_ids:
            if tid in matched_tracks:
                continue
            self._tracks[tid].missed += 1

        dead = [
            tid for tid, tr in self._tracks.items()
            if self._expired(tr, now)
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
            # Until the vote window fills, VotedSlot.pose is None — keep the
            # last measured pose so slot_assign still sees the cube.
            pose = v.pose if v.pose is not None else tr.last_pose
            out.append(
                TrackedCube(
                    track_id=tid,
                    label=v.label,
                    score=v.score,
                    n_frames=v.n_frames,
                    pose=pose,
                    yaw_deg=v.yaw_deg,
                    yaw_consistency=v.yaw_consistency,
                    detection=tr.last_detection,
                    missed=tr.missed,
                    last_seen=tr.last_seen,
                )
            )
        return out

    def _expired(self, track: _Track, now: float) -> bool:
        """Check whether ``track`` was unseen too long (seconds or frames)."""
        if self.max_missed_s is not None:
            return now - track.last_seen > self.max_missed_s
        return track.missed > self.max_missed

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
                dist = self._gap(p, tr.last_position)
                if dist is not None:
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

    def _gap(
        self,
        position: np.ndarray,
        track_position: np.ndarray,
    ) -> float | None:
        """Return the matching distance, or None beyond the gate."""
        diff = np.asarray(position, dtype=np.float64) - track_position
        if self.match_max_dz_m is None:
            dist = float(np.linalg.norm(diff))
        elif abs(float(diff[2])) > self.match_max_dz_m:
            return None
        else:
            dist = float(np.hypot(diff[0], diff[1]))
        return dist if dist <= self.match_dist_m else None

    def _second_mask(
        self,
        obs: SlotResult,
        matched_tracks: set[int],
    ) -> bool:
        """Return True when a track matched in this frame is that close."""
        assert obs.pose is not None
        p = np.asarray(obs.pose.position, dtype=np.float64)
        for tid in matched_tracks:
            tr = self._tracks[tid]
            if tr.last_position is not None and self._gap(
                p, tr.last_position
            ) is not None:
                return True
        return False
