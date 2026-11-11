"""Freeze the start layout of the cubes and keep their names during a task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from g1_perception.core.slot_assign import AssignmentResult
from g1_perception.core.tracker import TrackedCube
from g1_perception.core.types import COLOR_LABELS, Pose

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


@dataclass(frozen=True)
class NamedCube:
    """
    One cube of the frozen start layout.

    ``name`` is ``<prefix><place>``, ``label`` the colour seen at the start,
    ``start_pose`` the voted pose at the start. ``pose`` is the pose seen in
    the last update, or None when the cube was not seen in it.
    ``last_pose`` is the pose it was last seen at (the start pose until it
    is seen again), also while it is out of view. ``track_id`` is the track
    the name is bound to. All poses are in the frame of the tracked cubes
    passed to :meth:`CubeRegistry.update`.
    """

    name: str
    label: str
    start_pose: Pose
    pose: Pose | None
    track_id: int | None
    last_pose: Pose | None = None


@dataclass
class _Binding:
    name: str
    label: str
    start_pose: Pose
    track_id: int | None
    last_pose: Pose

    def named(self, pose: Pose | None) -> NamedCube:
        return NamedCube(
            self.name, self.label, self.start_pose, pose, self.track_id,
            self.last_pose,
        )


class CubeRegistry:
    """
    Name the cubes once at the start and follow them while they are pushed.

    Before the start layout is frozen, the layout has to stay usable (valid
    place assignment, only red / green labels, poses) with the same tracks,
    places and labels for ``latch_s`` seconds of frame stamps, so the
    decision does not depend on the frame rate. Then the layout is frozen:
    place ``n`` becomes the cube ``<name_prefix>n`` with its colour and
    start pose, and :attr:`start_id` grows by one.

    While frozen, a name follows its track. A name whose track was not seen
    in an update (occluded, dropped by the tracker, or seen with the other
    colour) is given to an unnamed track seen in that update whose
    coordinate along ``axis`` (the line of the cubes) is within
    ``reid_gate_m`` of the start one. Cubes are pushed across that line, so
    the coordinate barely changes, while neighbours are a cube size away.
    Colour only vetoes a match once the track has voted it, because a new
    track is ``uncertain`` until it has enough votes. Every name keeps the
    pose it was last seen at (:meth:`known`), so a cube out of view (the
    camera turned away with the torso) still has a position.

    A stamp older than the previous one (bag played with ``--loop``) makes
    :class:`CubeTracker` restart its track ids, so every name then forgets
    its track and is found again by position.
    """

    def __init__(
        self,
        *,
        axis: str = 'x',
        name_prefix: str = 'cube_',
        latch_s: float = 1.0,
        reid_gate_m: float = 0.025,
    ) -> None:
        axis_key = str(axis).strip().lower()
        if axis_key not in _AXIS_INDEX:
            raise ValueError(f'axis must be x/y/z, got {axis!r}')
        if not name_prefix:
            raise ValueError('name_prefix must not be empty')
        if latch_s < 0.0:
            raise ValueError(f'latch_s must be >= 0, got {latch_s}')
        if reid_gate_m <= 0.0:
            raise ValueError(f'reid_gate_m must be > 0, got {reid_gate_m}')
        self._axis = _AXIS_INDEX[axis_key]
        self.name_prefix = str(name_prefix)
        self.latch_s = float(latch_s)
        self.reid_gate_m = float(reid_gate_m)
        self._bindings: list[_Binding] = []
        self._candidate: tuple[tuple[int, int | None, str], ...] | None = None
        self._candidate_since = 0.0
        self._latest: float | None = None
        self._start_id = 0

    @property
    def latched(self) -> bool:
        """Return True while a start layout is frozen."""
        return bool(self._bindings)

    @property
    def start_id(self) -> int:
        """Return the number of start layouts frozen so far (0 = none yet)."""
        return self._start_id

    def known(self) -> list[NamedCube]:
        """
        Return the cubes of the frozen layout at their last seen poses.

        ``pose`` is None (nothing new seen), ``last_pose`` is set for every
        cube. Empty before the layout is frozen.
        """
        return [b.named(None) for b in self._bindings]

    def reset(self) -> None:
        """Forget the frozen layout and wait for a new one (keeps start_id)."""
        self._bindings = []
        self._candidate = None

    def update(
        self,
        assignment: AssignmentResult,
        tracked: Sequence[TrackedCube],
        stamp: float,
    ) -> list[NamedCube]:
        """
        Process one frame taken at ``stamp`` (s) and return the named cubes.

        The cubes are in place order. Returns an empty list until the start
        layout is frozen; in the update that freezes it every cube is seen
        at its start pose.
        """
        stamp = float(stamp)
        if self._latest is not None and stamp < self._latest:
            self._candidate = None
            for binding in self._bindings:
                binding.track_id = None
        self._latest = stamp

        if not self._bindings:
            if not self._try_latch(assignment, stamp):
                return []
            return [b.named(b.start_pose) for b in self._bindings]
        return self._follow(tracked)

    def _try_latch(self, assignment: AssignmentResult, stamp: float) -> bool:
        """Freeze the layout once it has been stable for latch_s."""
        slots = assignment.slots
        usable = assignment.valid and bool(slots) and all(
            s.label in COLOR_LABELS and s.pose is not None for s in slots
        )
        if not usable:
            self._candidate = None
            return False

        candidate = tuple((s.slot, s.track_id, s.label) for s in slots)
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = stamp
        if stamp - self._candidate_since < self.latch_s:
            return False

        self._bindings = [
            _Binding(
                name=f'{self.name_prefix}{s.slot}',
                label=s.label,
                start_pose=s.pose,  # type: ignore[arg-type]
                track_id=s.track_id,
                last_pose=s.pose,  # type: ignore[arg-type]
            )
            for s in slots
        ]
        self._start_id += 1
        self._candidate = None
        return True

    def _follow(self, tracked: Sequence[TrackedCube]) -> list[NamedCube]:
        """Update the bound tracks and give lost names to new tracks."""
        visible = {
            t.track_id: t for t in tracked
            if t.missed == 0 and t.pose is not None
        }
        seen: dict[str, TrackedCube] = {}
        for binding in self._bindings:
            track = visible.get(binding.track_id)  # type: ignore[arg-type]
            if track is not None and not _contradicts(track, binding.label):
                seen[binding.name] = track

        used = {t.track_id for t in seen.values()}
        free = [t for tid, t in visible.items() if tid not in used]
        candidates: list[tuple[float, int, int]] = []
        for b_index, binding in enumerate(self._bindings):
            if binding.name in seen:
                continue
            start = float(binding.start_pose.position[self._axis])
            for t_index, track in enumerate(free):
                if _contradicts(track, binding.label):
                    continue
                assert track.pose is not None
                offset = abs(float(track.pose.position[self._axis]) - start)
                if offset <= self.reid_gate_m:
                    candidates.append((offset, b_index, t_index))

        candidates.sort()
        taken: set[int] = set()
        for _offset, b_index, t_index in candidates:
            binding = self._bindings[b_index]
            if binding.name in seen or t_index in taken:
                continue
            taken.add(t_index)
            binding.track_id = free[t_index].track_id
            seen[binding.name] = free[t_index]

        cubes = []
        for binding in self._bindings:
            pose = seen[binding.name].pose if binding.name in seen else None
            if pose is not None:
                binding.last_pose = pose
            cubes.append(binding.named(pose))
        return cubes


def _contradicts(track: TrackedCube, label: str) -> bool:
    """Return True when the track has voted the other colour."""
    return track.label in COLOR_LABELS and track.label != label
