"""Offline TF tree from recorded /tf and /tf_static (no ROS, nearest stamp)."""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from g1_perception.core.geometry import (
    quat_to_rotation_matrix,
    rotation_matrix_to_quat_xyzw,
)


class TfChain:
    """
    Look up ``target <- source`` at a stamp from recorded transforms.

    Every edge ``parent -> child`` keeps its samples sorted by stamp; a
    lookup takes, on each edge of the path, the sample nearest to the stamp
    (no interpolation) and fails when it is more than ``max_dt_s`` away.
    Static edges match any stamp.
    """

    def __init__(self, max_dt_s: float = 0.05) -> None:
        self.max_dt_s = float(max_dt_s)
        self._parent: dict[str, str] = {}
        self._stamps: dict[str, list[float]] = {}
        self._mats: dict[str, list[NDArray[np.float64]]] = {}
        self._static: dict[str, NDArray[np.float64]] = {}

    def add(
        self,
        parent: str,
        child: str,
        stamp: float,
        translation: Sequence[float],
        quat_xyzw: Sequence[float],
        *,
        static: bool = False,
    ) -> None:
        """Record ``parent -> child`` (``p_parent = R p_child + t``)."""
        parent, child = parent.lstrip('/'), child.lstrip('/')
        mat = np.eye(4)
        mat[:3, :3] = quat_to_rotation_matrix(np.asarray(quat_xyzw, float))
        mat[:3, 3] = np.asarray(translation, dtype=np.float64)
        self._parent[child] = parent
        if static:
            self._static[child] = mat
            return
        stamps = self._stamps.setdefault(child, [])
        mats = self._mats.setdefault(child, [])
        i = bisect.bisect(stamps, stamp)
        stamps.insert(i, float(stamp))
        mats.insert(i, mat)

    def lookup(
        self,
        target: str,
        source: str,
        stamp: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(translation, rotation_xyzw)`` of ``target <- source``."""
        target, source = target.lstrip('/'), source.lstrip('/')
        root_src, root_s = self._to_root(source, stamp)
        root_tgt, root_t = self._to_root(target, stamp)
        if root_s != root_t:
            raise LookupError(
                f'{source} and {target} are in different trees '
                f'({root_s}, {root_t})'
            )
        mat = np.linalg.inv(root_tgt) @ root_src
        return mat[:3, 3].copy(), rotation_matrix_to_quat_xyzw(mat[:3, :3])

    def _to_root(
        self,
        frame: str,
        stamp: float,
    ) -> tuple[NDArray[np.float64], str]:
        """Return ``root <- frame`` and the root frame name."""
        mat = np.eye(4)
        seen = {frame}
        while frame in self._parent:
            mat = self._edge(frame, stamp) @ mat
            frame = self._parent[frame]
            if frame in seen:
                raise LookupError(f'TF loop at {frame}')
            seen.add(frame)
        return mat, frame

    def _edge(self, child: str, stamp: float) -> NDArray[np.float64]:
        if child in self._static:
            return self._static[child]
        stamps = self._stamps[child]
        i = bisect.bisect(stamps, stamp)
        best = min(
            (j for j in (i - 1, i) if 0 <= j < len(stamps)),
            key=lambda j: abs(stamps[j] - stamp),
        )
        if abs(stamps[best] - stamp) > self.max_dt_s:
            raise LookupError(
                f'no TF {self._parent[child]} -> {child} within '
                f'{self.max_dt_s * 1000:.0f} ms of {stamp:.3f}'
            )
        return self._mats[child][best]
