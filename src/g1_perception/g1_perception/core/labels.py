"""Map raw segmenter / model class names to canonical slot labels."""

from __future__ import annotations

from collections.abc import Mapping

from g1_perception.core.types import CANONICAL_LABELS, LABEL_UNCERTAIN


def normalize_label_key(raw: str) -> str:
    """Normalize a class name for lookup (trimmed, lower-case)."""
    return str(raw).strip().lower()


def canonical_label(raw: str, label_map: Mapping[str, str]) -> str:
    """
    Return the canonical label for ``raw``.

    ``label_map`` keys must already be normalized (see
    :func:`normalize_label_key`). Canonical names map to themselves even if
    absent from the map; anything else unknown becomes ``uncertain``.
    """
    key = normalize_label_key(raw)
    if key in label_map:
        return label_map[key]
    if key in CANONICAL_LABELS:
        return key
    return LABEL_UNCERTAIN
