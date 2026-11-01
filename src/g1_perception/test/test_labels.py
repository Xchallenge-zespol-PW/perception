"""Tests for canonical label mapping and labels.yaml loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from g1_perception.core.config import load_label_map
from g1_perception.core.labels import canonical_label
from g1_perception.core.types import (
    LABEL_EMPTY,
    LABEL_GREEN,
    LABEL_RED,
    LABEL_UNCERTAIN,
)


def test_canonical_label_maps_known_names_case_insensitively():
    label_map = {'red cube': LABEL_RED, 'zielony': LABEL_GREEN}
    assert canonical_label('Red Cube ', label_map) == LABEL_RED
    assert canonical_label('ZIELONY', label_map) == LABEL_GREEN


def test_canonical_names_map_to_themselves():
    for label in (LABEL_RED, LABEL_GREEN, LABEL_EMPTY, LABEL_UNCERTAIN):
        assert canonical_label(label, {}) == label


def test_unknown_name_is_uncertain():
    assert canonical_label('blue cube', {'red cube': LABEL_RED}) == (
        LABEL_UNCERTAIN
    )


def test_load_label_map_normalizes_keys(tmp_path: Path):
    path = tmp_path / 'labels.yaml'
    path.write_text(
        'label_map:\n  "  Red Cube": red\n  Brak: EMPTY\n',
        encoding='utf-8',
    )
    assert load_label_map(path) == {'red cube': 'red', 'brak': 'empty'}


def test_load_label_map_rejects_non_canonical_value(tmp_path: Path):
    path = tmp_path / 'labels.yaml'
    path.write_text('label_map:\n  red cube: crimson\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid label'):
        load_label_map(path)


def test_load_label_map_rejects_conflicting_keys(tmp_path: Path):
    path = tmp_path / 'labels.yaml'
    path.write_text(
        'label_map:\n  red cube: red\n  RED CUBE: green\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='Conflicting'):
        load_label_map(path)


def test_package_labels_yaml_covers_model_names():
    label_map = load_label_map()
    for raw, expected in (
        ('red cube', LABEL_RED),
        ('green cube', LABEL_GREEN),
        ('czerwona', LABEL_RED),
        ('czerwony', LABEL_RED),
        ('zielona', LABEL_GREEN),
        ('zielony', LABEL_GREEN),
        ('nieznany', LABEL_UNCERTAIN),
    ):
        assert canonical_label(raw, label_map) == expected
