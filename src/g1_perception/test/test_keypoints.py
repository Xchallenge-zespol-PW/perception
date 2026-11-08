"""Tests for the canonical top-face corner order."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from g1_perception.core.keypoints import is_clockwise, order_top_corners

# Diamond (cube rotated ~45°): bottom corner is a clear c0.
DIAMOND = np.array([[50.0, 10.0], [90.0, 50.0], [50.0, 90.0], [10.0, 50.0]])


def test_c0_is_lowest_corner_and_order_is_clockwise():
    ordered = order_top_corners(DIAMOND)
    np.testing.assert_allclose(ordered[0], [50.0, 90.0])
    # Clockwise on screen from the bottom: left, top, right.
    np.testing.assert_allclose(ordered[1], [10.0, 50.0])
    np.testing.assert_allclose(ordered[2], [50.0, 10.0])
    np.testing.assert_allclose(ordered[3], [90.0, 50.0])
    assert is_clockwise(ordered)


def test_input_order_does_not_matter():
    expected = order_top_corners(DIAMOND)
    for perm in itertools.permutations(range(4)):
        np.testing.assert_allclose(order_top_corners(DIAMOND[list(perm)]), expected)


def test_near_tie_picks_larger_u():
    # Edge facing the camera: bottom corners differ by 2 px in v (< 3 px).
    square = np.array([[10.0, 10.0], [60.0, 11.0], [61.0, 60.0], [9.0, 62.0]])
    ordered = order_top_corners(square)
    np.testing.assert_allclose(ordered[0], [61.0, 60.0])
    np.testing.assert_allclose(ordered[1], [9.0, 62.0])
    assert is_clockwise(ordered)


def test_beyond_tie_picks_largest_v():
    square = np.array([[10.0, 10.0], [60.0, 11.0], [61.0, 58.0], [9.0, 62.0]])
    np.testing.assert_allclose(order_top_corners(square)[0], [9.0, 62.0])


def test_is_clockwise_detects_reversed_order():
    assert not is_clockwise(order_top_corners(DIAMOND)[::-1])


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match='shape'):
        order_top_corners(np.zeros((3, 2)))
