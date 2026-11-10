"""Tests for the offline TF tree used on recorded bags."""

from __future__ import annotations

import numpy as np
import pytest

from g1_perception.core.tf_chain import TfChain

IDENTITY = (0.0, 0.0, 0.0, 1.0)
YAW_90 = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))


def test_chain_composes_through_the_root():
    tf = TfChain()
    tf.add('pelvis', 'torso', 0.0, (0.0, 0.0, 0.3), YAW_90, static=True)
    tf.add('torso', 'camera', 1.0, (0.1, 0.0, 0.0), IDENTITY)
    tf.add('pelvis', 'foot', 1.0, (0.0, 0.1, -0.7), IDENTITY)

    t, q = tf.lookup('pelvis', 'camera', 1.0)
    assert np.allclose(t, [0.0, 0.1, 0.3])
    assert np.allclose(np.abs(q), np.abs(YAW_90))
    t, _q = tf.lookup('foot', 'camera', 1.0)
    assert np.allclose(t, [0.0, 0.0, 1.0])


def test_nearest_stamp_and_max_dt():
    tf = TfChain(max_dt_s=0.05)
    tf.add('a', 'b', 1.0, (1.0, 0.0, 0.0), IDENTITY)
    tf.add('a', 'b', 2.0, (2.0, 0.0, 0.0), IDENTITY)

    assert np.isclose(tf.lookup('a', 'b', 1.04)[0][0], 1.0)
    assert np.isclose(tf.lookup('a', 'b', 1.97)[0][0], 2.0)
    with pytest.raises(LookupError):
        tf.lookup('a', 'b', 1.5)


def test_different_trees_raise():
    tf = TfChain()
    tf.add('a', 'b', 0.0, (0.0, 0.0, 0.0), IDENTITY, static=True)
    tf.add('c', 'd', 0.0, (0.0, 0.0, 0.0), IDENTITY, static=True)
    with pytest.raises(LookupError):
        tf.lookup('a', 'd', 0.0)
