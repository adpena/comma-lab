# SPDX-License-Identifier: MIT
"""NO-FAKE behavior tests for the REAL reversible pose-trajectory carrier codec.

These verify the coder's actual byte output and bit-exact roundtrip on hand-built
trajectories with KNOWN structure, never an asserted constant. If
``encode_pose_trajectory`` / ``decode_pose_trajectory`` were replaced by a stub the
roundtrip assertions would fail. The codec underpins the sufficient-statistic floor
probe's pose-H term (``experiments/sufficient_statistic_floor_probe.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.pose_trajectory_entropy import (
    decode_pose_trajectory,
    encode_pose_trajectory,
    induced_d_pose_from_steps,
    pose_carrier_real_bytes,
)


def _quant(traj: np.ndarray, deltas) -> np.ndarray:
    dv = np.full(traj.shape[1], float(deltas)) if np.isscalar(deltas) else np.asarray(deltas)
    return np.round(traj / dv).astype(np.int64)


def test_roundtrip_bit_exact_smooth_trajectory():
    rng = np.random.default_rng(7)
    traj = np.cumsum(rng.normal(0, 0.02, (200, 6)), axis=0)
    deltas = np.full(6, 0.01)
    payload = encode_pose_trajectory(traj, deltas=deltas)
    out = decode_pose_trajectory(payload)
    assert np.array_equal(out, _quant(traj, deltas))


def test_roundtrip_bit_exact_per_dim_steps():
    rng = np.random.default_rng(11)
    traj = np.cumsum(rng.normal(0, 0.05, (150, 6)) * np.array([1, 0.1, 0.05, 0.03, 0.02, 0.04]), axis=0)
    deltas = np.array([0.05, 0.005, 0.0025, 0.0015, 0.001, 0.002])
    nbytes, quant = pose_carrier_real_bytes(traj, deltas=deltas)
    assert nbytes == len(encode_pose_trajectory(traj, deltas=deltas))
    assert np.array_equal(quant, _quant(traj, deltas))


def test_constant_trajectory_size_one_alphabet_roundtrips():
    # Every temporal delta is 0 -> size-1 alphabet per dim. Must still roundtrip
    # (the size-1-alphabet guard) and stay tiny (only seeds + model headers).
    traj = np.full((100, 6), 0.5)
    nbytes, quant = pose_carrier_real_bytes(traj, deltas=0.01)
    assert np.array_equal(quant, _quant(traj, 0.01))
    assert nbytes < 400  # no stream payload; just 6 small per-dim headers


def test_single_pair_edge_case():
    traj = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    nbytes, quant = pose_carrier_real_bytes(traj, deltas=0.01)
    assert quant.shape == (1, 6)
    assert np.array_equal(quant, _quant(traj, 0.01))


def test_mixed_constant_and_varying_dims_roundtrip():
    rng = np.random.default_rng(3)
    cols = []
    for k in range(6):
        if k == 2:
            cols.append(np.full(120, 0.3))  # a constant dim mixed in
        else:
            cols.append(np.cumsum(rng.normal(0, 0.02, 120)))
    traj = np.column_stack(cols)
    _, quant = pose_carrier_real_bytes(traj, deltas=0.01)
    assert np.array_equal(quant, _quant(traj, 0.01))


def test_smoother_trajectory_codes_smaller():
    # A smoother (smaller-delta) trajectory must cost FEWER real coded bytes than a
    # noisier one at the same step -> the coder actually exploits temporal redundancy
    # (this fails if the body returns a constant byte count).
    rng = np.random.default_rng(5)
    smooth = np.cumsum(rng.normal(0, 0.005, (300, 6)), axis=0)
    noisy = np.cumsum(rng.normal(0, 0.05, (300, 6)), axis=0)
    nb_smooth, _ = pose_carrier_real_bytes(smooth, deltas=0.01)
    nb_noisy, _ = pose_carrier_real_bytes(noisy, deltas=0.01)
    assert nb_smooth < nb_noisy


def test_finer_step_holds_lower_d_pose_and_costs_more_bytes():
    # Finer quantization -> lower induced d_pose AND more real coded bytes (the
    # rate/distortion tradeoff is real, not asserted).
    rng = np.random.default_rng(9)
    traj = np.cumsum(rng.normal(0, 0.02, (300, 6)), axis=0)
    coarse = np.full(6, 0.02)
    fine = np.full(6, 0.002)
    nb_coarse, _ = pose_carrier_real_bytes(traj, deltas=coarse)
    nb_fine, _ = pose_carrier_real_bytes(traj, deltas=fine)
    assert induced_d_pose_from_steps(fine) < induced_d_pose_from_steps(coarse)
    assert nb_fine > nb_coarse


def test_decode_rejects_bad_magic():
    with pytest.raises(ValueError):
        decode_pose_trajectory(b"XXXX" + b"\x00" * 8)


def test_encode_rejects_bad_shape():
    with pytest.raises(ValueError):
        encode_pose_trajectory(np.zeros((100,)), deltas=0.01)


def test_encode_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        encode_pose_trajectory(np.zeros((10, 6)), deltas=0.0)
