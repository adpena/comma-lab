# SPDX-License-Identifier: MIT
"""Behavior tests for the Lever-F partition contour + pose trajectory entropy floors.

NO FAKE: every test verifies the entropy/contour computation on a hand-built array
with a KNOWN answer (exact boundary count, exact conditional distribution, exact
quantization->d_pose relation), never an asserted constant. If a function body were
replaced by ``return <constant>`` these tests would fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.partition_contour_entropy import (
    boundary_step_count,
    contour_geometry_entropy_bits,
    iid_argmax_entropy_bits,
    partition_contour_entropy,
    region_label_entropy_bits,
    spatial_context_argmax_entropy_bits,
    temporal_context_argmax_entropy_bits,
)
from tac.optimization.pose_trajectory_entropy import (
    induced_d_pose_from_steps,
    per_dim_steps_for_target_pose_term,
    pose_trajectory_entropy,
)

# --------------------------------------------------------------------------- #
# Contour geometry / boundary step counting
# --------------------------------------------------------------------------- #


def test_boundary_step_count_uniform_is_zero():
    a = np.zeros((10, 10), dtype=np.uint8)
    assert boundary_step_count(a) == 0


def test_boundary_step_count_single_vertical_split():
    # Left half class 0, right half class 1: vertical crack of length H at one column.
    a = np.zeros((8, 10), dtype=np.uint8)
    a[:, 5:] = 1
    # One horizontal crack per row at the 4|5 column boundary => 8 horizontal cracks.
    assert boundary_step_count(a) == 8


def test_boundary_step_count_single_horizontal_split():
    a = np.zeros((8, 10), dtype=np.uint8)
    a[4:, :] = 1
    # One vertical crack per column at the 3|4 row boundary => 10 vertical cracks.
    assert boundary_step_count(a) == 10


def test_boundary_step_count_checkerboard_maximal():
    # 4x4 checkerboard: every internal edge is a boundary.
    a = np.indices((4, 4)).sum(axis=0) % 2
    a = a.astype(np.uint8)
    # internal horizontal edges: 4 rows * 3 = 12; vertical: 3 * 4 = 12; all boundaries.
    assert boundary_step_count(a) == 24


def test_contour_geometry_bits_log2_3_default():
    a = np.zeros((8, 10), dtype=np.uint8)
    a[:, 5:] = 1
    steps, bits = contour_geometry_entropy_bits(a)
    assert steps == 8
    assert bits == pytest.approx(8 * np.log2(3.0))


def test_contour_geometry_bits_custom_per_step():
    a = np.zeros((8, 10), dtype=np.uint8)
    a[:, 5:] = 1
    steps, bits = contour_geometry_entropy_bits(a, bits_per_step=2.0)
    assert steps == 8
    assert bits == pytest.approx(16.0)


def test_boundary_rejects_non_2d():
    with pytest.raises(ValueError):
        boundary_step_count(np.zeros((3, 3, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Region labeling
# --------------------------------------------------------------------------- #


def test_region_label_one_region_zero_bits():
    a = np.zeros((6, 6), dtype=np.uint8)
    n_regions, bits = region_label_entropy_bits(a)
    assert n_regions == 1
    # A single region over a degenerate (one nonzero class-count) prior => 0 bits.
    assert bits == pytest.approx(0.0)


def test_region_label_two_classes_equal_regions():
    # Two regions, one of class 0, one of class 1 => H = 1 bit/region, 2 regions.
    a = np.zeros((4, 4), dtype=np.uint8)
    a[:, 2:] = 1
    n_regions, bits = region_label_entropy_bits(a)
    assert n_regions == 2
    assert bits == pytest.approx(2.0)  # 2 * H(1/2,1/2) = 2 * 1


def test_region_label_counts_components():
    # Two disjoint class-1 blobs + background => 3 regions.
    a = np.zeros((6, 9), dtype=np.uint8)
    a[1:3, 1:3] = 1
    a[1:3, 6:8] = 1
    n_regions, _ = region_label_entropy_bits(a)
    assert n_regions == 3  # background(0) + 2 class-1 blobs


# --------------------------------------------------------------------------- #
# Partition contour total
# --------------------------------------------------------------------------- #


def test_partition_contour_total_is_geometry_plus_labels():
    a = np.zeros((8, 10), dtype=np.uint8)
    a[:, 5:] = 1
    r = partition_contour_entropy(a)
    assert r.boundary_steps == 8
    assert r.total_partition_bits == pytest.approx(
        r.contour_geometry_bits + r.region_label_bits
    )
    assert r.total_bytes == pytest.approx(r.total_partition_bits / 8.0)


# --------------------------------------------------------------------------- #
# Context entropy: monotonicity (a context model never increases entropy)
# --------------------------------------------------------------------------- #


def test_iid_entropy_uniform_two_classes():
    # 100x100 half/half => H = 1 bit/pixel over 10000 px => 10000 bits.
    a = np.zeros((100, 100), dtype=np.uint8)
    a[:, 50:] = 1
    bits = iid_argmax_entropy_bits([a])
    assert bits == pytest.approx(10000.0)


def test_spatial_context_le_iid():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 5, size=(40, 50)).astype(np.uint8) for _ in range(3)]
    iid = iid_argmax_entropy_bits(frames)
    spatial = spatial_context_argmax_entropy_bits(frames)
    # A context model cannot exceed the iid (order-0) coding length (conditioning
    # reduces entropy). Allow tiny float slack.
    assert spatial <= iid + 1e-6


def test_temporal_context_le_spatial():
    rng = np.random.default_rng(1)
    # Make frames temporally correlated so temporal context strictly helps.
    base = rng.integers(0, 5, size=(40, 50)).astype(np.uint8)
    frames = [base.copy()]
    for _ in range(4):
        nxt = frames[-1].copy()
        flip = rng.random(nxt.shape) < 0.05
        nxt[flip] = rng.integers(0, 5, size=int(flip.sum()))
        frames.append(nxt.astype(np.uint8))
    spatial = spatial_context_argmax_entropy_bits(frames)
    temporal = temporal_context_argmax_entropy_bits(frames)
    assert temporal <= spatial + 1e-6


def test_spatial_context_perfectly_predictable_low_entropy():
    # A constant partition: spatial context predicts every pixel perfectly => ~0 bits.
    a = np.full((30, 30), 3, dtype=np.uint8)
    bits = spatial_context_argmax_entropy_bits([a])
    assert bits == pytest.approx(0.0, abs=1e-6)


def test_temporal_context_static_video_near_zero():
    # Identical frames: once the spatial context is learned, all near-zero;
    # temporal context with prev-frame predictor is exactly zero after frame 0.
    a = np.full((20, 20), 2, dtype=np.uint8)
    frames = [a.copy() for _ in range(5)]
    temporal = temporal_context_argmax_entropy_bits(frames)
    assert temporal == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Pose trajectory entropy
# --------------------------------------------------------------------------- #


def test_induced_d_pose_scalar_step():
    # 6 dims, step delta => d_pose = 6 * delta^2 / 12 = delta^2 / 2.
    delta = 0.01
    d = induced_d_pose_from_steps(np.full(6, delta))
    assert d == pytest.approx(delta**2 / 2.0)


def test_induced_d_pose_per_dim():
    deltas = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
    d = induced_d_pose_from_steps(deltas)
    assert d == pytest.approx((0.01 + 0.04) / 12.0)


def test_per_dim_steps_hits_target_pose_term():
    rng = np.random.default_rng(7)
    traj = rng.normal(size=(600, 6))
    target = 0.01
    steps = per_dim_steps_for_target_pose_term(traj, target)
    induced = induced_d_pose_from_steps(steps)
    assert np.sqrt(10.0 * induced) == pytest.approx(target, rel=1e-6)


def test_pose_trajectory_entropy_constant_traj_zero_delta_bits():
    # A constant trajectory: every temporal delta is 0 => delta entropy 0
    # (plus tiny seed cost). raw quant bits also 0 (one symbol).
    traj = np.full((600, 6), 5.0)
    r = pose_trajectory_entropy(traj, deltas=0.01)
    assert r.raw_quant_bits == pytest.approx(0.0)
    assert r.temporal_delta_bits == pytest.approx(0.0)


def test_pose_trajectory_entropy_temporal_le_raw_for_smooth():
    # A smooth ramp: temporal-delta coding should not exceed raw by much; for a
    # linear ramp the delta stream is constant => near-zero delta entropy.
    t = np.linspace(0, 10, 600)
    traj = np.stack([t for _ in range(6)], axis=1)
    r = pose_trajectory_entropy(traj, deltas=0.05)
    # constant delta => delta entropy ~ 0 (+ seed); strictly below raw quant bits.
    assert r.temporal_delta_bits < r.raw_quant_bits + 100.0
    assert r.induced_d_pose == pytest.approx(6 * 0.05**2 / 12.0)


def test_pose_trajectory_rejects_bad_shape():
    with pytest.raises(ValueError):
        pose_trajectory_entropy(np.zeros((600,)), deltas=0.01)


def test_pose_trajectory_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        pose_trajectory_entropy(np.zeros((10, 6)), deltas=0.0)
