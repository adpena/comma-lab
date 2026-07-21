# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.optimization.predict_project_receiver import predict_cell_field
from tac.optimization.predict_project_schema import derive_morse_smale_raster, serialize_constraint_seed
from tac.optimization.s2_partition_seed import PartitionEvent
from tac.optimization.seed_compose_b2 import (
    SCORER_HEIGHT,
    SCORER_WIDTH,
    _constraints,
    _ordered_with_pair_coverage,
    _predict_prevalidated,
    _seed_base,
    _trajectory,
    compatibility_site_chart,
)


def _five_stripe_mode() -> np.ndarray:
    mode = np.zeros((SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    for class_id, columns in enumerate(np.array_split(np.arange(SCORER_WIDTH), 5)):
        mode[:, columns] = class_id
    return mode


def test_compact_compatibility_chart_roundtrips_and_fast_predictor_is_exact() -> None:
    chart = compatibility_site_chart(_five_stripe_mode())
    gt_poses = np.zeros((600, 6), dtype=np.float64)
    trajectory, _ = _trajectory(gt_poses, 0, s_t=-0.00143, s_r=0.0)
    seed = _seed_base(chart, [], trajectory)
    raster = np.frombuffer(derive_morse_smale_raster(chart), dtype=np.uint8).reshape(
        SCORER_HEIGHT, SCORER_WIDTH
    )
    assert len(serialize_constraint_seed(seed)) < 100_000
    for pair in (0, 300, 599):
        assert np.array_equal(_predict_prevalidated(seed, raster, pair), predict_cell_field(seed, pair))


def test_compatibility_chart_uses_full_occupancy_centroid_when_mode_omits_class() -> None:
    mode = np.zeros((SCORER_HEIGHT, SCORER_WIDTH), dtype=np.uint8)
    centroids = {class_id: (class_id * 256, class_id * 256) for class_id in range(5)}
    chart = compatibility_site_chart(mode, occupancy_centroids_q=centroids)
    assert [cell["site_y_q"] for cell in chart["cells"]][1:] == [class_id * 256 for class_id in range(1, 5)]


def test_common_event_order_is_nested_pair_complete_and_pose_tubes_are_n600() -> None:
    rows = []
    for pair in range(600):
        event = PartitionEvent(pair, pair % SCORER_HEIGHT, pair % SCORER_WIDTH, 1, 0)
        rows.append(
            {
                "event": event,
                "target": 1,
                "margin": float(pair + 1),
                "stratum": "boundary_codim1",
                "rank": (0, 0, float(pair + 1), pair, event.row, event.col),
            }
        )
    ordered = _ordered_with_pair_coverage(rows)
    pose_q = np.arange(600 * 6, dtype=np.int64).reshape(600, 6)
    constraints = _constraints(ordered, pose_q, radius_q=7)
    assert len(constraints) == 600
    assert all(row["pose_tube"] is not None for row in constraints)
    for pair, row in enumerate(constraints):
        assert row["pose_tube"]["lower_q"] == (pose_q[pair] - 7).tolist()
        assert row["pose_tube"]["upper_q"] == (pose_q[pair] + 7).tolist()


def test_xi_curve_has_data_derived_nested_control_and_residual_quantization() -> None:
    gt_poses = np.zeros((600, 6), dtype=np.float64)
    gt_poses[:, 0] = np.sin(np.linspace(0.0, 8.0, 600))
    gt_poses[:, 2] = np.cos(np.linspace(0.0, 5.0, 600))
    rows = [_trajectory(gt_poses, index, s_t=-0.00143, s_r=0.0)[1] for index in range(3)]
    assert [row["control_count"] for row in rows] == sorted(row["control_count"] for row in rows)
    assert [row["residual_quantum_q"] for row in rows] == sorted(
        (row["residual_quantum_q"] for row in rows), reverse=True
    )
    assert rows[-1]["residual_quantum_q"] == 1
