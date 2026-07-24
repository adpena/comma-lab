from __future__ import annotations

import math

import numpy as np
import pytest

from tac.optimization.ddm_metric_producers import (
    MetricProducerError,
    audit_pf2_bucket_assignments,
    composite_r_second_order_row,
    dual_metric_diagnostic_row,
    non_converged_pose_row,
    padded_batch32,
    pose_quadratic_row,
    seg_margin_fisher_row,
    validate_hard_pair_schedule,
)


def _pf2_rows(*, assigned: bool) -> dict:
    rows = []
    assignment = {
        "pair_ids": list(range(600)),
        "receiver_actuator_id": "receiver-A",
        "direction_id": "direction-A",
    }
    for index in range(1200):
        row = {"bucket_id": f"bucket-{index:04d}"}
        if assigned:
            row["measurement_assignment"] = assignment
        rows.append(row)
    return {"typed_split_atlas": {"rows": rows}}


def test_pf2_semantic_rows_are_not_measurement_assignments() -> None:
    audit = audit_pf2_bucket_assignments(_pf2_rows(assigned=False))
    assert audit.bucket_count == 1200
    assert audit.assigned_count == 0
    assert not audit.complete
    assert len(audit.unassigned_bucket_ids) == 1200


def test_pf2_assignment_requires_full_pair_and_direction_custody() -> None:
    value = _pf2_rows(assigned=True)
    assert audit_pf2_bucket_assignments(value).complete
    value["typed_split_atlas"]["rows"][17]["measurement_assignment"]["pair_ids"] = [0]
    audit = audit_pf2_bucket_assignments(value)
    assert audit.assigned_count == 0  # shared test mapping was invalidated everywhere


def test_pose_factor_is_exact_mean_squared_quadratic() -> None:
    row = pose_quadratic_row(
        7,
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        observed_against_registered_center_max_abs=1e-7,
    )
    factor = np.asarray(row["low_rank_factors"], dtype=np.float64)
    delta = np.asarray([0.7, -0.1, 0.2, 0.9, -0.3, 0.4])
    assert row["rank"] == 6
    assert row["tube_radius"] > 0.0
    assert np.linalg.matrix_rank(factor) == 6
    assert math.isclose(float(np.linalg.norm(factor.T @ delta) ** 2), float(np.mean(delta**2)))
    assert row["converged"] is True
    assert row["convergence_status"] == "CONVERGED"


def test_pose_nonconvergence_is_explicit() -> None:
    row = non_converged_pose_row(11, "SCORER_FORWARD")
    assert row["converged"] is False
    assert row["convergence_status"] == "NON_CONVERGED_SCORER_FORWARD"
    with pytest.raises(MetricProducerError):
        non_converged_pose_row(11, "not stable!")


def test_smoke_batch_is_exactly_32_and_deterministic() -> None:
    batch, padding = padded_batch32([523, 54, 1])
    assert len(batch) == 32
    assert batch[:3] == (523, 54, 1)
    assert padding == (0, *range(2, 30))
    assert len(set(batch)) == 32


def test_g3_schedule_preserves_hard_prefix_and_full_order() -> None:
    top64 = list(range(64))
    schedule = validate_hard_pair_schedule(
        {
            "top24": top64[:24],
            "top64": top64,
            "stratified_control24": list(range(100, 124)),
        }
    )
    assert schedule["top24"] == tuple(range(24))
    assert schedule["full_n600"] == tuple(range(600))
    broken = {
        "top24": top64[:24],
        "top64": [99, *top64[1:]],
        "stratified_control24": list(range(100, 124)),
    }
    with pytest.raises(MetricProducerError):
        validate_hard_pair_schedule(broken)


def test_seg_producer_preserves_pf2_key_and_computes_psd_gram() -> None:
    atlas_row = {
        "bucket_id": "cp01__target__visible__stationary__residual",
        "class_pair": "0-1",
        "class_stratum": "target",
        "visibility": "visible",
        "g4_temporal_class": "stationary",
        "representation_type": "residual",
    }
    source = np.arange(2400, dtype=np.float64).reshape(600, 4) / 2400.0
    row = seg_margin_fisher_row(atlas_row, source)
    gram = np.asarray(row["margin_fisher_gram"])
    assert row["bucket_id"] == atlas_row["bucket_id"]
    assert row["sample_count"] == 600
    assert gram.shape == (4, 4)
    assert np.linalg.eigvalsh(gram).min() >= -1e-10
    assert np.allclose(row["eigenvalues_ascending"], np.linalg.eigvalsh(gram))


def test_composite_r_producer_keeps_model_and_both_realized_secants() -> None:
    row = composite_r_second_order_row(
        "bucket-A",
        model_hessian=[[2.0, 0.5], [0.5, 1.0]],
        adjoint_readback=[1.0, -1.0],
        realized_secant_positive=[0.4, 0.3],
        realized_secant_negative=[-0.2, -0.5],
        secant_amplitude=0.25,
    )
    assert row["dimension"] == 2
    assert row["realized_secant_positive"] != row["realized_secant_negative"]
    with pytest.raises(MetricProducerError):
        composite_r_second_order_row(
            "bucket-A",
            model_hessian=[[1.0, 2.0], [0.0, 1.0]],
            adjoint_readback=[1.0, 1.0],
            realized_secant_positive=[1.0, 1.0],
            realized_secant_negative=[-1.0, -1.0],
            secant_amplitude=1.0,
        )


def test_dual_producer_retains_sign_and_control_only_label() -> None:
    row = dual_metric_diagnostic_row(
        "bucket-B",
        fisher_vector=[1.0, 0.0],
        euclidean_control_vector=[-1.0, 0.0],
    )
    assert row["fisher_euclidean_cosine"] == -1.0
    assert row["fisher_to_euclidean_rel_norm"] == 1.0
    assert row["euclidean_role"] == "LABELED_CONTROL_ONLY"
