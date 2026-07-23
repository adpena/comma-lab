# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.iterative_realized_trust_region import (
    HardCandidate,
    IterativeRealizedTrustError,
    TemperingStatus,
    TemplateBasis,
    TrustRegionPolicy,
    TrustUpdate,
    bounded_parallel_tempering,
    build_template_basis_projection,
    categorical_fisher_trace_from_margin,
    fisher_margin_debt,
    quantized_babai_candidates,
    ranked_prefix_sign_candidates,
    select_realized_improvement,
    summarize_validity_curve,
    update_trust_radius,
)


def test_template_basis_projections_preserve_integer_receiver_lattice() -> None:
    values = np.asarray([[[[10, 20, 30]] * 2] * 2], dtype=np.uint8)
    compensation = np.asarray([[1, -2, 3]], dtype=np.int8)
    expected_latent = {
        TemplateBasis.ROWBAND_1X1_CONTROL: 3,
        TemplateBasis.CONTEXTUAL_2X2: 12,
        TemplateBasis.BOUNDARY_NORMAL_2X2: 6,
    }
    for basis, template_count in expected_latent.items():
        axes = ("x",) if basis is TemplateBasis.BOUNDARY_NORMAL_2X2 else ()
        projection = build_template_basis_projection(values, compensation, basis=basis, boundary_axes=axes)
        assert projection.template_latent_count == template_count
        step = np.ones(projection.current.size, dtype=np.int64)
        lifted = projection.lift_step(step)
        assert lifted.shape == (values.size + compensation.size,)
        assert np.array_equal(lifted, np.ones_like(lifted))


def test_boundary_normal_projection_rejects_state_outside_subspace() -> None:
    values = np.asarray([[[[10, 20, 30], [11, 20, 30]], [[10, 20, 30], [12, 20, 30]]]], dtype=np.uint8)
    with pytest.raises(IterativeRealizedTrustError, match="side-equality"):
        build_template_basis_projection(
            values,
            np.zeros((0, 3), dtype=np.int8),
            basis=TemplateBasis.BOUNDARY_NORMAL_2X2,
            boundary_axes=("x",),
        )


def test_ranked_prefix_sign_candidates_are_unique_and_equal_budget() -> None:
    step = np.asarray([-9.0, 8.0, -7.0, 6.0, -5.0, 4.0])
    rows = ranked_prefix_sign_candidates(
        step,
        current=np.zeros(6, dtype=np.int64),
        lower=np.full(6, -20, dtype=np.int64),
        upper=np.full(6, 20, dtype=np.int64),
        trust_radii=(1.0, 2.0, 4.0, 8.0),
    )
    assert len(rows) == 4
    assert len({row.integer_step.tobytes() for row in rows}) == 4
    assert [int(np.max(np.abs(row.integer_step))) for row in rows] == [1, 2, 4, 8]
    assert all(np.sign(row.integer_step[0]) == -1 for row in rows)


def test_ranked_prefix_sign_candidates_fail_closed_on_fake_duplicate_budget() -> None:
    with pytest.raises(IterativeRealizedTrustError, match="one nonzero coordinate"):
        ranked_prefix_sign_candidates(
            np.asarray([1.0, 0.0]),
            current=np.zeros(2, dtype=np.int64),
            lower=np.full(2, -2, dtype=np.int64),
            upper=np.full(2, 2, dtype=np.int64),
            trust_radii=(1.0, 2.0),
        )


def test_fisher_margin_debt_prioritizes_tie_tight_deficit() -> None:
    curvature = categorical_fisher_trace_from_margin(np.asarray([0.0, 8.0]))
    assert curvature[0] > 100.0 * curvature[1]
    tight = fisher_margin_debt(np.asarray([-0.1]), np.asarray([0.0]))
    far = fisher_margin_debt(np.asarray([-8.0]), np.asarray([0.0]))
    assert tight > far


def test_babai_candidates_are_unique_integer_and_box_safe() -> None:
    candidates = quantized_babai_candidates(
        np.asarray([2.4, -1.7]),
        np.asarray([[2.0, 0.25], [0.25, 1.0]]),
        current=np.asarray([254, 1]),
        lower=np.asarray([0, 0]),
        upper=np.asarray([255, 255]),
        trust_radius=2.0,
        scales=(0.25, 0.5, 1.0, 2.0),
        maximum_candidates=4,
    )
    assert candidates
    states = [np.asarray([254, 1]) + row.integer_step for row in candidates]
    assert len({row.integer_step.tobytes() for row in candidates}) == len(candidates)
    assert all(np.issubdtype(row.integer_step.dtype, np.integer) for row in candidates)
    assert all(np.all((state >= 0) & (state <= 255)) for state in states)


def test_hard_selection_cannot_accept_proxy_only_or_inadmissible_step() -> None:
    candidates = (
        HardCandidate("proxy_good_hard_bad", 10.1, 0.2, 100, True, 3.0, 2.0, np.asarray([1])),
        HardCandidate("hard_good_inadmissible", 9.0, 0.1, 100, False, -3.0, 4.0, np.asarray([2])),
        HardCandidate("hard_good", 9.5, 0.15, 101, True, 1.0, 0.5, np.asarray([3])),
    )
    selected = select_realized_improvement(10.0, candidates)
    assert selected.accepted
    assert selected.selected is not None and selected.selected.candidate_id == "hard_good"
    assert selected.rho == 0.5


def test_negative_rho_hard_shrinks_even_when_caller_marks_accepted() -> None:
    policy = TrustRegionPolicy()
    decision = update_trust_radius(8.0, rho=-0.25, accepted=True, policy=policy)
    assert decision.update is TrustUpdate.HARD_SHRINK_NEGATIVE_RHO
    assert decision.new_radius == 2.0
    grown = update_trust_radius(2.0, rho=0.9, accepted=True, policy=policy)
    assert grown.update is TrustUpdate.GROW_ACCEPTED_HIGH_RHO
    assert grown.new_radius == 4.0


def test_validity_curve_keeps_negative_rho_and_missing_prediction_explicit() -> None:
    curve = summarize_validity_curve(
        (
            {"lattice_quanta": 1, "predicted_reduction": 2.0, "realized_reduction": 1.0, "rho": 0.5},
            {"lattice_quanta": 1, "predicted_reduction": 1.0, "realized_reduction": -1.0, "rho": -1.0},
            {"lattice_quanta": 2, "predicted_reduction": 0.0, "realized_reduction": 0.0, "rho": None},
        )
    )
    assert curve[0]["rho_median"] == -0.25
    assert curve[0]["negative_rho_count"] == 1
    assert curve[1]["rho_count"] == 0


def test_parallel_tempering_is_seed_deterministic_and_hard_selected() -> None:
    kwargs = {
        "lower": np.asarray([-4, -4]),
        "upper": np.asarray([4, 4]),
        "coordinates": (0, 1),
        "cheap_energy": lambda state: float((state[0] - 2) ** 2 + 0.5 * (state[1] + 1) ** 2),
        "hard_key": lambda state: (float((state[0] - 1) ** 2 + (state[1] + 1) ** 2),),
        "seed": 7,
        "sweeps": 12,
    }
    first = bounded_parallel_tempering(np.asarray([0, 0]), **kwargs)
    second = bounded_parallel_tempering(np.asarray([0, 0]), **kwargs)
    assert first.status == second.status
    assert first.temperatures == second.temperatures
    assert first.proposals == second.proposals
    assert [row.state.tolist() for row in first.terminals] == [row.state.tolist() for row in second.terminals]
    assert [row.hard_key for row in first.terminals] == [row.hard_key for row in second.terminals]
    assert first.status in {TemperingStatus.HARD_IMPROVEMENT, TemperingStatus.NO_HARD_IMPROVEMENT}
    assert first.proposals == 48
    assert first.terminals


def test_parallel_tempering_refuses_degenerate_energy_scale() -> None:
    result = bounded_parallel_tempering(
        np.asarray([0]),
        lower=np.asarray([-1]),
        upper=np.asarray([1]),
        coordinates=(0,),
        cheap_energy=lambda _state: 1.0,
        hard_key=lambda state: (float(abs(state[0])),),
        seed=0,
    )
    assert result.status is TemperingStatus.N_A_DEGENERATE_ENERGY_SPREAD
