import math

import pytest

from tac.canonical_equations.witness_modular_norm_assignment_20260713 import (
    MANIFOLD_MUON_AB_TICKET,
    MODULE_NORM_ASSIGNMENTS,
    build_witness_modular_norm_assignment_v1,
    current_muon_parameter_count,
    current_trainable_parameter_count,
    rms_to_linf_operator_norm,
    rms_to_rms_operator_norm,
    weighted_modular_product_norm,
)


def test_current_v9_inventory_is_complete_and_reconciles_muon_partition() -> None:
    assert len(MODULE_NORM_ASSIGNMENTS) == 12
    assert current_trainable_parameter_count() == 87_575
    assert current_muon_parameter_count() == 59_136
    deltas = {row.module_pattern: row.candidate_delta for row in MODULE_NORM_ASSIGNMENTS}
    assert "exact polar-chart Manifold Muon" in deltas["film.weight"]
    assert all(value == "none" for key, value in deltas.items() if key in {
        "in_proj.weight", "hidden.{0,1,2,3}.weight"
    })


def test_induced_norm_scalings_are_exact() -> None:
    assert rms_to_rms_operator_norm(2.0, fan_in=96, fan_out=96) == 2.0
    assert rms_to_rms_operator_norm(2.0, fan_in=80, fan_out=96) == pytest.approx(
        2.0 * math.sqrt(80.0 / 96.0)
    )
    assert rms_to_linf_operator_norm(3.0, fan_in=96) == pytest.approx(3.0 * math.sqrt(96.0))


def test_modular_product_norm_fails_closed_on_uncustodied_or_invalid_weights() -> None:
    assert weighted_modular_product_norm({"trunk": 2.0, "film": 3.0}, {"trunk": 4.0, "film": 0.5}) == 8.0
    with pytest.raises(ValueError, match="identical block keys"):
        weighted_modular_product_norm({"trunk": 2.0}, {"film": 1.0})
    with pytest.raises(ValueError, match="finite and positive"):
        weighted_modular_product_norm({"trunk": 2.0}, {"trunk": 0.0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        weighted_modular_product_norm({"trunk": -1.0}, {"trunk": 1.0})


def test_equation_and_ticket_are_explicitly_unmeasured_and_build_gated() -> None:
    equation = build_witness_modular_norm_assignment_v1()
    assert equation.equation_id == "witness_modular_norm_assignment_v1"
    assert equation.empirical_anchors == ()
    assert equation.provenance.promotion_eligible is False
    assert MANIFOLD_MUON_AB_TICKET.status == "WIRING_NEEDED"
    assert MANIFOLD_MUON_AB_TICKET.measurement_status == "UNMEASURED"
    assert MANIFOLD_MUON_AB_TICKET.score_claim is False
    assert MANIFOLD_MUON_AB_TICKET.pointer_moved is False
    assert "TUNED incumbent" in MANIFOLD_MUON_AB_TICKET.control_definition
