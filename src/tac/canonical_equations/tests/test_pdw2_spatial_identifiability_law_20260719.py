# SPDX-License-Identifier: MIT
"""Triality checks for PDW2 coefficient-only non-identifiability law."""
from __future__ import annotations

from tac.boundary_math.pdw2_spatial_receiver import PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY
from tac.canonical_equations.pdw2_spatial_identifiability_law_20260719 import (
    EQUATION_ID,
    build_pdw2_spatial_identifiability_law_v1,
    pdw2_spatial_nonidentifiability_admissibility,
)


def test_nonidentifiability_law_builds_focuses_on_blocker_scope() -> None:
    equation = build_pdw2_spatial_identifiability_law_v1()
    assert equation.equation_id == EQUATION_ID
    assert PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY in equation.domain_of_validity["blocker"]
    assert equation.units_in["packet_to_partition_claim"] == "bool"
    assert equation.units_out["admissible"] == "bool"
    assert "tools.probe_pdw2_spatial_receiver" in equation.canonical_consumers
    assert equation.provenance.measurement_axis == "[macOS-CPU advisory]"
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.empirical_output["same_packet_two_partitions"] is True
    assert anchor.empirical_output["deterministic_n600_replay"] is True
    assert anchor.empirical_output["d_seg"] is None


def test_nonidentifiability_evaluator_blocks_packet_only_claims() -> None:
    blocked = pdw2_spatial_nonidentifiability_admissibility(
        packet_to_partition_claim=True,
        through_r_field_present=False,
    )
    assert blocked["admissible"] is False
    assert blocked["blocker_id"] == PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY

    open = pdw2_spatial_nonidentifiability_admissibility(
        packet_to_partition_claim=False,
        through_r_field_present=True,
    )
    assert open["admissible"] is False
    assert open["blocker_id"] == PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY
