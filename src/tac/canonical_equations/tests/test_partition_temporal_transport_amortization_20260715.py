# SPDX-License-Identifier: MIT
"""Tests for the jitter-bound partition temporal-transport amortization law."""
from __future__ import annotations

import pytest

from tac.canonical_equations.equation import VERIFIED_VIA_EMPIRICAL_ANCHOR
from tac.canonical_equations.partition_temporal_transport_amortization_20260715 import (
    EQUATION_ID,
    NAIVE_BYTES_PER_FRAME,
    RESIDUAL_BYTES_PER_FRAME_PERSIST,
    RESIDUAL_BYTES_PER_FRAME_SCREW,
    amortization_ratio,
    build_partition_temporal_transport_amortization_v1,
    transport_pays,
)


def test_equation_id_stable() -> None:
    assert EQUATION_ID == "partition_temporal_transport_amortization_jitter_bound_v1"


def test_measured_constants_encode_the_negative() -> None:
    # The law's content: the measured residual out-costs the per-frame coder.
    assert RESIDUAL_BYTES_PER_FRAME_PERSIST > NAIVE_BYTES_PER_FRAME
    assert RESIDUAL_BYTES_PER_FRAME_SCREW > NAIVE_BYTES_PER_FRAME


def test_transport_pays_decision_rule() -> None:
    assert transport_pays(NAIVE_BYTES_PER_FRAME, RESIDUAL_BYTES_PER_FRAME_SCREW) is False
    assert transport_pays(1003.0, 500.0) is True
    assert transport_pays(1003.0, 1000.0, xi_marginal_bytes_per_frame=10.0) is False


def test_amortization_ratio_measured_below_one() -> None:
    assert amortization_ratio(601_931, 846_116) == pytest.approx(0.7114, abs=1e-3)
    assert amortization_ratio(601_931, 846_116) < 1.0


def test_amortization_ratio_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        amortization_ratio(0, 1)
    with pytest.raises(ValueError):
        amortization_ratio(1, -1)


def test_build_equation_structure() -> None:
    eq = build_partition_temporal_transport_amortization_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 1
    anchor = eq.empirical_anchors[0]
    assert anchor.empirical_verification_status == VERIFIED_VIA_EMPIRICAL_ANCHOR
    assert anchor.source_artifact.endswith("results.json")
    # verdict_scope FORMULATION must be carried on the anchor + the domain of validity.
    assert "FORMULATION" in anchor.empirical_output["verdict_scope"]
    assert "FORMULATION" in eq.domain_of_validity["verdict_scope"]
    # non-promotable advisory axis only.
    assert eq.domain_of_validity["measurement_axis"] == ["[macOS advisory / research-signal]"]
    assert eq.canonical_producers and eq.canonical_consumers


def test_registered_in_live_registry() -> None:
    from tac.canonical_equations import query_equations

    ids = [e.equation_id for e in query_equations()]
    assert ids.count(EQUATION_ID) == 1
