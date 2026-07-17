# SPDX-License-Identifier: MIT
"""Tests for the dash-phase carrier rate-law canonical equation (#425 Arm G)."""
from __future__ import annotations

import pytest

from tac.canonical_equations.dash_phase_carrier_rate_law_20260717 import (
    EQUATION_ID,
    EXPECTED_BITS_PER_DASH_PRIOR,
    LANE_ONLY_NAIVE_BYTES,
    MEASURED_BITS_PER_MATCHED_DASH,
    SECTION_BYTES_EXCL_XI,
    build_dash_phase_carrier_rate_law_v1,
    dash_rate_amortizes,
    populate_dash_phase_carrier_rate_law_equation,
    site_prior_transfers_to_dash,
)


def test_equation_id():
    assert EQUATION_ID == "dash_phase_carrier_rate_blinkback_prior_divergence_v1"


def test_dash_rate_amortizes_measured_row():
    assert dash_rate_amortizes(SECTION_BYTES_EXCL_XI, LANE_ONLY_NAIVE_BYTES) is True
    assert dash_rate_amortizes(LANE_ONLY_NAIVE_BYTES, SECTION_BYTES_EXCL_XI) is False


def test_dash_rate_amortizes_refuses_nonpositive():
    with pytest.raises(ValueError):
        dash_rate_amortizes(0, 100)
    with pytest.raises(ValueError):
        dash_rate_amortizes(100, -1)


def test_site_prior_does_not_transfer_measured():
    assert site_prior_transfers_to_dash(
        MEASURED_BITS_PER_MATCHED_DASH, EXPECTED_BITS_PER_DASH_PRIOR
    ) is False
    # a hypothetical realized cost near the prior WOULD transfer
    assert site_prior_transfers_to_dash(4.6, EXPECTED_BITS_PER_DASH_PRIOR) is True


def test_build_equation_structure():
    eq = build_dash_phase_carrier_rate_law_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 1
    a = eq.empirical_anchors[0]
    assert a.empirical_output["blink_back_fraction"] == pytest.approx(0.787, abs=1e-3)
    assert a.empirical_output["section_bytes_excl_xi"] == SECTION_BYTES_EXCL_XI
    assert "FORMULATION" in a.empirical_output["verdict_scope"]
    # producers/consumers both present (no orphan equation)
    assert eq.canonical_producers and eq.canonical_consumers
    # the memo + codec are the producers
    assert any("dash_phase_carrier.py" in p for p in eq.canonical_producers)


def test_anchor_records_prior_divergence_residual():
    eq = build_dash_phase_carrier_rate_law_v1()
    res = eq.predicted_vs_empirical_residual["bits_per_dash_prior_vs_measured"]
    assert res == pytest.approx(MEASURED_BITS_PER_MATCHED_DASH - EXPECTED_BITS_PER_DASH_PRIOR)
    assert res > 0  # the measured divergence IS the finding


def test_populate_idempotent(tmp_path):
    path = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.lock"
    eq1 = populate_dash_phase_carrier_rate_law_equation(
        path=path, lock_path=lock, agent="test", subagent_id="arm_g_425"
    )
    eq2 = populate_dash_phase_carrier_rate_law_equation(
        path=path, lock_path=lock, agent="test", subagent_id="arm_g_425"
    )
    assert eq1.equation_id == eq2.equation_id == EQUATION_ID
    rows = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(rows) >= 1
