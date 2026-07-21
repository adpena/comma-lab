# SPDX-License-Identifier: MIT
"""Offline Pontryagin conformance fixture for costate ORGAN v2.

This synthetic scalar LQR test validates adjoint signs and bounded control
projection only.  It is deliberately incapable of authorizing a curriculum or
live-run control update.
"""
from __future__ import annotations

import pytest

from tac.witness_control.costate_organ_v2 import pontryagin_lqr_conformance_fixture


def test_forward_backward_sweep_matches_analytic_lqr_and_hamiltonian_fd():
    row = pontryagin_lqr_conformance_fixture()
    assert row["fixture_only"] and not row["live_control_authority"]
    assert row["learned_parameters"] == 0 and row["actuation"] == "NONE"
    assert row["sweep_residual_monotone"]
    assert row["sweep_residuals"][-1] <= 1e-11
    assert row["control_max_abs_error_vs_analytic"] < 4e-11
    assert row["costate_max_abs_error_vs_analytic"] < 2e-11
    assert row["hamiltonian_fd_x_max_abs_error"] < 3e-11
    assert row["hamiltonian_fd_u_max_abs_error"] < 2e-11
    assert row["projected_control_max_abs_error"] < 4e-11


def test_nonstabilizing_riccati_root_is_explicitly_rejected():
    rows = pontryagin_lqr_conformance_fixture()["riccati_roots"]
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    assert len(accepted) == len(rejected) == 1
    assert accepted[0]["closed_loop_abs"] < 1.0
    assert rejected[0]["closed_loop_abs"] > 1.0
    assert rejected[0]["rejection"] == "non_stabilizing_riccati_root"


def test_fixture_refuses_relaxation_that_drops_monotone_convergence_contract():
    with pytest.raises(ValueError, match="relaxation"):
        pontryagin_lqr_conformance_fixture(relaxation=0.5)
