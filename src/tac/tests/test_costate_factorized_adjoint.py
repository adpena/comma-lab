# SPDX-License-Identifier: MIT
"""Task #516 regression tests: exact factorization, admission, and consumption."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.hybrid_factorized_costate_adjoint_20260716 import (
    build_hybrid_exact_factorized_costate_adjoint_v1,
)
from tac.witness_control.factorized_adjoint import (
    EXACT_CLASS_OPERATOR,
    VISIBLE_SUPPORT_FRAC,
    ZERO_WEIGHT_FRAC,
    ExactFactorizedAdjoint,
    apply_ker_a_mask,
    exact_response_direction,
    factorization_provenance,
    morse_smale_event_prior,
)
from tac.witness_control.lambda_net import (
    ARCHITECTURES,
    Interval,
    lever_features,
    make_model,
)

REPO = Path(__file__).resolve().parents[3]


def test_exact_operator_is_rank4_psd_and_gauge_null():
    assert EXACT_CLASS_OPERATOR.shape == (5, 5)
    assert np.allclose(EXACT_CLASS_OPERATOR, EXACT_CLASS_OPERATOR.T)
    assert np.linalg.matrix_rank(EXACT_CLASS_OPERATOR, tol=1e-10) == 4
    assert np.linalg.eigvalsh(EXACT_CLASS_OPERATOR).min() >= -1e-12
    assert np.max(np.abs(EXACT_CLASS_OPERATOR @ np.ones(5))) < 1e-12
    assert ZERO_WEIGHT_FRAC == pytest.approx(0.226969)
    assert VISIBLE_SUPPORT_FRAC == pytest.approx(0.773031)


def test_ker_a_mask_zeros_only_certified_coordinates_and_refuses_broadcast():
    x = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    mask = np.asarray([[False, True], [True, False]])
    assert np.array_equal(apply_ker_a_mask(x, mask), [[1.0, 0.0], [0.0, 4.0]])
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_ker_a_mask(x, np.asarray([True, False]))


def _intervals() -> tuple[list[Interval], np.ndarray]:
    names = ("seg", "horizon_margin", "persistence")
    phis = np.stack([lever_features(n) for n in names])
    out = []
    for i in range(5):
        x0 = np.asarray([.20, .10, .30, .15, .25, math.log(100_000 - 50 * i)])
        # deterministic, mildly curved class path; sufficient for the closed-form GP
        direction = np.asarray([-.003, -.002, -.001, -.0015, -.001, -1e-4])
        x1 = x0 + 10.0 * direction * (1.0 + 0.04 * i)
        u = np.asarray([.7 - .02 * i, .1 + .01 * i, .2 + .01 * i])
        out.append(Interval(
            ep0=float(10 * i), ep1=float(10 * (i + 1)), x0=x0, x1=x1,
            ctx=np.asarray([10 * i / 3000.0, 0.5 + .01 * i, 0.4]),
            u_mean=u, path=np.zeros((2, len(names) + 1)),
        ))
    return out, phis


def test_model_learns_only_shared_amplitude_and_never_new_class_direction():
    intervals, phis = _intervals()
    m = ExactFactorizedAdjoint()
    m.fit(intervals, phis)
    assert m.diagnostics is not None
    assert m.diagnostics.n_parameters == 5
    assert len(m.diagnostics.amplitude) == 5
    phi = lever_features("horizon_margin")
    exact = exact_response_direction(phi)
    got = m.response(intervals[-1].x1, intervals[-1].ctx, phi)
    nz = np.abs(exact) > 1e-12
    if np.any(nz):
        ratios = got[nz] / exact[nz]
        assert np.max(ratios) - np.min(ratios) < 1e-12
        assert ratios[0] >= 0.0
    # Uniform class pressure is the exact rank-4 gauge null direction.
    assert np.max(np.abs(exact_response_direction(lever_features("seg")))) < 1e-12


def test_architecture_and_dsl_panel_register_existing_organ_extension():
    assert "V_exact_factorized_residual" in ARCHITECTURES
    assert isinstance(make_model("V_exact_factorized_residual"), ExactFactorizedAdjoint)
    from tac.witness_control.costate_panel import LENSES
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1

    assert any(s.architecture == "V_exact_factorized_residual" for s in LENSES)
    compiled = derive_costate_agent_v1(".").compile()
    assert any(e.architecture == "V_exact_factorized_residual" for e in compiled.program.experts)


def test_provenance_and_event_prior_are_typed_and_nonpromotable():
    p = factorization_provenance()
    assert p["exact"]["head_rank"] == 4
    assert p["derived"]["road_lane_gain_only_lambda_ratio_vs_other_median"] == pytest.approx(
        2.0896226415094343)
    assert p["learned"]["shared_amplitude_parameters"] == 5
    assert p["learned"]["new_class_direction_parameters"] == 0
    assert p["learned"]["residual_ridge"] == 10.0
    assert "POST_HOC_DEVELOPMENT_ON_205" in p["learned"][
        "residual_ridge_selection_scope"]
    assert p["score_claim"] is False
    e = morse_smale_event_prior()
    assert e["derived"]["event_turnover_per_island_step_upper_bound"] == pytest.approx(
        (9.43 + 9.50) / 19.1)
    assert "not an exact event identity" in e["derived"]["interpretation"]


def test_canonical_equation_carries_three_lawrefs_and_scope_caveat():
    eq = build_hybrid_exact_factorized_costate_adjoint_v1()
    assert eq.equation_id == "hybrid_exact_factorized_costate_adjoint_v1"
    assert len(eq.empirical_anchors) == 2
    scope = eq.domain_of_validity["verdict_scope"]
    assert "per-class walk-forward loses" in scope
    assert eq.domain_of_validity["promotion_eligible"] is False


def test_durable_backtest_receipt_preserves_incompatible_run_boundaries():
    p = REPO / ".omx/research/costate_organ_elevation_backtest_20260716.json"
    assert p.is_file()
    d = json.loads(p.read_text())
    rows = {r["label"]: r for r in d["runs"]}
    n205 = rows["#205_live_v752"]
    assert n205["status"] == "BACKTESTED-PASS"
    assert n205["backtest"]["walkforward_mae_model"] < n205["backtest"][
        "walkforward_mae_heuristic"]
    assert n205["backtest"]["walkforward_perclass_mae_model"] > n205["backtest"][
        "walkforward_perclass_mae_heuristic"]
    assert n205["source_bytes_unchanged"] is True
    assert rows["mod32cap"]["status"] == "UNAVAILABLE_INSUFFICIENT_INTERVAL_SCHEMA"
    assert all(r["status"] == "PENDING_NO_RUN_LOG" for k, r in rows.items()
               if k.startswith("c2:"))
    audit = {r["surface"]: r["verdict"] for r in d["consumption_audit"]}
    assert audit["exact-factorized rank4 x ker(A) x gain adjoint"] == "CONSUMED"
    assert audit["CostateAgent DSL/panel"] == "ORPHANED_FROM_ALWAYS_ON_PRODUCTION"
    assert audit["regime_dispatch"] == "INERT_FOR_ALWAYS_ON_RECOMMENDATION"


def test_new_source_has_structural_no_actuation_surface():
    src = (REPO / "src/tac/witness_control/factorized_adjoint.py").read_text()
    for token in ("import subprocess", "os.system(", "os.exec", "os.kill(", "Popen("):
        assert token not in src
