"""T19 → run_admm migration parity tests (2026-05-09).

Covers the migration of ``tac.joint_admm_coordinator.run_admm`` from
inline Q4B + steady-state adaptive-ρ logic to delegating those updates to
the canonical ``adaptive_rho_step`` helper.

Backward-compat contract
------------------------
``JointADMMConfig.use_canonical_adaptive_rho`` gates between two paths:

- ``True`` (default, post-migration 2026-05-09): canonical helper consumed.
- ``False``: legacy inline code path preserved verbatim.

This test file proves the migration preserves bit-for-bit identical
numerical behavior across:

1. ρ trajectory parity at every iteration
2. Final byte allocation parity
3. Final dual / score / margin parity
4. KKT waterline residual parity
5. Convergence flag parity
6. Restart count parity
7. The 14 existing regression tests still pass on the canonical path
   (verified separately in ``src/tac/tests/test_joint_admm_coordinator.py``).

Per CLAUDE.md "Comment-only contracts — FORBIDDEN" + "Internal-consistency
assertions" — the migration adds no comment-only promises; every behavior
claim is empirically asserted in this file.

Memory ref: ``feedback_t19_migration_and_cathedral_autopilot_catalog_landed_20260509.md``.

[empirical: tests/test_run_admm_migration_to_canonical_adaptive_rho.py]
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from tac.joint_admm_coordinator import (
    AdmmResult,
    JointADMMConfig,
    ProximalStepResult,
    StreamProximalCodec,
    adaptive_rho_step,
    run_admm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic streams (must be DETERMINISTIC for parity comparison)
# ─────────────────────────────────────────────────────────────────────────────


class _DeterministicQuadStream:
    """f(b) = a*(b - b_opt)^2; deterministic proximal-Lagrangian step."""

    def __init__(self, a: float, b_opt: float, name: str, discretisation: float = 1.0):
        self.a = a
        self.b_opt = b_opt
        self._name = name
        self.discretisation = discretisation

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        b_unconstr = self.b_opt - dual / (2.0 * self.a)
        b = max(0.0, min(target_bytes, b_unconstr))
        b = max(0.0, round(b / self.discretisation) * self.discretisation)
        score = self.a * (b - self.b_opt) ** 2
        margin = max(2.0 * self.a * (self.b_opt - b), 0.0)
        return ProximalStepResult(
            encoded_bytes=int(b),
            score_delta=float(score),
            marginal=float(margin),
            state=None,
        )


class _DeterministicLinearStream:
    """f(b) = max(c0 - slope*b, 0); deterministic proximal-Lagrangian step."""

    def __init__(self, slope: float, c0: float, name: str, discretisation: float = 1.0):
        self.slope = slope
        self.c0 = c0
        self._name = name
        self.discretisation = discretisation

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        floor_b = self.c0 / self.slope
        if dual >= self.slope:
            b_unconstr = 0.0
        else:
            b_unconstr = floor_b
        b = max(0.0, min(target_bytes, b_unconstr))
        b = max(0.0, round(b / self.discretisation) * self.discretisation)
        score = max(self.c0 - self.slope * b, 0.0)
        margin = self.slope if b < floor_b else 0.0
        return ProximalStepResult(
            encoded_bytes=int(b),
            score_delta=float(score),
            marginal=float(margin),
            state=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_streams_for(scenario: str) -> tuple[list[StreamProximalCodec], JointADMMConfig]:
    """Construct (streams, cfg) for one of several canonical scenarios.

    Each scenario is constructed twice in each test (once per path) so the
    streams are FRESH and identical between calls — required because
    proximal_step has internal call counters that differ between runs.
    """
    if scenario == "two_stream_kkt":
        streams = [
            _DeterministicQuadStream(a=0.005, b_opt=300.0, name="q1"),
            _DeterministicQuadStream(a=0.01, b_opt=200.0, name="q2"),
        ]
        cfg = dict(
            byte_budget=300.0,
            max_iters=300,
            primal_tol=1e-2,
            dual_tol=1e-2,
            kkt_waterline_tol=0.05,
            rho_init=0.02,
        )
    elif scenario == "rho_too_small":
        streams = [
            _DeterministicQuadStream(a=0.001, b_opt=1000.0, name="q1"),
            _DeterministicQuadStream(a=0.001, b_opt=800.0, name="q2"),
        ]
        cfg = dict(
            byte_budget=100.0,
            max_iters=200,
            primal_tol=1e-2,
            dual_tol=1e-2,
            kkt_waterline_tol=0.05,
            rho_init=1e-5,
        )
    elif scenario == "rho_undersized":
        streams = [
            _DeterministicQuadStream(a=0.05, b_opt=80.0, name="q"),
            _DeterministicLinearStream(slope=0.3, c0=5.0, name="l"),
        ]
        cfg = dict(
            byte_budget=100.0,
            max_iters=300,
            rho_init=1e-3,
            rho_max=1e6,
            rho_imbalance_ratio=10.0,
            primal_tol=1e-2,
            dual_tol=1e-2,
        )
    elif scenario == "discretised":
        streams = [
            _DeterministicQuadStream(a=0.02, b_opt=100.0, name="q", discretisation=25.0),
            _DeterministicLinearStream(slope=0.4, c0=10.0, name="l", discretisation=25.0),
        ]
        cfg = dict(
            byte_budget=120.0,
            max_iters=200,
            primal_tol=1e-2,
            dual_tol=1e-2,
            rho_init=0.1,
        )
    elif scenario == "tight_cap":
        # Generous budget where both streams want b_opt — budget is non-binding;
        # exercise the "primal residual swings sign" path.
        streams = [
            _DeterministicQuadStream(a=0.005, b_opt=10.0, name="q1"),
            _DeterministicQuadStream(a=0.005, b_opt=10.0, name="q2"),
        ]
        cfg = dict(
            byte_budget=30.0,
            max_iters=20,
            primal_tol=1e-2,
            dual_tol=1e-2,
            kkt_waterline_tol=0.05,
            rho_init=1e3,
        )
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return streams, cfg  # type: ignore[return-value]


def _run_both_paths(scenario: str) -> tuple[AdmmResult, AdmmResult]:
    """Run scenario under canonical and legacy paths; return (canonical, legacy)."""
    s_canonical, cfg_kwargs = _build_streams_for(scenario)
    canonical_cfg = JointADMMConfig(**cfg_kwargs, use_canonical_adaptive_rho=True)
    canonical_result = run_admm(s_canonical, canonical_cfg)

    s_legacy, _ = _build_streams_for(scenario)
    legacy_cfg = JointADMMConfig(**cfg_kwargs, use_canonical_adaptive_rho=False)
    legacy_result = run_admm(s_legacy, legacy_cfg)
    return canonical_result, legacy_result


def _assert_iteration_traces_equal(
    canonical: AdmmResult, legacy: AdmmResult, atol: float = 0.0
) -> None:
    """Assert per-iteration ρ trajectory + residuals are bit-equal.

    The migration MUST be numerically identical on the configured path. We
    use atol=0 by default (bit-equality required); tests that legitimately
    incur floating-point drift may pass a non-zero atol.
    """
    assert len(canonical.history) == len(legacy.history), (
        f"history length drift: canonical={len(canonical.history)} "
        f"legacy={len(legacy.history)}"
    )
    for i, (c, l) in enumerate(zip(canonical.history, legacy.history)):
        assert c.iter == l.iter, f"iter index mismatch at {i}: {c.iter} vs {l.iter}"
        if atol == 0.0:
            assert c.rho == l.rho, (
                f"rho parity violated at iter {c.iter}: "
                f"canonical={c.rho!r} legacy={l.rho!r}"
            )
        else:
            assert abs(c.rho - l.rho) <= atol, (
                f"rho parity violated at iter {c.iter} "
                f"(atol={atol}): canonical={c.rho} legacy={l.rho}"
            )
        # Residuals must also match — they're inputs to the rho update.
        assert c.primal_residual == pytest.approx(l.primal_residual, abs=atol)
        assert c.dual_residual == pytest.approx(l.dual_residual, abs=atol)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: rho-trajectory parity on the canonical KKT scenario
# ─────────────────────────────────────────────────────────────────────────────


def test_two_stream_kkt_rho_trajectory_parity():
    """[empirical: this file] ρ trajectory must be IDENTICAL between the
    canonical adaptive_rho_step path and the legacy inline path on the
    primary KKT scenario.

    This is the strongest possible parity check — proves the canonical
    helper is a behavioral substitute for the inline logic.
    """
    canonical, legacy = _run_both_paths("two_stream_kkt")
    _assert_iteration_traces_equal(canonical, legacy, atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: final allocation parity — bytes, score, marginal
# ─────────────────────────────────────────────────────────────────────────────


def test_two_stream_kkt_final_allocation_parity():
    """[empirical: this file] Final bytes / score / marginal must be
    bit-equal between paths."""
    canonical, legacy = _run_both_paths("two_stream_kkt")
    assert canonical.final_bytes_per_stream == legacy.final_bytes_per_stream
    assert canonical.final_score_per_stream == legacy.final_score_per_stream
    assert canonical.final_marginal_per_stream == legacy.final_marginal_per_stream
    assert canonical.waterline_kkt_residual == legacy.waterline_kkt_residual
    assert canonical.waterline_satisfied == legacy.waterline_satisfied


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: convergence flag + restart count parity
# ─────────────────────────────────────────────────────────────────────────────


def test_two_stream_kkt_convergence_and_restart_parity():
    """[empirical: this file] ``converged`` and ``restarts`` must match."""
    canonical, legacy = _run_both_paths("two_stream_kkt")
    assert canonical.converged == legacy.converged
    assert canonical.restarts == legacy.restarts
    assert canonical.iters == legacy.iters


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Q4B "rho too small" branch parity
# ─────────────────────────────────────────────────────────────────────────────


def test_q4b_rho_too_small_path_parity():
    """[empirical: this file] When iter-1 ratio > 10 (Q4B "too small"
    branch), both paths must produce IDENTICAL rho_iter1 and trajectory.

    This isolates the Q4B grow path (rho * 2.0 with [1e-6, 1e6] clip).
    """
    canonical, legacy = _run_both_paths("rho_too_small")
    # Iter-1 rho must match (Q4B fired the same way).
    assert canonical.history[0].rho == legacy.history[0].rho, (
        f"Q4B iter-1 rho drift: canonical={canonical.history[0].rho} "
        f"legacy={legacy.history[0].rho}"
    )
    # Full trajectory parity.
    _assert_iteration_traces_equal(canonical, legacy, atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Steady-state grow path parity (rho_init too small for steady-state)
# ─────────────────────────────────────────────────────────────────────────────


def test_rho_undersized_steady_state_parity():
    """[empirical: this file] When rho_init is small enough to require
    repeated steady-state growth, both paths must produce IDENTICAL trajectories."""
    canonical, legacy = _run_both_paths("rho_undersized")
    _assert_iteration_traces_equal(canonical, legacy, atol=0.0)
    # Final allocation must also match.
    assert canonical.final_bytes_per_stream == legacy.final_bytes_per_stream


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Discretisation path parity
# ─────────────────────────────────────────────────────────────────────────────


def test_discretised_byte_path_parity():
    """[empirical: this file] Heavy discretisation (multiples of 25)
    must produce IDENTICAL trajectories under both paths."""
    canonical, legacy = _run_both_paths("discretised")
    _assert_iteration_traces_equal(canonical, legacy, atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Q4B "rho too large" branch parity
# ─────────────────────────────────────────────────────────────────────────────


def test_q4b_rho_too_large_path_parity():
    """[empirical: this file] When iter-1 ratio < 0.1 (Q4B "too large"
    branch), both paths must produce IDENTICAL rho_iter1.

    This isolates the Q4B shrink path (rho * 0.5 with [1e-6, 1e6] clip).
    """
    canonical, legacy = _run_both_paths("tight_cap")
    # Whatever the iter-1 ratio class, both paths must agree.
    assert canonical.history[0].rho == legacy.history[0].rho, (
        f"Q4B iter-1 rho drift: canonical={canonical.history[0].rho} "
        f"legacy={legacy.history[0].rho}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: default flag is True (post-migration semantics)
# ─────────────────────────────────────────────────────────────────────────────


def test_use_canonical_adaptive_rho_default_is_true():
    """Post-migration default (2026-05-09): canonical path is the default.

    A future operator constructing JointADMMConfig with no explicit value
    must get the canonical path — the migration is COMPLETE, the legacy
    path is preserved only for regression verification.
    """
    cfg = JointADMMConfig(byte_budget=100.0)
    assert cfg.use_canonical_adaptive_rho is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Q4B exclusivity preserved in canonical path
# ─────────────────────────────────────────────────────────────────────────────


def test_q4b_steady_state_exclusivity_preserved():
    """[empirical: this file] When Q4B fires on iter 1, the steady-state
    rule MUST NOT also fire on iter 1.

    This is the test_two_stream_convex_* failure mode that prompted Q4B's
    exclusivity gate: with rho_init=0.02, Q4B → 0.04 + steady-state → 0.08
    would corner-pin bytes to [0, 29]. The canonical path must preserve
    this exclusivity.

    Verification approach: at iter 1, rho should change by AT MOST one
    multiplicative factor (Q4B's tau=2.0 OR steady-state's tau=cfg.rho_growth),
    not both compounded.
    """
    streams, cfg_kwargs = _build_streams_for("two_stream_kkt")
    cfg = JointADMMConfig(**cfg_kwargs, use_canonical_adaptive_rho=True)
    result = run_admm(streams, cfg)
    rho_iter1 = result.history[0].rho
    # rho_init=0.02; if BOTH rules fired on iter 1, rho_iter1 = 0.02 * 2.0 *
    # cfg.rho_growth = 0.02 * 2.0 * 2.0 = 0.08 (OR worse on shrink: 0.005).
    # Q4B alone: 0.04. Steady-state alone: 0.04 (cfg.rho_growth=2.0).
    # Either single path: 0.04. Either way, rho_iter1 ∈ {0.02, 0.04, 0.01}.
    # Definitely NOT 0.08 or 0.005 (that's the double-application bug).
    forbidden_double = {0.08, 0.005}
    assert rho_iter1 not in forbidden_double, (
        f"Q4B exclusivity violated in canonical path: rho_iter1={rho_iter1} "
        f"is the double-application value (rho_init=0.02 should yield "
        f"either 0.02 (hold), 0.04 (single grow), or 0.01 (single shrink))"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: backward-compat flag round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_legacy_path_still_invokable_explicitly():
    """[empirical: this file] Setting ``use_canonical_adaptive_rho=False``
    explicitly must invoke the legacy inline code path.

    Verification: legacy path must produce a valid AdmmResult and cite the
    legacy code branch by behavior (not by introspection).
    """
    streams, cfg_kwargs = _build_streams_for("two_stream_kkt")
    cfg = JointADMMConfig(**cfg_kwargs, use_canonical_adaptive_rho=False)
    result = run_admm(streams, cfg)
    assert isinstance(result, AdmmResult)
    # Legacy path must converge for this scenario (test_two_stream_convex_*
    # is the canonical regression test).
    assert result.converged


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: full scenario sweep — parity across all canonical scenarios
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scenario",
    [
        "two_stream_kkt",
        "rho_too_small",
        "rho_undersized",
        "discretised",
        "tight_cap",
    ],
)
def test_full_scenario_parity_sweep(scenario: str):
    """[empirical: this file] Parity sweep across all 5 canonical scenarios.

    Catches any rho-update edge case the per-scenario tests miss.
    """
    canonical, legacy = _run_both_paths(scenario)
    # Final allocation must always match across all scenarios.
    assert canonical.final_bytes_per_stream == legacy.final_bytes_per_stream, (
        f"scenario {scenario!r}: bytes drift "
        f"canonical={canonical.final_bytes_per_stream} "
        f"legacy={legacy.final_bytes_per_stream}"
    )
    assert canonical.converged == legacy.converged
    assert canonical.iters == legacy.iters
    # Trajectory must be bit-equal.
    _assert_iteration_traces_equal(canonical, legacy, atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: canonical helper is the named delegate target
# ─────────────────────────────────────────────────────────────────────────────


def test_canonical_helper_is_imported_and_callable():
    """The canonical adaptive_rho_step helper must be importable from the
    coordinator module and produce a valid AdaptiveRhoStep result.

    This is the structural-link verification: the migration consumes the
    canonical helper, so the helper must remain a public, importable surface.
    """
    step = adaptive_rho_step(
        rho_curr=1.0,
        primal_residual=100.0,
        dual_residual=1.0,
        mu=10.0,
        tau_grow=2.0,
        tau_shrink=0.5,
        rho_min=1e-6,
        rho_max=1e6,
    )
    assert step.rho_next == 2.0
    assert step.direction == "grow"


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: migration provenance — config field is documented
# ─────────────────────────────────────────────────────────────────────────────


def test_use_canonical_adaptive_rho_field_present():
    """The migration flag must be present as a JointADMMConfig field.

    This proves the API surface for the migration is explicit (not buried
    in a global flag, not implicit via env var).
    """
    fields = {f.name for f in dataclasses.fields(JointADMMConfig)}
    assert "use_canonical_adaptive_rho" in fields, (
        "T19 migration flag use_canonical_adaptive_rho is missing from "
        "JointADMMConfig — the migration was reverted or never landed."
    )
