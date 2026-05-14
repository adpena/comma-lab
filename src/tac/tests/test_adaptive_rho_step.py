# SPDX-License-Identifier: MIT
"""Tests for T19 — adaptive-ρ ADMM step (Boyd grand-council eureka).

Per Fields-medal council 2026-05-09 (Boyd shower-thought): the existing
Joint-ADMM coordinator embeds adaptive-ρ logic in the inner loop. T19
extracts it as a standalone, reusable helper for any ADMM consumer that
wants the Boyd §3.4.1 / He-Yang 2000 update without dragging in the full
coordinator.

Tests verify:

* Closed-form correctness on the three branches (grow / shrink / hold).
* Monotone-convergence: starting from any rho > 0, repeated calls under
  decreasing residual ratios stabilize.
* rho_min / rho_max clipping is monotone-respecting.
* Primal-dual balance: when ratio is in (1/mu, mu), rho is held.
* Robustness to zero/near-zero residuals via the eps floor.
* Strict input validation per CLAUDE.md fail-loud.
"""
from __future__ import annotations


import pytest

from tac.joint_admm_coordinator import adaptive_rho_step


# ---------------------------------------------------------------------------
# Closed-form correctness on the three branches
# ---------------------------------------------------------------------------


def test_grow_when_primal_far_exceeds_dual() -> None:
    r = adaptive_rho_step(1.0, primal_residual=100.0, dual_residual=1.0)
    assert r.direction == "grow"
    assert r.rho_next == pytest.approx(2.0, rel=1e-12)


def test_shrink_when_dual_far_exceeds_primal() -> None:
    r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=100.0)
    assert r.direction == "shrink"
    assert r.rho_next == pytest.approx(0.5, rel=1e-12)


def test_hold_when_residuals_balanced() -> None:
    r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=1.0)
    assert r.direction == "hold"
    assert r.rho_next == pytest.approx(1.0, rel=1e-12)


def test_hold_within_mu_band_either_side() -> None:
    # Ratio in (1/mu, mu) = (0.1, 10) at default mu=10.
    for primal, dual in [(1.0, 1.0), (1.0, 5.0), (5.0, 1.0), (1.0, 9.0), (9.0, 1.0)]:
        r = adaptive_rho_step(2.5, primal_residual=primal, dual_residual=dual)
        assert r.direction == "hold", (
            f"primal={primal} dual={dual} should hold within mu band"
        )
        assert r.rho_next == 2.5


def test_grow_at_exact_threshold_plus_epsilon() -> None:
    # Strict inequality: primal > mu * dual. Use eps offset to dodge tie.
    r = adaptive_rho_step(1.0, primal_residual=10.0001, dual_residual=1.0)
    assert r.direction == "grow"


def test_shrink_at_exact_threshold_plus_epsilon() -> None:
    r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=10.0001)
    assert r.direction == "shrink"


# ---------------------------------------------------------------------------
# Custom tau and mu values
# ---------------------------------------------------------------------------


def test_custom_tau_grow() -> None:
    r = adaptive_rho_step(
        1.0,
        primal_residual=100.0,
        dual_residual=1.0,
        tau_grow=5.0,
    )
    assert r.rho_next == pytest.approx(5.0, rel=1e-12)


def test_custom_tau_shrink() -> None:
    r = adaptive_rho_step(
        1.0,
        primal_residual=1.0,
        dual_residual=100.0,
        tau_shrink=0.1,
    )
    assert r.rho_next == pytest.approx(0.1, rel=1e-12)


def test_custom_mu_changes_band() -> None:
    # Tighter mu=2 means ratio 3 should grow.
    r = adaptive_rho_step(
        1.0, primal_residual=3.0, dual_residual=1.0, mu=2.0
    )
    assert r.direction == "grow"
    # Default mu=10 holds at the same ratio.
    r2 = adaptive_rho_step(1.0, primal_residual=3.0, dual_residual=1.0)
    assert r2.direction == "hold"


# ---------------------------------------------------------------------------
# Clipping behavior
# ---------------------------------------------------------------------------


def test_clip_to_rho_max() -> None:
    r = adaptive_rho_step(
        100.0,
        primal_residual=1000.0,
        dual_residual=1.0,
        rho_max=150.0,
    )
    # Proposed: 100 * 2 = 200, clipped to 150.
    assert r.rho_next == pytest.approx(150.0, rel=1e-12)
    assert any("clipped" in n for n in r.notes)


def test_clip_to_rho_min() -> None:
    r = adaptive_rho_step(
        0.01,
        primal_residual=1.0,
        dual_residual=1000.0,
        rho_min=0.008,
    )
    # Proposed: 0.01 * 0.5 = 0.005, clipped to 0.008.
    assert r.rho_next == pytest.approx(0.008, rel=1e-12)
    assert any("clipped" in n for n in r.notes)


def test_no_clip_in_normal_band() -> None:
    r = adaptive_rho_step(
        1.0, primal_residual=100.0, dual_residual=1.0
    )
    # 1.0 * 2.0 = 2.0; default rho_max=1e6.
    assert r.rho_next == pytest.approx(2.0, rel=1e-12)
    # No clipping note expected.
    assert not any("clipped" in n for n in r.notes)


# ---------------------------------------------------------------------------
# Numerical stability with zero residuals
# ---------------------------------------------------------------------------


def test_zero_dual_residual_grows_via_eps_floor() -> None:
    # Dual=0 + primal>0 → ratio=primal/eps which is huge; should grow.
    r = adaptive_rho_step(
        1.0, primal_residual=1.0, dual_residual=0.0
    )
    # Strictly: primal=1 > 10 * dual=0 → True; rho grows.
    assert r.direction == "grow"


def test_zero_primal_residual_shrinks_via_eps_floor() -> None:
    r = adaptive_rho_step(
        1.0, primal_residual=0.0, dual_residual=1.0
    )
    # Strictly: dual=1 > 10 * primal=0 → True; rho shrinks.
    assert r.direction == "shrink"


def test_both_zero_residuals_holds() -> None:
    # KKT-equilibrium branch: primal=0 AND dual=0 → hold; notes carry the
    # explicit equilibrium reason rather than the standard "ratio within mu band" string.
    r = adaptive_rho_step(1.0, primal_residual=0.0, dual_residual=0.0)
    assert r.direction == "hold"
    assert r.rho_next == 1.0
    assert any("equilibrium" in n.lower() for n in r.notes)


# ---------------------------------------------------------------------------
# Provenance fields
# ---------------------------------------------------------------------------


def test_notes_include_provenance_tag() -> None:
    r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=1.0)
    assert any("Boyd" in n for n in r.notes)
    assert any("adaptive-rho" in n.lower() for n in r.notes)


def test_ratio_field_correct() -> None:
    r = adaptive_rho_step(1.0, primal_residual=8.0, dual_residual=2.0)
    assert r.ratio == pytest.approx(4.0, rel=1e-12)


def test_dataclass_is_frozen() -> None:
    r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=1.0)
    with pytest.raises(Exception):  # FrozenInstanceError
        r.rho_next = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input validation — fail loud per CLAUDE.md
# ---------------------------------------------------------------------------


def test_rejects_non_finite_inputs() -> None:
    for arg in [
        {"rho_curr": float("nan")},
        {"primal_residual": float("inf")},
        {"dual_residual": float("nan")},
        {"mu": float("inf")},
        {"tau_grow": float("nan")},
    ]:
        kwargs: dict = {
            "rho_curr": 1.0,
            "primal_residual": 1.0,
            "dual_residual": 1.0,
        }
        kwargs.update(arg)
        with pytest.raises(ValueError):
            adaptive_rho_step(**kwargs)


def test_rejects_zero_or_negative_rho() -> None:
    with pytest.raises(ValueError, match="rho_curr"):
        adaptive_rho_step(0.0, primal_residual=1.0, dual_residual=1.0)
    with pytest.raises(ValueError, match="rho_curr"):
        adaptive_rho_step(-0.01, primal_residual=1.0, dual_residual=1.0)


def test_rejects_negative_residuals() -> None:
    with pytest.raises(ValueError, match="primal_residual"):
        adaptive_rho_step(1.0, primal_residual=-0.1, dual_residual=1.0)
    with pytest.raises(ValueError, match="dual_residual"):
        adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=-0.1)


def test_rejects_invalid_mu() -> None:
    with pytest.raises(ValueError, match="mu"):
        adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=1.0, mu=1.0)
    with pytest.raises(ValueError, match="mu"):
        adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=1.0, mu=0.5)


def test_rejects_invalid_tau_grow() -> None:
    with pytest.raises(ValueError, match="tau_grow"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, tau_grow=1.0
        )
    with pytest.raises(ValueError, match="tau_grow"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, tau_grow=0.5
        )


def test_rejects_invalid_tau_shrink() -> None:
    with pytest.raises(ValueError, match="tau_shrink"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, tau_shrink=0.0
        )
    with pytest.raises(ValueError, match="tau_shrink"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, tau_shrink=1.0
        )
    with pytest.raises(ValueError, match="tau_shrink"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, tau_shrink=2.0
        )


def test_rejects_inverted_clip_range() -> None:
    with pytest.raises(ValueError, match="rho_min"):
        adaptive_rho_step(
            1.0,
            primal_residual=1.0,
            dual_residual=1.0,
            rho_min=10.0,
            rho_max=1.0,
        )


def test_rejects_non_positive_rho_min() -> None:
    # Closes R2.4: rho_min<=0 would let shrinks drive rho to 0/negative,
    # invalidating the rho_curr > 0 invariant for the next call.
    with pytest.raises(ValueError, match="rho_min"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, rho_min=0.0
        )
    with pytest.raises(ValueError, match="rho_min"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, rho_min=-1.0
        )


def test_rejects_invalid_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        adaptive_rho_step(
            1.0, primal_residual=1.0, dual_residual=1.0, eps=0.0
        )


def test_rejects_bool_inputs() -> None:
    # bool is a subclass of int/float; explicitly rejected.
    with pytest.raises(ValueError):
        adaptive_rho_step(True, primal_residual=1.0, dual_residual=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        adaptive_rho_step(1.0, primal_residual=True, dual_residual=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Convergence properties (monotone)
# ---------------------------------------------------------------------------


def test_repeated_grow_converges_to_clip() -> None:
    # Pin primal/dual ratio = 100 (always > mu); rho doubles each step
    # until it hits rho_max=1024.
    rho = 1.0
    for _ in range(20):
        rho = adaptive_rho_step(
            rho, primal_residual=100.0, dual_residual=1.0, rho_max=1024.0
        ).rho_next
    assert rho == 1024.0


def test_repeated_shrink_converges_to_clip() -> None:
    # Pin reverse ratio; rho halves each step until rho_min=0.001.
    rho = 1.0
    for _ in range(50):
        rho = adaptive_rho_step(
            rho, primal_residual=1.0, dual_residual=100.0, rho_min=0.001
        ).rho_next
    assert rho == pytest.approx(0.001, rel=1e-12)


def test_balanced_residuals_no_drift() -> None:
    # Ratio = 1; rho should hold across many steps.
    rho = 3.14
    for _ in range(100):
        result = adaptive_rho_step(
            rho, primal_residual=1.0, dual_residual=1.0
        )
        rho = result.rho_next
        assert result.direction == "hold"
    assert rho == pytest.approx(3.14, rel=1e-12)


# ---------------------------------------------------------------------------
# Symmetric monotonicity: tau_grow * tau_shrink should be invariant
# ---------------------------------------------------------------------------


def test_default_tau_pair_is_symmetric_inverse() -> None:
    # Default tau_grow=2.0, tau_shrink=0.5 → product=1.
    # One grow + one shrink should return to the start.
    r1 = adaptive_rho_step(1.0, primal_residual=100.0, dual_residual=1.0)
    r2 = adaptive_rho_step(
        r1.rho_next, primal_residual=1.0, dual_residual=100.0
    )
    assert r2.rho_next == pytest.approx(1.0, rel=1e-12)
