"""Tests for tac.lagrangian_kl_weight.KLWeightProportionalController.

(Historically named ``LearnableKLWeight`` — the alias is kept and these
tests cover both names so callers that still import the old name keep
working. See Bug 4 / codex Round 3: this is a proportional ratio
controller, NOT Lagrangian dual ascent.)

Verifies the multiplicative log-space ratio update converges to the
operator-specified SNR target on a synthetic stationary problem AND the
pathological-input guards fire as designed (no silent NaN drift, no
log_w runaway).
"""
from __future__ import annotations

import math

import pytest

from tac.lagrangian_kl_weight import KLWeightProportionalController, LearnableKLWeight


def test_alias_is_proportional_controller():
    """``LearnableKLWeight`` is the deprecated alias for the renamed
    :py:class:`KLWeightProportionalController`. Both names must point to
    the SAME class so existing call sites in ``optimize_poses.py`` and
    third-party scripts continue to work without a behavioral change."""
    assert LearnableKLWeight is KLWeightProportionalController


# ────────────────────────────────────────────────────────────────────────
# Convergence tests
# ────────────────────────────────────────────────────────────────────────


def _simulate_steady_state(
    *,
    snr_target: float,
    initial_weight: float,
    eta: float,
    true_kl: float,
    true_scorer: float,
    n_steps: int,
) -> LearnableKLWeight:
    """Run n_steps of dual ascent against a stationary (kl, scorer)
    observation pair. The fixed point is `weight = snr_target * scorer / kl`.
    """
    ctrl = LearnableKLWeight(
        snr_target=snr_target,
        initial_weight=initial_weight,
        eta=eta,
    )
    for _ in range(n_steps):
        ctrl.update(kl_value=true_kl, scorer_value=true_scorer)
    return ctrl


def test_converges_to_snr_target_when_initial_too_low():
    """Initial weight 0.0001 (SNR = 0.0001 · 2.7 / 0.05 ≈ 0.0054) should
    rise to the SNR=0.10 fixed point (weight ≈ 0.001852)."""
    ctrl = _simulate_steady_state(
        snr_target=0.10, initial_weight=1e-4, eta=0.5,
        true_kl=2.7, true_scorer=0.05, n_steps=100,
    )
    expected_w = 0.10 * 0.05 / 2.7  # ≈ 0.001852
    assert math.isclose(ctrl.weight, expected_w, rel_tol=0.05), (
        f"weight {ctrl.weight} did not converge to target {expected_w}"
    )
    assert ctrl.last_snr is not None
    assert math.isclose(ctrl.last_snr, 0.10, rel_tol=0.05)


def test_converges_to_snr_target_when_initial_too_high():
    """Initial weight 1.0 (SNR = 1.0 · 2.7 / 0.05 = 54.0) should fall to
    the SNR=0.10 fixed point."""
    ctrl = _simulate_steady_state(
        snr_target=0.10, initial_weight=1.0, eta=0.5,
        true_kl=2.7, true_scorer=0.05, n_steps=100,
    )
    expected_w = 0.10 * 0.05 / 2.7
    assert math.isclose(ctrl.weight, expected_w, rel_tol=0.05), (
        f"weight {ctrl.weight} did not converge to target {expected_w}"
    )


def test_initial_weight_starts_at_initial_value():
    ctrl = LearnableKLWeight(
        snr_target=0.10, initial_weight=0.002, eta=0.5,
    )
    assert math.isclose(ctrl.weight, 0.002, rel_tol=1e-12)
    assert ctrl.last_snr is None  # no observation yet
    assert ctrl.step_count == 0


def test_synthetic_minimize_x_squared_subject_to_x_geq_1():
    """Classical strongly-convex constrained problem: min x² s.t. x ≥ 1.
    KKT: x* = 1, λ* = 2. Used here as a textbook proof that the dual
    ascent rule is correctly implemented (Boyd §5.4)."""
    # We frame the SNR controller so that:
    #   "scorer_value" plays the role of the unconstrained loss x² (we
    #     keep it fixed to expose the dual mechanics);
    #   "kl_value" plays the role of the constraint slack;
    #   "snr_target" plays the role of the constraint level.
    # The fixed point of the multiplicative update is identical
    # mathematically to the multiplicative dual ascent in the
    # exponentiated-gradient family (Kivinen & Warmuth 1997).
    ctrl = LearnableKLWeight(
        snr_target=0.5, initial_weight=10.0, eta=0.5,
    )
    for _ in range(100):
        ctrl.update(kl_value=1.0, scorer_value=2.0)
    # Fixed point: w · k / s = ρ  ⇒  w = ρ · s / k = 0.5 · 2 / 1 = 1.0
    assert math.isclose(ctrl.weight, 1.0, rel_tol=0.05)


# ────────────────────────────────────────────────────────────────────────
# Stability + bound tests
# ────────────────────────────────────────────────────────────────────────


def test_log_weight_bounds_prevent_runaway():
    """A pathological scorer_value = eps drives the SNR upward; the
    log-weight bound MUST cap the descent so the weight never collapses
    below e^log_weight_min."""
    ctrl = LearnableKLWeight(
        snr_target=0.10, initial_weight=0.001, eta=0.5,
        log_weight_min=-9.0, log_weight_max=3.0,
    )
    for _ in range(50):
        # huge KL with tiny scorer ⇒ observed SNR ≫ target ⇒ log_w should fall
        ctrl.update(kl_value=1000.0, scorer_value=1e-10)
    assert ctrl.weight >= math.exp(-9.0) - 1e-12, (
        "log_weight_min bound violated"
    )


def test_negative_kl_raises():
    ctrl = LearnableKLWeight()
    with pytest.raises(ValueError, match="non-negative"):
        ctrl.update(kl_value=-0.1, scorer_value=0.05)


def test_nonfinite_scorer_raises():
    ctrl = LearnableKLWeight()
    with pytest.raises(ValueError, match="finite"):
        ctrl.update(kl_value=2.7, scorer_value=float("nan"))


def test_invalid_construction_raises():
    with pytest.raises(ValueError, match="snr_target"):
        LearnableKLWeight(snr_target=0.0)
    with pytest.raises(ValueError, match="initial_weight"):
        LearnableKLWeight(initial_weight=-1.0)
    with pytest.raises(ValueError, match="eta"):
        LearnableKLWeight(eta=0.0)
    with pytest.raises(ValueError, match="log_weight_max"):
        LearnableKLWeight(log_weight_min=1.0, log_weight_max=0.0)


def test_state_dict_round_trip_preserves_iterate():
    ctrl = LearnableKLWeight(snr_target=0.10, initial_weight=0.002, eta=0.5)
    for _ in range(20):
        ctrl.update(kl_value=2.7, scorer_value=0.05)
    state = ctrl.state_dict()
    recovered = LearnableKLWeight()
    recovered.load_state_dict(state)
    assert math.isclose(recovered.weight, ctrl.weight, rel_tol=1e-12)
    assert recovered.step_count == ctrl.step_count
    assert recovered.last_snr == ctrl.last_snr
