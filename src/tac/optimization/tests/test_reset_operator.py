# SPDX-License-Identifier: MIT
"""BEHAVIOUR tests for the boundary reset operator (ddm_sb2, task #819).

NO-FAKE class #2 discipline ("tests-verify-constants-not-behavior"): every test here
would FAIL if the module body were replaced by ``return <canonical markers>``. The
load-bearing tests run the REAL ``mlx.optimizers.Adam`` and the real Adam recursion and
compare measured displacement — they do not assert that a constant equals itself.

The mutation guard is explicit at the bottom (:func:`test_mutation_guard_*`): it asserts
the properties that a marker-returning stub could not satisfy.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.reset_operator import (
    ARM_A_NO_RESET,
    ARM_B_ZERO_RESET,
    ARM_BPRIME_BIAS_CORRECTED,
    ARM_C_MOMENTUM_ONLY,
    ARMS,
    DEFAULT_BETAS,
    ResetOperatorConfig,
    ResetOperatorError,
    apply_reset,
    arm_config,
    boundary_impulse_epochs,
    cumulative_displacement_ratio,
    cumulative_excess_sign_steps,
    effective_lr_multiplier,
    load_diagonal_prior,
    resolve_arm_name,
    solve_norm_match_scalar,
)

mx = pytest.importorskip("mlx.core")
optim = pytest.importorskip("mlx.optimizers")


# ── the load-bearing behaviour test: eta(t) vs the REAL MLX optimizer ────────────────
def _real_adam_displacement(steps: int, *, bias_correction: bool, lr: float = 1e-3):
    """Run the REAL mlx Adam on a constant gradient; return per-step |Δparam| / lr."""
    p = mx.zeros((4,))
    g = mx.ones((4,)) * 0.37  # constant, non-unit gradient
    opt = optim.Adam(learning_rate=lr, bias_correction=bias_correction)
    out = []
    for _ in range(steps):
        new = opt.apply_gradients({"w": g}, {"w": p})["w"]
        mx.eval(new)
        out.append(float(np.abs(np.asarray(new) - np.asarray(p)).mean()) / lr)
        p = new
    return out


def test_eta_matches_real_mlx_adam_step_ratio() -> None:
    """eta(t) IS the measured uncorrected/bias-corrected step ratio of the real optimizer.

    This is the whole derivation. If ``effective_lr_multiplier`` returned a canned table
    it would not track the real optimizer across every step.
    """
    n = 40
    uncorr = _real_adam_displacement(n, bias_correction=False)
    corr = _real_adam_displacement(n, bias_correction=True)
    for t in range(1, n + 1):
        measured = uncorr[t - 1] / corr[t - 1]
        predicted = effective_lr_multiplier(t)
        assert measured == pytest.approx(predicted, rel=2e-3), (
            f"step {t}: measured ratio {measured} != closed form {predicted}")


def test_bias_corrected_adam_takes_unit_sign_steps() -> None:
    """The corrected optimizer's step magnitude is lr (a pure sign step) at constant gradient.

    This is the premise the whole eta derivation rests on; it is MEASURED, not assumed.
    """
    corr = _real_adam_displacement(25, bias_correction=True)
    for t, d in enumerate(corr, start=1):
        assert d == pytest.approx(1.0, rel=5e-3), f"step {t} displacement/lr = {d}"


def test_eta_shape_is_measured_not_asserted() -> None:
    """eta rises then decays; peak near t=12; eta(1)=3.162. Derived by scanning, not hardcoded.

    The tail is SLOW: eta(1000)=1.258, eta(3000)=1.026, and only by ~8k steps is it within
    2e-4 of unity (b2^t must be << 1, i.e. t >> 1/(1-b2)=1000). Measured here, so a future
    beta change cannot silently invalidate the window arithmetic downstream.
    """
    vals = [effective_lr_multiplier(t) for t in range(1, 3000)]
    peak_idx = int(np.argmax(vals)) + 1
    assert peak_idx == 12
    assert vals[0] == pytest.approx(3.1623, rel=1e-3)
    assert max(vals) == pytest.approx(6.5685, rel=1e-3)
    assert vals[-1] > 1.02, "the tail is slow — eta is still >2% high at t=3000"
    assert effective_lr_multiplier(8000) == pytest.approx(1.0, abs=1e-3)
    assert effective_lr_multiplier(20000) == pytest.approx(1.0, abs=1e-8)


def test_cumulative_excess_converges_and_prices_the_boundary() -> None:
    """The impulse CONVERGES => it is set by reset COUNT, not window length (gc15 §5.1).

    Independent re-derivation of gc15's headline: the asymptote is **1212.57** extra
    sign-steps, matching the memo's quoted 1,212.6 to four significant figures. Convergence
    is slow (a 2000-step sum reaches only 94% of it), so the test brackets the ASYMPTOTE
    rather than a truncated sum.
    """
    a = cumulative_excess_sign_steps(20_000)
    b = cumulative_excess_sign_steps(40_000)
    assert b > cumulative_excess_sign_steps(2_000)
    assert (b - a) / b < 1e-4, "excess must converge, not grow with window length"
    assert b == pytest.approx(1212.57, rel=1e-3)
    # priced in epochs at the burn geometry (600 pairs / batch 8 => 75 steps/epoch)
    assert boundary_impulse_epochs(40_000, 75) == pytest.approx(b / 75.0, rel=1e-9)
    assert 16.0 < boundary_impulse_epochs(40_000, 75) < 16.2


def test_excess_is_front_loaded() -> None:
    """>=80% of the impulse lands in the first 13 epochs — measured by integration."""
    total = cumulative_excess_sign_steps(8000)
    first13 = cumulative_excess_sign_steps(13 * 75)
    assert first13 / total > 0.80


def test_eta_is_identically_one_when_beta_moments_are_unbiased() -> None:
    """Degenerate betas => no bias => eta == 1 at every step (a property, not a constant)."""
    for t in (1, 5, 50):
        assert effective_lr_multiplier(t, (0.0, 0.0)) == pytest.approx(1.0)


# ── displacement simulation vs the real optimizer ────────────────────────────────────
def test_cumulative_displacement_ratio_tracks_real_adam() -> None:
    """The numpy Adam recursion reproduces the real optimizer's cumulative displacement."""
    n = 30
    real = sum(_real_adam_displacement(n, bias_correction=False))
    g = np.full((1,), 0.37)
    sim = cumulative_displacement_ratio(np.zeros((1,)), g, n)
    assert sim == pytest.approx(real, rel=5e-3)


def test_displacement_decreases_monotonically_in_v0() -> None:
    """A larger starting v shrinks every step — the monotonicity the bisection relies on."""
    g = np.full((8,), 0.2)
    prev = float("inf")
    for s in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        d = cumulative_displacement_ratio(np.full((8,), s), g, 50)
        assert d < prev
        prev = d


# ── norm matching: it must actually MATCH ────────────────────────────────────────────
def test_norm_match_scalar_actually_achieves_the_match() -> None:
    """The returned scalar, fed back through the simulation, reproduces the target within tol.

    A stub returning a canonical scalar could not satisfy this round trip.
    """
    rng = np.random.default_rng(118)
    prior = rng.uniform(0.05, 4.0, size=64)
    grad = rng.uniform(0.01, 0.5, size=64)
    res = solve_norm_match_scalar(prior, grad, n_steps=100, tol=0.02)
    assert res.converged
    achieved = cumulative_displacement_ratio(res.scalar * prior, grad, 100)
    assert achieved == pytest.approx(res.target_displacement, rel=0.02)
    assert abs(res.achieved_ratio - 1.0) <= 0.02


def test_norm_match_is_prior_dependent() -> None:
    """Different priors need different scalars — proof the solve consumes its input."""
    grad = np.full((32,), 0.1)
    s_small = solve_norm_match_scalar(np.full((32,), 0.01), grad).scalar
    s_large = solve_norm_match_scalar(np.full((32,), 10.0), grad).scalar
    assert s_small > s_large * 10.0


def test_norm_match_refuses_non_positive_prior() -> None:
    with pytest.raises(ResetOperatorError, match="strictly positive"):
        solve_norm_match_scalar(np.array([1.0, 0.0]), np.array([0.1, 0.1]))


# ── the operator: arms are genuinely distinct ────────────────────────────────────────
_SHAPES = {"tokens_base": (2, 3), "renderer.w": (4,)}


def _prev_state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    out: dict[str, np.ndarray] = {}
    for k, s in _SHAPES.items():
        out[f"{k}::m"] = rng.normal(size=s).astype(np.float32)
        out[f"{k}::v"] = rng.uniform(0.1, 1.0, size=s).astype(np.float32)
    return out


def test_arm_b_is_byte_identical_incumbent() -> None:
    """Arm B returns an EMPTY state => the optimizer lazily zero-inits => incumbent behaviour."""
    assert apply_reset(_prev_state(), _SHAPES, ARM_B_ZERO_RESET) == {}
    assert ARM_B_ZERO_RESET.is_incumbent
    assert not ARM_B_ZERO_RESET.requires_persistence


def test_arm_bprime_differs_from_b_only_in_bias_correction() -> None:
    """B' is the surgical isolator: identical knobs, one field flipped."""
    b, bp = ARM_B_ZERO_RESET, ARM_BPRIME_BIAS_CORRECTED
    assert (b.what, b.to, b.structure) == (bp.what, bp.to, bp.structure)
    assert bp.bias_correction and not b.bias_correction
    assert not bp.is_incumbent  # it is NOT the incumbent, despite identical knobs


def test_arm_a_preserves_previous_moments_exactly() -> None:
    prev = _prev_state()
    out = apply_reset(prev, _SHAPES, ARM_A_NO_RESET)
    for k in _SHAPES:
        np.testing.assert_array_equal(out[f"{k}::m"], prev[f"{k}::m"])
        np.testing.assert_array_equal(out[f"{k}::v"], prev[f"{k}::v"])


def test_arm_c_zeros_m_but_keeps_v() -> None:
    """Arm C's defining behaviour — separating knob 1's two components."""
    prev = _prev_state()
    out = apply_reset(prev, _SHAPES, ARM_C_MOMENTUM_ONLY)
    for k in _SHAPES:
        assert np.all(out[f"{k}::m"] == 0.0)
        np.testing.assert_array_equal(out[f"{k}::v"], prev[f"{k}::v"])
        assert np.any(prev[f"{k}::m"] != 0.0)  # the m we zeroed was genuinely non-zero


def test_arm_d_uses_the_prior_and_the_scalar() -> None:
    prior = {k: np.full(s, 3.0, dtype=np.float32) for k, s in _SHAPES.items()}
    cfg = arm_config("Dminus", prior_path="x.npz")
    out = apply_reset({}, _SHAPES, cfg, prior=prior, scalar=0.5)
    for k in _SHAPES:
        assert np.allclose(out[f"{k}::v"], 1.5)
        assert np.all(out[f"{k}::m"] == 0.0)


def test_all_six_arms_are_pairwise_distinct_configs() -> None:
    seen = {(a.what, a.to, a.structure, a.bias_correction, a.prior_exponent)
            for a in ARMS.values()}
    assert len(seen) == len(ARMS) == 6


def test_resolve_arm_name_round_trips() -> None:
    for name, cfg in ARMS.items():
        assert resolve_arm_name(cfg) == name


# ── fail-closed contracts (NO-FAKE: never silently become arm B) ─────────────────────
def test_arm_a_refuses_when_state_was_not_persisted() -> None:
    with pytest.raises(ResetOperatorError, match="previous optimizer state"):
        apply_reset({}, _SHAPES, ARM_A_NO_RESET)


def test_arm_c_refuses_when_state_was_not_persisted() -> None:
    with pytest.raises(ResetOperatorError, match="previous optimizer state"):
        apply_reset({}, _SHAPES, ARM_C_MOMENTUM_ONLY)


def test_prior_arm_refuses_without_prior_path() -> None:
    with pytest.raises(ResetOperatorError, match="silently degrade"):
        ResetOperatorConfig(what="both", to="prior")


def test_prior_arm_refuses_without_loaded_prior() -> None:
    cfg = arm_config("Dplus", prior_path="x.npz")
    with pytest.raises(ResetOperatorError, match="requires a loaded prior"):
        apply_reset({}, _SHAPES, cfg)


def test_requires_persistence_is_exactly_arms_a_and_c() -> None:
    need = {n for n, c in ARMS.items() if c.requires_persistence}
    assert need == {"A", "C"}


def test_shape_mismatch_in_persisted_state_raises() -> None:
    prev = _prev_state()
    prev["renderer.w::m"] = np.zeros((99,), dtype=np.float32)
    with pytest.raises(ResetOperatorError, match="shape"):
        apply_reset(prev, _SHAPES, ARM_A_NO_RESET)


def test_invalid_knob_values_raise() -> None:
    with pytest.raises(ResetOperatorError, match="what must be"):
        ResetOperatorConfig(what="bogus")
    with pytest.raises(ResetOperatorError, match="to must be"):
        ResetOperatorConfig(to="bogus")
    with pytest.raises(ResetOperatorError, match="structure must be"):
        ResetOperatorConfig(structure="bogus")


def test_unknown_arm_name_lists_the_real_ones() -> None:
    with pytest.raises(ResetOperatorError, match="pre-registered arms"):
        arm_config("Z")


# ── prior loader ─────────────────────────────────────────────────────────────────────
def test_load_diagonal_prior_applies_exponent_and_broadcasts(tmp_path) -> None:
    p = tmp_path / "prior.npz"
    np.savez(p, **{"tokens_base": np.full((3,), 2.0), "renderer.w": np.full((4,), 4.0)})
    out = load_diagonal_prior(p, _SHAPES, exponent=-2.0)
    assert out["tokens_base"].shape == (2, 3)          # broadcast along the last axis
    assert np.allclose(out["tokens_base"], 0.25)       # 2^-2
    assert np.allclose(out["renderer.w"], 0.0625)      # 4^-2


def test_load_diagonal_prior_refuses_partial_coverage(tmp_path) -> None:
    p = tmp_path / "prior.npz"
    np.savez(p, **{"tokens_base": np.full((3,), 2.0)})
    with pytest.raises(ResetOperatorError, match="missing"):
        load_diagonal_prior(p, _SHAPES)


def test_load_diagonal_prior_refuses_non_positive(tmp_path) -> None:
    p = tmp_path / "prior.npz"
    np.savez(p, **{"tokens_base": np.zeros((3,)), "renderer.w": np.full((4,), 1.0)})
    with pytest.raises(ResetOperatorError, match="strictly positive"):
        load_diagonal_prior(p, _SHAPES)


def test_load_diagonal_prior_refuses_unbroadcastable(tmp_path) -> None:
    p = tmp_path / "prior.npz"
    np.savez(p, **{"tokens_base": np.full((7,), 2.0), "renderer.w": np.full((4,), 1.0)})
    with pytest.raises(ResetOperatorError, match="broadcastable"):
        load_diagonal_prior(p, _SHAPES)


# ── the explicit mutation guard ──────────────────────────────────────────────────────
def test_mutation_guard_stub_body_would_fail() -> None:
    """Assert the properties a marker-returning stub could NOT satisfy (NO-FAKE class #2).

    (1) eta varies with its argument; (2) the norm-match scalar varies with its input;
    (3) different arms produce different states from the SAME inputs.
    """
    assert len({round(effective_lr_multiplier(t), 6) for t in (1, 12, 500)}) == 3
    g = np.full((16,), 0.2)
    s1 = solve_norm_match_scalar(np.full((16,), 0.5), g).scalar
    s2 = solve_norm_match_scalar(np.full((16,), 5.0), g).scalar
    assert s1 != s2
    prev = _prev_state()
    states = [
        tuple(np.asarray(v).tobytes() for v in sorted_state(apply_reset(prev, _SHAPES, c)))
        for c in (ARM_A_NO_RESET, ARM_C_MOMENTUM_ONLY)
    ]
    assert states[0] != states[1]
    assert apply_reset(prev, _SHAPES, ARM_B_ZERO_RESET) == {}


def sorted_state(d: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [d[k] for k in sorted(d)]


def test_betas_are_validated() -> None:
    with pytest.raises(ResetOperatorError, match="b2"):
        effective_lr_multiplier(1, (0.9, 1.0))
    with pytest.raises(ResetOperatorError, match="2-sequence"):
        cumulative_excess_sign_steps(5, (0.9,))
    assert DEFAULT_BETAS == (0.9, 0.999)
