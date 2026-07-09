# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the #310 step-native activation lever (mechanism + DSL factory).

Proves (the NO-FAKE contract): the step-native basis ACTUALLY computes on real inputs and DIFFERS from
sine; the β anneal schedule is the numpy twin of the trainer; the DSL factory maps ONLY to EXISTING
trainer flags, is byte-identical when NOT composed, and FAILS CLOSED on the MEASURED fixed-β
saturation-death (NEVER emits a constant β). $0 numpy (no MLX/GPU)."""
from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.step_native_activation import (
    ANNEALED_HOSC,
    STEP_BASIS,
    beta_anneal_schedule,
    hosc_activation,
    resolve_beta_end,
    sine_basis,
    step_native_limit,
    validate_step_native_config,
)


# ── mechanism: the step-native basis genuinely computes + DIFFERS from sine ─────────────────────────
def test_hosc_activation_is_tanh_beta_sin():
    u = np.linspace(-np.pi, np.pi, 101)
    got = hosc_activation(u, beta=2.0, omega=1.0)
    assert np.allclose(got, np.tanh(2.0 * np.sin(u)))


def test_high_beta_hosc_approaches_step_limit_not_sine():
    # β→∞ ⇒ tanh(β·sin) → sign(sin) (a STEP), which is NOT sine. Avoid the exact zero-crossings
    # (measure-zero set where sign is 0) so the a.e. convergence is what we assert.
    u = np.linspace(-3.0, 3.0, 200)
    u = u[np.abs(np.sin(u)) > 0.05]
    step = step_native_limit(u)
    hi = hosc_activation(u, beta=64.0, omega=1.0)
    # hosc at high β is CLOSE to the step limit ...
    assert np.max(np.abs(hi - step)) < 1e-2
    # ... and FAR from the smooth sine basis (proving it is not a relabeled sine).
    assert np.max(np.abs(hi - sine_basis(u))) > 0.3


def test_step_native_limit_is_square_wave():
    u = np.linspace(-3.0, 3.0, 200)
    assert np.array_equal(step_native_limit(u, omega=1.0), np.sign(np.sin(u)))
    # a square wave takes only {-1, 0, +1}
    assert set(np.unique(step_native_limit(u))).issubset({-1.0, 0.0, 1.0})


def test_low_beta_hosc_is_near_linear_in_sin():
    # small β: tanh(β·sin) ≈ β·sin (gradients flow — why the anneal STARTS small).
    u = np.linspace(-1.0, 1.0, 50)
    lo = hosc_activation(u, beta=0.01, omega=1.0)
    assert np.max(np.abs(lo - 0.01 * np.sin(u))) < 1e-4


def test_hosc_activation_rejects_nonpositive_beta_omega():
    u = np.linspace(-1, 1, 10)
    with pytest.raises(ValueError):
        hosc_activation(u, beta=0.0)
    with pytest.raises(ValueError):
        hosc_activation(u, beta=1.0, omega=0.0)


# ── β anneal schedule = numpy twin of the trainer _hosc_beta_for_epoch ───────────────────────────────
def test_beta_anneal_linear_endpoints_and_monotone():
    epochs = 100
    assert beta_anneal_schedule(1, epochs, 1.0, 4.0, "linear") == pytest.approx(1.0)
    assert beta_anneal_schedule(epochs, epochs, 1.0, 4.0, "linear") == pytest.approx(4.0)
    seq = [beta_anneal_schedule(e, epochs, 1.0, 4.0, "linear") for e in range(1, epochs + 1)]
    assert all(b <= a + 1e-9 for b, a in zip(seq, seq[1:]))  # monotone increasing


def test_beta_anneal_matches_trainer_linear_form():
    # (ep-1)/max(A-1,1) linear form, mirroring _hosc_beta_for_epoch.
    epochs, b0, b1 = 50, 1.0, 8.0
    for ep in (1, 7, 23, 50):
        prog = (ep - 1) / max(epochs - 1, 1)
        assert beta_anneal_schedule(ep, epochs, b0, b1, "linear") == pytest.approx(b0 + (b1 - b0) * prog)


def test_beta_anneal_cosine_endpoints():
    epochs = 40
    assert beta_anneal_schedule(1, epochs, 1.0, 8.0, "cosine") == pytest.approx(1.0)
    assert beta_anneal_schedule(epochs, epochs, 1.0, 8.0, "cosine") == pytest.approx(8.0)


def test_beta_anneal_constant_when_no_end_or_equal():
    # β_end None OR == β_start ⇒ CONSTANT β_start (the no-anneal / bit-identical path).
    for ep in (1, 10, 100):
        assert beta_anneal_schedule(ep, 100, 3.0, None, "linear") == 3.0
        assert beta_anneal_schedule(ep, 100, 3.0, 3.0, "linear") == 3.0


def test_beta_anneal_honors_anneal_epochs_denominator():
    # (review C2) denominator is anneal_epochs when set, NOT epochs — mirrors the trainer.
    assert beta_anneal_schedule(21, 100, 1.0, 5.0, "linear", anneal_epochs=21) == pytest.approx(5.0)


def test_beta_anneal_clamps_ep_beyond_schedule():
    # ep past the schedule length stays at β_end (clamp, not overshoot).
    assert beta_anneal_schedule(200, 50, 1.0, 4.0, "linear") == pytest.approx(4.0)


def test_beta_anneal_rejects_bad_shape():
    with pytest.raises(ValueError):
        beta_anneal_schedule(1, 10, 1.0, 4.0, "quadratic")


# ── resolve_beta_end + validate: the fail-closed SAFETY predicate ────────────────────────────────────
def test_resolve_beta_end_per_basis_defaults():
    assert resolve_beta_end(ANNEALED_HOSC, None) == 4.0
    assert resolve_beta_end(STEP_BASIS, None) == 8.0
    assert resolve_beta_end(STEP_BASIS, 6.0) == 6.0
    with pytest.raises(ValueError):
        resolve_beta_end("bogus", None)


def test_validate_accepts_annealed_hosc_default():
    assert validate_step_native_config(ANNEALED_HOSC, 1.0, None, "linear", 1.0, True) == []
    assert validate_step_native_config(ANNEALED_HOSC, 1.0, 4.0, "cosine", 1.0, False) == []


def test_validate_accepts_step_basis_with_finer():
    assert validate_step_native_config(STEP_BASIS, 1.0, None, "linear", 1.0, True) == []


def test_validate_rejects_fixed_beta_the_saturation_death():
    # β_end == β_start is the MEASURED fixed-β saturation-death — MUST fail closed.
    prob = validate_step_native_config(ANNEALED_HOSC, 4.0, 4.0, "linear", 1.0, True)
    assert prob and any("FIXED-beta" in p for p in prob)
    # the classic forbidden default (fixed β=4): resolving to the annealed default (4.0) with start 4.0.
    prob2 = validate_step_native_config(ANNEALED_HOSC, 4.0, None, "linear", 1.0, True)
    assert prob2 and any("FIXED-beta" in p for p in prob2)


def test_validate_rejects_step_basis_without_finer():
    prob = validate_step_native_config(STEP_BASIS, 1.0, None, "linear", 1.0, False)
    assert prob and any("finer_bias_init=True" in p for p in prob)


def test_validate_rejects_invented_basis_and_shape():
    assert validate_step_native_config("wire", 1.0, 4.0, "linear", 1.0, True)
    assert validate_step_native_config(ANNEALED_HOSC, 1.0, 4.0, "sigmoid", 1.0, True)


def test_validate_rejects_nonpositive_beta():
    assert validate_step_native_config(ANNEALED_HOSC, 0.0, 4.0, "linear", 1.0, True)


# ── DSL factory (the EXISTING FEED-07b lever, ENHANCED): maps to EXISTING flags, byte-identity when
#    off, fail-closed on the forbidden FIXED-beta config ───────────────────────────────────────────────
_LEVER = "FEED_07b_step_native_activation"


def test_dsl_factory_default_emits_existing_flags_only():
    from tac.witness_dsl.curriculum_dsl import StepNativeActivation, real_trainer_flags

    lv = StepNativeActivation()
    assert lv.name == _LEVER
    assert lv.overrides["--activation"] == "hosc"
    assert lv.overrides["--hosc-beta"] == 1.0                # lowered default (start where grads flow)
    assert lv.overrides["--hosc-beta-end"] == 8.0            # anneal TARGET (step limit)
    assert lv.overrides["--hosc-beta-anneal"] == "linear"
    assert lv.overrides["--hosc-omega"] == 1.0
    # FINER is OFF by default for annealed_hosc → the finer flags are NOT emitted (byte-neutral default).
    assert "--finer-bias-init" not in lv.overrides
    # every emitted flag must be a REAL trainer flag (never-invent-flags).
    trainer = set(real_trainer_flags())
    for flag in lv.overrides:
        assert flag in trainer, f"{flag} is not a real trainer flag"


def test_dsl_factory_finer_arms_finer_flags_in_same_lever():
    from tac.witness_dsl.curriculum_dsl import StepNativeActivation

    lv = StepNativeActivation(finer_bias_init=True, finer_bias_k=12.0)
    assert lv.overrides["--finer-bias-init"] is True
    assert lv.overrides["--finer-bias-k"] == 12.0


def test_dsl_factory_step_basis_requires_finer():
    from tac.witness_dsl.curriculum_dsl import StepNativeActivation

    lv = StepNativeActivation(basis="step_basis", finer_bias_init=True)
    assert lv.overrides["--finer-bias-init"] is True
    # step_basis WITHOUT FINER is the un-stabilized sharper step → fail closed.
    with pytest.raises(ValueError):
        StepNativeActivation(basis="step_basis", finer_bias_init=False)


def test_dsl_factory_fails_closed_on_fixed_beta():
    from tac.witness_dsl.curriculum_dsl import StepNativeActivation

    # constant β (the saturation-death) MUST raise — the lever can NEVER emit a fixed β. The prior
    # guard (beta_start <= beta_end) ALLOWED beta_start == beta_end; the enhanced guard rejects it.
    with pytest.raises(ValueError):
        StepNativeActivation(beta_start=4.0, beta_end=4.0)
    with pytest.raises(ValueError):
        StepNativeActivation(beta_start=8.0, beta_end=8.0)


def test_dsl_factory_rejects_invented_basis_and_anneal():
    from tac.witness_dsl.curriculum_dsl import StepNativeActivation

    with pytest.raises(ValueError):
        StepNativeActivation(basis="relu")
    with pytest.raises(ValueError):
        StepNativeActivation(anneal="sigmoid")


def test_lever_absent_is_byte_identical_argv():
    # BYTE-IDENTITY WHEN OFF: a program that does NOT compose StepNativeActivation compiles argv
    # identical to the same program without it (the lever only changes argv when ON).
    import dataclasses

    from tac.witness_dsl.curriculum_dsl import BASELINE, StepNativeActivation

    argv_off = BASELINE.compile_trainer_argv()
    prog_on = dataclasses.replace(BASELINE, levers=(*BASELINE.levers, StepNativeActivation()))
    argv_on = prog_on.compile_trainer_argv()
    assert argv_off != argv_on                              # composing it DOES change argv
    assert "--activation" in argv_on
    # BASELINE itself is UNMUTATED (the OFF path is untouched by the lever existing).
    assert BASELINE.compile_trainer_argv() == argv_off


def test_factory_is_discovered_as_known_lever():
    # the activation ledger derives known levers from the DSL AST by FACTORY def name → the
    # StepNativeActivation factory must be a KNOWN, NEVER-FIRED, duty-to-measure lever (the whole point).
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers, never_fired

    assert "StepNativeActivation" in known_levers()
    assert "StepNativeActivation" in never_fired()
    assert "StepNativeActivation" in duty_to_measure()


def test_completeness_maps_the_step_native_flags():
    # completeness() must now MARK the hosc/finer flags as mapped (the DSL holds them) — no longer
    # unmapped orphans.
    from tac.witness_dsl.lever_registry import completeness

    comp = completeness()
    for flag in ("--activation", "--hosc-beta", "--hosc-beta-end", "--hosc-beta-anneal",
                 "--hosc-omega", "--finer-bias-init", "--finer-bias-k"):
        assert flag not in comp.unmapped, f"{flag} still unmapped — DSL factory did not reference it"
