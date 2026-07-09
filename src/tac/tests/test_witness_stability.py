# SPDX-License-Identifier: MIT
"""Tests for tac.witness_stability — the #146/#211 AMBER deep-unroll collapse-fix levers.

Covers: the pose-eps <-> max-grad-coefficient LAW (exact + invertible), the byte-identity default
contract, the amber preset composition, the Cells2Pixels per-param normalize + overflow primitives, a
SYNTHETIC reproduction of the 5e4-coefficient blowup that the fix TAMES, and the stage-boundary
loss-weight guard. All $0, no GPU, no mlx (framework-free primitives)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.witness_stability import (
    AMBER,
    PRESETS,
    SCORE_POSE_CONST,
    StabilityViolation,
    assert_loss_weights_stage_boundary_only,
    max_pose_grad_coeff,
    overflow_state_penalty,
    per_param_normalize_grads,
    pose_eps_floor_for_coeff_max,
    resolve_effective_pose_eps,
    resolve_stability_config,
)


# --- the coefficient <-> eps LAW (exact) -----------------------------------------------------------
def test_max_coeff_at_1em8_is_the_5e4_blowup():
    # 5/sqrt(1e-8) = 5e4 — the exact coefficient the AMBER diagnosis saw.
    assert max_pose_grad_coeff(1e-8) == pytest.approx(5.0e4, rel=1e-9)


def test_max_coeff_at_incumbent_1em2_is_50():
    # the live #205 default pose_eps=1e-2 bounds the coefficient to 50 (1000x below the blowup).
    assert max_pose_grad_coeff(1e-2) == pytest.approx(50.0, rel=1e-12)


def test_eps_floor_inverts_max_coeff_roundtrip():
    for c in (10.0, 25.0, 50.0, 137.0):
        eps = pose_eps_floor_for_coeff_max(c)
        assert max_pose_grad_coeff(eps) == pytest.approx(c, rel=1e-9)


def test_eps_floor_known_values():
    assert pose_eps_floor_for_coeff_max(50.0) == pytest.approx(1e-2, rel=1e-12)
    assert pose_eps_floor_for_coeff_max(25.0) == pytest.approx(4e-2, rel=1e-12)


def test_max_coeff_rejects_nonpositive_eps():
    with pytest.raises(ValueError):
        max_pose_grad_coeff(0.0)
    with pytest.raises(ValueError):
        max_pose_grad_coeff(-1e-3)


def test_eps_floor_rejects_nonpositive_coeff():
    with pytest.raises(ValueError):
        pose_eps_floor_for_coeff_max(0.0)


def test_score_pose_const_is_ten():
    assert SCORE_POSE_CONST == 10.0


# --- resolve_effective_pose_eps: byte-identity + tighten-only ---------------------------------------
def test_effective_eps_default_off_is_byte_identical():
    # coeff_max=0 (default) => pose_eps returned UNCHANGED (identity, byte-identical #205 path).
    assert resolve_effective_pose_eps(1e-2, 0.0) == 1e-2
    assert resolve_effective_pose_eps(1e-8, 0.0) == 1e-8


def test_effective_eps_only_raises_never_lowers():
    # a LOOSER coeff bound whose floor is below the incumbent eps must NOT lower it.
    # coeff_max=100 => floor (5/100)^2 = 2.5e-3 < 1e-2 => eps unchanged.
    assert resolve_effective_pose_eps(1e-2, pose_grad_coeff_max=100.0) == 1e-2
    # a TIGHTER bound raises it.
    assert resolve_effective_pose_eps(1e-2, pose_grad_coeff_max=25.0) == pytest.approx(4e-2, rel=1e-12)


# --- resolve_stability_config: presets + byte-identity contract -------------------------------------
def test_config_default_none_is_unchanged():
    c = resolve_stability_config(grad_clip=1.0, pose_eps=1e-2, pose_grad_coeff_max=0.0,
                                 stability_preset="none", per_group_grad_clip=False)
    assert c.changed is False
    assert c.grad_clip == 1.0 and c.effective_pose_eps == 1e-2 and c.per_group_grad_clip is False


def test_config_amber_composes_the_tighter_cures():
    c = resolve_stability_config(grad_clip=1.0, pose_eps=1e-2, stability_preset="amber")
    assert c.changed is True
    assert c.grad_clip == 0.5
    assert c.per_group_grad_clip is True
    assert c.effective_pose_eps == pytest.approx(4e-2, rel=1e-12)
    assert c.max_pose_grad_coeff_effective == pytest.approx(25.0, rel=1e-9)
    assert AMBER.pose_grad_coeff_max == 25.0 and "amber" in PRESETS


def test_config_explicit_coeff_max_overrides_preset():
    # explicit coeff-max (40) beats amber's default (25) => eps floor (5/40)^2 = 0.015625.
    c = resolve_stability_config(grad_clip=1.0, pose_eps=1e-2, pose_grad_coeff_max=40.0,
                                 stability_preset="amber")
    assert c.pose_grad_coeff_max == 40.0
    assert c.effective_pose_eps == pytest.approx((SCORE_POSE_CONST / (2 * 40.0)) ** 2, rel=1e-12)


def test_config_rejects_unknown_preset():
    with pytest.raises(ValueError):
        resolve_stability_config(grad_clip=1.0, pose_eps=1e-2, stability_preset="bogus")


def test_config_coeff_max_only_no_preset():
    c = resolve_stability_config(grad_clip=1.0, pose_eps=1e-2, pose_grad_coeff_max=25.0,
                                 stability_preset="none")
    assert c.changed is True
    assert c.grad_clip == 1.0  # preset none => grad_clip untouched
    assert c.effective_pose_eps == pytest.approx(4e-2, rel=1e-12)


# --- the SYNTHETIC blowup reproduction the fix tames ------------------------------------------------
def test_synthetic_blowup_is_tamed_by_the_coeff_bound():
    # Reproduce the diagnosis on an EASY (near-zero-pose) pair: the score-domain pose gradient
    # coefficient 5/sqrt(10*p+eps) explodes at eps=1e-8, and the amber coeff bound caps it at 25.
    def pose_grad_coeff(p, eps):
        return SCORE_POSE_CONST / (2.0 * math.sqrt(SCORE_POSE_CONST * p + eps))

    easy_p = 0.0  # a near-zero-pose (easy) pair — the worst case
    blown = pose_grad_coeff(easy_p, 1e-8)
    assert blown > 4.9e4  # the 5e4 blowup the batch=1 update exposes

    # the fix: raise eps to the amber floor so the SAME easy pair's coefficient is bounded.
    amber_eps = pose_eps_floor_for_coeff_max(25.0)
    tamed = pose_grad_coeff(easy_p, amber_eps)
    assert tamed == pytest.approx(25.0, rel=1e-9)
    assert tamed < blown / 1000.0  # >1000x reduction


# --- Cells2Pixels per-param normalize ---------------------------------------------------------------
def _dict_tree_map(fn, tree):
    return {k: fn(v) for k, v in tree.items()}


def test_per_param_normalize_makes_each_tensor_unit_norm():
    grads = {"a": np.array([3.0, 4.0]), "b": np.array([1.0, 0.0, 0.0])}
    out = per_param_normalize_grads(
        grads, tree_map=lambda fn, g=grads: _dict_tree_map(fn, g),
        leaf_norm=lambda x: float(np.sqrt((x * x).sum())), eps=0.0)
    assert np.sqrt((out["a"] ** 2).sum()) == pytest.approx(1.0, rel=1e-12)
    assert np.sqrt((out["b"] ** 2).sum()) == pytest.approx(1.0, rel=1e-12)


def test_per_param_normalize_zero_grad_stays_finite():
    grads = {"z": np.array([0.0, 0.0])}
    out = per_param_normalize_grads(
        grads, tree_map=lambda fn, g=grads: _dict_tree_map(fn, g),
        leaf_norm=lambda x: float(np.sqrt((x * x).sum())), eps=1e-8)
    assert np.all(np.isfinite(out["z"]))
    assert np.allclose(out["z"], 0.0)


# --- Cells2Pixels overflow / state-clamp penalty ----------------------------------------------------
def test_overflow_penalty_zero_when_below_clamp():
    assert overflow_state_penalty(np.array([0.0, 0.5, 1.0]), clamp=1.0) == 0.0


def test_overflow_penalty_hinge_above_clamp():
    # (|2|-1 + |1|-1)/2 = (1 + 0)/2 = 0.5
    assert overflow_state_penalty(np.array([2.0, 1.0]), clamp=1.0) == pytest.approx(0.5, rel=1e-12)


# --- stage-boundary loss-weight guard (SPEC_v75 §8-C) -----------------------------------------------
def test_stage_boundary_guard_passes_when_weights_constant():
    events = [(e, 100.0, 0.0) for e in range(10)]
    assert_loss_weights_stage_boundary_only(events, stage_boundary_epochs={0, 5})  # no raise


def test_stage_boundary_guard_passes_change_at_boundary():
    events = [(0, 100.0, 0.0), (4, 100.0, 0.0), (5, 100.0, 1.0), (6, 100.0, 1.0)]
    assert_loss_weights_stage_boundary_only(events, stage_boundary_epochs={0, 5})  # change at 5 = ok


def test_stage_boundary_guard_raises_on_midstage_change():
    events = [(0, 100.0, 0.0), (3, 100.0, 5.0)]  # weight changed at 3 (not a boundary)
    with pytest.raises(StabilityViolation):
        assert_loss_weights_stage_boundary_only(events, stage_boundary_epochs={0, 5})


# --- the canonical equation leg registers + the DSL leg auto-discovers ------------------------------
def test_canonical_equation_builds_and_is_valid():
    from tac.canonical_equations.witness_pose_grad_coeff_stability_20260709 import (
        build_witness_pose_grad_coeff_stability_v1,
    )

    eq = build_witness_pose_grad_coeff_stability_v1()
    assert eq.equation_id == "witness_pose_grad_coeff_stability_v1"
    assert eq.canonical_producers and eq.canonical_consumers  # no orphan


def test_dsl_lever_auto_discovered_never_fired():
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers

    assert "WitnessStability" in known_levers()
    assert "WitnessStability" in duty_to_measure()  # duty-to-measure (un-collapse A/B owed)


def test_dsl_lever_flags_are_all_held_no_orphan():
    # never-invent-flags / config-orphan: every flag the lever emits must be held by the DSL registry.
    from tac.witness_dsl.curriculum_dsl import WitnessStability
    from tac.witness_dsl.lever_registry import completeness

    unmapped = set(completeness().unmapped or [])
    for flag in ("--stability-preset", "--pose-grad-coeff-max", "--grad-normalize"):
        assert flag not in unmapped, f"{flag} is an unmapped config-orphan"
    # grad_normalize is opt-in + a distinct A/B arm (NOT in amber's default overrides)
    assert "--grad-normalize" not in WitnessStability().overrides
    assert WitnessStability(grad_normalize="per-param").overrides["--grad-normalize"] == "per-param"


def test_dsl_lever_rejects_bad_grad_normalize():
    from tac.witness_dsl.curriculum_dsl import WitnessStability

    with pytest.raises(ValueError):
        WitnessStability(grad_normalize="bogus")
