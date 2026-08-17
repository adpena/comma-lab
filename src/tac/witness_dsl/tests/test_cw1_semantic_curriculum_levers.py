# SPDX-License-Identifier: MIT
"""Tests for the CW1 semantic-renderer curriculum / lr / cadence levers.

These verify BEHAVIOUR, not constants.  The two closed forms the levers rest on (the
phase share of the integrated LR budget, and its scale-invariance) are RECOMPUTED here
from the trainer's own source semantics -- so if ``curriculum_loss``'s progress rule or
the ``CosineAnnealingLR`` construction ever changes, these tests fail rather than keep
asserting a stale number.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tac.witness_dsl.cw1_semantic_curriculum_levers_20260817 import (
    ALIGNED_LR_BUDGET_EF,
    ALIGNED_LR_BUDGET_STOCK,
    COS_SIGN_AT_INIT,
    LADDER_FLIPS_AT_600,
    TRAINER_DEFAULT_CE_FRACTION,
    TRAINER_DEFAULT_SOFTPLUS_FRACTION,
    TRAINER_RELPATH,
    lever_cw1_aligned_objective,
    lever_cw1_lr_budget,
    lever_cw1_observation_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


# --------------------------------------------------------------------------- forms


def _lr_at(k: int, n: int, base: float = 1.0) -> float:
    """CosineAnnealingLR(T_max=n, eta_min=0.01*base) at iteration k."""
    eta_min = base * 0.01
    return eta_min + (base - eta_min) * (1 + math.cos(math.pi * k / n)) / 2


def _phase(k: int, n: int, ce_f: float, sp_f: float) -> str:
    """curriculum_loss's phase selector, verbatim."""
    progress = k / max(n - 1, 1)
    if progress < ce_f:
        return "ce"
    if progress < sp_f:
        return "softplus_margin"
    return "expected_flip"


def _aligned_budget(n: int, ce_f: float, sp_f: float) -> float:
    per = {"ce": 0.0, "softplus_margin": 0.0, "expected_flip": 0.0}
    for k in range(n):
        per[_phase(k, n, ce_f, sp_f)] += _lr_at(k, n)
    total = sum(per.values())
    return sum(per[p] / total * COS_SIGN_AT_INIT[p] for p in per)


# ------------------------------------------------------------------ source contract


def test_trainer_relpath_exists_and_declares_the_flags_we_emit():
    trainer = REPO_ROOT / TRAINER_RELPATH
    assert trainer.is_file(), f"TRAINER_RELPATH does not resolve: {trainer}"
    src = trainer.read_text()
    declared = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))
    for flag in ("--ce-fraction", "--softplus-fraction", "--lr", "--eval-every", "--checkpoint-every"):
        assert flag in declared, f"never-invent-flags: {flag} is not in the trainer argparse"


def test_trainer_defaults_recorded_here_match_the_trainer():
    src = (REPO_ROOT / TRAINER_RELPATH).read_text()
    ce = re.search(r'add_argument\("--ce-fraction",\s*type=float,\s*default=([0-9.]+)\)', src)
    sp = re.search(r'add_argument\("--softplus-fraction",\s*type=float,\s*default=([0-9.]+)\)', src)
    assert ce and sp, "could not read the curriculum defaults from the trainer argparse"
    assert float(ce.group(1)) == TRAINER_DEFAULT_CE_FRACTION
    assert float(sp.group(1)) == TRAINER_DEFAULT_SOFTPLUS_FRACTION


def test_scheduler_is_still_cosine_with_eta_min_tied_to_lr():
    """The lr lever's whole rationale is eta_min = 0.01*lr. Guard the premise."""
    src = (REPO_ROOT / TRAINER_RELPATH).read_text()
    assert "CosineAnnealingLR(" in src
    assert "T_max=args.steps" in src
    assert "eta_min=args.lr * 0.01" in src


# ------------------------------------------------------------- the aligned objective


def test_aligned_objective_emits_both_fractions_at_zero():
    lever = lever_cw1_aligned_objective()
    assert lever.overrides == {"--ce-fraction": 0.0, "--softplus-fraction": 0.0}


def test_aligned_objective_carries_measured_anchor_provenance_on_both_flags():
    lever = lever_cw1_aligned_objective()
    for flag in ("--ce-fraction", "--softplus-fraction"):
        assert lever.lawrefs[flag].ladder_class == "measured_anchor"
        assert lever.constant_manifest[flag]["equation_id"] == "semantic_curriculum_alignment_ladder_v1"
    assert lever.policy_contracts["score_claim"] is False
    assert lever.policy_contracts["is_frontier_vehicle"] is False


def test_zero_fractions_actually_select_expected_flip_from_step_one():
    """Behaviour, not a constant: progress < 0.0 is False at every step."""
    for n in (600, 3000, 6000):
        assert {_phase(k, n, 0.0, 0.0) for k in range(n)} == {"expected_flip"}


def test_the_ladder_numbers_recorded_here_are_internally_consistent():
    stock, ef0 = LADDER_FLIPS_AT_600["stock"], LADDER_FLIPS_AT_600["ef0"]
    assert stock / ef0 == pytest.approx(13.6069, abs=1e-3)
    assert (stock - ef0) / stock == pytest.approx(0.92651, abs=1e-4)
    assert LADDER_FLIPS_AT_600["ce0"] < stock, "the ladder must be monotone"
    assert ef0 < LADDER_FLIPS_AT_600["ce0"], "the ladder must be monotone"


# ------------------------------------------------------- the closed form, recomputed


@pytest.mark.parametrize("steps", [600, 3000, 30000])
def test_stock_aligned_budget_matches_the_recorded_constant(steps):
    assert _aligned_budget(steps, TRAINER_DEFAULT_CE_FRACTION, TRAINER_DEFAULT_SOFTPLUS_FRACTION) == (
        pytest.approx(ALIGNED_LR_BUDGET_STOCK, abs=5e-4)
    )


@pytest.mark.parametrize("steps", [600, 3000, 6000])
def test_ef_aligned_budget_is_exactly_the_expected_flip_cosine(steps):
    assert _aligned_budget(steps, 0.0, 0.0) == pytest.approx(ALIGNED_LR_BUDGET_EF, abs=1e-9)


def test_the_split_is_scale_invariant_which_is_why_longer_windows_cannot_fix_it():
    budgets = [
        _aligned_budget(n, TRAINER_DEFAULT_CE_FRACTION, TRAINER_DEFAULT_SOFTPLUS_FRACTION)
        for n in (600, 3000, 30000)
    ]
    assert max(budgets) - min(budgets) < 2e-4, (
        "the aligned budget must be invariant across a 50x window range; if this fails the "
        "scale-invariance argument -- and therefore the whole 'no window reaches parity' "
        "reading -- no longer holds"
    )


def test_alignment_strictly_improves_from_stock_to_ce0_to_ef():
    stock = _aligned_budget(3000, 0.50, 0.85)
    ce0 = _aligned_budget(3000, 0.00, 0.85)
    ef = _aligned_budget(3000, 0.00, 0.00)
    assert stock < ce0 < ef, "the ladder's alignment ordering must match its measured seg ordering"


# -------------------------------------------------------------------- the lr lever


def test_lr_lever_emits_lr_and_is_typed_as_an_unmeasured_swept_axis():
    lever = lever_cw1_lr_budget(6.0e-5)
    assert lever.overrides == {"--lr": 6.0e-5}
    ref = lever.lawrefs["--lr"]
    assert ref.ladder_class == "hardcoded_waiver"
    assert ref.fallback == 6.0e-5
    assert "rederivation_trigger" in ref.fallback_waiver_reason
    assert lever.policy_contracts["optimum_measured"] is False
    assert lever.constant_manifest["--lr"]["eta_min"] == pytest.approx(6.0e-7)


def test_lr_lever_names_are_distinct_per_value_so_two_rungs_cannot_collide():
    assert lever_cw1_lr_budget(6.0e-5).name != lever_cw1_lr_budget(2.0e-4).name


@pytest.mark.parametrize("bad", [0.0, -1e-5])
def test_lr_lever_refuses_non_positive_peak(bad):
    with pytest.raises(ValueError, match="must be > 0"):
        lever_cw1_lr_budget(bad)


def test_integrated_lr_is_steps_times_half_the_peak_which_is_the_ladder_currency():
    """steps * 0.505 * lr, the identity the geometric rung spacing is derived from."""
    for n, base in ((3000, 2.0e-5), (6000, 2.0e-5), (3000, 6.0e-5)):
        integrated = sum(_lr_at(k, n, base) for k in range(n))
        assert integrated == pytest.approx(n * 0.505 * base, rel=2e-3)


# ------------------------------------------------------------- the cadence lever


def test_cadence_lever_emits_both_cadences():
    lever = lever_cw1_observation_budget(250, 200)
    assert lever.overrides == {"--eval-every": 250, "--checkpoint-every": 200}
    assert lever.policy_contracts["trajectory_neutral"] is True


def test_cadence_lever_refuses_collinear_cadences():
    """The exact defect that produced a wall-clock law implying a negative fixed cost."""
    with pytest.raises(ValueError, match="COLLINEAR"):
        lever_cw1_observation_budget(250, 250)


@pytest.mark.parametrize("ev,ck", [(0, 200), (250, 0), (-1, 200)])
def test_cadence_lever_refuses_sub_one_cadences(ev, ck):
    with pytest.raises(ValueError, match=">= 1"):
        lever_cw1_observation_budget(ev, ck)


def test_cadence_default_is_non_collinear():
    lever = lever_cw1_observation_budget()
    assert lever.overrides["--eval-every"] != lever.overrides["--checkpoint-every"]


# ------------------------------------------------------------ registry integration


def test_package_registry_binds_every_factory_to_the_semantic_trainer_with_no_missing_flags():
    from tac.witness_dsl import lever_registry as registry

    builds = [b for b in registry.package_lever_factories() if b.factory.startswith("lever_cw1_")]
    assert len(builds) == 3, f"expected 3 CW1 factories, saw {[b.factory for b in builds]}"
    for build in builds:
        assert build.trainer_declared is True
        assert build.trainer == TRAINER_RELPATH
        assert build.missing_flags == (), (
            f"{build.factory} emits flags the trainer argparse does not declare: {build.missing_flags}"
        )
        assert build.stub_marker is False


def test_every_emitted_flag_is_covered_by_a_constant_manifest_entry():
    levers = [
        lever_cw1_aligned_objective(),
        lever_cw1_lr_budget(2.0e-4),
        lever_cw1_observation_budget(),
    ]
    for lever in levers:
        assert set(lever.overrides) == set(lever.constant_manifest), (
            f"{lever.name}: every emitted flag needs a provenance rung "
            f"(emitted {sorted(lever.overrides)}, custodied {sorted(lever.constant_manifest)})"
        )
        for entry in lever.constant_manifest.values():
            assert entry.get("vehicle"), "every constant must carry its vehicle label"
