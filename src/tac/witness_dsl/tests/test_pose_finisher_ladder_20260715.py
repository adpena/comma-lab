# SPDX-License-Identifier: MIT
"""Pose-finisher ladder levers (#248/#366 finisher-phase prep, 2026-07-15) — pure-argv tests.

LOAD-BEARING contract (no MLX; Lever is a frozen dataclass):
1. ``PoseFinisherFilmReadbackArm`` emits EXACTLY the one finisher-window delta
   ``--pose-carrier-residual-mode film`` (a REAL trainer choice — never-invent-flags), so the
   composed argv over a pose-carrier-active base is the incumbent config +- ONE flag value.
2. ``PoseFinisherLiveGap`` delegates to the single ``--verdict-live-gap-every`` emitter
   (``VerdictLiveGap``) at the DERIVED cadence — no duplicate flag home, no bare constant.
3. ``pose_finisher_live_gap_cadence`` is DERIVED from (ema_decay, num_pairs, accum_pairs) via
   ``tac.confound_observability.ema_warmup_updates`` and fail-closes on malformed inputs.
4. Both factories are zero-required-arg single-Lever => DISCOVERED composable by the lever
   registry (``--dsl-lever`` resolvable) — the config-orphan discipline, structurally checked.

Adversarial note (would these pass if broken?): test 1 fails if the arm grows extra overrides
(silent config drift) or the mode value drifts from the trainer's table|film choices; test 4
fails if either factory gains a required arg or returns a composite (both would crash
``--dsl-lever`` at launch). Advisory; the pointer moves only via upstream/evaluate.py.
"""
from __future__ import annotations

import math

import pytest

from tac.witness_dsl.curriculum_dsl import (
    PoseFinisherFilmReadbackArm,
    PoseFinisherLiveGap,
    StoreNothingPoseCarrier,
    VerdictLiveGap,
    pose_finisher_live_gap_cadence,
)


# ── 1. the film read-back arm: exactly ONE flag delta ─────────────────────────────────────────
def test_film_arm_emits_exactly_one_residual_mode_flip():
    lev = PoseFinisherFilmReadbackArm()
    assert lev.name == "pose_finisher_film_readback_arm"
    assert lev.overrides == {"--pose-carrier-residual-mode": "film"}
    assert lev.epochs_delta == 0  # rides the existing finisher span; no extra epoch budget


def test_film_arm_value_is_a_real_trainer_choice():
    # The trainer argparse choices are exactly table|film — the DSL's own carrier factory
    # fail-closes on anything else; the arm's value must be inside that set.
    mode = PoseFinisherFilmReadbackArm().overrides["--pose-carrier-residual-mode"]
    assert StoreNothingPoseCarrier(residual_mode=mode).overrides[
        "--pose-carrier-residual-mode"] == "film"


def test_film_arm_composes_over_table_base_by_merge_semantics():
    """Lever composition = later lever wins on conflict (curriculum_dsl.Lever docstring)."""
    base = StoreNothingPoseCarrier()  # R1-proven table
    arm = PoseFinisherFilmReadbackArm()
    merged = {**base.overrides, **arm.overrides}
    assert merged["--pose-carrier-residual-mode"] == "film"
    # the rest of the carrier block is UNTOUCHED by the arm:
    assert merged["--pose-carrier"] is True
    assert merged["--pose-carrier-source"] == "generated"
    assert merged["--w-pose"] == 1.0


# ── 2./3. the derived live-gap cadence ─────────────────────────────────────────────────────────
def test_cadence_defaults_derive_to_four():
    # ceil(2/(1-0.997)) = 667 updates; steps/epoch = ceil(600/8) = 75; warmup_epochs = 9; 9//2 = 4.
    assert pose_finisher_live_gap_cadence() == 4


def test_cadence_matches_the_written_derivation_formula():
    from tac.confound_observability import ema_warmup_updates
    for decay, pairs, accum in ((0.997, 600, 8), (0.99, 600, 8), (0.997, 24, 8), (0.9995, 600, 1)):
        steps = math.ceil(pairs / accum)
        expect = max(1, math.ceil(ema_warmup_updates(decay) / steps) // 2)
        assert pose_finisher_live_gap_cadence(decay, pairs, accum) == expect


def test_cadence_never_below_one():
    # tiny warmup vs huge epochs => still a valid every-K cadence (K >= 1).
    assert pose_finisher_live_gap_cadence(ema_decay=0.0, num_pairs=600, accum_pairs=1) >= 1


def test_cadence_fail_closed_on_malformed_inputs():
    with pytest.raises(ValueError):
        pose_finisher_live_gap_cadence(num_pairs=0)
    with pytest.raises(ValueError):
        pose_finisher_live_gap_cadence(accum_pairs=0)
    with pytest.raises(ValueError):
        pose_finisher_live_gap_cadence(ema_decay=1.0)  # ema_warmup_updates refuses beta >= 1


def test_live_gap_lever_delegates_to_the_single_emitter():
    lev = PoseFinisherLiveGap()
    # Same lever name as VerdictLiveGap => ONE flag home (no parallel emitter to drift).
    assert lev.name == VerdictLiveGap().name == "verdict_live_gap"
    assert lev.overrides == {"--verdict-live-gap-every": pose_finisher_live_gap_cadence()}
    assert lev.overrides["--verdict-live-gap-every"] > 0  # explicit all-run mode, never 0/-1


# ── 4. composability: discovered by the registry, resolvable via --dsl-lever ──────────────────
def test_both_factories_are_registry_composable():
    from tac.witness_dsl.lever_registry import name_composable_levers, resolve_composable_lever
    names = name_composable_levers()
    assert "PoseFinisherFilmReadbackArm" in names
    assert "PoseFinisherLiveGap" in names
    assert resolve_composable_lever("PoseFinisherFilmReadbackArm").overrides == {
        "--pose-carrier-residual-mode": "film"}
    assert resolve_composable_lever("PoseFinisherLiveGap").overrides == {
        "--verdict-live-gap-every": 4}
