# SPDX-License-Identifier: MIT
"""StoreNothingPoseCarrier DSL Lever tests (R1 #245/#238 P0 joint-descent recipe).

LOAD-BEARING contract under test (pure-argv; no MLX needed — Lever is a frozen dataclass):
1. The factory emits EXACTLY R1's proven store-nothing pose-carrier flags: --pose-carrier True,
   --pose-carrier-source generated, --pose-carrier-residual-mode table, --w-pose (never-invent-flags
   — each override key is a REAL levelset-trainer argparse flag).
2. It is DISTINCT from carrier B (WarpRealLumaFrame0): source=generated (store-nothing, ~0 bytes) vs
   carrier B's stored-real-keyframe path; and it carries the --pose-carrier boolean B carrier omits.
3. residual_mode is fail-closed to the trainer's exact choices (table|film); malformed refused.

Adversarial note (would these pass if broken?): test 1 fails if any flag name/value drifts from R1's
launch.sh; the refusal test fails if the choices guard is dropped. Advisory; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import pytest

from tac.witness_dsl.curriculum_dsl import StoreNothingPoseCarrier, WarpRealLumaFrame0


def test_emits_r1_proven_store_nothing_flags():
    lev = StoreNothingPoseCarrier(w_pose=1.0)
    assert lev.name == "store_nothing_pose_carrier"
    ov = lev.overrides
    # EXACTLY R1's launch.sh pose-carrier block (verified real flags, trainer L10572-10585).
    assert ov["--pose-carrier"] is True
    assert ov["--pose-carrier-source"] == "generated"
    assert ov["--pose-carrier-residual-mode"] == "table"
    assert ov["--w-pose"] == 1.0


def test_w_pose_is_configurable_and_defaults_to_joint():
    assert StoreNothingPoseCarrier().overrides["--w-pose"] == 1.0  # joint by default
    assert StoreNothingPoseCarrier(w_pose=0.5).overrides["--w-pose"] == 0.5


def test_distinct_from_carrier_b_warp_real_luma():
    a = StoreNothingPoseCarrier()
    b = WarpRealLumaFrame0()
    assert a.name != b.name
    # carrier A carries the store-nothing source + the --pose-carrier boolean; carrier B does not.
    assert "--pose-carrier-source" in a.overrides
    assert "--pose-carrier-source" not in b.overrides
    assert a.overrides["--pose-carrier-source"] == "generated"


def test_residual_mode_film_allowed():
    assert StoreNothingPoseCarrier(residual_mode="film").overrides["--pose-carrier-residual-mode"] == "film"


def test_residual_mode_fail_closed():
    with pytest.raises(ValueError, match="residual_mode"):
        StoreNothingPoseCarrier(residual_mode="mlp")  # not a trainer choice


# ── v7.5 D.9 TERMINAL POSE-FINISH (the R1 two-phase; SPEC §D.9) ────────────────────────────────
def test_terminal_pose_finish_emits_gate_and_finish_weight():
    """v7.5 D.9: TerminalPoseFinish holds --pose-finish-start-epoch (the terminal gate) + --w-pose (the
    finish-phase weight) — both REAL trainer flags (never-invent-flags)."""
    from tac.witness_dsl.curriculum_dsl import TerminalPoseFinish
    lev = TerminalPoseFinish(start_epoch=726, w_pose=1.0)
    assert lev.name == "terminal_pose_finish"
    assert lev.overrides["--pose-finish-start-epoch"] == 726
    assert lev.overrides["--w-pose"] == 1.0


def test_terminal_pose_finish_rejects_zero_w_pose():
    """The pose carrier trains its dxi only on the realized d_pose term => the finish weight MUST be > 0
    (a w_pose=0 finish is a no-op; fail-closed)."""
    from tac.witness_dsl.curriculum_dsl import TerminalPoseFinish
    with pytest.raises(ValueError, match="w_pose must be > 0"):
        TerminalPoseFinish(start_epoch=726, w_pose=0.0)
    with pytest.raises(ValueError, match="start_epoch must be >= 0"):
        TerminalPoseFinish(start_epoch=-1)
