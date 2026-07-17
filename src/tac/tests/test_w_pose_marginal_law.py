# SPDX-License-Identifier: MIT
"""w_pose(t) = 5/sqrt(10*d_pose(t)) derived-weight law (SPEC_v10 §13.3; arm B 2026-07-17).

Behavior tests for the equation module + the derived clamp + the DSL lever/LawRef custody + the
trainer wiring surfaces (never-invent-flags).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tac.canonical_equations.w_pose_marginal_weight_law_20260717 import (
    D_POSE_CROSSOVER,
    SEG_MARGINAL,
    clamp_from_crossover,
    verify_crossover_identity,
    w_pose_law,
    w_pose_marginal,
)

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


# ── the law itself ─────────────────────────────────────────────────────────── #
def test_marginal_is_exact_score_derivative():
    """Finite-difference check against S_pose(d) = sqrt(10*d) at several operating points."""
    for d in (1e-4, 1.6e-3, 0.01, 0.5, 112.0):   # incl. the live-run d_pose 112 + R1 1.6e-3
        h = d * 1e-6
        fd = (math.sqrt(10 * (d + h)) - math.sqrt(10 * (d - h))) / (2 * h)
        assert w_pose_marginal(d) == pytest.approx(fd, rel=1e-5)


def test_crossover_identity_and_value():
    assert verify_crossover_identity()
    assert D_POSE_CROSSOVER == pytest.approx(2.5e-4)
    assert w_pose_marginal(D_POSE_CROSSOVER) == pytest.approx(SEG_MARGINAL)


def test_clamp_binds_only_below_crossover():
    assert w_pose_law(D_POSE_CROSSOVER * 4.0) == pytest.approx(SEG_MARGINAL / 2.0)  # unclamped
    assert w_pose_law(D_POSE_CROSSOVER / 100.0) == SEG_MARGINAL                      # clamped
    assert w_pose_law(3.4e-5) == SEG_MARGINAL   # the ancestor operating point clamps at 100


def test_law_monotone_decreasing_in_d_pose():
    vals = [w_pose_law(d) for d in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 112.0)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    # at the live-run start d_pose 112 the marginal is WEAK (~0.149) — the correct physics.
    assert w_pose_law(112.0) == pytest.approx(5.0 / math.sqrt(1120.0))


def test_marginal_rejects_nonpositive_and_nonfinite():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            w_pose_marginal(bad)


def test_clamp_from_crossover_validation():
    assert clamp_from_crossover() == 100.0
    assert clamp_from_crossover(50.0) == 50.0
    with pytest.raises(ValueError):
        clamp_from_crossover(0.0)


def test_w_pose_law_custom_clamp_validation():
    assert w_pose_law(1e-6, clamp=7.0) == 7.0
    with pytest.raises(ValueError):
        w_pose_law(1e-3, clamp=-1.0)


# ── DSL lever + LawRef custody ─────────────────────────────────────────────── #
def test_dsl_lever_emits_flags_and_lawref():
    from tac.witness_dsl.curriculum_dsl import PoseMarginalWeightLaw

    lever = PoseMarginalWeightLaw()
    assert lever.overrides == {"--w-pose-marginal-law": True, "--w-pose-marginal-clamp": 100.0}
    ref = lever.constant_refs["--w-pose-marginal-clamp"]
    assert ref.equation_id == "w_pose_marginal_weight_law_v1"


def test_dsl_lever_custom_clamp_and_validation():
    from tac.witness_dsl.curriculum_dsl import PoseMarginalWeightLaw

    assert PoseMarginalWeightLaw(clamp=42.0).overrides["--w-pose-marginal-clamp"] == 42.0
    with pytest.raises(ValueError, match="clamp must be"):
        PoseMarginalWeightLaw(clamp=0.0)


# ── trainer wiring surfaces ────────────────────────────────────────────────── #
def test_trainer_declares_law_flags_default_off():
    src = _TRAINER.read_text(errors="ignore")
    m = re.search(r"add_argument\(\s*\"--w-pose-marginal-law\".*?default=(\w+)", src, re.S)
    assert m is not None and m.group(1) == "False"
    m2 = re.search(r"add_argument\(\s*\"--w-pose-marginal-clamp\".*?default=([0-9.]+)", src, re.S)
    assert m2 is not None and float(m2.group(1)) == 100.0


def test_trainer_consumes_law_at_pose_finish_and_fail_louds_inert_arm():
    src = _TRAINER.read_text(errors="ignore")
    # consumption: the engage block computes min(clamp, 5/sqrt(10*d_pose)) into _w_pose_now
    assert "5.0 / math.sqrt(10.0 * float(_w_pose_law_state[\"last_d_pose\"]))" in src
    # sense: the verdict path records the latest measured d_pose
    assert '_w_pose_law_state["last_d_pose"] = float(v["d_pose"])' in src
    # inert-arm NO-FAKE guard
    assert "--w-pose-marginal-law requires --pose-finish-start-epoch > 0" in src
    # telemetry row for observability (verdict-cadence piecewise-constant updates)
    assert '"stage": "w_pose_marginal_law"' in src
