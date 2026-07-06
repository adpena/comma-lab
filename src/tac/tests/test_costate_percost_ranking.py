"""Tests for the ΔS-PER-COST ranking (task #247) — the costate controller's missing
per-cost argmax.

The shadow controller previously ranked recommendations by predicted ΔS alone
(``ranked.sort(key=lambda c: c["predicted_dS"])``). Task #247 adds the per-cost
divisor: rank by ``predicted_dS / max(cost, COST_EPSILON)`` so the cheapest-biggest
-drop wins. These tests cover: the pure cost + per-cost helpers, the divisor
reordering candidates vs pure-ΔS, the never-regress refusal being preserved (raw ΔS
> 0 still refused; the divisor never rescues it), the default-cost path, the epsilon
guard, and the interpretable ``cost`` + ``predicted_dS_per_cost`` fields on every row.

Everything here is $0 apparatus / advisory shadow-mode; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.witness_control.shadow_controller import (
    COST_EPSILON,
    DEFAULT_ACTION_COST,
    build_shadow_report,
    candidate_cost,
    load_run_inputs,
    per_cost_score,
)

# Real #205 CE-descent + tau-creep row snippets (verbatim values; same as the sister
# costate test module — the CE stage is CONVERGING (CONTINUE_STAGE, a HEAVY move),
# CE+TAU together is DIVERGING (ROLLBACK/STOP, LIGHT moves)).
CE_ROWS = [
    {"stage": "verdict", "epoch": 200, "seg_form": "ce", "d_seg": 0.005088,
     "d_pose": 0.003081, "blob_bytes": 100526, "implied_S": 0.7513, "ep_loss": 521.239},
    {"stage": "verdict", "epoch": 225, "seg_form": "ce", "d_seg": 0.004964,
     "d_pose": 0.002489, "blob_bytes": 100233, "implied_S": 0.7209, "ep_loss": 517.161},
    {"stage": "verdict", "epoch": 250, "seg_form": "ce", "d_seg": 0.004835,
     "d_pose": 0.002761, "blob_bytes": 99760, "implied_S": 0.7161, "ep_loss": 514.665},
    {"stage": "verdict", "epoch": 275, "seg_form": "ce", "d_seg": 0.004763,
     "d_pose": 0.002447, "blob_bytes": 99709, "implied_S": 0.6991, "ep_loss": 513.461},
]
TAU_ROWS = [
    {"stage": "verdict", "epoch": 300, "seg_form": "tau_softplus", "d_seg": 0.004752,
     "d_pose": 0.002099, "blob_bytes": 99550, "implied_S": 0.6864, "ep_loss": 148.553},
    {"stage": "verdict", "epoch": 325, "seg_form": "tau_softplus", "d_seg": 0.005923,
     "d_pose": 0.003826, "blob_bytes": 99580, "implied_S": 0.8542, "ep_loss": 147.512},
    {"stage": "verdict", "epoch": 350, "seg_form": "tau_softplus", "d_seg": 0.006267,
     "d_pose": 0.002529, "blob_bytes": 99489, "implied_S": 0.8519, "ep_loss": 141.335},
    {"stage": "verdict", "epoch": 375, "seg_form": "tau_softplus", "d_seg": 0.006424,
     "d_pose": 0.002411, "blob_bytes": 99365, "implied_S": 0.8638, "ep_loss": 137.598},
    {"stage": "verdict", "epoch": 400, "seg_form": "tau_softplus", "d_seg": 0.006568,
     "d_pose": 0.002277, "blob_bytes": 98995, "implied_S": 0.8736, "ep_loss": 134.122},
]


def _make_run_dir(tmp_path: Path, rows: list[dict], flags_line: str = "--mod-dim 19") -> Path:
    d = tmp_path / "run"
    d.mkdir()
    (d / "run.log").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "launch.sh").write_text(f".venv/bin/python trainer.py {flags_line}\n")
    return d


# ─────────────────────── pure cost model ───────────────────────
def test_candidate_cost_heavy_multi_epoch_is_horizon():
    c = candidate_cost({"action": "CONTINUE_STAGE", "predicted_dS": -0.1,
                        "horizon_epochs": 25})
    assert c == 25.0
    # a different horizon flows through
    assert candidate_cost({"action": "CONTINUE_STAGE", "horizon_epochs": 40}) == 40.0


def test_candidate_cost_light_action_is_unit():
    for act in ("ROLLBACK_TO_BEST_CHECKPOINT", "STOP_OR_RETREAT_STAGE",
                "WATCH_NO_ACTION", "INVESTIGATE_BINDING_TERM_DEADLOCK",
                "WIDEN_WINDOW_OR_CHECK_COLLISION"):
        assert candidate_cost({"action": act, "predicted_dS": -0.02}) == 1.0


def test_candidate_cost_default_when_unknown_or_absent():
    # absent action -> DEFAULT_ACTION_COST
    assert candidate_cost({"predicted_dS": -0.01}) == DEFAULT_ACTION_COST
    # unknown (non-heavy) action -> light unit baseline (also 1.0)
    assert candidate_cost({"action": "SOME_NOVEL_ACTION", "predicted_dS": -0.01}) == 1.0


def test_candidate_cost_explicit_positive_field_wins():
    # an explicit positive cost overrides the per-action model
    assert candidate_cost({"action": "CONTINUE_STAGE", "cost": 3.0,
                           "horizon_epochs": 25}) == 3.0
    assert candidate_cost({"action": "ROLLBACK_TO_BEST_CHECKPOINT", "cost": 7.5}) == 7.5


def test_candidate_cost_non_positive_explicit_field_ignored():
    # 0 / negative explicit cost is ignored -> falls back to the model (never ≤ 0)
    assert candidate_cost({"action": "CONTINUE_STAGE", "cost": 0.0,
                           "horizon_epochs": 25}) == 25.0
    assert candidate_cost({"action": "CONTINUE_STAGE", "cost": -5.0,
                           "horizon_epochs": 25}) == 25.0


def test_candidate_cost_always_strictly_positive():
    for cand in ({}, {"action": "CONTINUE_STAGE"}, {"action": "WATCH_NO_ACTION"},
                 {"action": "X", "cost": -1}, {"cost": 0.0}, {"action": None}):
        assert candidate_cost(cand) > 0.0


# ─────────────────────── the per-cost divisor ───────────────────────
def test_per_cost_divisor_reorders_vs_pure_ds():
    """The core of #247: a cheaper SMALLER-drop out-ranks a costlier BIGGER-drop."""
    heavy = {"action": "CONTINUE_STAGE", "predicted_dS": -0.10, "horizon_epochs": 25}
    light = {"action": "ROLLBACK_TO_BEST_CHECKPOINT", "predicted_dS": -0.02}
    ch, cl = candidate_cost(heavy), candidate_cost(light)
    assert ch == 25.0 and cl == 1.0
    # pure-ΔS ranking: the HEAVY move wins (its raw drop is more negative)
    assert heavy["predicted_dS"] < light["predicted_dS"]
    # per-cost ranking: the LIGHT move wins (-0.02/1 = -0.02 < -0.10/25 = -0.004)
    heavy_pc = per_cost_score(heavy["predicted_dS"], ch)
    light_pc = per_cost_score(light["predicted_dS"], cl)
    assert light_pc < heavy_pc
    assert light_pc == pytest.approx(-0.02)
    assert heavy_pc == pytest.approx(-0.004)


def test_per_cost_epsilon_guard_no_div_zero():
    val = per_cost_score(-1.0, 0.0)
    assert math.isfinite(val)
    assert val == pytest.approx(-1.0 / COST_EPSILON)
    # sign-preserving: a positive ΔS stays positive under the guard (refusal intact)
    assert per_cost_score(1.0, 0.0) > 0.0
    assert per_cost_score(0.0, 0.0) == 0.0


def test_per_cost_score_sign_preserving():
    # cost is strictly positive -> the divisor never flips a candidate's sign
    assert per_cost_score(-0.5, 25.0) < 0.0
    assert per_cost_score(0.5, 25.0) > 0.0
    assert per_cost_score(0.0, 25.0) == 0.0


# ─────────────────────── integration through the shadow controller ───────────────────────
def test_recommendations_expose_cost_and_per_cost(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS)   # CONVERGING -> CONTINUE_STAGE
    rep = build_shadow_report(load_run_inputs(d))
    assert rep.recommendations
    top = rep.recommendations[0]
    assert "cost" in top and "predicted_dS_per_cost" in top
    assert top["cost"] > 0.0
    assert top["predicted_dS_per_cost"] == pytest.approx(top["predicted_dS"] / top["cost"])


def test_continue_stage_cost_is_its_horizon(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS)
    rep = build_shadow_report(load_run_inputs(d))
    cs = next(r for r in rep.recommendations if r["action"] == "CONTINUE_STAGE")
    # the HEAVY multi-epoch move costs its horizon (not the unit baseline)
    assert cs["cost"] == float(cs["horizon_epochs"])
    assert cs["cost"] > 1.0


def test_light_rollback_cost_is_unit(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS + TAU_ROWS)   # DIVERGING -> ROLLBACK (light)
    rep = build_shadow_report(load_run_inputs(d))
    rb = next(r for r in rep.recommendations if r["action"] == "ROLLBACK_TO_BEST_CHECKPOINT")
    assert rb["cost"] == 1.0


def test_ranked_is_sorted_by_per_cost(tmp_path):
    d = _make_run_dir(tmp_path, CE_ROWS + TAU_ROWS)
    rep = build_shadow_report(load_run_inputs(d))
    keys = [r["predicted_dS_per_cost"] for r in rep.recommendations]
    assert keys == sorted(keys)   # most-negative (best bang/buck) first
    # and consistent with the exposed cost + raw ΔS on every row
    for r in rep.recommendations:
        assert r["predicted_dS_per_cost"] == pytest.approx(
            r["predicted_dS"] / max(r["cost"], COST_EPSILON))


def test_never_regress_preserved_under_per_cost(tmp_path):
    # d_seg gently converging BUT d_pose rising hard -> chained dS/dep central > 0.
    # CONTINUE_STAGE must be REFUSED (raw ΔS > 0); the per-cost divisor (cost 25)
    # keeps it positive and cannot rescue it into the ranked list.
    rows = [
        {"stage": "verdict", "epoch": 0, "seg_form": "ce", "d_seg": 0.010000,
         "d_pose": 0.001, "blob_bytes": 100000, "ep_loss": 100.0},
        {"stage": "verdict", "epoch": 25, "seg_form": "ce", "d_seg": 0.009900,
         "d_pose": 0.010, "blob_bytes": 100000, "ep_loss": 99.0},
        {"stage": "verdict", "epoch": 50, "seg_form": "ce", "d_seg": 0.009800,
         "d_pose": 0.030, "blob_bytes": 100000, "ep_loss": 98.0},
        {"stage": "verdict", "epoch": 75, "seg_form": "ce", "d_seg": 0.009700,
         "d_pose": 0.060, "blob_bytes": 100000, "ep_loss": 97.0},
    ]
    d = _make_run_dir(tmp_path, rows)
    rep = build_shadow_report(load_run_inputs(d))
    assert not any(r["action"] == "CONTINUE_STAGE" for r in rep.recommendations)
    ref = next(r for r in rep.refused if r["action"] == "CONTINUE_STAGE")
    assert "NEVER_REGRESS" in ref["refusal_reason"]
    # refused rows ALSO carry the interpretable cost + per-cost (Rudin readback),
    # and both stay positive (the divisor never flipped the sign).
    assert ref["cost"] > 0.0
    assert ref["predicted_dS_per_cost"] > 0.0
    assert ref["predicted_dS_per_cost"] == pytest.approx(
        ref["predicted_dS"] / max(ref["cost"], COST_EPSILON))
