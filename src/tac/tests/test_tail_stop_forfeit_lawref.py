# SPDX-License-Identifier: MIT
"""Tests for the TAIL PowerPlay attribution-floor reconcile (SEAL-v7-r1 MAJOR-1/2).

FIX-1: ``stop_marginal_s`` is the DERIVED s* = ν·forfeit ≈ 6.897e-6 (forfeit_matched_exit_v1),
not the struck HARDCODED 1e-4 — as a LawRef, a DERIVED provenance row, the dataclass default,
and a round-trippable canonical equation.
FIX-2: the turnpike operating point (τ_0 = τ_end) emits a one-time dead-knob note; the
descending-ladder path is unchanged.
"""
from __future__ import annotations

import math

from tac.canonical_equations.evaluators import eval_forfeit_matched_exit_s_star
from tac.canonical_equations.registry import (
    load_registry_events_lenient,
    register_canonical_equation,
)
from tac.canonical_equations.tail_stop_forfeit_floor_20260708 import (
    FORFEIT_MATCHED_EXIT_EQUATION_ID,
    FORFEIT_S,
    NU_MUON_FIN,
    NU_TAU_SOFTPLUS,
    S_STAR_DERIVED,
    build_forfeit_matched_exit_v1,
    forfeit_matched_exit_s_star,
)
from tac.witness_control.tail_cycles import (
    STOP_MARGINAL_S_DERIVED,
    TAIL_CONSTANT_PROVENANCE,
    TailCycleConfig,
    TailController,
    next_tau,
    stop_marginal_s_lawref,
)
from tac.witness_dsl.lawref import LADDER_DERIVED_AT_CONFIG, resolve

_EXPECT = 6.897e-6  # the seal's target floor, to 4 sig figs


# ── FIX-1: the LawRef + provenance + default + registry ──────────────────────────────────────
def test_stop_marginal_s_lawref_resolves_6_897e_6() -> None:
    rc = resolve(stop_marginal_s_lawref())
    assert round(rc.value, 9) == _EXPECT
    assert math.isclose(rc.value, NU_TAU_SOFTPLUS * FORFEIT_S, rel_tol=0.0, abs_tol=0.0)
    assert rc.value == S_STAR_DERIVED == STOP_MARGINAL_S_DERIVED


def test_lawref_is_derived_at_config_with_no_fallback() -> None:
    rc = resolve(stop_marginal_s_lawref())
    assert rc.ladder_class == LADDER_DERIVED_AT_CONFIG
    assert rc.fallback_used is False  # pure literal inputs → no artifact I/O, no fallback


def test_provenance_row_flipped_to_derived() -> None:
    row = TAIL_CONSTANT_PROVENANCE["stop_marginal_s"]
    assert row["ladder_class"] == "derived_at_config"
    assert row["equation_id"] == FORFEIT_MATCHED_EXIT_EQUATION_ID
    assert row["value"] == STOP_MARGINAL_S_DERIVED
    assert round(row["value"], 9) == _EXPECT


def test_dataclass_default_is_derived_not_hardcoded_1e_4() -> None:
    default = TailCycleConfig(k_max=1).stop_marginal_s
    assert default == STOP_MARGINAL_S_DERIVED
    assert default != 1e-4
    # 14.5× finer than the struck hardcode (the seal's magnitude).
    assert math.isclose(1e-4 / default, 14.5, abs_tol=0.1)


def test_evaluator_matches_builder_callable() -> None:
    via_eval = eval_forfeit_matched_exit_s_star({"nu": NU_TAU_SOFTPLUS, "forfeit": FORFEIT_S})
    via_callable = forfeit_matched_exit_s_star(NU_TAU_SOFTPLUS, FORFEIT_S)
    assert via_eval == via_callable == S_STAR_DERIVED
    # the seal's ν(muon_fin) reformulation is a DIFFERENT (finer, run-2) number, not 6.897e-6.
    assert round(NU_MUON_FIN * FORFEIT_S, 9) != _EXPECT


def test_registry_round_trip(tmp_path) -> None:
    reg = tmp_path / "eq.jsonl"
    lock = tmp_path / "eq.jsonl.lock"
    eq = build_forfeit_matched_exit_v1()  # builds + validates
    register_canonical_equation(eq, path=reg, lock_path=lock, subagent_id="test")
    events = load_registry_events_lenient(path=reg)
    matching = [e for e in events if e.get("equation_id") == FORFEIT_MATCHED_EXIT_EQUATION_ID]
    assert matching, "registered equation not found on reload"
    payload = matching[-1]["equation_payload"]
    assert payload == eq.to_dict()
    assert payload["latex_form"] and payload["one_line_summary"]
    assert len(payload["empirical_anchors"]) == 2


# ── FIX-2: turnpike vs descending-ladder honesty ─────────────────────────────────────────────
def _turnpike_cfg() -> TailCycleConfig:
    return TailCycleConfig(k_max=2, tau_end=0.31, cycle_floor_epochs=5, dwell_min=0, min_points=4)


def test_turnpike_note_emitted_exactly_once(capsys) -> None:
    ctrl = TailController(_turnpike_cfg(), tau_ref=0.31, lr_ref=1e-3, tau0=0.31)  # τ_0 == τ_end
    assert ctrl.is_turnpike is True
    assert ctrl.turnpike_note()["dead_knobs"] == ["tau_halving", "lr_prop_coeff"]
    ctrl.step(0, [(0, 0.005)])
    ctrl.step(1, [(0, 0.005), (1, 0.0049)])
    out = capsys.readouterr().out
    assert out.count('"tail_turnpike_active"') == 1  # one-time only


def test_descending_ladder_no_turnpike_note_and_math_unchanged(capsys) -> None:
    cfg = TailCycleConfig(k_max=5, tau_end=0.31)
    # τ_0 > τ_end ⇒ genuine descending ladder; the halving math is UNCHANGED by the fix.
    seq, t = [], 1.24
    for _ in range(4):
        t = next_tau(t, cfg, None)
        seq.append(round(t, 4))
    assert seq == [0.62, 0.31, 0.31, 0.31]
    ctrl = TailController(cfg, tau_ref=1.24, lr_ref=1e-3, tau0=1.24)
    assert ctrl.is_turnpike is False
    assert ctrl.turnpike_note() is None
    ctrl.step(0, [(0, 0.005)])
    assert '"tail_turnpike_active"' not in capsys.readouterr().out


# ── FIX-1 behavior: the finer floor mines longer + composes with the rate-aware stop ─────────
def _drive(cfg, rows, brows=None, tau0=0.62):
    ctrl = TailController(cfg, tau_ref=0.62, lr_ref=1e-3, tau0=tau0)
    last = None
    for ep in range(0, max(e for e, _ in rows) + 1):
        vis = [(e, d) for e, d in rows if e <= ep]
        vb = [(e, b) for e, b in brows if e <= ep] if brows is not None else None
        last = ctrl.step(ep, vis, byte_rows=vb)
    return last


def test_rate_aware_stop_composes_with_finer_floor() -> None:
    cfg = TailCycleConfig(k_max=5, tau_end=0.31, cycle_floor_epochs=5, dwell_min=0, min_points=4)
    # d_seg marginal ≈ 5e-5: below the OLD 1e-4 (would have stopped) but ABOVE the finer
    # 6.897e-6 floor ⇒ the tail MINES LONGER (does not stop) — the seal's point.
    assert _drive(cfg, [(0, 0.005), (4, 0.004998)]).stop is False
    # d_seg marginal ≈ 5e-6 < 6.897e-6 ⇒ stop.
    assert _drive(cfg, [(0, 0.005), (4, 0.0049998)]).stop is True
    # rate-aware: d_seg marginal ≈ 1e-5 (> floor, would NOT stop d_seg-only) but a byte-INFLATING
    # cycle drags the NET-ΔS below the finer floor ⇒ stop (S1-R1 rate leg composes with the floor).
    assert _drive(cfg, [(0, 0.005), (4, 0.0049996)]).stop is False  # d_seg-only
    assert _drive(cfg, [(0, 0.005), (4, 0.0049996)], brows=[(0, 0.0), (4, 20000.0)]).stop is True
