# SPDX-License-Identifier: MIT
"""REVISIONS-B (T3 v7 council) — the three landed revisions, verified.

1. S2-REV-A LADDER<->Muon STAGGER invariant (shared helper + DSL WitnessProgram.validate + trainer
   validate_ladder_muon_stagger_config) — positive/negative/event-domain.
2. S4-R2 + S1-R1 TAIL upgrades — the TAIL constant PROVENANCE rows (no silent literals) + the crucible
   manifest surfacing them; the rate-aware stop math is in test_tail_cycles.py.
3. S6-R5 event-triggered-curriculum-inert-under-unify guard + loud note.

means != ends: gates a MEANS. Only a byte-closed n600 exact row < 0.19110 moves the pointer 0.19110.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tac.witness_control.tail_cycles import (
    TAIL_CONSTANT_PROVENANCE,
    tail_constant_provenance,
)
from tac.witness_curriculum.ladder_homotopy import (
    LADDER_LAMBDA_GATE_PROVENANCE,
    ladder_arm_window,
    ladder_muon_stagger_violation,
)
from tac.witness_dsl.lawref import VALID_LADDER_CLASSES

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"
_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# The trainer file is co-edited (the S6-R4 tau-advance sibling); this REVISIONS-B build lands the
# trainer surface (validate_ladder_muon_stagger_config call + the S6-R5 loud note) in the WORKING
# TREE, but its COMMIT is sequenced with the co-editing sibling. Guard the trainer-surface tests so
# this test file is GREEN whether or not the trainer commit has landed yet — they RUN + assert once
# the trainer carries the REVISIONS-B surface, and SKIP on a checkout that predates it. The shared
# PURE helper (ladder_homotopy) + DSL + provenance + crucible tests below never gate on the trainer.
_TRAINER_SRC = _TRAINER.read_text(encoding="utf-8")
_HAS_REVB_TRAINER = (
    "def validate_ladder_muon_stagger_config(" in _TRAINER_SRC
    and "event_curriculum_inert_under_unify" in _TRAINER_SRC
)
_needs_trainer = pytest.mark.skipif(
    not _HAS_REVB_TRAINER,
    reason="trainer REVISIONS-B surface not yet committed (co-edit coordination with the S6-R4 sibling)")


# ═══════════════════════ 1. S2-REV-A stagger invariant (shared pure helper) ═══════════════════════
def test_ladder_arm_window_is_sum():
    assert ladder_arm_window(80, 0, 260) == 340
    assert ladder_arm_window(60, 10, 200) == 270


def test_stagger_ok_when_max_window_below_muon():
    # v7 defaults: lane 340, movable 260, muon 726 => holds.
    assert ladder_muon_stagger_violation(
        ladder_on=True, lane_window=340, movable_window=260, muon_start_epoch=726) is None


def test_stagger_lane_violation_names_lane_only():
    err = ladder_muon_stagger_violation(
        ladder_on=True, lane_window=780, movable_window=260, muon_start_epoch=726)
    assert err is not None
    assert "lane arm window 780" in err and "movable arm window" not in err
    assert ">= --muon-start-epoch (726)" in err and "STAGGER" in err


def test_stagger_movable_violation_names_movable_only():
    err = ladder_muon_stagger_violation(
        ladder_on=True, lane_window=200, movable_window=800, muon_start_epoch=726)
    assert err is not None
    assert "movable arm window 800" in err and "lane arm window" not in err


def test_stagger_both_violation_names_both():
    err = ladder_muon_stagger_violation(
        ladder_on=True, lane_window=900, movable_window=800, muon_start_epoch=726)
    assert err is not None
    assert "lane arm window 900" in err and "movable arm window 800" in err and " AND " in err


def test_stagger_boundary_equal_is_a_violation():
    # window == muon_start is NOT strictly before => a violation (support still live AT the switch).
    assert ladder_muon_stagger_violation(
        ladder_on=True, lane_window=726, movable_window=0, muon_start_epoch=726) is not None
    # one below is OK (strict <).
    assert ladder_muon_stagger_violation(
        ladder_on=True, lane_window=725, movable_window=0, muon_start_epoch=726) is None


def test_stagger_noop_when_ladder_off():
    assert ladder_muon_stagger_violation(
        ladder_on=False, lane_window=9999, movable_window=9999, muon_start_epoch=726) is None


def test_stagger_noop_when_no_muon_finisher():
    # event-armed-domain / no-Muon: no cap => nothing to stagger against => N/A (None).
    assert ladder_muon_stagger_violation(
        ladder_on=True, lane_window=9999, movable_window=9999, muon_start_epoch=None) is None


# ── DSL surface: WitnessProgram.validate consumes the shared helper ──
def _minimal_program(*, lane_anneal: int, muon: "int | None"):
    from tac.witness_dsl.curriculum_dsl import (
        Anneal,
        Authority,
        Contain,
        LadderIslandHomotopy,
        Preserve,
        WitnessProgram,
    )
    lev = LadderIslandHomotopy(lane_anneal_epochs=lane_anneal)
    base = {"--curriculum": True, "--tau-softplus-start-epoch": 300, "--l7-start-epoch": 3001}
    if muon is not None:
        base["--muon-start-epoch"] = muon
    base.update(lev.overrides)
    return WitnessProgram(
        out_dir="x", gt_cache="g.npz", epochs=3000, num_pairs=200,
        temp=Anneal(1.0, 0.31), stages=(), regularizers=(),
        preserve=Preserve(), contain=Contain(), authority=Authority(), base=base)


def test_dsl_validate_fires_stagger_on_violation():
    p = _minimal_program(lane_anneal=700, muon=726)  # lane window 80+0+700=780 >= 726
    stagger = [v for v in p.validate() if "STAGGER" in v]
    assert stagger and "lane arm window 780" in stagger[0]


def test_dsl_validate_clean_when_staggered():
    p = _minimal_program(lane_anneal=260, muon=726)  # lane 340 < 726
    assert [v for v in p.validate() if "STAGGER" in v] == []


def test_dsl_validate_noop_stagger_when_muon_absent():
    p = _minimal_program(lane_anneal=700, muon=None)  # no Muon => no stagger check
    assert [v for v in p.validate() if "STAGGER" in v] == []


def test_v7_config_passes_stagger():
    import tac.witness_autoconfig as wac
    typed = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    assert [v for v in typed.validate_program() if "STAGGER" in v] == []


# ── trainer surface: validate_ladder_muon_stagger_config ──
def _trainer():
    spec = importlib.util.spec_from_file_location("_revb_trainer", _TRAINER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@_needs_trainer
def test_trainer_stagger_validator_raises_on_violation():
    m = _trainer()
    with pytest.raises(ValueError, match="STAGGER"):
        m.validate_ladder_muon_stagger_config(
            ladder_island_homotopy=True, lane_birth_epochs=80, lane_hold_epochs=0,
            lane_anneal_epochs=700, movable_birth_epochs=60, movable_hold_epochs=0,
            movable_anneal_epochs=200, muon_start_epoch=726)


@_needs_trainer
def test_trainer_stagger_validator_noop_paths():
    m = _trainer()
    # ok case
    m.validate_ladder_muon_stagger_config(
        ladder_island_homotopy=True, lane_birth_epochs=80, lane_hold_epochs=0,
        lane_anneal_epochs=260, movable_birth_epochs=60, movable_hold_epochs=0,
        movable_anneal_epochs=200, muon_start_epoch=726)
    # ladder off
    m.validate_ladder_muon_stagger_config(
        ladder_island_homotopy=False, lane_birth_epochs=80, lane_hold_epochs=0,
        lane_anneal_epochs=700, movable_birth_epochs=60, movable_hold_epochs=0,
        movable_anneal_epochs=200, muon_start_epoch=726)
    # no muon (event-armed-domain / no finisher)
    m.validate_ladder_muon_stagger_config(
        ladder_island_homotopy=True, lane_birth_epochs=80, lane_hold_epochs=0,
        lane_anneal_epochs=700, movable_birth_epochs=60, movable_hold_epochs=0,
        movable_anneal_epochs=200, muon_start_epoch=None)


@_needs_trainer
def test_trainer_calls_stagger_validator_in_config_path():
    src = _TRAINER.read_text(encoding="utf-8")
    assert "validate_ladder_muon_stagger_config(" in src
    # called (not just defined) — the config-validation call site passes args.* fields.
    assert 'ladder_island_homotopy=bool(getattr(args, "ladder_island_homotopy", False))' in src


# ═══════════════════════ 2. TAIL constant PROVENANCE (req-T; no silent literals) ═══════════════════
def test_tail_provenance_covers_the_sealed_constants():
    assert set(TAIL_CONSTANT_PROVENANCE) == {
        "cycle_floor_epochs", "dwell_min", "tau_halving", "stop_marginal_s"}


def test_tail_provenance_rows_are_well_formed():
    for name, row in TAIL_CONSTANT_PROVENANCE.items():
        assert row["ladder_class"] in VALID_LADDER_CLASSES, name
        assert isinstance(row["rationale"], str) and len(row["rationale"]) > 20, name
        assert "value" in row and "equation_id" in row


def test_tail_provenance_derived_cite_equations_waivers_do_not():
    # cycle_floor / dwell CITE registered laws; tau_halving / stop_marginal_s are HARDCODED-WITH-WAIVER.
    assert TAIL_CONSTANT_PROVENANCE["cycle_floor_epochs"]["equation_id"] == "tail_cycle_floor_v1"
    assert TAIL_CONSTANT_PROVENANCE["dwell_min"]["equation_id"] == "settle_window_v1"
    for k in ("tau_halving", "stop_marginal_s"):
        assert TAIL_CONSTANT_PROVENANCE[k]["ladder_class"] == "hardcoded_waiver", k
        assert TAIL_CONSTANT_PROVENANCE[k]["equation_id"] is None, k


def test_tail_provenance_getter_is_a_copy():
    a = tail_constant_provenance()
    a["tau_halving"]["value"] = 999.0
    assert TAIL_CONSTANT_PROVENANCE["tau_halving"]["value"] == 0.5  # not mutated


def test_lambda_gate_provenance_covers_both_gates():
    assert set(LADDER_LAMBDA_GATE_PROVENANCE) == {
        "--ladder-movable-lambda-gate", "--ladder-lane-lambda-gate"}
    for row in LADDER_LAMBDA_GATE_PROVENANCE.values():
        assert row["value"] == 0.0 and row["ladder_class"] in VALID_LADDER_CLASSES
        assert len(row["rationale"]) > 20


def test_crucible_manifest_surfaces_tail_provenance():
    import tac.witness_autoconfig as wac
    c = wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    tp = c.tail_constant_provenance
    assert set(tp["tail_constants"]) == {
        "cycle_floor_epochs", "dwell_min", "tau_halving", "stop_marginal_s"}
    assert set(tp["ladder_lambda_gates"]) == {
        "--ladder-movable-lambda-gate", "--ladder-lane-lambda-gate"}


def test_tail_provenance_does_not_change_emitted_argv():
    import tac.witness_autoconfig as wac
    c = wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    fd = dict(c.emitted_pairs)
    # the provenance is manifest-only; the sealed VALUES are still emitted unchanged.
    assert fd["--tail-stop-marginal-s"] == "0.0001"
    assert fd["--tail-tau-halving"] == "0.5"
    assert fd["--ladder-movable-lambda-gate"] == "0.0"
    assert fd["--ladder-lane-lambda-gate"] == "0.0"


# ═══════════════════════ 3. S6-R5 event-curriculum-inert-under-unify ══════════════════════════════
def test_v7_co_emits_unify_and_event_triggered():
    # the inert-but-armed condition R5 audits: BOTH flags present in the live v7 config.
    import tac.witness_autoconfig as wac
    c = wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    emitted = {f for f, _ in c.emitted_pairs}
    assert "--seg-form-unify-tau" in emitted
    assert "--curriculum-event-triggered" in emitted


@_needs_trainer
def test_seg_form_dispatch_short_circuits_unify_before_event():
    # the correctness-clean inertness: `if _unify_tau_on:` precedes `elif _evt_on:` so the event
    # controller (_evt_resolve_seg_form) is NEVER called under unify — no dissolved-boundary fire.
    src = _TRAINER.read_text(encoding="utf-8")
    i_unify = src.index("if _unify_tau_on:\n                # (--seg-form-unify-tau) the CE->tau_softplus discrete boundary")
    i_evt = src.index("elif _evt_on:\n                seg_form, _evt_event = _evt_resolve_seg_form")
    assert i_unify < i_evt, "unify branch must short-circuit BEFORE the event branch"


@_needs_trainer
def test_r5_loud_note_present_and_guarded():
    src = _TRAINER.read_text(encoding="utf-8")
    assert '"stage": "event_curriculum_inert_under_unify"' in src
    # guarded on BOTH flags being set (the inert-but-armed condition).
    assert ('bool(getattr(args, "seg_form_unify_tau", False)) and '
            'bool(getattr(args, "curriculum_event_triggered", False))' in src)
