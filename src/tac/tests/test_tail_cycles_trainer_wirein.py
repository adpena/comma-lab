"""TAIL_k trainer wire-in + adoption-seam + DSL-mapping tests.

Proves: (1) the trainer flags default OFF/sealed (byte-identical when unset); (2) the
default-off construction + loop guards are structurally present; (3) the RESUME path tolerates
the NEW tail flags absent from a frozen checkpoint cfg (the run-1 adoption seam); (4) the DSL
TailCycles factory holds all tail flags (lever_registry maps them out of unmapped — triality).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("_tail_trainer_under_test", _TRAINER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- default-off byte-identity
def test_tail_flags_declare_off_or_sealed_defaults():
    src = _TRAINER.read_text(encoding="utf-8")
    # --tail-cycles-max default 0 == OFF == byte-identical
    assert 'ap.add_argument("--tail-cycles-max", type=int, default=0,' in src
    # sealed law defaults (req T: tail_cycle_floor_v1 387.09 / settle_window_v1 237)
    assert 'ap.add_argument("--tail-cycle-floor-epochs", type=float, default=387.09,' in src
    assert 'ap.add_argument("--tail-dwell-min", type=int, default=237,' in src
    assert 'ap.add_argument("--tail-tau-halving", type=float, default=0.5,' in src
    assert 'ap.add_argument("--tail-lr-prop-tau", type=float, default=1.0,' in src
    assert 'ap.add_argument("--tail-start-epoch", type=int, default=0,' in src
    # --tail-live-mq default False (fail-closed when set; fallback = τ-halving)
    assert '"--tail-live-mq", action=argparse.BooleanOptionalAction, default=False,' in src


def test_construction_and_loop_are_default_off_guarded():
    src = _TRAINER.read_text(encoding="utf-8")
    # controller is None by default; only built when --tail-cycles-max > 0
    assert "_tail_ctrl = None" in src
    assert 'if int(getattr(args, "tail_cycles_max", 0) or 0) > 0:' in src
    # every loop tail block is guarded by `_tail_ctrl is not None` => skipped when off (byte-identical)
    assert "if _tail_ctrl is not None and muon_switched and ep >= _tail_start_epoch:" in src
    assert "if _tail_stop_now:" in src
    # requires the Muon finisher; --tail-live-mq is fail-closed (owed SC-3 render build)
    assert "--tail-cycles-max requires --muon-start-epoch" in src
    assert "--tail-live-mq is the OWED SC-3 live-margin render build" in src


# ----------------------------------------------------------------------------- adoption seam (item 3)
def test_resume_tolerates_new_tail_flags_absent_from_frozen_ckpt():
    """Run-1's frozen checkpoint (no tail keys) resumed by a NEW launcher invocation WITH tail flags
    appended must NOT be flagged as silent lever drift — the tail flags are a resume-time config
    EXTENSION, not a dropped/changed trained lever. _resume_lever_divergences checks only persisted
    __cfg_* keys, so tail keys (never persisted) are tolerated."""
    m = _load_trainer()
    args = types.SimpleNamespace(
        mod_dim=32, lane_render_band=False, lane_band_dash_comb=False, persistence_loss_weight=0.0,
        amplify_weight=0.0, witness_alone_island_loss=False, seed_anneal_epochs=0, render_aa="none",
        hosc_beta_end=None, seg_focal_gamma=0.0, boundary_distance_weight=0.0, logit_adjust_loss_tau=0.0,
        weight_entropy_penalty_lambda=0.0, film_stiefel=False,
        # the resume argv adds the tail stage:
        tail_cycles_max=5, tail_dwell_min=237, tail_start_epoch=0)
    resume_cfg = {"__cfg_mod_dim": 32}   # a PRE-tail checkpoint: no tail keys persisted
    div = m._resume_lever_divergences(resume_cfg, args)
    assert not any("tail" in d for d in div), f"tail flags must not be flagged as drift: {div}"


# ----------------------------------------------------------------------------- DSL triality mapping
def test_dsl_tailcycles_factory_holds_all_tail_flags():
    from tac.witness_dsl.curriculum_dsl import TailCycles
    lev = TailCycles()
    assert lev.name == "tail_k_warm_restart"
    # sealed law defaults consumed (req T)
    assert lev.overrides["--tail-cycle-floor-epochs"] == pytest.approx(387.09)
    assert lev.overrides["--tail-dwell-min"] == 237
    assert lev.overrides["--tail-cycles-max"] == 5


def test_lever_registry_maps_the_tail_flags():
    from tac.witness_dsl.lever_registry import completeness
    c = completeness()
    tail_flags = {"--tail-cycles-max", "--tail-start-epoch", "--tail-cycle-floor-epochs",
                  "--tail-dwell-min", "--tail-tau-halving", "--tail-lr-prop-tau",
                  "--tail-stop-marginal-s"}
    unmapped = tail_flags & set(c.unmapped)
    assert unmapped == set(), f"tail flags must be DSL-held (mapped), still unmapped: {unmapped}"


def test_law_consistency_factory_default_equals_evaluator():
    """req T: the factory's cycle-floor default is the tail_cycle_floor_v1 value (settle 3/ν + 150)."""
    from tac.canonical_equations.evaluators import eval_settle_window, eval_tail_cycle_floor
    from tac.witness_dsl.curriculum_dsl import TailCycles
    floor = eval_tail_cycle_floor({"coeff": 3.0, "nu": 0.012653, "tail_extra": 150.0})
    settle = eval_settle_window({"coeff": 3.0, "nu": 0.012653})
    lev = TailCycles()
    assert lev.overrides["--tail-cycle-floor-epochs"] == pytest.approx(floor, abs=0.01)
    assert lev.overrides["--tail-dwell-min"] == pytest.approx(settle, abs=0.5)
