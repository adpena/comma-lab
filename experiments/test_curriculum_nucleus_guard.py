"""Regression guard for the PER-CLASS CRITICAL-NUCLEUS hand-off (#302, T3 curriculum-derivation).

The CE->tau hand-off law (symposium 2026-07-05 §B.2; canonical equation
``curriculum_handoff_critical_nucleus_v1``): a loss plateau is NECESSARY BUT NOT SUFFICIENT to fire
CE->tau — every scored class must ALSO be above its critical nucleus (BORN part_frac>0 AND FORMED
within-flip<=thresh), else mean-curvature flow erodes a half-formed partition (Allen-Cahn: MCF
grows nothing below the critical nucleus; MEASURED #205 lane creep). This file proves the PURE
functions that implement the guard + the boundary re-anchor:

  * ``_evt_nucleus_counts`` / ``_evt_counts_add`` — chunk-additive per-class pixel counts (the
    interchange format the chunked verdict accumulates), VERBATIM the annulus-tool arithmetic.
  * ``_evt_nucleus_stats`` — part_frac (predicted partition area) + within_flip (per-class flip).
  * ``_evt_nucleus_satisfied`` — BORN ∧ FORMED per class; a zero-mass or high-flip class blocks;
    an unscored (gt_px==0) class is VACUOUSLY satisfied.
  * ``_evt_readiness_row`` — the handoff_readiness telemetry schema (ready = plateau ∧ nucleus).
  * ``_evt_reanchor_epoch`` — (#302 M1) shift a tau-relative lever into the fired-boundary frame;
    the byte-identity contract (unfired / fired-at-cap => ep unchanged).
  * ``_evt_resolve_seg_form`` nucleus gate — plateau + nucleus-NOT-ready => HOLD; cap still fires;
    guard OFF => byte-identical to the pure-loss #292 build-2 trigger.

Binding non-negotiables proved:
  * BYTE-IDENTITY OFF: nucleus_ready defaults True; the guard gate is inert unless nucleus_gate;
    _evt_reanchor_epoch is the identity when unfired/at-cap.
  * DETERMINISM: same argmax inputs => same counts/stats/readiness; same losses+nucleus_ready =>
    same fired epoch.
  * NEVER HANGS: the CAP fires unconditionally even when nucleus never satisfies.

Loads the trainer by file path (mirrors test_closed_loop_control.py). Run:
``.venv/bin/python -m pytest experiments/test_curriculum_nucleus_guard.py``
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"

_spec = importlib.util.spec_from_file_location("_lv_ncg", _TRAINER)
_m = importlib.util.module_from_spec(_spec)
sys.modules["_lv_ncg"] = _m
_spec.loader.exec_module(_m)

_counts = _m._evt_nucleus_counts
_counts_add = _m._evt_counts_add
_stats = _m._evt_nucleus_stats
_satisfied = _m._evt_nucleus_satisfied
_readiness = _m._evt_readiness_row
_reanchor = _m._evt_reanchor_epoch
_resolve = _m._evt_resolve_seg_form


class _Args:
    """Minimal argparse-namespace stand-in for _evt_resolve_seg_form."""

    def __init__(self, **kw):
        self.tau_softplus_start_epoch = 300
        self.l7_start_epoch = 10_000          # "never" (>= epochs) => l7 guard active
        self.epochs = 1000
        self.curriculum_min_stage_epochs = 3
        self.curriculum_plateau_rel_eps = 1e-4
        self.curriculum_plateau_windows = 3
        self.curriculum_nucleus_guard = False
        self.__dict__.update(kw)


# a 2x3 argmax fixture (5 classes 0..4); class 3 is ABSENT from the prediction (zero-mass).
_GT = np.array([[0, 0, 1], [2, 3, 4]], dtype=np.int64)
_PRED_PERFECT = _GT.copy()
_PRED_NO3 = np.array([[0, 0, 1], [2, 2, 4]], dtype=np.int64)   # class-3 pixel mislabeled as 2


# ─────────────────────────── counts / stats ───────────────────────────
def test_counts_perfect_prediction():
    c = _counts([_PRED_PERFECT], [_GT])
    assert c["total_px"] == 6
    assert c["gt_px"][0] == 2 and c["gt_px"][3] == 1
    assert c["wrong_px"] == [0, 0, 0, 0, 0]           # perfect => no flips
    assert c["pred_px"][0] == 2 and c["pred_px"][3] == 1


def test_counts_zero_mass_class():
    c = _counts([_PRED_NO3], [_GT])
    assert c["pred_px"][3] == 0                        # class 3 NEVER predicted (zero-mass)
    assert c["wrong_px"][3] == 1                       # its 1 GT pixel is mislabeled
    assert c["gt_px"][3] == 1


def test_counts_add_is_chunk_additive():
    a = _counts([_PRED_PERFECT], [_GT])
    b = _counts([_PRED_NO3], [_GT])
    both = _counts([_PRED_PERFECT, _PRED_NO3], [_GT, _GT])
    added = _counts_add(a, b)
    assert added == both                              # accumulate(chunk_a, chunk_b) == whole-batch
    assert _counts_add(None, b) == b                  # None seed => copy of b


def test_stats_part_frac_and_within_flip():
    st = _stats(_counts([_PRED_NO3], [_GT]))
    assert st[3]["part_frac"] == 0.0                  # zero-mass => part_frac 0
    assert st[3]["within_flip"] == 1.0                # its only GT pixel flipped
    assert st[0]["within_flip"] == 0.0               # road perfect
    assert abs(st[2]["part_frac"] - (2 / 6)) < 1e-12  # class2 predicted at 2/6 px (its own + class3's)


def test_stats_matches_annulus_tool_arithmetic():
    # per-class within_flip == wrong_c / gt_c (the canonical stage_stats formula).
    st = _stats(_counts([_PRED_NO3], [_GT]))
    for c in range(5):
        cc = _counts([_PRED_NO3], [_GT])
        gt = cc["gt_px"][c]
        exp = (cc["wrong_px"][c] / gt) if gt else 0.0
        assert abs(st[c]["within_flip"] - exp) < 1e-12


# ─────────────────────────── nucleus satisfaction ───────────────────────────
def test_satisfied_perfect_all_ok():
    _pc, all_ok = _satisfied(_stats(_counts([_PRED_PERFECT], [_GT])), within_flip_thresh=0.5)
    assert all_ok is True


def test_satisfied_zero_mass_class_blocks():
    pc, all_ok = _satisfied(_stats(_counts([_PRED_NO3], [_GT])), within_flip_thresh=0.5)
    assert pc[3] is False                             # BORN gate: part_frac 0 => not satisfied
    assert all_ok is False


def test_satisfied_high_flip_class_blocks():
    # class 0 predicted (part_frac>0) but every GT-0 pixel flipped => FORMED gate fails.
    pred = np.array([[1, 1, 1], [2, 3, 4]], dtype=np.int64)   # both GT-0 pixels -> class1
    pc, all_ok = _satisfied(_stats(_counts([pred], [_GT])), within_flip_thresh=0.5)
    assert pc[0] is False                             # within_flip 1.0 > 0.5
    assert all_ok is False


def test_satisfied_unscored_class_vacuous():
    # a GT with NO class-4 pixels => class 4 is vacuously satisfied (never blocks).
    gt = np.array([[0, 0], [1, 2]], dtype=np.int64)
    pred = gt.copy()
    pc, all_ok = _satisfied(_stats(_counts([pred], [gt])), within_flip_thresh=0.01)
    assert pc[4] is True and pc[3] is True            # absent-in-GT classes vacuously ok
    assert all_ok is True


def test_satisfied_min_part_frac_gate():
    # a tiny-mass class (1/6) is BORN only if part_frac strictly exceeds min_part_frac.
    st = _stats(_counts([_PRED_PERFECT], [_GT]))     # class1 part_frac = 1/6 ~ 0.1667
    pc_lo, _ = _satisfied(st, within_flip_thresh=1.0, min_part_frac=0.1)
    pc_hi, _ = _satisfied(st, within_flip_thresh=1.0, min_part_frac=0.2)
    assert pc_lo[1] is True                           # 0.1667 > 0.1
    assert pc_hi[1] is False                          # 0.1667 !> 0.2


# ─────────────────────────── readiness row ───────────────────────────
def test_readiness_row_schema_and_ready():
    st = _stats(_counts([_PRED_PERFECT], [_GT]))
    sat, all_ok = _satisfied(st, within_flip_thresh=0.5)
    row = _readiness(150, "ce", st, sat, all_ok, plateau_ok=True,
                     within_flip_thresh=0.5, min_part_frac=0.0, guard_active=True)
    assert row["stage"] == "handoff_readiness" and row["epoch"] == 150
    assert row["ready"] is True                       # plateau ∧ nucleus
    assert set(row["per_class"].keys()) == {"0", "1", "2", "3", "4"}   # string keys
    assert row["per_class"]["0"]["nucleus_ok"] is True


def test_readiness_ready_requires_both_plateau_and_nucleus():
    st = _stats(_counts([_PRED_NO3], [_GT]))          # nucleus NOT all-ok (class3 zero-mass)
    sat, all_ok = _satisfied(st, within_flip_thresh=0.5)
    assert all_ok is False
    row = _readiness(150, "ce", st, sat, all_ok, plateau_ok=True,
                     within_flip_thresh=0.5, min_part_frac=0.0, guard_active=True)
    assert row["ready"] is False                      # plateau True but nucleus False => not ready


# ─────────────────────────── boundary re-anchor (M1) ───────────────────────────
def test_reanchor_identity_when_unfired():
    assert _reanchor(123, None, 300) == 123           # byte-identity: unfired => ep unchanged


def test_reanchor_identity_when_fired_at_cap():
    assert _reanchor(300, 300, 300) == 300            # fired exactly at cap => no shift


def test_reanchor_early_fire_shifts_forward():
    # tau calibrated @300, fires @200 => shift +100 => a lever completing at virtual 300 => real 200;
    # analytic band at virtual 350 (tau+50) => real 250 (fired+50).
    assert _reanchor(200, 200, 300) == 300            # real 200 maps to virtual 300 (completion)
    assert _reanchor(250, 200, 300) == 350            # real 250 maps to virtual 350 (band engage)


def test_reanchor_late_fire_shifts_backward():
    # tau fires LATE @400 => shift -100 => a lever's virtual-300 event lands at real 400.
    assert _reanchor(400, 400, 300) == 300


# ─────────────────────────── event-trigger nucleus gate ───────────────────────────
def _ce_plateau_state(nucleus_ready: bool) -> dict:
    # a flat CE loss window (plateau) long enough to pass min_stage_epochs.
    losses = [10.0, 10.0, 10.0, 10.0, 10.0]
    return {"tau": None, "l7": None, "stage_start": 1, "losses": losses,
            "nucleus_ready": nucleus_ready}


def test_trigger_guard_off_fires_on_plateau_regardless_of_nucleus():
    # guard OFF => byte-identical to #292 build-2: a plateau fires CE->tau even if nucleus not ready.
    args = _Args(curriculum_nucleus_guard=False)
    state = _ce_plateau_state(nucleus_ready=False)
    form, evt = _resolve(6, state, args)
    assert form == "tau_softplus" and evt is not None and evt["trigger"] == "loss_plateau"


def test_trigger_guard_on_holds_when_nucleus_not_ready():
    # guard ON + plateau reached BUT nucleus NOT ready => HOLD in CE (no fire).
    args = _Args(curriculum_nucleus_guard=True)
    state = _ce_plateau_state(nucleus_ready=False)
    form, evt = _resolve(6, state, args)
    assert form == "ce" and evt is None               # held: plateau necessary, nucleus not sufficient
    assert state["tau"] is None


def test_trigger_guard_on_fires_when_nucleus_ready():
    args = _Args(curriculum_nucleus_guard=True)
    state = _ce_plateau_state(nucleus_ready=True)
    form, evt = _resolve(6, state, args)
    assert form == "tau_softplus" and evt is not None
    assert evt["nucleus_gated"] is True and evt["nucleus_ready"] is True


def test_trigger_cap_fires_unconditionally_even_if_nucleus_never_ready():
    # NEVER HANGS: at the hardcoded cap the boundary fires by CAP even with the guard on + not ready.
    args = _Args(curriculum_nucleus_guard=True, tau_softplus_start_epoch=6)
    # losses NOT plateaued (falling) => only the cap can fire.
    state = {"tau": None, "l7": None, "stage_start": 1,
             "losses": [100.0, 80.0, 60.0, 40.0, 20.0], "nucleus_ready": False}
    form, evt = _resolve(6, state, args)              # ep == cap
    assert form == "tau_softplus" and evt is not None and evt["trigger"] == "cap"


def test_determinism_same_inputs_same_counts():
    a = _counts([_PRED_NO3, _PRED_PERFECT], [_GT, _GT])
    b = _counts([_PRED_NO3, _PRED_PERFECT], [_GT, _GT])
    assert a == b
    assert _stats(a) == _stats(b)
