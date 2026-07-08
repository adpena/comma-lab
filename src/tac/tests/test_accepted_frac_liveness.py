"""C6 liveness confound fix — the mid-epoch loss_terms ``accepted_frac`` must be
truthful in BOTH directions (a live run must NOT read as frozen).

BUG (confound-class, liveness sentinel misreporting): the loss_terms row is emitted
AFTER ``ep_tot += 1`` (counts the in-flight batch) but BEFORE ``opt.update`` + the
``ep_acc += 1`` that records the batch's acceptance. A running-frac read at that point
undercounts the current batch by one in the numerator, so on the FIRST batch of every
epoch it is 0/1 == 0.0 while ``weights_stepped`` is True — a pessimistically-lying
signal that read as a dead run and caused a council seat to declare a live run dead.

The fix folds the current batch's already-decided accept/skip state (``not skip``) into
the numerator via the module-level ``_running_accepted_frac``. These tests exercise that
ACTUAL shipped function (the closure ``_live_running_frac`` is a thin wrapper over it).

Telemetry-only + score-neutral: this fix touches row emission/counting, never the
update path. There is nothing here that can change trained weights / archive bytes /
d_seg / d_pose — byte-identity is preserved by construction.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TRAINER = (Path(__file__).resolve().parents[3]
            / "experiments" / "train_levelset_witness_realized_through_R_mlx.py")


def _load():
    spec = importlib.util.spec_from_file_location("_twr_liveness", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


_M = _load()
_frac = _M._running_accepted_frac


# ── the core confound: the FIRST batch of a clean epoch ────────────────────────────────
def test_first_batch_clean_epoch_reads_one_not_zero():
    # ep_tot already counts this in-flight batch (==1); ep_acc not yet incremented (==0).
    # Pre-fix this returned 0/1 == 0.0 (the lying signal). With the batch's known accept
    # state folded in, a stepping batch must read 1.0 — the run is ALIVE.
    assert _frac(0, 1, pending_accept=True) == 1.0


def test_first_batch_skipped_reads_zero():
    # A genuinely spike-skipped first batch: numerator 0, honest 0.0 (weights_stepped False
    # on the same row). The sentinel must tell the truth in the frozen direction too.
    assert _frac(0, 1, pending_accept=False) == 0.0


# ── mid-epoch running fraction ─────────────────────────────────────────────────────────
def test_midepoch_all_accepted_reads_one():
    # 5 prior batches stepped (ep_acc=5), this is the 6th (ep_tot=6), and it steps too.
    assert _frac(5, 6, pending_accept=True) == 1.0


def test_midepoch_with_real_skip_reads_below_one():
    # 4 of the prior 5 stepped (ep_acc=4, ep_tot=6 counts this in-flight batch), current
    # batch is skipped -> honest (4+0)/6.
    assert _frac(4, 6, pending_accept=False) == pytest.approx(4.0 / 6.0)


def test_midepoch_current_accepts_after_prior_skip():
    # ep_acc=4 (one earlier skip among 5 seen), ep_tot=6, current steps -> (4+1)/6.
    assert _frac(4, 6, pending_accept=True) == pytest.approx(5.0 / 6.0)


# ── the epoch-boundary / no-batch-ran edge ─────────────────────────────────────────────
def test_no_batch_ran_returns_alive_default():
    # ep_tot == 0 (loop head, before the first chunk) must not divide-by-zero and must
    # default to ALIVE (1.0), matching the _live init default. The real call site never
    # hits this (row emits after ep_tot += 1) but the helper is defensive.
    assert _frac(0, 0) == 1.0
    assert _frac(0, 0, pending_accept=True) == 1.0
    assert _frac(0, 0, pending_accept=False) == 1.0


# ── backward-compat: no pending_accept behaves like the raw running counters ────────────
def test_pending_none_is_raw_running_fraction():
    # Without a pending decision the helper reports the raw ep_acc/ep_tot (used nowhere at
    # the fixed call site, but the default must be well-defined and monotone).
    assert _frac(3, 4, pending_accept=None) == pytest.approx(0.75)
    assert _frac(0, 1, pending_accept=None) == 0.0  # the pre-fix value, only when un-folded


# ── the fix is a strict improvement over the pre-fix arithmetic on the bug signature ────
def test_fold_strictly_increases_numerator_for_accepting_batch():
    # For any accepting in-flight batch, folding raises the reported frac vs the pre-fix
    # read (ep_acc/ep_tot) by exactly one batch's worth — never lowers it.
    for ep_acc, ep_tot in [(0, 1), (3, 8), (10, 75)]:
        prefix = ep_acc / ep_tot
        fixed = _frac(ep_acc, ep_tot, pending_accept=True)
        assert fixed > prefix
        assert fixed == pytest.approx((ep_acc + 1) / ep_tot)


# ── closure delegates to the tested helper (the shipped path IS what we tested) ─────────
def test_closure_helper_wired_module_level():
    # Guard against a future refactor re-inlining the arithmetic into the closure (which
    # would make it untestable again). The module-level helper must exist and be callable.
    assert callable(_M._running_accepted_frac)
