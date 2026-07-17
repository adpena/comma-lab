# SPDX-License-Identifier: MIT
"""Event-fallback phase supervision force (SPEC_v10 §13.1 row 1; FEED-lane-gain §4b).

Behavior tests for ``tac.boundary_math.phase_primitives.event_fallback_ref_and_weight_numpy``
(the "advect-where-persistent, target-where-born" composer) + the DSL lever amendment
(``PhaseAdvectionConsistency(ref='gt_advected_with_own_tie_fallback')``) + the trainer argparse
choice. Every test verifies BEHAVIOR (the composed target/weight fields), not constants.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.phase_primitives import event_fallback_ref_and_weight_numpy

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _mk(H=4, W=5):
    """A small deterministic fixture with all four site classes present:
    (a) advected-covered, (b) fallback-covered (own tie, no ref), (c) both, (d) neither."""
    t_own = np.full((H, W), -1.0, np.float32)
    own_active = np.zeros((H, W), bool)
    ref_adv = np.zeros((H, W), np.float32)
    ref_active = np.zeros((H, W), bool)
    ann = np.ones((H, W), bool)
    ground = np.ones((H, W), bool)
    # (a) pure transport site
    ref_active[0, 0] = True
    ref_adv[0, 0] = 0.25
    # (b) pure birth site (own tie only)
    own_active[1, 1] = True
    t_own[1, 1] = 0.75
    # (c) both channels present -> transport wins (fallback is a FALLBACK, never gates advection off)
    own_active[2, 2] = True
    t_own[2, 2] = 0.9
    ref_active[2, 2] = True
    ref_adv[2, 2] = 0.1
    # (d) neither -> unsupervised
    return t_own, own_active, ref_adv, ref_active, ann, ground


def test_transport_channel_kept_verbatim():
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert ref[0, 0] == np.float32(0.25)
    assert w[0, 0] == 1.0


def test_birth_site_gets_own_tie_target():
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert ref[1, 1] == np.float32(0.75)   # own GT tie, not the (zero) advected field
    assert w[1, 1] == 1.0


def test_advected_wins_where_both_channels_active():
    """Fallback NEVER gates advection off: where a valid advected reference exists it is used."""
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert ref[2, 2] == np.float32(0.1)    # advected value (0.1), NOT own tie (0.9)
    assert w[2, 2] == 1.0


def test_uncovered_site_stays_sentinel_zero_weight():
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert ref[3, 3] == np.float32(-1.0)
    assert w[3, 3] == 0.0


def test_channel_counts_are_disjoint_and_sum_to_weight():
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    # (a)+(c) advected = 2; (b) fallback = 1; disjoint by construction.
    assert n_adv == 2
    assert n_fb == 1
    assert int(w.sum()) == n_adv + n_fb


def test_annulus_masks_both_channels():
    t_own, ao, rv, ra, ann, gnd = _mk()
    ann = np.zeros_like(ann)
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert w.sum() == 0.0 and n_adv == 0 and n_fb == 0
    assert (ref == -1.0).all()


def test_ground_masks_both_channels():
    t_own, ao, rv, ra, ann, gnd = _mk()
    gnd = np.zeros_like(gnd)
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert w.sum() == 0.0 and n_adv == 0 and n_fb == 0


def test_pair0_case_all_ref_inactive_supervises_own_ties():
    """Pair 0 (no previous scored frame): the fallback covers own-tie sites — the incumbent
    mode's all-zero-weight no-op for pair 0 is replaced by birth coverage."""
    t_own, ao, rv, ra, ann, gnd = _mk()
    ra = np.zeros_like(ra)  # no advected reference anywhere (pair 0)
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert n_adv == 0
    assert n_fb == 2                       # sites (b) and (c) own-active
    assert ref[1, 1] == np.float32(0.75)
    assert ref[2, 2] == np.float32(0.9)    # own tie now used at (c) — no ref exists


def test_stateless_pure_function_no_input_mutation():
    """Anti-scope honored: no per-island persistence hold, no cross-call memory. Same inputs
    -> same outputs, and inputs are not mutated."""
    args = _mk()
    copies = tuple(a.copy() for a in args)
    r1 = event_fallback_ref_and_weight_numpy(*args)
    r2 = event_fallback_ref_and_weight_numpy(*args)
    assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])
    assert r1[2:] == r2[2:]
    for a, c in zip(args, copies):
        assert np.array_equal(np.asarray(a), np.asarray(c))


def test_output_dtypes_and_shapes():
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(t_own, ao, rv, ra, ann, gnd)
    assert ref.shape == t_own.shape and w.shape == t_own.shape
    assert ref.dtype == np.float32 and w.dtype == np.float32
    assert isinstance(n_adv, int) and isinstance(n_fb, int)
    assert set(np.unique(w)).issubset({0.0, 1.0})


def test_float_masks_accepted():
    """The trainer passes {0,1} float masks in places; the composer must coerce robustly."""
    t_own, ao, rv, ra, ann, gnd = _mk()
    ref, w, n_adv, n_fb = event_fallback_ref_and_weight_numpy(
        t_own, ao.astype(np.float32), rv, ra.astype(np.float32),
        ann.astype(np.float32), gnd.astype(np.float32))
    assert n_adv == 2 and n_fb == 1


def test_dsl_lever_accepts_fallback_ref_and_emits_override():
    from tac.witness_dsl.curriculum_dsl import PhaseAdvectionConsistency

    lever = PhaseAdvectionConsistency(weight=0.4, start_epoch=700,
                                      ref="gt_advected_with_own_tie_fallback")
    assert lever.overrides["--seg-phase-advect-ref"] == "gt_advected_with_own_tie_fallback"
    assert lever.overrides["--seg-phase-advect-weight"] == 0.4


def test_dsl_lever_rejects_unknown_ref():
    from tac.witness_dsl.curriculum_dsl import PhaseAdvectionConsistency

    with pytest.raises(ValueError, match="ref must be"):
        PhaseAdvectionConsistency(weight=0.4, ref="not_a_mode")


def test_trainer_argparse_declares_fallback_choice():
    """never-invent-flags: the trainer's --seg-phase-advect-ref choices include the new mode."""
    src = _TRAINER.read_text(errors="ignore")
    m = re.search(r"add_argument\(\"--seg-phase-advect-ref\".*?choices=\[([^\]]*)\]", src, re.S)
    assert m is not None
    assert "gt_advected_with_own_tie_fallback" in m.group(1)


def test_trainer_validation_accepts_fallback_mode_string():
    src = _TRAINER.read_text(errors="ignore")
    # the fail-closed validation tuple includes the new mode (not just argparse choices)
    assert src.count('"gt_advected_with_own_tie_fallback"') >= 2
