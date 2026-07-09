"""#220 AACoverageRender COMPLETION — mode∈{supersample,ipe}, grid≥384 enforcement, and the
F2 resume-divergence guard for the AA supersample FACTOR.

Reconcile context: the ``AACoverageRender`` DSL Lever factory + its #224 trainer wire-in already
EXISTED (FEED-07b, 2026-07-07) — the lever was fireable via ``--dsl-lever AACoverageRender`` and
auto-tracked by the activation ledger. This suite locks the 2026-07-09 COMPLETION: (1) the factory
now EXPRESSES the ``ipe`` chart (was supersample-only, forcing the ``render_aa: "ipe"`` raw-override
config-orphan in witness_autoconfig v6), (2) ``grid>=384`` is ENFORCED (was in the name only), and
(3) the resume-divergence guard covers the supersample FACTOR (mode-only before → a 2→3 resume
silently re-gridded). NO-FAKE: every assertion drives the REAL factory / REAL trainer argparse /
REAL pure guard fn — none checks a constant. means != ends: plumbing, NOT a score; pointer UNMOVED.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as LR

_REPO = Path(__file__).resolve().parents[3]
_TRAINER_PATH = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"


# ─── factory: byte-identical default + both charts ──────────────────────────────────────────────
def test_zero_arg_is_supersample_grid384_ss2_byte_identical_to_pre_completion():
    """Zero-arg default == the pre-#220-completion lever exactly (no behavior drift for existing
    composes / autoconfig): supersample chart, base grid 384x512, ss=2."""
    lv = cd.AACoverageRender()
    assert lv.name == "FEED_07b_aa_coverage_render"
    assert lv.overrides == {
        "--render-aa": "supersample", "--render-h": 384, "--render-w": 512, "--aa-supersample": 2,
    }


def test_supersample_mode_emits_aa_supersample_not_ipe_footprint():
    lv = cd.AACoverageRender(mode="supersample", ss=3)
    assert lv.overrides["--render-aa"] == "supersample"
    assert lv.overrides["--aa-supersample"] == 3
    assert "--aa-ipe-footprint" not in lv.overrides


def test_ipe_mode_expressible_emits_ipe_footprint_not_supersample():
    """The #220 completion: the factory now HOLDS the ipe chart (kills the raw-override orphan)."""
    lv = cd.AACoverageRender(mode="ipe", ipe_footprint=1.5)
    assert lv.overrides["--render-aa"] == "ipe"
    assert lv.overrides["--aa-ipe-footprint"] == 1.5
    assert "--aa-supersample" not in lv.overrides


def test_ipe_default_footprint_is_one():
    assert cd.AACoverageRender(mode="ipe").overrides["--aa-ipe-footprint"] == 1.0


def test_window_maps_to_epochs_delta():
    assert cd.AACoverageRender(window=250).epochs_delta == 250


# ─── grid≥384 enforcement (the named "+ grid≥384" half of #220) ─────────────────────────────────
def test_grid_below_384_height_refused():
    with pytest.raises(ValueError, match="grid>=384"):
        cd.AACoverageRender(grid_h=192)


def test_grid_below_384_width_refused():
    with pytest.raises(ValueError, match="grid>=384"):
        cd.AACoverageRender(grid_w=256)


def test_grid_exactly_384_accepted_boundary():
    lv = cd.AACoverageRender(grid_h=384, grid_w=384)
    assert lv.overrides["--render-h"] == 384 and lv.overrides["--render-w"] == 384


def test_grid_above_384_accepted_and_enforced_for_ipe_too():
    lv = cd.AACoverageRender(mode="ipe", grid_h=512, grid_w=768)
    assert lv.overrides == {
        "--render-aa": "ipe", "--render-h": 512, "--render-w": 768, "--aa-ipe-footprint": 1.0,
    }
    with pytest.raises(ValueError, match="grid>=384"):
        cd.AACoverageRender(mode="ipe", grid_h=200)


# ─── input validation ───────────────────────────────────────────────────────────────────────────
def test_bad_mode_refused():
    with pytest.raises(ValueError, match="mode must"):
        cd.AACoverageRender(mode="point")


def test_ss_below_one_refused():
    with pytest.raises(ValueError, match="ss must"):
        cd.AACoverageRender(ss=0)


def test_ss_one_is_byte_identical_downsample_identity():
    """ss=1 supersample is the identity box-downsample (byte-identical render); still a valid arm."""
    lv = cd.AACoverageRender(ss=1)
    assert lv.overrides["--aa-supersample"] == 1


# ─── composability + never-invent-flags (real trainer argparse) ─────────────────────────────────
def test_composable_via_dsl_lever_bare_name():
    assert "AACoverageRender" in LR.name_composable_levers()
    assert isinstance(LR.resolve_composable_lever("AACoverageRender"), cd.Lever)


def test_all_emitted_flags_are_real_trainer_flags_both_modes():
    """never-invent-flags: every override key parses through the trainer's REAL argparse (AST-built,
    no MLX import). Covers supersample AND the newly-expressible ipe chart + the grid knob."""
    real = cd.real_trainer_flags(_TRAINER_PATH)
    for lv in (cd.AACoverageRender(), cd.AACoverageRender(mode="ipe"),
               cd.AACoverageRender(grid_h=512, grid_w=768)):
        for flag in lv.overrides:
            assert flag in real, f"{flag} is not a real trainer flag"


def test_auto_registered_in_activation_ledger_never_fired_duty_to_measure():
    """The lever is auto-tracked (lever_registry AST) — never-fired + owed-a-measurement by
    construction until a DSL-path launch records a 'fired' event. The 'off is a tracked queue' bar."""
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers, never_fired
    assert "AACoverageRender" in known_levers()
    assert "AACoverageRender" in never_fired()
    assert "AACoverageRender" in duty_to_measure()


# ─── F2 resume-divergence guard: the AA supersample FACTOR (real pure trainer fn) ───────────────
def test_resume_guard_flags_supersample_factor_change():
    """A resume that keeps mode='supersample' but changes --aa-supersample 2->3 renders a DIFFERENT
    grid — the mode-only guard missed it; the #220 factor guard fails closed. Drives the REAL pure
    ``_resume_lever_divergences`` (MLX import; skipped where unavailable)."""
    pytest.importorskip("mlx")
    from types import SimpleNamespace

    import experiments.train_levelset_witness_realized_through_R_mlx as T

    sidecar = {"__cfg_aa_supersample": np.asarray(2), "__cfg_render_aa": np.asarray("supersample")}
    div = T._resume_lever_divergences(sidecar, SimpleNamespace(aa_supersample=3, render_aa="supersample"))
    assert any("aa_supersample" in d for d in div), div
    # same factor => no divergence
    same = T._resume_lever_divergences(sidecar, SimpleNamespace(aa_supersample=2, render_aa="supersample"))
    assert not any("aa_supersample" in d for d in same), same
    # legacy sidecar lacking the key => NO spurious divergence (only present keys checked)
    legacy = T._resume_lever_divergences(
        {"__cfg_render_aa": np.asarray("supersample")},
        SimpleNamespace(aa_supersample=3, render_aa="supersample"))
    assert not any("aa_supersample" in d for d in legacy), legacy
