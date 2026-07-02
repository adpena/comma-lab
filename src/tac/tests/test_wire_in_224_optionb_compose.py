#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#224 Option-B compose-fix tests — lane-band+self-orient shape-compat, AA-supersample
+self-orient FAIL-CLOSED shape contract, and gauge-selectability round-trip.

Option-B made the render-side levers compose with the DEFAULT --self-orient basis so the
from-scratch witness launch can use ALL d_seg levers. These tests certify:

  (A) lane-band + self-orient: the model trunk (in_proj) accepts the PER-PAIR self-orient feats
      (base curvelet ⊕ dir_w) the fixed ``_band_feats`` provider feeds -> call_margin/
      render_lane_appearance are finite + correctly-shaped; the PRE-FIX path (base-only feats into
      a base+dir_w model) SHAPE-CRASHES (proves the fix is load-bearing, not cosmetic).
  (B) AA-supersample + self-orient: the FAIL-CLOSED shape contract — plain fine-grid curvelet feats
      are base-width (NOT base+dir_w), so feeding them to a self-orient model crashes; this is why
      the trainer keeps the guard. The correct (base+dir_w) fine feats WOULD render finite+shaped.
      The trainer source retains the guard (regression: it must NOT silently render).
  (C) gauge selectability: GaugeChoice gains render_aa/lane_band/head_geometry OPTIONAL fields
      (defaults = OFF/byte-identical); old 6-field callers unbroken; every new-enum member has a
      compliant+deterministic cost cell; fix_gauge ranks RENDER_AA->SUPERSAMPLE_2X; with_gauge
      round-trips the new kwargs.

MLX-only, CPU-friendly (no scorer / no GPU / no dispatch). Advisory; pointer 0.19110 UNMOVED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for p in ("src", "upstream", str(REPO), str(REPO / "experiments")):
    sp = str((REPO / p) if not p.startswith("/") and p != str(REPO) else p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# small deterministic witness + curvelet feats (mirrors the byte-identical smoke setup)
# ---------------------------------------------------------------------------
def _build_bank_and_feats(render_h: int, render_w: int):
    from train_witness_realized_through_R_mlx import _build_render_coords

    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
    )

    coords = _build_render_coords(render_h, render_w)
    bank = CurveletBankConfig(n_scales=2, n_orient0=8, f0=1.0, base=2.0, n_iso=4)
    B = curvelet_directional_B(bank, max_freq=None)
    feats = curvelet_feats(coords, B).astype(np.float32)
    return coords, B, feats


def _build_model(in_feat: int, num_pairs: int = 2):
    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness

    m = build_levelset_rgb_witness(
        num_pairs=num_pairs, in_feat=in_feat, hidden_dim=96, n_hidden=4, mod_dim=32,
        n_classes=5, activation="relu", softmax_temp=1.0, chroma=False,
        wire_w0=30.0, wire_s0=10.0, hosc_beta=1.0, hosc_omega=30.0)
    mx.eval(m.parameters())
    return m


# ---------------------------------------------------------------------------
# (A) lane-band + self-orient shape-compat
# ---------------------------------------------------------------------------
def test_lane_band_self_orient_feats_shape_compat_finite():
    """The Option-B fix feeds base⊕dir_w feats to call_margin/render_lane_appearance. A model built
    with in_feat=base+dir_w consumes them (finite, correct shape) — the crash the guard blocked."""
    rh, rw = 24, 32
    _coords, _B, curv = _build_bank_and_feats(rh, rw)
    base = curv.shape[1]
    n_dir_freqs = 4
    dir_w = 4 * n_dir_freqs
    in_feat = base + dir_w  # self-orient width
    m = _build_model(in_feat)
    # pre-first-reorient the dir feats are zeros (pure-curvelet iso pass) -> width base+dir_w.
    feats = np.concatenate([curv, np.zeros((curv.shape[0], dir_w), np.float32)], axis=-1)
    cf = mx.array(feats)
    margin = m.call_margin(cf, 1)
    lane_rgb = m.render_lane_appearance(cf, 1, lane_cls=1)
    assert margin.shape == (rh * rw,)
    assert lane_rgb.shape == (rh * rw, 3)
    assert np.all(np.isfinite(np.asarray(margin)))
    assert np.all(np.isfinite(np.asarray(lane_rgb)))


def test_lane_band_pre_fix_base_only_feats_crash_proves_fix_needed():
    """PRE-FIX: feeding base-only coord_feats to a self-orient (base+dir_w) model SHAPE-CRASHES at
    in_proj. This is the exact bug the Option-B ``_band_feats`` per-pair swap fixes."""
    rh, rw = 24, 32
    _coords, _B, curv = _build_bank_and_feats(rh, rw)
    base = curv.shape[1]
    dir_w = 4 * 4
    m = _build_model(base + dir_w)  # model expects base+dir_w
    cf_base_only = mx.array(curv)   # OLD coord_feats_mx: base-only width
    with pytest.raises(Exception):
        out = m.call_margin(cf_base_only, 1)
        mx.eval(out)  # force the lazy matmul to run so the shape error surfaces


def test_lane_band_no_self_orient_feats_unchanged_width():
    """When --self-orient is OFF, the band feats are the shared base-width curvelet feats — the fix
    preserves the measured no-self-orient config (same width, same values)."""
    rh, rw = 24, 32
    _coords, _B, curv = _build_bank_and_feats(rh, rw)
    m = _build_model(curv.shape[1])  # in_feat = base only (no self-orient)
    cf = mx.array(curv)
    margin = m.call_margin(cf, 1)
    assert margin.shape == (rh * rw,)
    assert np.all(np.isfinite(np.asarray(margin)))


# ---------------------------------------------------------------------------
# (B) AA-supersample + self-orient FAIL-CLOSED shape contract
# ---------------------------------------------------------------------------
def test_aa_supersample_fine_curvelet_feats_are_base_width_only():
    """Plain fine-grid curvelet feats are BASE width (2*cols) — NOT base+dir_w. So under self-orient
    they would crash a base+dir_w model; the correct fix requires recomputing the fine dir feats.
    This documents WHY the trainer fail-closes AA-supersample + self-orient."""
    from tac.boundary_math.aa_sdf_observation_render import build_supersampled_coords
    from tac.boundary_math.lever_b_levelset_generator import curvelet_feats

    rh, rw, ss = 16, 24, 2
    _coords, B, curv_base = _build_bank_and_feats(rh, rw)
    fine = build_supersampled_coords(rh, rw, ss)
    assert fine.shape[0] == (ss * rh) * (ss * rw)
    curv_fine = curvelet_feats(fine, B).astype(np.float32)
    # fine curvelet width == base curvelet width (only the row count scales by ss^2).
    assert curv_fine.shape[1] == curv_base.shape[1]
    assert curv_fine.shape[0] == (ss ** 2) * curv_base.shape[0]


def test_aa_supersample_correct_fine_feats_render_finite_and_shaped():
    """SANITY: WITH correctly-widthed fine feats (base+dir_w) the AA render is finite + correctly
    shaped -> the shape math is sound; the fail-close is a MEMORY/WALL-CLOCK-at-n600 decision, not a
    shape-impossibility (NO-FAKE: the guard blocks an n600-infeasible path, not a broken one)."""
    from tac.boundary_math.aa_sdf_observation_render import (
        SEG_H,
        SEG_W,
        build_supersampled_coords,
        render_aa_through_R_mlx,
    )
    from tac.boundary_math.lever_b_levelset_generator import curvelet_feats

    rh, rw, ss = 16, 24, 2
    _coords, B, curv_base = _build_bank_and_feats(rh, rw)
    base = curv_base.shape[1]
    dir_w = 4 * 4
    m = _build_model(base + dir_w)
    fine = build_supersampled_coords(rh, rw, ss)
    curv_fine = curvelet_feats(fine, B).astype(np.float32)
    # correct fine self-orient feats = fine curvelet ⊕ fine dir feats (zeros here == pre-reorient).
    fine_feats = np.concatenate([curv_fine, np.zeros((curv_fine.shape[0], dir_w), np.float32)], axis=-1)
    out = render_aa_through_R_mlx(m, mx.array(fine_feats), 1, rh, rw, ss)
    mx.eval(out)
    assert out.shape == (1, SEG_H, SEG_W, 3)
    assert np.all(np.isfinite(np.asarray(out)))


def test_aa_supersample_wrong_width_fine_feats_crash():
    """A base-width fine feats into a base+dir_w model crashes — the shape mismatch the guard blocks."""
    from tac.boundary_math.aa_sdf_observation_render import (
        build_supersampled_coords,
        render_aa_through_R_mlx,
    )
    from tac.boundary_math.lever_b_levelset_generator import curvelet_feats

    rh, rw, ss = 16, 24, 2
    _coords, B, curv_base = _build_bank_and_feats(rh, rw)
    m = _build_model(curv_base.shape[1] + 4 * 4)  # expects base+dir_w
    fine = build_supersampled_coords(rh, rw, ss)
    curv_fine = curvelet_feats(fine, B).astype(np.float32)  # base width only
    with pytest.raises(Exception):
        out = render_aa_through_R_mlx(m, mx.array(curv_fine), 1, rh, rw, ss)
        mx.eval(out)


def test_trainer_retains_aa_supersample_self_orient_guard():
    """Regression: the trainer MUST keep AA-supersample + self-orient FAIL-CLOSED BY DEFAULT (Wave B
    sharpened the blocker + BUILT the opt-in fine-feat modes; the refuse default still refuses so no
    unverified OOM / 50x-slow n600 path fires by accident), not silently render a wrong-width path."""
    src = (REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py").read_text()
    # fail-closed by default (refuse) is preserved (message updated in Wave B).
    assert "supersample + --self-orient is fail-closed by DEFAULT (Wave B)" in src
    assert 'use_self_orient and _aa_fine_mode == "refuse"' in src
    # the opt-in memory-safe (batch) + wall-clock-viable (full) fine-feat modes ARE built.
    assert "--aa-self-orient-fine-mode" in src
    assert "def _cf_fine_mx" in src and "def _fine_dir_feats_np" in src
    # the --structured-init supersample guard must ALSO remain.
    assert "--render-aa supersample is incompatible with --structured-init" in src


# ---------------------------------------------------------------------------
# (C) gauge selectability
# ---------------------------------------------------------------------------
def test_gauge_choice_6_field_backward_compat():
    from tac.witness_dsl.gauge import (
        CarrierGauge,
        GaugeChoice,
        GenerationGauge,
        HeadGeometryGauge,
        LaneGauge,
        MovablesGauge,
        PoseGauge,
        RenderAAGauge,
        ResidualGauge,
        WarpGauge,
        default_cost_table,
    )

    g = GaugeChoice(
        warp=WarpGauge.SCREW_TWIST, carrier=CarrierGauge.SINGLE_SDF,
        residual=ResidualGauge.CONDITIONAL_ON_LANE_PRIOR, pose=PoseGauge.RANGE_DELTA,
        movables=MovablesGauge.STORE, generation=GenerationGauge.DETERMINISTIC_FREE)
    # new fields default to the OFF/byte-identical members.
    assert g.render_aa is RenderAAGauge.NONE
    assert g.lane_band is LaneGauge.NONE
    assert g.head_geometry is HeadGeometryGauge.SOFTMAX
    assert len(g.items()) == 9
    g.validate(default_cost_table())  # must not raise


def test_canonical_gauge_validates():
    from tac.witness_dsl.gauge import CANONICAL_GAUGE, default_cost_table

    CANONICAL_GAUGE.validate(default_cost_table())


def test_new_enum_members_all_have_compliant_deterministic_cells():
    from tac.witness_dsl.gauge import (
        HeadGeometryGauge,
        LaneGauge,
        RenderAAGauge,
        default_cost_table,
    )

    t = default_cost_table()
    for enum_cls in (RenderAAGauge, LaneGauge, HeadGeometryGauge):
        for m in enum_cls:
            cell = t.lookup(m)
            assert cell is not None, f"no cost cell for {m}"
            assert cell.passes_hard_gates(), f"{m} fails rule-118/determinism hard gates"
            assert isinstance(cell.provenance, str) and cell.provenance.strip()


def test_fix_gauge_render_aa_ranks_supersample_2x():
    from tac.witness_dsl.gauge import (
        GaugeComponent,
        RenderAAGauge,
        default_cost_table,
        fix_gauge,
    )

    v = fix_gauge(GaugeComponent.RENDER_AA, default_cost_table())
    assert v.chosen is RenderAAGauge.SUPERSAMPLE_2X
    # NONE (byte-identical baseline; witness's own floor, not a chart delta) is unranked (pending).
    assert RenderAAGauge.NONE in v.pending
    assert "SUPERSAMPLE_2X" in v.explain()


def test_fix_gauge_lane_and_head_pending_no_selectable():
    from tac.witness_dsl.gauge import GaugeComponent, default_cost_table, fix_gauge

    t = default_cost_table()
    assert fix_gauge(GaugeComponent.LANE_BAND, t).chosen is None
    assert fix_gauge(GaugeComponent.HEAD_GEOMETRY, t).chosen is None


def test_with_gauge_new_kwargs_round_trip():
    from tac.witness_dsl.curriculum_dsl import BASELINE
    from tac.witness_dsl.gauge import HeadGeometryGauge, LaneGauge, RenderAAGauge

    p = BASELINE.with_gauge(
        render_aa=RenderAAGauge.SUPERSAMPLE_2X,
        lane_band=LaneGauge.BAND_RENDER_AUTHORITY,
        head_geometry=HeadGeometryGauge.ETF)
    assert p.gauge.render_aa is RenderAAGauge.SUPERSAMPLE_2X
    assert p.gauge.lane_band is LaneGauge.BAND_RENDER_AUTHORITY
    assert p.gauge.head_geometry is HeadGeometryGauge.ETF
    # unspecified components inherit the canonical (measured-winner) gauge.
    from tac.witness_dsl.gauge import WarpGauge
    assert p.gauge.warp is WarpGauge.SCREW_TWIST


def test_with_gauge_default_off_charts_byte_identical():
    """A with_gauge that overrides only non-render components leaves the render/head charts at their
    OFF/byte-identical defaults (no accidental lever activation)."""
    from tac.witness_dsl.curriculum_dsl import BASELINE
    from tac.witness_dsl.gauge import (
        CarrierGauge,
        HeadGeometryGauge,
        LaneGauge,
        RenderAAGauge,
    )

    p = BASELINE.with_gauge(carrier=CarrierGauge.SINGLE_SDF)
    assert p.gauge.render_aa is RenderAAGauge.NONE
    assert p.gauge.lane_band is LaneGauge.NONE
    assert p.gauge.head_geometry is HeadGeometryGauge.SOFTMAX


def test_render_aa_trainer_flags_none_is_empty():
    """NONE/SOFTMAX charts emit NO trainer flags (they ARE the byte-identical default)."""
    from tac.witness_dsl.gauge import (
        HeadGeometryGauge,
        LaneGauge,
        RenderAAGauge,
        head_geometry_trainer_flags,
        lane_band_trainer_flags,
        render_aa_trainer_flags,
    )

    assert render_aa_trainer_flags(RenderAAGauge.NONE) == ()
    assert lane_band_trainer_flags(LaneGauge.NONE) == ()
    assert head_geometry_trainer_flags(HeadGeometryGauge.SOFTMAX) == ()
    assert render_aa_trainer_flags(RenderAAGauge.SUPERSAMPLE_2X) == (
        "--render-aa", "supersample", "--aa-supersample", "2")
    assert lane_band_trainer_flags(LaneGauge.BAND_RENDER_AUTHORITY) == ("--lane-render-band",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
