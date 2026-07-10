"""T5 CRUCIBLE-3 v8 increment-1a MEASUREMENT config (P7 DELIVERABLE, task #380).

The committed standing form of the P7 compile: the SEALED v8 increment-1a decoupling-screen
config (`SYNTHESIS_v3_v8_20260709.md`, SEALED P6-R7) authored AS a provenance-tagged
`Inc1aScreenConfig` (via `witness_autoconfig.derive_crucible_v8_inc1a_config`) VALIDATES clean,
resolves EVERY constant on the value-provenance ladder (LawRef anchor where one exists; live
`decoupling_screen` harness constants otherwise — REUSE-not-rederive), and carries 0 unknowns.

This is the crucible-2 P7 pattern (config authored from the SEALED synthesis, fail-closed, 0
unknown) applied to a $0 MEASUREMENT config (NOT a trainer launch — no argparse argv, no GPU).
Every kill-gate threshold is asserted to match the sealed doc, and the REFUSE paths (a silently
diverging constant, a refuted b_c arm, a sub-3 seed count) are guarded so a future edit cannot
quietly break the screen.

means != ends: config plumbing, NOT a score. Pointer contest-CPU 0.19110 UNMOVED; only a
byte-closed upstream/evaluate.py n600 row < 0.19110 moves it. The 1a screen is a NECESSARY-condition
partition test (mask-optimal != score-optimal; the SUFFICIENT through-R test is 1b).
"""
from __future__ import annotations

import dataclasses

import pytest

from tac import witness_autoconfig as wac
from tac.canonical_equations.laguerre_ot_head_offset_20260709 import (
    GATE_N600_D_SEG_FLIP_MEDIAN,
    GATE_N600_D_SEG_FLIP_WEIGHTED,
    GATE_N600_D_SEG_NO_OFFSET,
)
from tac.inc1a_harness.decoupling_screen import (
    DELTA_MASK_FRAME_SAMPLING_FLOOR,
    DELTA_R_PROXY_RETIRED,
)
from tac.through_r.scaffold_assembler import N_SEG_CLASSES
from tac.witness_autoconfig import ProvenancedValue

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _cfg():
    return wac.derive_crucible_v8_inc1a_config(_GT, num_pairs=600)


# ----------------------------------------------------------------------------
# 1 — validates clean, named, 0-unknown (the crucible-2 291-token-0-unknown bar)
# ----------------------------------------------------------------------------
def test_inc1a_config_validates_clean_and_zero_unknown():
    cfg = _cfg()
    assert cfg.program_name == "crucible_v8_inc1a"
    assert cfg.validate() == [], "the sealed inc1a config must be validate-clean"
    assert cfg.unknown_count() == 0, "no field may be UNRESOLVED/TBD (the 0-unknown bar)"
    assert cfg.num_pairs == 600


# ----------------------------------------------------------------------------
# 2 — no_offset d_seg resolves THROUGH the LawRef anchor (value-identity)
# ----------------------------------------------------------------------------
def test_no_offset_dseg_traces_to_lawref_anchor():
    cfg = _cfg()
    assert cfg.no_offset_d_seg.value == GATE_N600_D_SEG_NO_OFFSET == 0.0031436
    assert cfg.no_offset_d_seg.source == wac.SRC_MEASURED
    assert "laguerre_ot_head_offset_v1" in cfg.no_offset_d_seg.provenance


# ----------------------------------------------------------------------------
# 3 — δ_mask floor is the R7 harness MEASURED-ANCHOR, NOT the retired δ_R proxy
# ----------------------------------------------------------------------------
def test_delta_mask_floor_is_r7_not_retired_proxy():
    cfg = _cfg()
    assert cfg.delta_mask_frame_sampling_floor.value == DELTA_MASK_FRAME_SAMPLING_FLOOR == 3.46e-6
    assert cfg.delta_mask_frame_sampling_floor.value != DELTA_R_PROXY_RETIRED
    # the retired proxy is RECORDED (as retired) but never the floor.
    assert cfg.delta_mask_retired_proxy.value == DELTA_R_PROXY_RETIRED
    assert "RETIRED" in cfg.delta_mask_retired_proxy.provenance
    # the operative floor is DERIVED-LIVE in-run (not a config-time number).
    assert cfg.delta_mask_operative_spec.source == wac.SRC_DERIVED
    assert "in-run seed spread" in cfg.delta_mask_operative_spec.value


# ----------------------------------------------------------------------------
# 4 — b_c mode is the sealed safe default; the flip/OT arms are the never-list
# ----------------------------------------------------------------------------
def test_bc_mode_is_no_offset_and_flip_arms_are_never():
    cfg = _cfg()
    assert cfg.bc_mode.value == "no_offset"
    assert set(cfg.bc_never.value) == {"menon", "ot_newton", "flip_weighted", "flip_median"}
    # the sealed both-flip-arms-REFUTED anchors are cited in the never provenance.
    assert str(GATE_N600_D_SEG_FLIP_WEIGHTED) in cfg.bc_never.provenance
    assert str(GATE_N600_D_SEG_FLIP_MEDIAN) in cfg.bc_never.provenance


# ----------------------------------------------------------------------------
# 5 — the sealed >=3 seed-replicate pin; a sub-3 count fails validate + derive
# ----------------------------------------------------------------------------
def test_seed_replicates_pin_is_three_and_sub_three_refuses():
    cfg = _cfg()
    assert cfg.seed_replicates_per_arm.value == 3
    # derive REFUSES a sub-3 count (F-P5-2 / sealed §B).
    with pytest.raises(ValueError, match="seed_replicates_per_arm must be >= 3"):
        wac.derive_crucible_v8_inc1a_config(_GT, num_pairs=600, seed_replicates_per_arm=1)


# ----------------------------------------------------------------------------
# 6 — the measurement grid PIN (scorer-authoritative 512×384)
# ----------------------------------------------------------------------------
def test_scorer_grid_pin_512x384():
    cfg = _cfg()
    assert tuple(cfg.scorer_grid.value) == (512, 384)
    assert cfg.scorer_grid.source == wac.SRC_MEASURED
    # camera-res is the #149 PLACEMENT grid, distinct from the compare grid.
    assert tuple(cfg.generator_render_grid.value) == (1164, 874)
    assert "#149" in cfg.generator_render_grid.provenance


# ----------------------------------------------------------------------------
# 7 — class count matches the live assembler
# ----------------------------------------------------------------------------
def test_n_seg_classes_matches_live_assembler():
    cfg = _cfg()
    assert cfg.n_seg_classes.value == N_SEG_CLASSES == 5


# ----------------------------------------------------------------------------
# 8 — F-P5-1: measure the BYTE-CLOSED composite argmax
# ----------------------------------------------------------------------------
def test_measures_byte_closed_composite():
    cfg = _cfg()
    assert cfg.measure_byte_closed_composite.value is True
    assert "BYTE-CLOSED" in cfg.measure_byte_closed_composite.provenance


# ----------------------------------------------------------------------------
# 9 — the carrier is LATERAL-CAPABLE (F-P5-1: not single-valued) + owns the seam
# ----------------------------------------------------------------------------
def test_carrier_is_lateral_capable_three_curve():
    cfg = _cfg()
    c = cfg.carrier.value
    assert c["owns_explicitly"] == "lateral_side_undrivable"
    assert "lateral" in c["mode"]
    assert set(c["curves"]) == {"top_arc", "left_lateral_extent", "right_lateral_extent"}
    assert c["dominant_arc_S"] == 0.00277  # F-P5-6 PIN (code-emitted 4167 B)
    lo, hi = c["carrier_total_S"]
    assert lo >= c["dominant_arc_S"] and hi > lo == 0.0040 and hi == 0.0083
    assert "owed-measurement" in c["carrier_total_S_grade"]


# ----------------------------------------------------------------------------
# 10 — kill-gate thresholds + falsifier match the sealed doc
# ----------------------------------------------------------------------------
def test_kill_criterion_and_falsifier_match_sealed_doc():
    cfg = _cfg()
    kc = cfg.kill_criterion.value
    assert "decoupled_mask_dseg > control_mask_dseg + delta_mask" in kc
    assert "DECOUPLING-CONFIRMED" in kc and "KILLED" in kc and "INCONCLUSIVE" in kc
    f = cfg.falsifier.value
    assert "necessary condition" in f["falsifies"]
    assert "SUFFICIENT test is 1b" in f["cannot_falsify"]
    assert f["flat_paint_confound"] == "EXCLUDED_BY_CONSTRUCTION (both arms paint-free)"
    # the A/B baseline is the IN-RUN control arm, never run-1's 0.312.
    assert "IN-RUN" in cfg.design.provenance and "0.312" in cfg.design.provenance


# ----------------------------------------------------------------------------
# 11 — the temporal section (F-P5-4) + residual operating point (F-P5-3)
# ----------------------------------------------------------------------------
def test_temporal_and_residual_riders():
    cfg = _cfg()
    t = cfg.temporal.value
    assert t["tie_flicker"].startswith("per-class")
    assert "LeverD" in t["instrument"]
    assert set(t["spec_v8_1_owed"]) == {"slot_churn", "GOP_keyframe", "dash_phase_ego_distance"}
    r = cfg.residual_operating_point.value
    assert tuple(r["r_star_RANGE"]) == (0.061, 0.135)
    assert r["shippable_rate"] == 0.135 and "WASH" in r["shippable_is"]


# ----------------------------------------------------------------------------
# 12 — the provenance manifest: schema + per-field ladder class
# ----------------------------------------------------------------------------
def test_provenance_manifest_schema_and_per_field_ladder():
    cfg = _cfg()
    m = cfg.provenance_manifest()
    assert m["schema"] == "dsl_program_manifest.v1"
    assert m["program_name"] == "crucible_v8_inc1a"
    assert m["kind"] == "measurement_screen"  # NOT a trainer launch
    assert m["unknown_count"] == 0
    assert len(m["fields"]) == len(cfg.provenanced_values())
    # every field carries a ladder class + a non-empty provenance string.
    _valid = {wac.SRC_MEASURED, wac.SRC_DERIVED, wac.SRC_DESIGN,
              wac.SRC_RECALLED, wac.SRC_HELDOUT, wac.SRC_FALLBACK}
    for name, row in m["fields"].items():
        assert row["ladder_class"] in _valid, f"{name} has an off-ladder class {row['ladder_class']}"
        assert row["provenance"].strip(), f"{name} has no provenance"


# ----------------------------------------------------------------------------
# 13 — REFUSE: a b_c mode mutated to a refuted arm flags validate
# ----------------------------------------------------------------------------
def test_validate_refuses_refuted_bc_arm():
    cfg = _cfg()
    bad = dataclasses.replace(
        cfg,
        bc_mode=ProvenancedValue(value="flip_weighted", source=wac.SRC_MEASURED,
                                 provenance="mutated", portability="scorer_fixed"),
    )
    v = bad.validate()
    assert any("no_offset" in s for s in v)
    assert any("REFUTED" in s for s in v)


# ----------------------------------------------------------------------------
# 14 — REFUSE: no_offset d_seg diverging from the LawRef anchor flags validate
# ----------------------------------------------------------------------------
def test_validate_refuses_dseg_drift_from_anchor():
    cfg = _cfg()
    bad = dataclasses.replace(
        cfg,
        no_offset_d_seg=ProvenancedValue(value=0.00272, source=wac.SRC_MEASURED,
                                         provenance="the n24/n48 SUBSET (WRONG scale)",
                                         portability="instance_conditioned"),
    )
    v = bad.validate()
    assert any("LawRef anchor" in s for s in v), "a d_seg off the n600 anchor must be refused"


# ----------------------------------------------------------------------------
# 15 — the P2 seed-honesty guard: the operative floor is DERIVED-LIVE (REFUSE at config time)
# ----------------------------------------------------------------------------
def test_operative_floor_is_derived_live_not_config_resolvable():
    # validate() (clean above) exercises the operative_delta_mask REFUSE-at-config-time contract:
    # with the declared >=3 replicates and no in-run seed_spread, the harness guard RAISES, proving
    # the operative floor is honestly NOT a config-resolvable number. Assert it directly here too.
    from tac.inc1a_harness.decoupling_screen import DecouplingScreenError, operative_delta_mask
    with pytest.raises(DecouplingScreenError):
        operative_delta_mask(seed_spread=None, n_seed_replicates=3)
    # a single seed => the frame-sampling LOWER BOUND only (INSTANCE-level, no seed component).
    assert operative_delta_mask(seed_spread=None, n_seed_replicates=1) == DELTA_MASK_FRAME_SAMPLING_FLOOR


# ----------------------------------------------------------------------------
# 16 — unknown_count detects an injected TBD placeholder
# ----------------------------------------------------------------------------
def test_unknown_count_detects_tbd_placeholder():
    cfg = _cfg()
    assert cfg.unknown_count() == 0
    bad = dataclasses.replace(
        cfg,
        metric=ProvenancedValue(value="TBD", source=wac.SRC_DESIGN,
                                provenance="placeholder", portability="scorer_fixed"),
    )
    assert bad.unknown_count() == 1
    assert any("unknown_count" in s for s in bad.validate())
