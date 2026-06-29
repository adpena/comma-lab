"""Tests for the GAUGE meta-layer (DAG FEED-ji) — the equations→GAUGE→DSL→DAG bridge.

Locks: the chart enums, the NO-FAKE GaugeCost provenance + pending-cell contracts, the
seeded MEASURED cost cells (regression-guarded against their cited FEEDs), the BY-CONSTRUCTION
compliance/determinism rejection (GaugeChoice.validate), the fix_gauge selection rule
(hard-gates → drop pending → min-S → enum-order tiebreak) with rule-chain readback, and the
pure ``with_gauge`` / ``compose_gauged_program`` composition with the DSL.
"""
import pytest

from tac.witness_dsl import (
    BASELINE,
    CANONICAL_GAUGE,
    COMPONENT_GAUGES,
    POLY_BASE_DSEG_FLOOR,
    CarrierGauge,
    GaugeChoice,
    GaugeComponent,
    GaugeCost,
    GaugeCostError,
    GaugeCostTable,
    GaugeViolation,
    GenerationGauge,
    MovablesGauge,
    PoseGauge,
    ResidualGauge,
    WarpGauge,
    compose_gauged_program,
    component_of,
    default_cost_table,
    fix_gauge,
)

_S_RATE_DENOM = 37_545_489


# --- enum membership --------------------------------------------------------
def test_chart_enum_membership_complete():
    assert {g.value for g in WarpGauge} == {"screw_twist", "per_class_homography", "learned"}
    assert {g.value for g in CarrierGauge} == {"single_sdf", "msdf", "hard_bitmap"}
    assert {g.value for g in ResidualGauge} == {
        "alard_lupton", "direct_learned", "persistence_events", "conditional_on_lane_prior"}
    assert {g.value for g in PoseGauge} == {"scalar_store", "range_delta", "low_rank"}
    assert {g.value for g in MovablesGauge} == {"store", "warp_predict"}
    assert {g.value for g in GenerationGauge} == {"deterministic_free", "learned_counted"}


def test_component_gauges_registry_and_component_of():
    assert set(COMPONENT_GAUGES) == set(GaugeComponent)
    assert COMPONENT_GAUGES[GaugeComponent.WARP] is WarpGauge
    assert component_of(CarrierGauge.SINGLE_SDF) is GaugeComponent.CARRIER
    assert component_of(ResidualGauge.CONDITIONAL_ON_LANE_PRIOR) is GaugeComponent.RESIDUAL
    with pytest.raises(TypeError):
        component_of("not a chart")


# --- GaugeCost contracts (NO-FAKE) ------------------------------------------
def test_gauge_cost_rejects_empty_provenance():
    with pytest.raises(GaugeCostError):
        GaugeCost(counted_bytes=10, d_seg_through_R=None, conditioning=None,
                  compliant=True, deterministic=True, measured=True, provenance="")
    with pytest.raises(GaugeCostError):
        GaugeCost(counted_bytes=10, d_seg_through_R=None, conditioning=None,
                  compliant=True, deterministic=True, measured=True, provenance="   ")


def test_gauge_cost_pending_cell_cannot_carry_fabricated_number():
    # NO-FAKE: an un-measured (PENDING) cell must have None numeric fields.
    for kwargs in (
        dict(counted_bytes=123, d_seg_through_R=None, conditioning=None),
        dict(counted_bytes=None, d_seg_through_R=0.001, conditioning=None),
        dict(counted_bytes=None, d_seg_through_R=None, conditioning=0.5),
    ):
        with pytest.raises(GaugeCostError):
            GaugeCost(compliant=True, deterministic=True, measured=False,
                      provenance="probe running", **kwargs)
    # a measured cell MAY carry numbers (no raise)
    ok = GaugeCost(counted_bytes=123, d_seg_through_R=0.01, conditioning=None,
                   compliant=True, deterministic=True, measured=True, provenance="x")
    assert ok.counted_bytes == 123


def test_gauge_cost_hard_gates_and_s_contribution():
    both = GaugeCost(counted_bytes=37_545_489, d_seg_through_R=0.01, conditioning=None,
                     compliant=True, deterministic=True, measured=True, provenance="x")
    assert both.passes_hard_gates() is True
    # S = 100*0.01 + 25*N/N = 1.0 + 25.0
    assert both.s_contribution() == pytest.approx(26.0)
    # byte-only (pose-style): d_seg None → only the byte term
    byteonly = GaugeCost(counted_bytes=37_545_489, d_seg_through_R=None, conditioning=None,
                         compliant=True, deterministic=True, measured=True, provenance="x")
    assert byteonly.s_contribution() == pytest.approx(25.0)
    # no rankable cost → None
    nocost = GaugeCost(counted_bytes=None, d_seg_through_R=None, conditioning=None,
                       compliant=True, deterministic=True, measured=True, provenance="x")
    assert nocost.s_contribution() is None
    # hard-gate fail
    bad = GaugeCost(counted_bytes=1, d_seg_through_R=None, conditioning=None,
                    compliant=False, deterministic=True, measured=True, provenance="x")
    assert bad.passes_hard_gates() is False


# --- default_cost_table: seeded MEASURED cells regression-guarded -----------
def test_default_table_seeded_measured_cells_match_cited_feeds():
    t = default_cost_table()
    # carrier (F1 FEED-iw 0e44b4e8d)
    assert t.lookup(CarrierGauge.SINGLE_SDF).d_seg_through_R == 0.0319
    assert t.lookup(CarrierGauge.HARD_BITMAP).d_seg_through_R == 0.166
    # pose (F4) — byte sidecars
    assert t.lookup(PoseGauge.RANGE_DELTA).counted_bytes == 875
    assert t.lookup(PoseGauge.SCALAR_STORE).counted_bytes == 4800
    # movables (F3 FEED-je 930b6d348)
    assert t.lookup(MovablesGauge.STORE).counted_bytes == 2700
    assert t.lookup(MovablesGauge.STORE).d_seg_through_R == 0.0
    assert t.lookup(MovablesGauge.WARP_PREDICT).d_seg_through_R == 0.00082
    # warp (FEED-jk screw-warp-through-R): screw/twist MEASURED-WIN ~0 marginal bytes. The REAL
    # through-R bulk-warp d_seg is now MEASURED (≈0.0048 n96; full screw total 0.0076, lane
    # 0.39 flip→CARRIER/RESIDUAL, movable→MOVABLES) and recorded in provenance, BUT the numeric
    # d_seg_through_R ranking field stays None ON PURPOSE: SCREW and PER_CLASS_HOMOGRAPHY share the
    # same generalizable bulk d_seg (the oracle edge is non-physical overfit), so fix_gauge must
    # rank them on the BYTE axis (SCREW 0 < 6600) — feeding only SCREW's measured d_seg mis-ranks it.
    assert t.lookup(WarpGauge.SCREW_TWIST).counted_bytes == 0
    assert t.lookup(WarpGauge.SCREW_TWIST).d_seg_through_R is None
    assert "FEED-jk" in t.lookup(WarpGauge.SCREW_TWIST).provenance
    assert t.lookup(WarpGauge.SCREW_TWIST).measured is True
    assert t.lookup(WarpGauge.PER_CLASS_HOMOGRAPHY).counted_bytes == 6600
    assert t.lookup(WarpGauge.PER_CLASS_HOMOGRAPHY).d_seg_through_R is None
    # generation rule-118: deterministic-free = 0 bytes
    assert t.lookup(GenerationGauge.DETERMINISTIC_FREE).counted_bytes == 0
    # residual head-start (a99f41f0 389f84f6f + a5b83c730): base measured 0.00214, coeffs ~5KB
    clp = t.lookup(ResidualGauge.CONDITIONAL_ON_LANE_PRIOR)
    assert clp.d_seg_through_R == POLY_BASE_DSEG_FLOOR == 0.00214
    assert clp.counted_bytes == 5000
    assert clp.measured is True


def test_residual_head_start_cites_wyner_ziv_pipeline_provenance():
    clp = default_cost_table().lookup(ResidualGauge.CONDITIONAL_ON_LANE_PRIOR)
    assert "a99f41f0" in clp.provenance and "a5b83c730" in clp.provenance
    assert "PENDING-GPU" in clp.provenance  # the trained residual-on-top stays pending


def test_default_table_pending_cells_are_none_and_provenanced():
    # NO-FAKE guard at the table level: every measured=False cell has all-None numbers AND
    # a non-empty provenance (no fabricated float hides in a pending cell).
    t = default_cost_table()
    pending = [(c, cost) for c, cost in t.cells.items() if not cost.measured]
    assert pending, "expected some PENDING cells in the seeded table"
    for chart, cost in pending:
        assert cost.counted_bytes is None, chart
        assert cost.d_seg_through_R is None, chart
        assert cost.conditioning is None, chart
        assert isinstance(cost.provenance, str) and cost.provenance.strip(), chart


def test_default_table_known_pending_charts():
    t = default_cost_table()
    for chart in (WarpGauge.LEARNED, CarrierGauge.MSDF,
                  ResidualGauge.ALARD_LUPTON, ResidualGauge.DIRECT_LEARNED,
                  ResidualGauge.PERSISTENCE_EVENTS, PoseGauge.LOW_RANK,
                  GenerationGauge.LEARNED_COUNTED):
        assert t.lookup(chart).measured is False, chart


# --- GaugeChoice.validate (BY-CONSTRUCTION rejection) -----------------------
def test_gauge_choice_validate_accepts_canonical_and_returns_self():
    out = CANONICAL_GAUGE.validate()
    assert out is CANONICAL_GAUGE


def test_canonical_gauge_residual_is_the_head_start():
    assert CANONICAL_GAUGE.residual is ResidualGauge.CONDITIONAL_ON_LANE_PRIOR
    assert CANONICAL_GAUGE.generation is GenerationGauge.DETERMINISTIC_FREE


def test_gauge_choice_validate_rejects_non_compliant_chart():
    t = default_cost_table().with_cell(
        CarrierGauge.SINGLE_SDF,
        GaugeCost(counted_bytes=None, d_seg_through_R=0.0319, conditioning=None,
                  compliant=False, deterministic=True, measured=True, provenance="hostile"))
    with pytest.raises(GaugeViolation) as ei:
        CANONICAL_GAUGE.validate(t)
    assert "NON-COMPLIANT" in str(ei.value)


def test_gauge_choice_validate_rejects_non_deterministic_chart():
    # LEARNED_COUNTED is seeded deterministic=False (fail-closed until certified bit-identical)
    choice = GaugeChoice(
        warp=WarpGauge.PER_CLASS_HOMOGRAPHY, carrier=CarrierGauge.SINGLE_SDF,
        residual=ResidualGauge.CONDITIONAL_ON_LANE_PRIOR, pose=PoseGauge.RANGE_DELTA,
        movables=MovablesGauge.STORE, generation=GenerationGauge.LEARNED_COUNTED)
    with pytest.raises(GaugeViolation) as ei:
        choice.validate()
    assert "NON-DETERMINISTIC" in str(ei.value)


def test_gauge_choice_validate_rejects_missing_cost_cell():
    empty = GaugeCostTable({})
    with pytest.raises(GaugeViolation) as ei:
        CANONICAL_GAUGE.validate(empty)
    assert "no GaugeCost cell" in str(ei.value)


# --- fix_gauge selection rule -----------------------------------------------
def test_fix_gauge_picks_min_s_contribution_carrier():
    v = fix_gauge(GaugeComponent.CARRIER)
    assert v.chosen is CarrierGauge.SINGLE_SDF          # 100*0.0319=3.19 < 100*0.166=16.6
    assert v.runner_up is CarrierGauge.HARD_BITMAP
    assert CarrierGauge.MSDF in v.pending               # unmeasured → pending, not selected
    assert v.chosen not in v.pending


def test_fix_gauge_pose_is_byte_only_min_bytes_wins():
    v = fix_gauge(GaugeComponent.POSE)
    assert v.chosen is PoseGauge.RANGE_DELTA            # 875B < 4800B
    assert v.runner_up is PoseGauge.SCALAR_STORE
    assert PoseGauge.LOW_RANK in v.pending


def test_fix_gauge_movables_store_dominates_warp_predict():
    v = fix_gauge(GaugeComponent.MOVABLES)
    # STORE: 25*2700/N ≈ 0.0018 ; WARP_PREDICT: 100*0.00082 = 0.082 → STORE wins (FEED-je)
    assert v.chosen is MovablesGauge.STORE
    assert v.runner_up is MovablesGauge.WARP_PREDICT


def test_fix_gauge_residual_selects_head_start_base_others_pending():
    v = fix_gauge(GaugeComponent.RESIDUAL)
    assert v.chosen is ResidualGauge.CONDITIONAL_ON_LANE_PRIOR  # measured base 0.00214
    # the binding GPU move + the other coders stay pending (NO-FAKE: unmeasured)
    assert ResidualGauge.DIRECT_LEARNED in v.pending
    assert ResidualGauge.ALARD_LUPTON in v.pending
    assert ResidualGauge.PERSISTENCE_EVENTS in v.pending


def test_fix_gauge_warp_screw_twist_wins_on_bytes():
    # a513372a FEED-jj: screw/twist (~0 marginal bytes) beats per-class homography (≈6600
    # params, overfit) on S-contribution; both d_seg PRE-R so the byte term decides.
    v = fix_gauge(GaugeComponent.WARP)
    assert v.chosen is WarpGauge.SCREW_TWIST
    assert v.runner_up is WarpGauge.PER_CLASS_HOMOGRAPHY
    assert WarpGauge.LEARNED in v.pending              # high-cost fallback, unmeasured


def test_fix_gauge_drops_hard_gate_failures():
    # generation: LEARNED_COUNTED is deterministic=False → HARD-gate dropped (not pending)
    v = fix_gauge(GaugeComponent.GENERATION)
    assert v.chosen is GenerationGauge.DETERMINISTIC_FREE
    assert GenerationGauge.LEARNED_COUNTED in v.hard_gate_dropped
    assert GenerationGauge.LEARNED_COUNTED not in v.pending


def test_fix_gauge_no_selectable_chart_returns_none():
    # a table where every carrier chart is unmeasured → chosen None, all pending
    t = default_cost_table()
    for ch in CarrierGauge:
        t = t.with_cell(ch, GaugeCost(
            counted_bytes=None, d_seg_through_R=None, conditioning=None,
            compliant=True, deterministic=True, measured=False, provenance="all pending"))
    v = fix_gauge(GaugeComponent.CARRIER, t)
    assert v.chosen is None and v.runner_up is None
    assert set(v.pending) == set(CarrierGauge)
    assert "NO selectable chart" in v.explain()


def test_fix_gauge_tiebreak_is_enum_declaration_order():
    # two carrier charts with IDENTICAL S → the earlier-declared enum member wins
    same = dict(counted_bytes=None, d_seg_through_R=0.01, conditioning=None,
                compliant=True, deterministic=True, measured=True, provenance="tie")
    t = (default_cost_table()
         .with_cell(CarrierGauge.SINGLE_SDF, GaugeCost(**same))
         .with_cell(CarrierGauge.MSDF, GaugeCost(**same))
         .with_cell(CarrierGauge.HARD_BITMAP, GaugeCost(**same)))
    v = fix_gauge(GaugeComponent.CARRIER, t)
    # SINGLE_SDF is declared first in CarrierGauge → wins the tie
    assert v.chosen is CarrierGauge.SINGLE_SDF


def test_fix_gauge_hard_gate_drop_on_custom_table():
    t = default_cost_table().with_cell(
        CarrierGauge.SINGLE_SDF,
        GaugeCost(counted_bytes=None, d_seg_through_R=0.0319, conditioning=None,
                  compliant=True, deterministic=False, measured=True, provenance="non-det"))
    v = fix_gauge(GaugeComponent.CARRIER, t)
    assert CarrierGauge.SINGLE_SDF in v.hard_gate_dropped
    assert v.chosen is CarrierGauge.HARD_BITMAP  # next-best measured-compliant


def test_verdict_explain_names_the_rule_chain():
    text = fix_gauge(GaugeComponent.CARRIER).explain()
    assert "fix_gauge(carrier)" in text
    assert "hard_gate_drop" in text
    assert "pending" in text
    assert "S-contribution" in text
    assert "CHOOSE" in text and "tiebreak" in text


# --- composition with the DSL (pure; baseline unmutated) --------------------
def test_with_gauge_attaches_validated_choice_and_leaves_baseline_unmutated():
    g = BASELINE.with_gauge(carrier=CarrierGauge.SINGLE_SDF)
    assert isinstance(g.gauge, GaugeChoice)
    assert g.gauge.carrier is CarrierGauge.SINGLE_SDF
    # unspecified components inherit the canonical gauge
    assert g.gauge.residual is ResidualGauge.CONDITIONAL_ON_LANE_PRIOR
    # BASELINE is unmutated (pure composition)
    assert BASELINE.gauge is None


def test_with_gauge_does_not_leak_into_trainer_flags():
    # the gauge is the meta-layer ABOVE the trainer flags → flag_dict is byte-identical
    before = dict(BASELINE.flag_dict())
    g = BASELINE.with_gauge(carrier=CarrierGauge.SINGLE_SDF, residual=ResidualGauge.DIRECT_LEARNED)
    assert g.flag_dict() == before
    assert g.validate() == []  # still a valid DSL program


def test_with_gauge_full_choice_positional():
    g = BASELINE.with_gauge(CANONICAL_GAUGE)
    assert g.gauge is CANONICAL_GAUGE


def test_with_gauge_rejects_non_deterministic_choice():
    with pytest.raises(GaugeViolation):
        BASELINE.with_gauge(generation=GenerationGauge.LEARNED_COUNTED)


def test_with_gauge_then_with_lever_composes_and_preserves_gauge():
    from tac.witness_dsl import PoseDecouple
    prog = BASELINE.with_gauge(carrier=CarrierGauge.SINGLE_SDF).with_lever(PoseDecouple())
    assert prog.gauge.carrier is CarrierGauge.SINGLE_SDF   # gauge survives with_lever
    assert prog.flag_dict()["--w-pose"] == 0.0             # lever applied
    assert prog.validate() == []


def test_compose_gauged_program_validates_and_is_pure():
    g = compose_gauged_program(BASELINE, CANONICAL_GAUGE)
    assert g.gauge is CANONICAL_GAUGE
    assert BASELINE.gauge is None
    # a non-compliant choice raises through compose_gauged_program too
    bad = GaugeChoice(
        warp=WarpGauge.PER_CLASS_HOMOGRAPHY, carrier=CarrierGauge.SINGLE_SDF,
        residual=ResidualGauge.CONDITIONAL_ON_LANE_PRIOR, pose=PoseGauge.RANGE_DELTA,
        movables=MovablesGauge.STORE, generation=GenerationGauge.LEARNED_COUNTED)
    with pytest.raises(GaugeViolation):
        compose_gauged_program(BASELINE, bad)
