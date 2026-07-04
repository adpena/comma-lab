# SPDX-License-Identifier: MIT
"""Triality-sync tests for the Lever-D flicker-residual reactivation (FEED-03w).

Locks the THREE legs AGREE:
  * DSL  — FlickerTreatmentGauge THREE-bucket treatment (BUILT #274 down-weight / NOT-WARRANTED
    replicate / #279 DESIGN-STAGE store-regional), the trainer-flag accessor (never-invent-flags: the
    BUILT chart emits its REAL levelset flags incl. the --seg-spike-reweight gate; the two UNBUILT charts
    fail closed), and the seeded cost cells (BUILT measured=True; the two UNBUILT PENDING with None
    numerics + non-empty provenance = the NO-FAKE honest state).
  * equations — the leverd_flicker_residual_reactivation_economics_v1 contract (orphan invariant,
    PREDICTED provenance, callables reproduce the memo's break-even 34/40/54% + the b<0.65 coder GO).
"""
import pytest

from tac.canonical_equations import (
    CanonicalEquation,
    build_leverd_flicker_residual_reactivation_economics_v1,
    get_equation_by_id,
    leverd_break_even_recovery,
    leverd_coder_go,
    leverd_net_delta_s,
    leverd_survival_threshold,
    register_canonical_equation,
)
from tac.witness_dsl import (
    COMPONENT_GAUGES,
    FlickerTreatmentGauge,
    GaugeComponent,
    component_of,
    default_cost_table,
    flicker_treatment_trainer_flags,
    real_trainer_flags,
)
from tac.witness_dsl.curriculum_dsl import _REPO_ROOT
from tac.witness_dsl.gauge import (
    FLICKER_TREATMENT_TRAINER_FLAGS,
    LEVELSET_TRAINER_REL,
    SEG_SPIKE_DOWNWEIGHT_DEFAULT,
)
from tac.provenance.contract import Provenance, ProvenanceEvidenceGrade


# --- DSL leg: enum + component registration ---------------------------------
def test_flicker_treatment_three_bucket_enum_membership():
    # NONE baseline + the THREE treatment buckets.
    assert {g.value for g in FlickerTreatmentGauge} == {
        "none", "downweight_irreducible", "replicate_predictable", "store_regional_leverd"}


def test_flicker_treatment_component_registered_and_round_trips():
    assert COMPONENT_GAUGES[GaugeComponent.FLICKER_TREATMENT] is FlickerTreatmentGauge
    assert component_of(FlickerTreatmentGauge.STORE_REGIONAL_LEVERD) is GaugeComponent.FLICKER_TREATMENT


# --- DSL leg: trainer-flag emission (never-invent-flags) --------------------
def test_flicker_none_is_byte_identical_default():
    assert flicker_treatment_trainer_flags(FlickerTreatmentGauge.NONE) == ()


def test_flicker_downweight_emits_real_built_flags_with_gate():
    # BUILT #274: the value flag NO-OPs without the --seg-spike-reweight gate -> BOTH emitted.
    flags = flicker_treatment_trainer_flags(FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE)
    assert flags[0] == "--seg-spike-reweight"                 # the gate is present (not a silent no-op)
    assert "--seg-spike-downweight" in flags
    assert flags == ("--seg-spike-reweight", "--seg-spike-downweight",
                     str(SEG_SPIKE_DOWNWEIGHT_DEFAULT))
    assert SEG_SPIKE_DOWNWEIGHT_DEFAULT < 1.0                 # a real down-weight, not the no-op default
    # a threaded campaign override in [0, 1)
    assert flicker_treatment_trainer_flags(
        FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE, downweight=0.1) == (
        "--seg-spike-reweight", "--seg-spike-downweight", "0.1")


def test_flicker_downweight_refuses_no_op_downweight():
    # >= 1.0 IS the byte-identical no-op (== NONE), not a down-weight -> refuse (fail-closed).
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            flicker_treatment_trainer_flags(
                FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE, downweight=bad)


def test_flicker_unbuilt_charts_fail_closed():
    # REPLICATE_PREDICTABLE (NOT-WARRANTED) + STORE_REGIONAL_LEVERD (#279 DESIGN-STAGE) are UNBUILT ->
    # raise, do NOT fabricate a flag (never-invent-flags).
    for chart in (FlickerTreatmentGauge.REPLICATE_PREDICTABLE,
                  FlickerTreatmentGauge.STORE_REGIONAL_LEVERD):
        with pytest.raises(NotImplementedError):
            flicker_treatment_trainer_flags(chart)


def test_flicker_downweight_flags_are_real_levelset_trainer_flags():
    levelset_flags = real_trainer_flags(_REPO_ROOT / LEVELSET_TRAINER_REL)
    built = FLICKER_TREATMENT_TRAINER_FLAGS[FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE]
    # the flag NAMES (not the value token) must exist on the REAL levelset argparse.
    assert "--seg-spike-reweight" in levelset_flags
    assert "--seg-spike-downweight" in levelset_flags
    for tok in built:
        if tok.startswith("--"):
            assert tok in levelset_flags, f"DOWNWEIGHT emits non-levelset flag {tok}"


# --- DSL leg: seeded cost cells (BUILT vs PENDING, NO-FAKE) ------------------
def test_flicker_cost_cells_built_vs_pending():
    t = default_cost_table()
    dn = t.lookup(FlickerTreatmentGauge.DOWNWEIGHT_IRREDUCIBLE)
    rep = t.lookup(FlickerTreatmentGauge.REPLICATE_PREDICTABLE)
    store = t.lookup(FlickerTreatmentGauge.STORE_REGIONAL_LEVERD)
    # DOWNWEIGHT is BUILT (#274) -> measured=True, but net d_seg is #205-gated -> numeric None.
    assert dn.measured is True
    assert dn.counted_bytes is None and dn.d_seg_through_R is None
    assert "#274" in dn.provenance
    # the two UNBUILT charts are PENDING (measured=False) with None numerics + non-empty provenance.
    for c in (rep, store):
        assert c.measured is False
        assert c.counted_bytes is None and c.d_seg_through_R is None and c.conditioning is None
        assert c.provenance.strip()
        assert c.compliant and c.deterministic         # rule-118 compliant, deterministic decode
    assert "NOT-WARRANTED" in rep.provenance
    assert "0.65" in store.provenance and "#279" in store.provenance


# --- equations leg: the economics contract ----------------------------------
def test_leverd_equation_contract_and_orphan_invariant():
    eq = build_leverd_flicker_residual_reactivation_economics_v1()
    assert isinstance(eq, CanonicalEquation)
    assert eq.equation_id == "leverd_flicker_residual_reactivation_economics_v1"
    # orphan-equation invariant: non-empty producers AND consumers.
    assert len(eq.canonical_producers) >= 1 and len(eq.canonical_consumers) >= 1
    assert "tac.witness_dsl.gauge" in eq.canonical_consumers          # the DSL leg consumes it
    assert any("margin_conditional_residual" in p for p in eq.canonical_producers)  # #72 MCR coder
    assert any("levelset_byte_close_and_eval" in p for p in eq.canonical_producers)  # #202 A/B
    # two MEASURED-pending anchors (the net-S band + the coder floor).
    assert len(eq.empirical_anchors) == 2


def test_leverd_equation_carries_predicted_provenance():
    eq = build_leverd_flicker_residual_reactivation_economics_v1()
    assert isinstance(eq.provenance, Provenance)
    # SPEC/derivation -> PREDICTED, non-promotable, no score claim (means != ends; pointer UNMOVED).
    assert eq.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
    assert eq.provenance.promotion_eligible is False
    assert eq.provenance.score_claim_valid is False


def test_leverd_callables_reproduce_the_memo_numbers():
    # break-even recovery: 34% @250 KB / 40% @300 KB / 54% @400 KB (design memo).
    assert leverd_break_even_recovery(250_000) == pytest.approx(0.34, abs=0.005)
    assert leverd_break_even_recovery(300_000) == pytest.approx(0.40, abs=0.005)
    assert leverd_break_even_recovery(400_000) == pytest.approx(0.54, abs=0.005)
    # net ΔS sign: expected corner (r=0.50, 300 KB) is a (marginal) win; pessimistic (r=0.30, 400 KB) worse.
    assert leverd_net_delta_s(0.50, 300_000) == pytest.approx(-0.048, abs=0.01)
    assert leverd_net_delta_s(0.30, 400_000) > 0            # net-NEGATIVE (WORSE)
    # coder GO: spatial-only 0.90 B/flip is NO-GO; the joint-entropy target <0.65 is GO.
    assert leverd_coder_go(0.90) is False
    assert leverd_coder_go(0.64) is True
    # survival threshold σ* = b/WATERLINE: at the current b=0.99, σ*≈0.778 (> measured σ_eff 0.51 = NO-GO).
    assert leverd_survival_threshold(0.99) == pytest.approx(0.778, abs=0.005)


def test_leverd_equation_register_query_roundtrip(tmp_path):
    path = tmp_path / "leverd_eq.jsonl"
    lock = tmp_path / "leverd_eq.lock"
    eq = build_leverd_flicker_residual_reactivation_economics_v1()
    register_canonical_equation(eq, path=path, lock_path=lock, agent="test_leverd")
    got = get_equation_by_id("leverd_flicker_residual_reactivation_economics_v1", path=path)
    assert got is not None
    assert got.equation_id == eq.equation_id
