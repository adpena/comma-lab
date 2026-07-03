"""Tests for the triality-sync gauge levers (FEED-03n..03q): LEVER-4 S_R reachability + Muon
finishing-stage schedule.

Locks: the three new chart enums (MarginSaliencyGauge / MuonMomentumGauge / MuonLRGauge), their
COMPONENT_GAUGES registration + component_of round-trip, the trainer-flag accessors (never-invent-
flags: the exact levelset/base argparse flag names), the byte-identical DEFAULT charts emit (), the
seeded MEASURED cost cells (numbers in provenance; numeric fields None => unrankable = the NO-FAKE
honest #205-gated state), and the flag-validation against the REAL argparse (margin-saliency vs the
LEVELSET trainer; the two --muon-* charts vs the BASE trainer).
"""
import pytest

from tac.witness_dsl import (
    CANONICAL_GAUGE,
    COMPONENT_GAUGES,
    GaugeComponent,
    MarginSaliencyGauge,
    MuonLRGauge,
    MuonMomentumGauge,
    component_of,
    default_cost_table,
    fix_gauge,
    margin_saliency_trainer_flags,
    muon_lr_trainer_flags,
    muon_momentum_trainer_flags,
    real_trainer_flags,
)
from tac.witness_dsl.curriculum_dsl import _REPO_ROOT
from tac.witness_dsl.gauge import (
    BASE_TRAINER_REL,
    LEVELSET_TRAINER_REL,
    MUON_LR_FINAL_FRAC_DEFAULT,
)


# --- enum membership + registration -----------------------------------------
def test_triality_sync_chart_enum_membership():
    assert {g.value for g in MarginSaliencyGauge} == {"texture_proxy", "through_r_reachability"}
    assert {g.value for g in MuonMomentumGauge} == {"cold_momentum", "warm_start"}
    assert {g.value for g in MuonLRGauge} == {"flat_lr", "anneal_lr"}


def test_components_registered_and_round_trip():
    # COMPONENT_GAUGES stays complete (set(GaugeComponent)) with the 3 new members.
    assert set(COMPONENT_GAUGES) == set(GaugeComponent)
    for comp, cls in ((GaugeComponent.MARGIN_SALIENCY, MarginSaliencyGauge),
                      (GaugeComponent.MUON_MOMENTUM, MuonMomentumGauge),
                      (GaugeComponent.MUON_LR, MuonLRGauge)):
        assert COMPONENT_GAUGES[comp] is cls
        assert component_of(next(iter(cls))) is comp
    # they are loss/optimizer-side, NOT GaugeChoice STORE fields (every GaugeChoice caller unbroken)
    for attr in ("margin_saliency", "muon_momentum", "muon_lr"):
        assert not hasattr(CANONICAL_GAUGE, attr)
    assert CANONICAL_GAUGE.validate() is CANONICAL_GAUGE


# --- trainer-flag emission (never-invent-flags) -----------------------------
def test_margin_saliency_trainer_flags_emit_real_levelset_flags():
    assert margin_saliency_trainer_flags(MarginSaliencyGauge.TEXTURE_PROXY) == (
        "--margin-saliency-uniward",)
    assert margin_saliency_trainer_flags(MarginSaliencyGauge.THROUGH_R_REACHABILITY) == (
        "--margin-saliency-reachability",)


def test_muon_momentum_trainer_flags_cold_is_byte_identical_default():
    assert muon_momentum_trainer_flags(MuonMomentumGauge.COLD_MOMENTUM) == ()
    assert muon_momentum_trainer_flags(MuonMomentumGauge.WARM_START) == (
        "--muon-warm-start-momentum",)


def test_muon_lr_trainer_flags_flat_default_and_anneal_value():
    assert muon_lr_trainer_flags(MuonLRGauge.FLAT_LR) == ()
    # ANNEAL_LR bakes the DAG-cited #205-arm default 0.1
    assert MUON_LR_FINAL_FRAC_DEFAULT == 0.1
    assert muon_lr_trainer_flags(MuonLRGauge.ANNEAL_LR) == ("--muon-lr-final-frac", "0.1")
    # a threaded campaign override
    assert muon_lr_trainer_flags(MuonLRGauge.ANNEAL_LR, final_frac=0.05) == (
        "--muon-lr-final-frac", "0.05")


def test_muon_lr_anneal_refuses_no_decay_fraction():
    # frac >= 1.0 IS the FLAT_LR default (no decay), not an anneal -> refuse (fail-closed).
    with pytest.raises(ValueError):
        muon_lr_trainer_flags(MuonLRGauge.ANNEAL_LR, final_frac=1.0)
    with pytest.raises(ValueError):
        muon_lr_trainer_flags(MuonLRGauge.ANNEAL_LR, final_frac=1.5)


# --- flag-validation against the REAL argparse (the core coherence check) ----
def test_margin_saliency_flags_are_real_levelset_trainer_flags():
    levelset_flags = real_trainer_flags(_REPO_ROOT / LEVELSET_TRAINER_REL)
    for chart in MarginSaliencyGauge:
        for flag in margin_saliency_trainer_flags(chart):
            assert flag in levelset_flags, f"{chart.name} emits non-levelset flag {flag}"


def test_muon_flags_are_real_base_trainer_flags():
    # the two --muon-* GAP flags live on the BASE trainer (the Muon-stage carrier), NOT the levelset.
    base_flags = real_trainer_flags(_REPO_ROOT / BASE_TRAINER_REL)
    assert "--muon-warm-start-momentum" in base_flags
    assert "--muon-lr-final-frac" in base_flags
    for chart in MuonMomentumGauge:
        for flag in muon_momentum_trainer_flags(chart):
            assert flag in base_flags, f"{chart.name} emits non-base flag {flag}"
    # muon_lr emits a valued flag; validate the flag name (index 0) is real on the base trainer.
    anneal = muon_lr_trainer_flags(MuonLRGauge.ANNEAL_LR)
    assert anneal[0] in base_flags


# --- seeded MEASURED cost cells (numbers in provenance; unrankable => #205-gated honest state) ---
def test_cost_cells_seeded_measured_with_numbers_in_provenance():
    t = default_cost_table()
    tex = t.lookup(MarginSaliencyGauge.TEXTURE_PROXY)
    reach = t.lookup(MarginSaliencyGauge.THROUGH_R_REACHABILITY)
    assert tex.measured and reach.measured
    # NO-FAKE: the discriminator is a #205-gated d_seg, not a byte cost -> numeric fields None.
    for c in (tex, reach):
        assert c.counted_bytes is None and c.d_seg_through_R is None
        assert c.s_contribution() is None      # unrankable
        assert c.compliant and c.deterministic
    # the MEASURED diagnostics live in provenance.
    assert "-0.033" in tex.provenance and "INERT" in tex.provenance
    assert "3.0x" in reach.provenance and "S_R" in reach.provenance
    # muon cells carry the measured spike + plateau grounding.
    cold = t.lookup(MuonMomentumGauge.COLD_MOMENTUM)
    warm = t.lookup(MuonMomentumGauge.WARM_START)
    assert "+0.000357" in cold.provenance and "SPIKE" in cold.provenance
    assert "warm-start" in warm.provenance and warm.s_contribution() is None
    flat = t.lookup(MuonLRGauge.FLAT_LR)
    anneal = t.lookup(MuonLRGauge.ANNEAL_LR)
    assert "plateaus" in flat.provenance
    assert "cosine-DECAY" in anneal.provenance


def test_fix_gauge_lists_new_levers_as_pending_until_205():
    # honest NO-FAKE state: the discriminating d_seg is #205-gated => no S-rankable winner yet;
    # both charts of each new component are measured-but-unrankable -> all pending, chosen None.
    for comp, cls in ((GaugeComponent.MARGIN_SALIENCY, MarginSaliencyGauge),
                      (GaugeComponent.MUON_MOMENTUM, MuonMomentumGauge),
                      (GaugeComponent.MUON_LR, MuonLRGauge)):
        v = fix_gauge(comp)
        assert v.chosen is None, comp
        assert set(v.pending) == set(cls), comp
        assert "NO selectable chart" in v.explain()
