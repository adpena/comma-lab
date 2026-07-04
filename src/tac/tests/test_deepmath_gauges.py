"""Tests for the #284 deep-math-pass gauge levers (FEED-03y/03z): GammaTauEikonalGauge (Ch.4 phase-
field: geometric tau-floor + eikonal, COUPLED) + StageTransitionEasingGauge (Ch.6 dynamics: deconflict
the ep300 collision + LR re-warmup). Both are the §7 cross-chapter-converged config-B (#285) A/Bs.

Locks (DSL leg of the #284 triality): the two new chart enums + COMPONENT_GAUGES registration +
component_of round-trip; the trainer-flag accessors emit the EXACT config-B tuples (READ from the
trainer defaults so BASELINE/NONE is byte-identical ()); the never-invent-flags guard (every emitted
--flag is a REAL levelset-trainer flag AND the UNBUILT NTK-whitening / dash-comb flags are NEVER
emitted); the PENDING (measured=False, None-numbers) cost cells honor the NO-FAKE __post_init__
invariant and fix_gauge lists them pending (net-S #205-gated => no rankable winner); and the equation
cross-refs on the existing gauges (along-tangent shearlet + Muon != natural-gradient false-friend).

means != ends: all levers are net-S #205-gated + operator-GO-gated; the contest-CPU pointer 0.19110 is
UNMOVED. These tests lock the DSL leg's structure, not a score claim.
"""
import pytest

from tac.witness_dsl import (
    COMPONENT_GAUGES,
    AlongTangentFrequencyGauge,
    GammaTauEikonalGauge,
    GaugeComponent,
    GaugeCost,
    GaugeCostError,
    MuonLRGauge,
    MuonMomentumGauge,
    StageTransitionEasingGauge,
    component_of,
    default_cost_table,
    fix_gauge,
    gamma_tau_eikonal_trainer_flags,
    real_trainer_flags,
    stage_transition_easing_trainer_flags,
)
from tac.witness_dsl.curriculum_dsl import _REPO_ROOT
from tac.witness_dsl.gauge import (
    DECONFLICT_LANE_BAND_START_EPOCH_DEFAULT,
    EIKONAL_TAU_FLOOR_WEIGHT_DEFAULT,
    LEVELSET_TRAINER_REL,
    STAGE_TRANSITION_REWARMUP_EPOCHS_DEFAULT,
    STAGE_TRANSITION_REWARMUP_FLOOR_DEFAULT,
    TAU_FLOOR_SOFTMAX_TEMP_END_DEFAULT,
)

# The exact config-B tuples (deepmath_converged_next_run_config_20260704.md §7 / #285).
_GAMMA_ACTIVE = ("--tau-anneal-shape", "geometric", "--softmax-temp-end", "1.0", "--eikonal-weight", "0.05")
_STAGE_ACTIVE = ("--lane-band-start-epoch", "350", "--stage-transition-rewarmup-epochs", "20",
                 "--stage-transition-rewarmup-floor", "0.1", "--stage-transition-rewarmup-shape", "cosine")
# UNBUILT deep-math levers (Ch.5-M2 NTK-whitening + Ch.1 dash-comb, #286/#287): NO flag exists ->
# emitting one would be an invented flag. The never-invent-flags guard must never surface these.
_UNBUILT_INVENTED_FLAGS = ("--ntk-whiten", "--ntk-whitening", "--dash-comb", "--dash-comb-basis")


# --- enum membership + registration -----------------------------------------
def test_deepmath_chart_enum_membership():
    assert {g.value for g in GammaTauEikonalGauge} == {"baseline", "geometric_tau_floor_eikonal"}
    assert {g.value for g in StageTransitionEasingGauge} == {"none", "deconflict_rewarmup"}


def test_components_registered_and_round_trip():
    # COMPONENT_GAUGES stays complete (set(GaugeComponent)) with the 2 new members.
    assert set(COMPONENT_GAUGES) == set(GaugeComponent)
    for comp, cls in ((GaugeComponent.GAMMA_TAU_EIKONAL, GammaTauEikonalGauge),
                      (GaugeComponent.STAGE_TRANSITION_EASING, StageTransitionEasingGauge)):
        assert COMPONENT_GAUGES[comp] is cls
        assert component_of(next(iter(cls))) is comp


# --- trainer-flag emission (byte-identical baseline + exact config-B active tuple) ------------------
def test_gamma_tau_eikonal_baseline_is_byte_identical():
    # BASELINE = the trainer's CURRENT defaults (cosine / temp-end 0.05 / eikonal 0.01) -> emits ().
    assert gamma_tau_eikonal_trainer_flags(GammaTauEikonalGauge.BASELINE) == ()


def test_gamma_tau_eikonal_active_emits_exact_coupled_tuple():
    # config-B defaults baked from module constants (READ from the trainer + §7).
    assert TAU_FLOOR_SOFTMAX_TEMP_END_DEFAULT == 1.0
    assert EIKONAL_TAU_FLOOR_WEIGHT_DEFAULT == 0.05
    assert gamma_tau_eikonal_trainer_flags(GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL) == _GAMMA_ACTIVE


def test_gamma_tau_eikonal_override_and_guards():
    # a threaded campaign override (mirrors muon_lr_trainer_flags / head_geometry_trainer_flags).
    assert gamma_tau_eikonal_trainer_flags(
        GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL, temp_end=0.5, eikonal_weight=0.02) == (
        "--tau-anneal-shape", "geometric", "--softmax-temp-end", "0.5", "--eikonal-weight", "0.02")
    # temp_end must be > 0 (geometric requires --softmax-temp-end > 0); eikonal_weight must be >= 0.
    with pytest.raises(ValueError):
        gamma_tau_eikonal_trainer_flags(GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL, temp_end=0.0)
    with pytest.raises(ValueError):
        gamma_tau_eikonal_trainer_flags(GammaTauEikonalGauge.GEOMETRIC_TAU_FLOOR_EIKONAL, eikonal_weight=-1.0)


def test_stage_transition_easing_none_is_byte_identical():
    # NONE = the CURRENT #205 config (band@300 + rewarmup off) -> emits ().
    assert stage_transition_easing_trainer_flags(StageTransitionEasingGauge.NONE) == ()


def test_stage_transition_easing_active_emits_exact_config_b_tuple():
    assert DECONFLICT_LANE_BAND_START_EPOCH_DEFAULT == 350
    assert STAGE_TRANSITION_REWARMUP_EPOCHS_DEFAULT == 20
    assert STAGE_TRANSITION_REWARMUP_FLOOR_DEFAULT == 0.1
    assert stage_transition_easing_trainer_flags(
        StageTransitionEasingGauge.DECONFLICT_REWARMUP) == _STAGE_ACTIVE


# --- never-invent-flags (the core coherence check: every --flag is REAL on the levelset trainer) ----
def test_emitted_flags_are_real_levelset_trainer_flags():
    levelset_flags = real_trainer_flags(_REPO_ROOT / LEVELSET_TRAINER_REL)
    emitters = (
        [gamma_tau_eikonal_trainer_flags(c) for c in GammaTauEikonalGauge]
        + [stage_transition_easing_trainer_flags(c) for c in StageTransitionEasingGauge]
    )
    for argv in emitters:
        for tok in argv:
            if tok.startswith("--"):  # value tokens (e.g. "1.0", "cosine") are not flags
                assert tok in levelset_flags, f"emitted non-levelset flag {tok}"


def test_unbuilt_ntk_and_dash_comb_flags_never_emitted():
    # the Ch.5-M2 NTK-whitening + Ch.1 dash-comb levers are UNBUILT (#286/#287) -> no flag -> never
    # surfaced by any chart of the two new gauges (never-invent-flags).
    all_tokens = set()
    for c in GammaTauEikonalGauge:
        all_tokens.update(gamma_tau_eikonal_trainer_flags(c))
    for c in StageTransitionEasingGauge:
        all_tokens.update(stage_transition_easing_trainer_flags(c))
    for bad in _UNBUILT_INVENTED_FLAGS:
        assert bad not in all_tokens
    # and defensively: none of the unbuilt flags even exist on the real trainer argparse.
    levelset_flags = real_trainer_flags(_REPO_ROOT / LEVELSET_TRAINER_REL)
    for bad in _UNBUILT_INVENTED_FLAGS:
        assert bad not in levelset_flags


# --- PENDING cost cells honor the NO-FAKE None-numbers invariant; fix_gauge lists them pending ------
def test_cost_cells_pending_and_honor_none_numbers_invariant():
    t = default_cost_table()
    for cls in (GammaTauEikonalGauge, StageTransitionEasingGauge):
        for chart in cls:
            c = t.lookup(chart)
            assert c is not None, chart
            # UNMEASURED deep-math levers -> PENDING (measured=False) with None numeric fields.
            assert c.measured is False, chart
            assert c.counted_bytes is None and c.d_seg_through_R is None and c.conditioning is None, chart
            assert c.s_contribution() is None, chart          # unrankable
            assert c.compliant and c.deterministic, chart      # 0-byte train-time, bit-identical decode
            assert c.provenance.strip()                        # provenance names the pending probe


def test_pending_cell_with_a_fabricated_number_is_rejected():
    # sanity: the NO-FAKE __post_init__ invariant refuses a PENDING cell that smuggles a number.
    with pytest.raises(GaugeCostError):
        GaugeCost(counted_bytes=7, d_seg_through_R=None, conditioning=None,
                  compliant=True, deterministic=True, measured=False, provenance="fabricated")


def test_fix_gauge_lists_new_levers_as_pending_until_205():
    # honest NO-FAKE state: no isolated-measured probe -> both charts pending -> chosen None.
    for comp, cls in ((GaugeComponent.GAMMA_TAU_EIKONAL, GammaTauEikonalGauge),
                      (GaugeComponent.STAGE_TRANSITION_EASING, StageTransitionEasingGauge)):
        v = fix_gauge(comp)
        assert v.chosen is None, comp
        assert set(v.pending) == set(cls), comp
        assert "NO selectable chart" in v.explain()


# --- equation cross-refs on the two new gauges + the existing gauges (the DSL-leg deliverable) ------
def test_new_gauge_docstrings_cite_their_canonical_equations():
    gdoc = GammaTauEikonalGauge.__doc__ or ""
    for eq in ("tau_eps_hbar_one_dequantization_two_scales_v1",
               "multiphase_modica_mortola_perimeter_gamma_limit_v1",
               "mcf_minority_erasure_inevitability_v1"):
        assert eq in gdoc, eq
    sdoc = StageTransitionEasingGauge.__doc__ or ""
    assert "FEED-ft" in sdoc  # the MEASURED ep300 bump the lever attacks
    for eq in ("ce_softmax_mirror_descent_natural_gradient_v1",
               "muon_finisher_schedule_warmstart_and_lr_anneal_v1"):
        assert eq in sdoc, eq


def test_along_tangent_docstring_cross_refs_shearlet_equation():
    doc = AlongTangentFrequencyGauge.__doc__ or ""
    assert "shearlet_nterm_upper_bounds_task_rate_v1" in doc
    assert "SHEARLET" in doc.upper()


def test_muon_docstrings_flag_natural_gradient_false_friend():
    for cls in (MuonMomentumGauge, MuonLRGauge):
        doc = cls.__doc__ or ""
        assert "FALSE FRIEND" in doc, cls
        assert "ce_softmax_mirror_descent_natural_gradient_v1" in doc, cls
        assert "fisher_curvature_equals_categorical_fisher_trace_caustic_v1" in doc, cls
