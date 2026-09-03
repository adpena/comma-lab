"""Tests for ddm_eq1's two backfilled canonical equations.

Both laws were measured by arms whose memos are the primary artifacts, so these tests
are RE-DERIVATION guards, not restatements: every headline constant in each module is
recomputed from the four (or six) component numbers the source receipt published.  If a
constant is ever edited without its components, a test fails.

They also pin the two things each law exists to decide:
  * the coupling law's GATE -- a seg-only renderer move is unpayable at BOTH ends of
    the measured band, so the closure does not rest on the pessimistic anchor;
  * the detector law's SCREEN -- a global-statistic agreement does not clear a
    restricted-set constant measured on a contiguous prefix.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904 import (
    ALL_PIXEL_MEAN_N96,
    ALL_PIXEL_MEAN_N600,
    ALL_PIXEL_P95_N96,
    ALL_PIXEL_P95_N600,
    AMPLIFICATION,
    DEFAULT_AMPLIFICATION_THRESHOLD,
    DELTA_R_BY_BAND,
    DELTA_R_N96,
    DELTA_R_N600,
    GLOBAL_BIAS,
    POSITIVE_CONTROL_REL_DIFF,
    PREREGISTERED_TOLERANCE,
    RESTRICTED_BIAS,
    bias_amplification,
    build_annulus_restricted_prefix_bias_detector_v1,
    global_check_is_blind,
    prefix_constant_is_suspect,
    relative_bias,
)
from tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904 import (
    EQUATION_ID as DETECTOR_EQUATION_ID,
)
from tac.canonical_equations.renderer_seg_pose_coupling_20260903 import (
    AFR1_D_POSE,
    AFR1_D_SEG,
    AFR1_S,
    CARRIER_RECOVERY_MAX,
    CARRIER_RECOVERY_MIN,
    COUPLING_CENTRE,
    COUPLING_DISPERSION,
    COUPLING_MAX,
    COUPLING_MIN,
    FT1_BASE_D_POSE,
    FT1_BASE_D_SEG,
    FT1_CANDIDATE_D_POSE,
    FT1_CANDIDATE_D_SEG,
    FT1_COUPLING,
    RF1_BASE_D_POSE,
    RF1_BASE_D_SEG,
    RF1_CANDIDATE_D_POSE,
    RF1_CANDIDATE_D_SEG,
    RF1_COUPLING,
    build_renderer_seg_pose_coupling_shipped_object_v1,
    coupling_from_components,
    overshoot_multiple,
    payable_pose_ceiling,
    predicted_delta_d_pose,
    seg_only_move_is_payable,
)
from tac.canonical_equations.renderer_seg_pose_coupling_20260903 import (
    EQUATION_ID as COUPLING_EQUATION_ID,
)

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# renderer seg->pose coupling — re-derivation guards
# --------------------------------------------------------------------------
def test_rf1_coupling_is_rederivable_from_its_published_components() -> None:
    assert coupling_from_components(
        RF1_BASE_D_SEG, RF1_CANDIDATE_D_SEG, RF1_BASE_D_POSE, RF1_CANDIDATE_D_POSE
    ) == pytest.approx(RF1_COUPLING, rel=1e-12)


def test_ft1_coupling_is_rederivable_from_its_retained_components() -> None:
    assert coupling_from_components(
        FT1_BASE_D_SEG, FT1_CANDIDATE_D_SEG, FT1_BASE_D_POSE, FT1_CANDIDATE_D_POSE
    ) == pytest.approx(FT1_COUPLING, rel=1e-12)


def test_band_is_ordered_and_its_centre_is_geometric() -> None:
    assert COUPLING_MIN < COUPLING_CENTRE < COUPLING_MAX
    assert pytest.approx(math.sqrt(COUPLING_MIN * COUPLING_MAX), rel=1e-15) == COUPLING_CENTRE
    assert pytest.approx(1.3027143045049543, rel=1e-12) == COUPLING_DISPERSION


def test_the_two_anchors_are_independent_arms_within_1_31x() -> None:
    """Two arms, two mechanisms, one object -- the transfer is the whole result."""
    assert COUPLING_DISPERSION < 1.31
    assert COUPLING_MIN > 100.0  # the >=100 branch of ft1's own pre-registered rule


def test_payable_pose_ceiling_matches_ft1s_published_table() -> None:
    """ft1 sec 7: 25% cut -> 1.694e-05 (2.66x base); 10% cut -> 9.990e-06 (1.57x)."""
    assert payable_pose_ceiling(0.25 * AFR1_D_SEG) == pytest.approx(1.694e-05, rel=1e-3)
    assert payable_pose_ceiling(0.10 * AFR1_D_SEG) == pytest.approx(9.990e-06, rel=1e-3)
    assert payable_pose_ceiling(0.25 * AFR1_D_SEG) / AFR1_D_POSE == pytest.approx(2.66, rel=1e-2)


def test_ceiling_is_the_exact_score_inequality_not_an_approximation() -> None:
    """S_candidate == S_base exactly at the ceiling, at dB = 0."""
    delta_seg = 0.25 * AFR1_D_SEG
    ceiling = payable_pose_ceiling(delta_seg)
    lhs = -100.0 * delta_seg + math.sqrt(10.0 * ceiling) - math.sqrt(10.0 * AFR1_D_POSE)
    assert lhs == pytest.approx(0.0, abs=1e-15)


def test_byte_credit_raises_the_ceiling_and_seg_only_gets_none() -> None:
    with_credit = payable_pose_ceiling(0.25 * AFR1_D_SEG, delta_bytes=-1_000)
    without = payable_pose_ceiling(0.25 * AFR1_D_SEG)
    assert with_credit > without


def test_25pc_seg_cut_costs_1318x_base_pose_at_the_smallest_coupling() -> None:
    """ft1 sec 7: 'a 25% seg cut costs Delta d_pose ~ 8.4e-3 = 1,318x base'."""
    cost = predicted_delta_d_pose(0.25 * AFR1_D_SEG, coupling=COUPLING_MIN)
    assert cost == pytest.approx(8.4e-03, rel=1e-2)
    assert cost / AFR1_D_POSE == pytest.approx(1318.0, rel=1e-2)


def test_seg_only_renderer_move_is_unpayable_at_BOTH_ends_of_the_band() -> None:
    """The closure must not rest on the pessimistic anchor, or it is one arm's number."""
    delta_seg = 0.25 * AFR1_D_SEG
    for coupling in (COUPLING_MIN, COUPLING_CENTRE, COUPLING_MAX):
        for recovery in (CARRIER_RECOVERY_MIN, CARRIER_RECOVERY_MAX):
            assert not seg_only_move_is_payable(
                delta_seg, coupling=coupling, carrier_recovery=recovery
            )
    # ft1 sec 5.2: ~81x over at the best measured recovery.
    assert overshoot_multiple(
        delta_seg, coupling=COUPLING_MAX, carrier_recovery=CARRIER_RECOVERY_MAX
    ) == pytest.approx(81.0, rel=5e-2)


def test_a_hypothetically_small_coupling_would_reopen_the_formulation() -> None:
    """The gate must be able to say YES, or it is a tautology, not a test."""
    assert seg_only_move_is_payable(0.25 * AFR1_D_SEG, coupling=1.0)


def test_coupling_equation_builds_with_two_anchors_and_excludes_joint() -> None:
    eq = build_renderer_seg_pose_coupling_shipped_object_v1()
    assert eq.equation_id == COUPLING_EQUATION_ID
    assert len(eq.empirical_anchors) == 2
    assert {a.anchor_id for a in eq.empirical_anchors} == set(
        eq.predicted_vs_empirical_residual
    )
    excluded = " ".join(eq.domain_of_validity["excluded"]).lower()
    assert "joint" in excluded and "pose-priced" in excluded
    assert "assumption_stated_not_measured" in eq.domain_of_validity
    assert eq.provenance.promotion_eligible is False


def test_coupling_equation_prices_against_the_shipped_afr1_object() -> None:
    eq = build_renderer_seg_pose_coupling_shipped_object_v1()
    priced = eq.domain_of_validity["priced_against"]
    assert priced["S"] == AFR1_S
    assert priced["d_seg"] == AFR1_D_SEG
    assert priced["d_pose"] == AFR1_D_POSE


def _memo_text(relpath: str) -> str:
    path = REPO / relpath
    if not path.is_file():  # pragma: no cover - memo custody is outside this test's scope
        pytest.skip(f"{relpath} not present in this checkout")
    return path.read_text(encoding="utf-8", errors="replace")


def test_rf1s_memo_carries_the_components_not_the_ratio() -> None:
    """rf1 never printed 166.8 -- it printed the four components. The ratio is DERIVED.

    That distinction is load-bearing: if a future reader greps rf1 for '166.8' and finds
    nothing, this test says why. The number first appears in print in ft1, which
    re-derived it from exactly these components.
    """
    text = _memo_text(".omx/research/ddm_rf1_renderer_film_rung_20260824.md")
    for component in ("0.0003474", "0.00043022", "0.00014701", "0.01396208"):
        assert component in text, f"rf1 no longer publishes {component}"
    assert "166.8" not in text


def test_ft1s_memo_carries_both_couplings_and_calls_them_independent() -> None:
    text = _memo_text(".omx/research/ddm_ft1_shipped_renderer_aligned_finetune_20260903.md")
    assert "217.30" in text
    assert "166.8" in text
    assert "independent" in text.lower()


# --------------------------------------------------------------------------
# restricted-statistic prefix-bias detector — re-derivation guards
# --------------------------------------------------------------------------
def test_biases_are_rederivable_from_the_published_quantiles() -> None:
    assert relative_bias(DELTA_R_N96, DELTA_R_N600) == pytest.approx(RESTRICTED_BIAS, rel=1e-12)
    assert relative_bias(ALL_PIXEL_P95_N96, ALL_PIXEL_P95_N600) == pytest.approx(
        GLOBAL_BIAS, rel=1e-12
    )


def test_the_measured_amplification_is_25_94x() -> None:
    assert pytest.approx(25.941672329772032, rel=1e-12) == AMPLIFICATION
    assert bias_amplification(
        restricted_prefix=DELTA_R_N96,
        restricted_population=DELTA_R_N600,
        global_prefix=ALL_PIXEL_P95_N96,
        global_population=ALL_PIXEL_P95_N600,
    ) == pytest.approx(AMPLIFICATION, rel=1e-12)


def test_the_restricted_bias_fired_the_preregistered_falsifier_and_global_did_not() -> None:
    assert abs(RESTRICTED_BIAS) > PREREGISTERED_TOLERANCE  # +11.698% vs +/-10%
    assert abs(GLOBAL_BIAS) < PREREGISTERED_TOLERANCE  # +0.451%
    assert abs(relative_bias(ALL_PIXEL_MEAN_N96, ALL_PIXEL_MEAN_N600)) < 1e-4


def test_the_positive_control_makes_this_a_cohort_result_not_an_instrument_one() -> None:
    assert POSITIVE_CONTROL_REL_DIFF == 0.0


def test_global_check_is_blind_fires_here_and_stays_quiet_when_both_move_together() -> None:
    assert global_check_is_blind(
        restricted_prefix=DELTA_R_N96,
        restricted_population=DELTA_R_N600,
        global_prefix=ALL_PIXEL_P95_N96,
        global_population=ALL_PIXEL_P95_N600,
    )
    # A restriction that tracks the global statistic is NOT a detection.
    assert not global_check_is_blind(
        restricted_prefix=1.0,
        restricted_population=1.10,
        global_prefix=1.0,
        global_population=1.09,
    )
    assert DEFAULT_AMPLIFICATION_THRESHOLD < AMPLIFICATION


def test_zero_global_bias_is_the_pure_form_not_an_error() -> None:
    assert math.isinf(
        bias_amplification(
            restricted_prefix=1.0,
            restricted_population=1.10,
            global_prefix=1.0,
            global_population=1.0,
        )
    )


def test_the_screen_needs_both_facts() -> None:
    assert prefix_constant_is_suspect(
        statistic_is_restricted=True, cohort_is_contiguous_prefix=True
    )
    assert not prefix_constant_is_suspect(
        statistic_is_restricted=True, cohort_is_contiguous_prefix=False
    )
    assert not prefix_constant_is_suspect(
        statistic_is_restricted=False, cohort_is_contiguous_prefix=True
    )


def test_band_robustness_the_choice_of_annulus_is_not_the_confound() -> None:
    """Narrowing 4x moves delta_R only -8.28%, and the narrowest n600 band still exceeds n96."""
    narrow = DELTA_R_BY_BAND[0.25]
    wide = DELTA_R_BY_BAND[1.00]
    assert narrow / wide - 1.0 == pytest.approx(-0.0828, abs=5e-4)
    assert narrow > DELTA_R_N96


def test_detector_equation_builds_and_names_its_two_sister_costumes() -> None:
    eq = build_annulus_restricted_prefix_bias_detector_v1()
    assert eq.equation_id == DETECTOR_EQUATION_ID
    assert len(eq.empirical_anchors) == 1
    sisters = " ".join(eq.domain_of_validity["sister_laws"])
    assert "wallclock_fixed_cost_prefix_bias_v1" in sisters
    assert "seed_ensemble_falsifier_band_v1" in sisters
    assert eq.provenance.promotion_eligible is False
    assert "known_boundary" in eq.domain_of_validity


def test_detector_equation_declares_the_cure_not_only_the_defect() -> None:
    eq = build_annulus_restricted_prefix_bias_detector_v1()
    cure = eq.domain_of_validity["cure"].lower()
    assert "full population" in cure or "random draw" in cure


# --------------------------------------------------------------------------
# both laws reached the registry
# --------------------------------------------------------------------------
def test_both_equations_are_present_in_the_tracked_registry() -> None:
    registry = REPO / ".omx/state/canonical_equations_registry.jsonl"
    if not registry.is_file():  # pragma: no cover - registry custody outside test scope
        pytest.skip("registry JSONL not present in this checkout")
    ids: set[str] = set()
    for line in registry.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = row.get("equation") or row.get("payload") or row
        eq_id = payload.get("equation_id") if isinstance(payload, dict) else None
        if eq_id:
            ids.add(eq_id)
    assert COUPLING_EQUATION_ID in ids
    assert DETECTOR_EQUATION_ID in ids
