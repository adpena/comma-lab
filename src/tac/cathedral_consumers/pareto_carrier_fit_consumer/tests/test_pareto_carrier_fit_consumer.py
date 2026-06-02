# SPDX-License-Identifier: MIT
"""Tests for the Pareto carrier-fit ranking consumer.

NO-FAKE discipline: every synthetic frontier here is built from
``fixture_not_real=True`` rows. Tests assert the fixture flag is preserved and
threaded into the verdict so a fixture frontier can never be read as a real
measurement. No test fabricates a score claim — the consumer's output is a
RANKING / PLANNING surface with canonical false-authority markers.
"""
from __future__ import annotations

import math

import pytest

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.cathedral.consumer_contract import (
    ConsumerTier,
    validate_consumer_module,
)
from tac.cathedral_consumers import pareto_carrier_fit_consumer as mod
from tac.cathedral_consumers.pareto_carrier_fit_consumer import (
    CONTEST_BYTE_PRICE_SCORE,
    CarrierFitConsumerError,
    CarrierFitRow,
    ParetoCarrierFitFrontier,
    compute_across_carrier_pareto_frontier,
    consume_candidate,
    parse_carrier_fit_rows,
    rate_term_for_bytes,
    update_from_anchor,
)

# ---------------------------------------------------------------------------
# Fixtures — all clearly labeled fixture_not_real.
# ---------------------------------------------------------------------------


def _fixture_rows() -> list[CarrierFitRow]:
    """A clearly-labeled synthetic across-carrier ladder.

    NOT a real measurement: every row carries ``fixture_not_real=True``.
    """
    return [
        CarrierFitRow("hi_nerv", 160000, d_seg=0.0015, d_pose=3e-5,
                      budget_id="b160k", fixture_not_real=True),
        # More bytes AND better distortion than b160k → b160k stays on frontier
        # (b160k has lower rate), this stays on frontier (lower distortion).
        CarrierFitRow("hi_nerv", 200000, d_seg=0.0012, d_pose=2.5e-5,
                      budget_id="b200k", fixture_not_real=True),
        # Fewer bytes than hi_nerv b160k but worse distortion → frontier (lower
        # rate). Not dominated by anything with both lower.
        CarrierFitRow("snerv", 150000, d_seg=0.0020, d_pose=5e-5,
                      budget_id="b150k", fixture_not_real=True),
        # Best distortion at exactly baseline bytes.
        CarrierFitRow("pr101", 178493, d_seg=0.0009, d_pose=2e-5,
                      budget_id="baseline_match", fixture_not_real=True),
    ]


# ---------------------------------------------------------------------------
# NO-FAKE: fixtures are labeled, flag is threaded.
# ---------------------------------------------------------------------------


def test_all_fixture_rows_are_labeled_not_real():
    rows = _fixture_rows()
    assert all(r.fixture_not_real for r in rows), (
        "NO-FAKE invariant: every synthetic fixture row MUST set "
        "fixture_not_real=True so it cannot be read as a real measurement"
    )


def test_frontier_threads_fixture_flag_into_verdict():
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(), baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert f.any_fixture_not_real is True
    assert any(
        "fixture_not_real_rows_present" in b for b in f.blockers
    ), "fixture frontier must carry the fixture blocker"
    payload = f.as_jsonable()
    assert payload["any_fixture_not_real"] is True


def test_real_labeled_rows_do_not_set_fixture_blocker():
    rows = [
        CarrierFitRow("hi_nerv", 160000, d_seg=0.0015, d_pose=3e-5,
                      budget_id="b160k", fixture_not_real=False),
    ]
    f = compute_across_carrier_pareto_frontier(
        rows, baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert f.any_fixture_not_real is False
    assert not any("fixture_not_real_rows_present" in b for b in f.blockers)


# ---------------------------------------------------------------------------
# Canonical contract validation (Catalog #335 / #341).
# ---------------------------------------------------------------------------


def test_consumer_module_satisfies_canonical_contract():
    reg = validate_consumer_module(mod)
    assert reg.contract_compliant is True
    assert reg.validation_errors == ()
    assert reg.consumer_tier == ConsumerTier.TIER_A_OBSERVABILITY_ONLY


def test_consume_candidate_returns_tier_a_markers():
    rows = [r.as_jsonable() for r in _fixture_rows()]
    out = consume_candidate(
        {
            "carrier_fit_rows": rows,
            "baseline_archive_bytes": 178493,
            "baseline_S": 0.192,
            "carrier_fit_key": "hi_nerv::b160k",
        }
    )
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert out["axis_tag"] == "[predicted]"
    assert "pareto_carrier_fit_frontier" in out


def test_consume_candidate_no_payload_is_observability_acknowledgment():
    out = consume_candidate({"some": "other"})
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert "no carrier_fit_rows" in out["rationale"]


def test_consume_candidate_rejects_bad_payload_gracefully():
    out = consume_candidate(
        {"carrier_fit_rows": [{"carrier_id": "", "modelsize_bytes": 100}]}
    )
    assert out["predicted_delta_adjustment"] == 0.0
    assert out["promotable"] is False
    assert "input rejected" in out["rationale"]


def test_update_from_anchor_is_noop_acknowledgment():
    # Stateless per invocation; must not raise.
    assert update_from_anchor({"any": "anchor"}) is None


# ---------------------------------------------------------------------------
# Pareto frontier correctness.
# ---------------------------------------------------------------------------


def test_dominated_point_detected():
    # b200k: 200000 bytes (higher rate) AND d_seg 0.0012 / d_pose 2.5e-5.
    # pr101 baseline_match: 178493 bytes (lower rate) AND d_seg 0.0009 /
    # d_pose 2e-5 (lower distortion) → dominates b200k on BOTH axes.
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(), baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert "hi_nerv::b200k" in f.dominated_keys
    assert f.dominated_by["hi_nerv::b200k"] == "pr101::baseline_match"


def test_frontier_points_are_non_dominated():
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(), baseline_archive_bytes=178493, baseline_S=0.192
    )
    by_key = {r.key: r for r in f.rows}
    for fk in f.frontier_keys:
        cand = by_key[fk]
        for other_key in f.frontier_keys:
            if other_key == fk:
                continue
            other = by_key[other_key]
            # No frontier point may dominate another frontier point.
            dominated = (
                other.rate_term <= cand.rate_term + 1e-12
                and other.nonrate_score <= cand.nonrate_score + 1e-12
                and (
                    other.rate_term < cand.rate_term - 1e-12
                    or other.nonrate_score < cand.nonrate_score - 1e-12
                )
            )
            assert not dominated, f"{other_key} dominates frontier point {fk}"


def test_min_s_selection_is_global_min():
    rows = _fixture_rows()
    f = compute_across_carrier_pareto_frontier(
        rows, baseline_archive_bytes=178493, baseline_S=0.192
    )
    expected = min(rows, key=lambda r: r.advisory_S)
    assert f.selected_key == expected.key
    assert math.isclose(f.selected_advisory_S, expected.advisory_S, rel_tol=1e-12)


def test_beats_baseline_flag_true_when_below_baseline():
    # A clearly-labeled fixture carrier that genuinely beats 0.192.
    rows = [
        CarrierFitRow("super_carrier", 100000, d_seg=0.0005, d_pose=1e-5,
                      budget_id="tiny", fixture_not_real=True),
    ]
    f = compute_across_carrier_pareto_frontier(
        rows, baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert f.selected_advisory_S < 0.192
    assert f.selected_beats_baseline is True
    assert "beats baseline" in f.selection_reason


def test_beats_baseline_flag_false_when_above_baseline():
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(), baseline_archive_bytes=178493, baseline_S=0.192
    )
    # The fixture ladder min-S (~0.223) does not beat 0.192.
    assert f.selected_advisory_S > 0.192
    assert f.selected_beats_baseline is False
    assert "does NOT beat" in f.selection_reason


# ---------------------------------------------------------------------------
# Dykstra feasibility consumed (not reinvented).
# ---------------------------------------------------------------------------


def test_selected_feasibility_consumes_dykstra_verdict():
    # Supply explicit baseline distortion components so the "no worse than
    # baseline" polytope is non-degenerate. The selected point (pr101
    # baseline_match: d_seg=0.0009 <= 0.0015, d_pose=2e-5 <= 4e-5, bytes ==
    # baseline) lies inside the cone → feasible. The tight/slack partition is
    # read from the consumed Dykstra verdict (per-axis duals, not reinvented).
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(),
        baseline_archive_bytes=178493,
        baseline_d_seg=0.0015,
        baseline_d_pose=4e-5,
    )
    assert f.selected_feasible_vs_baseline is True
    assert set(f.selected_tight_axes) | set(f.selected_slack_axes) == {
        "seg", "pose", "rate"
    }


def test_infeasible_when_candidate_worse_than_baseline_on_all_axes():
    # Worse d_seg, worse d_pose, MORE bytes than baseline → not in the
    # "no worse than baseline" cone → infeasible projection.
    rows = [
        CarrierFitRow("bad_carrier", 250000, d_seg=0.05, d_pose=0.01,
                      budget_id="bloated", fixture_not_real=True),
    ]
    f = compute_across_carrier_pareto_frontier(
        rows,
        baseline_archive_bytes=178493,
        baseline_d_seg=0.0009,
        baseline_d_pose=2e-5,
    )
    assert f.selected_feasible_vs_baseline is False


# ---------------------------------------------------------------------------
# Canonical rate-term / contest-formula consumption (not reinvented).
# ---------------------------------------------------------------------------


def test_rate_term_uses_canonical_denominator():
    assert math.isclose(
        rate_term_for_bytes(178493),
        25.0 * 178493 / float(ORIGINAL_VIDEO_BYTES),
        rel_tol=1e-12,
    )
    assert math.isclose(CONTEST_BYTE_PRICE_SCORE, 25.0 / float(ORIGINAL_VIDEO_BYTES))


def test_advisory_S_matches_canonical_contest_formula():
    row = CarrierFitRow("c", 160000, d_seg=0.0015, d_pose=3e-5,
                        fixture_not_real=True)
    expected = contest_formula_score(
        seg_dist=0.0015, pose_dist=3e-5, archive_bytes=160000
    )
    assert math.isclose(row.advisory_S, expected, rel_tol=1e-12)


def test_parse_rejects_mismatched_advisory_S_no_fake():
    # An upstream-claimed advisory_S that does NOT match the canonical formula
    # on the supplied components must be rejected, not silently overwritten.
    with pytest.raises(CarrierFitConsumerError, match="does not match"):
        parse_carrier_fit_rows(
            [
                {
                    "carrier_id": "c",
                    "modelsize_bytes": 160000,
                    "d_seg": 0.0015,
                    "d_pose": 3e-5,
                    "advisory_S": 999.0,  # bogus
                }
            ]
        )


def test_parse_accepts_consistent_advisory_S():
    consistent = contest_formula_score(
        seg_dist=0.0015, pose_dist=3e-5, archive_bytes=160000
    )
    rows = parse_carrier_fit_rows(
        [
            {
                "carrier_id": "c",
                "modelsize_bytes": 160000,
                "d_seg": 0.0015,
                "d_pose": 3e-5,
                "advisory_S": consistent,
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0].carrier_id == "c"


def test_parse_accepts_archive_bytes_alias():
    rows = parse_carrier_fit_rows(
        [{"carrier_id": "c", "archive_bytes": 160000, "d_seg": 0.0, "d_pose": 0.0}]
    )
    assert rows[0].modelsize_bytes == 160000


# ---------------------------------------------------------------------------
# Input-contract validation.
# ---------------------------------------------------------------------------


def test_carrier_fit_row_rejects_empty_carrier():
    with pytest.raises(CarrierFitConsumerError, match="carrier_id"):
        CarrierFitRow("", 1000, d_seg=0.0, d_pose=0.0)


def test_carrier_fit_row_rejects_nonpositive_bytes():
    with pytest.raises(CarrierFitConsumerError, match="modelsize_bytes"):
        CarrierFitRow("c", 0, d_seg=0.0, d_pose=0.0)


def test_carrier_fit_row_rejects_negative_distortion():
    with pytest.raises(CarrierFitConsumerError, match="d_seg"):
        CarrierFitRow("c", 1000, d_seg=-0.1, d_pose=0.0)


def test_carrier_fit_row_rejects_nan_distortion():
    with pytest.raises(CarrierFitConsumerError, match="NaN"):
        CarrierFitRow("c", 1000, d_seg=float("nan"), d_pose=0.0)


def test_carrier_fit_row_rejects_bool_bytes():
    with pytest.raises(CarrierFitConsumerError):
        CarrierFitRow("c", True, d_seg=0.0, d_pose=0.0)  # type: ignore[arg-type]


def test_frontier_rejects_nonpositive_baseline_bytes():
    with pytest.raises(CarrierFitConsumerError, match="baseline_archive_bytes"):
        compute_across_carrier_pareto_frontier(
            _fixture_rows(), baseline_archive_bytes=0
        )


def test_frontier_empty_rows_returns_no_selection():
    f = compute_across_carrier_pareto_frontier(
        [], baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert f.selected_key is None
    assert f.selected_advisory_S is None
    assert "no_carrier_fit_rows_supplied" in f.blockers


def test_frontier_accepts_dict_rows():
    rows = [r.as_jsonable() for r in _fixture_rows()]
    f = compute_across_carrier_pareto_frontier(
        rows, baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert f.selected_key is not None


def test_frontier_rejects_mixed_row_types():
    typed = _fixture_rows()
    mixed = [typed[0], {"carrier_id": "c", "modelsize_bytes": 1000}]
    with pytest.raises(CarrierFitConsumerError, match="mixed row types"):
        compute_across_carrier_pareto_frontier(
            mixed, baseline_archive_bytes=178493  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Serialization carries false-authority markers.
# ---------------------------------------------------------------------------


def test_verdict_serialization_carries_false_authority_markers():
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(), baseline_archive_bytes=178493, baseline_S=0.192
    )
    payload = f.as_jsonable()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["promotable"] is False
    assert payload["axis_tag"] == "[predicted]"


def test_row_serialization_carries_false_authority_markers():
    row = CarrierFitRow("c", 160000, d_seg=0.0015, d_pose=3e-5,
                        fixture_not_real=True)
    payload = row.as_jsonable()
    assert payload["score_claim"] is False
    assert payload["promotable"] is False
    assert payload["axis_tag"] == "[predicted]"


def test_baseline_S_derived_when_not_supplied():
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(),
        baseline_archive_bytes=178493,
        baseline_d_seg=0.0009,
        baseline_d_pose=2e-5,
    )
    expected = contest_formula_score(
        seg_dist=0.0009, pose_dist=2e-5, archive_bytes=178493
    )
    assert math.isclose(f.baseline_S, expected, rel_tol=1e-12)


def test_verdict_is_frozen_dataclass():
    f = compute_across_carrier_pareto_frontier(
        _fixture_rows(), baseline_archive_bytes=178493, baseline_S=0.192
    )
    assert isinstance(f, ParetoCarrierFitFrontier)
    with pytest.raises((AttributeError, TypeError)):
        f.selected_key = "mutated"  # type: ignore[misc]
