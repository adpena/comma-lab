"""Adversarial tests for the EV1 exclusive evidence-join schema."""

from __future__ import annotations

import copy
from itertools import pairwise

import pytest

from tac.ddm_campaign_evidence_join import (
    G4_CLASSES,
    METRIC_ID,
    PAIR_COUNT,
    RD1_BUCKET_COUNT,
    RD1_SCHEMA,
    SCHEMA,
    STRATA,
    V19_SCHEMA,
    VISIBILITIES,
    CampaignEvidenceJoinError,
    allocate_exclusive_byte_homes,
    bucket_key,
    derive_g4_reuse_profiles,
    pack_histogram_rows,
    unpack_histogram_rows,
    validate_campaign_evidence_join,
)


def _fixture() -> dict:
    bucket_rows = []
    profiles = derive_g4_reuse_profiles(
        recurrence_k_distribution={
            "exact_k": [
                {"k": 1, "flip_event_mass": 7, "locus_count": 7},
                {"k": 2, "flip_event_mass": 10, "locus_count": 5},
                {"k": 4, "flip_event_mass": 12, "locus_count": 3},
            ]
        },
        xi_track_lengths=[2],
    )
    for dual in range(1, 4):
        keys = [
            (dual, stratum, visibility, g4) for stratum in STRATA for visibility in VISIBILITIES for g4 in G4_CLASSES
        ]
        homes = allocate_exclusive_byte_homes(
            delta_counted_bytes=(16, 962, 409_388_124)[dual - 1],
            weights={key: (1.0 if index == 0 else 0.0) for index, key in enumerate(keys)},
        )
        for key in keys:
            _, stratum, visibility, g4 = key
            hist = [0] * 256
            hist[1] = 1 if visibility in {"seg-visible", "pose-visible"} else 0
            home = homes[key]
            profile = profiles[g4]
            bucket_rows.append(
                {
                    "dual_index": dual,
                    "stratum": stratum,
                    "scorer_visibility": visibility,
                    "g4_temporal_class": g4,
                    "receiver_uint8_abs_step_histogram": hist,
                    "receiver_changed_channel_values": sum(hist),
                    "receiver_uint8_abs_step_sum": sum(index * value for index, value in enumerate(hist)),
                    "delta_D_dimension": 0.0,
                    **home,
                    **profile,
                    "amortized_bytes_per_frame": (
                        home["delta_counted_bytes_dimension"]
                        * profile["k_denominator"]
                        / profile["k_numerator"]
                    ),
                    "lambda_bytes_per_D_dimension": None,
                    "pricing_owner": "ddm_ms2r_tolerance_capped_solve_r2",
                    "score_claim": False,
                }
            )
    edge_summaries = []
    for dual, delta in enumerate((16, 962, 409_388_124), start=1):
        assigned = sum(row["delta_counted_bytes_dimension"] for row in bucket_rows if row["dual_index"] == dual)
        edge_summaries.append(
            {
                "dual_index": dual,
                "delta_counted_bytes": delta,
                "assigned_counted_bytes": assigned,
                "counted_exactly_once": True,
                "accounting_home_not_physical_zip_separability": True,
            }
        )
    return {
        "schema": SCHEMA,
        "run_id": "fixture_ev1",
        "pair_count": PAIR_COUNT,
        "metric_custody": {
            "metric_id": METRIC_ID,
            "margin_fisher_gram_bound": True,
            "composite_R_hessian_bound": True,
            "euclidean_naive_rows_admitted": False,
        },
        "implementation_custody": {
            "git_head_at_measurement": ("UNCOMMITTED_DELEGATED_WORKTREE; LANDING_COMMIT_REQUIRED"),
            "source_files": [{"sha256": "a" * 64}, {"sha256": "b" * 64}],
        },
        "rd1_endpoint_remeasurement": {
            "status": "FRESH_CURRENT_RECEIVER_REPLAY_MEASURED",
            "prior_menu_score_batches_reused": False,
            "receiver_scope": (
                "EXACT_MEASUREMENT_HARNESS_PAYLOAD_DECODE_THROUGH_UINT8_R; "
                "MENU1_PAYLOAD_ENDPOINTS_ARE_NOT_CONTEST_ARCHIVES"
            ),
        },
        "v19_pair_join": {
            "schema": V19_SCHEMA,
            "rows": [
                {
                    "source_pair_id": pair_id,
                    "receiver_closed": True,
                    "per_pair_byte_allocation": None,
                    "score_claim": False,
                }
                for pair_id in range(PAIR_COUNT)
            ],
            "shared_rate_home": {
                "home_id": "v19_n600_archive_delta",
                "delta_counted_bytes": 1588,
                "counted_exactly_once": True,
                "per_pair_allocation": None,
                "status": "MEASURED_EXACT_GLOBAL_ARCHIVE_DELTA_HOME",
            },
        },
        "rd1_evidence": {
            "schema": RD1_SCHEMA,
            "bucket_rows": bucket_rows,
            "edge_summaries": edge_summaries,
            "amortization_custody": {
                "profiles": profiles,
                "single_owner_across_reach": True,
                "shared_clip_bucket_count": 0,
                "shared_clip_absence_reason": "NO_AGGREGATED_G4_BUCKET_IS_EXCLUSIVELY_K_EQ_600",
                "application_scope": (
                    "G4_CLASS_LEVEL_REUSE_PRIOR_APPLIED_TO_TYPED_HOMES_NOT_EDGE_SPECIFIC_K_REMEASUREMENT"
                ),
            },
            "histogram_coder": {
                "codec": "BROTLI_Q11",
                "parse_back_identical": True,
            },
        },
        "free_interpreter_custody": {
            "generic_decoder_and_transport_code_counted_bytes": 0,
            "irreducible_video_statistic_bytes_counted": True,
            "xi_video_parameters_free": False,
            "independent_physical_bev_claim": False,
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "pointer_moved": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }


def test_complete_receipt_closes_both_campaign_evidence_gaps() -> None:
    counts = validate_campaign_evidence_join(_fixture())
    assert counts == {
        "exact_pair_rows": 600,
        "required_pair_rows": 600,
        "missing_pair_rows": 0,
        "typed_dimension_rows": 162,
        "dimension_byte_homes": 162,
        "shared_across_frame_byte_homes": 108,
        "per_frame_byte_homes": 54,
        "shared_clip_byte_homes": 0,
        "receiver_uint8_histograms": 162,
    }


def test_largest_remainder_is_exact_disjoint_and_deterministic() -> None:
    keys = {
        (1, "Road", "seg-visible", "STATIC_IN_IMAGE"): 2.0,
        (1, "Lane", "seg-visible", "TRANSIENT"): 1.0,
        (1, "POSE6_GLOBAL", "pose-visible", "TRANSIENT"): 1.0,
    }
    homes = allocate_exclusive_byte_homes(delta_counted_bytes=7, weights=keys)
    assert sum(row["delta_counted_bytes_dimension"] for row in homes.values()) == 7
    assert homes == allocate_exclusive_byte_homes(delta_counted_bytes=7, weights=keys)
    intervals = sorted(interval for row in homes.values() for interval in row["byte_home_ranges"])
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 7
    assert all(left[1] == right[0] for left, right in pairwise(intervals))


def test_histogram_binary_roundtrip_is_exact() -> None:
    receipt = _fixture()
    rows = receipt["rd1_evidence"]["bucket_rows"]
    packed = pack_histogram_rows(rows)
    keys = sorted(bucket_key(row) for row in rows)
    restored = unpack_histogram_rows(packed, keys)
    assert restored == {bucket_key(row): row["receiver_uint8_abs_step_histogram"] for row in rows}


def test_duplicate_pair_id_and_per_pair_rate_allocation_fail_closed() -> None:
    receipt = _fixture()
    receipt["v19_pair_join"]["rows"][1]["source_pair_id"] = 0
    with pytest.raises(CampaignEvidenceJoinError, match=r"0\.\.599"):
        validate_campaign_evidence_join(receipt)
    receipt = _fixture()
    receipt["v19_pair_join"]["rows"][7]["per_pair_byte_allocation"] = 3
    with pytest.raises(CampaignEvidenceJoinError, match="receiver/rate"):
        validate_campaign_evidence_join(receipt)


def test_byte_home_gap_or_duplicate_fails_closed() -> None:
    receipt = _fixture()
    row = next(row for row in receipt["rd1_evidence"]["bucket_rows"] if row["delta_counted_bytes_dimension"] > 0)
    row["byte_home_ranges"][0][0] += 1
    with pytest.raises(CampaignEvidenceJoinError, match=r"interval size|overlap|gap"):
        validate_campaign_evidence_join(receipt)


def test_histogram_moment_drift_and_price_trespass_fail_closed() -> None:
    receipt = _fixture()
    receipt["rd1_evidence"]["bucket_rows"][0]["receiver_changed_channel_values"] += 1
    with pytest.raises(CampaignEvidenceJoinError, match="moments"):
        validate_campaign_evidence_join(receipt)


def test_amortized_home_scope_and_rate_drift_fail_closed() -> None:
    receipt = _fixture()
    row = receipt["rd1_evidence"]["bucket_rows"][0]
    row["scope"] = "shared_clip"
    with pytest.raises(CampaignEvidenceJoinError, match="scope"):
        validate_campaign_evidence_join(receipt)
    receipt = _fixture()
    receipt["rd1_evidence"]["bucket_rows"][0]["amortized_bytes_per_frame"] += 1.0
    with pytest.raises(CampaignEvidenceJoinError, match="amortized bytes"):
        validate_campaign_evidence_join(receipt)
    receipt = _fixture()
    receipt["rd1_evidence"]["bucket_rows"][0]["lambda_bytes_per_D_dimension"] = 4.0
    with pytest.raises(CampaignEvidenceJoinError, match="ms2r"):
        validate_campaign_evidence_join(receipt)


def test_missing_typed_cell_is_not_silently_accepted() -> None:
    receipt = _fixture()
    receipt["rd1_evidence"]["bucket_rows"].pop()
    with pytest.raises(CampaignEvidenceJoinError, match="162"):
        validate_campaign_evidence_join(receipt)


def test_fixture_itself_has_exact_cube_size() -> None:
    assert len(_fixture()["rd1_evidence"]["bucket_rows"]) == RD1_BUCKET_COUNT


def test_zero_total_weight_is_refused() -> None:
    with pytest.raises(CampaignEvidenceJoinError, match="positive"):
        allocate_exclusive_byte_homes(
            delta_counted_bytes=1,
            weights={(1, "Road", "seg-visible", "TRANSIENT"): 0.0},
        )


def test_receipt_authority_firewall_is_not_mutable_by_copy() -> None:
    receipt = copy.deepcopy(_fixture())
    receipt["pointer_moved"] = True
    with pytest.raises(CampaignEvidenceJoinError, match="pointer_moved"):
        validate_campaign_evidence_join(receipt)
    receipt = copy.deepcopy(_fixture())
    receipt["rd1_endpoint_remeasurement"]["prior_menu_score_batches_reused"] = True
    with pytest.raises(CampaignEvidenceJoinError, match="score/camera"):
        validate_campaign_evidence_join(receipt)
