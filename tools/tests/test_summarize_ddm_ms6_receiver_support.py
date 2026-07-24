# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.summarize_ddm_ms6_receiver_support import (
    EXPECTED_BASE_SHA256,
    _assigned_bucket_summary,
    _candidate_coordinate_families,
    _coordinate_derivation,
    _distribution,
    _g3_coverage,
    _load_checkpoints,
    _signed_asymmetry,
)


def test_signed_asymmetry_preserves_direction_and_zero_support() -> None:
    assert _signed_asymmetry(10, 30) == pytest.approx(0.5)
    assert _signed_asymmetry(30, 10) == pytest.approx(-0.5)
    assert _signed_asymmetry(0, 0) == 0.0


def test_distribution_reports_direction_counts_and_quantiles() -> None:
    value = _distribution([-1.0, 0.0, 0.5, 1.0])
    assert value["count"] == 4
    assert value["negative_dominant_count"] == 1
    assert value["exact_tie_count"] == 1
    assert value["positive_dominant_count"] == 2
    assert value["median"] == pytest.approx(0.25)


def test_g3_coverage_requires_exact_pair_in_joined_pair_ids() -> None:
    rows = [
        {
            "bucket_id": "a",
            "pf2_membership_pair_ids": [1, 2],
            "pair_ids": [1],
        },
        {
            "bucket_id": "b",
            "pf2_membership_pair_ids": [1],
            "pair_ids": [1],
        },
        {
            "bucket_id": "c",
            "pf2_membership_pair_ids": [2],
            "pair_ids": [],
        },
    ]
    value = _g3_coverage(rows, [1, 2])
    assert value["coverage_proven"] is False
    assert value["fully_joined_pair_count"] == 1
    assert value["missing_blocks"] == [{"pair_id": 2, "bucket_id": "a"}, {"pair_id": 2, "bucket_id": "c"}]


def test_g3_coverage_ignores_buckets_without_hard_pair_membership() -> None:
    rows = [
        {
            "bucket_id": "a",
            "pf2_membership_pair_ids": [7],
            "pair_ids": [7],
        },
        {
            "bucket_id": "unrelated",
            "pf2_membership_pair_ids": [9],
            "pair_ids": [],
        },
    ]
    value = _g3_coverage(rows, [7])
    assert value["coverage_proven"] is True
    assert value["missing_block_count"] == 0


def test_candidate_coordinate_families_are_derived_from_typed_key() -> None:
    assert _candidate_coordinate_families(
        {
            "class_pair": "Lane--Movable",
            "class_stratum": "boundary",
            "g4_temporal_class": "TRANSIENT",
        }
    ) == [
        "LANE_PROGRAM_BAND_COORDINATE",
        "BOUNDED_G1_POLYGON_COORDINATE",
        "PAIR_LOCAL_POST_SOLVE_CORRECTION",
        "EVENT_LOCAL_SKELETON_BOUNDARY_PRODUCTION",
    ]


def test_assigned_bucket_summary_labels_probe_event_incidence() -> None:
    value = _assigned_bucket_summary(
        {
            "bucket_id": "a",
            "assignment_status": "RECOVERED_COMPLETE",
            "receiver_actuator_ids": ["x", "y"],
            "direction_ids": ["NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM"],
            "pair_ids": [1, 2],
            "measured_probe_assignments": [
                {"perturbed_event_count": 5},
                {"perturbed_event_count": 7},
            ],
        }
    )
    assert value["probe_event_incidence_count"] == 12
    assert "not unique-event cardinality" in value["probe_event_incidence_semantics"]


def test_rg2_coordinate_derivation_separates_unreachable_birth_from_finer_amplitude() -> None:
    rows = [
        {
            "bucket_id": "boundary",
            "atlas_key": {"class_stratum": "boundary"},
        },
        {
            "bucket_id": "cell",
            "atlas_key": {"class_stratum": "cell"},
        },
    ]
    coverage = {
        "missing_blocks": [
            {"pair_id": 1, "bucket_id": "boundary"},
            {"pair_id": 2, "bucket_id": "cell"},
        ]
    }
    assignment = {
        "rows": [
            {
                "pair_id": 1,
                "bucket_id": "boundary",
                "causal_join_status": "UNREACHABLE_NO_SHA_BOUND_RECEIVER_CLASS_PAIR_SUPPORT",
                "receiver_actuator_id": None,
                "receiver_derived_row_band": None,
            },
            {
                "pair_id": 2,
                "bucket_id": "cell",
                "causal_join_status": "READY_FOR_SIGNED_PROBE",
                "receiver_actuator_id": "rg2.skeleton.pair002.class0_1.cell.transient.band03",
                "receiver_derived_row_band": 3,
            },
        ]
    }

    value = _coordinate_derivation(rows, coverage, rg2_assignment=assignment)

    assert value["verdict_scope"] == "INSTANCE_EXTENDED_GRAMMAR_RG2"
    assert value["residual"][0]["candidate_coordinate_families"] == [
        "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION"
    ]
    assert value["residual"][1]["candidate_coordinate_families"] == [
        "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK"
    ]


def test_rg3_coordinate_derivation_reports_terminal_blocker_without_rg4() -> None:
    rows = [
        {
            "bucket_id": "cell",
            "atlas_key": {"class_stratum": "cell"},
        }
    ]
    coverage = {"missing_blocks": [{"pair_id": 2, "bucket_id": "cell"}]}
    rg3_assignment = {
        "rows": [
            {
                "pair_id": 2,
                "bucket_id": "cell",
                "family": "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK",
                "receiver_actuator_ids": [
                    "rg3.fisher_stratum.pair002.class0_1.cell.transient.band03.fine02.mag1",
                    "rg3.fisher_stratum.pair002.class0_1.cell.transient.band03.fine02.mag2",
                ],
            }
        ]
    }

    value = _coordinate_derivation(
        rows,
        coverage,
        rg2_assignment=None,
        rg3_assignment=rg3_assignment,
    )

    assert value["verdict_scope"] == "INSTANCE_EXTENDED_GRAMMAR_RG3"
    assert value["residual"][0]["candidate_coordinate_families"] == []
    assert value["next_authorized_family_status"] == "NO_RG4_AUTHORIZED"


def test_load_checkpoints_selects_assignment_bound_revision_across_roots(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "old", tmp_path / "new"]
    for root in roots:
        root.mkdir()
    key = ("j2.lane.line0.width_bias_q8", "POSITIVE_ONE_QUANTUM")
    expected_sha = ""
    for index, (root, status) in enumerate(
        zip(roots, ("MEASURED_ARGMAX_INVARIANT", "MEASURED_ARGMAX_PERTURBATION"), strict=True)
    ):
        row = {
            "schema": "ddm_ms6_receiver_support_probe_checkpoint.v2",
            "base_archive_sha256": EXPECTED_BASE_SHA256,
            "threads": 4,
            "seed": 1234,
            "deterministic_algorithms": True,
            "score_claim": False,
            "receiver_actuator_id": key[0],
            "direction_id": key[1],
            "event_artifact": None,
            "status": status,
            "revision": index,
        }
        path = root / "probe.json"
        path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        if index == 1:
            expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    rows, custody = _load_checkpoints(
        roots,
        expected_checkpoint_sha256={key: expected_sha},
    )

    assert rows[key]["status"] == "MEASURED_ARGMAX_PERTURBATION"
    assert custody["completed_probe_count"] == 1
    assert custody["superseded_checkpoint_count"] == 1
