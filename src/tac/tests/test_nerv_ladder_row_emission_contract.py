# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.analysis.nerv_ladder_row_emission_contract import (
    SCHEMA,
    build_nerv_ladder_row_emission_contract,
)


def test_contract_is_false_authority_and_family_scoped() -> None:
    payload = build_nerv_ladder_row_emission_contract(
        source_parity_contract={"family_rows": []},
        row_harvests=[],
    )

    assert payload["schema"] == SCHEMA
    assert payload["families"] == ("snerv", "hinerv")
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_trained_ladder_row_emission"] is False
    assert {
        "source_bound_modelsize_control",
        "archive_byte_custody",
        "receiver_replay_proof",
    }.issubset(
        {row["group_id"] for row in payload["generic_required_field_groups"]}
    )
    family_rows = {row["family"]: row for row in payload["family_rows"]}
    assert "snerv_official_source_controls" in {
        row["group_id"] for row in family_rows["snerv"]["required_field_groups"]
    }
    assert "hinerv_official_source_controls" in {
        row["group_id"] for row in family_rows["hinerv"]["required_field_groups"]
    }


def test_snerv_local_smoke_harvest_blocks_trained_ladder_emission() -> None:
    payload = build_nerv_ladder_row_emission_contract(
        families=("snerv",),
        source_parity_contract={
            "family_rows": [
                {"family": "snerv", "blockers": ["snerv_mfu_hfr_stride_stack_missing"]}
            ]
        },
        row_harvests=[
            {
                "schema": "nerv_receiver_closed_ladder_row_harvest.v1",
                "carrier_id": "snerv",
                "status": "receiver_closed_ladder_rows_blocked",
                "harvested_row_count": 30,
                "full_scope_row_count": 0,
                "receiver_proof_row_count": 0,
                "modelsize_present_row_count": 0,
                "ladder_candidate_row_count": 0,
            }
        ],
    )

    assert payload["ready_for_trained_ladder_row_emission"] is False
    row = payload["family_rows"][0]
    assert row["observed_harvest_summary"]["harvested_row_count"] == 30
    assert row["observed_harvest_summary"]["full_scope_row_count"] == 0
    assert "source_parity:snerv_mfu_hfr_stride_stack_missing" in row["blockers"]
    assert (
        "emission_gap:snerv:fewer_than_two_full600_rows_observed"
        in payload["blockers"]
    )
    assert (
        "emission_gap:snerv:fewer_than_two_modelsize_or_fc_dim_rows_observed"
        in payload["blockers"]
    )
    assert (
        "emission_gap:snerv:fewer_than_two_receiver_proof_rows_observed"
        in payload["blockers"]
    )


def test_hinerv_prefilter_harvest_blocks_receiver_proof_and_modelsize() -> None:
    payload = build_nerv_ladder_row_emission_contract(
        families=("hi_nerv",),
        source_parity_contract={"family_rows": [{"family": "hi_nerv", "blockers": []}]},
        row_harvests=[
            {
                "schema": "nerv_receiver_closed_ladder_row_harvest.v1",
                "carrier_id": "hinerv",
                "status": "receiver_closed_ladder_rows_blocked",
                "source_artifact_path": ".omx/research/hinerv_prefilter.json",
                "harvested_row_count": 2,
                "full_scope_row_count": 1,
                "receiver_proof_row_count": 0,
                "modelsize_present_row_count": 0,
                "ladder_candidate_row_count": 0,
            }
        ],
    )

    assert payload["families"] == ("hinerv",)
    row = payload["family_rows"][0]
    assert row["source_parity_blockers"] == []
    assert row["observed_harvest_summary"]["full_scope_row_count"] == 1
    assert row["observed_harvest_summary"]["source_paths"] == [
        ".omx/research/hinerv_prefilter.json"
    ]
    assert (
        "emission_gap:hinerv:fewer_than_two_full600_rows_observed"
        in payload["blockers"]
    )
    assert (
        "emission_gap:hinerv:fewer_than_two_receiver_proof_rows_observed"
        in payload["blockers"]
    )
    assert (
        "emission_gap:hinerv:fewer_than_two_modelsize_or_fc_dim_rows_observed"
        in payload["blockers"]
    )


def test_source_parity_family_aliases_are_normalized() -> None:
    payload = build_nerv_ladder_row_emission_contract(
        families=("hinerv",),
        source_parity_contract={
            "family_rows": [
                {
                    "family": "hi_nerv",
                    "blockers": ["hi_nerv_bitstream_roundtrip_missing"],
                }
            ]
        },
        row_harvests=[],
    )

    assert (
        "source_parity:hi_nerv_bitstream_roundtrip_missing" in payload["blockers"]
    )


def test_ready_contract_requires_source_parity_and_two_receiver_closed_rows() -> None:
    payload = build_nerv_ladder_row_emission_contract(
        families=("snerv",),
        source_parity_contract={"family_rows": [{"family": "snerv", "blockers": []}]},
        row_harvests=[
            {
                "schema": "nerv_receiver_closed_ladder_row_harvest.v1",
                "carrier_id": "snerv",
                "status": "receiver_closed_ladder_rows_ready",
                "harvested_row_count": 2,
                "full_scope_row_count": 2,
                "receiver_proof_row_count": 2,
                "modelsize_present_row_count": 2,
                "ladder_candidate_row_count": 2,
            }
        ],
    )

    assert payload["ready_for_trained_ladder_row_emission"] is True
    assert payload["blockers"] == []
    row = payload["family_rows"][0]
    assert row["ready_for_trained_ladder_row_emission"] is True
    assert row["blockers"] == []
