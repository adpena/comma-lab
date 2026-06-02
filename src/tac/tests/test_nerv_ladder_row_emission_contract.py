# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.analysis.nerv_ladder_row_emission_contract import (
    SCHEMA,
    build_nerv_ladder_row_emission_contract,
)
from tools.build_nerv_ladder_row_emission_contract import main as tool_main


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


def test_hinerv_archive_size_ladder_counts_receiver_proof_but_not_ladder_authority() -> None:
    payload = build_nerv_ladder_row_emission_contract(
        families=("hi_nerv",),
        source_parity_contract={"family_rows": [{"family": "hi_nerv", "blockers": []}]},
        row_harvests=[
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "family": "hi_nerv",
                "report_path": ".omx/research/hinerv_archive_size_ladder.json",
                "num_pairs": 600,
                "archive_export_backend_counts": {"pytorch_portable_fallback": 1},
                "archive_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "archive_bytes": 134908,
                        "archive_sha256": "5" * 64,
                        "num_parameters": 94764,
                        "runtime_consumption_proof_ready": True,
                        "backend_claim_blockers": ["archive_export_backend_not_mlx"],
                        "blockers": [
                            "hinerv_archive_size_row_has_no_nonrate_score",
                            "archive_export_backend_not_mlx",
                        ],
                    }
                ],
                "blockers": [
                    "hinerv_archive_size_ladder_false_authority_no_nonrate_score",
                    "archive_export_backend_not_mlx",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    )

    row = payload["family_rows"][0]
    observed = row["observed_harvest_summary"]
    assert observed["source_paths"] == [
        ".omx/research/hinerv_archive_size_ladder.json"
    ]
    assert observed["harvested_row_count"] == 1
    assert observed["full_scope_row_count"] == 1
    assert observed["local_receiver_replay_row_count"] == 1
    assert observed["receiver_proof_row_count"] == 1
    assert observed["modelsize_present_row_count"] == 1
    assert observed["ladder_candidate_row_count"] == 0
    assert observed["archive_export_backend_counts"] == {
        "pytorch_portable_fallback": 1
    }
    assert "archive_export_backend_not_mlx" in observed["row_blockers"]
    assert "no_harvested_rows_observed" not in row["emission_gap_ids"]
    assert (
        "emission_gap:hinerv:fewer_than_two_ladder_candidate_rows_observed"
        in payload["blockers"]
    )
    assert payload["ready_for_trained_ladder_row_emission"] is False


def test_snerv_receiver_packet_probe_is_hash_checked_but_not_ladder_authority(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "0000_explicit_fp163.snar"
    packet_bytes = b"snerv receiver packet"
    packet_path.write_bytes(packet_bytes)
    packet_sha = hashlib.sha256(packet_bytes).hexdigest()

    payload = build_nerv_ladder_row_emission_contract(
        families=("snerv",),
        source_parity_contract={"family_rows": [{"family": "snerv", "blockers": []}]},
        packet_probe_artifacts=[
            {
                "schema": "snerv_decoder_mode_assignment_probe.v1",
                "source_artifact_path": ".omx/research/snerv_packet_probe.json",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "candidates": [
                    {
                        "label": "explicit_fp163",
                        "receiver_archive_replay_verified": True,
                        "receiver_archive_packet_export": {
                            "schema": "snerv_receiver_packet_export.v1",
                            "kind": "snerv_receiver_packet_snar1_not_contest_archive_zip",
                            "path": packet_path.as_posix(),
                            "bytes": len(packet_bytes),
                            "sha256": packet_sha,
                            "expected_sha256": packet_sha,
                            "contest_archive_zip": False,
                        },
                        "blockers": [
                            "full_600_pair_receiver_replay_missing",
                            "not_packaged_as_contest_archive_zip",
                        ],
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    )

    row = payload["family_rows"][0]
    observed = row["observed_harvest_summary"]
    packet_summary = payload["receiver_packet_probe_summaries"][0]
    packet_row = packet_summary["packet_rows"][0]
    assert packet_summary["carrier_id"] == "snerv"
    assert observed["receiver_packet_probe_payload_count"] == 1
    assert observed["receiver_packet_export_verified_count"] == 1
    assert observed["local_receiver_replay_row_count"] == 1
    assert observed["receiver_proof_row_count"] == 0
    assert observed["ladder_candidate_row_count"] == 0
    assert packet_row["file_exists"] is True
    assert packet_row["sha256_matches_export"] is True
    assert packet_row["bytes_match_export"] is True
    assert packet_row["contest_archive_zip"] is False
    assert payload["ready_for_trained_ladder_row_emission"] is False
    assert (
        "emission_gap:snerv:fewer_than_two_full600_rows_observed"
        in payload["blockers"]
    )
    assert (
        "emission_gap:snerv:fewer_than_two_receiver_proof_rows_observed"
        in payload["blockers"]
    )


def test_ladder_row_contract_cli_accepts_packet_probe_json(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.snar"
    packet_bytes = b"snerv packet cli"
    packet_path.write_bytes(packet_bytes)
    packet_sha = hashlib.sha256(packet_bytes).hexdigest()
    probe_path = tmp_path / "probe.json"
    output_path = tmp_path / "contract.json"
    probe_path.write_text(
        json.dumps(
            {
                "schema": "snerv_decoder_mode_assignment_probe.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "candidates": [
                    {
                        "label": "explicit_fp163",
                        "receiver_archive_replay_verified": True,
                        "receiver_archive_packet_export": {
                            "path": packet_path.as_posix(),
                            "bytes": len(packet_bytes),
                            "sha256": packet_sha,
                            "expected_sha256": packet_sha,
                            "contest_archive_zip": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--family",
            "snerv",
            "--packet-probe-json",
            str(probe_path),
            "--out",
            str(output_path),
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["receiver_packet_probe_summaries"][0][
        "receiver_packet_export_verified_count"
    ] == 1
    assert payload["score_claim"] is False
    assert payload["ready_for_trained_ladder_row_emission"] is False


def test_packet_probe_without_carrier_or_schema_stays_unassigned(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.snar"
    packet_bytes = b"unknown carrier packet"
    packet_path.write_bytes(packet_bytes)
    packet_sha = hashlib.sha256(packet_bytes).hexdigest()

    payload = build_nerv_ladder_row_emission_contract(
        families=("snerv",),
        source_parity_contract={"family_rows": [{"family": "snerv", "blockers": []}]},
        packet_probe_artifacts=[
            {
                "source_artifact_path": ".omx/research/packet_probe.json",
                "n_pairs": 1,
                "candidates": [
                    {
                        "receiver_archive_packet_export": {
                            "path": packet_path.as_posix(),
                            "bytes": len(packet_bytes),
                            "sha256": packet_sha,
                            "expected_sha256": packet_sha,
                            "contest_archive_zip": False,
                        },
                    }
                ],
            }
        ],
    )

    packet_summary = payload["receiver_packet_probe_summaries"][0]
    observed = payload["family_rows"][0]["observed_harvest_summary"]
    assert packet_summary["carrier_id"] == "unknown"
    assert "packet_probe_carrier_missing_or_unknown" in packet_summary["blockers"]
    assert observed["receiver_packet_probe_payload_count"] == 0
    assert observed["receiver_packet_export_verified_count"] == 0
    assert observed["local_receiver_replay_row_count"] == 0


def test_packet_probe_without_declared_bytes_or_sha_blocks_custody(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "packet.snar"
    packet_path.write_bytes(b"packet")

    payload = build_nerv_ladder_row_emission_contract(
        families=("snerv",),
        source_parity_contract={"family_rows": [{"family": "snerv", "blockers": []}]},
        packet_probe_artifacts=[
            {
                "schema": "snerv_decoder_mode_assignment_probe.v1",
                "n_pairs": 1,
                "candidates": [
                    {
                        "receiver_archive_packet_export": {
                            "path": packet_path.as_posix(),
                            "contest_archive_zip": False,
                        },
                    }
                ],
            }
        ],
    )

    packet_summary = payload["receiver_packet_probe_summaries"][0]
    packet_row = packet_summary["packet_rows"][0]
    assert packet_summary["receiver_packet_export_verified_count"] == 0
    assert packet_row["bytes_match_export"] is False
    assert packet_row["sha256_matches_export"] is False
    assert "packet_export:receiver_packet_export_bytes_missing" in packet_summary[
        "blockers"
    ]
    assert "packet_export:receiver_packet_export_sha256_missing" in packet_summary[
        "blockers"
    ]


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
