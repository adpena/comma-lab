# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.analysis.nerv_receiver_closed_ladder_row_harvest import (
    SCHEMA,
    ReceiverRowSource,
    build_nerv_receiver_closed_ladder_row_harvest,
)


def test_local_receiver_replay_does_not_unlock_ladder_proof() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            ReceiverRowSource(
                path=".omx/research/snerv_smoke.json",
                payload={
                    "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                    "axis_tag": "[macOS-CPU advisory]",
                    "n_pairs": 4,
                    "rows": [
                        {
                            "sweep_label": "least_squares",
                            "archive_bytes_total": 1230018,
                            "receiver_archive_sha256": "a" * 64,
                            "receiver_archive_replay_verified": True,
                            "d_seg_mean_linf": 0.002,
                            "d_pose_mean_linf": 0.003,
                            "accepted": True,
                        }
                    ],
                },
            )
        ],
        carrier_id="snerv",
    )

    assert payload["schema"] == SCHEMA
    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    assert payload["harvested_row_count"] == 1
    assert payload["local_receiver_replay_row_count"] == 1
    assert payload["receiver_proof_row_count"] == 0
    row = payload["harvested_rows"][0]
    assert row["sample_scope"] == "local_pair_smoke"
    assert row["local_receiver_archive_replay_verified"] is True
    assert row["receiver_proof_passed"] is False
    assert row["receiver_closed"] is False
    assert row["archive_bytes"] == 1230018
    assert row["archive_sha256"] == "a" * 64
    assert row["nonrate_score"] is not None
    assert "local_smoke_only_not_full600_receiver_proof" in row["harvest_blockers"]
    assert "modelsize_or_fc_dim_missing" in row["harvest_blockers"]


def test_two_full600_modelsize_rows_are_ready_for_ladder_input() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[contest-CPU]",
                "n_pairs": 600,
                "rows": [
                    _full_row("tiny", 0.04, 16, 42_000, 0.004, 0.002, "b"),
                    _full_row("small", 0.08, 24, 80_000, 0.003, 0.002, "c"),
                ],
            }
        ],
        carrier_id="snerv",
    )

    assert payload["status"] == "receiver_closed_ladder_rows_ready"
    assert payload["receiver_proof_row_count"] == 2
    assert payload["ladder_candidate_row_count"] == 2
    assert payload["ready_for_receiver_closed_modelsize_ladder"] is True
    assert all(row["receiver_proof_passed"] for row in payload["harvested_rows"])
    assert all(row["sample_scope"] == "full600_or_better" for row in payload["harvested_rows"])
    assert payload["score_claim"] is False


def test_family_mismatch_is_preserved_in_harvest_rows() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 600,
                "rows": [
                    _full_row("snerv_row", 0.04, 16, 42_000, 0.004, 0.002, "d"),
                ],
            }
        ],
        carrier_id="hinerv",
    )

    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    row = payload["harvested_rows"][0]
    assert row["family"] == "snerv"
    assert "carrier_family_mismatch" in row["harvest_blockers"]
    assert payload["ladder_candidate_row_count"] == 0


def test_prefilter_profile_records_preserve_full_scope_without_receiver_proof() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "hprc_mlx_prefilter_coverage.v1",
                "required_pairs": 600,
                "profile_records": [
                    {
                        "path": "/ssd/profile.json",
                        "pair_count": 600,
                        "bytes": 19_962,
                        "sha256": "e" * 64,
                        "score_claim": False,
                        "promotion_eligible": False,
                    }
                ],
            }
        ],
        carrier_id="hinerv",
    )

    assert payload["harvested_row_count"] == 1
    assert payload["full_scope_row_count"] == 1
    assert payload["receiver_proof_row_count"] == 0
    row = payload["harvested_rows"][0]
    assert row["sample_scope"] == "full600_or_better"
    assert row["archive_bytes"] == 19_962
    assert row["archive_sha256"] == "e" * 64
    assert row["receiver_proof_passed"] is False
    assert "receiver_replay_or_contract_missing" in row["harvest_blockers"]
    assert payload["ready_for_receiver_closed_modelsize_ladder"] is False


def test_compact_runner_nested_snerv_evidence_is_harvested_without_signal_loss() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "compact_renderer_mlx_spine_runner.v1",
                "execute_family": "snerv",
                "num_pairs": 2,
                "archive_bytes": 57_892,
                "archive_sha256": "a" * 64,
                "receiver_proof_report_paths": ["/ssd/receiver_proof.json"],
                "modelsize_candidate_selection": {
                    "candidate": {
                        "family": "snerv",
                        "levels": 5,
                        "bits_per_coeff": 1.5,
                        "step_map_bits_per_coeff": 0.5,
                        "decoder_payload_codec": "int8_symmetric",
                    }
                },
                "score_aware_training": {
                    "d_seg_mean_linf": 0.273602806,
                    "d_pose_mean_linf": 196.590911865,
                    "receiver_archive_replay_verified": True,
                    "target_bits_per_coeff": 1.5,
                },
                "snerv_mlx_native_export": {
                    "receiver_proof_passed": True,
                    "receiver_contract_satisfied": True,
                },
            }
        ],
        carrier_id="snerv",
    )

    row = payload["harvested_rows"][0]
    assert row["sample_pair_count"] == 2
    assert row["sample_scope"] == "local_pair_smoke"
    assert row["archive_bytes"] == 57_892
    assert row["d_seg"] == 0.273602806
    assert row["d_pose"] == 196.590911865
    assert row["nonrate_score"] is not None
    assert row["snerv_levels"] == 5
    assert row["snerv_bits_per_coeff"] == 1.5
    assert row["snerv_step_map_bits_per_coeff"] == 0.5
    assert row["decoder_payload_codec"] == "int8_symmetric"
    assert row["local_receiver_archive_replay_verified"] is True
    assert row["receiver_proof_passed"] is False
    assert "modelsize_or_fc_dim_missing" not in row["harvest_blockers"]
    assert "receiver_replay_or_contract_missing" not in row["harvest_blockers"]
    assert "nonrate_score_or_component_distortions_missing" not in row[
        "harvest_blockers"
    ]
    assert "local_smoke_only_not_full600_receiver_proof" in row["harvest_blockers"]


def test_hinerv_archive_size_ladder_archive_rows_are_harvested_as_capacity_axis() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "family": "hi_nerv",
                "num_pairs": 600,
                "archive_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "family": "hi_nerv",
                        "modelsize_scale": 0.25,
                        "archive_bytes": 134_842,
                        "archive_sha256": "f" * 64,
                        "runtime_consumption_proof_ready": True,
                        "d_seg": 0.01,
                        "d_pose": 0.0025,
                        "blockers": [],
                    },
                    {
                        "row_id": "hi_nerv_local_small",
                        "family": "hi_nerv",
                        "modelsize_scale": 0.5,
                        "archive_bytes": 247_815,
                        "archive_sha256": "1" * 64,
                        "runtime_consumption_proof_ready": True,
                        "d_seg": 0.008,
                        "d_pose": 0.002,
                        "blockers": [],
                    },
                ],
            }
        ],
        carrier_id="hi_nerv",
    )

    assert payload["status"] == "receiver_closed_ladder_rows_ready"
    assert payload["harvested_row_count"] == 2
    assert payload["receiver_proof_row_count"] == 2
    assert payload["modelsize_present_row_count"] == 2
    assert payload["ladder_candidate_row_count"] == 2
    first = payload["harvested_rows"][0]
    assert first["row_id"] == "hi_nerv_local_tiny"
    assert first["sample_pair_count"] == 600
    assert first["sample_scope"] == "full600_or_better"
    assert first["modelsize_mparams"] is None
    assert first["modelsize_scale"] == 0.25
    assert first["fc_dim"] is None
    assert first["receiver_proof_passed"] is True
    assert first["nonrate_score"] is not None
    assert first["harvest_blockers"] == []
    assert payload["score_claim"] is False


def test_hinerv_archive_size_ladder_without_nonrate_stays_blocked() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "family": "hi_nerv",
                "num_pairs": 600,
                "archive_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "family": "hi_nerv",
                        "modelsize_scale": 0.25,
                        "archive_bytes": 134_842,
                        "archive_sha256": "f" * 64,
                        "runtime_consumption_proof_ready": True,
                    }
                ],
            }
        ],
        carrier_id="hi_nerv",
    )

    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    assert payload["modelsize_present_row_count"] == 1
    assert payload["ladder_candidate_row_count"] == 0
    row = payload["harvested_rows"][0]
    assert row["modelsize_scale"] == 0.25
    assert "nonrate_score_or_component_distortions_missing" in row[
        "harvest_blockers"
    ]


def _full_row(
    row_id: str,
    modelsize: float,
    fc_dim: int,
    archive_bytes: int,
    d_seg: float,
    d_pose: float,
    sha_char: str,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "modelsize_mparams": modelsize,
        "fc_dim": fc_dim,
        "archive_bytes": archive_bytes,
        "archive_sha256": sha_char * 64,
        "receiver_archive_replay_verified": True,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "accepted": True,
    }
