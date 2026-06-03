# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path

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


def test_path_only_receiver_proof_report_does_not_unlock_full600_row() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_path_only_receiver_report.v1",
                "axis_tag": "[contest-CPU]",
                "n_pairs": 600,
                "rows": [
                    {
                        "row_id": "path_only",
                        "modelsize_mparams": 0.04,
                        "fc_dim": 16,
                        "archive_bytes": 42_000,
                        "archive_sha256": "8" * 64,
                        "receiver_proof_report_paths": [
                            "/ssd/receiver_proof_path_only.json"
                        ],
                        "d_seg": 0.004,
                        "d_pose": 0.002,
                        "accepted": True,
                    }
                ],
            }
        ],
        carrier_id="snerv",
    )

    row = payload["harvested_rows"][0]
    assert row["sample_scope"] == "full600_or_better"
    assert row["local_receiver_archive_replay_verified"] is False
    assert row["receiver_proof_passed"] is False
    assert row["receiver_closed"] is False
    assert "receiver_replay_or_contract_missing" in row["harvest_blockers"]
    assert payload["ladder_candidate_row_count"] == 0
    assert payload["ready_for_receiver_closed_modelsize_ladder"] is False


def test_advisory_axis_full600_rows_do_not_unlock_ladder_candidates(
    tmp_path: Path,
) -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_advisory_full600.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 600,
                "rows": [
                    _full_row(tmp_path, "tiny", 0.04, 16, 42_000, 0.004, 0.002, "8"),
                    _full_row(tmp_path, "small", 0.08, 24, 80_000, 0.003, 0.002, "9"),
                ],
            }
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    assert payload["receiver_proof_row_count"] == 0
    assert payload["ladder_candidate_row_count"] == 0
    row = payload["harvested_rows"][0]
    assert row["source_axis_receiver_closed_authority"] is False
    assert row["receiver_closed"] is False
    assert (
        "source_axis_not_receiver_closed_contest_authority"
        in row["harvest_blockers"]
    )


def test_true_authority_flags_block_harvest_candidates(tmp_path: Path) -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_authority_leak.v1",
                "axis_tag": "[contest-CPU]",
                "n_pairs": 600,
                "rows": [
                    {
                        **_full_row(
                            tmp_path,
                            "tiny",
                            0.04,
                            16,
                            42_000,
                            0.004,
                            0.002,
                            "8",
                        ),
                        "promotion_eligible": True,
                    },
                    _full_row(tmp_path, "small", 0.08, 24, 80_000, 0.003, 0.002, "9"),
                ],
            }
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    rows = {row["row_id"]: row for row in payload["harvested_rows"]}
    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    assert payload["ladder_candidate_row_count"] == 1
    assert "source_authority_flag_true:promotion_eligible" in rows["tiny"][
        "harvest_blockers"
    ]


def test_boolean_only_full600_rows_do_not_unlock_ladder_candidates() -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_boolean_only_full600.v1",
                "axis_tag": "[contest-CPU]",
                "n_pairs": 600,
                "rows": [
                    _boolean_full_row("tiny", 0.04, 16, 42_000, 0.004, 0.002, "8"),
                    _boolean_full_row("small", 0.08, 24, 80_000, 0.003, 0.002, "9"),
                ],
            }
        ],
        carrier_id="snerv",
    )

    rows = {row["row_id"]: row for row in payload["harvested_rows"]}
    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    assert payload["receiver_proof_row_count"] == 0
    assert payload["ladder_candidate_row_count"] == 0
    assert rows["tiny"]["local_receiver_archive_replay_verified"] is True
    assert rows["tiny"]["receiver_proof_identity_bound"] is False
    assert "receiver_proof_identity_missing" in rows["tiny"]["harvest_blockers"]
    assert "receiver_proof_path_missing" in rows["tiny"]["harvest_blockers"]


def test_two_full600_modelsize_rows_are_ready_for_ladder_input(tmp_path: Path) -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[contest-CPU]",
                "n_pairs": 600,
                "rows": [
                    _full_row(tmp_path, "tiny", 0.04, 16, 42_000, 0.004, 0.002, "b"),
                    _full_row(tmp_path, "small", 0.08, 24, 80_000, 0.003, 0.002, "c"),
                ],
            }
        ],
        carrier_id="snerv",
        repo_root=tmp_path,
    )

    assert payload["status"] == "receiver_closed_ladder_rows_ready"
    assert payload["receiver_proof_row_count"] == 2
    assert payload["ladder_candidate_row_count"] == 2
    assert payload["ready_for_receiver_closed_modelsize_ladder"] is True
    assert all(row["receiver_proof_passed"] for row in payload["harvested_rows"])
    assert all(row["sample_scope"] == "full600_or_better" for row in payload["harvested_rows"])
    assert payload["score_claim"] is False


def test_family_mismatch_is_preserved_in_harvest_rows(tmp_path: Path) -> None:
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 600,
                "rows": [
                    _full_row(
                        tmp_path,
                        "snerv_row",
                        0.04,
                        16,
                        42_000,
                        0.004,
                        0.002,
                        "d",
                    ),
                ],
            }
        ],
        carrier_id="hinerv",
        repo_root=tmp_path,
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


def test_hinerv_archive_size_ladder_archive_rows_are_harvested_as_capacity_axis(
    tmp_path: Path,
) -> None:
    tiny_proof = _receiver_proof(tmp_path, "hi_nerv_local_tiny", 134_842, "f")
    small_proof = _receiver_proof(tmp_path, "hi_nerv_local_small", 247_815, "1")
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "axis_tag": "[contest-CPU]",
                "family": "hi_nerv",
                "num_pairs": 600,
                "archive_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "family": "hi_nerv",
                        "modelsize_scale": 0.25,
                        "modelsize_control_contract": {
                            "schema": "nerv_modelsize_control_contract.v1",
                            "authority_split": {
                                "schema": "nerv_modelsize_control_authority_split.v1",
                                "modelsize_mparams_semantics": (
                                    "local_nearest_parameter_count_target"
                                ),
                                "modelsize_mparams_caps_archive_zip_bytes": False,
                                "archive_byte_authority_surface": (
                                    "archive_zip_bytes_after_receiver_export_and_"
                                    "inflate_proof"
                                ),
                                "score_claim": False,
                                "ready_for_exact_eval_dispatch": False,
                            },
                        },
                        "archive_bytes": 134_842,
                        "archive_sha256": "f" * 64,
                        "runtime_consumption_proof_ready": True,
                        **tiny_proof,
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
                        **small_proof,
                        "d_seg": 0.008,
                        "d_pose": 0.002,
                        "blockers": [],
                    },
                ],
            }
        ],
        carrier_id="hi_nerv",
        repo_root=tmp_path,
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
    assert first["modelsize_authority_split"]["modelsize_mparams_semantics"] == (
        "local_nearest_parameter_count_target"
    )
    assert first["modelsize_authority_split"][
        "modelsize_mparams_caps_archive_zip_bytes"
    ] is False
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


def test_harvest_blocks_modelsize_contract_without_authority_split(
    tmp_path: Path,
) -> None:
    proof = _receiver_proof(tmp_path, "hi_nerv_ambiguous_contract", 134_842, "a")
    payload = build_nerv_receiver_closed_ladder_row_harvest(
        [
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "axis_tag": "[contest-CPU]",
                "family": "hi_nerv",
                "num_pairs": 600,
                "archive_rows": [
                    {
                        "row_id": "hi_nerv_ambiguous_contract",
                        "family": "hi_nerv",
                        "modelsize_scale": 0.25,
                        "modelsize_control_contract": {
                            "schema": "nerv_modelsize_control_contract.v1",
                            "modelsize_mparams_caps_archive_zip_bytes": False,
                        },
                        "archive_bytes": 134_842,
                        "archive_sha256": "a" * 64,
                        "runtime_consumption_proof_ready": True,
                        **proof,
                        "d_seg": 0.01,
                        "d_pose": 0.0025,
                        "blockers": [],
                    }
                ],
            }
        ],
        carrier_id="hi_nerv",
        repo_root=tmp_path,
    )

    assert payload["status"] == "receiver_closed_ladder_rows_blocked"
    row = payload["harvested_rows"][0]
    assert row["modelsize_control_contract"]["schema"] == (
        "nerv_modelsize_control_contract.v1"
    )
    assert row["modelsize_authority_split"] is None
    assert "modelsize_authority_split_missing" in row["harvest_blockers"]


def _boolean_full_row(
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


def _full_row(
    tmp_path: Path,
    row_id: str,
    modelsize: float,
    fc_dim: int,
    archive_bytes: int,
    d_seg: float,
    d_pose: float,
    sha_char: str,
) -> dict[str, object]:
    return {
        **_boolean_full_row(
            row_id,
            modelsize,
            fc_dim,
            archive_bytes,
            d_seg,
            d_pose,
            sha_char,
        ),
        **_receiver_proof(tmp_path, row_id, archive_bytes, sha_char),
    }


def _receiver_proof(
    tmp_path: Path,
    row_id: str,
    archive_bytes: int,
    sha_char: str,
) -> dict[str, object]:
    proof = tmp_path / f"{row_id}.receiver_proof.json"
    proof.write_text(
        (
            '{"schema":"snerv_inverse_steg_generated_receiver_proof.v1",'
            '"receiver_contract_satisfied":true,'
            '"runtime_consumption_proof_ready":true,'
            '"runtime_consumption_proof_passed":true,'
            f'"archive_bytes":{archive_bytes},'
            f'"archive_sha256":"{sha_char * 64}",'
            '"receiver_output_bytes":123,'
            '"expected_receiver_output_bytes":123,'
            '"blockers":[]}\n'
        ),
        encoding="utf-8",
    )
    return {
        "receiver_proof_path": proof.as_posix(),
        "receiver_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
    }
