# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.master_gradient_operator_plan import (
    CandidateModificationSpec,
    build_master_gradient_operator_plan,
    build_master_gradient_operator_plan_payload,
)


def _layout_manifest() -> dict[str, object]:
    return {
        "schema": "tac_frontier_archive_layout_v1",
        "archive_path": "experiments/results/fake/archive.zip",
        "archive_bytes": 178_517,
        "archive_sha256": "a" * 64,
        "logical_layout": {
            "grammar": "a1_prefixed_hnerv_microcodec",
            "sections": [
                {
                    "name": "decoder_section_total_u32le",
                    "role": "internal_length_header",
                    "offset": 0,
                    "len": 4,
                    "sha256": "b" * 64,
                },
                {
                    "name": "decoder_blob",
                    "role": "renderer_decoder_weights",
                    "offset": 4,
                    "len": 162_164,
                    "sha256": "c" * 64,
                },
                {
                    "name": "latent_blob",
                    "role": "latent_motion_or_frame_conditioning",
                    "offset": 162_168,
                    "len": 15_387,
                    "sha256": "d" * 64,
                },
                {
                    "name": "sidecar_blob",
                    "role": "latent_sidecar_not_separate_pose_or_mask_member",
                    "offset": 177_555,
                    "len": 607,
                    "sha256": "e" * 64,
                },
            ],
        },
    }


def test_operator_plan_never_emits_raw_archive_byte_rows() -> None:
    plan = build_master_gradient_operator_plan(_layout_manifest())

    assert plan["raw_byte_gradient_valid"] is False
    assert plan["raw_archive_byte_rows_emitted"] == 0
    assert plan["operator_row_count"] == 3
    assert plan["candidate_modification_spec_count"] == 3
    assert plan["skipped_section_count"] == 1
    assert all(row["mutation_grain"] == "grammar_aware_operator" for row in plan["rows"])
    assert all(
        spec["mutation_grain"] == "grammar_aware_operator"
        for spec in plan["candidate_modification_specs"]
    )
    assert all(
        spec["raw_archive_byte_coordinates_allowed"] is False
        for spec in plan["candidate_modification_specs"]
    )
    assert {row["mutation_operator"] for row in plan["rows"]} == {
        "decoder_codec_coordinate_response",
        "latent_conditioning_codebook_sweep",
        "latent_sidecar_stream_entropy_tournament",
    }


def test_operator_plan_fails_closed_until_packet_proofs_exist() -> None:
    plan = build_master_gradient_operator_plan(_layout_manifest())

    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_provider_dispatch"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["dispatch_attempted"] is False
    assert "operator_rows_blocked_until_packet_proofs_land" in plan["blockers"]
    assert "zip_headers_not_rebuilt" in plan["blockers"]
    assert "zip_crc_not_rebuilt" in plan["blockers"]
    assert "inflate_success_not_proven" in plan["blockers"]
    assert all(row["operator_response_valid"] is False for row in plan["rows"])
    assert all(row["ready_for_operator_probe"] is False for row in plan["rows"])
    assert all(
        spec["ready_for_operator_probe"] is False
        for spec in plan["candidate_modification_specs"]
    )
    assert all(
        spec["score_claim"] is False and spec["ready_for_provider_dispatch"] is False
        for spec in plan["candidate_modification_specs"]
    )


def test_operator_plan_can_become_probe_ready_without_becoming_score_claim() -> None:
    plan = build_master_gradient_operator_plan(
        _layout_manifest(),
        packet_proofs_available=True,
    )

    assert plan["blockers"] == ()
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_provider_dispatch"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["dispatch_attempted"] is False
    assert all(row["operator_response_valid"] is True for row in plan["rows"])
    assert all(row["ready_for_operator_probe"] is True for row in plan["rows"])
    assert all(row["score_claim"] is False for row in plan["rows"])
    assert all(row["ready_for_provider_dispatch"] is False for row in plan["rows"])
    assert all(
        spec["packet_proofs_available"] is True
        for spec in plan["candidate_modification_specs"]
    )
    assert all(
        spec["ready_for_operator_probe"] is True
        and spec["ready_for_exact_eval_dispatch"] is False
        for spec in plan["candidate_modification_specs"]
    )


def test_operator_plan_blocks_when_logical_layout_is_missing() -> None:
    plan = build_master_gradient_operator_plan(
        {
            "schema": "tac_frontier_archive_layout_v1",
            "archive_path": "unknown.zip",
            "archive_sha256": "f" * 64,
            "logical_layout": None,
        }
    )

    assert plan["operator_row_count"] == 0
    assert plan["candidate_modification_spec_count"] == 0
    assert plan["raw_archive_byte_rows_emitted"] == 0
    assert plan["rows"] == []
    assert plan["candidate_modification_specs"] == []
    assert "logical_layout_missing" in plan["blockers"]


def test_batch_layout_payload_builds_one_plan_per_run() -> None:
    batch = {
        "schema": "tac_frontier_archive_layout_batch_v1",
        "runs": [
            _layout_manifest(),
            {
                "schema": "tac_frontier_archive_layout_v1",
                "archive_path": "unknown.zip",
                "archive_sha256": "f" * 64,
                "logical_layout": None,
            },
        ],
    }

    payload = build_master_gradient_operator_plan_payload(batch)

    assert payload["schema"] == "tac_master_gradient_operator_plan_batch_v1"
    assert payload["plan_count"] == 2
    assert payload["operator_row_count"] == 3
    assert payload["candidate_modification_spec_count"] == 3
    assert payload["raw_archive_byte_rows_emitted"] == 0
    assert payload["score_claim"] is False
    assert payload["ready_for_provider_dispatch"] is False
    assert "logical_layout_missing" in payload["blockers"]
    assert "operator_rows_blocked_until_packet_proofs_land" in payload["blockers"]


def test_candidate_modification_spec_is_the_typed_non_raw_byte_contract() -> None:
    spec = CandidateModificationSpec(
        spec_id="candidate_modification::latent_blob::codebook",
        source_archive_path="archive.zip",
        source_archive_sha256="a" * 64,
        source_archive_bytes=178_517,
        operator_id="latent_blob::latent_conditioning_codebook_sweep",
        section_name="latent_blob",
        section_role="latent_motion_or_frame_conditioning",
        mutation_grain="grammar_aware_operator",
        mutation_operator="latent_conditioning_codebook_sweep",
        axis_label="paired_contest_cpu_cuda",
        response_matrix_columns=("seg_dist_delta", "pose_dist_delta", "rate_bytes_delta"),
        packet_proofs_required=("repacked_archive", "inflate_success_proof"),
        packet_proofs_available=False,
        ready_for_operator_probe=False,
        ready_for_provider_dispatch=False,
        score_claim=False,
        promotion_eligible=False,
        rank_or_kill_eligible=False,
        ready_for_exact_eval_dispatch=False,
        dispatch_attempted=False,
        coordinate_system="grammar_aware_operator_response",
        raw_archive_byte_coordinates_allowed=False,
        blockers=("inflate_success_not_proven",),
        rationale=("not a raw byte delta",),
    )

    payload = spec.to_manifest()

    assert payload["coordinate_system"] == "grammar_aware_operator_response"
    assert payload["raw_archive_byte_coordinates_allowed"] is False
    assert "byte_idx" not in payload
    assert payload["score_claim"] is False
    assert payload["ready_for_provider_dispatch"] is False
