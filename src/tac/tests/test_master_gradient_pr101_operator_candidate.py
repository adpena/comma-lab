# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.master_gradient_pr101_operator_candidate import (
    MUTATION_MODE_RAW_BYTE_DELTA,
    MasterGradientPR101OperatorError,
    build_pr101_pose_axis_decoder_recompression_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PR101_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
OP7_MANIFEST = (
    REPO_ROOT / ".omx/research/pose_axis_operator_pr101_manifest_20260519T074500Z.json"
)


def test_pr101_pose_axis_builder_materializes_same_length_packet_candidate(
    tmp_path: Path,
) -> None:
    if not PR101_ARCHIVE.exists():
        pytest.skip("public PR101 intake archive is an ignored local custody artifact")
    if not OP7_MANIFEST.exists():
        pytest.skip("OP-7 pose-axis operator manifest is unavailable")

    manifest = build_pr101_pose_axis_decoder_recompression_candidate(
        source_archive=PR101_ARCHIVE,
        operator_manifest=json.loads(OP7_MANIFEST.read_text(encoding="utf-8")),
        output_dir=tmp_path / "candidate",
        candidate_id="unit-pr101-op7-rank1",
        candidate_rank=1,
        operator_manifest_path=OP7_MANIFEST,
    )

    archive = Path(manifest["candidate_archive"]["path"])
    assert archive.is_file()
    assert manifest["schema"] == "tac_pr101_pose_axis_decoder_recompression_candidate_v1"
    assert manifest["mutation_mode"] == "raw_equivalent"
    assert manifest["component_moving_candidate"] is False
    assert manifest["semantic_equivalence_expected"] is True
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["selected_pose_axis_candidate"]["rank"] == 1
    assert manifest["selected_pose_axis_candidate"]["section_relative_offset"] == 35773
    assert manifest["selected_stream"]["stream_index"] == 2
    assert manifest["selected_stream"]["compressed_start"] <= 35773
    assert manifest["selected_stream"]["compressed_end"] > 35773
    assert manifest["replacement_stream"]["compressed_bytes"] == manifest["selected_stream"]["compressed_bytes"]
    assert manifest["replacement_stream"]["compressed_sha256"] != manifest["selected_stream"]["compressed_sha256"]
    assert manifest["replacement_stream"]["raw_equivalent_to_source"] is True
    assert manifest["replacement_stream"]["raw_mutation"]["mutation_kind"] == "none"
    assert manifest["replacement_decoder_section"]["section_byte_delta"] == 0
    assert manifest["candidate_archive"]["archive_byte_delta"] == 0
    assert manifest["decoder_brotli_raw_equivalence"]["all_stream_raw_sha256_match"] is True
    assert manifest["packet_proofs"]["repacked_archive"] is True
    assert manifest["packet_proofs"]["updated_zip_headers"] is True
    assert manifest["packet_proofs"]["updated_zip_crc"] is True
    assert manifest["packet_proofs"]["parser_reparse_success"] is True
    assert manifest["packet_proofs"]["structural_non_noop_section_changed"] is True
    assert manifest["packet_proofs"]["decoder_brotli_roundtrip_raw_equivalent"] is True
    assert manifest["packet_proofs"]["decoder_brotli_raw_changed"] is False
    assert manifest["packet_proofs"]["component_moving_operator"] is False
    assert manifest["packet_proofs"]["inflate_success_proof"] is False
    assert manifest["packet_proofs"]["runtime_byte_consumption_noop_detector"] is False
    assert "runtime_byte_consumption_noop_detector_missing" in manifest["dispatch_blockers"]
    assert "score_response_matrix_missing" in manifest["dispatch_blockers"]
    assert "contest_cuda_auth_eval_missing" in manifest["promotion_blockers"]


def test_pr101_pose_axis_builder_materializes_component_moving_packet_candidate(
    tmp_path: Path,
) -> None:
    if not PR101_ARCHIVE.exists():
        pytest.skip("public PR101 intake archive is an ignored local custody artifact")
    if not OP7_MANIFEST.exists():
        pytest.skip("OP-7 pose-axis operator manifest is unavailable")

    manifest = build_pr101_pose_axis_decoder_recompression_candidate(
        source_archive=PR101_ARCHIVE,
        operator_manifest=json.loads(OP7_MANIFEST.read_text(encoding="utf-8")),
        output_dir=tmp_path / "candidate",
        candidate_id="unit-pr101-op7-rank1-raw-byte-delta",
        candidate_rank=1,
        mutation_mode=MUTATION_MODE_RAW_BYTE_DELTA,
        raw_byte_delta=-1,
        operator_manifest_path=OP7_MANIFEST,
    )

    archive = Path(manifest["candidate_archive"]["path"])
    assert archive.is_file()
    assert manifest["mutation_mode"] == MUTATION_MODE_RAW_BYTE_DELTA
    assert manifest["component_moving_candidate"] is True
    assert manifest["semantic_equivalence_expected"] is False
    assert manifest["ready_for_score_response_probe"] is False
    assert manifest["replacement_stream"]["compressed_bytes"] == manifest["selected_stream"]["compressed_bytes"]
    assert manifest["replacement_stream"]["raw_equivalent_to_source"] is False
    assert manifest["replacement_stream"]["raw_sha256"] != manifest["selected_stream"]["raw_sha256"]
    raw_mutation = manifest["replacement_stream"]["raw_mutation"]
    assert raw_mutation["mutation_kind"] == "single_raw_byte_delta"
    assert raw_mutation["raw_byte_delta"] == -1
    assert raw_mutation["source_value"] != raw_mutation["candidate_value"]
    assert manifest["candidate_archive"]["archive_byte_delta"] == 0
    assert manifest["packet_proofs"]["decoder_brotli_roundtrip_raw_equivalent"] is False
    assert manifest["packet_proofs"]["decoder_brotli_raw_changed"] is True
    assert manifest["packet_proofs"]["component_moving_operator"] is True
    assert "component_moving_candidate_requires_score_response_matrix" in manifest["dispatch_blockers"]
    assert "score_response_matrix_missing" in manifest["dispatch_blockers"]
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False


def test_pr101_pose_axis_builder_rejects_zero_raw_delta(tmp_path: Path) -> None:
    if not PR101_ARCHIVE.exists():
        pytest.skip("public PR101 intake archive is an ignored local custody artifact")
    if not OP7_MANIFEST.exists():
        pytest.skip("OP-7 pose-axis operator manifest is unavailable")

    with pytest.raises(MasterGradientPR101OperatorError, match="raw_byte_delta"):
        build_pr101_pose_axis_decoder_recompression_candidate(
            source_archive=PR101_ARCHIVE,
            operator_manifest=json.loads(OP7_MANIFEST.read_text(encoding="utf-8")),
            output_dir=tmp_path / "candidate",
            candidate_id="bad-zero-raw-delta",
            candidate_rank=1,
            mutation_mode=MUTATION_MODE_RAW_BYTE_DELTA,
            raw_byte_delta=0,
            operator_manifest_path=OP7_MANIFEST,
        )


def test_pr101_pose_axis_builder_rejects_non_decoder_rows(tmp_path: Path) -> None:
    operator_manifest = {
        "schema": "pose_byte_hoist_op7_manifest_v1",
        "grammar_aware_operator_candidate_resolution": {
            "resolved_pose_axis_candidates": [
                {
                    "rank": 1,
                    "section_name": "latent",
                    "section_role": "lzma_temporal_delta",
                    "section_relative_offset": 7,
                    "mutation_operator": "latent_codec_coordinate_response",
                    "spec_id": "bad",
                }
            ],
            "candidate_modification_specs": [],
        },
    }

    with pytest.raises(MasterGradientPR101OperatorError, match="unsupported resolved section"):
        build_pr101_pose_axis_decoder_recompression_candidate(
            source_archive=tmp_path / "missing.zip",
            operator_manifest=operator_manifest,
            output_dir=tmp_path / "candidate",
            candidate_id="bad-row",
        )
