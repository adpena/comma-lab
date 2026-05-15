# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.hnerv_lowlevel_packer import write_stored_single_member_zip
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_NO_OP_DIM,
    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106SidecarPacket,
    canonicalize_brotli_dim_delta_sidecar_arrays,
    decode_brotli_dim_delta_sidecar_payload,
    decode_pr101_ranked_sidecar_payload_to_dim_delta,
    decode_pr106_sidecar_packet_dim_delta,
    emit_pr106_sidecar_packet,
    emit_pr106_sidecar_recode_candidate_archive,
    encode_brotli_dim_delta_sidecar_payload,
    encode_pr101_ranked_sidecar_payload,
    lossless_pr106_sidecar_recode_candidates,
    parse_pr106_sidecar_packet,
    pr106_sidecar_recode_candidate_manifest,
    read_single_stored_member_archive,
)

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "profile_pr106_latent_sidecar_recode.py"
PR106_R2_ARCHIVE = REPO / "submissions/pr106_latent_sidecar_r2/archive.zip"
PR106_R2_PR101_ARCHIVE = (
    REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip"
)
PR106_R2_PR101_RUNTIME_PROOF = (
    REPO / "experiments/results/pr106_r2_pr101_grammar_runtime_consumption_proof.json"
)
PR106_R2_PR101_FULL_FRAME_PARITY = (
    REPO / "experiments/results/pr106_r2_same_runtime_full_frame_parity_local_cpu.json"
)
PR106_HDM8_FORMAT07_FIXTURE = REPO / "src/tac/tests/fixtures/pr106_hdm8_format07.archive.zip"


def _sample_arrays(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    dims = np.array([0, 1, 2, 3, 4, 5, 6, PR106_NO_OP_DIM, 7, 8, 9, 10], dtype=np.uint8)[
        :n
    ]
    deltas = np.array([-2, -1, 1, 2, -2, 1, -1, 0, 2, -1, 1, -2], dtype=np.int8)[:n]
    return dims, deltas


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("profile_pr106_latent_sidecar_recode", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_sidecar_encoder_maps_zero_delta_to_noop() -> None:
    dims = np.array([0, 5, 9], dtype=np.uint8)
    deltas = np.array([1, 0, -1], dtype=np.int8)

    payload = encode_brotli_dim_delta_sidecar_payload(dims, deltas)
    got_dims, got_deltas = decode_brotli_dim_delta_sidecar_payload(payload)

    assert got_dims.tolist() == [0, PR106_NO_OP_DIM, 9]
    assert got_deltas.tolist() == [1, 0, -1]


def test_canonical_sidecar_rejects_noop_with_nonzero_delta() -> None:
    with pytest.raises(ValueError, match="no-op dim"):
        canonicalize_brotli_dim_delta_sidecar_arrays(
            np.array([PR106_NO_OP_DIM], dtype=np.uint8),
            np.array([1], dtype=np.int8),
        )


def test_lossless_recode_candidates_roundtrip_semantics() -> None:
    dims, deltas = _sample_arrays()
    source_dims, source_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(dims, deltas)

    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    names = {candidate.name for candidate in candidates}

    assert "current_pr100_dim_delta_brotli_q11" in names
    assert "vocab_bitpack_dim_delta_raw" in names
    assert "split_dim_stream_delta_stream_brotli_q11" in names
    for candidate in candidates:
        if not candidate.encoded_bytes:
            continue
        np.testing.assert_array_equal(candidate.decoded_dims, source_dims)
        np.testing.assert_array_equal(candidate.decoded_delta_q, source_deltas)
        assert candidate.charged_bytes > 0


def test_pr101_ranked_sidecar_candidate_roundtrips_600_pair_payload() -> None:
    dims = np.arange(600, dtype=np.uint16) % 28
    deltas = np.resize(np.array([-2, -1, 1, 2], dtype=np.int8), 600)
    payload, framing_meta = encode_pr101_ranked_sidecar_payload(
        dims.astype(np.uint8),
        deltas,
    )

    got_dims, got_deltas = decode_pr101_ranked_sidecar_payload_to_dim_delta(
        payload,
        framing_meta,
    )

    np.testing.assert_array_equal(got_dims, dims.astype(np.uint8))
    np.testing.assert_array_equal(got_deltas, deltas)
    assert len(framing_meta) == 6


def test_recode_profile_tool_writes_nonpromotable_report(tmp_path: Path) -> None:
    dims, deltas = _sample_arrays()
    sidecar = encode_brotli_dim_delta_sidecar_payload(dims, deltas)
    sidecar_path = tmp_path / "sidecar.bin"
    sidecar_path.write_bytes(sidecar)
    out = tmp_path / "profile.json"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--sidecar-bin",
            str(sidecar_path),
            "--json-out",
            str(out),
        ],
        check=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "pr106_latent_sidecar_recode_profile_v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "no_candidate_archive_emitted" in report["dispatch_blockers"]
    assert report["semantic_arrays"]["n_pairs"] == len(dims)
    assert report["best_lossless_candidate"]["lossless_semantic_equivalence_proven"] is True


def test_recode_profile_tool_reads_sidecar_archive(tmp_path: Path) -> None:
    dims, deltas = _sample_arrays()
    sidecar = encode_brotli_dim_delta_sidecar_payload(dims, deltas)
    payload = emit_pr106_sidecar_packet(
        PR106SidecarPacket(
            format_id=0x01,
            pr106_bytes=b"\xfffixture-pr106-payload",
            sidecar_payload=sidecar,
        )
    )
    archive = tmp_path / "archive.zip"
    write_stored_single_member_zip(archive, member_name="0.bin", payload=payload)
    tool = _load_tool_module()

    report = tool.build_report(
        tool.parse_args(
            [
                "--sidecar-archive",
                str(archive),
                "--json-out",
                str(tmp_path / "unused.json"),
            ]
        )
    )

    assert report["source"]["mode"] == "sidecar_archive"
    assert report["source"]["pr106_inner_payload_bytes"] == len(b"\xfffixture-pr106-payload")
    assert report["score_claim"] is False


def test_recode_profile_tool_reads_implicit_len_fixed_meta_archive(tmp_path: Path) -> None:
    archive_bytes = PR106_R2_PR101_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    source_packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_pr101_ranked_sidecar_payload_to_dim_delta(
        source_packet.sidecar_payload,
        source_packet.framing_meta,
    )
    candidate = next(
        item
        for item in lossless_pr106_sidecar_recode_candidates(dims, deltas)
        if item.sidecar_format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    )
    _candidate_member, candidate_archive = emit_pr106_sidecar_recode_candidate_archive(
        member,
        source_packet,
        candidate,
    )
    archive = tmp_path / "format06.zip"
    archive.write_bytes(candidate_archive)
    tool = _load_tool_module()

    report = tool.build_report(
        tool.parse_args(
            [
                "--sidecar-archive",
                str(archive),
                "--json-out",
                str(tmp_path / "unused.json"),
            ]
        )
    )

    assert report["source"]["sidecar_format_id"] == "0x06"
    assert (
        report["source"]["semantic_source_format"]
        == "pr101_ranked_no_op_implicit_len_fixed_meta_rank_elided_decoded_then_profiled"
    )
    assert report["current_charged_sidecar_bytes"] == 526
    implicit_row = next(
        row
        for row in report["candidate_rows"]
        if row["name"] == "pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06"
    )
    assert implicit_row["delta_bytes_vs_current_charged_sidecar"] == 0
    assert implicit_row["runtime_decoder_implemented"] is True
    assert report["score_claim"] is False


def test_recode_profile_tool_reads_hdm8_and_hdm9_headerless_archives(
    tmp_path: Path,
) -> None:
    fixture_member = read_single_stored_member_archive(PR106_HDM8_FORMAT07_FIXTURE.read_bytes())
    fixture_packet = parse_pr106_sidecar_packet(fixture_member.payload)
    dims, deltas = decode_pr106_sidecar_packet_dim_delta(fixture_packet)
    candidates = lossless_pr106_sidecar_recode_candidates(dims, deltas)
    tool = _load_tool_module()

    for format_id in (
        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    ):
        candidate = next(
            item for item in candidates if item.sidecar_format_id == format_id
        )
        candidate_member, candidate_archive = emit_pr106_sidecar_recode_candidate_archive(
            fixture_member,
            fixture_packet,
            candidate,
        )
        candidate_packet = parse_pr106_sidecar_packet(candidate_member.payload)
        archive = tmp_path / f"format{format_id:02x}.zip"
        archive.write_bytes(candidate_archive)

        report = tool.build_report(
            tool.parse_args(
                [
                    "--sidecar-archive",
                    str(archive),
                    "--json-out",
                    str(tmp_path / f"unused{format_id:02x}.json"),
                ]
            )
        )

        assert report["source"]["sidecar_format_id"] == f"0x{format_id:02X}"
        assert (
            report["source"]["semantic_source_format"]
            == f"{candidate_packet.sidecar_kind}_decoded_then_profiled"
        )
        assert report["current_charged_sidecar_bytes"] == 526
        assert report["score_claim"] is False


def test_recode_candidate_manifest_carries_packetir_consumed_byte_proof() -> None:
    archive_bytes = PR106_R2_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    source_packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_brotli_dim_delta_sidecar_payload(source_packet.sidecar_payload)
    candidate = next(
        item
        for item in lossless_pr106_sidecar_recode_candidates(dims, deltas)
        if item.name == "pr101_ranked_no_op_sidecar_format_0x02"
    )

    manifest = pr106_sidecar_recode_candidate_manifest(
        source_packet,
        candidate,
        source_archive_sha256="7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f",
    )

    assert manifest["schema"] == "pr106_sidecar_recode_candidate_manifest_v1"
    assert manifest["candidate_name"] == "pr101_ranked_no_op_sidecar_format_0x02"
    assert manifest["lossless_semantic_equivalence_proven"] is True
    assert manifest["candidate_packet_ir_identity_passed"] is True
    assert manifest["runtime_consumption_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "exact_cuda_auth_eval_missing" in manifest["exact_eval_blockers"]
    proof = manifest["candidate_packet_ir_consumed_byte_proof"]
    assert isinstance(proof, dict)
    assert proof["all_payload_bytes_accounted"] is True
    assert proof["runtime_consumption_claim"] is False
    sections = proof["sections"]
    assert [row["name"] for row in sections][-1] == "framing_meta"
    for row in sections:
        assert row["offset_start"] == row["offset"]
        assert row["byte_count"] == row["bytes"]
        assert row["offset_end_exclusive"] == row["end_offset"]

    candidate_member, _candidate_archive = emit_pr106_sidecar_recode_candidate_archive(
        member,
        source_packet,
        candidate,
    )
    canonical_pr101_member = read_single_stored_member_archive(
        PR106_R2_PR101_ARCHIVE.read_bytes()
    )
    assert candidate_member.payload == canonical_pr101_member.payload


def test_recode_profile_tool_emits_runtime_candidate_archives(tmp_path: Path) -> None:
    out = tmp_path / "profile.json"
    emitted_dir = tmp_path / "runtime_candidates"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--sidecar-archive",
            str(PR106_R2_ARCHIVE),
            "--json-out",
            str(out),
            "--emit-runtime-candidates-dir",
            str(emitted_dir),
        ],
        check=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    emitted = report["emitted_runtime_candidate_manifests"]
    assert emitted
    assert all(row["score_claim"] is False for row in emitted)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in emitted)
    assert "no_candidate_archive_emitted" not in report["dispatch_blockers"]
    assert "exact_cuda_result_review_already_exists_for_candidate" in report[
        "dispatch_blockers"
    ]
    pr101_emitted = next(
        row
        for row in emitted
        if row["candidate_name"] == "pr101_ranked_no_op_sidecar_format_0x02"
    )
    archive_path = Path(pr101_emitted["archive_path"])
    manifest_path = Path(pr101_emitted["manifest_path"])
    assert archive_path.is_file()
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_packet_ir_identity_passed"] is True
    assert manifest["candidate_packet_ir_consumed_byte_proof"][
        "all_payload_bytes_accounted"
    ] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_recode_profile_links_matching_runtime_and_parity_proofs(tmp_path: Path) -> None:
    out = tmp_path / "profile.json"
    emitted_dir = tmp_path / "runtime_candidates"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--sidecar-archive",
            str(PR106_R2_ARCHIVE),
            "--json-out",
            str(out),
            "--emit-runtime-candidates-dir",
            str(emitted_dir),
            "--runtime-consumption-proof",
            str(PR106_R2_PR101_RUNTIME_PROOF),
            "--same-runtime-full-frame-parity",
            str(PR106_R2_PR101_FULL_FRAME_PARITY),
        ],
        check=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "missing_no_op_runtime_consumption_proof_for_new_grammar" not in report[
        "dispatch_blockers"
    ]
    assert "exact_cuda_result_review_already_exists_for_candidate" in report[
        "dispatch_blockers"
    ]
    assert report["linked_proof_inputs"]["runtime_consumption_proofs"] == [
        str(PR106_R2_PR101_RUNTIME_PROOF)
    ]

    row = next(
        item
        for item in report["candidate_rows"]
        if item["name"] == "pr101_ranked_no_op_sidecar_format_0x02"
    )
    assert row["runtime_consumption_claim"] is True
    assert row["runtime_decode_apply_proof_claim"] is True
    assert row["full_frame_inflate_output_parity_claim"] is True
    assert row["runtime_consumption_proof"]["valid_for_candidate_archive"] is True
    assert row["same_runtime_full_frame_parity_proof"]["valid_for_candidate_archive"] is True
    assert "runtime_decode_apply_proof_required_for_new_candidate_archive" not in row[
        "candidate_exact_eval_blockers"
    ]
    assert "full_frame_same_runtime_parity_or_same_runtime_auth_eval_missing" not in row[
        "candidate_exact_eval_blockers"
    ]
    assert row["candidate_exact_eval_blockers"] == [
        "exact_cuda_auth_eval_missing",
        "contest_auth_eval_adjudication_missing",
    ]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    manifest = json.loads(
        Path(row["emitted_candidate_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["runtime_consumption_claim"] is True
    assert manifest["full_frame_inflate_output_parity_claim"] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_recode_profile_links_matching_exact_cuda_result_review(tmp_path: Path) -> None:
    archive_bytes = PR106_R2_ARCHIVE.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    source_packet = parse_pr106_sidecar_packet(member.payload)
    dims, deltas = decode_brotli_dim_delta_sidecar_payload(source_packet.sidecar_payload)
    candidate = next(
        item
        for item in lossless_pr106_sidecar_recode_candidates(dims, deltas)
        if item.name == "pr101_ranked_no_op_sidecar_format_0x02"
    )
    _candidate_member, candidate_archive = emit_pr106_sidecar_recode_candidate_archive(
        member,
        source_packet,
        candidate,
    )
    candidate_sha256 = hashlib.sha256(candidate_archive).hexdigest()
    review = {
        "schema": "tac_result_review_packet_v1",
        "lane_id": "test_pr106_recode_exact_review_join",
        "job_id": "test-job",
        "score_axis": "contest_cuda",
        "exact_cuda_evidence": True,
        "score_claim_valid": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "measured_config_status": "exact_cuda_result_reviewed",
        "custody": {
            "archive_sha256": candidate_sha256,
            "archive_bytes": len(candidate_archive),
        },
        "runtime_custody": {
            "runtime_tree_sha256": "a" * 64,
            "runtime_content_tree_sha256": "b" * 64,
        },
        "score_recomputation": {
            "available": True,
            "recomputed_score": 0.2063310355127786,
        },
    }
    review_path = tmp_path / "exact_review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    out = tmp_path / "profile.json"
    emitted_dir = tmp_path / "runtime_candidates"

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--sidecar-archive",
            str(PR106_R2_ARCHIVE),
            "--json-out",
            str(out),
            "--emit-runtime-candidates-dir",
            str(emitted_dir),
            "--no-default-exact-result-review-scan",
            "--exact-result-review",
            str(review_path),
        ],
        check=True,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert "exact_cuda_result_review_already_exists_for_candidate" in report[
        "dispatch_blockers"
    ]
    row = next(
        item
        for item in report["candidate_rows"]
        if item["name"] == "pr101_ranked_no_op_sidecar_format_0x02"
    )
    assert row["exact_cuda_auth_eval_claim"] is True
    assert row["exact_cuda_result_reviews"][0]["archive_sha256"] == candidate_sha256
    assert row["exact_cuda_result_reviews"][0]["canonical_score"] == 0.2063310355127786
    assert "exact_cuda_auth_eval_missing" not in row["candidate_exact_eval_blockers"]
    assert "contest_auth_eval_adjudication_missing" not in row[
        "candidate_exact_eval_blockers"
    ]
    assert "exact_cuda_result_review_already_exists" in row[
        "candidate_exact_eval_blockers"
    ]
    manifest = json.loads(
        Path(row["emitted_candidate_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["exact_cuda_auth_eval_claim"] is True
    assert manifest["exact_cuda_result_reviews"][0]["path"] == str(review_path)
    assert "exact_cuda_auth_eval_missing" not in manifest["exact_eval_blockers"]
    assert "contest_auth_eval_adjudication_missing" not in manifest[
        "exact_eval_blockers"
    ]
    assert "exact_cuda_result_review_already_exists" in manifest["exact_eval_blockers"]
