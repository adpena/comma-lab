# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import math
import zipfile
from pathlib import Path

import pytest

from tac.packet_compiler import PR106_PACKET_IR_SECTION_HASH_DOMAIN
from tac.packetir_exact_closure import build_packetir_exact_closure

SHA_SOURCE = "b" * 64
SHA_BEST = "c" * 64
BYTES_CANDIDATE = 900
BYTES_SOURCE = 1000
BYTES_BEST = 850
SEG = 0.001
POSE = 0.00004
RUNTIME_INFLATE_PY_SHA = "d" * 64
RUNTIME_CONTENT_TREE_SHA = "e" * 64
INNER_PR106_PAYLOAD_SHA = "f" * 64
HEADERLESS_SECTION_SHA = "a" * 64
FORMAT0D_PR106_SHA = "1" * 64
FORMAT0D_BASE_SHA = "2" * 64
FORMAT0D_EXTRA_SHA = "3" * 64
FORMAT0D_EXTRA_META_SHA = "4" * 64
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLOSURE_TOOL = _REPO_ROOT / "tools" / "build_pr106_r2_packetir_exact_closure.py"


def test_packetir_exact_closure_closes_measured_not_current_frontier(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        cpu_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cpu", claim=False, pose=0.00009),
        source_cuda_eval=_eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True),
        current_best_cuda_eval=_eval(SHA_BEST, BYTES_BEST, "contest_cuda", claim=True),
        runtime_consumption_proof=_runtime_consumption(archive),
        full_frame_parity_proof=_full_frame_parity(archive),
        recode_profile=_profile(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_not_current_frontier"
    assert closure["score_claim"] is False
    assert closure["promotion_eligible"] is False
    assert closure["ready_for_exact_eval_dispatch"] is False
    assert closure["blockers"] == []
    assert "same_candidate_archive_already_exact_evaluated" in closure["duplicate_dispatch_blockers"]
    assert "candidate_not_current_frontier_on_contest_cuda" in closure["duplicate_dispatch_blockers"]
    assert closure["comparisons"]["improves_packetir_source_cuda"] is True
    assert closure["comparisons"]["not_current_frontier"] is True
    assert closure["axes"]["contest_cpu"]["score_axis"] == "contest_cpu"
    assert closure["axes"]["axis_gap"]["interpretation"].startswith("axis divergence")
    assert closure["packetir"]["runtime_consumption_proof"]["valid"] is True
    assert closure["packetir"]["same_runtime_full_frame_parity"]["valid"] is True
    assert closure["exact_eval_duplicate_keys"][0]["key"].endswith(":contest_cuda")
    assert closure["exact_eval_duplicate_keys"][0]["runtime_tree_sha256"]
    assert closure["exact_eval_duplicate_keys"][0]["runtime_tree_sha256"] in closure[
        "exact_eval_duplicate_keys"
    ][0]["key"]
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_fails_closed_on_archive_sha_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    result = _candidate_result(archive)
    result["candidate_archive_sha256"] = "0" * 64

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "candidate_archive_file_matches_packetir_result" in closure["blockers"]
    assert "closure_evidence_inconsistent_fail_closed" in closure["duplicate_dispatch_blockers"]


def test_packetir_exact_closure_rejects_cuda_axis_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cpu", claim=False),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "cuda_eval_is_valid_contest_cuda_score_claim" in closure["blockers"]


def test_packetir_exact_closure_rejects_cuda_claim_without_cuda_device_semantics(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    cuda_eval = _eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True)
    cuda_eval["scorer_device"] = "cpu"
    cuda_eval["provenance_device"] = "cpu"
    cuda_eval["gpu_model"] = "linux-cpu"

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=cuda_eval,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "cuda_eval_axis_semantics_are_contest_cuda" in closure["blockers"]
    axis_check = next(
        check
        for check in closure["checks"]
        if check["id"] == "cuda_eval_axis_semantics_are_contest_cuda"
    )
    assert "eval_device_not_cuda" in axis_check["evidence"]["axis_semantics_blockers"]
    assert "hardware_not_cuda" in axis_check["evidence"]["axis_semantics_blockers"]


def test_packetir_exact_closure_rejects_partial_sample_cuda_eval(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    cuda_eval = _eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True)
    cuda_eval["n_samples"] = 1

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=cuda_eval,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "cuda_eval_axis_semantics_are_contest_cuda" in closure["blockers"]
    axis_check = next(
        check
        for check in closure["checks"]
        if check["id"] == "cuda_eval_axis_semantics_are_contest_cuda"
    )
    assert (
        "n_samples_not_contest_exact"
        in axis_check["evidence"]["axis_semantics_blockers"]
    )


def test_packetir_exact_closure_rejects_cpu_eval_with_cuda_device_semantics(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    cpu_eval = _eval(_sha256_file(archive), archive.stat().st_size, "contest_cpu", claim=False)
    cpu_eval["scorer_device"] = "cuda"
    cpu_eval["provenance_device"] = "cuda"

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        cpu_eval=cpu_eval,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "cpu_eval_is_axis_labeled_diagnostic_not_cuda_claim" in closure["blockers"]
    cpu_check = next(
        check
        for check in closure["checks"]
        if check["id"] == "cpu_eval_is_axis_labeled_diagnostic_not_cuda_claim"
    )
    assert "eval_device_not_cpu" in cpu_check["evidence"]["axis_semantics_blockers"]


def test_packetir_exact_closure_rejects_score_recompute_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    eval_payload = _eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True)
    eval_payload["canonical_score"] += 0.01

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=eval_payload,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "cuda_eval_is_valid_contest_cuda_score_claim" in closure["blockers"]


def test_packetir_exact_closure_surfaces_cuda_promotion_rank_blockers_top_level(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    cuda_eval = _eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True)
    cuda_eval["promotion_blockers"] = ["runtime_tree_not_submission_ready"]
    cuda_eval["rank_or_kill_blockers"] = ["needs_adversarial_review"]
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=cuda_eval,
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=_runtime_consumption(archive),
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "cuda_eval_promotion_and_rank_authority_blockers_absent" in closure["blockers"]
    assert closure["closure_authority_blockers"] == [
        "runtime_tree_not_submission_ready",
        "needs_adversarial_review",
    ]


def test_packetir_exact_closure_requires_runtime_and_full_frame_proofs(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=_eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True),
        current_best_cuda_eval=_eval(SHA_BEST, BYTES_BEST, "contest_cuda", claim=True),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert (
        "runtime_consumption_proof_binds_candidate_and_score_affecting_sections"
        in closure["blockers"]
    )
    assert (
        "same_runtime_full_frame_parity_binds_candidate_source_and_runtime"
        in closure["blockers"]
    )
    assert "runtime_identity_matches_cuda_eval_runtime" in closure["blockers"]


def test_packetir_exact_closure_rejects_runtime_proofs_without_exact_cuda_content_tree(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    runtime_consumption = _runtime_consumption(archive)
    full_frame_parity = _full_frame_parity(archive)
    runtime_consumption["runtime_source_manifest"].pop("runtime_content_tree_sha256")
    full_frame_parity["runtime_source_manifest"].pop("runtime_content_tree_sha256")
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=full_frame_parity,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "runtime_identity_matches_cuda_eval_runtime" in closure["blockers"]
    assert closure["packetir"]["runtime_consumption_proof"]["valid"] is True
    assert closure["packetir"]["runtime_consumption_proof"][
        "runtime_content_tree_sha256"
    ] is None
    assert closure["packetir"]["same_runtime_full_frame_parity"][
        "runtime_content_tree_sha256"
    ] is None


def test_packetir_exact_closure_rejects_runtime_consumption_runtime_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["runtime_inflate_py_sha256"] = "0" * 64
    runtime_consumption["runtime_source_manifest"]["files"][0]["sha256"] = "0" * 64

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=_eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True),
        current_best_cuda_eval=_eval(SHA_BEST, BYTES_BEST, "contest_cuda", claim=True),
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "runtime_identity_matches_cuda_eval_runtime" in closure["blockers"]


def test_packetir_exact_closure_rejects_runtime_section_omission(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    runtime_consumption = _runtime_consumption(archive)
    del runtime_consumption["runtime_consumed_score_affecting_sections"]["framing_meta"]

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=_eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True),
        current_best_cuda_eval=_eval(SHA_BEST, BYTES_BEST, "contest_cuda", claim=True),
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in closure["blockers"]


def test_packetir_exact_closure_rejects_prefix_or_short_frame_parity(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    parity = _full_frame_parity(archive)
    parity["prefix_parity_claim"] = True
    parity["candidate"]["n_pairs_hashed"] = 1

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=_eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True),
        current_best_cuda_eval=_eval(SHA_BEST, BYTES_BEST, "contest_cuda", claim=True),
        runtime_consumption_proof=_runtime_consumption(archive),
        full_frame_parity_proof=parity,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "blocked_inconsistent_or_missing_evidence"
    assert (
        "same_runtime_full_frame_parity_binds_candidate_source_and_runtime"
        in closure["blockers"]
    )
    assert "runtime_identity_matches_cuda_eval_runtime" in closure["blockers"]


def test_packetir_exact_closure_accepts_profile_row_for_packetir_source(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    profile = _profile_for_sha(SHA_SOURCE)

    closure = build_packetir_exact_closure(
        lane_id="lane",
        candidate_result=_candidate_result(archive),
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        current_best_cuda_eval=_eval(SHA_BEST, BYTES_BEST, "contest_cuda", claim=True),
        recode_profile=profile,
        repo_root=tmp_path,
    )

    assert "recode_profile_keeps_candidate_nonpromotable_before_exact_eval" not in closure["blockers"]
    profile_check = next(
        check
        for check in closure["checks"]
        if check["id"] == "recode_profile_keeps_candidate_nonpromotable_before_exact_eval"
    )
    assert profile_check["evidence"]["matches_packetir_source_archive"] is True


def test_packetir_closure_tool_merges_hlm1_manifest_shape() -> None:
    mod = _load_closure_tool()
    merged = mod._candidate_result_with_packetir_identity(
        candidate_result={
            "candidate_archive_byte_delta": -8,
            "archive_build_blockers": [],
            "candidate_archive_sha256": "a" * 64,
            "candidate_archive_bytes": 10,
            "source_archive_sha256": "b" * 64,
            "source_archive_bytes": 18,
            "score_claim": False,
            "dispatch_attempted": False,
        },
        packetir_identity={
            "packet": {
                "packet_ir_consumed_byte_proof": {
                    "all_payload_bytes_accounted": True,
                    "runtime_consumption_claim": False,
                    "score_affecting_section_names": ["pr106_payload"],
                },
            },
        },
    )

    assert merged["candidate_diff_audit"] == {
        "blockers": [],
        "total_byte_delta": -8,
    }
    assert merged["packet_ir_consumed_byte_proof"]["all_payload_bytes_accounted"] is True
    assert merged["packet_ir_consumed_byte_proof"]["score_affecting_section_names"] == [
        "pr106_payload"
    ]


def test_packetir_closure_tool_merges_format06_candidate_proof_shape() -> None:
    mod = _load_closure_tool()
    merged = mod._candidate_result_with_packetir_identity(
        candidate_result={
            "candidate_archive_byte_delta_vs_source": -13,
            "candidate_archive_sha256": "a" * 64,
            "candidate_archive_bytes": 186382,
            "source_archive_sha256": "b" * 64,
            "score_claim": False,
            "dispatch_attempted": False,
            "candidate_packet_ir_consumed_byte_proof": {
                "all_payload_bytes_accounted": True,
                "runtime_consumption_claim": False,
                "score_affecting_section_names": ["pr106_payload", "sidecar_payload"],
            },
        },
        packetir_identity=None,
    )

    assert merged["source_archive_bytes"] == 186395
    assert merged["candidate_diff_audit"] == {
        "blockers": [],
        "total_byte_delta": -13,
    }
    assert merged["packet_ir_consumed_byte_proof"]["all_payload_bytes_accounted"] is True
    assert merged["packet_ir_consumed_byte_proof"]["score_affecting_section_names"] == [
        "pr106_payload",
        "sidecar_payload",
    ]


def test_packetir_exact_closure_accepts_format06_manifest_without_source_bytes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result_format06_shape(archive)
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 13, "contest_cuda", claim=True)
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["runtime_consumed_score_affecting_sections"]["framing_meta"] = None
    parity = _full_frame_parity(archive)
    parity["source_archive"]["bytes"] = archive.stat().st_size + 13

    closure = build_packetir_exact_closure(
        lane_id="lane_format06",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=parity,
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_improves_packetir_source_cuda"
    assert closure["blockers"] == []
    assert closure["archive"]["source_archive_bytes"] == archive.stat().st_size + 13
    assert closure["archive"]["byte_delta_vs_packetir_source"] == -13
    assert closure["packetir"]["runtime_consumption_proof"]["valid"] is True
    assert closure["packetir"]["runtime_consumption_proof"][
        "actual_score_affecting_sections"
    ] == ["pr106_payload", "sidecar_payload"]
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_accepts_format08_hdm8_runtime_section_alias(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm8_hlm2_payload_without_inner_header",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm8_hlm2_payload_without_inner_header",
    )
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x08"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format08",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_improves_packetir_source_cuda"
    assert closure["blockers"] == []
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is True
    assert proof["score_affecting_section_match_mode"] == (
        "format_0x08_hdm8_hlm2_reconstructed_pr106_payload_alias"
    )
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_accepts_format09_hdm9_runtime_section_alias(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm2_payload_without_inner_header",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm9_hlm2_payload_without_inner_header",
    )
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x09"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format09",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_improves_packetir_source_cuda"
    assert closure["blockers"] == []
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is True
    assert proof["score_affecting_section_match_mode"] == (
        "format_0x09_hdm9_hlm2_reconstructed_pr106_payload_alias"
    )
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_accepts_format0b_magicless_runtime_section_alias(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
    )
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x0B"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0b",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_improves_packetir_source_cuda"
    assert closure["blockers"] == []
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is True
    assert proof["score_affecting_section_match_mode"] == (
        "format_0x0b_hdm9_hlm3_magicless_reconstructed_pr106_payload_alias"
    )
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_accepts_format0c_exact_radix_magicless_alias(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
    )
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x0C"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0c",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_improves_packetir_source_cuda"
    assert closure["blockers"] == []
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is True
    assert proof["score_affecting_section_match_mode"] == (
        "format_0x0c_hdm9_hlm3_magicless_exact_radix_reconstructed_pr106_payload_alias"
    )
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_accepts_format0d_base_then_extra_runtime_closure(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result_format0d_shape(archive)
    runtime_consumption = _runtime_consumption_format0d(archive)
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0d",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert closure["classification"] == "exact_measured_improves_packetir_source_cuda"
    assert closure["blockers"] == []
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is True
    assert proof["score_affecting_section_match_mode"] == (
        "format_0x0d_base_then_extra_runtime_closure"
    )
    identity = proof["score_affecting_section_match_evidence"][
        "format0d_closure_identity"
    ]
    assert identity["runtime_apply_order_valid"] is True
    assert identity["candidate_base_extra_section_ordered"] is True
    assert all(row["identity_valid"] for row in identity["sections"].values())
    assert all(check["passed"] for check in closure["checks"])


def test_packetir_exact_closure_rejects_format0d_without_runtime_extra_identity(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result_format0d_shape(archive)
    runtime_consumption = _runtime_consumption_format0d(archive)
    runtime_consumption["runtime_consumed_score_affecting_section_identities"] = [
        row
        for row in runtime_consumption[
            "runtime_consumed_score_affecting_section_identities"
        ]
        if row["name"] != "extra_pr101_ranked_no_op_payload"
    ]
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0d_missing_extra_runtime_identity",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is False
    assert proof["expected_score_affecting_sections"] == [
        "base_format0c_sidecar_payload",
        "extra_framing_meta",
        "extra_pr101_ranked_no_op_payload",
        "pr106_payload",
    ]
    assert proof["actual_score_affecting_sections"] == [
        "base_format0c_sidecar_payload",
        "extra_framing_meta",
        "extra_pr101_ranked_no_op_payload",
        "pr106_payload",
    ]
    identity = proof["score_affecting_section_match_evidence"][
        "format0d_closure_identity"
    ]
    extra = identity["sections"]["extra_pr101_ranked_no_op_payload"]
    assert extra["candidate_section_found"] is True
    assert extra["runtime_section_found"] is False
    assert extra["identity_valid"] is False


def test_packetir_exact_closure_rejects_format0d_runtime_offset_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result_format0d_shape(archive)
    runtime_consumption = _runtime_consumption_format0d(archive)
    for row in runtime_consumption["runtime_consumed_score_affecting_section_identities"]:
        if row["name"] == "extra_pr101_ranked_no_op_payload":
            row["offset"] = row["offset"] + 1
            row["offset_start"] = row["offset"]
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0d_offset_mismatch",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    identity = closure["packetir"]["runtime_consumption_proof"][
        "score_affecting_section_match_evidence"
    ]["format0d_closure_identity"]
    extra = identity["sections"]["extra_pr101_ranked_no_op_payload"]
    assert extra["sha256_matches"] is True
    assert extra["length_matches"] is True
    assert extra["offset_matches"] is False
    assert extra["identity_valid"] is False


def test_packetir_exact_closure_rejects_format0d_runtime_hash_domain_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result_format0d_shape(archive)
    runtime_consumption = _runtime_consumption_format0d(archive)
    for row in runtime_consumption["runtime_consumed_score_affecting_section_identities"]:
        if row["name"] == "extra_pr101_ranked_no_op_payload":
            row["hash_domain"] = "runtime_correction_digest_bytes_v1"
            row["sha256_domain"] = row["hash_domain"]
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0d_hash_domain_mismatch",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    identity = closure["packetir"]["runtime_consumption_proof"][
        "score_affecting_section_match_evidence"
    ]["format0d_closure_identity"]
    extra = identity["sections"]["extra_pr101_ranked_no_op_payload"]
    assert extra["sha256_matches"] is True
    assert extra["hash_domain_matches"] is False
    assert extra["runtime_hash_domain_valid"] is False
    assert extra["identity_valid"] is False


def test_packetir_exact_closure_rejects_format0d_candidate_hash_domain_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result_format0d_shape(archive)
    for row in candidate_result["packet_ir_consumed_byte_proof"]["sections"]:
        if row["name"] == "base_format0c_sidecar_payload":
            row["hash_domain"] = "whole_member_payload_bytes_v1"
            row["sha256_domain"] = row["hash_domain"]
    runtime_consumption = _runtime_consumption_format0d(archive)
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0d_candidate_hash_domain_mismatch",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    identity = closure["packetir"]["runtime_consumption_proof"][
        "score_affecting_section_match_evidence"
    ]["format0d_closure_identity"]
    base = identity["sections"]["base_format0c_sidecar_payload"]
    assert base["sha256_matches"] is True
    assert base["hash_domain_matches"] is False
    assert base["candidate_hash_domain_valid"] is False
    assert base["identity_valid"] is False


@pytest.mark.parametrize("inner_unchanged", [False, None])
def test_packetir_exact_closure_rejects_format0b_alias_without_inner_identity(
    tmp_path: Path,
    inner_unchanged: bool | None,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
    )
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x0B"
    if inner_unchanged is None:
        runtime_consumption.pop("inner_pr106_payload_sha256_unchanged", None)
    else:
        runtime_consumption["inner_pr106_payload_sha256_unchanged"] = inner_unchanged
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0b_reject",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is False
    assert proof["score_affecting_section_set_matches_packetir"] is False


def test_packetir_exact_closure_rejects_headerless_alias_sha_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x0B"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["runtime_inner_pr106_payload_sha256"] = "0" * 64
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0b_sha_mismatch",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is False
    assert proof["score_affecting_section_set_matches_packetir"] is False


def test_packetir_exact_closure_rejects_headerless_alias_candidate_section_sha_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
    )
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x0B"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["candidate_headerless_section_sha256"] = "0" * 64
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0b_candidate_section_sha_mismatch",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    proof = closure["packetir"]["runtime_consumption_proof"]
    assert proof["valid"] is False
    assert proof["score_affecting_section_set_matches_packetir"] is False
    identity = proof["score_affecting_section_match_evidence"][
        "headerless_alias_identity"
    ]
    assert identity["candidate_consumed_section_sha256"] == HEADERLESS_SECTION_SHA
    assert identity["candidate_headerless_section_sha256"] == "0" * 64
    assert identity["candidate_section_bound_to_consumed_byte_proof"] is False


def test_packetir_exact_closure_rejects_headerless_alias_candidate_hash_domain_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "candidate.zip"
    _write_zip(archive, b"x" * BYTES_CANDIDATE)
    candidate_result = _candidate_result(archive)
    candidate_result["packet_ir_consumed_byte_proof"][
        "score_affecting_section_names"
    ] = [
        "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
        "sidecar_payload",
    ]
    _bind_headerless_consumed_section(
        candidate_result,
        section_name="pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
    )
    candidate_result["packet_ir_consumed_byte_proof"]["sections"][0][
        "hash_domain"
    ] = "whole_member_payload_bytes_v1"
    candidate_result["packet_ir_consumed_byte_proof"]["sections"][0][
        "sha256_domain"
    ] = "whole_member_payload_bytes_v1"
    runtime_consumption = _runtime_consumption(archive)
    runtime_consumption["format_id"] = "0x0B"
    runtime_consumption["inner_pr106_payload_sha256_unchanged"] = True
    _add_headerless_alias_identity(runtime_consumption)
    runtime_consumption["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "sidecar_payload": True,
        "framing_meta": None,
    }
    source_eval = _eval(SHA_SOURCE, archive.stat().st_size + 100, "contest_cuda", claim=True)

    closure = build_packetir_exact_closure(
        lane_id="lane_format0b_candidate_hash_domain_mismatch",
        candidate_result=candidate_result,
        candidate_archive_path=archive,
        cuda_eval=_eval(_sha256_file(archive), archive.stat().st_size, "contest_cuda", claim=True),
        source_cuda_eval=source_eval,
        current_best_cuda_eval=source_eval,
        runtime_consumption_proof=runtime_consumption,
        full_frame_parity_proof=_full_frame_parity(archive),
        repo_root=tmp_path,
    )

    assert "runtime_consumption_proof_binds_candidate_and_score_affecting_sections" in (
        closure["blockers"]
    )
    identity = closure["packetir"]["runtime_consumption_proof"][
        "score_affecting_section_match_evidence"
    ]["headerless_alias_identity"]
    assert identity["candidate_hash_domain_valid"] is False
    assert identity["candidate_section_bound_to_consumed_byte_proof"] is False


def _add_headerless_alias_identity(proof: dict) -> None:
    proof["source_inner_pr106_payload_sha256"] = INNER_PR106_PAYLOAD_SHA
    proof["runtime_inner_pr106_payload_sha256"] = INNER_PR106_PAYLOAD_SHA
    proof["candidate_headerless_section_sha256"] = HEADERLESS_SECTION_SHA
    proof["candidate_headerless_section_offset"] = 16
    proof["candidate_headerless_section_length"] = 128


def _bind_headerless_consumed_section(
    candidate_result: dict,
    *,
    section_name: str,
    sha256: str = HEADERLESS_SECTION_SHA,
    offset: int = 16,
    length: int = 128,
    score_affecting: bool = True,
) -> None:
    proof = candidate_result["packet_ir_consumed_byte_proof"]
    proof["sections"] = [
        {
            "name": section_name,
            "offset": offset,
            "offset_start": offset,
            "bytes": length,
            "byte_count": length,
            "sha256": sha256,
            "hash_domain": PR106_PACKET_IR_SECTION_HASH_DOMAIN,
            "sha256_domain": PR106_PACKET_IR_SECTION_HASH_DOMAIN,
            "score_affecting": score_affecting,
        }
    ]


def _candidate_result(archive: Path) -> dict:
    archive_sha = _sha256_file(archive)
    archive_bytes = archive.stat().st_size
    return {
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive_bytes,
        "source_archive_sha256": SHA_SOURCE,
        "source_archive_bytes": archive_bytes + 100,
        "candidate_diff_audit": {
            "blockers": [],
            "total_byte_delta": -100,
        },
        "packet_ir_consumed_byte_proof": {
            "all_payload_bytes_accounted": True,
            "runtime_consumption_claim": False,
            "unconsumed_trailing_bytes": 0,
            "proof_scope": "packet_ir_parser_accounting_not_runtime_inflate_consumption",
            "score_affecting_section_names": [
                "pr106_payload",
                "sidecar_payload",
                "framing_meta",
            ],
        },
    }


def _candidate_result_format06_shape(archive: Path) -> dict:
    archive_sha = _sha256_file(archive)
    return {
        "score_claim": False,
        "dispatch_attempted": False,
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive.stat().st_size,
        "source_archive_sha256": SHA_SOURCE,
        "candidate_packet_ir_consumed_byte_proof": {
            "all_payload_bytes_accounted": True,
            "runtime_consumption_claim": False,
            "unconsumed_trailing_bytes": 0,
            "proof_scope": "packet_ir_parser_accounting_not_runtime_inflate_consumption",
            "score_affecting_section_names": [
                "pr106_payload",
                "sidecar_payload",
            ],
        },
    }


def _candidate_result_format0d_shape(archive: Path) -> dict:
    result = _candidate_result(archive)
    result["packet_ir_consumed_byte_proof"] = {
        "all_payload_bytes_accounted": True,
        "runtime_consumption_claim": False,
        "unconsumed_trailing_bytes": 0,
        "proof_scope": "packet_ir_parser_accounting_not_runtime_inflate_consumption",
        "score_affecting_section_names": [
            "pr106_payload",
            "base_format0c_sidecar_payload",
            "extra_pr101_ranked_no_op_payload",
            "extra_framing_meta",
        ],
        "sections": _format0d_section_rows(),
    }
    return result


def _load_closure_tool():
    spec = importlib.util.spec_from_file_location("build_pr106_r2_packetir_exact_closure", _CLOSURE_TOOL)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _profile(archive: Path) -> dict:
    return _profile_for_sha(_sha256_file(archive))


def _profile_for_sha(archive_sha: str) -> dict:
    return {
        "candidate_rows": [
            {
                "name": "candidate",
                "emitted_candidate_archive_sha256": archive_sha,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "runtime_decoder_implemented": True,
            }
        ]
    }


def _eval(
    archive_sha: str,
    archive_bytes: int,
    axis: str,
    *,
    claim: bool,
    pose: float = POSE,
) -> dict:
    score = 100.0 * SEG + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / 37_545_489.0
    return {
        "score_axis": axis,
        "lane_tag": "[contest-CUDA]" if axis == "contest_cuda" else "[contest-CPU]",
        "evidence_grade": "contest-CUDA" if axis == "contest_cuda" else "contest-CPU",
        "archive_size_bytes": archive_bytes,
        "avg_segnet_dist": SEG,
        "avg_posenet_dist": pose,
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "n_samples": 600,
        "scorer_device": "cuda" if axis == "contest_cuda" else "cpu",
        "provenance_device": "cuda" if axis == "contest_cuda" else "cpu",
        "gpu_model": "Tesla T4" if axis == "contest_cuda" else "linux-cpu",
        "score_claim": claim,
        "score_claim_valid": claim,
        "exact_cuda_eval_complete": claim and axis == "contest_cuda",
        "promotion_eligible": False,
        "provenance": {
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_bytes,
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": "f" * 64,
                "runtime_content_tree_sha256": RUNTIME_CONTENT_TREE_SHA,
                "files": [
                    {
                        "relative_path": "inflate.py",
                        "sha256": RUNTIME_INFLATE_PY_SHA,
                    },
                    {
                        "relative_path": "inflate.sh",
                        "sha256": "a" * 64,
                    },
                    {
                        "relative_path": "src/codec.py",
                        "sha256": "1" * 64,
                    },
                    {
                        "relative_path": "src/model.py",
                        "sha256": "2" * 64,
                    },
                    {
                        "relative_path": "src/pr101_grammar.py",
                        "sha256": "3" * 64,
                    },
                ],
            },
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": "9" * 64,
                    "total_bytes": 9876,
                }
            },
        },
    }


def _runtime_consumption(archive: Path) -> dict:
    return {
        "archive": {
            "sha256": _sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "blockers": [],
        "schema": "pr106_sidecar_runtime_decode_consumption_proof_v1",
        "runtime_dir": "submission",
        "runtime_inflate_py_sha256": RUNTIME_INFLATE_PY_SHA,
        "runtime_source_manifest": {
            "runtime_source_tree_sha256": "4" * 64,
            "runtime_content_tree_sha256": RUNTIME_CONTENT_TREE_SHA,
            "files": [
                {
                    "path": "inflate.py",
                    "sha256": RUNTIME_INFLATE_PY_SHA,
                    "bytes": 100,
                },
                {
                    "path": "src/codec.py",
                    "sha256": "1" * 64,
                    "bytes": 101,
                },
                {
                    "path": "src/model.py",
                    "sha256": "2" * 64,
                    "bytes": 102,
                },
                {
                    "path": "src/pr101_grammar.py",
                    "sha256": "3" * 64,
                    "bytes": 103,
                },
            ],
        },
        "parser_consumed_byte_accounting_passed": True,
        "runtime_all_score_affecting_sections_consumed": True,
        "runtime_consumed_score_affecting_sections": {
            "pr106_payload": True,
            "sidecar_payload": True,
            "framing_meta": True,
        },
        "runtime_corrected_latents_digest_changed": True,
        "proof_scope": "actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame",
        "score_claim": False,
        "contest_axis_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _runtime_consumption_format0d(archive: Path) -> dict:
    runtime = _runtime_consumption(archive)
    runtime["format_id"] = "0x0D"
    runtime["runtime_apply_order"] = [
        "base_format0c_corrections",
        "extra_pr101_ranked_no_op_corrections",
    ]
    runtime["runtime_consumed_score_affecting_sections"] = {
        "pr106_payload": True,
        "base_format0c_sidecar_payload": True,
        "extra_pr101_ranked_no_op_payload": True,
        "extra_framing_meta": True,
    }
    runtime["runtime_consumed_score_affecting_section_identities"] = [
        {
            "name": row["name"],
            "sha256": row["sha256"],
            "hash_domain": row["hash_domain"],
            "sha256_domain": row["sha256_domain"],
            "bytes": row["bytes"],
            "offset": row["offset"],
            "offset_start": row["offset_start"],
            "consumed": True,
        }
        for row in _format0d_section_rows()
    ]
    return runtime


def _format0d_section_rows() -> list[dict]:
    return [
        _format0d_section_row(
            "pr106_payload",
            offset=6,
            length=64,
            sha256=FORMAT0D_PR106_SHA,
        ),
        _format0d_section_row(
            "base_format0c_sidecar_payload",
            offset=70,
            length=32,
            sha256=FORMAT0D_BASE_SHA,
        ),
        _format0d_section_row(
            "extra_pr101_ranked_no_op_payload",
            offset=104,
            length=16,
            sha256=FORMAT0D_EXTRA_SHA,
        ),
        _format0d_section_row(
            "extra_framing_meta",
            offset=120,
            length=6,
            sha256=FORMAT0D_EXTRA_META_SHA,
        ),
    ]


def _format0d_section_row(
    name: str,
    *,
    offset: int,
    length: int,
    sha256: str,
) -> dict:
    return {
        "name": name,
        "offset": offset,
        "offset_start": offset,
        "bytes": length,
        "byte_count": length,
        "sha256": sha256,
        "hash_domain": PR106_PACKET_IR_SECTION_HASH_DOMAIN,
        "sha256_domain": PR106_PACKET_IR_SECTION_HASH_DOMAIN,
        "score_affecting": True,
    }


def _full_frame_parity(archive: Path) -> dict:
    return {
        "runtime_dir": "submission",
        "runtime_inflate_py_sha256": RUNTIME_INFLATE_PY_SHA,
        "runtime_source_manifest": {
            "runtime_source_tree_sha256": "4" * 64,
            "runtime_content_tree_sha256": RUNTIME_CONTENT_TREE_SHA,
            "files": [
                {
                    "path": "inflate.py",
                    "sha256": RUNTIME_INFLATE_PY_SHA,
                    "bytes": 100,
                },
                {
                    "path": "src/codec.py",
                    "sha256": "1" * 64,
                    "bytes": 101,
                },
                {
                    "path": "src/model.py",
                    "sha256": "2" * 64,
                    "bytes": 102,
                },
                {
                    "path": "src/pr101_grammar.py",
                    "sha256": "3" * 64,
                    "bytes": 103,
                },
            ],
        },
        "schema": "pr106_same_runtime_streaming_frame_parity_v1",
        "proof_scope": "same_runtime_streaming_full_frame_hash",
        "device_axis_label": "local-cpu-streaming-runtime",
        "candidate_archive": {
            "sha256": _sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "source_archive": {
            "sha256": SHA_SOURCE,
            "bytes": archive.stat().st_size + 100,
        },
        "full_frame_inflate_output_parity_claim": True,
        "prefix_parity_claim": False,
        "streaming_output_sha256_equal": True,
        "streaming_output_total_bytes_equal": True,
        "candidate": {
            "streaming_raw_sha256": "8" * 64,
            "total_bytes": 9876,
            "n_pairs_hashed": 600,
            "n_pairs_total": 600,
            "full_frame_digest": True,
        },
        "source": {
            "streaming_raw_sha256": "8" * 64,
            "total_bytes": 9876,
            "n_pairs_hashed": 600,
            "n_pairs_total": 600,
            "full_frame_digest": True,
        },
        "score_claim": False,
        "contest_axis_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_zip(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("0.bin", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, payload)


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
