from __future__ import annotations

import math
import zipfile
from pathlib import Path

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


def _full_frame_parity(archive: Path) -> dict:
    return {
        "runtime_dir": "submission",
        "runtime_inflate_py_sha256": RUNTIME_INFLATE_PY_SHA,
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
