# SPDX-License-Identifier: MIT
"""Tests for the PR101/FEC6 frontier PacketIR authority matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.packet_compiler.pr101_frontier_packetir_matrix import (
    PR101_FRONTIER_PACKETIR_MATRIX_SCHEMA,
    PR101FEC6FrontierMatrixSpec,
    build_pr101_frontier_packetir_matrix,
    render_pr101_frontier_packetir_matrix_markdown,
    write_pr101_frontier_packetir_matrix,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str = "artifact\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _exact_eval_payload(
    *,
    axis: str,
    archive_sha: str,
    runtime_content_sha: str,
) -> dict[str, object]:
    device = "cpu" if axis == "contest_cpu" else "cuda"
    return {
        "schema_version": 1,
        "score_axis": axis,
        "canonical_score": 0.192 if axis == "contest_cpu" else 0.226,
        "score_recomputed_from_components": 0.192
        if axis == "contest_cpu"
        else 0.226,
        "archive_size_bytes": 123,
        "avg_segnet_dist": 0.0005,
        "avg_posenet_dist": 0.00003,
        "n_samples": 600,
        "evidence_grade": "contest-CPU" if axis == "contest_cpu" else "contest-CUDA",
        "score_claim": False,
        "score_claim_valid": axis != "contest_cpu",
        "promotion_eligible": False,
        "provenance": {
            "archive_sha256": archive_sha,
            "device": device,
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "gpu_model": "Tesla T4" if axis == "contest_cuda" else "",
            "gpu_t4_match": axis == "contest_cuda",
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": ("a" if axis == "contest_cpu" else "b") * 64,
                "runtime_content_tree_sha256": runtime_content_sha,
            },
            "inflated_output_manifest": {
                "path": "/tmp/inflated_outputs_manifest.json",
                "sha256": "c" * 64,
                "payload": {
                    "aggregate_sha256": "d" * 64,
                    "raw_file_count": 1,
                    "total_bytes": 456,
                },
            },
        },
    }


def _fixture_spec(tmp_path: Path, *, include_queue: bool = False) -> PR101FEC6FrontierMatrixSpec:
    archive_bytes = b"fec6 archive bytes"
    archive_sha = _sha(archive_bytes)
    archive = tmp_path / "fec6" / "archive.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(archive_bytes)

    _write_json(
        tmp_path / "fec6" / "archive_manifest.json",
        {
            "schema": "archive_manifest",
            "archive_bytes": len(archive_bytes),
            "archive_sha256": archive_sha,
            "source_archive_sha256": "e" * 64,
            "score_claim": False,
        },
    )
    _write_json(
        tmp_path / "fec6" / "packet_manifest.json",
        {
            "schema": "packet_manifest",
            "archive": {"bytes": len(archive_bytes), "sha256": archive_sha},
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    runtime_content_sha = "f" * 64
    cpu_eval = tmp_path / "eval" / "cpu" / "contest_auth_eval.json"
    cuda_eval = tmp_path / "eval" / "cuda" / "contest_auth_eval.json"
    _write_json(
        cpu_eval,
        _exact_eval_payload(
            axis="contest_cpu",
            archive_sha=archive_sha,
            runtime_content_sha=runtime_content_sha,
        ),
    )
    _write_json(
        cuda_eval,
        _exact_eval_payload(
            axis="contest_cuda",
            archive_sha=archive_sha,
            runtime_content_sha=runtime_content_sha,
        ),
    )

    profile_json = tmp_path / "profiles" / "profile.json"
    parser_md = tmp_path / "profiles" / "parser.md"
    _write_json(profile_json, {"schema": "profile", "score_claim": False})
    _write_text(parser_md, "parser evidence\n")

    queue_path = tmp_path / "fec6" / "packetir_candidate_queue.json"
    if include_queue:
        _write_json(
            queue_path,
            {
                "schema": "pr101_fec6_packetir_candidate_queue_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )

    return PR101FEC6FrontierMatrixSpec(
        archive_path=str(archive),
        archive_manifest_path=str(tmp_path / "fec6" / "archive_manifest.json"),
        packet_manifest_path=str(tmp_path / "fec6" / "packet_manifest.json"),
        deterministic_compiler_manifest_path=str(
            tmp_path / "fec6" / "deterministic_packet_compiler_manifest.json"
        ),
        candidate_queue_path=str(queue_path),
        exact_eval_paths={
            "contest_cpu": str(cpu_eval),
            "contest_cuda": str(cuda_eval),
        },
        parser_profile_paths=(str(profile_json), str(parser_md)),
    )


def test_pr101_fec6_matrix_answers_authority_without_candidate_queue(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    assert matrix["schema"] == PR101_FRONTIER_PACKETIR_MATRIX_SCHEMA
    assert matrix["score_claim"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    assert matrix["dispatch_attempted"] is False
    summary = matrix["authority_summary"]
    assert summary["pr101_packet_compiler_packetir_primitives_are_general"] is True
    assert summary["fec6_has_contest_cpu_evidence"] is True
    assert summary["fec6_has_contest_cuda_evidence"] is True
    assert summary["fec6_has_parser_profile_evidence"] is True
    assert summary["fec6_has_deterministic_compiler_identity_evidence"] is False
    assert summary["fec6_has_pr106_style_packetir_candidate_queue"] is False
    assert (
        matrix["status"]
        == "parser_profile_no_compiler_identity_no_packetir_candidate_queue"
    )
    assert "deterministic_compiler_identity_manifest_missing" in matrix["blockers"]
    assert "pr106_style_packetir_candidate_queue_missing" in matrix["blockers"]
    assert matrix["archive"]["sha256_matches_manifest"] is True
    assert (
        matrix["exact_eval_artifacts"]["contest_cpu"]["inflated_output_manifest"][
            "aggregate_sha256"
        ]
        == "d" * 64
    )
    assert matrix["next_actions"][0]["id"] == "run_compile_packet_identity_closure"
    assert matrix["next_actions"][0]["status"] == "pending"
    assert all(action["score_claim"] is False for action in matrix["next_actions"])


def test_pr101_fec6_matrix_records_candidate_queue_when_present(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path, include_queue=True)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    summary = matrix["authority_summary"]
    assert summary["fec6_has_pr106_style_packetir_candidate_queue"] is True
    assert matrix["candidate_queue"]["exists"] is True
    assert matrix["candidate_queue"]["score_claim"] is False
    assert (
        matrix["status"]
        == "parser_profile_no_compiler_identity_candidate_queue_present_needs_review"
    )
    assert "deterministic_compiler_identity_manifest_missing" in matrix["blockers"]
    assert "pr106_style_packetir_candidate_queue_missing" not in matrix["blockers"]
    assert matrix["next_actions"][0]["status"] == "pending"
    assert matrix["next_actions"][1]["status"] == "done"


def test_pr101_fec6_matrix_blocks_mismatched_exact_eval_archive(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path)
    bad_cuda = tmp_path / "eval" / "cuda_bad" / "contest_auth_eval.json"
    _write_json(
        bad_cuda,
        _exact_eval_payload(
            axis="contest_cuda",
            archive_sha="0" * 64,
            runtime_content_sha="f" * 64,
        ),
    )
    spec = PR101FEC6FrontierMatrixSpec(
        archive_path=spec.archive_path,
        archive_manifest_path=spec.archive_manifest_path,
        packet_manifest_path=spec.packet_manifest_path,
        deterministic_compiler_manifest_path=(
            spec.deterministic_compiler_manifest_path
        ),
        candidate_queue_path=spec.candidate_queue_path,
        exact_eval_paths={
            "contest_cpu": spec.exact_eval_paths["contest_cpu"],
            "contest_cuda": str(bad_cuda),
        },
        parser_profile_paths=spec.parser_profile_paths,
    )

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    cuda = matrix["exact_eval_artifacts"]["contest_cuda"]
    assert cuda["valid_axis_evidence"] is False
    assert "archive_sha256_mismatch" in cuda["blockers"]
    assert "contest_cuda_evidence_missing_or_blocked" in matrix["blockers"]


def test_pr101_fec6_matrix_writes_json_and_markdown(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    output_json = tmp_path / "out" / "matrix.json"
    output_md = tmp_path / "out" / "matrix.md"

    matrix = write_pr101_frontier_packetir_matrix(
        output_json=output_json,
        output_md=output_md,
        repo_root=tmp_path,
        spec=spec,
    )

    assert output_json.is_file()
    assert output_md.is_file()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["schema"] == PR101_FRONTIER_PACKETIR_MATRIX_SCHEMA
    assert loaded["artifact_paths"] == matrix["artifact_paths"]
    assert loaded["artifact_sha256"]["json"] is None
    assert (
        loaded["artifact_sha256"]["json_omitted_reason"]
        == "self_referential_json_hash_requires_external_manifest"
    )
    assert matrix["written_artifact_sha256"]["json"] == _sha(output_json.read_bytes())
    markdown = output_md.read_text(encoding="utf-8")
    assert "PR101/FEC6 frontier PacketIR authority matrix" in markdown
    assert "score_claim=false" in markdown
    assert "PR106-style PacketIR candidate queue: `False`" in markdown
    assert "run_compile_packet_identity_closure" in markdown


def test_pr101_fec6_matrix_markdown_is_non_dispatching(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    markdown = render_pr101_frontier_packetir_matrix_markdown(matrix)

    assert "ready_for_exact_eval_dispatch=false" in markdown
    assert "dispatch commands are emitted" in markdown
    assert "contest_cpu" in markdown
    assert "contest_cuda" in markdown
