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


def _valid_candidate_queue_payload(
    *,
    archive_sha: str,
    archive_bytes: int,
    runtime_consumption_proven: bool = True,
) -> dict[str, object]:
    """Build a queue fixture using the v2 schema.

    Per codex adversarial review 2026-05-19 F1: ``runtime_consumption_proven``
    defaults True for the legacy-equivalent fixture (callers that want to
    exercise the runtime-unproven branch flip the kwarg to False).
    """

    return {
        "schema": "pr101_fec6_packetir_candidate_queue_v2",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "expected_archive_sha256_matches": True,
        "candidate_count": 1,
        "operator_candidate_count": 0,
        "blockers": [],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "byte_accounting": {
            "schema": "pr101_fec6_packetir_byte_accounting_v2",
            "member_payload_bytes": 11,
            "accounted_primary_payload_bytes": 11,
            "all_payload_bytes_accounted": True,
            "parser_byte_accounting_passed": True,
            "runtime_consumption_proven": runtime_consumption_proven,
            "runtime_consumed_byte_accounting_passed": runtime_consumption_proven,
            "parser_runtime_candidate_surfaces": ["inflate.py::parse"],
            "runtime_consumer_surfaces": ["inflate.py::parse"],
            "queue_consumer_surfaces": [
                "tac.cathedral_consumers.packetir_candidate_queue_consumer"
            ],
            "sections": [
                {
                    "name": "source_pr101_payload",
                    "offset": 0,
                    "length": 11,
                    "primary_payload_section": True,
                    "parser_section_runtime_candidate": True,
                    "runtime_consumption_proven": runtime_consumption_proven,
                    "runtime_consumed": runtime_consumption_proven,
                    "parser_runtime_candidate_surfaces": ["inflate.py::parse"],
                }
            ],
            "byte_accounting_blockers": (
                []
                if runtime_consumption_proven
                else ["runtime_byte_consumption_noop_detector_missing"]
            ),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "consumer_surfaces": [
            "tac.cathedral_consumers.packetir_candidate_queue_consumer"
        ],
        "candidates": [
            {
                "candidate_id": "identity",
                "candidate_kind": "identity_reference",
                "consumer_surfaces": [
                    "tac.cathedral_consumers.packetir_candidate_queue_consumer"
                ],
                "blockers": [],
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "ready_for_operator_probe": False,
                "ready_for_provider_dispatch": False,
                "dispatch_attempted": False,
            }
        ],
    }


def _valid_parser_section_manifest(
    *,
    archive_sha: str,
) -> dict[str, object]:
    """Canonical parser_section_manifest with one populated section."""

    return {
        "schema_version": "deterministic_parser_section_manifest.v1",
        "section_count": 1,
        "section_names": ["x"],
        "lengths": [11],
        "section_sha256s": [archive_sha],
        "offsets": [0],
        "compress_types": [0],
        "entropy_estimates": "per-member entropy deferred",
        "old_new_section_boundaries": "ZIP central directory",
    }


def _valid_golden_vectors(
    *,
    archive_sha: str,
    runtime_tree_sha: str,
) -> dict[str, object]:
    """Canonical golden_vectors with one populated member_vector."""

    return {
        "schema_version": "deterministic_golden_vectors.v1",
        "tool_name": "deterministic_packet_compiler",
        "tool_schema_version": "deterministic_packet_compiler.v1",
        "mode": "identity",
        "target_profile": "contest_one_video_replay",
        "archive_sha256": archive_sha,
        "runtime_tree_sha256": runtime_tree_sha,
        "member_vectors": [
            {
                "name": "x",
                "payload_sha256": archive_sha,
                "compressed_payload_sha256": archive_sha,
                "compress_type": 0,
                "uncompressed_bytes": 11,
                "compressed_bytes": 11,
                "data_offset": 0,
            }
        ],
    }


def _valid_compiler_manifest_payload(
    *,
    archive_sha: str,
    archive_bytes: int,
    parser_section_manifest: dict[str, object] | None = None,
    golden_vectors: dict[str, object] | None = None,
    runtime_tree_sha: str = "1" * 64,
) -> dict[str, object]:
    """Canonical compiler manifest fixture per codex F2 schema validation.

    Defaults populate parser_section_manifest + golden_vectors with valid
    schema-conformant data.  Tests that need to exercise the F2 blocker
    branches override either kwarg with `{}` or a malformed mapping.
    """

    if parser_section_manifest is None:
        parser_section_manifest = _valid_parser_section_manifest(
            archive_sha=archive_sha
        )
    if golden_vectors is None:
        golden_vectors = _valid_golden_vectors(
            archive_sha=archive_sha,
            runtime_tree_sha=runtime_tree_sha,
        )
    return {
        "schema_version": "deterministic_packet_compiler.v1",
        "mode": "identity",
        "target_profile": "contest_one_video_replay",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "runtime_tree_sha256": runtime_tree_sha,
        "parser_section_manifest": parser_section_manifest,
        "golden_vectors": golden_vectors,
        "no_op_proof": {
            "schema_version": "deterministic_no_op_proof.v1",
            "mode": "identity",
            "new_archive_sha256": archive_sha,
            "new_archive_size_bytes": archive_bytes,
            "baseline_archive_sha256": archive_sha,
            "baseline_archive_size_bytes": None,
            "no_op_detector_passed": True,
        },
        "blockers": [],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _fixture_spec(
    tmp_path: Path,
    *,
    include_identity: bool = True,
    include_queue: bool = False,
    include_compiler: bool = False,
) -> PR101FEC6FrontierMatrixSpec:
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

    identity_path = tmp_path / "fec6" / "packetir_identity_proof.json"
    if include_identity:
        _write_json(
            identity_path,
            {
                "schema": "pr101_fec6_packetir_identity_proof_v1",
                "archive_sha256": archive_sha,
                "member_name": "x",
                "member_bytes": len(archive_bytes),
                "member_sha256": archive_sha,
                "packet_ir_identity_passed": True,
                "reemit_identity": True,
                "member_reemit_identity": True,
                "archive_reemit_identity": True,
                "runtime_consumption_claim": False,
                "full_frame_inflate_output_parity_claim": False,
                "contest_axis_claim": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )

    queue_path = tmp_path / "fec6" / "packetir_candidate_queue.json"
    if include_queue:
        _write_json(
            queue_path,
            _valid_candidate_queue_payload(
                archive_sha=archive_sha,
                archive_bytes=len(archive_bytes),
            ),
        )

    compiler_path = tmp_path / "fec6" / "deterministic_packet_compiler_manifest.json"
    if include_compiler:
        _write_json(
            compiler_path,
            _valid_compiler_manifest_payload(
                archive_sha=archive_sha,
                archive_bytes=len(archive_bytes),
            ),
        )

    return PR101FEC6FrontierMatrixSpec(
        archive_path=str(archive),
        archive_manifest_path=str(tmp_path / "fec6" / "archive_manifest.json"),
        packet_manifest_path=str(tmp_path / "fec6" / "packet_manifest.json"),
        packetir_identity_proof_path=str(identity_path),
        deterministic_compiler_manifest_path=str(
            compiler_path
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
    assert summary["fec6_has_paired_exact_same_archive_runtime"] is True
    assert summary["fec6_has_parser_profile_evidence"] is True
    assert summary["fec6_has_packetir_identity_evidence"] is True
    assert summary["fec6_has_deterministic_compiler_identity_evidence"] is False
    assert summary["fec6_has_pr106_style_packetir_candidate_queue"] is False
    assert summary["fec6_has_candidate_byte_accounting_evidence"] is False
    assert (
        matrix["status"]
        == "parser_profile_no_compiler_identity_no_packetir_candidate_queue"
    )
    assert "deterministic_compiler_identity_manifest_missing" in matrix["blockers"]
    assert "pr106_style_packetir_candidate_queue_missing" in matrix["blockers"]
    assert "packetir_identity_proof_missing_or_blocked" not in matrix["blockers"]
    assert "paired_exact_same_archive_runtime_missing_or_blocked" not in matrix["blockers"]
    assert matrix["archive"]["sha256_matches_manifest"] is True
    assert matrix["packetir_identity_proof"]["packetir_identity_passed"] is True
    assert matrix["paired_exact_eval"]["paired_exact_same_archive_runtime"] is True
    assert (
        matrix["exact_eval_artifacts"]["contest_cpu"]["inflated_output_manifest"][
            "aggregate_sha256"
        ]
        == "d" * 64
    )
    assert matrix["next_actions"][0]["id"] == "run_compile_packet_identity_closure"
    assert matrix["next_actions"][0]["status"] == "pending"
    assert matrix["next_actions"][2]["id"] == "prove_parser_consumption_and_byte_accounting"
    assert matrix["next_actions"][2]["status"] == "pending"
    assert all(action["score_claim"] is False for action in matrix["next_actions"])


def test_pr101_fec6_matrix_records_candidate_queue_when_present(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path, include_queue=True)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    summary = matrix["authority_summary"]
    assert summary["fec6_has_pr106_style_packetir_candidate_queue"] is True
    assert summary["fec6_has_candidate_byte_accounting_evidence"] is True
    assert matrix["candidate_queue"]["exists"] is True
    assert matrix["candidate_queue"]["candidate_byte_accounting_present"] is True
    assert matrix["candidate_queue"]["score_claim"] is False
    assert matrix["candidate_queue"]["score_claim_valid"] is False
    assert (
        matrix["status"]
        == "parser_profile_no_compiler_identity_candidate_queue_present_needs_review"
    )
    assert "deterministic_compiler_identity_manifest_missing" in matrix["blockers"]
    assert "pr106_style_packetir_candidate_queue_missing" not in matrix["blockers"]
    assert matrix["next_actions"][0]["status"] == "pending"
    assert matrix["next_actions"][1]["status"] == "done"
    assert matrix["next_actions"][2]["status"] == "done"
    assert matrix["next_actions"][4]["status"] == "done"


def test_pr101_fec6_matrix_records_compiler_and_queue_when_valid(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path, include_queue=True, include_compiler=True)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    summary = matrix["authority_summary"]
    assert summary["fec6_has_deterministic_compiler_identity_evidence"] is True
    assert summary["fec6_has_pr106_style_packetir_candidate_queue"] is True
    assert summary["fec6_has_candidate_byte_accounting_evidence"] is True
    assert (
        summary["current_authority"]
        == "packetir_identity_deterministic_compiler_identity_and_candidate_queue_validated_no_score_claim"
    )
    assert matrix["status"] == "packetir_compiler_identity_and_candidate_queue_validated"
    assert "deterministic_compiler_identity_manifest_missing" not in matrix["blockers"]
    assert "pr106_style_packetir_candidate_queue_missing" not in matrix["blockers"]
    assert matrix["next_actions"][0]["status"] == "done"
    assert matrix["next_actions"][1]["status"] == "done"
    assert matrix["next_actions"][2]["status"] == "done"
    assert matrix["next_actions"][4]["id"] == "local_identity_profile_smoke"
    assert matrix["next_actions"][4]["status"] == "done"


def test_pr101_fec6_matrix_separates_parser_queue_from_runtime_consumption(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path, include_queue=True, include_compiler=True)
    queue_path = Path(spec.candidate_queue_path)
    payload = _valid_candidate_queue_payload(
        archive_sha=_sha(Path(spec.archive_path).read_bytes()),
        archive_bytes=Path(spec.archive_path).stat().st_size,
        runtime_consumption_proven=False,
    )
    payload["blockers"] = ["runtime_byte_consumption_noop_detector_missing"]
    _write_json(queue_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    summary = matrix["authority_summary"]
    assert summary["fec6_has_packetir_candidate_queue_artifact"] is True
    assert summary["fec6_has_parser_byte_accounting_evidence"] is True
    assert summary["fec6_has_runtime_consumption_evidence"] is False
    assert summary["fec6_has_pr106_style_packetir_candidate_queue"] is False
    assert summary["fec6_has_candidate_byte_accounting_evidence"] is False
    assert matrix["candidate_queue"]["candidate_queue_generated"] is True
    assert matrix["candidate_queue"]["parser_byte_accounting_present"] is True
    assert matrix["candidate_queue"]["runtime_consumption_proven"] is False
    assert (
        matrix["status"]
        == "packetir_candidate_queue_runtime_consumption_unproven"
    )
    assert "pr106_style_packetir_candidate_queue_missing" in matrix["blockers"]
    assert (
        "packetir_candidate_runtime_consumption_missing_or_blocked"
        in matrix["blockers"]
    )
    assert matrix["next_actions"][1]["status"] == "done"
    assert matrix["next_actions"][2]["status"] == "done"
    assert matrix["next_actions"][3]["id"] == "prove_runtime_byte_consumption_noop_detector"
    assert matrix["next_actions"][3]["status"] == "pending"
    assert matrix["next_actions"][4]["status"] == "blocked"


def test_pr101_fec6_matrix_blocks_overclaiming_candidate_queue(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path, include_queue=True)
    queue_path = Path(spec.candidate_queue_path)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    payload["score_claim_valid"] = True
    payload["candidates"][0]["ready_for_exact_eval_dispatch"] = True
    _write_json(queue_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    assert matrix["authority_summary"]["fec6_has_pr106_style_packetir_candidate_queue"] is False
    assert "pr106_style_packetir_candidate_queue_missing" in matrix["blockers"]
    assert any(
        "ready_for_exact_eval_dispatch_overclaimed" in blocker
        for blocker in matrix["candidate_queue"]["blockers"]
    )
    assert "candidate_queue_score_claim_valid_overclaimed" in matrix["candidate_queue"][
        "blockers"
    ]


def test_pr101_fec6_matrix_blocks_bad_compiler_manifest(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["no_op_proof"]["no_op_detector_passed"] = False
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    assert (
        matrix["authority_summary"]["fec6_has_deterministic_compiler_identity_evidence"]
        is False
    )
    assert "deterministic_compiler_identity_manifest_missing" in matrix["blockers"]
    assert (
        "deterministic_compiler_no_op_detector_not_passed"
        in matrix["deterministic_compiler_manifest"]["blockers"]
    )


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
        packetir_identity_proof_path=spec.packetir_identity_proof_path,
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
    assert "paired_exact_same_archive_runtime_missing_or_blocked" in matrix["blockers"]


def test_pr101_fec6_matrix_blocks_mismatched_exact_eval_runtime(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path)
    archive_sha = Path(spec.archive_path).read_bytes()
    bad_cuda = tmp_path / "eval" / "cuda_runtime_bad" / "contest_auth_eval.json"
    _write_json(
        bad_cuda,
        _exact_eval_payload(
            axis="contest_cuda",
            archive_sha=_sha(archive_sha),
            runtime_content_sha="9" * 64,
        ),
    )
    spec = PR101FEC6FrontierMatrixSpec(
        archive_path=spec.archive_path,
        archive_manifest_path=spec.archive_manifest_path,
        packet_manifest_path=spec.packet_manifest_path,
        packetir_identity_proof_path=spec.packetir_identity_proof_path,
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

    assert matrix["exact_eval_artifacts"]["contest_cuda"]["valid_axis_evidence"] is True
    assert matrix["authority_summary"]["fec6_has_contest_cuda_evidence"] is True
    assert (
        matrix["authority_summary"]["fec6_has_paired_exact_same_archive_runtime"]
        is False
    )
    assert (
        "paired_exact_runtime_content_sha256_mismatch_or_missing"
        in matrix["paired_exact_eval"]["blockers"]
    )
    assert "paired_exact_same_archive_runtime_missing_or_blocked" in matrix["blockers"]


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
    assert "PacketIR identity evidence: `True`" in markdown


def test_pr101_fec6_matrix_blocks_overclaiming_identity_proof(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    identity_path = Path(spec.packetir_identity_proof_path)
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    payload["reemit_identity"] = False
    payload["contest_axis_claim"] = True
    payload["runtime_consumption_claim"] = True
    _write_json(identity_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    identity = matrix["packetir_identity_proof"]
    assert identity["packetir_identity_passed"] is False
    assert "identity_proof_reemit_identity_not_true" in identity["blockers"]
    assert "identity_proof_overclaims_contest_axis" in identity["blockers"]
    assert "identity_proof_overclaims_runtime_consumption" in identity["blockers"]
    assert "packetir_identity_proof_missing_or_blocked" in matrix["blockers"]
    assert matrix["authority_summary"]["fec6_has_packetir_identity_evidence"] is False


def test_pr101_fec6_matrix_blocks_missing_identity_proof(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path, include_identity=False)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    assert matrix["authority_summary"]["fec6_has_packetir_identity_evidence"] is False
    assert "packetir_identity_proof_missing_or_blocked" in matrix["blockers"]
    assert matrix["next_actions"][2]["id"] == "prove_parser_consumption_and_byte_accounting"
    assert matrix["next_actions"][2]["status"] == "pending"


def test_pr101_fec6_matrix_markdown_is_non_dispatching(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    markdown = render_pr101_frontier_packetir_matrix_markdown(matrix)

    assert "ready_for_exact_eval_dispatch=false" in markdown
    assert "dispatch commands are emitted" in markdown
    assert "contest_cpu" in markdown
    assert "contest_cuda" in markdown


# --- Codex adversarial review 2026-05-19 F2 regression tests ---


def test_pr101_fec6_matrix_blocks_empty_parser_section_manifest(
    tmp_path: Path,
) -> None:
    """Empty `{}` parser_section_manifest must add a blocker (F2 regression).

    Pre-fix the v1 isinstance(_, Mapping) check accepted `{}` as valid.
    """

    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parser_section_manifest"] = {}
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    compiler_blockers = matrix["deterministic_compiler_manifest"]["blockers"]
    assert "deterministic_compiler_parser_manifest_schema_mismatch" in compiler_blockers
    assert (
        matrix["authority_summary"]["fec6_has_deterministic_compiler_identity_evidence"]
        is False
    )


def test_pr101_fec6_matrix_blocks_empty_golden_vectors(
    tmp_path: Path,
) -> None:
    """Empty `{}` golden_vectors must add a blocker (F2 regression)."""

    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["golden_vectors"] = {}
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    compiler_blockers = matrix["deterministic_compiler_manifest"]["blockers"]
    assert (
        "deterministic_compiler_golden_vectors_schema_mismatch" in compiler_blockers
    )
    assert (
        matrix["authority_summary"]["fec6_has_deterministic_compiler_identity_evidence"]
        is False
    )


def test_pr101_fec6_matrix_blocks_golden_vectors_empty_member_list(
    tmp_path: Path,
) -> None:
    """golden_vectors with empty member_vectors list must add a blocker."""

    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["golden_vectors"]["member_vectors"] = []
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    compiler_blockers = matrix["deterministic_compiler_manifest"]["blockers"]
    assert (
        "deterministic_compiler_golden_vectors_member_vectors_empty"
        in compiler_blockers
    )


def test_pr101_fec6_matrix_blocks_golden_vectors_archive_sha_mismatch(
    tmp_path: Path,
) -> None:
    """golden_vectors archive_sha256 must equal top-level archive_sha256."""

    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["golden_vectors"]["archive_sha256"] = "9" * 64
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    compiler_blockers = matrix["deterministic_compiler_manifest"]["blockers"]
    assert (
        "deterministic_compiler_golden_vectors_archive_sha_mismatch"
        in compiler_blockers
    )


def test_pr101_fec6_matrix_blocks_golden_vectors_runtime_tree_sha_mismatch(
    tmp_path: Path,
) -> None:
    """golden_vectors runtime_tree_sha256 must equal top-level value."""

    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["golden_vectors"]["runtime_tree_sha256"] = "a" * 64
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    compiler_blockers = matrix["deterministic_compiler_manifest"]["blockers"]
    assert (
        "deterministic_compiler_golden_vectors_runtime_tree_sha_mismatch"
        in compiler_blockers
    )


def test_pr101_fec6_matrix_blocks_parser_manifest_section_count_invalid(
    tmp_path: Path,
) -> None:
    """parser_section_manifest must declare section_count >= 1."""

    spec = _fixture_spec(tmp_path, include_compiler=True)
    manifest_path = Path(spec.deterministic_compiler_manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parser_section_manifest"]["section_count"] = 0
    _write_json(manifest_path, payload)

    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)

    compiler_blockers = matrix["deterministic_compiler_manifest"]["blockers"]
    assert (
        "deterministic_compiler_parser_manifest_section_count_invalid"
        in compiler_blockers
    )


def test_pr101_fec6_matrix_accepts_canonical_compiler_manifest(
    tmp_path: Path,
) -> None:
    """Sanity: canonical fixture data still PASSES (no regression)."""

    spec = _fixture_spec(
        tmp_path,
        include_queue=True,
        include_compiler=True,
    )
    matrix = build_pr101_frontier_packetir_matrix(repo_root=tmp_path, spec=spec)
    assert (
        matrix["authority_summary"]["fec6_has_deterministic_compiler_identity_evidence"]
        is True
    )
