# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np

from tac.optimization.family_agnostic_materializers import (
    ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
    TENSOR_FACTORIZE_SCHEMA,
    materialize_archive_section_entropy_recode_candidate,
    materialize_packet_member_recompress_candidate,
    materialize_packet_member_zip_header_elide_candidate,
    materialize_tensor_factorize_candidate,
)
from tac.repo_io import sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_zip(path: Path, members: dict[str, bytes], *, compression: int = zipfile.ZIP_STORED) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def _write_zip_with_member_header_overhead(
    path: Path,
    *,
    payload: bytes,
    member_name: str = "payload.bin",
    compression: int = zipfile.ZIP_STORED,
) -> None:
    info = zipfile.ZipInfo(member_name)
    info.compress_type = compression
    info.extra = b"\x7f\x7f\x04\x00abcd"
    info.comment = b"deterministic member comment"
    with zipfile.ZipFile(path, "w") as zf:
        zf.comment = b"deterministic archive comment"
        zf.writestr(info, payload)


def test_packet_member_recompress_materializer_preserves_member_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    payload = b"A" * 8192
    _write_zip(archive, {"payload.bin": payload})

    result = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        repo_root=tmp_path,
    )

    assert result["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["source_member"]["sha256"] == sha256_bytes(payload)
    assert result["candidate_member"]["sha256"] == sha256_bytes(payload)
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["selected_compression"]["saved_bytes"] > 0
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in result["readiness_blockers"]


def test_packet_member_recompress_materializer_emits_payload_identity_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    proof = tmp_path / "runtime_consumption_proof.json"
    payload = b"A" * 8192
    _write_zip(archive, {"payload.bin": payload})

    result = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        runtime_consumption_proof_out=proof,
        repo_root=tmp_path,
    )

    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert proof_payload["schema"] == "family_agnostic_runtime_consumption_proof_v1"
    assert proof_payload["proof_kind"] == (
        "packet_member_recompress_payload_identity_receiver_proof.v1"
    )
    assert proof_payload["candidate_archive"]["sha256"] == result["candidate_archive"]["sha256"]
    assert proof_payload["candidate_member"]["sha256"] == result["candidate_member"]["sha256"]
    assert proof_payload["candidate_member_payload_identical_to_source"] is True
    assert proof_payload["receiver_contract_satisfied"] is True
    assert proof_payload["score_claim"] is False
    assert proof_payload["ready_for_exact_eval_dispatch"] is False
    assert result["receiver_contract_satisfied"] is True
    assert result["receiver_verification"]["proof_present"] is True
    assert result["receiver_verification"]["candidate_archive_sha256"] == (
        result["candidate_archive"]["sha256"]
    )
    assert result["receiver_verification"]["candidate_member_sha256"] == (
        result["candidate_member"]["sha256"]
    )
    assert "runtime_consumption_proof_missing" not in result["readiness_blockers"]
    assert "packet_member_recompress_receiver_contract_not_satisfied" not in (
        result["readiness_blockers"]
    )
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_packet_member_recompress_materializer_rejects_stale_member_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    first_output = tmp_path / "candidate_a.zip"
    second_output = tmp_path / "candidate_b.zip"
    proof = tmp_path / "stale_runtime_consumption_proof.json"
    payload = b"A" * 8192
    _write_zip(archive, {"payload.bin": payload})
    anchor = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=first_output,
        member_name="payload.bin",
        runtime_consumption_proof_out=tmp_path / "anchor_proof.json",
        repo_root=tmp_path,
    )
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "proof_kind": "packet_member_recompress_payload_identity_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "candidate_archive_sha256": anchor["candidate_archive"]["sha256"],
                "candidate_member_sha256": "0" * 64,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            }
        ),
        encoding="utf-8",
    )

    result = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=second_output,
        member_name="payload.bin",
        runtime_consumption_proof=proof,
        repo_root=tmp_path,
    )

    assert result["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_candidate_member_sha_mismatch" in (
        result["readiness_blockers"]
    )


def test_packet_member_recompress_materializer_rejects_truthy_proof_authority(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    first_output = tmp_path / "candidate_a.zip"
    second_output = tmp_path / "candidate_b.zip"
    proof = tmp_path / "authority_leaking_runtime_consumption_proof.json"
    payload = b"A" * 8192
    _write_zip(archive, {"payload.bin": payload})
    anchor = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=first_output,
        member_name="payload.bin",
        runtime_consumption_proof_out=tmp_path / "anchor_proof.json",
        repo_root=tmp_path,
    )
    proof.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "proof_kind": "packet_member_recompress_payload_identity_receiver_proof.v1",
                "receiver_contract_satisfied": True,
                "candidate_archive_sha256": anchor["candidate_archive"]["sha256"],
                "candidate_member_sha256": anchor["candidate_member"]["sha256"],
                "score_claim": True,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            }
        ),
        encoding="utf-8",
    )

    result = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=second_output,
        member_name="payload.bin",
        runtime_consumption_proof=proof,
        repo_root=tmp_path,
    )

    assert result["receiver_contract_satisfied"] is False
    assert any("score_claim=truthy" in blocker for blocker in result["readiness_blockers"])


def test_packet_member_recompress_materializer_preserves_stored_method_zero(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    payload = b"payload-bytes" * 8
    _write_zip(archive, {"payload.bin": payload})

    result = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        compression_methods=("stored",),
        allow_size_regression=True,
        repo_root=tmp_path,
    )

    assert result["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert result["selected_compression"]["compression_method"] == "stored"
    assert result["candidate_trials"][0]["compression_method"] == "stored"
    assert result["candidate_member"]["sha256"] == sha256_bytes(payload)
    assert result["byte_closed_candidate_emitted"] is True


def test_packet_member_zip_header_elide_materializer_preserves_payload(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    payload = b"header-elide-payload" * 64
    _write_zip_with_member_header_overhead(
        archive,
        payload=payload,
        compression=zipfile.ZIP_DEFLATED,
    )

    result = materialize_packet_member_zip_header_elide_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        repo_root=tmp_path,
    )

    assert result["schema"] == PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["source_member"]["sha256"] == sha256_bytes(payload)
    assert result["candidate_member"]["sha256"] == sha256_bytes(payload)
    assert result["candidate_member"]["zip_compressed_bytes"] == (
        result["source_member"]["zip_compressed_bytes"]
    )
    assert result["candidate_member"]["zip_compressed_sha256"] == (
        result["source_member"]["zip_compressed_sha256"]
    )
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["selected_elision"]["saved_bytes"] > 0
    assert result["selected_elision"]["elided_header_bytes"] > 0
    assert result["candidate_zip_header"]["total_elidable_header_bytes"] == 0
    assert "runtime_consumption_proof_missing" in result["readiness_blockers"]
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_packet_member_zip_header_elide_materializer_can_elide_all_members(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    payloads = {
        "renderer.bin": b"renderer-payload" * 64,
        "weights.bin": b"weights-payload" * 64,
    }
    with zipfile.ZipFile(archive, "w") as zf:
        zf.comment = b"archive comment"
        for name, payload in payloads.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.extra = b"\x7f\x7f\x04\x00abcd"
            info.comment = f"{name} comment".encode()
            zf.writestr(info, payload)

    result = materialize_packet_member_zip_header_elide_candidate(
        archive_path=archive,
        output_archive=output,
        all_members=True,
        runtime_consumption_proof_out=tmp_path / "runtime_consumption_proof.json",
        repo_root=tmp_path,
    )

    assert result["schema"] == PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA
    assert result["selection_scope"] == "all_members"
    assert result["selected_member_names"] == ["renderer.bin", "weights.bin"]
    assert result["source_zip_header_summary"]["member_count"] == 2
    assert result["candidate_zip_header_summary"]["total_elidable_header_bytes"] == 0
    assert result["selected_elision"]["saved_bytes"] > 0
    assert result["selected_elision"]["elided_header_bytes"] > (
        result["source_zip_header"]["total_elidable_header_bytes"]
    )
    source_by_name = {row["name"]: row for row in result["source_members"]}
    candidate_by_name = {row["name"]: row for row in result["candidate_members"]}
    for name, payload in payloads.items():
        assert source_by_name[name]["sha256"] == sha256_bytes(payload)
        assert candidate_by_name[name]["sha256"] == sha256_bytes(payload)
        assert candidate_by_name[name]["zip_compressed_bytes"] == (
            source_by_name[name]["zip_compressed_bytes"]
        )
        assert candidate_by_name[name]["zip_compressed_sha256"] == (
            source_by_name[name]["zip_compressed_sha256"]
        )
        assert candidate_by_name[name]["zip_compressed_payload_sha256"] == (
            source_by_name[name]["zip_compressed_payload_sha256"]
        )
    assert result["receiver_contract_satisfied"] is True
    proof_payload = json.loads(
        Path(result["runtime_consumption_proof_path"]).read_text(encoding="utf-8")
    )
    assert proof_payload["all_member_compressed_streams_identical_to_source"] is True
    assert result["receiver_verification"]["candidate_member_sha256s"] == {
        name: sha256_bytes(payload)
        for name, payload in payloads.items()
    }
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_packet_member_zip_header_elide_materializer_emits_runtime_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    proof = tmp_path / "runtime_consumption_proof.json"
    payload = b"header-elide-payload" * 64
    _write_zip_with_member_header_overhead(archive, payload=payload)

    result = materialize_packet_member_zip_header_elide_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        runtime_consumption_proof_out=proof,
        repo_root=tmp_path,
    )

    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert proof_payload["schema"] == "family_agnostic_runtime_consumption_proof_v1"
    assert proof_payload["proof_kind"] == (
        "packet_member_zip_header_elide_payload_identity_receiver_proof.v1"
    )
    assert proof_payload["candidate_archive"]["sha256"] == result["candidate_archive"]["sha256"]
    assert proof_payload["candidate_member"]["sha256"] == result["candidate_member"]["sha256"]
    assert proof_payload["candidate_member_payload_identical_to_source"] is True
    assert proof_payload["receiver_contract_satisfied"] is True
    assert result["receiver_contract_satisfied"] is True
    assert result["receiver_verification"]["proof_present"] is True
    assert result["receiver_verification"]["candidate_member_sha256"] == (
        result["candidate_member"]["sha256"]
    )
    assert "runtime_consumption_proof_missing" not in result["readiness_blockers"]
    assert "packet_member_zip_header_elide_receiver_contract_not_satisfied" not in (
        result["readiness_blockers"]
    )
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_archive_section_entropy_recode_materializer_uses_section_manifest(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    section_a = brotli.compress(b"A" * 4096, quality=0)
    section_b = brotli.compress(b"B" * 4096, quality=0)
    member_payload = section_a + section_b
    _write_zip(archive, {"0.raw": member_payload})
    manifest = {
        "schema": "fixture_section_manifest.v1",
        "member": {"name": "0.raw"},
        "sections": [
            {
                "name": "section_a",
                "index": 0,
                "offset": 0,
                "length": len(section_a),
                "sha256": sha256_bytes(section_a),
                "optimization_role": "decoder_weight_stream",
            },
            {
                "name": "section_b",
                "index": 1,
                "offset": len(section_a),
                "length": len(section_b),
                "sha256": sha256_bytes(section_b),
                "optimization_role": "latent_stream",
            },
        ],
    }

    result = materialize_archive_section_entropy_recode_candidate(
        archive_path=archive,
        section_manifest=manifest,
        output_archive=output,
        section_names=("section_a",),
        brotli_qualities=(11,),
        repo_root=tmp_path,
    )

    assert result["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["section_recode"]["changed_section_count"] == 1
    assert result["sections"][0]["raw_payload_sha256"] == sha256_bytes(b"A" * 4096)
    assert result["sections"][1]["selected"] is False
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in result["readiness_blockers"]


def test_archive_section_entropy_recode_materializer_emits_raw_identity_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    proof = tmp_path / "runtime_consumption_proof.json"
    raw_a = b"A" * 4096
    raw_b = b"B" * 4096
    section_a = brotli.compress(raw_a, quality=0)
    section_b = brotli.compress(raw_b, quality=0)
    member_payload = section_a + section_b
    _write_zip(archive, {"0.raw": member_payload})
    manifest = {
        "schema": "fixture_section_manifest.v1",
        "member": {"name": "0.raw"},
        "sections": [
            {
                "name": "section_a",
                "index": 0,
                "offset": 0,
                "length": len(section_a),
                "sha256": sha256_bytes(section_a),
            },
            {
                "name": "section_b",
                "index": 1,
                "offset": len(section_a),
                "length": len(section_b),
                "sha256": sha256_bytes(section_b),
            },
        ],
    }

    result = materialize_archive_section_entropy_recode_candidate(
        archive_path=archive,
        section_manifest=manifest,
        output_archive=output,
        section_names=("section_a",),
        brotli_qualities=(11,),
        runtime_consumption_proof_out=proof,
        repo_root=tmp_path,
    )

    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert proof_payload["schema"] == "family_agnostic_runtime_consumption_proof_v1"
    assert proof_payload["proof_kind"] == (
        "archive_section_entropy_recode_raw_payload_identity_receiver_proof.v1"
    )
    assert proof_payload["candidate_archive"]["sha256"] == result["candidate_archive"]["sha256"]
    assert proof_payload["candidate_member"]["sha256"] == result["candidate_member"]["sha256"]
    assert proof_payload["section_proofs"][0]["source_raw_payload_sha256"] == sha256_bytes(raw_a)
    assert proof_payload["section_proofs"][0]["candidate_raw_payload_sha256"] == sha256_bytes(raw_a)
    assert proof_payload["all_selected_sections_raw_payload_identical"] is True
    assert proof_payload["all_selected_section_lengths_preserved"] is False
    assert result["receiver_contract_satisfied"] is False
    assert result["receiver_verification"]["proof_present"] is True
    assert result["receiver_verification"]["candidate_member_sha256"] == (
        result["candidate_member"]["sha256"]
    )
    assert "runtime_consumption_proof_missing" not in result["readiness_blockers"]
    assert "section_length_changed_requires_runtime_consumption_proof" in (
        result["readiness_blockers"]
    )
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_archive_section_entropy_recode_materializer_accepts_length_preserved_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    proof = tmp_path / "runtime_consumption_proof.json"
    raw = b"already-quality-zero" * 256
    section = brotli.compress(raw, quality=0)
    _write_zip(archive, {"0.raw": section})
    manifest = {
        "schema": "fixture_section_manifest.v1",
        "member": {"name": "0.raw"},
        "sections": [
            {
                "name": "section_a",
                "index": 0,
                "offset": 0,
                "length": len(section),
                "sha256": sha256_bytes(section),
            }
        ],
    }

    result = materialize_archive_section_entropy_recode_candidate(
        archive_path=archive,
        section_manifest=manifest,
        output_archive=output,
        section_names=("section_a",),
        brotli_qualities=(0,),
        runtime_consumption_proof_out=proof,
        allow_size_regression=True,
        repo_root=tmp_path,
    )

    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert proof_payload["all_selected_sections_raw_payload_identical"] is True
    assert proof_payload["all_selected_section_lengths_preserved"] is True
    assert result["receiver_contract_satisfied"] is True
    assert "archive_section_entropy_recode_receiver_contract_not_satisfied" not in (
        result["readiness_blockers"]
    )
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_archive_section_entropy_recode_materializer_preserves_brotli_quality_zero(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    raw = b"quality-zero-still-valid" * 256
    section = brotli.compress(raw, quality=11)
    _write_zip(archive, {"0.raw": section})
    manifest = {
        "schema": "fixture_section_manifest.v1",
        "member": {"name": "0.raw"},
        "sections": [
            {
                "name": "section_a",
                "index": 0,
                "offset": 0,
                "length": len(section),
                "sha256": sha256_bytes(section),
            }
        ],
    }

    result = materialize_archive_section_entropy_recode_candidate(
        archive_path=archive,
        section_manifest=manifest,
        output_archive=output,
        section_names=("section_a",),
        brotli_qualities=(0,),
        allow_size_regression=True,
        repo_root=tmp_path,
    )

    assert result["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert result["sections"][0]["candidate_quality"] == 0
    assert result["sections"][0]["raw_payload_sha256"] == sha256_bytes(raw)
    assert result["byte_closed_candidate_emitted"] is True


def test_tensor_factorize_materializer_emits_cooperative_receiver_packet(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    tensor_path = tmp_path / "tensor.npy"
    vector_a = np.arange(256, dtype=np.float32)[:, None]
    vector_b = np.linspace(0.25, 2.0, 256, dtype=np.float32)[None, :]
    np.save(tensor_path, vector_a @ vector_b)
    _write_zip(archive, {"weights.npy": tensor_path.read_bytes()})

    result = materialize_tensor_factorize_candidate(
        archive_path=archive,
        tensor_manifest={"member_name": "weights.npy"},
        factorization_contract={"rank": 1},
        output_archive=output,
        repo_root=tmp_path,
    )

    assert result["schema"] == TENSOR_FACTORIZE_SCHEMA
    assert result["byte_closed_candidate_emitted"] is True
    assert result["factorization"]["rank"] == 1
    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert "tensor_factorized_payload_requires_cooperative_receiver" in (
        result["readiness_blockers"]
    )


def test_tensor_factorize_materializer_emits_cooperative_receiver_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    proof = tmp_path / "runtime_consumption_proof.json"
    tensor_path = tmp_path / "tensor.npy"
    vector_a = np.arange(256, dtype=np.float32)[:, None]
    vector_b = np.linspace(0.25, 2.0, 256, dtype=np.float32)[None, :]
    np.save(tensor_path, vector_a @ vector_b)
    _write_zip(archive, {"weights.npy": tensor_path.read_bytes()})

    result = materialize_tensor_factorize_candidate(
        archive_path=archive,
        tensor_manifest={"member_name": "weights.npy"},
        factorization_contract={
            "rank": 1,
            "cooperative_receiver_id": "fixture_tensor_factorize_receiver",
            "receiver_adapter_kind": "npz_svd_low_rank_v1",
            "max_abs_error_tolerance": 1.0e-3,
        },
        output_archive=output,
        runtime_consumption_proof_out=proof,
        repo_root=tmp_path,
    )

    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert proof_payload["schema"] == "family_agnostic_runtime_consumption_proof_v1"
    assert proof_payload["proof_kind"] == (
        "tensor_factorize_cooperative_receiver_reconstruction_proof.v1"
    )
    assert proof_payload["candidate_archive"]["sha256"] == result["candidate_archive"]["sha256"]
    assert proof_payload["candidate_member"]["sha256"] == result["candidate_member"]["sha256"]
    assert proof_payload["runtime_consumption_probe"]["passed"] is True
    assert proof_payload["runtime_consumption_probe"]["max_abs_error"] <= 1.0e-3
    assert proof_payload["cooperative_receiver"]["declared"] is True
    assert result["receiver_contract_satisfied"] is True
    assert result["runtime_consumption_proof_path"] == str(proof)
    assert "tensor_factorized_payload_requires_cooperative_receiver" not in (
        result["readiness_blockers"]
    )
    assert result["score_claim"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_tensor_factorize_materializer_proof_requires_declared_receiver(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    proof = tmp_path / "runtime_consumption_proof.json"
    tensor_path = tmp_path / "tensor.npy"
    vector_a = np.arange(64, dtype=np.float32)[:, None]
    vector_b = np.linspace(0.25, 2.0, 64, dtype=np.float32)[None, :]
    np.save(tensor_path, vector_a @ vector_b)
    _write_zip(archive, {"weights.npy": tensor_path.read_bytes()})

    result = materialize_tensor_factorize_candidate(
        archive_path=archive,
        tensor_manifest={"member_name": "weights.npy"},
        factorization_contract={"rank": 1, "max_abs_error_tolerance": 1.0e-3},
        output_archive=output,
        runtime_consumption_proof_out=proof,
        repo_root=tmp_path,
    )

    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert proof_payload["runtime_consumption_probe"]["cooperative_receiver_declared"] is False
    assert proof_payload["runtime_consumption_probe"]["passed"] is False
    assert result["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_not_passed" in result["readiness_blockers"]
    assert "tensor_factorized_payload_requires_cooperative_receiver" in (
        result["readiness_blockers"]
    )


def test_family_agnostic_materializer_cli_writes_false_authority_manifest(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate.json"
    proof = tmp_path / "candidate.runtime_consumption_proof.json"
    _write_zip(archive, {"payload.bin": b"A" * 4096})

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer.py"),
            "--target-kind",
            "packet_member_recompress_v1",
            "--archive-path",
            str(archive),
            "--output-archive",
            str(output),
            "--output-manifest",
            str(manifest),
            "--member-name",
            "payload.bin",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout)
    assert payload["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert stdout_payload["schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert proof.is_file()
    assert payload["receiver_contract_satisfied"] is True
    assert payload["receiver_verification"]["proof_present"] is True
    assert payload["runtime_consumption_proof_path"] == str(proof)
    assert payload["tool_run_manifest"]["tool"] == "tools/run_family_agnostic_materializer.py"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_packet_member_zip_header_elide_materializer_cli_auto_writes_runtime_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate.json"
    proof = tmp_path / "candidate.runtime_consumption_proof.json"
    _write_zip_with_member_header_overhead(archive, payload=b"A" * 4096)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer.py"),
            "--target-kind",
            "packet_member_zip_header_elide_v1",
            "--archive-path",
            str(archive),
            "--output-archive",
            str(output),
            "--output-manifest",
            str(manifest),
            "--member-name",
            "payload.bin",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout)
    assert payload["schema"] == PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA
    assert stdout_payload["schema"] == PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA
    assert proof.is_file()
    assert payload["receiver_contract_satisfied"] is True
    assert payload["receiver_verification"]["proof_present"] is True
    assert payload["runtime_consumption_proof_path"] == str(proof)
    assert payload["selected_elision"]["saved_bytes"] > 0
    assert payload["tool_run_manifest"]["tool"] == "tools/run_family_agnostic_materializer.py"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_packet_member_zip_header_elide_cli_can_elide_all_members(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest = tmp_path / "candidate.json"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.comment = b"archive comment"
        for name in ("renderer.bin", "weights.bin"):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.extra = b"\x7f\x7f\x04\x00abcd"
            info.comment = f"{name} comment".encode()
            zf.writestr(info, name.encode() * 128)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer.py"),
            "--target-kind",
            "packet_member_zip_header_elide_v1",
            "--archive-path",
            str(archive),
            "--output-archive",
            str(output),
            "--output-manifest",
            str(manifest),
            "--all-members",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["selection_scope"] == "all_members"
    assert payload["selected_member_names"] == ["renderer.bin", "weights.bin"]
    assert payload["receiver_contract_satisfied"] is True
    assert payload["selected_elision"]["saved_bytes"] > 0
    assert payload["score_claim"] is False


def test_archive_section_materializer_cli_auto_writes_runtime_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "candidate.json"
    proof = tmp_path / "candidate.runtime_consumption_proof.json"
    raw = b"section-raw" * 512
    section = brotli.compress(raw, quality=0)
    section_manifest_path = tmp_path / "sections.json"
    _write_zip(archive, {"0.raw": section})
    section_manifest_path.write_text(
        json.dumps(
            {
                "schema": "fixture_section_manifest.v1",
                "member": {"name": "0.raw"},
                "sections": [
                    {
                        "name": "section_a",
                        "index": 0,
                        "offset": 0,
                        "length": len(section),
                        "sha256": sha256_bytes(section),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer.py"),
            "--target-kind",
            "archive_section_entropy_recode_v1",
            "--archive-path",
            str(archive),
            "--output-archive",
            str(output),
            "--output-manifest",
            str(manifest_path),
            "--section-manifest",
            str(section_manifest_path),
            "--section-name",
            "section_a",
            "--brotli-quality",
            "11",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout)
    assert payload["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert stdout_payload["schema"] == ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA
    assert proof.is_file()
    assert payload["receiver_contract_satisfied"] is False
    assert payload["receiver_verification"]["proof_present"] is True
    assert payload["runtime_consumption_proof_path"] == str(proof)
    assert "section_length_changed_requires_runtime_consumption_proof" in (
        payload["readiness_blockers"]
    )
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_tensor_factorize_materializer_cli_auto_writes_runtime_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    output = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "candidate.json"
    proof = tmp_path / "candidate.runtime_consumption_proof.json"
    tensor_manifest_path = tmp_path / "tensor_manifest.json"
    contract_path = tmp_path / "factorization_contract.json"
    tensor_path = tmp_path / "tensor.npy"
    vector_a = np.arange(64, dtype=np.float32)[:, None]
    vector_b = np.linspace(0.25, 2.0, 64, dtype=np.float32)[None, :]
    np.save(tensor_path, vector_a @ vector_b)
    _write_zip(archive, {"weights.npy": tensor_path.read_bytes()})
    tensor_manifest_path.write_text(
        json.dumps({"member_name": "weights.npy"}),
        encoding="utf-8",
    )
    contract_path.write_text(
        json.dumps(
            {
                "rank": 1,
                "cooperative_receiver_id": "fixture_tensor_factorize_receiver",
                "receiver_adapter_kind": "npz_svd_low_rank_v1",
                "max_abs_error_tolerance": 1.0e-3,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_family_agnostic_materializer.py"),
            "--target-kind",
            "tensor_factorize_v1",
            "--archive-path",
            str(archive),
            "--output-archive",
            str(output),
            "--output-manifest",
            str(manifest_path),
            "--tensor-manifest",
            str(tensor_manifest_path),
            "--factorization-contract",
            str(contract_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout)
    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert payload["schema"] == TENSOR_FACTORIZE_SCHEMA
    assert stdout_payload["schema"] == TENSOR_FACTORIZE_SCHEMA
    assert proof.is_file()
    assert payload["receiver_contract_satisfied"] is True
    assert payload["receiver_verification"]["proof_present"] is True
    assert payload["runtime_consumption_proof_path"] == str(proof)
    assert proof_payload["proof_kind"] == (
        "tensor_factorize_cooperative_receiver_reconstruction_proof.v1"
    )
    assert proof_payload["runtime_consumption_probe"]["passed"] is True
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
