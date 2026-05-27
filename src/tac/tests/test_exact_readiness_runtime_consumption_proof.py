# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tac.optimizer.exact_readiness import validate_runtime_consumption_proof


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _row(proof_path: str = "proofs/runtime_consumption_proof.json") -> dict[str, object]:
    return {
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": proof_path,
        "target_kind": "packet_member_zip_header_elide_v1",
        "materializer_id": "packet_member_zip_header_elide_adapter",
        "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
    }


def _proof(*, archive_sha256: str, archive_path: Path | None = None) -> dict[str, object]:
    candidate_archive: dict[str, object] = {"sha256": archive_sha256}
    if archive_path is not None:
        candidate_archive["path"] = archive_path.as_posix()
        candidate_archive["bytes"] = archive_path.stat().st_size
    return {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "target_kind": "packet_member_zip_header_elide_v1",
        "materializer_id": "packet_member_zip_header_elide_adapter",
        "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": archive_sha256,
        "receiver_contract_satisfied": True,
        "runtime_consumption_proof_passed": True,
        "passed": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def test_family_agnostic_runtime_proof_rejects_top_level_boolean_only(
    tmp_path: Path,
) -> None:
    archive_sha = "a" * 64
    _write_json(
        tmp_path / "proofs/runtime_consumption_proof.json",
        _proof(archive_sha256=archive_sha),
    )

    blockers, _facts = validate_runtime_consumption_proof(
        _row(),
        repo_root=tmp_path,
        queue_dir=tmp_path,
        submission_dir=None,
        archive_sha256=archive_sha,
    )

    assert "runtime_consumption_proof_not_proven" in blockers
    assert "runtime_consumption_proof_runtime_probe_missing" in blockers


def test_family_agnostic_runtime_proof_rejects_empty_probe_evidence(
    tmp_path: Path,
) -> None:
    archive_sha = "b" * 64
    proof = _proof(archive_sha256=archive_sha)
    proof["runtime_consumption_probe"] = {"passed": True}
    _write_json(tmp_path / "proofs/runtime_consumption_proof.json", proof)

    blockers, _facts = validate_runtime_consumption_proof(
        _row(),
        repo_root=tmp_path,
        queue_dir=tmp_path,
        submission_dir=None,
        archive_sha256=archive_sha,
    )

    assert "runtime_consumption_proof_not_proven" in blockers
    assert "runtime_consumption_probe_evidence_missing" in blockers


def test_family_agnostic_runtime_proof_accepts_file_backed_probe_evidence(
    tmp_path: Path,
) -> None:
    archive_bytes = b"candidate archive bytes"
    archive_path = tmp_path / "candidate.zip"
    archive_path.write_bytes(archive_bytes)
    archive_sha = _sha256_bytes(archive_bytes)
    member_sha = _sha256_bytes(b"member payload")
    proof = _proof(archive_sha256=archive_sha, archive_path=archive_path)
    proof["runtime_consumption_probe"] = {
        "schema": "packet_member_payload_identity_probe.v1",
        "passed": True,
        "candidate_member_sha256": member_sha,
        "source_member_sha256": member_sha,
    }
    _write_json(tmp_path / "proofs/runtime_consumption_proof.json", proof)

    blockers, _facts = validate_runtime_consumption_proof(
        _row(),
        repo_root=tmp_path,
        queue_dir=tmp_path,
        submission_dir=None,
        archive_sha256=archive_sha,
    )

    assert blockers == []
