# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from pathlib import Path

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    build_archive_bound_candidate_runtime_package,
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import sha256_file


def test_generated_receiver_proof_rejects_raw_output_directory(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "0.bin").write_bytes(b"payload")
    inflate = submission / "inflate.sh"
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$2/0.raw\"\n"
        "printf frame0 > \"$2/0.raw/0.png\"\n",
        encoding="utf-8",
    )
    os.chmod(inflate, 0o755)

    proof = run_generated_inflate_receiver_proof(
        archive_zip_path=archive,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        submission_dir=submission,
        output_dir=tmp_path / "proof",
        repo_root=tmp_path,
        candidate_label="directory_receiver",
        retain_receiver_output=False,
    )

    assert proof["runtime_consumption_proof_passed"] is False
    assert proof["receiver_contract_satisfied"] is False
    assert proof["receiver_output_kind"] == "directory"
    assert proof["receiver_output_bytes"] == len(b"frame0")
    assert "directory_receiver_generated_inflate_sh_output_not_raw_file" in proof[
        "blockers"
    ]
    assert not (tmp_path / "proof" / "receiver_proof" / "runtime_out" / "0.raw").exists()


def test_generated_receiver_proof_accepts_raw_output_file(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "0.bin").write_bytes(b"payload")
    inflate = submission / "inflate.sh"
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf rawbytes > \"$2/0.raw\"\n",
        encoding="utf-8",
    )
    os.chmod(inflate, 0o755)

    proof = run_generated_inflate_receiver_proof(
        archive_zip_path=archive,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        submission_dir=submission,
        output_dir=tmp_path / "proof_file",
        repo_root=tmp_path,
        candidate_label="file_receiver",
        expected_receiver_output_bytes=len(b"rawbytes"),
        retain_receiver_output=False,
    )

    assert proof["runtime_consumption_proof_passed"] is True
    assert proof["receiver_contract_satisfied"] is True
    assert proof["receiver_output_kind"] == "file"
    assert proof["receiver_output_bytes"] == len(b"rawbytes")
    assert proof["blockers"] == []


def test_archive_bound_runtime_package_preserves_changedness_without_score_authority(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "0.bin").write_bytes(b"payload")
    proof = {
        "proof_path": "receiver_proof/proof.json",
        "runtime_consumption_proof_ready": True,
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "inflate_argv": ["inflate.sh", "archive_dir", "out", "file_list"],
    }

    package = build_archive_bound_candidate_runtime_package(
        adapter_id="unit_adapter",
        candidate_family="unit_family",
        candidate_id_prefix="unit_candidate",
        transform_kind="unit_transform",
        archive_zip_path=archive,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        submission_dir=submission,
        output_dir=tmp_path / "package",
        repo_root=tmp_path,
        receiver_proof=proof,
        receiver_contract_kind="unit_receiver_contract",
    )

    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    assert row["semantic_payload_changed"] is True
    assert row["score_affecting_payload_changed"] is True
    assert row["exact_axis_score_affecting_adjudication_required"] is True
    assert row["charged_bits_changed"] is True
    assert row["score_claim"] is False
    assert row["score_claim_valid"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_archive_bound_runtime_package_runtime_ready_requires_receiver_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "0.bin").write_bytes(b"payload")
    proof = {
        "proof_path": "receiver_proof/proof.json",
        "runtime_consumption_proof_ready": False,
        "receiver_contract_satisfied": False,
        "blockers": ["unit_receiver_proof_failed"],
        "inflate_argv": ["inflate.sh", "archive_dir", "out", "file_list"],
    }

    package = build_archive_bound_candidate_runtime_package(
        adapter_id="unit_adapter",
        candidate_family="unit_family",
        candidate_id_prefix="unit_candidate",
        transform_kind="unit_transform",
        archive_zip_path=archive,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        submission_dir=submission,
        output_dir=tmp_path / "package",
        repo_root=tmp_path,
        receiver_proof=proof,
        receiver_contract_kind="unit_receiver_contract",
    )

    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    manifest = row["runtime_adapter_manifest"]
    assert manifest["runtime_adapter_present"] is True
    assert manifest["contest_runtime_decoder_adapter_present"] is True
    assert manifest["runtime_adapter_ready"] is False
    assert manifest["contest_runtime_decoder_adapter_ready"] is False
    assert row["runtime_adapter_present"] is True
    assert row["contest_runtime_decoder_adapter_present"] is True
    assert row["runtime_adapter_ready"] is False
    assert row["contest_runtime_decoder_adapter_ready"] is False
    assert row["runtime_consumption_proof_status"] == "blocked"
    assert row["runtime_consumption_proof_ready"] is False
    assert row["runtime_consumption_proof_passed"] is False
    assert row["receiver_contract_satisfied"] is False
    assert "unit_receiver_proof_failed" in row["blockers"]
    assert "runtime_adapter_ready_requires_receiver_proof" in row["blockers"]


def test_archive_bound_runtime_package_rejects_ready_only_receiver_proof(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "0.bin").write_bytes(b"payload")
    proof = {
        "proof_path": "receiver_proof/proof.json",
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "inflate_argv": ["inflate.sh", "archive_dir", "out", "file_list"],
    }

    package = build_archive_bound_candidate_runtime_package(
        adapter_id="unit_adapter",
        candidate_family="unit_family",
        candidate_id_prefix="unit_candidate",
        transform_kind="unit_transform",
        archive_zip_path=archive,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        submission_dir=submission,
        output_dir=tmp_path / "package",
        repo_root=tmp_path,
        receiver_proof=proof,
        receiver_contract_kind="unit_receiver_contract",
    )

    row = package["archive_bound_candidate_adapter_package"]["candidate_rows"][0]
    assert row["runtime_consumption_proof_status"] == "blocked"
    assert row["runtime_consumption_proof_ready"] is False
    assert row["runtime_consumption_proof_passed"] is False
    assert row["receiver_contract_satisfied"] is False
