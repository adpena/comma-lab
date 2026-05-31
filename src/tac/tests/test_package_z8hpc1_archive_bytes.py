# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools import package_z8hpc1_archive_bytes as tool


def test_package_z8hpc1_archive_bin_wraps_existing_packet_with_custody(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source" / "0.bin"
    source.parent.mkdir()
    source.write_bytes(b"Z8HPC1-test-packet")
    output_dir = tmp_path / "wrapped"

    def fake_export(
        archive_bytes: bytes,
        output_dir_arg: Path,
        *,
        repo_root: Path,
        emit_archive_bound_candidate_package: bool,
        emit_byte_mutation_proof: bool,
        emit_runtime_payload_bridge_report: bool,
        retain_receiver_proof_output: bool,
        mlx_triage_argv: list[str],
    ) -> tuple[Path, str, int]:
        assert archive_bytes == source.read_bytes()
        assert emit_archive_bound_candidate_package is True
        assert emit_byte_mutation_proof is True
        assert emit_runtime_payload_bridge_report is True
        del repo_root, retain_receiver_proof_output, mlx_triage_argv
        output_dir_arg.mkdir(parents=True, exist_ok=True)
        (output_dir_arg / "0.bin").write_bytes(archive_bytes)
        submission = output_dir_arg / "submission"
        submission.mkdir()
        (submission / "0.bin").write_bytes(archive_bytes)
        (submission / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        for rel in (
            "z8_hpc1_runtime_payload_bridge_report.json",
            "z8_hpc1_byte_mutation_proof.json",
        ):
            path = output_dir_arg / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        receiver_proof = output_dir_arg / "receiver_proof" / "z8_hpc1_receiver_proof.json"
        receiver_proof.parent.mkdir(parents=True, exist_ok=True)
        receiver_proof.write_text(
            json.dumps(
                {
                    "runtime_consumption_proof_ready": True,
                    "receiver_contract_satisfied": True,
                    "blockers": [],
                }
            ),
            encoding="utf-8",
        )
        adapter_package = output_dir_arg / "archive_bound_candidate_adapter_package.json"
        adapter_package.write_text(
            json.dumps(
                {
                    "archive_bound_candidate_adapter_package": {
                        "candidate_rows": [
                            {
                                "runtime_consumption_proof_ready": True,
                                "receiver_contract_satisfied": True,
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        archive_zip = output_dir_arg / "archive.zip"
        with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("0.bin", archive_bytes)
            zf.writestr("inflate.sh", "#!/usr/bin/env bash\n")
        return (archive_zip, tool._sha256_file(archive_zip), archive_zip.stat().st_size)

    monkeypatch.setattr(tool, "export_z8hpc1_archive_bytes", fake_export)

    manifest = tool.package_z8hpc1_archive_bin(
        archive_bin=source,
        output_dir=output_dir,
        repo_root=tmp_path,
        argv=["package_z8hpc1_archive_bytes.py", "--unit"],
    )

    assert manifest["schema"] == tool.SCHEMA
    assert manifest["custody_repaired"] is True
    assert manifest["blockers"] == []
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["receiver_proof_ready"] is True
    assert manifest["receiver_contract_satisfied"] is True
    assert manifest["zip_custody"]["zip_custody_ok"] is True
    assert manifest["zip_custody"]["zero_bin_member_sha256"] == tool._sha256_file(source)
    manifest_path = output_dir / "package_z8hpc1_archive_bytes_manifest.json"
    assert manifest_path.is_file()
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["input_archive_bin_sha256"] == tool._sha256_file(source)


def test_zip_zero_bin_custody_rejects_mismatched_member(tmp_path: Path) -> None:
    archive_zip = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", b"different")

    custody = tool._zip_zero_bin_custody(
        archive_zip,
        expected_zero_bin_sha256=tool._sha256_bytes(b"expected"),
    )

    assert custody["zip_custody_ok"] is False
    assert custody["zero_bin_member_count"] == 1


def test_package_z8hpc1_archive_bin_fails_closed_on_blocked_receiver_proof(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source" / "0.bin"
    source.parent.mkdir()
    source.write_bytes(b"Z8HPC1-test-packet")
    output_dir = tmp_path / "wrapped"

    def fake_export(
        archive_bytes: bytes,
        output_dir_arg: Path,
        *,
        repo_root: Path,
        emit_archive_bound_candidate_package: bool,
        emit_byte_mutation_proof: bool,
        emit_runtime_payload_bridge_report: bool,
        retain_receiver_proof_output: bool,
        mlx_triage_argv: list[str],
    ) -> tuple[Path, str, int]:
        del (
            repo_root,
            emit_archive_bound_candidate_package,
            emit_byte_mutation_proof,
            emit_runtime_payload_bridge_report,
            retain_receiver_proof_output,
            mlx_triage_argv,
        )
        output_dir_arg.mkdir(parents=True, exist_ok=True)
        (output_dir_arg / "0.bin").write_bytes(archive_bytes)
        submission = output_dir_arg / "submission"
        submission.mkdir()
        (submission / "0.bin").write_bytes(archive_bytes)
        receiver_proof = output_dir_arg / "receiver_proof" / "z8_hpc1_receiver_proof.json"
        receiver_proof.parent.mkdir(parents=True, exist_ok=True)
        receiver_proof.write_text(
            json.dumps(
                {
                    "runtime_consumption_proof_ready": False,
                    "receiver_contract_satisfied": False,
                    "blockers": ["z8_hpc1_generated_inflate_sh_output_missing"],
                }
            ),
            encoding="utf-8",
        )
        for rel in (
            "archive_bound_candidate_adapter_package.json",
            "z8_hpc1_runtime_payload_bridge_report.json",
            "z8_hpc1_byte_mutation_proof.json",
        ):
            path = output_dir_arg / rel
            path.write_text(
                json.dumps(
                    {
                        "archive_bound_candidate_adapter_package": {
                            "candidate_rows": [
                                {
                                    "runtime_consumption_proof_ready": False,
                                    "receiver_contract_satisfied": False,
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
        archive_zip = output_dir_arg / "archive.zip"
        with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("0.bin", archive_bytes)
        return (archive_zip, tool._sha256_file(archive_zip), archive_zip.stat().st_size)

    monkeypatch.setattr(tool, "export_z8hpc1_archive_bytes", fake_export)

    manifest = tool.package_z8hpc1_archive_bin(
        archive_bin=source,
        output_dir=output_dir,
        repo_root=tmp_path,
        argv=["package_z8hpc1_archive_bytes.py", "--unit"],
    )

    assert manifest["custody_repaired"] is False
    assert "receiver_proof_not_ready" in manifest["blockers"]
    assert "receiver_contract_not_satisfied" in manifest["blockers"]
    assert "archive_bound_candidate_runtime_proof_not_ready" in manifest["blockers"]
    assert manifest["receiver_proof_blockers"] == [
        "z8_hpc1_generated_inflate_sh_output_missing"
    ]
