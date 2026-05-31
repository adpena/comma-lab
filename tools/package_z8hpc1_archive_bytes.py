#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package arbitrary Z8HPC1 ``0.bin`` bytes into a receiver-proven archive.

This is the custody repair path for rate probes that emitted a valid inner
Z8HPC1 packet without the contest-shaped ``archive.zip`` and shared
archive-bound candidate contract. It does not compute or claim a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
    export_z8hpc1_archive_bytes,
)

SCHEMA = "z8_hpc1_archive_bytes_packaging_manifest.v1"
TOOL = "tools/package_z8hpc1_archive_bytes.py"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repo_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def _capture_env() -> dict[str, str]:
    keys = (
        "PYTHONPATH",
        "PYTHON",
        "PACT_INFLATE_DEVICE",
        "MLX_METAL_DEVICE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _zip_zero_bin_custody(
    archive_zip_path: Path,
    *,
    expected_zero_bin_sha256: str,
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            payload = zf.read(info.filename)
            members.append(
                {
                    "filename": info.filename,
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "compression_type": int(info.compress_type),
                    "sha256": _sha256_bytes(payload),
                    "matches_expected_zero_bin_sha256": (
                        info.filename == "0.bin"
                        and _sha256_bytes(payload) == expected_zero_bin_sha256
                    ),
                }
            )
    zero_bin_members = [row for row in members if row["filename"] == "0.bin"]
    zero_bin_sha = zero_bin_members[0]["sha256"] if len(zero_bin_members) == 1 else None
    custody_ok = (
        len(zero_bin_members) == 1 and zero_bin_sha == expected_zero_bin_sha256
    )
    return {
        "archive_zip_path": archive_zip_path.as_posix(),
        "archive_zip_bytes": archive_zip_path.stat().st_size,
        "archive_zip_sha256": _sha256_file(archive_zip_path),
        "member_count": len(members),
        "members": sorted(members, key=lambda row: row["filename"]),
        "zero_bin_member_count": len(zero_bin_members),
        "zero_bin_member_sha256": zero_bin_sha,
        "expected_zero_bin_sha256": expected_zero_bin_sha256,
        "zip_custody_ok": custody_ok,
    }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def package_z8hpc1_archive_bin(
    *,
    archive_bin: Path,
    output_dir: Path,
    repo_root: Path,
    manifest_path: Path | None = None,
    retain_receiver_output: bool = False,
    emit_byte_mutation_proof: bool = True,
    emit_runtime_payload_bridge_report: bool = True,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Package an existing Z8HPC1 packet and return a custody manifest."""

    archive_bin = archive_bin.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or (output_dir / "package_z8hpc1_archive_bytes_manifest.json")
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    input_bytes = archive_bin.read_bytes()
    input_sha = _sha256_bytes(input_bytes)
    archive_zip_path, archive_zip_sha, archive_zip_bytes = export_z8hpc1_archive_bytes(
        input_bytes,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=True,
        emit_byte_mutation_proof=emit_byte_mutation_proof,
        emit_runtime_payload_bridge_report=emit_runtime_payload_bridge_report,
        retain_receiver_proof_output=retain_receiver_output,
        mlx_triage_argv=argv or sys.argv,
    )
    zip_custody = _zip_zero_bin_custody(
        archive_zip_path,
        expected_zero_bin_sha256=input_sha,
    )
    copied_bin = output_dir / "0.bin"
    submission_bin = output_dir / "submission" / "0.bin"
    blockers: list[str] = []
    if not zip_custody["zip_custody_ok"]:
        blockers.append("archive_zip_zero_bin_custody_mismatch")
    if not copied_bin.is_file() or _sha256_file(copied_bin) != input_sha:
        blockers.append("output_0bin_sha256_mismatch")
    if not submission_bin.is_file() or _sha256_file(submission_bin) != input_sha:
        blockers.append("submission_0bin_sha256_mismatch")
    receiver_proof = output_dir / "receiver_proof" / "z8_hpc1_receiver_proof.json"
    adapter_package = output_dir / "archive_bound_candidate_adapter_package.json"
    bridge_report = output_dir / "z8_hpc1_runtime_payload_bridge_report.json"
    byte_mutation_proof = output_dir / "z8_hpc1_byte_mutation_proof.json"
    for label, path in (
        ("receiver_proof_missing", receiver_proof),
        ("archive_bound_candidate_adapter_package_missing", adapter_package),
        ("runtime_payload_bridge_report_missing", bridge_report),
    ):
        if not path.is_file():
            blockers.append(label)
    if emit_byte_mutation_proof and not byte_mutation_proof.is_file():
        blockers.append("byte_mutation_proof_missing")
    receiver_proof_payload = _read_json_object(receiver_proof)
    if receiver_proof_payload is None:
        receiver_proof_blockers: list[str] = []
    else:
        receiver_proof_blockers = [
            str(blocker) for blocker in (receiver_proof_payload.get("blockers") or [])
        ]
        if receiver_proof_payload.get("runtime_consumption_proof_ready") is not True:
            blockers.append("receiver_proof_not_ready")
        if receiver_proof_payload.get("receiver_contract_satisfied") is not True:
            blockers.append("receiver_contract_not_satisfied")
    adapter_package_payload = _read_json_object(adapter_package)
    if adapter_package_payload is not None:
        wrapped = adapter_package_payload.get("archive_bound_candidate_adapter_package")
        rows = wrapped.get("candidate_rows") if isinstance(wrapped, dict) else None
        if not rows:
            blockers.append("archive_bound_candidate_rows_missing")
        else:
            first = rows[0]
            if not isinstance(first, dict):
                blockers.append("archive_bound_candidate_row_invalid")
            else:
                if first.get("runtime_consumption_proof_ready") is not True:
                    blockers.append("archive_bound_candidate_runtime_proof_not_ready")
                if first.get("receiver_contract_satisfied") is not True:
                    blockers.append("archive_bound_candidate_receiver_contract_not_satisfied")

    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "purpose": (
            "Repair custody for an existing Z8HPC1 inner archive packet by "
            "emitting contest-shaped archive.zip, decode-only runtime, receiver "
            "proof, runtime payload bridge report, and shared archive-bound "
            "candidate package."
        ),
        "input_archive_bin": _repo_relative(archive_bin, repo_root),
        "input_archive_bin_bytes": len(input_bytes),
        "input_archive_bin_sha256": input_sha,
        "output_dir": _repo_relative(output_dir, repo_root),
        "archive_zip_path": _repo_relative(archive_zip_path, repo_root),
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": int(archive_zip_bytes),
        "zip_custody": zip_custody,
        "copied_0bin_path": _repo_relative(copied_bin, repo_root),
        "submission_0bin_path": _repo_relative(submission_bin, repo_root),
        "receiver_proof_path": (
            _repo_relative(receiver_proof, repo_root) if receiver_proof.is_file() else None
        ),
        "receiver_proof_ready": (
            receiver_proof_payload.get("runtime_consumption_proof_ready") is True
            if receiver_proof_payload is not None
            else False
        ),
        "receiver_contract_satisfied": (
            receiver_proof_payload.get("receiver_contract_satisfied") is True
            if receiver_proof_payload is not None
            else False
        ),
        "receiver_proof_blockers": receiver_proof_blockers,
        "archive_bound_candidate_adapter_package_path": (
            _repo_relative(adapter_package, repo_root)
            if adapter_package.is_file()
            else None
        ),
        "runtime_payload_bridge_report_path": (
            _repo_relative(bridge_report, repo_root) if bridge_report.is_file() else None
        ),
        "byte_mutation_proof_path": (
            _repo_relative(byte_mutation_proof, repo_root)
            if byte_mutation_proof.is_file()
            else None
        ),
        "emit_byte_mutation_proof": bool(emit_byte_mutation_proof),
        "emit_runtime_payload_bridge_report": bool(emit_runtime_payload_bridge_report),
        "retain_receiver_output": bool(retain_receiver_output),
        "cleanup_policy": {
            "scratch_deleted_by_default": True,
            "receiver_raw_output_retained": bool(retain_receiver_output),
            "evidence_artifacts_are_durable": True,
            "large_rebuildable_artifacts": [],
            "no_signal_loss_provenance": True,
        },
        "reproducibility": {
            "argv": list(argv or sys.argv),
            "cwd": Path.cwd().as_posix(),
            "env": _capture_env(),
        },
        "blockers": blockers,
        "custody_repaired": not blockers,
        "exact_axis_blocker": (
            "contest_cpu_cuda_exact_eval_not_executed"
            if not blockers
            else "archive_bound_candidate_custody_incomplete"
        ),
        **FALSE_AUTHORITY,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = _repo_relative(manifest_path, repo_root)
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--retain-receiver-output", action="store_true")
    parser.add_argument(
        "--skip-byte-mutation-proof",
        action="store_true",
        help="Debug-only: package without the Z8 byte-mutation pixel-consumption proof.",
    )
    parser.add_argument(
        "--skip-runtime-payload-bridge-report",
        action="store_true",
        help="Debug-only: omit the Z8 runtime payload bridge report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = package_z8hpc1_archive_bin(
        archive_bin=args.archive_bin,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        manifest_path=args.manifest_path,
        retain_receiver_output=bool(args.retain_receiver_output),
        emit_byte_mutation_proof=not bool(args.skip_byte_mutation_proof),
        emit_runtime_payload_bridge_report=not bool(
            args.skip_runtime_payload_bridge_report
        ),
        argv=sys.argv,
    )
    print(_json_text(manifest), end="")
    return 0 if manifest["custody_repaired"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
