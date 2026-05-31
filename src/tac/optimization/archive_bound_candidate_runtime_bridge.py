# SPDX-License-Identifier: MIT
"""Runtime-proof bridge for archive-bound candidate exporters.

Substrate archive exporters that already emit a contest-shaped ``archive.zip``
and ``submission/inflate.sh`` can use this module to produce the shared
archive-bound candidate package: receiver proof, replay bundle, exact blocker,
and posterior hook. It is intentionally score-authority false.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.archive_bound_candidate_adapter_spine import (
    build_archive_bound_candidate_adapter_package,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import ordered_unique
from tac.repo_io import sha256_file, tree_sha256, write_json

ARCHIVE_BOUND_RUNTIME_ADAPTER_PACKAGE_SCHEMA = (
    "tac_archive_bound_candidate_runtime_adapter_package.v1"
)
ARCHIVE_BOUND_RUNTIME_RECEIVER_PROOF_SCHEMA = (
    "tac_archive_bound_candidate_generated_receiver_proof.v1"
)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repo_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def _safe_text(value: object, *, max_chars: int = 4096) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def run_generated_inflate_receiver_proof(
    *,
    archive_zip_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    submission_dir: Path,
    output_dir: Path,
    repo_root: Path,
    proof_schema: str = ARCHIVE_BOUND_RUNTIME_RECEIVER_PROOF_SCHEMA,
    proof_filename: str = "archive_bound_candidate_receiver_proof.json",
    candidate_label: str = "archive_bound_candidate",
    video_name: str = "0.mkv",
    retain_receiver_output: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, object]:
    """Run generated ``inflate.sh`` and persist a receiver proof.

    The proof establishes only that the emitted runtime consumes the archive
    packet and produces raw bytes. It never grants score, promotion, or exact
    dispatch authority.
    """

    proof_dir = output_dir / "receiver_proof"
    proof_dir.mkdir(parents=True, exist_ok=True)
    file_list = proof_dir / "file_list.txt"
    file_list.write_text(f"{video_name}\n", encoding="utf-8")
    receiver_out_dir = proof_dir / "runtime_out"
    receiver_out_dir.mkdir(parents=True, exist_ok=True)
    output_name = Path(video_name).stem
    receiver_raw = receiver_out_dir / output_name
    inflate_argv = [
        str(submission_dir / "inflate.sh"),
        str(submission_dir),
        str(receiver_out_dir),
        str(file_list),
    ]

    started = time.monotonic()
    returncode: int | None = None
    stdout = ""
    stderr = ""
    timed_out = False
    try:
        result = subprocess.run(
            inflate_argv,
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHON": sys.executable},
            text=True,
            timeout=int(timeout_seconds),
        )
        returncode = int(result.returncode)
        stdout = _safe_text(result.stdout)
        stderr = _safe_text(result.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _safe_text(exc.stdout)
        stderr = _safe_text(exc.stderr)
    wall_seconds = round(time.monotonic() - started, 6)

    output_present = receiver_raw.is_file()
    output_sha256 = sha256_file(receiver_raw) if output_present else None
    output_bytes = receiver_raw.stat().st_size if output_present else None
    if output_present and not retain_receiver_output:
        receiver_raw.unlink()
    if not retain_receiver_output:
        try:
            receiver_out_dir.rmdir()
        except OSError:
            pass

    blockers: list[str] = []
    if timed_out:
        blockers.append(f"{candidate_label}_generated_inflate_sh_timed_out")
    if returncode not in (0, None):
        blockers.append(f"{candidate_label}_generated_inflate_sh_returned_nonzero")
    if returncode is None and not timed_out:
        blockers.append(f"{candidate_label}_generated_inflate_sh_not_executed")
    if not output_present:
        blockers.append(f"{candidate_label}_generated_inflate_sh_output_missing")
    if output_present and not output_bytes:
        blockers.append(f"{candidate_label}_generated_inflate_sh_output_empty")
    passed = returncode == 0 and output_present and bool(output_bytes) and not timed_out
    proof_path = proof_dir / proof_filename
    proof = {
        "schema": proof_schema,
        "candidate_label": candidate_label,
        "archive_path": _repo_relative(archive_zip_path, repo_root),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_bytes),
        "submission_dir": _repo_relative(submission_dir, repo_root),
        "runtime_tree_sha256": tree_sha256(submission_dir),
        "inflate_argv": [
            _repo_relative(Path(inflate_argv[0]), repo_root),
            *inflate_argv[1:],
        ],
        "file_list_path": _repo_relative(file_list, repo_root),
        "receiver_output_dir": _repo_relative(receiver_out_dir, repo_root),
        "receiver_output_path": _repo_relative(receiver_raw, repo_root),
        "receiver_output_present_during_proof": output_present,
        "receiver_output_retained": bool(retain_receiver_output and output_present),
        "receiver_output_sha256": output_sha256,
        "receiver_output_bytes": output_bytes,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "runtime_consumption_proof_ready": passed,
        "receiver_contract_satisfied": passed,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    write_json(proof_path, proof)
    proof["proof_path"] = _repo_relative(proof_path, repo_root)
    write_json(proof_path, proof)
    return proof


class _SingleArchiveBoundRuntimeCandidateAdapter:
    def __init__(
        self,
        *,
        adapter_id: str,
        candidate_family: str,
        row: Mapping[str, Any],
    ) -> None:
        self.adapter_id = adapter_id
        self.candidate_family = candidate_family
        self._row = dict(row)

    def emit_archive_bound_candidate_rows(
        self,
        context: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        return [dict(self._row)]


def build_archive_bound_candidate_runtime_package(
    *,
    adapter_id: str,
    candidate_family: str,
    candidate_id_prefix: str,
    transform_kind: str,
    archive_zip_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    submission_dir: Path,
    output_dir: Path,
    repo_root: Path,
    receiver_proof: Mapping[str, Any],
    receiver_contract_kind: str,
    runtime_adapter_manifest_extra: Mapping[str, Any] | None = None,
    candidate_row_schema: str = "archive_bound_runtime_candidate_row.v1",
    wrapper_schema: str = ARCHIVE_BOUND_RUNTIME_ADAPTER_PACKAGE_SCHEMA,
    package_filename: str = "archive_bound_candidate_adapter_package.json",
    input_artifacts: Sequence[str] | None = None,
    extra_blockers: Sequence[str] | None = None,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build and persist the shared archive-bound candidate package."""

    proof_passed = receiver_proof.get("runtime_consumption_proof_ready") is True
    proof_path = str(receiver_proof.get("proof_path") or "")
    runtime_tree_sha256 = tree_sha256(submission_dir)
    runtime_manifest = {
        "schema": "archive_bound_runtime_adapter_manifest.v1",
        "runtime_adapter_ready": True,
        "contest_runtime_decoder_adapter_ready": True,
        "decode_only_receiver_contract": True,
        "submission_dir": _repo_relative(submission_dir, repo_root),
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_receiver_proof_path": proof_path,
        **dict(runtime_adapter_manifest_extra or {}),
    }
    row = {
        "schema": candidate_row_schema,
        "candidate_id": f"{candidate_id_prefix}_{archive_sha256[:16]}",
        "candidate_family": candidate_family,
        "target_kind": transform_kind,
        "archive_native_transform_kind": transform_kind,
        "candidate_archive_path": _repo_relative(archive_zip_path, repo_root),
        "candidate_archive_sha256": archive_sha256,
        "candidate_archive_bytes": int(archive_bytes),
        "byte_closed_candidate_emitted": True,
        "byte_closed_candidate_materialized": True,
        "candidate_archive_materialized": True,
        "runtime_consumption_proof_status": "present" if proof_passed else "blocked",
        "runtime_consumption_proof_ready": proof_passed,
        "runtime_consumption_proof_path": proof_path,
        "receiver_contract_kind": receiver_contract_kind,
        "receiver_contract_satisfied": proof_passed,
        "runtime_adapter_ready": True,
        "contest_runtime_decoder_adapter_ready": True,
        "runtime_adapter_manifest": runtime_manifest,
        "semantic_payload_changed": True,
        "score_affecting_payload_changed": True,
        "exact_axis_score_affecting_adjudication_required": True,
        "charged_bits_changed": True,
        "replay_argv": list(receiver_proof.get("inflate_argv") or []),
        "mlx_triage_argv": list(mlx_triage_argv or []),
        "input_artifacts": list(
            input_artifacts
            or [
                _repo_relative(archive_zip_path, repo_root),
                _repo_relative(submission_dir / "0.bin", repo_root),
                proof_path,
            ]
        ),
        "blockers": ordered_unique(
            [
                *list(receiver_proof.get("blockers") or []),
                *list(extra_blockers or []),
            ]
        ),
        "score_claim_valid": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        **FALSE_AUTHORITY,
    }
    package = build_archive_bound_candidate_adapter_package(
        _SingleArchiveBoundRuntimeCandidateAdapter(
            adapter_id=adapter_id,
            candidate_family=candidate_family,
            row=row,
        ),
        repo_root=repo_root,
    )
    wrapped = {
        "schema": wrapper_schema,
        "archive_bound_candidate_adapter_package": package,
        "receiver_proof": dict(receiver_proof),
        **FALSE_AUTHORITY,
    }
    write_json(output_dir / package_filename, wrapped)
    return wrapped


__all__ = [
    "ARCHIVE_BOUND_RUNTIME_ADAPTER_PACKAGE_SCHEMA",
    "ARCHIVE_BOUND_RUNTIME_RECEIVER_PROOF_SCHEMA",
    "build_archive_bound_candidate_runtime_package",
    "run_generated_inflate_receiver_proof",
]
