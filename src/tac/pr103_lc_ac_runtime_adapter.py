# SPDX-License-Identifier: MIT
"""Runtime-adapter builder for PR103 ``hnerv_lc_ac`` section-length changes."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from tac.hnerv_lowlevel_packer import read_strict_single_member_zip
from tac.optimization.archive_bound_candidate_adapter_spine import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
    build_archive_bound_candidate_adapter_package,
)
from tac.pr103_arithmetic_transform_plan import CANDIDATE_SCHEMA
from tac.repo_io import (
    json_text,
    read_json,
    repo_relative,
    sha256_bytes,
    sha256_file,
    tree_sha256,
    write_json,
)

ADAPTER_SCHEMA = "pr103_lc_ac_runtime_adapter_v1"
PACKET_SCHEMA = "pr103_lc_ac_histogram_candidate_packet_v1"
ARCHIVE_MANIFEST_SCHEMA = "pr103_lc_ac_histogram_candidate_archive_manifest_v1"
FRAME_PARITY_SCHEMA = "pr103_lc_ac_frame_parity_probe_v1"
SHELL_PARITY_SCHEMA = "pact.inflate_shell_output_parity_v1"
REQUIRED_RUNTIME_CONSTANTS = (
    "BR_LEN",
    "HIST_LEN",
    "MERGED_AC_LEN",
    "LO_LEN",
    "HI_HIST_LEN",
)
EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")


class Pr103RuntimeAdapterError(ValueError):
    """Raised when a PR103 runtime adapter cannot be built safely."""


class _SinglePr103ArchiveCandidateAdapter:
    """Adapter-spine bridge for one PR103 archive/runtime packet."""

    def __init__(
        self,
        *,
        adapter_id: str,
        candidate_family: str,
        row: dict[str, Any],
    ) -> None:
        self.adapter_id = adapter_id
        self.candidate_family = candidate_family
        self._row = row

    def emit_archive_bound_candidate_rows(
        self,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [dict(self._row)]


def build_pr103_lc_ac_runtime_adapter(
    *,
    candidate_manifest: str | Path,
    source_runtime_dir: str | Path,
    output_runtime_dir: str | Path,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Copy and patch a PR103 runtime tree for a materialized candidate.

    The candidate manifest remains the source of truth for section lengths. The
    copied runtime is not a score claim and does not run the scorer.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest_path = _repo_path(Path(candidate_manifest), repo)
    source_dir = _repo_path(Path(source_runtime_dir), repo)
    output_dir = _repo_path(Path(output_runtime_dir), repo)
    manifest = _read_candidate_manifest(manifest_path)
    constants = _candidate_runtime_constants(manifest)
    candidate_record = _mapping(manifest.get("candidate_archive"))
    source_record = _mapping(manifest.get("source_archive"))
    candidate_archive = _resolve_manifest_path(
        Path(str(candidate_record.get("path") or "")),
        manifest_path=manifest_path,
        repo=repo,
    )
    source_archive = _resolve_manifest_path(
        Path(str(source_record.get("path") or "")),
        manifest_path=manifest_path,
        repo=repo,
    )
    _validate_inputs(source_dir, candidate_archive)
    _validate_candidate_archive_custody(source_record, source_archive, label="source archive")
    _validate_candidate_archive_custody(candidate_record, candidate_archive)
    _validate_output_location(source_dir, output_dir)
    _prepare_output_dir(output_dir, force=force)
    copied_files = _copy_runtime_tree(source_dir, output_dir)
    changes = _patch_inflate_constants(output_dir / "inflate.py", constants)
    shell_contract = _verify_inflate_shell_contract(output_dir / "inflate.sh")
    source_probe = _runtime_consumption_probe(
        source_dir / "inflate.py",
        source_archive,
        _source_runtime_constants(manifest),
    )
    probe = _runtime_consumption_probe(output_dir / "inflate.py", candidate_archive, constants)
    parity = _decoder_state_parity_proof(source_probe, probe)
    runtime_files = _runtime_file_records(output_dir)
    blockers = _blockers_for_probe(probe, parity)
    return {
        "schema": ADAPTER_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_manifest": {
            "path": repo_relative(manifest_path, repo),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "candidate_archive": {
            "path": repo_relative(candidate_archive, repo),
            "bytes": candidate_archive.stat().st_size,
            "sha256": sha256_file(candidate_archive),
        },
        "source_archive": {
            "path": repo_relative(source_archive, repo),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "source_runtime_dir": repo_relative(source_dir, repo),
        "output_runtime_dir": repo_relative(output_dir, repo),
        "copied_files": copied_files,
        "constant_changes": changes,
        "shell_patch": shell_contract,
        "source_runtime_consumption_probe": _public_probe_record(source_probe),
        "runtime_consumption_probe": probe,
        "decoder_state_parity_proof": parity,
        "semantic_stream_parity": _mapping(manifest.get("semantic_stream_parity")),
        "runtime_files": runtime_files,
        "runtime_file_records_sha256": _runtime_tree_sha256(runtime_files),
        "runtime_tree_sha256": tree_sha256(output_dir),
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            "pr103_runtime_adapter_is_not_dispatch_authorization",
            *blockers,
        ],
    }


def build_pr103_lc_ac_candidate_packet(
    *,
    runtime_adapter_manifest: str | Path,
    packet_dir: str | Path,
    frame_parity_report: str | Path | None = None,
    shell_parity_report: str | Path | None = None,
    repo_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a local packet surface for pre-submission compliance.

    This packages the byte-different candidate archive with the adapted runtime
    and writes custody sidecars. It does not run scorers, open a dispatch claim,
    or mark the packet exact-eval ready.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest_path = _repo_path(Path(runtime_adapter_manifest), repo)
    manifest = _read_adapter_manifest(manifest_path)
    frame_parity = _read_frame_parity_report(frame_parity_report, repo=repo)
    shell_parity = _read_shell_parity_report(shell_parity_report, repo=repo)
    output_dir = _repo_path(Path(packet_dir), repo)
    candidate_record = _mapping(manifest.get("candidate_archive"))
    source_runtime_dir = _repo_path(Path(str(manifest.get("output_runtime_dir") or "")), repo)
    candidate_archive = _repo_path(Path(str(candidate_record.get("path") or "")), repo)
    _validate_inputs(source_runtime_dir, candidate_archive)
    _validate_candidate_archive_custody(candidate_record, candidate_archive)
    _validate_frame_parity_report(frame_parity, candidate_record)
    _validate_shell_parity_report(shell_parity, candidate_record)
    _validate_output_location(source_runtime_dir, output_dir)
    _prepare_output_dir(output_dir, force=force)

    packet_archive = output_dir / "archive.zip"
    shutil.copyfile(candidate_archive, packet_archive)
    os.chmod(packet_archive, 0o644)
    runtime_files = _copy_runtime_tree(source_runtime_dir, output_dir)
    archive_manifest = _archive_manifest(packet_archive, repo=repo)
    archive_manifest.update(
        {
            "schema": ARCHIVE_MANIFEST_SCHEMA,
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
            "source_runtime_adapter_manifest": {
                "path": repo_relative(manifest_path, repo),
                "sha256": sha256_file(manifest_path),
            },
            "decoder_state_parity_proof": _mapping(manifest.get("decoder_state_parity_proof")),
            "frame_output_parity_proof": _frame_parity_summary(frame_parity, shell_parity),
        }
    )
    write_json(output_dir / "archive_manifest.json", archive_manifest)
    report_text = _packet_report_text(
        archive_manifest=archive_manifest,
        runtime_adapter_manifest=manifest,
    )
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    os.chmod(report_path, 0o644)
    packet_runtime_files = _runtime_file_records(output_dir)
    adapter_blockers = [
        str(blocker) for blocker in manifest.get("readiness_blockers") or [] if str(blocker)
    ]
    packet_blockers = _unique_ordered(
        [
            *(
                ["full_frame_render_output_parity_missing"]
                if not _full_frame_parity_passed(frame_parity, shell_parity)
                else []
            ),
            *(
                ["shell_inflate_output_parity_missing"]
                if (
                    "shell_inflate_output_parity_missing" in adapter_blockers
                    or "full_frame_inflate_output_parity_missing" in adapter_blockers
                )
                and not _shell_inflate_parity_passed(frame_parity, shell_parity)
                else []
            ),
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    packet_manifest = {
        "schema": PACKET_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "scorers_invoked": False,
        "packet_dir": repo_relative(output_dir, repo),
        "archive": _file_record(packet_archive, relpath="archive.zip"),
        "archive_manifest": _file_record(output_dir / "archive_manifest.json", relpath="archive_manifest.json"),
        "report": _file_record(report_path, relpath="report.txt"),
        "runtime_adapter_manifest": {
            "path": repo_relative(manifest_path, repo),
            "sha256": sha256_file(manifest_path),
        },
        "runtime_files_copied": runtime_files,
        "runtime_files": packet_runtime_files,
        "runtime_file_records_sha256": _runtime_tree_sha256(packet_runtime_files),
        "runtime_tree_sha256": tree_sha256(output_dir),
        "adapter_runtime_tree_sha256": manifest.get("runtime_tree_sha256"),
        "decoder_state_parity_proof": _mapping(manifest.get("decoder_state_parity_proof")),
        "frame_output_parity_proof": _frame_parity_summary(frame_parity, shell_parity),
        "readiness_blockers": packet_blockers,
        "dispatch_blockers": [
            "pr103_candidate_packet_is_not_dispatch_authorization",
            *packet_blockers,
        ],
    }
    adapter_package = _archive_bound_candidate_adapter_package_for_packet(
        packet_manifest=packet_manifest,
        runtime_adapter_manifest=manifest,
        runtime_adapter_manifest_path=manifest_path,
        repo=repo,
    )
    packet_manifest.update(
        {
            "archive_bound_candidate_adapter_package_schema": (
                ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA
            ),
            "archive_bound_candidate_adapter_package": adapter_package,
            "archive_bound_candidate_adapter_package_candidate_count": (
                adapter_package["candidate_row_count"]
            ),
            "archive_bound_candidate_adapter_package_receiver_gate_passed_count": (
                adapter_package["receiver_proof_gate_passed_count"]
            ),
            "archive_bound_candidate_adapter_package_exact_blocker_count": len(
                adapter_package["exact_axis_blockers"]
            ),
        }
    )
    write_json(output_dir / "packet_manifest.json", packet_manifest)
    return packet_manifest


def probe_pr103_lc_ac_frame_parity(
    *,
    source_runtime_py: str | Path,
    source_archive: str | Path,
    candidate_runtime_py: str | Path,
    candidate_archive: str | Path,
    pair_indices: list[int],
    device: str = "cpu",
    batch_size: int = 16,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash rendered frame bytes for selected PR103 pair indices.

    This is an in-process decoder/render sanity probe, not an auth-eval or
    shell-level ``inflate.sh`` substitute. A sampled pass catches adapter/parser
    mistakes before CUDA spend. A full-scope pass proves rendered-frame parity
    for this imported runtime module path, but it still cannot clear the
    shell-inflate parity blocker by itself.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    source_runtime = _repo_path(Path(source_runtime_py), repo)
    candidate_runtime = _repo_path(Path(candidate_runtime_py), repo)
    source_zip = _repo_path(Path(source_archive), repo)
    candidate_zip = _repo_path(Path(candidate_archive), repo)
    if not pair_indices:
        raise Pr103RuntimeAdapterError("pair_indices must not be empty")
    source_result = _render_frame_digest(
        runtime_py=source_runtime,
        archive_path=source_zip,
        pair_indices=pair_indices,
        device=device,
        batch_size=batch_size,
    )
    candidate_result = _render_frame_digest(
        runtime_py=candidate_runtime,
        archive_path=candidate_zip,
        pair_indices=pair_indices,
        device=device,
        batch_size=batch_size,
    )
    pair_hashes_match = source_result["pair_hashes"] == candidate_result["pair_hashes"]
    output_sha_match = source_result["output_sha256"] == candidate_result["output_sha256"]
    parsed_lengths_match = source_result["parsed_lengths"] == candidate_result["parsed_lengths"]
    full_scope = _is_full_pair_scope(pair_indices, int(source_result["n_pairs"]))
    parity_passed = bool(pair_hashes_match and output_sha_match)
    full_parity = bool(parity_passed and full_scope)
    blockers = _unique_ordered(
        [
            *(["frame_output_parity_mismatch"] if not parity_passed else []),
            *(["full_frame_render_output_parity_missing"] if not full_parity else []),
            "shell_inflate_output_parity_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return {
        "schema": FRAME_PARITY_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": device != "cpu",
        "ready_for_exact_eval_dispatch": False,
        "device": device,
        "batch_size": int(batch_size),
        "parity_method": "in_process_runtime_decoder_render",
        "shell_inflate_output_parity_proven": False,
        "pair_indices": [int(index) for index in pair_indices],
        "frame_output_parity_scope": "full" if full_scope else "sampled",
        "sampled_frame_output_parity_proven": parity_passed,
        "full_frame_output_parity_proven": full_parity,
        "pair_hashes_match": pair_hashes_match,
        "output_sha_match": output_sha_match,
        "parsed_lengths_match": parsed_lengths_match,
        "source": {
            "runtime_py": _file_record(source_runtime, relpath=repo_relative(source_runtime, repo)),
            "archive": _file_record(source_zip, relpath=repo_relative(source_zip, repo)),
            "render": source_result,
        },
        "candidate": {
            "runtime_py": _file_record(candidate_runtime, relpath=repo_relative(candidate_runtime, repo)),
            "archive": _file_record(candidate_zip, relpath=repo_relative(candidate_zip, repo)),
            "render": candidate_result,
        },
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            "pr103_frame_parity_probe_is_not_dispatch_authorization",
            *blockers,
        ],
    }


def _read_candidate_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise Pr103RuntimeAdapterError("candidate manifest must be a JSON object")
    if payload.get("schema") != CANDIDATE_SCHEMA:
        raise Pr103RuntimeAdapterError(
            "candidate manifest must come from PR103 histogram candidate materializer"
        )
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise Pr103RuntimeAdapterError("candidate manifest must be a no-score local artifact")
    return payload


def _read_adapter_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise Pr103RuntimeAdapterError("runtime adapter manifest must be a JSON object")
    if payload.get("schema") != ADAPTER_SCHEMA:
        raise Pr103RuntimeAdapterError(f"runtime adapter manifest must have schema {ADAPTER_SCHEMA}")
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise Pr103RuntimeAdapterError("runtime adapter manifest must be a no-score local artifact")
    parity = _mapping(payload.get("decoder_state_parity_proof"))
    if parity.get("passed") is not True:
        raise Pr103RuntimeAdapterError("runtime adapter manifest missing decoder-state parity proof")
    return payload


def _read_frame_parity_report(path: str | Path | None, *, repo: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    report_path = _repo_path(Path(path), repo)
    payload = read_json(report_path)
    if not isinstance(payload, dict):
        raise Pr103RuntimeAdapterError("frame parity report must be a JSON object")
    if payload.get("schema") != FRAME_PARITY_SCHEMA:
        raise Pr103RuntimeAdapterError(
            f"expected frame parity schema {FRAME_PARITY_SCHEMA}, got {payload.get('schema')!r}"
        )
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise Pr103RuntimeAdapterError("frame parity report must be a no-score local artifact")
    payload = dict(payload)
    payload["_report_path"] = repo_relative(report_path, repo)
    payload["_report_sha256"] = sha256_file(report_path)
    return payload


def _read_shell_parity_report(path: str | Path | None, *, repo: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    report_path = _repo_path(Path(path), repo)
    payload = read_json(report_path)
    if not isinstance(payload, dict):
        raise Pr103RuntimeAdapterError("shell parity report must be a JSON object")
    if payload.get("schema") != SHELL_PARITY_SCHEMA:
        raise Pr103RuntimeAdapterError(
            f"expected shell parity schema {SHELL_PARITY_SCHEMA}, got {payload.get('schema')!r}"
        )
    if payload.get("score_claim") is True or payload.get("dispatch_attempted") is True:
        raise Pr103RuntimeAdapterError("shell parity report must be a no-score local artifact")
    payload = dict(payload)
    payload["_report_path"] = repo_relative(report_path, repo)
    payload["_report_sha256"] = sha256_file(report_path)
    return payload


def _validate_frame_parity_report(
    report: dict[str, Any] | None,
    candidate_archive_record: dict[str, Any],
) -> None:
    if report is None:
        return
    if report.get("full_frame_output_parity_proven") is not True:
        raise Pr103RuntimeAdapterError("frame parity report does not prove full-frame parity")
    candidate = _mapping(report.get("candidate"))
    archive = _mapping(candidate.get("archive"))
    if archive.get("sha256") != candidate_archive_record.get("sha256"):
        raise Pr103RuntimeAdapterError(
            "frame parity candidate archive sha256 does not match packet archive"
        )
    if archive.get("bytes") != candidate_archive_record.get("bytes"):
        raise Pr103RuntimeAdapterError("frame parity candidate archive bytes do not match packet archive")


def _validate_shell_parity_report(
    report: dict[str, Any] | None,
    candidate_archive_record: dict[str, Any],
) -> None:
    if report is None:
        return
    if report.get("passed") is not True:
        raise Pr103RuntimeAdapterError("shell parity report did not pass")
    if report.get("output_mismatches"):
        raise Pr103RuntimeAdapterError("shell parity report contains output mismatches")
    candidate = _mapping(report.get("candidate"))
    archive = _mapping(candidate.get("archive"))
    if archive.get("sha256") != candidate_archive_record.get("sha256"):
        raise Pr103RuntimeAdapterError(
            "shell parity candidate archive sha256 does not match packet archive"
        )
    if archive.get("bytes") != candidate_archive_record.get("bytes"):
        raise Pr103RuntimeAdapterError("shell parity candidate archive bytes do not match packet archive")
    source_outputs = _shell_output_map(_mapping(report.get("source")).get("outputs"))
    candidate_outputs = _shell_output_map(candidate.get("outputs"))
    if not source_outputs or source_outputs != candidate_outputs:
        raise Pr103RuntimeAdapterError("shell parity report output records are not identical")


def _full_frame_parity_passed(
    frame_report: dict[str, Any] | None,
    shell_report: dict[str, Any] | None = None,
) -> bool:
    return bool(
        (frame_report is not None and frame_report.get("full_frame_output_parity_proven") is True)
        or _shell_inflate_parity_passed(frame_report, shell_report)
    )


def _shell_inflate_parity_passed(
    frame_report: dict[str, Any] | None,
    shell_report: dict[str, Any] | None = None,
) -> bool:
    return bool(
        (frame_report is not None and frame_report.get("shell_inflate_output_parity_proven") is True)
        or (shell_report is not None and shell_report.get("passed") is True)
    )


def _frame_parity_summary(
    frame_report: dict[str, Any] | None,
    shell_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame_report is None and shell_report is None:
        return {
            "provided": False,
            "full_frame_output_parity_proven": False,
            "shell_inflate_output_parity_proven": False,
        }
    if frame_report is None:
        source_outputs = _shell_output_map(_mapping(_mapping(shell_report).get("source")).get("outputs"))
        candidate_outputs = _shell_output_map(
            _mapping(_mapping(shell_report).get("candidate")).get("outputs")
        )
        return {
            "provided": True,
            "path": shell_report.get("_report_path") if shell_report else None,
            "sha256": shell_report.get("_report_sha256") if shell_report else None,
            "parity_method": shell_report.get("parity_method") if shell_report else None,
            "frame_output_parity_scope": "inflate_shell_full_outputs",
            "sampled_frame_output_parity_proven": False,
            "full_frame_output_parity_proven": _shell_inflate_parity_passed(None, shell_report),
            "shell_inflate_output_parity_proven": _shell_inflate_parity_passed(None, shell_report),
            "source_output_sha256": _single_output_sha(source_outputs),
            "candidate_output_sha256": _single_output_sha(candidate_outputs),
            "output_bytes": _single_output_bytes(source_outputs),
            "shell_output_count": len(source_outputs),
        }
    report = frame_report
    return {
        "provided": True,
        "path": report.get("_report_path"),
        "sha256": report.get("_report_sha256"),
        "parity_method": report.get("parity_method"),
        "frame_output_parity_scope": report.get("frame_output_parity_scope"),
        "sampled_frame_output_parity_proven": report.get("sampled_frame_output_parity_proven"),
        "full_frame_output_parity_proven": report.get("full_frame_output_parity_proven"),
        "shell_inflate_output_parity_proven": report.get("shell_inflate_output_parity_proven")
        is True or _shell_inflate_parity_passed(None, shell_report),
        "source_output_sha256": _mapping(_mapping(report.get("source")).get("render")).get(
            "output_sha256"
        ),
        "candidate_output_sha256": _mapping(_mapping(report.get("candidate")).get("render")).get(
            "output_sha256"
        ),
        "output_bytes": _mapping(_mapping(report.get("source")).get("render")).get("output_bytes"),
    }


def _shell_output_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for item in value:
        row = _mapping(item)
        relpath = str(row.get("relative_path") or "")
        if relpath:
            rows[relpath] = {
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            }
    return rows


def _single_output_sha(outputs: dict[str, dict[str, Any]]) -> str | None:
    if len(outputs) != 1:
        return None
    return next(iter(outputs.values())).get("sha256")


def _single_output_bytes(outputs: dict[str, dict[str, Any]]) -> int | None:
    if len(outputs) != 1:
        return None
    value = next(iter(outputs.values())).get("bytes")
    return int(value) if value is not None else None


def _candidate_runtime_constants(manifest: dict[str, Any]) -> dict[str, int]:
    contract = _mapping(manifest.get("runtime_adapter_contract"))
    raw = _mapping(contract.get("public_runtime_constants"))
    return _runtime_constants(raw, label="candidate")


def _source_runtime_constants(manifest: dict[str, Any]) -> dict[str, int]:
    contract = _mapping(manifest.get("runtime_adapter_contract"))
    raw = _mapping(contract.get("source_runtime_constants"))
    if not raw:
        raise Pr103RuntimeAdapterError("candidate manifest missing source runtime constants")
    return _runtime_constants(raw, label="source")


def _runtime_constants(raw: dict[str, Any], *, label: str) -> dict[str, int]:
    constants: dict[str, int] = {}
    for name in REQUIRED_RUNTIME_CONSTANTS:
        if name not in raw:
            raise Pr103RuntimeAdapterError(f"{label} manifest missing runtime constant {name}")
        constants[name] = int(raw[name])
        if constants[name] < 0:
            raise Pr103RuntimeAdapterError(f"runtime constant {name} must be nonnegative")
    return constants


def _validate_inputs(source_runtime_dir: Path, candidate_archive: Path) -> None:
    if not source_runtime_dir.is_dir():
        raise FileNotFoundError(f"source runtime directory not found: {source_runtime_dir}")
    for name in ("inflate.py", "inflate.sh"):
        if not (source_runtime_dir / name).is_file():
            raise FileNotFoundError(f"source runtime missing {name}: {source_runtime_dir}")
    if not candidate_archive.is_file():
        raise FileNotFoundError(f"candidate archive not found: {candidate_archive}")


def _validate_candidate_archive_custody(
    record: dict[str, Any],
    candidate_archive: Path,
    *,
    label: str = "candidate archive",
) -> None:
    expected_bytes = record.get("bytes")
    expected_sha = str(record.get("sha256") or "")
    if expected_bytes is None or not expected_sha:
        raise Pr103RuntimeAdapterError(f"candidate manifest missing {label} bytes or sha256")
    actual_bytes = candidate_archive.stat().st_size
    actual_sha = sha256_file(candidate_archive)
    if int(expected_bytes) != actual_bytes:
        raise Pr103RuntimeAdapterError(
            f"{label} byte mismatch: manifest={expected_bytes} actual={actual_bytes}"
        )
    if expected_sha != actual_sha:
        raise Pr103RuntimeAdapterError(
            f"{label} sha256 mismatch: manifest={expected_sha} actual={actual_sha}"
        )


def _validate_output_location(source_runtime_dir: Path, output_runtime_dir: Path) -> None:
    source = source_runtime_dir.resolve()
    output = output_runtime_dir.resolve()
    if output == source:
        raise Pr103RuntimeAdapterError("output runtime directory must differ from source")
    if _path_is_relative_to(output, source):
        raise Pr103RuntimeAdapterError("output runtime directory cannot be inside source runtime")
    if _path_is_relative_to(source, output):
        raise Pr103RuntimeAdapterError("source runtime directory cannot be inside output runtime")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise Pr103RuntimeAdapterError(
                f"output runtime directory exists; pass force to replace: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _copy_runtime_tree(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix()):
        rel = source.relative_to(source_dir)
        if _should_exclude(rel):
            continue
        target = output_dir / rel
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if source.is_symlink():
            raise Pr103RuntimeAdapterError(f"runtime adapter refuses symlink: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, stat.S_IMODE(source.stat().st_mode))
        copied.append(rel.as_posix())
    return copied


def _patch_inflate_constants(path: Path, constants: dict[str, int]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    assignments = _integer_constant_assignments(text)
    changes: list[dict[str, Any]] = []
    for name, value in constants.items():
        if name not in assignments:
            raise Pr103RuntimeAdapterError(
                f"inflate.py must contain one top-level integer assignment for {name}"
            )
        pattern = re.compile(rf"^(?P<prefix>{re.escape(name)}\s*=\s*)(?P<value>\d+)\s*$", re.M)
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise Pr103RuntimeAdapterError(
                f"expected exactly one integer assignment for {name}, found {len(matches)}"
            )
        old_value = int(matches[0].group("value"))
        if old_value != assignments[name]:
            raise Pr103RuntimeAdapterError(
                f"AST and text assignment disagree for {name}: ast={assignments[name]} text={old_value}"
            )
        text = pattern.sub(rf"\g<prefix>{value}", text)
        changes.append({"name": name, "old": old_value, "new": int(value), "changed": old_value != value})
    path.write_text(text, encoding="utf-8")
    return changes


def _verify_inflate_shell_contract(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    expected = 'python "$HERE/inflate.py" "$SRC" "$DST"'
    count = text.count(expected)
    if count != 1:
        raise Pr103RuntimeAdapterError(
            f"expected exactly one bare python inflate invocation, found {count}"
        )
    return {
        "changed": False,
        "contract": expected,
        "basis": "preserve_source_pr103_inflate_shell_interpreter_contract",
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _integer_constant_assignments(text: str) -> dict[str, int]:
    """Catalog #168 fix 2026-05-12: handle both `BLOCK_LEN = 16` (Assign) and
    `BLOCK_LEN: int = 16` (AnnAssign) module-level constants. The latter is
    increasingly common in modern Python and would have been silently missed."""
    module = ast.parse(text)
    assignments: dict[str, int] = {}
    seen: set[str] = set()
    for node in module.body:
        # Normalize Assign / AnnAssign into (target_name, value) pair.
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value_node = node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id not in REQUIRED_RUNTIME_CONSTANTS:
            continue
        if target.id in seen:
            raise Pr103RuntimeAdapterError(
                f"inflate.py contains duplicate top-level assignment for {target.id}"
            )
        seen.add(target.id)
        if (
            not isinstance(value_node, ast.Constant)
            or isinstance(value_node.value, bool)
            or not isinstance(value_node.value, int)
        ):
            raise Pr103RuntimeAdapterError(
                f"inflate.py assignment for {target.id} must be an integer literal"
            )
        assignments[target.id] = int(value_node.value)
    return assignments


def _runtime_consumption_probe(
    inflate_py: Path,
    candidate_archive: Path,
    constants: dict[str, int],
) -> dict[str, Any]:
    source = read_strict_single_member_zip(candidate_archive)
    runtime = _load_runtime_module(inflate_py)
    sections = runtime.parse_archive(source.payload)
    if len(sections) != 8:
        raise Pr103RuntimeAdapterError(f"runtime parse_archive returned {len(sections)} sections")
    sca, br_b, hists_b, merged_ac, mins_scales, lo_b, hi_hist_b, wrp_b = sections
    lengths = {
        "BR_LEN": len(br_b),
        "HIST_LEN": len(hists_b),
        "MERGED_AC_LEN": len(merged_ac),
        "LO_LEN": len(lo_b),
        "HI_HIST_LEN": len(hi_hist_b),
    }
    mismatches = {
        key: {"expected": value, "actual": lengths[key]}
        for key, value in constants.items()
        if lengths[key] != value
    }
    if mismatches:
        raise Pr103RuntimeAdapterError(f"runtime parse length mismatch: {mismatches}")
    hi_hist = np.frombuffer(runtime.brotli.decompress(hi_hist_b), dtype=np.uint16)
    state_dict, hi_decoded = runtime.build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist)
    latents = runtime.decode_latents(mins_scales, lo_b, hi_decoded)
    runtime.apply_corrections(latents, wrp_b)
    state_records = _state_dict_records(state_dict)
    latent_record = _tensor_record(latents)
    return {
        "passed": True,
        "member_name": source.member_name,
        "payload_bytes": source.member_bytes,
        "parsed_lengths": lengths,
        "sidecar_tail_bytes": len(wrp_b),
        "state_dict_tensors": len(state_dict),
        "state_dict_params": int(sum(_tensor_numel(tensor) for tensor in state_dict.values())),
        "state_dict_sha256": sha256_bytes(json_text(state_records).encode("utf-8")),
        "state_dict_tensor_records": state_records,
        "latents_shape": [int(value) for value in latents.shape],
        "latents_sha256": latent_record["sha256"],
        "latents_dtype": latent_record["dtype"],
        "full_frame_inflate_ran": False,
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _render_frame_digest(
    *,
    runtime_py: Path,
    archive_path: Path,
    pair_indices: list[int],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    constants = _integer_constant_assignments(runtime_py.read_text(encoding="utf-8"))
    missing = [name for name in REQUIRED_RUNTIME_CONSTANTS if name not in constants]
    if missing:
        raise Pr103RuntimeAdapterError(f"runtime missing required constants: {missing}")
    runtime = _load_runtime_module(runtime_py)
    source = read_strict_single_member_zip(archive_path)
    sections = runtime.parse_archive(source.payload)
    if len(sections) != 8:
        raise Pr103RuntimeAdapterError(f"runtime parse_archive returned {len(sections)} sections")
    sca, br_b, hists_b, merged_ac, mins_scales, lo_b, hi_hist_b, wrp_b = sections
    lengths = {
        "BR_LEN": len(br_b),
        "HIST_LEN": len(hists_b),
        "MERGED_AC_LEN": len(merged_ac),
        "LO_LEN": len(lo_b),
        "HI_HIST_LEN": len(hi_hist_b),
    }
    mismatches = {
        key: {"expected": value, "actual": lengths[key]}
        for key, value in constants.items()
        if lengths[key] != value
    }
    if mismatches:
        raise Pr103RuntimeAdapterError(f"runtime parse length mismatch: {mismatches}")
    hi_hist = np.frombuffer(runtime.brotli.decompress(hi_hist_b), dtype=np.uint16)
    state_dict, hi_decoded = runtime.build_state_dict(br_b, hists_b, merged_ac, sca, hi_hist)
    latents = runtime.decode_latents(mins_scales, lo_b, hi_decoded)
    runtime.apply_corrections(latents, wrp_b)
    n_pairs = int(latents.shape[0])
    normalized_indices = _normalize_pair_indices(pair_indices, n_pairs=n_pairs)
    if batch_size <= 0:
        raise Pr103RuntimeAdapterError("batch_size must be positive")
    torch = runtime.torch
    nn_device = torch.device(device)
    decoder = runtime.HNeRVDecoder(runtime.LATENT_DIM, runtime.BASE_CHANNELS, runtime.EVAL_SIZE).to(
        nn_device
    )
    decoder.load_state_dict(state_dict)
    decoder.eval()
    latents = latents.to(nn_device)
    pair_hashes = []
    chunks: list[bytes] = []
    with torch.inference_mode():
        for start in range(0, len(normalized_indices), batch_size):
            batch_indices = normalized_indices[start : start + batch_size]
            index_tensor = torch.tensor(batch_indices, dtype=torch.long, device=nn_device)
            decoded = decoder(latents.index_select(0, index_tensor))
            flat = decoded.reshape(len(batch_indices) * 2, 3, *runtime.EVAL_SIZE)
            up = runtime.F.interpolate(
                flat,
                size=(runtime.CAMERA_H, runtime.CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            frames = (
                up.clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            frames_by_pair = frames.reshape(len(batch_indices), 2, runtime.CAMERA_H, runtime.CAMERA_W, 3)
            for offset, pair_index in enumerate(batch_indices):
                payload = np.ascontiguousarray(frames_by_pair[offset]).tobytes()
                chunks.append(payload)
                pair_hashes.append(
                    {
                        "pair_index": pair_index,
                        "frame_count": 2,
                        "bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                )
    output = b"".join(chunks)
    return {
        "member_name": source.member_name,
        "payload_bytes": source.member_bytes,
        "parsed_lengths": lengths,
        "n_pairs": n_pairs,
        "pair_hashes": pair_hashes,
        "output_bytes": len(output),
        "output_sha256": sha256_bytes(output),
    }


def _load_runtime_module(path: Path) -> ModuleType:
    module_name = "pr103_lc_ac_runtime_adapter_" + sha256_file(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise Pr103RuntimeAdapterError(f"could not import runtime module: {path}")
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _runtime_file_records(runtime_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(runtime_dir.rglob("*"), key=lambda item: item.relative_to(runtime_dir).as_posix()):
        rel = path.relative_to(runtime_dir)
        if path.is_file() and not _should_exclude(rel):
            records.append(
                {
                    "relpath": rel.as_posix(),
                    "bytes": path.stat().st_size,
                    "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
                    "sha256": sha256_file(path),
                }
            )
    return records


def _file_record(path: Path, *, relpath: str) -> dict[str, Any]:
    return {
        "relpath": relpath,
        "bytes": path.stat().st_size,
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        "sha256": sha256_file(path),
    }


def _archive_manifest(path: Path, *, repo: Path) -> dict[str, Any]:
    record = {
        "candidate_archive": {
            "path": repo_relative(path, repo),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "members": [],
        }
    }
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            payload = zf.read(info)
            record["candidate_archive"]["members"].append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": int(info.CRC),
                    "sha256": sha256_bytes(payload),
                }
            )
    return record


def _archive_bound_candidate_adapter_package_for_packet(
    *,
    packet_manifest: dict[str, Any],
    runtime_adapter_manifest: dict[str, Any],
    runtime_adapter_manifest_path: Path,
    repo: Path,
) -> dict[str, Any]:
    archive = _mapping(packet_manifest.get("archive"))
    source_archive = _mapping(runtime_adapter_manifest.get("source_archive"))
    parity = _mapping(runtime_adapter_manifest.get("decoder_state_parity_proof"))
    runtime_probe = _mapping(runtime_adapter_manifest.get("runtime_consumption_probe"))
    runtime_adapter_record = {
        "path": repo_relative(runtime_adapter_manifest_path, repo),
        "sha256": sha256_file(runtime_adapter_manifest_path),
        "runtime_tree_sha256": runtime_adapter_manifest.get("runtime_tree_sha256"),
        "runtime_file_records_sha256": runtime_adapter_manifest.get(
            "runtime_file_records_sha256"
        ),
        "runtime_dir": runtime_adapter_manifest.get("output_runtime_dir"),
        "runtime_adapter_ready": True,
        "schema": runtime_adapter_manifest.get("schema"),
    }
    candidate_id = (
        "pr103_lc_ac_candidate_packet:"
        + str(archive.get("sha256") or "")[:16]
    )
    packet_archive_path = str(Path(packet_manifest["packet_dir"]) / "archive.zip")
    row = {
        "schema": "pr103_lc_ac_archive_bound_candidate_row.v1",
        "candidate_id": candidate_id,
        "candidate_family": "pr103_lc_ac",
        "archive_native_transform_kind": "pr103_lc_ac_range_arithmetic_runtime_packet",
        "candidate_archive_path": packet_archive_path,
        "candidate_archive_sha256": str(archive.get("sha256") or ""),
        "candidate_archive_bytes": archive.get("bytes"),
        "source_archive_path": source_archive.get("path"),
        "source_archive_sha256": source_archive.get("sha256"),
        "source_archive_bytes": source_archive.get("bytes"),
        "byte_closed_candidate_materialized": True,
        "candidate_archive_materialized": True,
        "runtime_consumption_proof_ready": runtime_probe.get("passed") is True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": repo_relative(runtime_adapter_manifest_path, repo),
        "receiver_contract_kind": "pr103_lc_ac_decoder_state_parity_runtime_adapter",
        "receiver_contract_satisfied": parity.get("passed") is True,
        "runtime_adapter_ready": True,
        "contest_runtime_decoder_adapter_ready": True,
        "runtime_adapter_manifest": runtime_adapter_record,
        "readiness_blockers": packet_manifest.get("readiness_blockers") or [],
        "dispatch_blockers": packet_manifest.get("dispatch_blockers") or [],
        "replay_argv": [
            sys.executable,
            "tools/build_pr103_lc_ac_candidate_packet.py",
            "--runtime-adapter-manifest",
            repo_relative(runtime_adapter_manifest_path, repo),
            "--packet-dir",
            str(packet_manifest["packet_dir"]),
        ],
        "input_artifacts": [
            repo_relative(runtime_adapter_manifest_path, repo),
            str(_mapping(runtime_adapter_manifest.get("candidate_manifest")).get("path") or ""),
            str(source_archive.get("path") or ""),
            packet_archive_path,
        ],
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }
    return build_archive_bound_candidate_adapter_package(
        _SinglePr103ArchiveCandidateAdapter(
            adapter_id="pr103_lc_ac_candidate_packet:archive_bound_adapter",
            candidate_family="pr103_lc_ac",
            row=row,
        ),
        repo_root=repo,
    )


def _packet_report_text(
    *,
    archive_manifest: dict[str, Any],
    runtime_adapter_manifest: dict[str, Any],
) -> str:
    archive = _mapping(archive_manifest.get("candidate_archive"))
    parity = _mapping(runtime_adapter_manifest.get("decoder_state_parity_proof"))
    frame_parity = _mapping(archive_manifest.get("frame_output_parity_proof"))
    return "\n".join(
        [
            "PR103 LC-AC histogram candidate packet",
            "",
            "score_claim: false",
            "dispatch_attempted: false",
            "ready_for_exact_eval_dispatch: false",
            f"archive_sha256: {archive.get('sha256')}",
            f"archive_size_bytes: {archive.get('bytes')}",
            f"runtime_tree_sha256: {runtime_adapter_manifest.get('runtime_tree_sha256')}",
            f"decoder_state_parity_passed: {parity.get('passed') is True}",
            "decoder_state_full_frame_output_parity_proven: "
            f"{parity.get('full_frame_output_parity_proven') is True}",
            "decoder_state_full_frame_output_parity_required: "
            f"{parity.get('full_frame_output_parity_required') is True}",
            f"render_frame_parity_report_provided: {frame_parity.get('provided') is True}",
            "render_frame_parity_full_output_proven: "
            f"{frame_parity.get('full_frame_output_parity_proven') is True}",
            "shell_inflate_output_parity_proven: "
            f"{frame_parity.get('shell_inflate_output_parity_proven') is True}",
            "",
            "This packet is a compliance-smoke artifact only. It is not a score claim.",
            "",
        ]
    )


def _state_dict_records(state_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": str(name), **_tensor_record(state_dict[name])}
        for name in sorted(state_dict)
    ]


def _tensor_record(value: Any) -> dict[str, Any]:
    array = (
        value.detach().cpu().contiguous().numpy()
        if hasattr(value, "detach")
        else np.asarray(value)
    )
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": [int(item) for item in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "sha256": sha256_bytes(contiguous.tobytes()),
    }


def _tensor_numel(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return int(np.asarray(value).size)


def _normalize_pair_indices(pair_indices: list[int], *, n_pairs: int) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_index in pair_indices:
        index = int(raw_index)
        if index < 0 or index >= n_pairs:
            raise Pr103RuntimeAdapterError(
                f"pair index {index} outside valid range [0, {n_pairs - 1}]"
            )
        if index not in seen:
            seen.add(index)
            normalized.append(index)
    return normalized


def _is_full_pair_scope(pair_indices: list[int], n_pairs: int) -> bool:
    return _normalize_pair_indices(pair_indices, n_pairs=n_pairs) == list(range(n_pairs))


def _decoder_state_parity_proof(
    source_probe: dict[str, Any],
    candidate_probe: dict[str, Any],
) -> dict[str, Any]:
    tensor_records_match = (
        source_probe.get("state_dict_tensor_records")
        == candidate_probe.get("state_dict_tensor_records")
    )
    state_sha_match = source_probe.get("state_dict_sha256") == candidate_probe.get("state_dict_sha256")
    latents_sha_match = source_probe.get("latents_sha256") == candidate_probe.get("latents_sha256")
    latents_shape_match = source_probe.get("latents_shape") == candidate_probe.get("latents_shape")
    passed = bool(tensor_records_match and state_sha_match and latents_sha_match and latents_shape_match)
    return {
        "passed": passed,
        "basis": "source_and_candidate_decode_to_identical_decoder_state_and_latents",
        "source_state_dict_sha256": source_probe.get("state_dict_sha256"),
        "candidate_state_dict_sha256": candidate_probe.get("state_dict_sha256"),
        "state_dict_sha_match": state_sha_match,
        "state_dict_tensor_records_match": tensor_records_match,
        "source_latents_sha256": source_probe.get("latents_sha256"),
        "candidate_latents_sha256": candidate_probe.get("latents_sha256"),
        "latents_sha_match": latents_sha_match,
        "latents_shape_match": latents_shape_match,
        "full_frame_inflate_ran": False,
        "full_frame_output_parity_proven": False,
        "full_frame_output_parity_required": True,
        "full_frame_output_parity_note": (
            "decoded state/latent parity is necessary but not sufficient; "
            "promotion requires source-vs-candidate inflate output parity or exact "
            "same-runtime auth eval on both packets"
        ),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _public_probe_record(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in probe.items()
        if key != "state_dict_tensor_records"
    }


def _runtime_tree_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json_text(records).encode("utf-8")).hexdigest()


def _blockers_for_probe(probe: dict[str, Any], parity: dict[str, Any]) -> list[str]:
    blockers = [
        "strict_pre_submission_compliance_json_missing",
        "full_frame_render_output_parity_missing",
        "shell_inflate_output_parity_missing",
        "lane_dispatch_claim_missing",
        "exact_cuda_auth_eval_missing",
    ]
    if probe.get("passed") is not True:
        blockers.insert(0, "runtime_consumption_probe_failed")
    if parity.get("passed") is not True:
        blockers.insert(0, "decoder_state_latent_parity_missing")
    return blockers


def _repo_path(path: Path, repo: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _resolve_manifest_path(path: Path, *, manifest_path: Path, repo: Path) -> Path:
    if path.is_absolute():
        return path
    repo_candidate = repo / path
    if repo_candidate.exists():
        return repo_candidate
    return manifest_path.parent / path


def _should_exclude(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _unique_ordered(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


__all__ = [
    "ADAPTER_SCHEMA",
    "ARCHIVE_MANIFEST_SCHEMA",
    "PACKET_SCHEMA",
    "SHELL_PARITY_SCHEMA",
    "Pr103RuntimeAdapterError",
    "build_pr103_lc_ac_candidate_packet",
    "build_pr103_lc_ac_runtime_adapter",
]
