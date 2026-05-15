#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a PR106 latent score-table output into a byte-closed candidate.

This tool is the local bridge between long-running Kaggle CUDA table jobs and
exact-eval dispatch. It consumes a completed ``score_table.npy`` plus its CUDA
provenance manifest, runs the canonical PR106 sidecar builder in scorer-free
``score_table`` mode, and writes a materialization manifest that remains
explicitly non-promotional until exact contest-CUDA adjudication scores the
emitted archive.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    apply_proxy_evidence_boundary,
    ordered_unique,
    validate_proxy_candidate,
)
from tac.packet_compiler.pr106_latent_sidecar_selection import (  # noqa: E402
    build_latent_candidate_grid,
    choose_latent_corrections_from_score_table_file,
    latent_candidate_grid_npy_sha256,
    validate_score_table_manifest,
)
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_LATENT_N_DIMS,
    PR106_LATENT_N_PAIRS,
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    emit_pr106_format0c_semantic_candidate_archive,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
    pr106_sidecar_manifest,
    read_single_stored_member_archive,
    sha256_hex,
)
from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
DEFAULT_SCORE_TABLE_ROOT = REPO_ROOT / "reports/raw/kaggle_pr106_latent_score_table_latest"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/results"
BUILDER = REPO_ROOT / "experiments/build_pr106_latent_sidecar.py"
SCORE_TABLE_FILENAME = "score_table.npy"
SCORE_TABLE_MANIFEST_FILENAME = "score_table_manifest.json"
MANIFEST_AUDIT_SCHEMA = "pr106_latent_score_table_materialization_audit_v1"
BUILDER_METADATA_AUDIT_SCHEMA = "pr106_latent_score_table_builder_metadata_audit_v1"
SCORE_TABLE_MANIFEST_REQUIRED_FIELDS = (
    "manifest_schema",
    "producer",
    "score_claim",
    "ready_for_builder",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "remote_jobs_dispatched",
    "source_archive_path",
    "source_archive_bytes",
    "source_archive_sha256",
    "source_archive_member_name",
    "source_archive_member_sha256",
    "source_payload_kind",
    "runtime_dir",
    "candidate_grid_path",
    "candidate_grid_sha256",
    "candidate_grid_npy_sha256",
    "score_table_npy_path",
    "score_table_npy_bytes",
    "score_table_npy_sha256",
    "delta_radius",
    "latent_dim",
    "candidate_count",
    "n_pairs",
    "score_table_shape",
    "objective",
    "pair_marginal_semantics",
    "noop_candidate_index",
    "strict_improvement_pair_count",
    "best_improvement_min",
    "best_improvement_mean",
    "best_improvement_max",
    "device",
    "torch_version",
    "cuda_version",
    "elapsed_seconds",
    "lane_claim_verified",
    "lane_claim",
    "dispatch_blockers",
)
MATERIALIZATION_DISPATCH_BLOCKERS = (
    "kaggle_score_table_materialization_is_proxy_evidence_boundary",
    "requires_lane_dispatch_claim_before_exact_eval",
    "requires_paired_contest_cuda_auth_eval_on_materialized_archive",
    "requires_paired_contest_cpu_auth_eval_on_materialized_archive",
    "requires_adjudicated_component_recompute_before_score_claim",
)


@dataclass(frozen=True)
class SourcePacketInfo:
    member_name: str
    payload_sha256: str
    format_id: int | None


def default_run_id() -> str:
    return "pr106_latent_score_table_materialized_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _candidate_roots(root: Path) -> Iterable[Path]:
    yield root
    yield root / "score_table"
    yield root / "latent_run" / "score_table"
    yield root / "pr106_latent_score_table" / "score_table"
    yield root / "pr106_latent_score_table" / "latent_run" / "score_table"


def resolve_score_table_artifacts(
    *,
    score_table_root: Path,
    score_table_npy: Path | None,
    score_table_manifest: Path | None,
) -> tuple[Path, Path]:
    """Resolve a score table pair from explicit paths or known Kaggle layouts."""

    if (score_table_npy is None) ^ (score_table_manifest is None):
        raise ValueError("--score-table-npy and --score-table-manifest must be supplied together")
    if score_table_npy is not None and score_table_manifest is not None:
        npy = score_table_npy
        manifest = score_table_manifest
    else:
        matches: list[tuple[Path, Path]] = []
        for candidate_root in _candidate_roots(score_table_root):
            npy_candidate = candidate_root / SCORE_TABLE_FILENAME
            manifest_candidate = candidate_root / SCORE_TABLE_MANIFEST_FILENAME
            if npy_candidate.is_file() and manifest_candidate.is_file():
                matches.append((npy_candidate, manifest_candidate))
        unique = sorted({(npy.resolve(), manifest.resolve()) for npy, manifest in matches})
        if not unique:
            raise FileNotFoundError(
                "no score_table.npy + score_table_manifest.json pair found under "
                f"{score_table_root}"
            )
        if len(unique) > 1:
            rendered = ", ".join(f"{npy.parent}" for npy, _manifest in unique)
            raise ValueError(
                "multiple score-table artifact pairs found; pass explicit paths: "
                f"{rendered}"
            )
        npy, manifest = unique[0]
    if not npy.is_file():
        raise FileNotFoundError(f"score table .npy not found: {npy}")
    if not manifest.is_file():
        raise FileNotFoundError(f"score table manifest not found: {manifest}")
    return npy, manifest


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": repo_relative(path, REPO_ROOT),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _manifest_mapping(path: Path) -> Mapping[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"score-table manifest must be a JSON object: {path}")
    return payload


def _authority_flags(payload: Mapping[str, Any]) -> dict[str, object]:
    keys = (
        "score_claim",
        "ready_for_builder",
        "ready_for_exact_eval_dispatch",
        "dispatch_attempted",
        "remote_jobs_dispatched",
        "promotion_eligible",
        "rank_or_kill_eligible",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _source_member(source_archive: Path):
    try:
        return read_single_stored_member_archive(source_archive.read_bytes())
    except Exception:
        return None


def _source_payload_sha256(source_archive: Path) -> str | None:
    member = _source_member(source_archive)
    if member is None:
        return None
    return sha256_hex(member.payload)


def _source_packet_info(source_archive: Path) -> SourcePacketInfo | None:
    member = _source_member(source_archive)
    if member is None:
        return None
    format_id: int | None = None
    try:
        format_id = parse_pr106_sidecar_packet(member.payload).format_id
    except ValueError:
        format_id = None
    return SourcePacketInfo(
        member_name=member.name,
        payload_sha256=sha256_hex(member.payload),
        format_id=format_id,
    )


def _source_packet_format_label(info: SourcePacketInfo | None) -> str | None:
    if info is None or info.format_id is None:
        return None
    return f"0x{info.format_id:02X}"


def audit_score_table_manifest(
    *,
    source_archive: Path,
    score_table_npy: Path,
    score_table_manifest: Path,
) -> dict[str, object]:
    """Return deterministic custody audit metadata for the Kaggle table manifest.

    This audit is deliberately stricter than the builder's minimum contract:
    missing Kaggle custody fields are recorded as dispatch blockers even when
    the builder can still reduce the table into bytes.
    """

    manifest = _manifest_mapping(score_table_manifest)
    missing_fields = [
        field for field in SCORE_TABLE_MANIFEST_REQUIRED_FIELDS if field not in manifest
    ]
    warnings: list[str] = []
    blockers: list[str] = []
    if missing_fields:
        blockers.append("score_table_manifest_missing_custody_fields")
    if manifest.get("manifest_schema") != "pr106_latent_score_table_manifest_v1":
        blockers.append("score_table_manifest_schema_mismatch")
    if manifest.get("producer") != "experiments/build_pr106_latent_score_table.py":
        blockers.append("score_table_manifest_producer_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("score_table_manifest_score_claim_not_false")
    if manifest.get("ready_for_builder") is not True:
        blockers.append("score_table_manifest_ready_for_builder_not_true")
    if manifest.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("score_table_manifest_claims_exact_eval_dispatch_ready")
    if manifest.get("dispatch_attempted") is True or manifest.get("remote_jobs_dispatched") is True:
        blockers.append("score_table_manifest_claims_dispatch_attempted")
    if manifest.get("lane_claim_verified") is not True:
        blockers.append("score_table_manifest_lane_claim_not_verified")

    score_table_sha256 = sha256_file(score_table_npy)
    score_table_bytes = int(score_table_npy.stat().st_size)
    score_table_sha256_matches = manifest.get("score_table_npy_sha256") == score_table_sha256
    score_table_bytes_match = manifest.get("score_table_npy_bytes") == score_table_bytes
    if not score_table_sha256_matches:
        blockers.append("score_table_manifest_score_table_npy_sha256_mismatch_or_missing")
    if not score_table_bytes_match:
        blockers.append("score_table_manifest_score_table_npy_bytes_mismatch_or_missing")

    source_archive_sha256 = sha256_file(source_archive)
    source_archive_bytes = int(source_archive.stat().st_size)
    source_member = _source_member(source_archive)
    source_member_name = source_member.name if source_member is not None else None
    source_payload_sha256 = (
        sha256_hex(source_member.payload) if source_member is not None else None
    )
    source_archive_sha256_matches = manifest.get("source_archive_sha256") == source_archive_sha256
    source_archive_bytes_match = manifest.get("source_archive_bytes") == source_archive_bytes
    source_member_name_matches = manifest.get("source_archive_member_name") in (
        None,
        source_member_name,
    )
    source_info = _source_packet_info(source_archive)
    source_payload_sha256_matches = (
        source_payload_sha256 is not None
        and (
            manifest.get("source_archive_member_sha256") == source_payload_sha256
        )
    )
    legacy_zero_bin_fallback_allowed = (
        source_member_name == "0.bin"
        and (source_info is None or source_info.format_id in (None, PR106_SIDECAR_FORMAT_BROTLI))
    )
    source_zero_bin_sha256_matches = (
        source_payload_sha256 is not None
        and legacy_zero_bin_fallback_allowed
        and manifest.get("source_zero_bin_sha256") == source_payload_sha256
    )
    source_payload_identity_matches = (
        source_payload_sha256_matches or source_zero_bin_sha256_matches
    )
    format0c_requires_member_sha = (
        source_info is not None
        and source_info.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    )
    if format0c_requires_member_sha and not source_payload_sha256_matches:
        blockers.append("score_table_manifest_format0c_member_sha256_mismatch_or_missing")
    if not source_member_name_matches:
        blockers.append("score_table_manifest_source_archive_member_name_mismatch")
    if not source_archive_sha256_matches and not source_payload_identity_matches:
        blockers.append("score_table_manifest_source_archive_payload_mismatch_or_missing")
    elif not source_archive_sha256_matches and source_payload_identity_matches:
        warnings.append("source_archive_zip_sha256_differs_but_payload_sha256_matches")
    if not source_archive_bytes_match:
        warnings.append("source_archive_bytes_missing_or_not_matching_local_archive")

    if not isinstance(manifest.get("dispatch_blockers"), list):
        blockers.append("score_table_manifest_dispatch_blockers_missing_or_invalid")

    blockers = ordered_unique(blockers)
    warnings = ordered_unique(warnings)
    return {
        "schema": MANIFEST_AUDIT_SCHEMA,
        "subject": repo_relative(score_table_manifest, REPO_ROOT),
        "authority_flags": _authority_flags(manifest),
        "required_custody_fields": list(SCORE_TABLE_MANIFEST_REQUIRED_FIELDS),
        "missing_custody_fields": missing_fields,
        "score_table_npy": {
            "path": repo_relative(score_table_npy, REPO_ROOT),
            "bytes": score_table_bytes,
            "sha256": score_table_sha256,
            "manifest_bytes": manifest.get("score_table_npy_bytes"),
            "manifest_sha256": manifest.get("score_table_npy_sha256"),
            "bytes_match": score_table_bytes_match,
            "sha256_match": score_table_sha256_matches,
        },
        "source_archive": {
            "path": repo_relative(source_archive, REPO_ROOT),
            "bytes": source_archive_bytes,
            "sha256": source_archive_sha256,
            "single_member_name": source_member_name,
            "single_member_payload_sha256": source_payload_sha256,
            "manifest_bytes": manifest.get("source_archive_bytes"),
            "manifest_sha256": manifest.get("source_archive_sha256"),
            "manifest_member_name": manifest.get("source_archive_member_name"),
            "manifest_member_sha256": manifest.get("source_archive_member_sha256"),
            "manifest_zero_bin_sha256": manifest.get("source_zero_bin_sha256"),
            "source_sidecar_format_id": _source_packet_format_label(source_info),
            "bytes_match": source_archive_bytes_match,
            "archive_sha256_match": source_archive_sha256_matches,
            "single_member_name_match": source_member_name_matches,
            "single_member_payload_sha256_match": source_payload_identity_matches,
            "single_member_explicit_member_sha256_match": source_payload_sha256_matches,
            "legacy_zero_bin_sha256_fallback_match": source_zero_bin_sha256_matches,
        },
        "warnings": warnings,
        "blockers": blockers,
        "dispatch_ready": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def audit_builder_metadata(build_metadata: Mapping[str, Any]) -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    score_table_section = build_metadata.get("score_table")
    if build_metadata.get("score_claim") is not False:
        blockers.append("builder_metadata_score_claim_not_false")
    if build_metadata.get("ready_for_exact_eval_dispatch") is True:
        blockers.append("builder_metadata_claims_exact_eval_dispatch_ready")
    if build_metadata.get("search_mode") != "score_table":
        blockers.append("builder_metadata_search_mode_not_score_table")
    if not isinstance(score_table_section, Mapping):
        blockers.append("builder_metadata_score_table_section_missing")
    elif score_table_section.get("score_table_manifest_validated") is not True:
        blockers.append("builder_metadata_score_table_manifest_not_validated")
    if not isinstance(build_metadata.get("dispatch_blockers"), list):
        warnings.append("builder_metadata_dispatch_blockers_missing_or_invalid")
    return {
        "schema": BUILDER_METADATA_AUDIT_SCHEMA,
        "authority_flags": _authority_flags(build_metadata),
        "warnings": ordered_unique(warnings),
        "blockers": ordered_unique(blockers),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _materialize_format0c_native(
    *,
    source_archive: Path,
    output_dir: Path,
    score_table_npy: Path,
    score_table_manifest: Path,
    delta_radius: int,
    top_k: int,
) -> dict[str, object]:
    source_archive_bytes = source_archive.read_bytes()
    source_member = read_single_stored_member_archive(
        source_archive_bytes,
        expected_member_name="x",
    )
    source_packet = parse_pr106_sidecar_packet(source_member.payload)
    if (
        source_packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        raise RuntimeError(
            "native score-table materialization only supports format0C source "
            f"packets; got 0x{source_packet.format_id:02X}"
        )

    candidates = build_latent_candidate_grid(
        latent_dim=PR106_LATENT_N_DIMS,
        delta_radius=delta_radius,
    )
    candidate_grid_sha256 = latent_candidate_grid_npy_sha256(candidates)
    validated_manifest = validate_score_table_manifest(
        score_table_manifest,
        score_table_npy=score_table_npy,
        source_archive=source_archive,
        n_pairs=PR106_LATENT_N_PAIRS,
        latent_dim=PR106_LATENT_N_DIMS,
        delta_radius=delta_radius,
        candidate_count=int(candidates.shape[0]),
    )
    dim_arr, delta_q_arr, diagnostics = choose_latent_corrections_from_score_table_file(
        score_table_npy,
        n_pairs=PR106_LATENT_N_PAIRS,
        latent_dim=PR106_LATENT_N_DIMS,
        delta_radius=delta_radius,
        top_k=top_k,
    )
    candidate_member, candidate_archive, semantic_diagnostics = (
        emit_pr106_format0c_semantic_candidate_archive(
            source_member,
            source_packet,
            dim_arr,
            delta_q_arr,
        )
    )
    candidate_packet = parse_pr106_sidecar_packet(candidate_member.payload)
    sidecar_bin_path = output_dir / "sidecar.bin"
    archive_path = output_dir / "sidecar_archive.zip"
    metadata_path = output_dir / "build_metadata.json"
    sidecar_bin_path.write_bytes(candidate_packet.sidecar_payload)
    archive_path.write_bytes(candidate_archive)
    archive_size = archive_path.stat().st_size
    source_archive_size = source_archive.stat().st_size
    delta = archive_size - source_archive_size
    score_delta = 25.0 * delta / 37_545_489.0
    selected_dim_bytes = dim_arr.astype("uint8", copy=False).tobytes()
    selected_delta_bytes = delta_q_arr.astype("int8", copy=False).tobytes()
    metadata: dict[str, object] = {
        "lane_id": "lane_pr106_latent_sidecar",
        "materialization_engine": "format0c_packet_ir_native",
        "wall_clock_seconds": 0.0,
        "wall_clock_seconds_note": "deterministic_local_packet_ir_materialization",
        "device": "cpu",
        "smoke_mode": True,
        "search_mode": "score_table",
        "top_k": int(top_k),
        "source_archive": str(source_archive),
        "source_archive_bytes": source_archive_size,
        "source_archive_sha256": sha256_file(source_archive),
        "source_archive_member_name": source_member.name,
        "source_archive_member_sha256": sha256_hex(source_member.payload),
        "source_sidecar_format_id": f"0x{source_packet.format_id:02X}",
        "pr106_bin_bytes": len(source_packet.pr106_bytes),
        "pr106_bin_sha256": sha256_hex(source_packet.pr106_bytes),
        "sidecar_path": str(sidecar_bin_path),
        "sidecar_bytes": len(candidate_packet.sidecar_payload),
        "archive_path": str(archive_path),
        "archive_blob_bytes": len(candidate_member.payload),
        "archive_zip_bytes": archive_size,
        "delta_bytes_vs_source_archive": delta,
        "rate_component_score_delta_vs_source_archive": score_delta,
        "predicted_total_score_delta_vs_source": None,
        "predicted_total_score": None,
        "score_claim": False,
        "dispatch_attempted": False,
        "remote_jobs_dispatched": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "requires_paired_contest_cuda_auth_eval",
            "requires_paired_contest_cpu_auth_eval",
            "requires_runtime_decode_apply_proof_for_semantic_format0c_candidate",
            "requires_adjudicated_component_recompute_before_score_claim",
        ],
        "score_table": {
            "score_table_npy_path": str(score_table_npy),
            "score_table_npy_bytes": int(score_table_npy.stat().st_size),
            "score_table_npy_sha256": sha256_file(score_table_npy),
            "score_table_manifest_path": str(score_table_manifest),
            "score_table_manifest_sha256": sha256_file(score_table_manifest),
            "score_table_manifest_validated": True,
            "score_table_manifest_schema": validated_manifest.get("manifest_schema"),
            "validated_source_archive_sha256_match": validated_manifest.get(
                "validated_source_archive_sha256_match"
            ),
            "validated_source_archive_member_sha256_match": validated_manifest.get(
                "validated_source_archive_member_sha256_match"
            ),
            "validated_source_sidecar_format_id": validated_manifest.get(
                "validated_source_sidecar_format_id"
            ),
            "candidate_grid_sha256": candidate_grid_sha256,
            "selected_dim_sha256": hashlib.sha256(selected_dim_bytes).hexdigest(),
            "selected_delta_q_sha256": hashlib.sha256(selected_delta_bytes).hexdigest(),
            "score_table_is_score_claim": False,
            "score_table_required_provenance": (
                "CUDA scorer table must be generated against the exact source "
                "archive member; this native materializer only composes the "
                "measured table entries into PacketIR bytes."
            ),
        },
        "semantic_materialization": semantic_diagnostics,
        "packet_ir": {
            "source": pr106_sidecar_manifest(
                source_packet,
                archive_sha256=sha256_file(source_archive),
            ),
            "candidate": pr106_sidecar_manifest(
                candidate_packet,
                archive_sha256=sha256_file(archive_path),
            ),
            "candidate_consumed_byte_proof": pr106_sidecar_consumed_byte_proof(
                candidate_packet
            ),
        },
        "evidence_grade": "packet_ir_materialization_only",
        "diagnostics": diagnostics,
        "tag": "[packet-ir-local-no-score]",
        "next_step": (
            "Run runtime decode/apply smoke, then paired [contest-CUDA] and "
            "[contest-CPU] auth eval on sidecar_archive.zip before any score or "
            "promotion claim."
        ),
    }
    write_json(metadata_path, metadata)
    return metadata


def materialize_candidate(
    *,
    source_archive: Path,
    output_dir: Path,
    score_table_root: Path,
    score_table_npy: Path | None,
    score_table_manifest: Path | None,
    delta_radius: int,
    top_k: int,
    python_executable: str,
) -> dict[str, object]:
    if not source_archive.is_file():
        raise FileNotFoundError(f"source archive not found: {source_archive}")
    npy_path, manifest_path = resolve_score_table_artifacts(
        score_table_root=score_table_root,
        score_table_npy=score_table_npy,
        score_table_manifest=score_table_manifest,
    )
    score_table_manifest_audit = audit_score_table_manifest(
        source_archive=source_archive,
        score_table_npy=npy_path,
        score_table_manifest=manifest_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_info = _source_packet_info(source_archive)
    source_format_id = None if source_info is None else source_info.format_id
    command: list[str] = []
    builder_stdout = ""
    builder_stderr = ""
    materialization_engine = "legacy_builder_subprocess"

    if (
        source_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        materialization_engine = "format0c_packet_ir_native"
        build_metadata = _materialize_format0c_native(
            source_archive=source_archive,
            output_dir=output_dir,
            score_table_npy=npy_path,
            score_table_manifest=manifest_path,
            delta_radius=delta_radius,
            top_k=top_k,
        )
    else:
        if source_format_id not in (None, PR106_SIDECAR_FORMAT_BROTLI):
            raise RuntimeError(
                "refusing to route non-0x01 PR106 PacketIR through the legacy "
                "0.bin materializer: source format_id="
                f"{_source_packet_format_label(source_info)}"
            )
        command = [
            python_executable,
            str(BUILDER),
            "--source-archive",
            str(source_archive),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--smoke",
            "--search-mode",
            "score_table",
            "--score-table-npy",
            str(npy_path),
            "--score-table-manifest",
            str(manifest_path),
            "--delta-radius",
            str(delta_radius),
            "--top-k",
            str(top_k),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        builder_stdout = completed.stdout
        builder_stderr = completed.stderr
        archive_path = output_dir / "sidecar_archive.zip"
        metadata_path = output_dir / "build_metadata.json"
        if not archive_path.is_file():
            raise FileNotFoundError(f"builder did not emit expected archive: {archive_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"builder did not emit expected metadata: {metadata_path}")
        build_metadata = read_json(metadata_path)
    archive_path = output_dir / "sidecar_archive.zip"
    metadata_path = output_dir / "build_metadata.json"
    if not archive_path.is_file():
        raise FileNotFoundError(f"materializer did not emit expected archive: {archive_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"materializer did not emit expected metadata: {metadata_path}")
    if build_metadata.get("score_claim") is not False:
        raise RuntimeError("builder metadata must keep score_claim=false")
    if build_metadata.get("search_mode") != "score_table":
        raise RuntimeError("builder metadata search_mode must be score_table")
    if build_metadata.get("score_table", {}).get("score_table_manifest_validated") is not True:
        raise RuntimeError("builder did not validate the score-table manifest")
    builder_metadata_audit = audit_builder_metadata(build_metadata)
    audit_blockers = ordered_unique(
        [
            *MATERIALIZATION_DISPATCH_BLOCKERS,
            *[str(item) for item in score_table_manifest_audit["blockers"]],
            *[str(item) for item in builder_metadata_audit["blockers"]],
        ]
    )
    audit_warnings = ordered_unique(
        [
            *[str(item) for item in score_table_manifest_audit["warnings"]],
            *[str(item) for item in builder_metadata_audit["warnings"]],
        ]
    )

    materialization = apply_proxy_evidence_boundary({
        "schema": "pr106_latent_score_table_candidate_materialization_v1",
        "lane_id": "lane_pr106_latent_sidecar",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotion_requires": (
            "paired_contest_cuda_and_contest_cpu_adjudication_on_materialized_archive"
        ),
        "score_claim_blockers": audit_blockers,
        "exact_eval_dispatch_blockers": audit_blockers,
        "audit_warnings": audit_warnings,
        "score_table_manifest_audit": score_table_manifest_audit,
        "builder_metadata_audit": builder_metadata_audit,
        "source_archive": _artifact(source_archive),
        "score_table_npy": _artifact(npy_path),
        "score_table_manifest": _artifact(manifest_path),
        "delta_radius": int(delta_radius),
        "top_k": int(top_k),
        "builder": {
            "path": repo_relative(BUILDER, REPO_ROOT),
            "command": command,
            "stdout": builder_stdout,
            "stderr": builder_stderr,
            "materialization_engine": materialization_engine,
            "source_packet_format_id": _source_packet_format_label(source_info),
        },
        "outputs": {
            "archive": _artifact(archive_path),
            "build_metadata": _artifact(metadata_path),
            "materialization_manifest": {
                "path": repo_relative(output_dir / "materialization_manifest.json", REPO_ROOT)
            },
        },
        "next_step": (
            "Claim lane_pr106_latent_sidecar, run paired [contest-CUDA] and "
            "[contest-CPU] auth eval on outputs.archive, then adjudicate component "
            "fields before any score claim."
        ),
    }, dispatch_blockers=audit_blockers)
    proxy_violations = validate_proxy_candidate(materialization)
    if proxy_violations:
        raise RuntimeError(
            "materialization manifest leaked proxy evidence authority: "
            + ", ".join(proxy_violations)
        )
    manifest_out = output_dir / "materialization_manifest.json"
    write_json(manifest_out, materialization)
    materialization["outputs"]["materialization_manifest"] = _artifact(manifest_out)
    write_json(manifest_out, materialization)
    return materialization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--score-table-root", type=Path, default=DEFAULT_SCORE_TABLE_ROOT)
    parser.add_argument("--score-table-npy", type=Path, default=None)
    parser.add_argument("--score-table-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--delta-radius", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=600)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / default_run_id())
    manifest = materialize_candidate(
        source_archive=args.source_archive,
        output_dir=output_dir,
        score_table_root=args.score_table_root,
        score_table_npy=args.score_table_npy,
        score_table_manifest=args.score_table_manifest,
        delta_radius=args.delta_radius,
        top_k=args.top_k,
        python_executable=args.python_executable,
    )
    print(json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
