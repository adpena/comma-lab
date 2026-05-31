#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# DETERMINISTIC_COMPILER_OK:non-promoting-research-or-probe-builder-emits-non-score-claiming-archive-FIX-WAVE-R1-META-1-closure
"""Build a local A5 PR101 runtime-consumption packet scaffold.

This tool is local-only. It does not run inflate end-to-end, does not invoke
the scorer, does not claim a dispatch lane, and does not promote score. It
materializes a packet-local copy of the PR101 runtime, patches only that copy
for the ``tac_frame_conditional_latent_wire.v1`` A5 wire contract, builds a
deterministic single-member archive, and emits the three local artifacts that
the A5 readiness planner can validate:

* candidate archive manifest
* packet-local runtime patch manifest
* runtime-consumption proof
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import lzma
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis import hnerv_packet_sections  # noqa: E402
from tac.codec.frame_conditional_bit_budget import (  # noqa: E402
    FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
    FRAME_CONDITIONAL_Q_BITS_ENCODING_BINARY_LOW_HIGH_MASK,
    FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3,
    FRAME_CONDITIONAL_Q_BITS_ENCODING_RAW3,
    allocate_per_frame_bits,
    build_frame_conditional_channel_wire_contract,
    build_frame_conditional_wire_contract,
    pack_frame_conditional_channel_latent_codes,
    pack_frame_conditional_channel_q_bits,
    pack_frame_conditional_latent_codes,
    pack_frame_conditional_q_bits_by_encoding,
)
from tac.optimization.archive_bound_candidate_contract import (  # noqa: E402
    archive_bound_candidate_contract_fields_for_row,
)
from tac.repo_io import (  # noqa: E402
    json_text,
    read_json,
    repo_relative,
    sha256_bytes,
    sha256_file,
    write_json,
)

TOOL_NAME = "tools/build_pr101_frame_conditional_runtime_packet.py"
SCHEMA_VERSION = "pr101_frame_conditional_runtime_packet.v1"
RUNTIME_PATCH_SCHEMA = "pr101_frame_conditional_packet_runtime_patch_manifest.v1"
RUNTIME_PROOF_SCHEMA = "pr101_frame_conditional_runtime_consumption_proof.v1"
A5_ANCHOR_SCHEMA = "pr101_frame_conditional_bit_anchor.v1"
A5_SCORE_MARGINAL_SCHEMA = "pr101_a5_per_pair_score_marginals.v1"
A5_SCORE_MARGINAL_QBITS_SCHEDULE_SCHEMA = "pr101_a5_score_marginal_qbits_schedule.v1"
Q_BITS_OVERRIDE_KEYS = ("per_pair_q_bits", "q_bits_per_pair", "q_bits")
CHANNEL_Q_BITS_OVERRIDE_KEYS = ("per_channel_q_bits", "q_bits_per_channel")

A5_MAGIC = b"A5FC"
A5_HEADER_LEN = 20
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_N_PAIRS = 600
PR101_LATENT_DIM = 28
PR101_LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]

DEFAULT_A5_MANIFEST = Path(
    "experiments/results/pr101_frame_conditional_bit_codex_20260508T_wire_contract_smoke/build_manifest.json"
)
DEFAULT_SOURCE_ARCHIVE = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_SOURCE_RUNTIME_DIR = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/"
    "source/submissions/hnerv_ft_microcodec"
)
DEFAULT_VIDEO_PATH = Path("upstream/videos/0.mkv")

EXCLUDED_DIR_NAMES = frozenset({"__pycache__", ".git", ".mypy_cache", ".pytest_cache"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = (".pyc", ".pyo")
PACKET_CUSTODY_FILENAMES = frozenset(
    {
        "archive.zip",
        "candidate_archive_manifest.json",
        "packet_runtime_patch_manifest.json",
        "runtime_consumption_proof.json",
        "readiness.with_runtime_packet.json",
        "report.txt",
    }
)
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class FrameConditionalRuntimePacketError(ValueError):
    """Raised when an A5 runtime packet scaffold cannot be built safely."""


def build_frame_conditional_runtime_packet(
    *,
    a5_manifest_path: Path,
    source_archive: Path,
    source_runtime_dir: Path,
    output_dir: Path,
    video_path: Path | None = None,
    candidate_id: str = "pr101_a5_frame_conditional_runtime_packet",
    force: bool = False,
    q_bits_override: Sequence[float] | np.ndarray | None = None,
    q_bits_override_source_path: Path | None = None,
    channel_q_bits_override: Sequence[float] | np.ndarray | None = None,
    channel_q_bits_override_source_path: Path | None = None,
    q_bits_sideinfo_encoding: str = FRAME_CONDITIONAL_Q_BITS_ENCODING_RAW3,
    allow_q_bits_wire_contract_override: bool = False,
    recorded_at_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic local A5 packet scaffold and custody manifests."""

    recorded_at_utc = (recorded_at_utc or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    a5_manifest_path = _repo_path(a5_manifest_path)
    source_archive = _repo_path(source_archive)
    source_runtime_dir = _repo_path(source_runtime_dir)
    output_dir = _repo_path(output_dir)
    video_path = _repo_path(video_path) if video_path is not None else None
    q_bits_override_source_path = (
        _repo_path(q_bits_override_source_path)
        if q_bits_override_source_path is not None
        else None
    )
    channel_q_bits_override_source_path = (
        _repo_path(channel_q_bits_override_source_path)
        if channel_q_bits_override_source_path is not None
        else None
    )

    _require_file(a5_manifest_path, "a5_manifest")
    _require_file(source_archive, "source_archive")
    if q_bits_override_source_path is not None:
        _require_file(q_bits_override_source_path, "q_bits_override_source")
        if q_bits_override is None:
            raise FrameConditionalRuntimePacketError(
                "q_bits_override_source_path requires q_bits_override"
            )
    if channel_q_bits_override_source_path is not None:
        _require_file(
            channel_q_bits_override_source_path,
            "channel_q_bits_override_source",
        )
        if channel_q_bits_override is None:
            raise FrameConditionalRuntimePacketError(
                "channel_q_bits_override_source_path requires channel_q_bits_override"
            )
    if q_bits_override is not None and channel_q_bits_override is not None:
        raise FrameConditionalRuntimePacketError(
            "per-pair and per-channel q-bit overrides are mutually exclusive"
        )
    if (
        q_bits_sideinfo_encoding == FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3
        and channel_q_bits_override is None
    ):
        raise FrameConditionalRuntimePacketError(
            "channel_raw3 q-bit side-info requires --channel-q-bits-json"
        )
    if (
        q_bits_sideinfo_encoding != FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3
        and channel_q_bits_override is not None
    ):
        raise FrameConditionalRuntimePacketError(
            "--channel-q-bits-json requires --q-bits-sideinfo-encoding channel_raw3"
        )
    if not source_runtime_dir.is_dir():
        raise FrameConditionalRuntimePacketError(
            f"source runtime directory not found: {source_runtime_dir}"
        )
    inputs = [a5_manifest_path, source_archive, source_runtime_dir]
    if q_bits_override_source_path is not None:
        inputs.append(q_bits_override_source_path)
    if channel_q_bits_override_source_path is not None:
        inputs.append(channel_q_bits_override_source_path)
    _assert_output_dir_isolated(
        output_dir=output_dir,
        inputs=inputs,
    )
    _prepare_empty_dir(output_dir, force=force)

    a5_manifest = _validate_a5_manifest(read_json(a5_manifest_path))
    best_row = _best_a5_row(a5_manifest)
    wire_contract = _require_mapping(
        best_row.get("frame_conditional_wire_contract"),
        "best_row.frame_conditional_wire_contract",
    )
    source_archive_record, source_member_name, source_member_payload = _read_single_member_zip(
        source_archive
    )
    source_decoder_blob, source_latent_blob, source_sidecar_blob = _split_pr101_member(
        source_member_payload
    )
    latent_meta_blob, q_pair_first = _extract_storage_ordered_latent_payload(
        source_latent_blob
    )
    q_bits_axis = (
        "channel"
        if q_bits_sideinfo_encoding == FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3
        else "pair"
    )
    if q_bits_axis == "channel":
        q_bits = _materialize_channel_q_bits(
            channel_q_bits_override=channel_q_bits_override,
            latent_dim=q_pair_first.shape[1],
        )
        q_bits_sideinfo = pack_frame_conditional_channel_q_bits(q_bits)
        latent_wire_payload = pack_frame_conditional_channel_latent_codes(
            q_pair_first,
            q_bits,
        )
        materialized_wire_contract = build_frame_conditional_channel_wire_contract(
            q_bits,
            n_pairs=q_pair_first.shape[0],
            q_pair_first=q_pair_first,
        )
    else:
        q_bits = _materialize_q_bits(
            a5_manifest=a5_manifest,
            best_row=best_row,
            video_path=video_path,
            q_bits_override=q_bits_override,
        )
        q_bits_sideinfo = pack_frame_conditional_q_bits_by_encoding(
            q_bits,
            encoding=q_bits_sideinfo_encoding,
        )
        latent_wire_payload = pack_frame_conditional_latent_codes(q_pair_first, q_bits)
        materialized_wire_contract = build_frame_conditional_wire_contract(
            q_bits,
            latent_dim=q_pair_first.shape[1],
            q_pair_first=q_pair_first,
            q_bits_sideinfo_encoding=q_bits_sideinfo_encoding,
        )
    wire_contract_reconciliation = _reconcile_wire_contract(
        source_wire_contract=wire_contract,
        materialized_wire_contract=materialized_wire_contract,
        q_bits_sideinfo=q_bits_sideinfo,
        latent_wire_payload=latent_wire_payload,
        allow_q_bits_wire_contract_override=allow_q_bits_wire_contract_override,
    )
    wire_contract = wire_contract_reconciliation["selected_wire_contract"]
    q_bits_schedule = _q_bits_schedule_record(
        q_bits=q_bits,
        q_bits_sideinfo=q_bits_sideinfo,
        q_bits_sideinfo_encoding=q_bits_sideinfo_encoding,
        q_bits_axis=q_bits_axis,
        q_bits_override=q_bits_override,
        q_bits_override_source_path=q_bits_override_source_path,
        channel_q_bits_override=channel_q_bits_override,
        channel_q_bits_override_source_path=channel_q_bits_override_source_path,
        a5_manifest=a5_manifest,
        best_row=best_row,
        video_path=video_path,
    )

    packet_dir = output_dir / "packet"
    packet_dir.mkdir(parents=True)
    runtime_files_before_patch = _copy_runtime_tree(source_runtime_dir, packet_dir)
    inflate_patch = _patch_packet_inflate_sh(packet_dir)
    runtime_patch = _patch_packet_codec(packet_dir)

    candidate_member_payload = _build_a5_member_payload(
        decoder_blob=source_decoder_blob,
        latent_meta_blob=latent_meta_blob,
        q_bits_sideinfo=q_bits_sideinfo,
        latent_wire_payload=latent_wire_payload,
        sidecar_blob=source_sidecar_blob,
    )
    candidate_archive_path = packet_dir / "archive.zip"
    candidate_archive_record = _write_single_member_zip(
        candidate_archive_path,
        source_member_name,
        candidate_member_payload,
    )
    parser_section_custody = _parser_section_custody_for_candidate(
        candidate_archive_path=candidate_archive_path,
        candidate_id=candidate_id,
    )

    source_parse_smoke = _packet_local_parse_smoke(
        packet_dir,
        source_archive,
        label="source_pr101_stock_fallback",
    )
    candidate_parse_smoke = _packet_local_parse_smoke(
        packet_dir,
        candidate_archive_path,
        label="candidate_a5fc_frame_conditional",
    )
    runtime_consumes_changed_archive_bytes = bool(
        source_parse_smoke.get("passed")
        and candidate_parse_smoke.get("passed")
        and candidate_parse_smoke.get("a5_magic") is True
        and candidate_parse_smoke.get("consumed_q_bits_sideinfo_sha256")
        == sha256_bytes(q_bits_sideinfo)
        and candidate_parse_smoke.get("consumed_latent_wire_payload_sha256")
        == sha256_bytes(latent_wire_payload)
        and candidate_parse_smoke.get("latent_wire_mutation_changed_latents") is True
        and candidate_parse_smoke.get("zero_sideinfo_negative_control_rejected") is True
        and source_archive_record["members"][0]["sha256"]
        != candidate_archive_record["members"][0]["sha256"]
    )

    runtime_files_after_patch = [
        _file_record(path, relpath=path.relative_to(packet_dir).as_posix())
        for path in sorted(
            packet_dir.rglob("*"), key=lambda item: item.relative_to(packet_dir).as_posix()
        )
        if path.is_file()
        and not _is_packet_custody_file(path.relative_to(packet_dir))
        and not _should_exclude(path.relative_to(packet_dir))
    ]
    runtime_tree_sha256 = _runtime_tree_sha256(runtime_files_after_patch)

    dispatch_blockers = [
        "strict_pre_submission_compliance_json_missing",
        "no_exact_cuda_auth_eval",
        "no_contest_cpu_auth_eval",
        "requires_level2_dispatch_claim_before_exact_eval",
        "operator_score_claim_review_not_done",
    ]
    if not q_bits_schedule["score_marginal_manifest_consumed"]:
        dispatch_blockers.insert(0, "per_pair_score_marginal_manifest_missing")
    cleared_blockers = [
        "candidate_archive_manifest",
        "packet_local_runtime_patch_manifest",
        "frame_conditional_runtime_consumption_proof",
    ]
    if q_bits_schedule["score_marginal_manifest_consumed"]:
        cleared_blockers.append("per_pair_score_marginal_manifest_consumed")
    report_record = _write_packet_report(
        packet_dir=packet_dir,
        candidate_id=candidate_id,
        candidate_archive_record=candidate_archive_record,
        source_archive_record=source_archive_record,
        runtime_tree_sha256=runtime_tree_sha256,
        dispatch_blockers=dispatch_blockers,
    )

    patch_manifest = _build_runtime_patch_manifest(
        packet_dir=packet_dir,
        runtime_patch=runtime_patch,
        runtime_tree_sha256=runtime_tree_sha256,
        runtime_files=runtime_files_after_patch,
    )
    patch_manifest_path = output_dir / "packet_runtime_patch_manifest.json"
    proof_manifest_path = output_dir / "runtime_consumption_proof.json"
    proof_manifest = _build_runtime_consumption_proof(
        candidate_id=candidate_id,
        candidate_archive_record=candidate_archive_record,
        candidate_member_payload=candidate_member_payload,
        q_bits_sideinfo=q_bits_sideinfo,
        latent_wire_payload=latent_wire_payload,
        source_parse_smoke=source_parse_smoke,
        candidate_parse_smoke=candidate_parse_smoke,
        runtime_consumes_changed_archive_bytes=runtime_consumes_changed_archive_bytes,
        runtime_consumption_proof_path=proof_manifest_path,
    )
    write_json(patch_manifest_path, patch_manifest)
    write_json(proof_manifest_path, proof_manifest)

    archive_member_manifest = _archive_member_manifest(
        source_decoder_blob=source_decoder_blob,
        source_latent_blob=source_latent_blob,
        source_sidecar_blob=source_sidecar_blob,
        latent_meta_blob=latent_meta_blob,
        q_bits_sideinfo=q_bits_sideinfo,
        latent_wire_payload=latent_wire_payload,
        candidate_member_payload=candidate_member_payload,
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "candidate_id": candidate_id,
        "created_utc": _format_utc(recorded_at_utc),
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "remote_gpu_run": False,
        "gpu_required": False,
        "evidence_grade": "empirical",
        "evidence_semantics": "local_packet_runtime_consumption_scaffold_no_score",
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "t4_contest_runtime_pending_score_marginals_and_exact_eval",
        "source_runtime_mutated": False,
        "charged_bits_changed": True,
        "score_affecting_payload_changed": True,
        "semantic_payload_changed": True,
        "dispatch_blockers": dispatch_blockers,
        "cleared_local_readiness_artifacts": cleared_blockers,
        "a5_manifest": {
            "path": _repo_rel(a5_manifest_path),
            "bytes": a5_manifest_path.stat().st_size,
            "sha256": sha256_file(a5_manifest_path),
            "schema": a5_manifest.get("schema"),
            "best_eta": best_row.get("eta"),
            "best_archive_delta_bytes": best_row.get("archive_delta_bytes"),
            "wire_schema": wire_contract.get("schema"),
            "score_claim": False,
        },
        "q_bits_schedule": q_bits_schedule,
        "source_archive": source_archive_record,
        "candidate_archive": candidate_archive_record,
        "candidate_archive_relpath": _repo_rel(candidate_archive_path),
        "packet_runtime_patch_manifest": _artifact_ref(patch_manifest_path),
        "runtime_consumption_proof": _artifact_ref(proof_manifest_path),
        "packet_local_runtime_patch": patch_manifest["packet_local_runtime_patch"],
        "packet_closure": {
            "byte_closed_packet_built": True,
            "runtime_consumes_changed_archive_bytes": runtime_consumes_changed_archive_bytes,
            "runtime_source_mutated": False,
            "wire_contract": FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
            "layout_magic": A5_MAGIC.decode("ascii"),
            "missing_wire_contracts": [],
            "parse_smoke_only": True,
            "inflate_parity_status": "not_run",
            "score_eval_status": "not_run",
        },
        "frame_conditional_wire_contract": wire_contract,
        "wire_contract_reconciliation": wire_contract_reconciliation[
            "manifest_record"
        ],
        "archive_member_manifest": archive_member_manifest,
        "parser_section_manifest": parser_section_custody["parser_section_manifest"],
        "parser_section_gate": parser_section_custody["parser_section_gate"],
        "parser_section_custody": parser_section_custody,
        "runtime_packet": {
            "packet_dir": _repo_rel(packet_dir),
            "inflate_patch": inflate_patch,
            "runtime_patch": runtime_patch,
            "runtime_custody": {
                "copied_file_count": len(runtime_files_before_patch),
                "runtime_tree_sha256": runtime_tree_sha256,
                "runtime_files": runtime_files_after_patch,
                "excluded_packet_custody_filenames": sorted(PACKET_CUSTODY_FILENAMES),
            },
            "runtime_checks": {
                "inflate_sh_bash_n": _bash_n(packet_dir / "inflate.sh"),
                "source_packet_local_parse_smoke": source_parse_smoke,
                "candidate_packet_local_parse_smoke": candidate_parse_smoke,
                "packet_local_parse_smoke": candidate_parse_smoke,
            },
            "report": report_record,
        },
        "next_required_actions": [
            (
                "produce per-pair score-marginal evidence for the A5 q-bit schedule"
                if not q_bits_schedule["score_marginal_manifest_consumed"]
                else "review consumed per-pair score-marginal schedule before exact eval"
            ),
            "run strict pre-submission compliance on reviewed packet surface",
            "claim Level-2 lane before any exact eval dispatch",
            "run exact CUDA auth eval before any score or promotion claim",
        ],
    }
    manifest.update(
        archive_bound_candidate_contract_fields_for_row(
            {
                "archive_native_transform_kind": SCHEMA_VERSION,
                "target_kind": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "candidate_archive": candidate_archive_record,
                "source_archive": source_archive_record,
                "byte_closed_candidate_emitted": True,
                "byte_closed_candidate_materialized": True,
                "candidate_archive_materialized": True,
                "runtime_consumption_proof_ready": proof_manifest[
                    "runtime_consumption_proven"
                ],
                "runtime_consumption_proof_path": _repo_rel(proof_manifest_path),
                "receiver_contract_kind": RUNTIME_PROOF_SCHEMA,
                "receiver_contract_satisfied": proof_manifest[
                    "runtime_consumption_proven"
                ],
                "runtime_adapter_ready": True,
                "runtime_adapter_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_tree_sha256": runtime_tree_sha256,
                    "packet_runtime_patch_manifest": _repo_rel(patch_manifest_path),
                },
                "semantic_payload_changed": True,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "readiness_blockers": proof_manifest["blockers"],
                "dispatch_blockers": dispatch_blockers,
                **FALSE_AUTHORITY_FIELDS,
            },
            repo_root=REPO_ROOT,
            selected_transform_kind=SCHEMA_VERSION,
            family_id="pr101_frame_conditional_runtime_packet",
            candidate_chain_id=candidate_id,
        )
    )
    manifest["manifest_sha256_excluding_self"] = _canonical_json_sha256(manifest)
    manifest_path = output_dir / "candidate_archive_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def _validate_a5_manifest(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FrameConditionalRuntimePacketError("A5 manifest must be a JSON object")
    if payload.get("schema") != A5_ANCHOR_SCHEMA:
        raise FrameConditionalRuntimePacketError(
            f"A5 manifest schema must be {A5_ANCHOR_SCHEMA}"
        )
    if payload.get("score_claim") is not False:
        raise FrameConditionalRuntimePacketError("A5 manifest must not claim score")
    if payload.get("byte_proxy_only") is not True:
        raise FrameConditionalRuntimePacketError("A5 manifest must be byte_proxy_only")
    if payload.get("ready_for_exact_eval_dispatch") is True:
        raise FrameConditionalRuntimePacketError(
            "A5 manifest must not claim exact-eval readiness"
        )
    return payload


def _best_a5_row(a5_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = a5_manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise FrameConditionalRuntimePacketError("A5 manifest rows missing")
    candidates = [row for row in rows if isinstance(row, Mapping)]
    candidates = [
        row
        for row in candidates
        if isinstance(row.get("frame_conditional_wire_contract"), Mapping)
        and row["frame_conditional_wire_contract"].get("schema")
        == FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA
        and row["frame_conditional_wire_contract"].get("decoder_helper_consumes_sideinfo_bytes")
        is True
    ]
    if not candidates:
        raise FrameConditionalRuntimePacketError(
            "A5 manifest has no side-info-consuming frame-conditional row"
        )
    return min(candidates, key=lambda row: float(row.get("archive_delta_bytes", 0)))


def _materialize_q_bits(
    *,
    a5_manifest: Mapping[str, Any],
    best_row: Mapping[str, Any],
    video_path: Path | None,
    q_bits_override: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    if q_bits_override is not None:
        raw_q_bits = np.asarray(q_bits_override, dtype=np.float64)
        if not np.isfinite(raw_q_bits).all():
            raise FrameConditionalRuntimePacketError("q_bits override must be finite")
        q_bits = np.floor(raw_q_bits).astype(np.uint8)
    else:
        if video_path is None:
            video = _repo_path(Path(str(a5_manifest.get("input_video") or DEFAULT_VIDEO_PATH)))
        else:
            video = video_path
        _require_file(video, "video_path")
        from tac.codec.frame_conditional_bit_budget import compute_per_frame_complexity

        n_pairs = _positive_int(a5_manifest.get("n_pairs"), "n_pairs")
        latent_dim = _positive_int(a5_manifest.get("latent_dim"), "latent_dim")
        per_frame = compute_per_frame_complexity(video, n_pairs * 2)
        pair_complexity = per_frame.reshape(-1, 2).mean(axis=1)
        bits_per_pair = allocate_per_frame_bits(
            pair_complexity,
            float(a5_manifest.get("total_bit_budget")),
            eta=float(best_row.get("eta")),
            floor=float(best_row.get("floor")),
            cap=float(best_row.get("cap")),
        )
        q_bits = np.floor(np.clip(bits_per_pair / latent_dim, 1.0, 8.0)).astype(np.uint8)
    if q_bits.ndim != 1:
        raise FrameConditionalRuntimePacketError(f"q_bits must be 1-D, got {q_bits.shape}")
    expected_pairs = _positive_int(a5_manifest.get("n_pairs"), "n_pairs")
    if q_bits.size != expected_pairs:
        raise FrameConditionalRuntimePacketError(
            f"q_bits length {q_bits.size} != expected pairs {expected_pairs}"
        )
    if (q_bits < 1).any() or (q_bits > 8).any():
        raise FrameConditionalRuntimePacketError("q_bits must be in [1, 8]")
    return q_bits


def _materialize_channel_q_bits(
    *,
    channel_q_bits_override: Sequence[float] | np.ndarray | None,
    latent_dim: int,
) -> np.ndarray:
    if channel_q_bits_override is None:
        raise FrameConditionalRuntimePacketError(
            "per-channel q-bit schedule requires an explicit override"
        )
    q_bits = _normalise_q_bits_override_values(channel_q_bits_override)
    if q_bits.ndim != 1:
        raise FrameConditionalRuntimePacketError(
            f"channel q_bits must be 1-D, got {q_bits.shape}"
        )
    if q_bits.size != latent_dim:
        raise FrameConditionalRuntimePacketError(
            f"channel q_bits length {q_bits.size} != latent_dim {latent_dim}"
        )
    return q_bits


def _q_bits_schedule_record(
    *,
    q_bits: np.ndarray,
    q_bits_sideinfo: bytes,
    q_bits_sideinfo_encoding: str,
    q_bits_axis: str,
    q_bits_override: Sequence[float] | np.ndarray | None,
    q_bits_override_source_path: Path | None,
    channel_q_bits_override: Sequence[float] | np.ndarray | None,
    channel_q_bits_override_source_path: Path | None,
    a5_manifest: Mapping[str, Any],
    best_row: Mapping[str, Any],
    video_path: Path | None,
) -> dict[str, Any]:
    q_bits = np.asarray(q_bits, dtype=np.uint8)
    if q_bits_axis not in {"pair", "channel"}:
        raise FrameConditionalRuntimePacketError(f"unknown q_bits_axis={q_bits_axis!r}")
    unique_values, unique_counts = np.unique(q_bits, return_counts=True)
    record: dict[str, Any] = {
        "schema": "pr101_frame_conditional_q_bits_schedule.v1",
        "q_bits_axis": q_bits_axis,
        "q_bits_count": int(q_bits.size),
        "q_bits_min": int(q_bits.min()),
        "q_bits_max": int(q_bits.max()),
        "q_bits_mean": float(np.mean(q_bits)),
        "q_bits_unique_counts": {
            str(int(value)): int(count)
            for value, count in zip(unique_values, unique_counts, strict=True)
        },
        "q_bits_sha256": sha256_bytes(q_bits.tobytes()),
        "q_bits_sideinfo_encoding": q_bits_sideinfo_encoding,
        "q_bits_sideinfo_bytes": len(q_bits_sideinfo),
        "q_bits_sideinfo_sha256": sha256_bytes(q_bits_sideinfo),
        "score_marginal_manifest_consumed": False,
    }
    if q_bits_axis == "channel":
        record["schedule_source"] = "channel_json_override"
        if channel_q_bits_override_source_path is None:
            raise FrameConditionalRuntimePacketError(
                "channel q-bit override requires source path for custody"
            )
        source_q_bits, source_info = _load_q_bits_override_json_with_info(
            channel_q_bits_override_source_path,
            expected_axis="channel",
        )
        if sha256_bytes(source_q_bits.tobytes()) != record["q_bits_sha256"]:
            raise FrameConditionalRuntimePacketError(
                "channel q_bits override source does not match materialized q_bits"
            )
        record.update(
            {
                "source_key": source_info["key"],
                "source_schema": source_info.get("schema"),
                "source_artifact": _artifact_ref(channel_q_bits_override_source_path),
                "score_marginal_manifest_consumed": False,
            }
        )
    elif q_bits_override is not None:
        record["schedule_source"] = "api_override"
        if q_bits_override_source_path is not None:
            source_q_bits, source_info = _load_q_bits_override_json_with_info(
                q_bits_override_source_path,
                expected_axis="pair",
            )
            if sha256_bytes(source_q_bits.tobytes()) != record["q_bits_sha256"]:
                raise FrameConditionalRuntimePacketError(
                    "q_bits override source does not match materialized q_bits"
                )
            source_schema = source_info.get("schema")
            record.update(
                {
                    "schedule_source": "json_override",
                    "source_key": source_info["key"],
                    "source_schema": source_schema,
                    "source_artifact": _artifact_ref(q_bits_override_source_path),
                    "score_marginal_manifest_consumed": (
                        source_schema
                        in {
                            A5_SCORE_MARGINAL_SCHEMA,
                            A5_SCORE_MARGINAL_QBITS_SCHEDULE_SCHEMA,
                        }
                        and source_info["key"] == "per_pair_q_bits"
                    ),
                }
            )
    else:
        record["schedule_source"] = "video_complexity_allocator"
        record["allocator"] = {
            "video_path": _repo_rel(
                video_path
                if video_path is not None
                else _repo_path(Path(str(a5_manifest.get("input_video") or DEFAULT_VIDEO_PATH)))
            ),
            "eta": float(best_row.get("eta")),
            "floor": float(best_row.get("floor")),
            "cap": float(best_row.get("cap")),
            "total_bit_budget": float(a5_manifest.get("total_bit_budget")),
        }
    return record


def _load_q_bits_override_json(path: Path) -> np.ndarray:
    q_bits, _info = _load_q_bits_override_json_with_info(path, expected_axis="pair")
    return q_bits


def _load_channel_q_bits_override_json(path: Path) -> np.ndarray:
    q_bits, _info = _load_q_bits_override_json_with_info(path, expected_axis="channel")
    return q_bits


def _load_q_bits_override_json_with_info(
    path: Path,
    *,
    expected_axis: str = "pair",
) -> tuple[np.ndarray, dict[str, Any]]:
    path = _repo_path(path)
    payload = read_json(path)
    values, info = _extract_q_bits_override_values(payload, expected_axis=expected_axis)
    q_bits = _normalise_q_bits_override_values(values)
    info["axis"] = expected_axis
    info["path"] = _repo_rel(path)
    info["q_bits_count"] = int(q_bits.size)
    info["q_bits_sha256"] = sha256_bytes(q_bits.tobytes())
    return q_bits, info


def _extract_q_bits_override_values(
    payload: Any,
    *,
    expected_axis: str = "pair",
) -> tuple[Any, dict[str, Any]]:
    if expected_axis not in {"pair", "channel"}:
        raise FrameConditionalRuntimePacketError(
            f"expected_axis must be pair or channel, got {expected_axis!r}"
        )
    allowed_keys = (
        Q_BITS_OVERRIDE_KEYS
        if expected_axis == "pair"
        else CHANNEL_Q_BITS_OVERRIDE_KEYS
    )
    if isinstance(payload, Mapping):
        present = [key for key in allowed_keys if key in payload]
        forbidden_keys = [
            key
            for key in (
                CHANNEL_Q_BITS_OVERRIDE_KEYS
                if expected_axis == "pair"
                else Q_BITS_OVERRIDE_KEYS
            )
            if key in payload
        ]
        if forbidden_keys:
            raise FrameConditionalRuntimePacketError(
                f"{expected_axis} q-bits JSON contains wrong-axis key(s): "
                + ", ".join(forbidden_keys)
            )
        if len(present) != 1:
            raise FrameConditionalRuntimePacketError(
                "q-bits JSON object must contain exactly one of "
                + ", ".join(allowed_keys)
            )
        key = present[0]
        return payload[key], {"key": key, "schema": payload.get("schema")}
    if isinstance(payload, list):
        return payload, {"key": "$root", "schema": None}
    raise FrameConditionalRuntimePacketError(
        "q-bits JSON must be an object or an array of integer q-bit widths"
    )


def _normalise_q_bits_override_values(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (bytes, str)):
        raise FrameConditionalRuntimePacketError("q-bits override must be a JSON array")
    out: list[int] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FrameConditionalRuntimePacketError(
                f"q-bits override value at index {idx} must be numeric"
            )
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise FrameConditionalRuntimePacketError(
                f"q-bits override value at index {idx} must be an integer"
            )
        integer = int(numeric)
        if integer < 1 or integer > 8:
            raise FrameConditionalRuntimePacketError(
                f"q-bits override value at index {idx} must be in [1, 8]"
            )
        out.append(integer)
    if not out:
        raise FrameConditionalRuntimePacketError("q-bits override must be non-empty")
    return np.asarray(out, dtype=np.uint8)


def _assert_wire_contract_matches_materialized_payloads(
    *,
    wire_contract: Mapping[str, Any],
    q_bits_sideinfo: bytes,
    latent_wire_payload: bytes,
) -> None:
    sideinfo = _require_mapping(wire_contract.get("q_bits_sideinfo"), "q_bits_sideinfo")
    latent = _require_mapping(wire_contract.get("latent_wire_payload"), "latent_wire_payload")
    checks = [
        ("q_bits_sideinfo.bytes", sideinfo.get("bytes"), len(q_bits_sideinfo)),
        ("latent_wire_payload.bytes", latent.get("bytes"), len(latent_wire_payload)),
        (
            "q_bits_sideinfo.sha256",
            sideinfo.get("sha256"),
            sha256_bytes(q_bits_sideinfo),
        ),
        (
            "latent_wire_payload.sha256",
            latent.get("sha256"),
            sha256_bytes(latent_wire_payload),
        ),
    ]
    for label, expected, actual in checks:
        if expected != actual:
            raise FrameConditionalRuntimePacketError(
                f"materialized {label} mismatch: expected {expected!r}, got {actual!r}"
            )


def _reconcile_wire_contract(
    *,
    source_wire_contract: Mapping[str, Any],
    materialized_wire_contract: Mapping[str, Any],
    q_bits_sideinfo: bytes,
    latent_wire_payload: bytes,
    allow_q_bits_wire_contract_override: bool,
) -> dict[str, Any]:
    source_check = _wire_contract_match_check(
        wire_contract=source_wire_contract,
        q_bits_sideinfo=q_bits_sideinfo,
        latent_wire_payload=latent_wire_payload,
    )
    materialized_check = _wire_contract_match_check(
        wire_contract=materialized_wire_contract,
        q_bits_sideinfo=q_bits_sideinfo,
        latent_wire_payload=latent_wire_payload,
    )
    if not materialized_check["matches"]:
        raise FrameConditionalRuntimePacketError(
            "materialized q-bits wire contract does not match materialized payloads: "
            + str(materialized_check["error"])
        )
    if source_check["matches"]:
        selected_wire_contract = source_wire_contract
        selected_source = "a5_manifest"
    else:
        if not allow_q_bits_wire_contract_override:
            raise FrameConditionalRuntimePacketError(str(source_check["error"]))
        selected_wire_contract = materialized_wire_contract
        selected_source = "materialized_q_bits_override"
    return {
        "selected_wire_contract": selected_wire_contract,
        "manifest_record": {
            "schema": "pr101_frame_conditional_wire_contract_reconciliation.v1",
            "selected_source": selected_source,
            "q_bits_wire_contract_override_allowed": bool(
                allow_q_bits_wire_contract_override
            ),
            "source_wire_contract_matches_materialized": bool(source_check["matches"]),
            "source_wire_contract_mismatch": source_check["error"],
            "materialized_wire_contract_matches_materialized": bool(
                materialized_check["matches"]
            ),
            "source_wire_contract_sha256": _canonical_json_sha256(
                source_wire_contract
            ),
            "materialized_wire_contract_sha256": _canonical_json_sha256(
                materialized_wire_contract
            ),
        },
    }


def _wire_contract_match_check(
    *,
    wire_contract: Mapping[str, Any],
    q_bits_sideinfo: bytes,
    latent_wire_payload: bytes,
) -> dict[str, Any]:
    try:
        _assert_wire_contract_matches_materialized_payloads(
            wire_contract=wire_contract,
            q_bits_sideinfo=q_bits_sideinfo,
            latent_wire_payload=latent_wire_payload,
        )
    except FrameConditionalRuntimePacketError as exc:
        return {"matches": False, "error": str(exc)}
    return {"matches": True, "error": None}


def _extract_storage_ordered_latent_payload(latent_blob: bytes) -> tuple[bytes, np.ndarray]:
    raw = lzma.decompress(
        latent_blob, format=lzma.FORMAT_RAW, filters=PR101_LATENT_LZMA_FILTERS
    )
    meta_len = PR101_LATENT_DIM * 4
    q_len = PR101_N_PAIRS * PR101_LATENT_DIM
    expected = meta_len + q_len
    if len(raw) != expected:
        raise FrameConditionalRuntimePacketError(
            f"decoded PR101 latent payload length {len(raw)} != expected {expected}"
        )
    latent_meta_blob = raw[:meta_len]
    stored = np.frombuffer(raw[meta_len:], dtype=np.uint8).reshape(
        PR101_LATENT_DIM, PR101_N_PAIRS
    )
    q_ordered = stored.copy()
    q_ordered[:, 1:] = (
        np.cumsum(
            ((stored[:, 1:].astype(np.int16) - 128) & 255),
            axis=1,
            dtype=np.uint16,
        ).astype(np.uint8)
        + stored[:, :1]
    )
    return latent_meta_blob, q_ordered.T.copy()


def _build_a5_member_payload(
    *,
    decoder_blob: bytes,
    latent_meta_blob: bytes,
    q_bits_sideinfo: bytes,
    latent_wire_payload: bytes,
    sidecar_blob: bytes,
) -> bytes:
    for label, data in (
        ("decoder_blob", decoder_blob),
        ("latent_meta_blob", latent_meta_blob),
        ("q_bits_sideinfo", q_bits_sideinfo),
        ("latent_wire_payload", latent_wire_payload),
    ):
        if not data:
            raise FrameConditionalRuntimePacketError(f"{label} must not be empty")
        if len(data) >= (1 << 32):
            raise FrameConditionalRuntimePacketError(f"{label} exceeds u32 length")
    return (
        A5_MAGIC
        + len(decoder_blob).to_bytes(4, "little")
        + len(latent_meta_blob).to_bytes(4, "little")
        + len(q_bits_sideinfo).to_bytes(4, "little")
        + len(latent_wire_payload).to_bytes(4, "little")
        + decoder_blob
        + latent_meta_blob
        + q_bits_sideinfo
        + latent_wire_payload
        + sidecar_blob
    )


_PARSE_ARCHIVE_RE = re.compile(
    r"def parse_archive\(archive_bytes\):\n"
    r"(?P<body>.*?)"
    r"    return decode_decoder_compact\(decoder_blob\), latents, meta\n",
    re.DOTALL,
)


def _patch_packet_codec(packet_dir: Path) -> dict[str, Any]:
    codec_path = packet_dir / "src" / "codec.py"
    if not codec_path.is_file():
        raise FrameConditionalRuntimePacketError(f"packet runtime missing {codec_path}")
    text = codec_path.read_text(encoding="utf-8")
    replacement = '''import hashlib as _a5_hashlib

A5_FRAME_CONDITIONAL_MAGIC = b"A5FC"
A5_FRAME_CONDITIONAL_HEADER_LEN = 20
A5_FRAME_CONDITIONAL_WIRE_SCHEMA = "tac_frame_conditional_latent_wire.v1"
A5_Q_BITS_RAW3_BYTES = (N_PAIRS * 3 + 7) // 8
A5_Q_BITS_BINARY_LOW_HIGH_MASK_BYTES = 2 + ((N_PAIRS + 7) // 8)
A5_Q_BITS_CHANNEL_RAW3_BYTES = (LATENT_DIM * 3 + 7) // 8


def _a5_sha256(data):
    return _a5_hashlib.sha256(data).hexdigest()


def _a5_read_msb_bits(data, bit_pos, width):
    value = 0
    for _ in range(width):
        if bit_pos >= len(data) * 8:
            raise ValueError("A5 frame-conditional bitstream truncated")
        value = (value << 1) | ((data[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1)
        bit_pos += 1
    return value, bit_pos


def _a5_assert_zero_padding(data, bit_pos):
    while bit_pos < len(data) * 8:
        if (data[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1:
            raise ValueError("non-zero A5 frame-conditional padding bits")
        bit_pos += 1


def _a5_unpack_q_bits(data):
    if len(data) == A5_Q_BITS_BINARY_LOW_HIGH_MASK_BYTES:
        low = data[0] + 1
        high = data[1] + 1
        if low < 1 or high > 8 or low > high:
            raise ValueError(f"bad A5 binary q-bit range low={low} high={high}")
        out = np.empty(N_PAIRS, dtype=np.uint8)
        mask = data[2:]
        bit_pos = 0
        for i in range(N_PAIRS):
            selector, bit_pos = _a5_read_msb_bits(mask, bit_pos, 1)
            out[i] = high if selector else low
        _a5_assert_zero_padding(mask, bit_pos)
        return out
    if len(data) != A5_Q_BITS_RAW3_BYTES:
        raise ValueError(
            "A5 q-bit side-info length "
            f"{len(data)} != raw3 {A5_Q_BITS_RAW3_BYTES} "
            f"or binary {A5_Q_BITS_BINARY_LOW_HIGH_MASK_BYTES}"
        )
    out = np.empty(N_PAIRS, dtype=np.uint8)
    bit_pos = 0
    for i in range(N_PAIRS):
        code, bit_pos = _a5_read_msb_bits(data, bit_pos, 3)
        out[i] = code + 1
    _a5_assert_zero_padding(data, bit_pos)
    return out


def _a5_unpack_channel_q_bits(data):
    if len(data) != A5_Q_BITS_CHANNEL_RAW3_BYTES:
        raise ValueError(
            "A5 channel q-bit side-info length "
            f"{len(data)} != expected {A5_Q_BITS_CHANNEL_RAW3_BYTES}"
        )
    out = np.empty(LATENT_DIM, dtype=np.uint8)
    bit_pos = 0
    for i in range(LATENT_DIM):
        code, bit_pos = _a5_read_msb_bits(data, bit_pos, 3)
        out[i] = code + 1
    _a5_assert_zero_padding(data, bit_pos)
    return out


def _a5_unpack_latent_codes(data, q_bits):
    expected_bits = int(q_bits.astype(np.uint64).sum()) * LATENT_DIM
    expected_bytes = (expected_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"A5 latent bitstream length {len(data)} != expected {expected_bytes}"
        )
    out = np.empty((N_PAIRS, LATENT_DIM), dtype=np.uint8)
    bit_pos = 0
    for pair_idx, bits in enumerate(q_bits):
        bit_count = int(bits)
        shift = 8 - bit_count
        for dim_idx in range(LATENT_DIM):
            code, bit_pos = _a5_read_msb_bits(data, bit_pos, bit_count)
            out[pair_idx, dim_idx] = (code << shift) & 255
    _a5_assert_zero_padding(data, bit_pos)
    return out


def _a5_unpack_channel_latent_codes(data, q_bits):
    expected_bits = int(q_bits.astype(np.uint64).sum()) * N_PAIRS
    expected_bytes = (expected_bits + 7) // 8
    if len(data) != expected_bytes:
        raise ValueError(
            f"A5 channel latent bitstream length {len(data)} != expected {expected_bytes}"
        )
    out = np.empty((N_PAIRS, LATENT_DIM), dtype=np.uint8)
    bit_pos = 0
    for pair_idx in range(N_PAIRS):
        for dim_idx, bits in enumerate(q_bits):
            bit_count = int(bits)
            shift = 8 - bit_count
            code, bit_pos = _a5_read_msb_bits(data, bit_pos, bit_count)
            out[pair_idx, dim_idx] = (code << shift) & 255
    _a5_assert_zero_padding(data, bit_pos)
    return out


def decode_latents_frame_conditional(latent_meta_blob, q_bits_sideinfo_blob, latent_wire_blob):
    expected_meta_len = LATENT_DIM * 4
    if len(latent_meta_blob) != expected_meta_len:
        raise ValueError(
            f"A5 latent meta length {len(latent_meta_blob)} != expected {expected_meta_len}"
        )
    mins = torch.from_numpy(
        np.frombuffer(latent_meta_blob[:LATENT_DIM * 2], dtype=np.float16).copy()
    ).float()
    scales = torch.from_numpy(
        np.frombuffer(latent_meta_blob[LATENT_DIM * 2:], dtype=np.float16).copy()
    ).float()
    if len(q_bits_sideinfo_blob) == A5_Q_BITS_CHANNEL_RAW3_BYTES:
        q_bits = _a5_unpack_channel_q_bits(q_bits_sideinfo_blob)
        q_ordered = _a5_unpack_channel_latent_codes(latent_wire_blob, q_bits)
    else:
        q_bits = _a5_unpack_q_bits(q_bits_sideinfo_blob)
        q_ordered = _a5_unpack_latent_codes(latent_wire_blob, q_bits)
    q = np.empty((N_PAIRS, LATENT_DIM), dtype=np.uint8)
    q[:, LATENT_DIM_ORDER] = q_ordered
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def parse_archive(archive_bytes):
    a5_frame_conditional = archive_bytes.startswith(A5_FRAME_CONDITIONAL_MAGIC)
    if a5_frame_conditional:
        if len(archive_bytes) < A5_FRAME_CONDITIONAL_HEADER_LEN:
            raise ValueError("bad A5 frame-conditional archive header")
        decoder_len = int.from_bytes(archive_bytes[4:8], "little")
        latent_meta_len = int.from_bytes(archive_bytes[8:12], "little")
        q_bits_sideinfo_len = int.from_bytes(archive_bytes[12:16], "little")
        latent_wire_len = int.from_bytes(archive_bytes[16:20], "little")
        decoder_start = A5_FRAME_CONDITIONAL_HEADER_LEN
        decoder_end = decoder_start + decoder_len
        latent_meta_start = decoder_end
        latent_meta_end = latent_meta_start + latent_meta_len
        q_bits_start = latent_meta_end
        q_bits_end = q_bits_start + q_bits_sideinfo_len
        latent_wire_start = q_bits_end
        latent_wire_end = latent_wire_start + latent_wire_len
        if (
            decoder_len <= 0
            or latent_meta_len <= 0
            or q_bits_sideinfo_len <= 0
            or latent_wire_len <= 0
            or latent_wire_end > len(archive_bytes)
        ):
            raise ValueError("bad A5 frame-conditional archive lengths")
        decoder_blob = archive_bytes[decoder_start:decoder_end]
        latent_meta_blob = archive_bytes[latent_meta_start:latent_meta_end]
        q_bits_sideinfo_blob = archive_bytes[q_bits_start:q_bits_end]
        latent_wire_blob = archive_bytes[latent_wire_start:latent_wire_end]
        sidecar_blob = archive_bytes[latent_wire_end:]
    else:
        decoder_blob = archive_bytes[:DECODER_BLOB_LEN]
        latent_blob = archive_bytes[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
        sidecar_blob = archive_bytes[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    if not decoder_blob:
        raise ValueError("bad compact archive")
    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    if a5_frame_conditional:
        latents_base = decode_latents_frame_conditional(
            latent_meta_blob,
            q_bits_sideinfo_blob,
            latent_wire_blob,
        )
        meta.update({
            "frame_conditional_wire_schema": A5_FRAME_CONDITIONAL_WIRE_SCHEMA,
            "parse_archive_consumes_q_bits_sideinfo": True,
            "decode_latents_consumes_variable_width_payload": True,
            "q_bits_sideinfo_sha256": _a5_sha256(q_bits_sideinfo_blob),
            "latent_wire_payload_sha256": _a5_sha256(latent_wire_blob),
        })
    else:
        if not latent_blob:
            raise ValueError("bad compact archive")
        latents_base = decode_latents_compact(latent_blob)
    latents = apply_latent_sidecar(latents_base, sidecar_blob)
    return decode_decoder_compact(decoder_blob), latents, meta
'''
    patched, count = _PARSE_ARCHIVE_RE.subn(replacement, text, count=1)
    if count != 1:
        raise FrameConditionalRuntimePacketError(
            "could not patch PR101 runtime parse_archive anchor for A5FC"
        )
    codec_path.write_text(patched, encoding="utf-8")
    return {
        "codec_path": _repo_rel(codec_path),
        "codec_sha256": sha256_file(codec_path),
        "consumes_schema": FRAME_CONDITIONAL_LATENT_WIRE_SCHEMA,
        "codec_parse_archive_supports_a5fc": True,
        "parse_archive_consumes_q_bits_sideinfo": True,
        "decode_latents_consumes_variable_width_payload": True,
        "supported_q_bits_sideinfo_encodings": [
            FRAME_CONDITIONAL_Q_BITS_ENCODING_RAW3,
            FRAME_CONDITIONAL_Q_BITS_ENCODING_BINARY_LOW_HIGH_MASK,
            FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3,
        ],
        "magic": A5_MAGIC.decode("ascii"),
        "header_bytes": A5_HEADER_LEN,
        "source_runtime_mutated": False,
        "score_claim": False,
    }


def _patch_packet_inflate_sh(packet_dir: Path) -> dict[str, Any]:
    inflate_path = packet_dir / "inflate.sh"
    if not inflate_path.is_file():
        raise FrameConditionalRuntimePacketError(f"packet runtime missing {inflate_path}")
    text = inflate_path.read_text(encoding="utf-8")
    old = 'python "$HERE/inflate.py" "$SRC" "$DST"'
    new = '"${PYTHON:-python3}" "$HERE/inflate.py" "$SRC" "$DST"'
    if old not in text and new not in text:
        raise FrameConditionalRuntimePacketError(
            "packet inflate.sh does not contain the expected PR101 inflate.py call"
        )
    if old in text:
        text = text.replace(old, new, 1)
        inflate_path.write_text(text, encoding="utf-8")
    return {
        "inflate_sh_path": _repo_rel(inflate_path),
        "inflate_sh_sha256": sha256_file(inflate_path),
        "portable_python_fallback": True,
        "python_invocation": new,
    }


def _packet_local_parse_smoke(
    packet_dir: Path, archive_path: Path, *, label: str
) -> dict[str, Any]:
    script = r'''
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

packet_dir = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
label = sys.argv[3]
src_dir = packet_dir / "src"
sys.path.insert(0, str(src_dir))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load_module("model", src_dir / "model.py")
codec = load_module("codec", src_dir / "codec.py")
with zipfile.ZipFile(archive_path) as zf:
    infos = [info for info in zf.infolist() if not info.is_dir()]
    if len(infos) != 1:
        raise RuntimeError(f"expected one archive member, found {len(infos)}")
    payload = zf.read(infos[0])

def tensor_sha256(tensor):
    if not hasattr(tensor, "detach"):
        return None
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(json.dumps(list(cpu.shape), sort_keys=True).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()

def parse_header(raw):
    if not raw.startswith(b"A5FC"):
        return None
    decoder_len = int.from_bytes(raw[4:8], "little")
    latent_meta_len = int.from_bytes(raw[8:12], "little")
    sideinfo_len = int.from_bytes(raw[12:16], "little")
    latent_wire_len = int.from_bytes(raw[16:20], "little")
    decoder_start = 20
    decoder_end = decoder_start + decoder_len
    latent_meta_start = decoder_end
    latent_meta_end = latent_meta_start + latent_meta_len
    sideinfo_start = latent_meta_end
    sideinfo_end = sideinfo_start + sideinfo_len
    latent_wire_start = sideinfo_end
    latent_wire_end = latent_wire_start + latent_wire_len
    return {
        "decoder_len": decoder_len,
        "latent_meta_len": latent_meta_len,
        "q_bits_sideinfo_len": sideinfo_len,
        "latent_wire_len": latent_wire_len,
        "decoder_start": decoder_start,
        "decoder_end": decoder_end,
        "latent_meta_start": latent_meta_start,
        "latent_meta_end": latent_meta_end,
        "q_bits_sideinfo_start": sideinfo_start,
        "q_bits_sideinfo_end": sideinfo_end,
        "latent_wire_start": latent_wire_start,
        "latent_wire_end": latent_wire_end,
    }

decoder_sd, latents, meta = codec.parse_archive(payload)
latents_sha = tensor_sha256(latents)
result = {
    "passed": True,
    "label": label,
    "member_name": infos[0].filename,
    "member_sha256": hashlib.sha256(payload).hexdigest(),
    "a5_magic": payload.startswith(b"A5FC"),
    "decoder_tensor_count": len(decoder_sd),
    "latents_shape": list(latents.shape) if hasattr(latents, "shape") else None,
    "latents_dtype": str(getattr(latents, "dtype", "")),
    "latents_sha256": latents_sha,
    "meta": {
        "n_pairs": int(meta["n_pairs"]),
        "latent_dim": int(meta["latent_dim"]),
        "base_channels": int(meta["base_channels"]),
        "eval_size": list(meta["eval_size"]),
    },
    "command": "python -c <a5_packet_local_parse_smoke> packet_dir archive.zip label",
}
header = parse_header(payload)
if header is not None:
    sideinfo = payload[header["q_bits_sideinfo_start"]:header["q_bits_sideinfo_end"]]
    latent_wire = payload[header["latent_wire_start"]:header["latent_wire_end"]]
    result.update({
        "a5_header": header,
        "frame_conditional_wire_schema": meta.get("frame_conditional_wire_schema"),
        "parse_archive_consumes_q_bits_sideinfo": meta.get("parse_archive_consumes_q_bits_sideinfo"),
        "decode_latents_consumes_variable_width_payload": meta.get("decode_latents_consumes_variable_width_payload"),
        "consumed_q_bits_sideinfo_sha256": meta.get("q_bits_sideinfo_sha256"),
        "consumed_latent_wire_payload_sha256": meta.get("latent_wire_payload_sha256"),
        "independent_q_bits_sideinfo_sha256": hashlib.sha256(sideinfo).hexdigest(),
        "independent_latent_wire_payload_sha256": hashlib.sha256(latent_wire).hexdigest(),
    })
    zero_sideinfo = bytes(len(sideinfo))
    zero_payload = (
        payload[:header["q_bits_sideinfo_start"]]
        + zero_sideinfo
        + payload[header["q_bits_sideinfo_end"]:]
    )
    try:
        codec.parse_archive(zero_payload)
    except Exception as exc:
        result["zero_sideinfo_negative_control_rejected"] = True
        result["zero_sideinfo_negative_control_error"] = str(exc)
    else:
        result["zero_sideinfo_negative_control_rejected"] = False
        result["zero_sideinfo_negative_control_error"] = ""
    if latent_wire:
        mutated = bytearray(latent_wire)
        mutated[0] ^= 0x80
        mutated_payload = (
            payload[:header["latent_wire_start"]]
            + bytes(mutated)
            + payload[header["latent_wire_end"]:]
        )
        _decoder2, mutated_latents, _meta2 = codec.parse_archive(mutated_payload)
        mutated_sha = tensor_sha256(mutated_latents)
        result["latent_wire_mutation_changed_latents"] = mutated_sha != latents_sha
        result["latent_wire_mutation_latents_sha256"] = mutated_sha
    else:
        result["latent_wire_mutation_changed_latents"] = False
print(json.dumps(result, sort_keys=True))
'''
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script, str(packet_dir), str(archive_path), label],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise FrameConditionalRuntimePacketError(
            f"packet-local parse smoke failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FrameConditionalRuntimePacketError(
            f"packet-local parse smoke emitted non-JSON output: {proc.stdout!r}"
        ) from exc


def _build_runtime_patch_manifest(
    *,
    packet_dir: Path,
    runtime_patch: Mapping[str, Any],
    runtime_tree_sha256: str,
    runtime_files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_PATCH_SCHEMA,
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "packet_dir": _repo_rel(packet_dir),
        "packet_local_runtime_patch": dict(runtime_patch),
        "runtime_tree_sha256": runtime_tree_sha256,
        "runtime_files": runtime_files,
        "score_claim": False,
    }


def _build_runtime_consumption_proof(
    *,
    candidate_id: str,
    candidate_archive_record: Mapping[str, Any],
    candidate_member_payload: bytes,
    q_bits_sideinfo: bytes,
    latent_wire_payload: bytes,
    source_parse_smoke: Mapping[str, Any],
    candidate_parse_smoke: Mapping[str, Any],
    runtime_consumes_changed_archive_bytes: bool,
    runtime_consumption_proof_path: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    expected_sideinfo_sha = sha256_bytes(q_bits_sideinfo)
    expected_latent_sha = sha256_bytes(latent_wire_payload)
    if candidate_parse_smoke.get("consumed_q_bits_sideinfo_sha256") != expected_sideinfo_sha:
        blockers.append("runtime_parse_missing_q_bits_sideinfo_sha")
    if candidate_parse_smoke.get("consumed_latent_wire_payload_sha256") != expected_latent_sha:
        blockers.append("runtime_parse_missing_latent_wire_payload_sha")
    if candidate_parse_smoke.get("zero_sideinfo_negative_control_rejected") is not True:
        blockers.append("zero_sideinfo_negative_control_not_rejected")
    if candidate_parse_smoke.get("latent_wire_mutation_changed_latents") is not True:
        blockers.append("latent_wire_mutation_did_not_change_latents")
    if not runtime_consumes_changed_archive_bytes:
        blockers.append("runtime_consumes_changed_archive_bytes_not_proven")
    ready = not blockers
    payload = {
        "schema": RUNTIME_PROOF_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_archive": dict(candidate_archive_record),
        "candidate_archive_path": candidate_archive_record.get("path"),
        "candidate_archive_bytes": candidate_archive_record.get("bytes"),
        "candidate_archive_sha256": candidate_archive_record.get("sha256"),
        "candidate_member_sha256": sha256_bytes(candidate_member_payload),
        "consumed_q_bits_sideinfo_sha256": expected_sideinfo_sha,
        "consumed_latent_wire_payload_sha256": expected_latent_sha,
        "ready_for_exact_eval_runtime": ready,
        "runtime_consumption_proven": ready,
        "runtime_consumes_changed_archive_bytes": runtime_consumes_changed_archive_bytes,
        "source_packet_local_parse_smoke": dict(source_parse_smoke),
        "candidate_packet_local_parse_smoke": dict(candidate_parse_smoke),
        "blockers": blockers,
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "score_claim": False,
    }
    payload.update(
        archive_bound_candidate_contract_fields_for_row(
            {
                "archive_native_transform_kind": SCHEMA_VERSION,
                "target_kind": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "candidate_archive": dict(candidate_archive_record),
                "byte_closed_candidate_emitted": ready,
                "byte_closed_candidate_materialized": ready,
                "candidate_archive_materialized": ready,
                "runtime_consumption_proof_ready": ready,
                "runtime_consumption_proof_path": _repo_rel(
                    runtime_consumption_proof_path
                ),
                "receiver_contract_kind": RUNTIME_PROOF_SCHEMA,
                "receiver_contract_satisfied": ready,
                "runtime_adapter_ready": ready,
                "semantic_payload_changed": True,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "readiness_blockers": blockers,
                **FALSE_AUTHORITY_FIELDS,
            },
            repo_root=REPO_ROOT,
            selected_transform_kind=SCHEMA_VERSION,
            family_id="pr101_frame_conditional_runtime_packet",
            candidate_chain_id=candidate_id,
        )
    )
    return payload


def _archive_member_manifest(
    *,
    source_decoder_blob: bytes,
    source_latent_blob: bytes,
    source_sidecar_blob: bytes,
    latent_meta_blob: bytes,
    q_bits_sideinfo: bytes,
    latent_wire_payload: bytes,
    candidate_member_payload: bytes,
) -> dict[str, Any]:
    decoder_start = A5_HEADER_LEN
    decoder_end = decoder_start + len(source_decoder_blob)
    latent_meta_start = decoder_end
    latent_meta_end = latent_meta_start + len(latent_meta_blob)
    q_bits_start = latent_meta_end
    q_bits_end = q_bits_start + len(q_bits_sideinfo)
    latent_wire_start = q_bits_end
    latent_wire_end = latent_wire_start + len(latent_wire_payload)
    return {
        "member_name": "x",
        "layout_magic": A5_MAGIC.decode("ascii"),
        "header_bytes": A5_HEADER_LEN,
        "decoder_len_field": len(source_decoder_blob),
        "latent_meta_len_field": len(latent_meta_blob),
        "q_bits_sideinfo_len_field": len(q_bits_sideinfo),
        "latent_wire_len_field": len(latent_wire_payload),
        "decoder_blob_offset": decoder_start,
        "decoder_blob_bytes": len(source_decoder_blob),
        "latent_meta_offset": latent_meta_start,
        "latent_meta_bytes": len(latent_meta_blob),
        "q_bits_sideinfo_offset": q_bits_start,
        "q_bits_sideinfo_bytes": len(q_bits_sideinfo),
        "latent_wire_payload_offset": latent_wire_start,
        "latent_wire_payload_bytes": len(latent_wire_payload),
        "sidecar_blob_offset": latent_wire_end,
        "sidecar_blob_bytes": len(source_sidecar_blob),
        "source_decoder_sha256": sha256_bytes(source_decoder_blob),
        "candidate_decoder_sha256": sha256_bytes(source_decoder_blob),
        "source_latent_sha256": sha256_bytes(source_latent_blob),
        "candidate_latent_meta_sha256": sha256_bytes(latent_meta_blob),
        "candidate_q_bits_sideinfo_sha256": sha256_bytes(q_bits_sideinfo),
        "candidate_latent_wire_payload_sha256": sha256_bytes(latent_wire_payload),
        "source_sidecar_sha256": sha256_bytes(source_sidecar_blob),
        "candidate_sidecar_sha256": sha256_bytes(source_sidecar_blob),
        "candidate_member_sha256": sha256_bytes(candidate_member_payload),
    }


def _parser_section_custody_for_candidate(
    *, candidate_archive_path: Path, candidate_id: str
) -> dict[str, Any]:
    section_record = hnerv_packet_sections.build_packet_section_manifest(
        candidate_archive_path,
        label=f"A5FC:{candidate_id}",
        parser=hnerv_packet_sections.PARSER_A5FC,
        repo_root=REPO_ROOT,
    )
    return {
        "source": "tac.analysis.hnerv_packet_sections.build_packet_section_manifest",
        "parser": section_record["parser"],
        "parser_section_gate": section_record["parser_section_gate"],
        "parser_section_manifest": section_record["parser_section_manifest"],
        "sections": section_record["sections"],
        "coverage": section_record["coverage"],
        "archive": section_record["archive"],
        "member": section_record["member"],
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _write_packet_report(
    *,
    packet_dir: Path,
    candidate_id: str,
    candidate_archive_record: Mapping[str, Any],
    source_archive_record: Mapping[str, Any],
    runtime_tree_sha256: str,
    dispatch_blockers: Sequence[str],
) -> dict[str, Any]:
    report_path = packet_dir / "report.txt"
    lines = [
        "A5 frame-conditional PR101 runtime packet scaffold",
        "",
        f"candidate_id: {candidate_id}",
        f"archive_bytes: {candidate_archive_record['bytes']}",
        f"archive_sha256: {candidate_archive_record['sha256']}",
        f"source_archive_sha256: {source_archive_record['sha256']}",
        f"packet_builder_runtime_file_sha256: {runtime_tree_sha256}",
        "score_claim: false",
        "promotion_eligible: false",
        "rank_or_kill_eligible: false",
        "ready_for_exact_eval_dispatch: false",
        "evidence_grade: empirical",
        "",
        "dispatch_blockers:",
    ]
    lines.extend(f"- {blocker}" for blocker in dispatch_blockers)
    lines.extend(
        [
            "",
            "This packet proves local runtime parsing and consumption of A5 q-bit "
            "side-info plus variable-width latent bytes. It has not run inflate "
            "parity, exact CUDA auth eval, contest-CPU auth eval, or operator "
            "score-claim review.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return _file_record(report_path, relpath="report.txt")


def _read_single_member_zip(path: Path) -> tuple[dict[str, Any], str, bytes]:
    archive_bytes = path.read_bytes()
    archive_record: dict[str, Any] = {
        "path": _repo_rel(path),
        "bytes": len(archive_bytes),
        "sha256": sha256_bytes(archive_bytes),
        "members": [],
    }
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise FrameConditionalRuntimePacketError(
                f"expected single-member archive, got {len(infos)} members"
            )
        info = infos[0]
        _validate_zip_member_name(info.filename)
        local = _local_header_record(path, info)
        if local["name"] != info.filename:
            raise FrameConditionalRuntimePacketError(
                f"ZIP local/central member-name mismatch: {local['name']} vs {info.filename}"
            )
        member = zf.read(info)
        archive_record["members"].append(
            {
                "name": info.filename,
                "bytes": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "crc": f"{info.CRC:08x}",
                "date_time": list(info.date_time),
                "external_attr": info.external_attr,
                "flag_bits": info.flag_bits,
                "local_header": local,
                "sha256": sha256_bytes(member),
            }
        )
    return archive_record, infos[0].filename, member


def _write_single_member_zip(path: Path, member_name: str, payload: bytes) -> dict[str, Any]:
    _validate_zip_member_name(member_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename=member_name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (0o644 & 0xFFFF) << 16
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload)
    archive_record, _member_name, _member = _read_single_member_zip(path)
    return archive_record


def _split_pr101_member(payload: bytes) -> tuple[bytes, bytes, bytes]:
    minimum = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    if len(payload) < minimum:
        raise FrameConditionalRuntimePacketError(
            f"PR101 member length {len(payload)} < fixed decoder+latent minimum {minimum}"
        )
    return (
        payload[:PR101_DECODER_BLOB_LEN],
        payload[PR101_DECODER_BLOB_LEN : PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN],
        payload[PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN :],
    )


def _copy_runtime_tree(source_runtime_dir: Path, packet_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel in _iter_runtime_files(source_runtime_dir):
        source = source_runtime_dir / rel
        target = packet_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, _copy_mode(source))
        records.append(_file_record(target, relpath=rel.as_posix()))
    return records


def _iter_runtime_files(runtime_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in runtime_dir.rglob("*"):
        rel = path.relative_to(runtime_dir)
        if _should_exclude(rel):
            continue
        if path.is_symlink():
            raise FrameConditionalRuntimePacketError(f"runtime refuses symlink: {path}")
        if path.is_file():
            files.append(rel)
    return sorted(files, key=lambda item: item.as_posix())


def _file_record(path: Path, *, relpath: str | None = None) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "mode": _mode_string(path.stat().st_mode),
        "relpath": relpath or path.name,
        "sha256": sha256_file(path),
    }


def _runtime_tree_sha256(runtime_files: list[dict[str, Any]]) -> str:
    basis = [
        {
            "bytes": row["bytes"],
            "mode": row["mode"],
            "relpath": row["relpath"],
            "sha256": row["sha256"],
        }
        for row in runtime_files
    ]
    return hashlib.sha256(json_text(basis).encode("utf-8")).hexdigest()


def _bash_n(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )
    return {
        "command": f"bash -n {path}",
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "stdout": proc.stdout.strip(),
    }


def _artifact_ref(path: Path) -> dict[str, Any]:
    return {
        "path": _repo_rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _prepare_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FrameConditionalRuntimePacketError(f"output is not a directory: {path}")
        if any(path.iterdir()):
            if not force:
                raise FrameConditionalRuntimePacketError(
                    f"output directory is not empty; pass --force: {path}"
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _assert_output_dir_isolated(*, output_dir: Path, inputs: Sequence[Path]) -> None:
    output = output_dir.resolve(strict=False)
    for input_path in inputs:
        resolved = input_path.resolve(strict=False)
        if output == resolved or _is_relative_to(output, resolved) or _is_relative_to(
            resolved, output
        ):
            raise FrameConditionalRuntimePacketError(
                f"output directory overlaps input path: output={output_dir} input={input_path}"
            )


def _local_header_record(path: Path, info: zipfile.ZipInfo) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise FrameConditionalRuntimePacketError(f"bad ZIP local header in {path}")
        flag_bits = struct.unpack_from("<H", header, 6)[0]
        method = struct.unpack_from("<H", header, 8)[0]
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        handle.read(extra_len)
    return {
        "name": _decode_zip_name(raw_name, flag_bits),
        "flag_bits": flag_bits,
        "compress_type": method,
        "extra_len": extra_len,
    }


def _validate_zip_member_name(name: str) -> None:
    path = Path(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or any(ord(ch) < 32 for ch in name)
        or re.match(r"^[A-Za-z]:", name)
        or name.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(part.startswith(".") or part == "__MACOSX" for part in path.parts)
    ):
        raise FrameConditionalRuntimePacketError(f"unsafe ZIP member name: {name!r}")


def _decode_zip_name(raw: bytes, flag_bits: int) -> str | None:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        return None


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FrameConditionalRuntimePacketError(f"{label} not found: {path}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FrameConditionalRuntimePacketError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FrameConditionalRuntimePacketError(f"{label} must be a positive integer")
    return value


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(path, REPO_ROOT)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _should_exclude(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return path.name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES


def _is_packet_custody_file(path: Path) -> bool:
    return path.name in PACKET_CUSTODY_FILENAMES


def _copy_mode(source: Path) -> int:
    return 0o755 if source.stat().st_mode & 0o111 else 0o644


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_ts() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-manifest", type=Path, default=DEFAULT_A5_MANIFEST)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME_DIR)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument(
        "--q-bits-json",
        type=Path,
        help=(
            "Optional JSON q-bit override. Accepts the A5 score-marginal "
            "manifest key per_pair_q_bits, or exactly one q_bits_per_pair/q_bits key."
        ),
    )
    parser.add_argument(
        "--channel-q-bits-json",
        type=Path,
        help=(
            "Optional JSON per-channel q-bit override. Requires "
            "--q-bits-sideinfo-encoding channel_raw3 and exactly one "
            "per_channel_q_bits/q_bits_per_channel key."
        ),
    )
    parser.add_argument(
        "--recompute-wire-contract-for-q-bits",
        action="store_true",
        help=(
            "Allow --q-bits-json to supersede the source A5 manifest wire-contract "
            "hashes. The output manifest records the mismatch and selected contract."
        ),
    )
    parser.add_argument(
        "--q-bits-sideinfo-encoding",
        choices=(
            FRAME_CONDITIONAL_Q_BITS_ENCODING_RAW3,
            FRAME_CONDITIONAL_Q_BITS_ENCODING_BINARY_LOW_HIGH_MASK,
            FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3,
        ),
        default=FRAME_CONDITIONAL_Q_BITS_ENCODING_RAW3,
        help=(
            "Encoding for q-bit side-info. raw3 preserves the original 3-bit "
            "per-pair stream; binary_low_high_mask is a compact two-level "
            "selector mask for one- or two-q-bit schedules; channel_raw3 "
            "stores one 3-bit q value per latent dimension."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--candidate-id",
        default="pr101_a5_frame_conditional_runtime_packet",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or Path(
        f"experiments/results/pr101_frame_conditional_runtime_packet_{_utc_ts()}"
    )
    try:
        q_bits_override = (
            _load_q_bits_override_json(args.q_bits_json)
            if args.q_bits_json is not None
            else None
        )
        channel_q_bits_override = (
            _load_channel_q_bits_override_json(args.channel_q_bits_json)
            if args.channel_q_bits_json is not None
            else None
        )
        manifest = build_frame_conditional_runtime_packet(
            a5_manifest_path=args.a5_manifest,
            source_archive=args.source_archive,
            source_runtime_dir=args.source_runtime_dir,
            output_dir=output_dir,
            video_path=args.video_path,
            q_bits_override=q_bits_override,
            q_bits_override_source_path=args.q_bits_json,
            channel_q_bits_override=channel_q_bits_override,
            channel_q_bits_override_source_path=args.channel_q_bits_json,
            q_bits_sideinfo_encoding=args.q_bits_sideinfo_encoding,
            allow_q_bits_wire_contract_override=args.recompute_wire_contract_for_q_bits,
            candidate_id=args.candidate_id,
            force=args.force,
        )
    except (
        FrameConditionalRuntimePacketError,
        OSError,
        json.JSONDecodeError,
        lzma.LZMAError,
        zipfile.BadZipFile,
    ) as exc:
        raise SystemExit(f"A5 runtime packet build failed: {exc}") from None
    archive = manifest["candidate_archive"]
    proof = manifest["runtime_consumption_proof"]
    print(
        f"candidate_archive_manifest: {output_dir / 'candidate_archive_manifest.json'}"
    )
    print(f"candidate_archive: {archive['path']} ({archive['bytes']} bytes)")
    print(f"candidate_archive_sha256: {archive['sha256']}")
    print(f"runtime_consumption_proof: {proof['path']}")
    print(f"ready_for_exact_eval_dispatch: {manifest['ready_for_exact_eval_dispatch']}")
    print(f"score_claim: {manifest['score_claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
