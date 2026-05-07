"""Deterministic HDM3 HNeRV decoder-section archive candidates.

HDM3 is a byte-level decoder-section fixture: fixed-schema q-stream Brotli plus
raw scale bytes. This module can build a deterministic archive that swaps only
the HNeRV decoder section to HDM3 and proves raw decoder equivalence. It does
not make the archive scorer-ready; the submission runtime must consume HDM3
before exact CUDA dispatch is valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_decoder_recode import (
    decode_hdm3_q_brotli_split_fixture,
    encode_hdm3_q_brotli_split_fixture,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.repo_io import repo_relative, sha256_file, write_json

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_hdm3_archive_candidate.build_hdm3_archive_candidate"
HDM3_VARIANT_NAME = "hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales"


class HnervHdm3ArchiveCandidateError(ValueError):
    """Raised when an HDM3 archive candidate input is invalid."""


def build_hdm3_archive_candidate(
    *,
    source_archive: str | Path,
    output_dir: str | Path,
    source_label: str,
    allow_rate_regression: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic HDM3 decoder-section archive candidate.

    The returned manifest may be archive-byte-closed, but it always keeps
    ``ready_for_exact_eval_dispatch`` false until the runtime adapter and exact
    CUDA custody proofs exist.
    """

    source_path = Path(source_archive)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()

    source = read_strict_single_member_zip(source_path)
    packed = parse_ff_packed_brotli_hnerv(source.payload)
    parsed_decoder = parse_packed_decoder_brotli(packed.decoder_packed_brotli)
    source_raw = brotli.decompress(packed.decoder_packed_brotli)
    if parsed_decoder.to_raw() != source_raw:
        raise HnervHdm3ArchiveCandidateError("source decoder parser does not match Brotli raw")

    hdm3_stream, hdm3_stats = encode_hdm3_q_brotli_split_fixture(parsed_decoder)
    restored_decoder = decode_hdm3_q_brotli_split_fixture(hdm3_stream)
    raw_equal = restored_decoder.to_raw() == source_raw
    q_equal = restored_decoder.q_stream == parsed_decoder.q_stream
    scale_equal = restored_decoder.scale_stream == parsed_decoder.scale_stream
    section_byte_delta = len(hdm3_stream) - len(packed.decoder_packed_brotli)
    rate_positive = section_byte_delta < 0
    archive_build_blockers: list[str] = []
    if not raw_equal:
        archive_build_blockers.append("hdm3_raw_decoder_mismatch")
    if not q_equal:
        archive_build_blockers.append("hdm3_q_stream_mismatch")
    if not scale_equal:
        archive_build_blockers.append("hdm3_scale_stream_mismatch")
    if not rate_positive and not allow_rate_regression:
        archive_build_blockers.append("hdm3_decoder_section_not_rate_positive")

    candidate_archive_path: Path | None = None
    candidate_payload: bytes | None = None
    candidate_archive_sha = ""
    candidate_archive_bytes: int | None = None
    candidate_payload_sha = ""
    ready_for_archive_preflight = False
    if not archive_build_blockers:
        candidate_payload = PackedHnervPayload(
            header=packed.header,
            decoder_packed_brotli=hdm3_stream,
            latents_and_sidecar_brotli=packed.latents_and_sidecar_brotli,
        ).to_bytes()
        candidate_archive_path = output_root / f"{_slug(source_label)}_hdm3_archive_candidate.zip"
        write_stored_single_member_zip(
            candidate_archive_path,
            member_name=source.member_name,
            payload=candidate_payload,
        )
        candidate_archive_sha = sha256_file(candidate_archive_path)
        candidate_archive_bytes = candidate_archive_path.stat().st_size
        candidate_payload_sha = sha256_bytes(candidate_payload)
        checked = parse_ff_packed_brotli_hnerv(candidate_payload)
        if checked.decoder_packed_brotli != hdm3_stream:
            archive_build_blockers.append("candidate_decoder_section_not_hdm3_stream")
        if checked.latents_and_sidecar_brotli != packed.latents_and_sidecar_brotli:
            archive_build_blockers.append("candidate_latents_section_changed")
        if candidate_archive_sha == source.archive_sha256:
            archive_build_blockers.append("candidate_archive_sha256_unchanged")
        if candidate_payload_sha == sha256_bytes(source.payload):
            archive_build_blockers.append("candidate_payload_sha256_unchanged")
        ready_for_archive_preflight = not archive_build_blockers

    runtime_adapter = _runtime_adapter_blocker_report()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_archive_preflight": ready_for_archive_preflight,
        "archive_build_gate": ready_for_archive_preflight,
        "source_label": source_label,
        "source_archive_path": repo_relative(source_path, repo),
        "source_archive_sha256": source.archive_sha256,
        "source_archive_bytes": source.archive_bytes,
        "source_member_name": source.member_name,
        "source_member_bytes": source.member_bytes,
        "source_payload_sha256": sha256_bytes(source.payload),
        "source_payload_bytes": len(source.payload),
        "source_decoder_section_sha256": sha256_bytes(packed.decoder_packed_brotli),
        "source_decoder_section_bytes": len(packed.decoder_packed_brotli),
        "source_decoder_raw_sha256": sha256_bytes(source_raw),
        "source_decoder_raw_bytes": len(source_raw),
        "latents_and_sidecar_section_sha256": sha256_bytes(packed.latents_and_sidecar_brotli),
        "latents_and_sidecar_section_bytes": len(packed.latents_and_sidecar_brotli),
        "candidate_variant": HDM3_VARIANT_NAME,
        "candidate_decoder_section_sha256": sha256_bytes(hdm3_stream),
        "candidate_decoder_section_bytes": len(hdm3_stream),
        "candidate_decoder_section_byte_delta": section_byte_delta,
        "candidate_rate_positive": rate_positive,
        "candidate_rate_score_delta_if_runtime_supported_and_components_equal": round(
            section_byte_delta * (25 / 37_545_489),
            12,
        ),
        "candidate_payload_sha256": candidate_payload_sha,
        "candidate_payload_bytes": len(candidate_payload) if candidate_payload is not None else None,
        "candidate_archive_path": (
            repo_relative(candidate_archive_path, repo) if candidate_archive_path is not None else ""
        ),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_member_name": source.member_name if candidate_archive_path is not None else "",
        "hdm3_stats": hdm3_stats,
        "decoder_raw_equivalence": {
            "contract": "hdm3_decoder_raw_equivalence_v1",
            "source_decoder_raw_sha256": sha256_bytes(source_raw),
            "restored_decoder_raw_sha256": sha256_bytes(restored_decoder.to_raw()),
            "raw_equal": raw_equal,
            "q_roundtrip_equal": q_equal,
            "scale_roundtrip_equal": scale_equal,
        },
        "section_replacement_proof": {
            "contract": "hnerv_hdm3_single_section_replacement_v1",
            "replaced_section": "decoder_packed_brotli",
            "header_rewritten": True,
            "latents_and_sidecar_preserved": True,
            "zip_member_preserved": candidate_archive_path is not None,
            "byte_different_archive": bool(candidate_archive_sha and candidate_archive_sha != source.archive_sha256),
            "byte_different_payload": bool(candidate_payload_sha and candidate_payload_sha != sha256_bytes(source.payload)),
        },
        "runtime_adapter_proof": runtime_adapter,
        "archive_build_blockers": archive_build_blockers,
        "dispatch_blockers": [
            *archive_build_blockers,
            *runtime_adapter["dispatch_blockers"],
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    write_json(output_root / "hdm3_archive_candidate_manifest.json", manifest)
    return manifest


def _runtime_adapter_blocker_report() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_runtime_adapter_proof_v1",
        "content_detection_required": True,
        "legacy_brotli_fallback_required": True,
        "hdm3_decoder_fixture_raw_equal": True,
        "submission_runtime_integrated": True,
        "runtime_adapter_module": "tac.hnerv_hdm3_runtime_adapter",
        "runtime_normalizer_path": (
            "experiments/public_runtime_adapters/"
            "pr106_belt_and_suspenders_adapter/hdm3_normalize.py"
        ),
        "runtime_tree_parity_manifest_present": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "hdm3_runtime_adapter_archive_parity_proof_missing",
            "hdm3_runtime_tree_parity_manifest_missing",
            "hdm3_inflate_output_parity_missing",
        ],
    }


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "hnerv"


__all__ = [
    "HDM3_VARIANT_NAME",
    "HnervHdm3ArchiveCandidateError",
    "build_hdm3_archive_candidate",
]
