"""Archive-candidate builder for PR106 HLM1 fixed-latent recodes."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import read_packed_archive_view
from tac.packet_compiler.pr106_fixed_latent_recode import (
    HLM1_MAGIC,
    encode_hlm1_fixed_latents_from_brotli,
)
from tac.repo_io import repo_relative, sha256_bytes, sha256_file, write_json

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_hlm1_archive_candidate.build_hlm1_latent_archive_candidate"
LANE_ID = "hnerv_hlm1_fixed_latent_recode_exact_eval"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489


class HnervHlm1ArchiveCandidateError(ValueError):
    """Raised when an HLM1 archive candidate cannot be built safely."""


def build_hlm1_latent_archive_candidate(
    *,
    source_archive: str | Path,
    output_dir: str | Path,
    source_label: str,
    candidate_member_name: str | None = None,
    allow_rate_regression: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a byte-closed archive candidate replacing only fixed latents.

    The candidate is archive-byte materialized and parser-consumption proven,
    but it is never exact-score authority. Exact CUDA auth eval and terminal
    lane-claim custody remain separate required gates.
    """
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    source_path = Path(source_archive)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    view = read_packed_archive_view(source_path)
    packed = view.packed
    emitted_member_name = candidate_member_name or view.archive.member_name
    if emitted_member_name not in {"0.bin", "x"}:
        raise HnervHlm1ArchiveCandidateError(
            f"unsupported HLM1 candidate member name: {emitted_member_name!r}"
        )
    source_is_hlm1 = packed.latents_and_sidecar_brotli.startswith(HLM1_MAGIC)
    member_name_changed = emitted_member_name != view.archive.member_name
    recode = encode_hlm1_fixed_latents_from_brotli(packed.latents_and_sidecar_brotli)
    blockers: list[str] = []
    if recode.source_raw_sha256 != recode.decoded_raw_sha256:
        blockers.append("hlm1_fixed_latent_raw_roundtrip_mismatch")

    candidate_archive_path: Path | None = None
    candidate_payload: bytes | None = None
    candidate_archive_sha = ""
    candidate_archive_bytes: int | None = None
    candidate_payload_sha = ""
    ready_for_archive_preflight = False
    if not blockers:
        candidate_latent_section = (
            packed.latents_and_sidecar_brotli
            if source_is_hlm1 and member_name_changed
            else recode.payload
        )
        candidate_packed = dataclasses.replace(
            packed,
            latents_and_sidecar_brotli=candidate_latent_section,
        )
        candidate_payload = view.emit_payload(candidate_packed)
        candidate_archive_path = output_root / f"{_slug(source_label)}_hlm1_latent_candidate.zip"
        view.write_archive(
            candidate_archive_path,
            candidate_payload,
            member_name=emitted_member_name,
        )
        candidate_archive_sha = sha256_file(candidate_archive_path)
        candidate_archive_bytes = candidate_archive_path.stat().st_size
        candidate_payload_sha = sha256_bytes(candidate_payload)
        checked_view = read_packed_archive_view(candidate_archive_path)
        if checked_view.packed.decoder_packed_brotli != packed.decoder_packed_brotli:
            blockers.append("candidate_decoder_section_changed")
        if checked_view.packed.latents_and_sidecar_brotli != candidate_latent_section:
            blockers.append("candidate_hlm1_latent_section_not_preserved")
        if not checked_view.packed.latents_and_sidecar_brotli.startswith(HLM1_MAGIC):
            blockers.append("candidate_hlm1_magic_missing")
        if checked_view.archive.member_name != emitted_member_name:
            blockers.append("candidate_member_name_mismatch")
        if checked_view.archive.archive_sha256 == view.archive.archive_sha256:
            blockers.append("candidate_archive_sha256_unchanged")
        if not recode.rate_positive and not member_name_changed and not allow_rate_regression:
            blockers.append("hlm1_fixed_latent_section_not_rate_positive")
        if (
            candidate_archive_bytes is not None
            and int(candidate_archive_bytes) >= int(view.archive.archive_bytes)
            and not allow_rate_regression
        ):
            blockers.append("candidate_archive_not_rate_positive")
        ready_for_archive_preflight = not blockers

    byte_delta = (
        int(candidate_archive_bytes) - int(view.archive.archive_bytes)
        if candidate_archive_bytes is not None
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "contract": "hnerv_hlm1_latent_archive_candidate_v1",
        "candidate_id": (
            "pr106_hlm1_xmember_rate_repack"
            if member_name_changed
            else "pr106_hlm1_fixed_latent_recode"
        ),
        "lane_id": LANE_ID,
        "family": "hnerv_fixed_latent_recode",
        "pareto_scope": "hnerv_rate_only_exact_archive",
        "candidate_transform_kind": (
            "zip_member_rate_only_repack"
            if member_name_changed and source_is_hlm1
            else "hlm1_fixed_latent_recode"
        ),
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": ready_for_archive_preflight,
        "ready_for_exact_eval_dispatch": False,
        "source_label": source_label,
        "source_archive_path": repo_relative(source_path, repo),
        "source_archive_sha256": view.archive.archive_sha256,
        "source_archive_bytes": view.archive.archive_bytes,
        "source_member_name": view.archive.member_name,
        "source_payload_kind": view.payload_kind,
        "source_payload_sha256": sha256_bytes(view.archive.payload),
        "source_payload_bytes": len(view.archive.payload),
        "source_latents_section_is_hlm1": source_is_hlm1,
        "source_decoder_section_sha256": sha256_bytes(packed.decoder_packed_brotli),
        "source_decoder_section_bytes": len(packed.decoder_packed_brotli),
        "source_latents_section_sha256": sha256_bytes(packed.latents_and_sidecar_brotli),
        "source_latents_section_bytes": len(packed.latents_and_sidecar_brotli),
        "candidate_archive_path": (
            repo_relative(candidate_archive_path, repo)
            if candidate_archive_path is not None
            else ""
        ),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_payload_sha256": candidate_payload_sha,
        "candidate_payload_bytes": len(candidate_payload) if candidate_payload is not None else None,
        "candidate_member_name": emitted_member_name if candidate_archive_path is not None else "",
        "candidate_member_name_changed": member_name_changed,
        "candidate_latents_section_sha256": sha256_bytes(
            packed.latents_and_sidecar_brotli
            if source_is_hlm1 and member_name_changed
            else recode.payload
        ),
        "candidate_latents_section_bytes": len(
            packed.latents_and_sidecar_brotli
            if source_is_hlm1 and member_name_changed
            else recode.payload
        ),
        "candidate_latents_section_byte_delta": (
            0 if source_is_hlm1 and member_name_changed else recode.byte_delta
        ),
        "member_payload_sha256_unchanged": (
            candidate_payload_sha == sha256_bytes(view.archive.payload)
            if candidate_payload is not None
            else False
        ),
        "member_payload_bytes_unchanged": (
            len(candidate_payload) == len(view.archive.payload)
            if candidate_payload is not None
            else False
        ),
        "candidate_archive_byte_delta": byte_delta,
        "candidate_rate_score_delta_if_components_equal": (
            byte_delta * RATE_SCORE_PER_BYTE if byte_delta is not None else None
        ),
        "decoder_section_preserved": candidate_payload is not None and not blockers,
        "zip_member_repack": {
            "enabled": member_name_changed,
            "source_member_name": view.archive.member_name,
            "candidate_member_name": emitted_member_name,
            "payload_sha256_unchanged": (
                candidate_payload_sha == sha256_bytes(view.archive.payload)
                if candidate_payload is not None
                else False
            ),
            "payload_bytes_unchanged": (
                len(candidate_payload) == len(view.archive.payload)
                if candidate_payload is not None
                else False
            ),
        },
        "hlm1_recode": recode.to_manifest(),
        "archive_build_blockers": blockers,
        "dispatch_blockers": [
            "static_release_surface_missing",
            "runtime_tree_custody_refresh_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    write_json(output_root / "hlm1_latent_archive_candidate_manifest.json", manifest)
    return manifest


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part) or "source"
