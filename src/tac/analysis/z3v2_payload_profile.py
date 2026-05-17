# SPDX-License-Identifier: MIT
"""Z3HV2 payload authority profiler.

This module classifies the bytes shipped by the Z3 v2 latent-replacement
archive grammar. It is intentionally analysis-only: no scorer loads, no device
selection, no dispatch, and no score claims.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_LATENT_BLOB_LEN,
    Z3HV2_HEADER_STRUCT,
    Z3HV2_PER_DIM_AFFINE_LEN,
    decode_z3hv2_section,
    split_z3v2_payload_bytes,
)

_SCHEMA = "z3v2_payload_profile_v1"


class Z3V2PayloadProfileError(ValueError):
    """Raised when an archive cannot be interpreted as a Z3HV2 payload."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_single_member_archive(archive_path: Path) -> tuple[str, bytes]:
    with ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise Z3V2PayloadProfileError(
                f"expected exactly one ZIP member, found {len(names)}: {names}"
            )
        name = names[0]
        return name, zf.read(name)


def _raw_z3hv2_blob_lengths(section: bytes) -> dict[str, int]:
    """Return compressed blob lengths without trusting decompressed payloads."""
    if len(section) < Z3HV2_HEADER_STRUCT.size:
        raise Z3V2PayloadProfileError("Z3HV2 section too short for header")
    pos = Z3HV2_HEADER_STRUCT.size
    if pos + 2 > len(section):
        raise Z3V2PayloadProfileError("Z3HV2 section truncated before weights length")
    (weights_blob_len,) = struct.unpack_from("<H", section, pos)
    pos += 2 + weights_blob_len
    if pos + 4 > len(section):
        raise Z3V2PayloadProfileError("Z3HV2 section truncated before w_hat length")
    (w_hat_blob_len,) = struct.unpack_from("<I", section, pos)
    pos += 4 + w_hat_blob_len
    if pos + 4 > len(section):
        raise Z3V2PayloadProfileError("Z3HV2 section truncated before residual length")
    (residual_blob_len,) = struct.unpack_from("<I", section, pos)
    pos += 4 + residual_blob_len
    if pos + Z3HV2_PER_DIM_AFFINE_LEN > len(section):
        raise Z3V2PayloadProfileError("Z3HV2 section truncated before affine payload")
    pos += Z3HV2_PER_DIM_AFFINE_LEN
    return {
        "weights_blob_compressed_bytes": int(weights_blob_len),
        "w_hat_blob_compressed_bytes": int(w_hat_blob_len),
        "residual_blob_compressed_bytes": int(residual_blob_len),
        "section_total_bytes_from_lengths": int(pos),
    }


def profile_z3v2_archive(archive_path: Path) -> dict[str, Any]:
    """Build a fail-closed authority profile for a Z3HV2 archive.zip.

    The profile deliberately distinguishes the current direct-residual control
    from a true Ballé entropy-coded residual. Current Z3HV2 inflate computes
    sigma from optional hyperprior side-info but does not use sigma to entropy-
    decode the residual; the residual is already a brotli-decompressed int8
    stream. That means the profile stays non-promotional even when side-info
    slots are non-empty.
    """
    archive_path = archive_path.resolve()
    member_name, payload = _read_single_member_archive(archive_path)
    try:
        decoder_section, z3hv2_section, sidecar_section = split_z3v2_payload_bytes(
            payload
        )
        (
            meta,
            weights_int8,
            w_hat_int8,
            residual_int8,
            _latent_offset,
            _latent_scale,
            decoded_section_total,
        ) = decode_z3hv2_section(z3hv2_section)
    except Exception as exc:  # pragma: no cover - exercised through caller errors
        raise Z3V2PayloadProfileError(
            f"{archive_path} is not a valid Z3HV2 payload: {exc}"
        ) from exc

    raw_lengths = _raw_z3hv2_blob_lengths(z3hv2_section)
    if raw_lengths["section_total_bytes_from_lengths"] != decoded_section_total:
        raise Z3V2PayloadProfileError(
            "Z3HV2 section length mismatch between raw lengths and decoder"
        )

    weights_present = len(weights_int8) > 0
    w_hat_present = len(w_hat_int8) > 0
    direct_residual_control = not weights_present and not w_hat_present
    section_delta = A1_LATENT_BLOB_LEN - len(z3hv2_section)

    result_review_blockers: list[str] = [
        "score_claim_requires_paired_contest_cpu_cuda_auth_eval",
        "result_review_required_before_promotion",
        "current_z3hv2_runtime_has_no_active_balle_entropy_residual_decoder",
    ]
    if direct_residual_control:
        result_review_blockers.append("hyperprior_weights_and_w_hat_slots_empty")
    if section_delta <= 0:
        result_review_blockers.append("z3hv2_section_not_smaller_than_a1_latent_blob")

    classification = (
        "direct_residual_control"
        if direct_residual_control
        else "hyperprior_sideinfo_present_but_residual_still_direct_brotli_int8"
    )

    return {
        "schema": _SCHEMA,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_bytes(archive_path.read_bytes()),
        "member_name": member_name,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
        "decoder_section_bytes": len(decoder_section),
        "z3hv2_section_bytes": len(z3hv2_section),
        "sidecar_section_bytes": len(sidecar_section),
        "a1_latent_blob_bytes_replaced": A1_LATENT_BLOB_LEN,
        "byte_savings_signed": int(section_delta),
        "byte_saving": section_delta > 0,
        "classification": classification,
        "direct_residual_control": direct_residual_control,
        "weights_int8_bytes": len(weights_int8),
        "w_hat_int8_bytes": len(w_hat_int8),
        "residual_int8_bytes": len(residual_int8),
        "raw_blob_lengths": raw_lengths,
        "meta": {
            "n_pairs": meta.n_pairs,
            "hyper_dim": meta.hyper_dim,
            "latent_dim": meta.latent_dim,
            "quant_step": meta.quant_step,
            "min_sigma": meta.min_sigma,
            "max_sigma": meta.max_sigma,
            "factorized_half_range": meta.factorized_half_range,
        },
        "residual_coding": "brotli_direct_int8_residual",
        "balle_entropy_residual_decoder_active": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": result_review_blockers,
    }


def render_markdown(profile: dict[str, Any]) -> str:
    """Render a compact operator-facing profile."""
    blockers = "\n".join(f"- `{b}`" for b in profile["result_review_blockers"])
    return (
        "# Z3HV2 Payload Authority Profile\n\n"
        f"- schema: `{profile['schema']}`\n"
        f"- archive: `{profile['archive_path']}`\n"
        f"- archive bytes: `{profile['archive_bytes']}`\n"
        f"- archive sha256: `{profile['archive_sha256']}`\n"
        f"- member: `{profile['member_name']}` ({profile['payload_bytes']} B)\n"
        f"- classification: `{profile['classification']}`\n"
        f"- Z3HV2 section bytes: `{profile['z3hv2_section_bytes']}`\n"
        f"- A1 latent bytes replaced: `{profile['a1_latent_blob_bytes_replaced']}`\n"
        f"- signed byte savings: `{profile['byte_savings_signed']}`\n"
        f"- residual coding: `{profile['residual_coding']}`\n"
        f"- Ballé entropy residual decoder active: "
        f"`{profile['balle_entropy_residual_decoder_active']}`\n"
        f"- score_claim: `{profile['score_claim']}`\n"
        f"- promotion_eligible: `{profile['promotion_eligible']}`\n"
        f"- ready_for_exact_eval_dispatch: "
        f"`{profile['ready_for_exact_eval_dispatch']}`\n\n"
        "## Blockers\n\n"
        f"{blockers}\n"
    )


def write_profile_outputs(
    profile: dict[str, Any],
    *,
    json_out: Path,
    markdown_out: Path | None = None,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(profile))
