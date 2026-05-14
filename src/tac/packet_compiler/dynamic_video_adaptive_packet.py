# SPDX-License-Identifier: MIT
"""Packet-manifest contracts for DVAR1 dynamic video adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "dynamic_packet_manifest_v1"
FAMILY = "dynamic_video_adaptive_receiver"
TARGET_PROFILES = (
    "contest_one_video_replay",
    "contest_generalized",
    "production_generalized",
    "production_edge_adaptive",
)
_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


class DynamicVideoAdaptivePacketError(ValueError):
    """Raised when a dynamic packet manifest would be non-compliant."""


def _sha256_hex(value: str, *, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in _SHA256_HEX_CHARS for ch in digest):
        raise DynamicVideoAdaptivePacketError(
            f"{field_name} must be a 64-character hex sha256"
        )
    return digest


def _section_manifest(
    parser_section_manifest: Mapping[str, Any],
    consumed_sections: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(parser_section_manifest, Mapping) or not parser_section_manifest:
        raise DynamicVideoAdaptivePacketError("parser_section_manifest must be non-empty")
    out: dict[str, dict[str, Any]] = {}
    for section in consumed_sections:
        if section not in parser_section_manifest:
            raise DynamicVideoAdaptivePacketError(
                f"consumed section {section!r} missing from parser_section_manifest"
            )
        row = parser_section_manifest[section]
        if not isinstance(row, Mapping):
            raise DynamicVideoAdaptivePacketError(
                f"parser_section_manifest[{section!r}] must be a mapping"
            )
        try:
            start = int(row["start"])
            length = int(row["length"])
        except KeyError as exc:
            raise DynamicVideoAdaptivePacketError(
                f"parser_section_manifest[{section!r}] requires start and length"
            ) from exc
        if start < 0 or length <= 0:
            raise DynamicVideoAdaptivePacketError(
                f"parser_section_manifest[{section!r}] has invalid start/length"
            )
        section_sha = _sha256_hex(str(row.get("sha256", "")), field_name=f"{section}.sha256")
        normalized = dict(row)
        normalized.update({"start": start, "length": length, "sha256": section_sha})
        out[section] = normalized
    return out


def _runtime_consumption_proof(
    runtime_consumption_proof: Mapping[str, Any],
    consumed_sections: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(runtime_consumption_proof, Mapping):
        raise DynamicVideoAdaptivePacketError("runtime_consumption_proof must be a mapping")
    proof_sections = runtime_consumption_proof.get("consumed_sections")
    if not isinstance(proof_sections, Sequence) or isinstance(proof_sections, (str, bytes)):
        raise DynamicVideoAdaptivePacketError(
            "runtime_consumption_proof.consumed_sections must be a sequence"
        )
    missing = sorted(set(consumed_sections) - {str(section) for section in proof_sections})
    if missing:
        raise DynamicVideoAdaptivePacketError(
            "runtime_consumption_proof missing consumed sections: " + ", ".join(missing)
        )
    if bool(runtime_consumption_proof.get("scorer_loaded_at_inflate", False)):
        raise DynamicVideoAdaptivePacketError("runtime proof cannot load scorer at inflate")
    proof_artifacts = runtime_consumption_proof.get("proof_artifacts")
    if not isinstance(proof_artifacts, Sequence) or isinstance(proof_artifacts, (str, bytes)):
        raise DynamicVideoAdaptivePacketError(
            "runtime_consumption_proof.proof_artifacts must be a non-empty sequence"
        )
    if not proof_artifacts:
        raise DynamicVideoAdaptivePacketError(
            "runtime_consumption_proof.proof_artifacts must be non-empty"
        )
    inflate_entrypoint = str(runtime_consumption_proof.get("inflate_entrypoint", "")).strip()
    if not inflate_entrypoint:
        raise DynamicVideoAdaptivePacketError(
            "runtime_consumption_proof.inflate_entrypoint must be non-empty"
        )
    return {
        **dict(runtime_consumption_proof),
        "consumed_sections": [str(section) for section in proof_sections],
        "proof_artifacts": [str(path) for path in proof_artifacts],
        "scorer_loaded_at_inflate": False,
        "inflate_entrypoint": inflate_entrypoint,
    }


def build_dynamic_packet_manifest(
    *,
    target_profile: str,
    baseline_archive_sha256: str,
    candidate_archive_sha256: str,
    baseline_archive_bytes: int,
    candidate_archive_bytes: int,
    runtime_tree_sha256: str,
    parser_section_manifest: Mapping[str, Any],
    consumed_sections: Sequence[str],
    runtime_consumption_proof: Mapping[str, Any],
    no_op_proof_inputs: Sequence[str],
    atom_ledger_ref: str,
    telemetry_ref: str,
    reproduction_commands: Sequence[Sequence[str] | str] = (),
) -> dict[str, Any]:
    """Build a scorer-free packet custody manifest.

    The manifest is promotion-blocked by design. Exact eval and the
    pre-submission compliance gate must attach score authority later.
    """

    if target_profile not in TARGET_PROFILES:
        raise DynamicVideoAdaptivePacketError(
            f"target_profile {target_profile!r} not in {TARGET_PROFILES!r}"
        )
    if baseline_archive_bytes <= 0 or candidate_archive_bytes <= 0:
        raise DynamicVideoAdaptivePacketError("archive byte sizes must be positive")
    baseline_archive_sha = _sha256_hex(
        baseline_archive_sha256,
        field_name="baseline_archive_sha256",
    )
    candidate_archive_sha = _sha256_hex(
        candidate_archive_sha256,
        field_name="candidate_archive_sha256",
    )
    runtime_tree_sha = _sha256_hex(
        runtime_tree_sha256,
        field_name="runtime_tree_sha256",
    )
    if not consumed_sections:
        raise DynamicVideoAdaptivePacketError("consumed_sections must be non-empty")
    consumed_sections_list = [str(section) for section in consumed_sections]
    parser_manifest = _section_manifest(parser_section_manifest, consumed_sections_list)
    runtime_proof = _runtime_consumption_proof(
        runtime_consumption_proof,
        consumed_sections_list,
    )
    if not no_op_proof_inputs:
        raise DynamicVideoAdaptivePacketError("no_op_proof_inputs must be non-empty")

    normalized_commands: list[list[str]] = []
    for command in reproduction_commands:
        if isinstance(command, str):
            normalized_commands.append([command])
        else:
            normalized_commands.append([str(part) for part in command])

    contest_candidate = target_profile.startswith("contest_")
    return {
        "schema": SCHEMA,
        "family": FAMILY,
        "target_profile": target_profile,
        "contest_dispatch_candidate": contest_candidate,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "inflate_allowed": True,
        "scorer_allowed_at_inflate": False,
        "baseline_archive_sha256": baseline_archive_sha,
        "candidate_archive_sha256": candidate_archive_sha,
        "baseline_archive_bytes": int(baseline_archive_bytes),
        "candidate_archive_bytes": int(candidate_archive_bytes),
        "byte_delta": int(candidate_archive_bytes) - int(baseline_archive_bytes),
        "runtime_tree_sha256": runtime_tree_sha,
        "parser_section_manifest": parser_manifest,
        "consumed_sections": consumed_sections_list,
        "runtime_consumption_proof": runtime_proof,
        "no_op_proof_inputs": list(no_op_proof_inputs),
        "atom_ledger_ref": atom_ledger_ref,
        "telemetry_ref": telemetry_ref,
        "reproduction_commands": normalized_commands,
        "dispatch_blockers": [
            "requires_lane_dispatch_claim",
            "requires_exact_eval_before_score_claim",
            "requires_pre_submission_compliance_check",
        ],
    }


__all__ = [
    "DynamicVideoAdaptivePacketError",
    "FAMILY",
    "SCHEMA",
    "TARGET_PROFILES",
    "build_dynamic_packet_manifest",
]
