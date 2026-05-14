# SPDX-License-Identifier: MIT
"""Bridge materialized CodecOp bytes into monolithic section replacements.

This module deliberately emits the existing ``--replacement-manifest`` shape
consumed by ``tools/build_monolithic_stack_candidate.py``. It does not build an
archive, infer missing bytes from planner rows, or mark anything dispatchable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import brotli

from tac.frontier_archive_layout import inspect_frontier_archive_layout

REPLACEMENT_MANIFEST_SCHEMA = "tac_monolithic_codec_op_replacement_manifest_v1"
SUPPORTED_GRAMMARS = frozenset(
    {
        "pr106_ff_packed_hnerv",
        "pr101_fixed_offset_hnerv_microcodec",
    }
)
PR101_FIXED_LENGTH_SECTIONS = frozenset({"decoder_blob", "latent_blob"})
PR106_BROTLI_SECTIONS = frozenset(
    {"decoder_packed_brotli", "latents_and_sidecar_brotli"}
)
KNOWN_CODEC_ENVELOPES = (b"COBM1", b"CPL1", b"JCSP", b"JCSK")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class MonolithicCodecOpReplacementError(ValueError):
    """Raised when a materialized payload is not safe as a section replacement."""


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest for ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest for ``path``."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_monolithic_codec_op_replacement_manifest(
    *,
    source_archive: Path,
    target_section: str,
    replacement_payload: Path,
    output_replacement_manifest: Path,
    candidate_id: str,
    section_payload_contract: str = "raw_section_bytes",
    evidence_json: Path | None = None,
    expected_source_archive_sha256: str | None = None,
    expected_source_archive_bytes: int | None = None,
) -> dict[str, Any]:
    """Emit a byte-closed replacement manifest for a parser-proven section."""

    if not candidate_id:
        raise MonolithicCodecOpReplacementError("candidate_id is required")
    if not target_section:
        raise MonolithicCodecOpReplacementError("target_section is required")

    source_archive = Path(source_archive)
    replacement_payload = Path(replacement_payload)
    output_replacement_manifest = Path(output_replacement_manifest)
    source_bytes = source_archive.stat().st_size
    source_sha = sha256_file(source_archive)
    if expected_source_archive_bytes is not None and source_bytes != expected_source_archive_bytes:
        raise MonolithicCodecOpReplacementError(
            f"source archive bytes mismatch: {source_bytes} != {expected_source_archive_bytes}"
        )
    if (
        expected_source_archive_sha256 is not None
        and source_sha.lower() != expected_source_archive_sha256.lower()
    ):
        raise MonolithicCodecOpReplacementError(
            f"source archive sha256 mismatch: {source_sha} != {expected_source_archive_sha256}"
        )

    layout = inspect_frontier_archive_layout(source_archive)
    logical = layout.get("logical_layout")
    if not isinstance(logical, dict):
        raise MonolithicCodecOpReplacementError(
            "source archive has no parser-proven logical layout"
        )
    grammar = str(logical.get("grammar"))
    if grammar not in SUPPORTED_GRAMMARS:
        raise MonolithicCodecOpReplacementError(
            f"unsupported monolithic grammar for CodecOp replacement: {grammar}"
        )
    if target_section == "ff_header":
        raise MonolithicCodecOpReplacementError(
            "ff_header is derived and cannot be a materialized replacement"
        )

    section = _find_section(logical, target_section)
    old_bytes = int(section["len"])
    old_sha = str(section["sha256"])
    new_payload = replacement_payload.read_bytes()
    new_bytes = len(new_payload)
    new_sha = sha256_bytes(new_payload)
    if new_sha.lower() == old_sha.lower() and new_bytes == old_bytes:
        raise MonolithicCodecOpReplacementError(
            f"no-op replacement rejected for {target_section}"
        )
    _validate_section_payload_contract(
        target_section=target_section,
        grammar=grammar,
        section_payload_contract=section_payload_contract,
        payload=new_payload,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
    )

    evidence_summary = _evidence_summary(
        evidence_json,
        replacement_bytes=new_bytes,
        replacement_sha256=new_sha,
    )
    replacement_record = {
        "section_name": target_section,
        "replacement_path": _path_for_manifest(
            replacement_payload,
            base_dir=output_replacement_manifest.parent,
        ),
        "expected_old_sha256": old_sha,
        "expected_old_bytes": old_bytes,
        "expected_new_sha256": new_sha,
        "expected_new_bytes": new_bytes,
    }
    manifest: dict[str, Any] = {
        "schema": REPLACEMENT_MANIFEST_SCHEMA,
        "candidate_id": candidate_id,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "empirical_materialized_payload_no_score",
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_bytes,
            "sha256": source_sha,
        },
        "monolithic_layout": {
            "grammar": grammar,
            "member_name": logical.get("single_member_name"),
            "parser_proof_strength": logical.get("parser_proof_strength"),
        },
        "target_section": {
            "name": target_section,
            "role": section.get("role"),
            "offset": int(section["offset"]),
            "old_bytes": old_bytes,
            "old_sha256": old_sha,
        },
        "replacement_payload": {
            "path": str(replacement_payload),
            "bytes": new_bytes,
            "sha256": new_sha,
            "section_payload_contract": section_payload_contract,
            "byte_delta": new_bytes - old_bytes,
        },
        "evidence_json": evidence_summary,
        "replacements": [replacement_record],
        "dispatch_blockers": [
            "replacement_manifest_only",
            "monolithic_candidate_archive_not_built",
            "runtime_consumption_proof_missing",
            "active_lane_claim_missing",
            "contest_cuda_auth_eval_missing",
        ],
        "promotion_blockers": ["contest_cuda_auth_eval_missing"],
        "notes": [
            "This manifest only binds materialized bytes to a parser-proven section.",
            "Build the candidate archive with tools/build_monolithic_stack_candidate.py.",
            "No score, dispatch, promotion, rank, or kill claim is implied.",
        ],
    }

    output_replacement_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_replacement_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _find_section(logical: dict[str, Any], target_section: str) -> dict[str, Any]:
    sections = logical.get("sections")
    if not isinstance(sections, list):
        raise MonolithicCodecOpReplacementError("logical layout has no sections")
    for section in sections:
        if isinstance(section, dict) and section.get("name") == target_section:
            return section
    raise MonolithicCodecOpReplacementError(
        f"unknown parser-proven section: {target_section}"
    )


def _validate_section_payload_contract(
    *,
    target_section: str,
    grammar: str,
    section_payload_contract: str,
    payload: bytes,
    old_bytes: int,
    new_bytes: int,
) -> None:
    if grammar == "pr101_fixed_offset_hnerv_microcodec" and (
        target_section in PR101_FIXED_LENGTH_SECTIONS and new_bytes != old_bytes
    ):
        raise MonolithicCodecOpReplacementError(
            "PR101 fixed-offset grammar only permits equal-length replacement "
            f"for {target_section}; got {new_bytes} vs {old_bytes}"
        )

    if section_payload_contract == "raw_section_bytes":
        if payload.startswith(KNOWN_CODEC_ENVELOPES):
            raise MonolithicCodecOpReplacementError(
                "replacement payload looks like a CodecOp/JCS envelope, not a "
                "runtime-consumed raw section"
            )
        return

    if section_payload_contract == "pr106_brotli_section":
        if grammar != "pr106_ff_packed_hnerv" or target_section not in PR106_BROTLI_SECTIONS:
            raise MonolithicCodecOpReplacementError(
                "pr106_brotli_section contract requires a PR106 Brotli payload section"
            )
        _require_brotli_payload(payload, section_payload_contract)
        return

    if section_payload_contract == "pr106_decoder_packed_brotli":
        if grammar != "pr106_ff_packed_hnerv" or target_section != "decoder_packed_brotli":
            raise MonolithicCodecOpReplacementError(
                "pr106_decoder_packed_brotli contract requires decoder_packed_brotli"
            )
        _require_brotli_payload(payload, section_payload_contract)
        return

    if section_payload_contract == "pr106_latents_and_sidecar_brotli":
        if (
            grammar != "pr106_ff_packed_hnerv"
            or target_section != "latents_and_sidecar_brotli"
        ):
            raise MonolithicCodecOpReplacementError(
                "pr106_latents_and_sidecar_brotli contract requires "
                "latents_and_sidecar_brotli"
            )
        _require_brotli_payload(payload, section_payload_contract)
        return

    raise MonolithicCodecOpReplacementError(
        f"unknown section_payload_contract: {section_payload_contract}"
    )


def _require_brotli_payload(payload: bytes, contract: str) -> None:
    try:
        brotli.decompress(payload)
    except brotli.error as exc:
        raise MonolithicCodecOpReplacementError(
            f"{contract} payload does not Brotli-decompress"
        ) from exc


def _evidence_summary(
    evidence_json: Path | None,
    *,
    replacement_bytes: int,
    replacement_sha256: str,
) -> dict[str, Any]:
    if evidence_json is None:
        return {"declared": False}
    path = Path(evidence_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MonolithicCodecOpReplacementError("evidence_json must contain an object")

    schema = payload.get("schema", payload.get("schema_version"))
    expected_bytes = _first_int(payload, ("bytes_out", "final_bytes_out", "payload_bytes"))
    if expected_bytes is None:
        expected_bytes = _first_int(
            payload,
            ("materialized_payload_bytes", "replacement_payload_bytes", "blob_bytes"),
        )
    expected_sha = _first_sha(
        payload,
        ("blob_sha256", "final_blob_sha256", "payload_sha256", "sha256"),
    )
    if expected_sha is None:
        expected_sha = _first_sha(
            payload,
            (
                "materialized_payload_sha256",
                "replacement_payload_sha256",
                "charged_byte_blob_sha256",
            ),
        )
    blockers: list[str] = []
    if expected_bytes is not None and expected_bytes != replacement_bytes:
        blockers.append("evidence_bytes_out_mismatch")
    if expected_sha is not None and expected_sha.lower() != replacement_sha256.lower():
        blockers.append("evidence_blob_sha256_mismatch")
    if blockers:
        raise MonolithicCodecOpReplacementError(
            "evidence_json does not match replacement payload: " + ", ".join(blockers)
        )
    return {
        "declared": True,
        "path": str(path),
        "schema": schema,
        "bytes_bound": expected_bytes == replacement_bytes if expected_bytes is not None else None,
        "sha256_bound": (
            expected_sha.lower() == replacement_sha256.lower()
            if expected_sha is not None
            else None
        ),
        "planning_only": payload.get("ready_for_exact_eval_dispatch") is False
        or payload.get("dispatchable") is False
        or payload.get("score_claim") is False,
    }


def _first_int(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _first_sha(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and _SHA256_RE.fullmatch(value):
            return value
    return None


def _path_for_manifest(path: Path, *, base_dir: Path) -> str:
    path = Path(path)
    try:
        rel = os.path.relpath(path.resolve(strict=False), base_dir.resolve(strict=False))
    except OSError:
        return str(path)
    if rel.startswith("..") or rel == os.curdir or os.path.isabs(rel):
        return str(path)
    return rel


__all__ = [
    "REPLACEMENT_MANIFEST_SCHEMA",
    "MonolithicCodecOpReplacementError",
    "build_monolithic_codec_op_replacement_manifest",
    "sha256_bytes",
    "sha256_file",
]
