#!/usr/bin/env python3
"""Build a deterministic non-score HNGP archive wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_generated_schema_packet import (  # noqa: E402
    HNeRVGeneratedSchemaPacketError,
    build_hnerv_generated_schema_packet,
)

ARCHIVE_MANIFEST_SCHEMA = "tac_hnerv_generated_schema_candidate_archive.v1"
MEMBER_SUFFIX = ".hngp"
_SAFE_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class HNeRVGeneratedSchemaCandidateError(ValueError):
    """Raised when the generated-schema archive wrapper would be unsafe."""


def build_hnerv_generated_schema_candidate_archive(
    *,
    hngs_decoder: Path,
    latent_blob: Path,
    sidecar_blob: Path,
    output_archive: Path,
    manifest_output: Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Build one deterministic ZIP_STORED archive containing one HNGP packet."""

    member_name = _member_name_from_candidate_id(candidate_id)
    packet = build_hnerv_generated_schema_packet(
        hngs_decoder=hngs_decoder.read_bytes(),
        latent_blob=latent_blob.read_bytes(),
        sidecar_blob=sidecar_blob.read_bytes(),
        metadata={"candidate_id": candidate_id},
    )

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    _write_single_member_zip(
        output_archive,
        member_name=member_name,
        payload=packet.packet,
    )
    archive_bytes = output_archive.stat().st_size
    archive_sha256 = _sha256_file(output_archive)

    packet_manifest = packet.manifest
    manifest: dict[str, Any] = {
        "schema": ARCHIVE_MANIFEST_SCHEMA,
        "candidate_id": candidate_id,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "runtime_inflate_included": False,
        "omx_state_touched": False,
        "evidence_grade": "empirical_archive_wrapper_no_score",
        "archive": {
            "path": str(output_archive),
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "member_count": 1,
            "member_name": member_name,
            "member_bytes": packet_manifest["packet_bytes"],
            "member_sha256": packet_manifest["packet_sha256"],
            "compression": "ZIP_STORED",
        },
        "packet": {
            "grammar": packet_manifest["packet_grammar"],
            "bytes": packet_manifest["packet_bytes"],
            "sha256": packet_manifest["packet_sha256"],
            "manifest_schema": packet_manifest["schema"],
        },
        "sections": packet_manifest["sections"],
        "payload_sections": packet_manifest["payload_sections"],
        "packet_manifest": packet_manifest,
        "dispatch_blockers": [
            "non_score_archive_wrapper_no_runtime_inflate",
            "runtime_loader_not_wired",
            "inflate_output_parity_not_proven",
            "exact_cuda_auth_eval_not_run",
            "dispatch_claim_not_created_for_non_score_wrapper",
        ],
        "promotion_blockers": [
            "non_score_generated_schema_archive_wrapper",
            "runtime_parity_proof_missing",
            "contest_cuda_auth_eval_missing",
        ],
        "notes": [
            "This archive is a deterministic single-member HNGP wrapper only.",
            "It does not include inflate/runtime integration and is not dispatch-ready.",
            "No score, promotion, rank, kill, GPU, or state claim is made.",
        ],
    }

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hngs-decoder", type=Path, required=True)
    parser.add_argument("--latent-blob", type=Path, required=True)
    parser.add_argument("--sidecar-blob", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args(argv)

    try:
        manifest = build_hnerv_generated_schema_candidate_archive(
            hngs_decoder=args.hngs_decoder,
            latent_blob=args.latent_blob,
            sidecar_blob=args.sidecar_blob,
            output_archive=args.output_archive,
            manifest_output=args.manifest_output,
            candidate_id=args.candidate_id,
        )
    except (
        HNeRVGeneratedSchemaCandidateError,
        HNeRVGeneratedSchemaPacketError,
        OSError,
    ) as exc:
        raise SystemExit(f"HNGP archive wrapper build failed: {exc}") from None

    archive = manifest["archive"]
    print(
        f"wrote {archive['path']} "
        f"({archive['bytes']} bytes, sha256={archive['sha256']})"
    )
    print(
        f"member {archive['member_name']} "
        f"({archive['member_bytes']} bytes, sha256={archive['member_sha256']})"
    )
    print(
        f"manifest {args.manifest_output} "
        f"ready_for_exact_eval_dispatch={manifest['ready_for_exact_eval_dispatch']}"
    )
    return 0


def _member_name_from_candidate_id(candidate_id: str) -> str:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise HNeRVGeneratedSchemaCandidateError("candidate_id is required")
    if not _SAFE_CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise HNeRVGeneratedSchemaCandidateError(
            f"unsafe candidate_id for ZIP member name: {candidate_id!r}"
        )
    member_name = f"{candidate_id}{MEMBER_SUFFIX}"
    _validate_zip_member_name(member_name)
    return member_name


def _write_single_member_zip(path: Path, *, member_name: str, payload: bytes) -> None:
    _validate_zip_member_name(member_name)
    info = zipfile.ZipInfo(member_name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.internal_attr = 0
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _validate_zip_member_name(member_name: str) -> None:
    if not member_name:
        raise HNeRVGeneratedSchemaCandidateError("ZIP member name is empty")
    if member_name.startswith(("/", "\\")) or "\\" in member_name:
        raise HNeRVGeneratedSchemaCandidateError(
            f"unsafe ZIP member name: {member_name!r}"
        )
    parts = member_name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise HNeRVGeneratedSchemaCandidateError(
            f"unsafe ZIP member name: {member_name!r}"
        )
    if any(part.startswith(".") for part in parts):
        raise HNeRVGeneratedSchemaCandidateError(
            f"hidden ZIP member name rejected: {member_name!r}"
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
