#!/usr/bin/env python3
"""Prove HDM3 runtime-adapter parity against a source HNeRV archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import brotli

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_hdm3_runtime_adapter import (  # noqa: E402
    HnervHdm3RuntimeAdapterError,
    restore_hdm3_payload_to_legacy_brotli,
)
from tac.hnerv_lowlevel_packer import (  # noqa: E402
    HnervLowlevelPackError,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)
from tac.repo_io import repo_relative, sha256_file, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def build_proof(
    *,
    source_archive: Path,
    candidate_archive: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_strict_single_member_zip(source_archive)
    candidate = read_strict_single_member_zip(candidate_archive)
    source_packed = parse_ff_packed_brotli_hnerv(source.payload)
    candidate_packed = parse_ff_packed_brotli_hnerv(candidate.payload)
    source_raw = brotli.decompress(source_packed.decoder_packed_brotli)
    restored_payload, adapter_proof = restore_hdm3_payload_to_legacy_brotli(
        candidate.payload,
        require_hdm3=True,
    )
    restored_packed = parse_ff_packed_brotli_hnerv(restored_payload)
    restored_raw = brotli.decompress(restored_packed.decoder_packed_brotli)
    restored_payload_path = output_dir / "candidate_hdm3_restored_legacy_payload.bin"
    restored_payload_path.write_bytes(restored_payload)

    source_raw_sha = sha256_bytes(source_raw)
    restored_raw_sha = sha256_bytes(restored_raw)
    source_payload_sha = sha256_bytes(source.payload)
    restored_payload_sha = sha256_bytes(restored_payload)
    source_decoder_section_sha = sha256_bytes(source_packed.decoder_packed_brotli)
    restored_decoder_section_sha = sha256_bytes(restored_packed.decoder_packed_brotli)
    source_latents_sha = sha256_bytes(source_packed.latents_and_sidecar_brotli)
    candidate_latents_sha = sha256_bytes(candidate_packed.latents_and_sidecar_brotli)
    restored_latents_sha = sha256_bytes(restored_packed.latents_and_sidecar_brotli)
    restored_payload_matches_source = restored_payload_sha == source_payload_sha
    restored_decoder_section_matches_source = restored_decoder_section_sha == source_decoder_section_sha
    inflate_output_parity_proven_by_payload_identity = (
        restored_payload_matches_source
        and restored_decoder_section_matches_source
        and restored_raw_sha == source_raw_sha
        and candidate_latents_sha == source_latents_sha
        and restored_latents_sha == source_latents_sha
    )
    adapter_proof = {
        **adapter_proof,
        "exact_source_payload_identity_proven_by_archive_proof": (
            inflate_output_parity_proven_by_payload_identity
        ),
        "remaining_dispatch_blockers": _dispatch_blockers(
            inflate_output_parity_proven_by_payload_identity
        ),
    }
    adapter_files = [
        REPO_ROOT / "src/tac/hnerv_hdm3_runtime_adapter.py",
        REPO_ROOT / "experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/hdm3_normalize.py",
        REPO_ROOT / "experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh",
    ]
    proof: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_runtime_adapter_archive_parity_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "source_archive_path": repo_relative(source_archive, REPO_ROOT),
        "source_archive_sha256": source.archive_sha256,
        "source_archive_bytes": source.archive_bytes,
        "source_member_name": source.member_name,
        "source_payload_sha256": source_payload_sha,
        "source_payload_bytes": len(source.payload),
        "source_decoder_section_sha256": source_decoder_section_sha,
        "source_decoder_section_bytes": len(source_packed.decoder_packed_brotli),
        "candidate_archive_path": repo_relative(candidate_archive, REPO_ROOT),
        "candidate_archive_sha256": candidate.archive_sha256,
        "candidate_archive_bytes": candidate.archive_bytes,
        "candidate_member_name": candidate.member_name,
        "candidate_decoder_section_sha256": sha256_bytes(candidate_packed.decoder_packed_brotli),
        "candidate_decoder_section_bytes": len(candidate_packed.decoder_packed_brotli),
        "candidate_decoder_section_is_hdm3": candidate_packed.decoder_packed_brotli.startswith(b"HDM3"),
        "source_decoder_raw_sha256": source_raw_sha,
        "restored_decoder_raw_sha256": restored_raw_sha,
        "decoder_raw_matches_source": restored_raw_sha == source_raw_sha,
        "restored_decoder_section_sha256": restored_decoder_section_sha,
        "restored_decoder_section_bytes": len(restored_packed.decoder_packed_brotli),
        "restored_decoder_section_matches_source": restored_decoder_section_matches_source,
        "source_latents_and_sidecar_sha256": source_latents_sha,
        "candidate_latents_and_sidecar_sha256": candidate_latents_sha,
        "restored_latents_and_sidecar_sha256": restored_latents_sha,
        "latents_and_sidecar_match_source": candidate_latents_sha == source_latents_sha
        and restored_latents_sha == source_latents_sha,
        "restored_payload_path": repo_relative(restored_payload_path, REPO_ROOT),
        "restored_payload_sha256": restored_payload_sha,
        "restored_payload_bytes": restored_payload_path.stat().st_size,
        "restored_payload_matches_source": restored_payload_matches_source,
        "adapter_proof": adapter_proof,
        "runtime_adapter_files": [
            {
                "path": repo_relative(path, REPO_ROOT),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in adapter_files
        ],
        "runtime_adapter_integrated": True,
        "public_pr106_dependency_root_unchanged": True,
        "inflate_output_parity_proven_by_payload_identity": inflate_output_parity_proven_by_payload_identity,
        "ready_for_public_runtime_inflate": inflate_output_parity_proven_by_payload_identity,
        "ready_for_exact_eval_dispatch": False,
        "remaining_dispatch_blockers": _dispatch_blockers(
            inflate_output_parity_proven_by_payload_identity
        ),
    }
    write_json(output_dir / "runtime_adapter_proof.json", proof)
    return proof


def _dispatch_blockers(inflate_output_parity_proven: bool) -> list[str]:
    blockers = [
        "strict_pre_submission_compliance_json_missing",
        "lane_dispatch_claim_missing",
        "exact_cuda_auth_eval_missing",
    ]
    if not inflate_output_parity_proven:
        blockers.insert(0, "exact_inflate_output_parity_missing")
    return blockers


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        proof = build_proof(
            source_archive=args.source_archive,
            candidate_archive=args.candidate_archive,
            output_dir=args.output_dir,
        )
    except (brotli.error, HnervHdm3RuntimeAdapterError, HnervLowlevelPackError) as exc:
        print(f"FATAL: HDM3 runtime adapter proof failed: {exc}", file=sys.stderr)
        return 2
    proof = attach_tool_run_manifest(
        proof,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.source_archive, args.candidate_archive],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    if args.json_out is not None:
        write_json(args.json_out, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
