#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run family-agnostic byte-shaving materializers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.family_agnostic_materializers import (  # noqa: E402
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
    FamilyAgnosticMaterializerError,
    materialize_archive_section_entropy_recode_candidate,
    materialize_packet_member_recompress_candidate,
    materialize_tensor_factorize_candidate,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-kind",
        required=True,
        choices=(
            ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            TENSOR_FACTORIZE_TARGET_KIND,
        ),
    )
    parser.add_argument("--archive-path", required=True, type=Path)
    parser.add_argument("--output-archive", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--runtime-consumption-proof", type=Path)
    parser.add_argument("--runtime-consumption-proof-out", type=Path)
    parser.add_argument("--section-manifest", type=Path)
    parser.add_argument("--section-name", action="append", default=[])
    parser.add_argument("--brotli-quality", action="append", type=int, default=[])
    parser.add_argument("--packet-member-manifest", type=Path)
    parser.add_argument("--member-name")
    parser.add_argument("--zip-compression-method", action="append", default=[])
    parser.add_argument("--zip-compresslevel", action="append", type=int, default=[])
    parser.add_argument("--tensor-manifest", type=Path)
    parser.add_argument("--factorization-contract", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--allow-size-regression", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-existing-output-sha256")
    parser.add_argument("--expected-existing-runtime-consumption-proof-sha256")
    parser.add_argument("--expected-existing-manifest-sha256")
    parser.add_argument("--min-free-bytes", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [args.archive_path]
    if args.runtime_consumption_proof is not None:
        input_paths.append(args.runtime_consumption_proof)
    if (
        args.runtime_consumption_proof is not None
        and args.runtime_consumption_proof_out is not None
    ):
        print(
            "FATAL: --runtime-consumption-proof and --runtime-consumption-proof-out "
            "are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = _run_materializer(args, input_paths=input_paths)
    except (OSError, ArtifactWriteError, FamilyAgnosticMaterializerError) as exc:
        print(f"FATAL: family-agnostic materializer failed: {exc}", file=sys.stderr)
        return 2
    existing_manifest_sha = (
        sha256_file(args.output_manifest) if args.output_manifest.is_file() else None
    )
    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.output_manifest,
    )
    try:
        write_json_artifact(
            args.output_manifest,
            manifest,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=(
                args.expected_existing_manifest_sha256
                if args.expected_existing_manifest_sha256 is not None
                else existing_manifest_sha
            ),
            min_free_bytes=args.min_free_bytes,
        )
    except (OSError, ArtifactWriteError) as exc:
        print(f"FATAL: manifest write failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(manifest), end="")
    return 0


def _run_materializer(
    args: argparse.Namespace,
    *,
    input_paths: list[Path],
) -> dict:
    if (
        args.runtime_consumption_proof_out is not None
        and args.target_kind != PACKET_MEMBER_RECOMPRESS_TARGET_KIND
    ):
        raise FamilyAgnosticMaterializerError(
            "--runtime-consumption-proof-out is currently supported only for "
            "packet_member_recompress_v1"
        )
    common = {
        "archive_path": args.archive_path,
        "output_archive": args.output_archive,
        "runtime_consumption_proof": args.runtime_consumption_proof,
        "repo_root": REPO_ROOT,
        "allow_size_regression": args.allow_size_regression,
        "allow_overwrite": args.allow_overwrite,
        "expected_existing_output_sha256": args.expected_existing_output_sha256,
        "min_free_bytes": args.min_free_bytes,
    }
    if args.target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        if args.section_manifest is None:
            raise FamilyAgnosticMaterializerError("--section-manifest is required")
        input_paths.append(args.section_manifest)
        return materialize_archive_section_entropy_recode_candidate(
            **common,
            section_manifest=args.section_manifest,
            section_names=args.section_name,
            brotli_qualities=tuple(args.brotli_quality or [9, 10, 11]),
        )
    if args.target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        if args.packet_member_manifest is not None:
            input_paths.append(args.packet_member_manifest)
        runtime_proof_out = args.runtime_consumption_proof_out
        if args.runtime_consumption_proof is None and runtime_proof_out is None:
            runtime_proof_out = args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_consumption_proof.json"
            )
        return materialize_packet_member_recompress_candidate(
            **common,
            packet_member_manifest=args.packet_member_manifest,
            member_name=args.member_name,
            compression_methods=tuple(args.zip_compression_method or ["stored", "deflated"]),
            compresslevels=tuple(args.zip_compresslevel or [9]),
            runtime_consumption_proof_out=runtime_proof_out,
            expected_existing_runtime_consumption_proof_sha256=(
                args.expected_existing_runtime_consumption_proof_sha256
            ),
        )
    if args.target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        if args.tensor_manifest is None:
            raise FamilyAgnosticMaterializerError("--tensor-manifest is required")
        if args.factorization_contract is None and args.rank is None:
            raise FamilyAgnosticMaterializerError(
                "--factorization-contract or --rank is required"
            )
        input_paths.append(args.tensor_manifest)
        contract = (
            args.factorization_contract
            if args.factorization_contract is not None
            else {"rank": args.rank}
        )
        if args.factorization_contract is not None:
            input_paths.append(args.factorization_contract)
        return materialize_tensor_factorize_candidate(
            **common,
            tensor_manifest=args.tensor_manifest,
            factorization_contract=contract,
        )
    raise FamilyAgnosticMaterializerError(f"unsupported target kind: {args.target_kind}")


if __name__ == "__main__":
    raise SystemExit(main())
