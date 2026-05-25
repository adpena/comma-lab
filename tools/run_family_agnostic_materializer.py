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
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
    FamilyAgnosticMaterializerError,
    materialize_archive_section_entropy_recode_candidate,
    materialize_packet_member_merge_candidate,
    materialize_packet_member_recompress_candidate,
    materialize_packet_member_zip_header_elide_candidate,
    materialize_tensor_factorize_candidate,
    verify_runtime_consumption_proof,
)
from tac.optimization.packet_member_merge_receiver import (  # noqa: E402
    PacketMemberMergeReceiverError,
    build_packet_member_merge_receiver_runtime,
    build_packet_member_merge_runtime_consumption_proof,
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
            PACKET_MEMBER_MERGE_TARGET_KIND,
            PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
            PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
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
    parser.add_argument("--member-names", action="append", default=[])
    parser.add_argument("--all-members", action="store_true")
    parser.add_argument("--merge-contract", type=Path)
    parser.add_argument("--merged-member-name")
    parser.add_argument("--packet-member-merge-source-runtime-dir", type=Path)
    parser.add_argument("--packet-member-merge-runtime-dir-out", type=Path)
    parser.add_argument("--packet-member-merge-runtime-manifest-out", type=Path)
    parser.add_argument(
        "--allow-packet-member-merge-runtime-sidecars",
        action="store_true",
    )
    parser.add_argument("--header-elision-contract", type=Path)
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
    except (
        OSError,
        ArtifactWriteError,
        FamilyAgnosticMaterializerError,
        PacketMemberMergeReceiverError,
    ) as exc:
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
    proof_out_target_kinds = {
        ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
        PACKET_MEMBER_MERGE_TARGET_KIND,
        PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
        TENSOR_FACTORIZE_TARGET_KIND,
    }
    if (
        args.runtime_consumption_proof_out is not None
        and args.target_kind not in proof_out_target_kinds
    ):
        raise FamilyAgnosticMaterializerError(
            "--runtime-consumption-proof-out is currently supported only for "
            "archive_section_entropy_recode_v1, packet_member_merge_v1, "
            "packet_member_recompress_v1, "
            "packet_member_zip_header_elide_v1, and tensor_factorize_v1"
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
        runtime_proof_out = args.runtime_consumption_proof_out
        if args.runtime_consumption_proof is None and runtime_proof_out is None:
            runtime_proof_out = args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_consumption_proof.json"
            )
        return materialize_archive_section_entropy_recode_candidate(
            **common,
            section_manifest=args.section_manifest,
            section_names=args.section_name,
            brotli_qualities=tuple(args.brotli_quality or [9, 10, 11]),
            runtime_consumption_proof_out=runtime_proof_out,
            expected_existing_runtime_consumption_proof_sha256=(
                args.expected_existing_runtime_consumption_proof_sha256
            ),
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
    if args.target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        if args.packet_member_manifest is not None:
            input_paths.append(args.packet_member_manifest)
        if args.merge_contract is not None:
            input_paths.append(args.merge_contract)
        runtime_proof_out = args.runtime_consumption_proof_out
        if (
            args.runtime_consumption_proof is None
            and runtime_proof_out is None
            and args.packet_member_merge_source_runtime_dir is None
        ):
            runtime_proof_out = args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_consumption_proof.json"
            )
        materializer_runtime_proof_out = (
            None
            if args.packet_member_merge_source_runtime_dir is not None
            else runtime_proof_out
        )
        manifest = materialize_packet_member_merge_candidate(
            **common,
            packet_member_manifest=args.packet_member_manifest,
            member_name=args.member_name,
            member_names=tuple(args.member_names),
            all_members=args.all_members,
            merge_contract=args.merge_contract,
            merged_member_name=args.merged_member_name,
            runtime_consumption_proof_out=materializer_runtime_proof_out,
            expected_existing_runtime_consumption_proof_sha256=(
                args.expected_existing_runtime_consumption_proof_sha256
            ),
        )
        if args.packet_member_merge_source_runtime_dir is None:
            return manifest
        runtime_dir = (
            args.packet_member_merge_runtime_dir_out
            if args.packet_member_merge_runtime_dir_out is not None
            else args.output_manifest.with_name(f"{args.output_manifest.stem}.runtime")
        )
        runtime_manifest_out = (
            args.packet_member_merge_runtime_manifest_out
            if args.packet_member_merge_runtime_manifest_out is not None
            else args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_adapter.json"
            )
        )
        proof_out = (
            args.runtime_consumption_proof_out
            if args.runtime_consumption_proof_out is not None
            else args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_consumption_proof.json"
            )
        )
        runtime_manifest = build_packet_member_merge_receiver_runtime(
            source_runtime_dir=args.packet_member_merge_source_runtime_dir,
            candidate_manifest=manifest,
            runtime_dir_out=runtime_dir,
            runtime_manifest_out=runtime_manifest_out,
            repo_root=REPO_ROOT,
            allow_runtime_sidecars=args.allow_packet_member_merge_runtime_sidecars,
            allow_overwrite=args.allow_overwrite,
            min_free_bytes=args.min_free_bytes,
        )
        runtime_proof = build_packet_member_merge_runtime_consumption_proof(
            runtime_adapter_manifest=runtime_manifest,
            candidate_manifest=manifest,
            repo_root=REPO_ROOT,
        )
        write_json_artifact(
            proof_out,
            runtime_proof,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=(
                args.expected_existing_runtime_consumption_proof_sha256
            ),
            min_free_bytes=args.min_free_bytes,
        )
        receiver_verification = verify_runtime_consumption_proof(
            runtime_consumption_proof=runtime_proof,
            required_candidate_archive_sha256=manifest["candidate_archive"]["sha256"],
            required_candidate_member_sha256=manifest["candidate_member"]["sha256"],
            repo_root=REPO_ROOT,
        )
        manifest["packet_member_merge_receiver_runtime"] = runtime_manifest
        manifest["receiver_verification"] = receiver_verification
        manifest["runtime_consumption_proof_path"] = proof_out.as_posix()
        manifest["reconstruction_proof_satisfied"] = (
            receiver_verification["receiver_contract_satisfied"] is True
        )
        manifest["receiver_contract_satisfied"] = (
            receiver_verification["receiver_contract_satisfied"] is True
            and receiver_verification.get("runtime_adapter_ready") is True
        )
        manifest["runtime_adapter_ready"] = (
            receiver_verification.get("runtime_adapter_ready") is True
        )
        manifest["readiness_blockers"] = [
            str(blocker)
            for blocker in (manifest.get("readiness_blockers") or [])
            if blocker
            not in {
                "runtime_consumption_proof_missing",
                "runtime_consumption_proof_not_passed",
                "packet_member_merge_receiver_contract_not_satisfied",
                "packet_member_merge_exact_readiness_refused_until_byte_closed_runtime_adapter_lands",
            }
        ]
        manifest["readiness_blockers"].extend(
            str(blocker) for blocker in receiver_verification.get("blockers") or []
        )
        if manifest["receiver_contract_satisfied"] is not True:
            manifest["readiness_blockers"].append(
                "packet_member_merge_receiver_contract_not_satisfied"
            )
        manifest["readiness_blockers"] = list(dict.fromkeys(manifest["readiness_blockers"]))
        return manifest
    if args.target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        if args.packet_member_manifest is not None:
            input_paths.append(args.packet_member_manifest)
        if args.header_elision_contract is not None:
            input_paths.append(args.header_elision_contract)
        runtime_proof_out = args.runtime_consumption_proof_out
        if args.runtime_consumption_proof is None and runtime_proof_out is None:
            runtime_proof_out = args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_consumption_proof.json"
            )
        return materialize_packet_member_zip_header_elide_candidate(
            **common,
            packet_member_manifest=args.packet_member_manifest,
            member_name=args.member_name,
            member_names=tuple(args.member_names),
            all_members=args.all_members,
            header_elision_contract=args.header_elision_contract,
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
        runtime_proof_out = args.runtime_consumption_proof_out
        if args.runtime_consumption_proof is None and runtime_proof_out is None:
            runtime_proof_out = args.output_manifest.with_name(
                f"{args.output_manifest.stem}.runtime_consumption_proof.json"
            )
        return materialize_tensor_factorize_candidate(
            **common,
            tensor_manifest=args.tensor_manifest,
            factorization_contract=contract,
            runtime_consumption_proof_out=runtime_proof_out,
            expected_existing_runtime_consumption_proof_sha256=(
                args.expected_existing_runtime_consumption_proof_sha256
            ),
        )
    raise FamilyAgnosticMaterializerError(f"unsupported target kind: {args.target_kind}")


if __name__ == "__main__":
    raise SystemExit(main())
