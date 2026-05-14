#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed HNeRV entropy candidate-packet manifest."""

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

from tac.hnerv_entropy_candidate_packet import (  # noqa: E402
    HDC2_STREAM_ARTIFACT_REQUIREMENTS,
    HnervEntropyCandidatePacketError,
    build_candidate_packet_manifest,
    build_hdc2_stream_byte_equivalence_work_product,
    discover_candidate_audit_inputs,
    discovery_report_input_paths,
    existing_artifact_input_paths,
    normalize_entropy_audit_payload,
)
from tac.repo_io import json_text, read_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

NAMED_ARTIFACT_FLAGS = {
    "source_archive_manifest": "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256",
    "source_stream_section_manifest": "source_stream_section_sha256_byte_range_and_symbol_count",
    "candidate_stream_section_manifest": "candidate_stream_section_sha256_byte_range_and_byte_count",
    "decoded_output_equivalence_report": "old_new_decoded_output_sha256_equality_report",
    "roundtrip_decode_validation_manifest": "roundtrip_decode_validation_manifest",
    "candidate_archive_manifest": "candidate_archive_manifest_with_member_sha256s",
    "runtime_tree_parity_manifest": "runtime_tree_parity_manifest",
    "strict_pre_submission_compliance_json": "strict_pre_submission_compliance_json",
    "meta_lagrangian_atom_json": "meta_lagrangian_atom_json_with_byte_delta_and_interaction_assumptions",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entropy-audit",
        type=Path,
        help=(
            "Entropy codec gap audit or stream profile JSON. If omitted, the "
            "tool discovers candidate inputs and either selects the first valid "
            "one or emits a fail-closed discovery report."
        ),
    )
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="Only emit the deterministic entropy-audit discovery report.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        type=Path,
        help="Root directory or JSON file to scan when --entropy-audit is omitted.",
    )
    parser.add_argument("--target-rank", type=int)
    parser.add_argument("--target-label")
    parser.add_argument("--target-kind")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="REQUIREMENT_ID=PATH",
        help="Attach a target-specific requirement artifact. Repeatable.",
    )
    parser.add_argument("--source-archive-manifest", type=Path)
    parser.add_argument("--source-stream-section-manifest", type=Path)
    parser.add_argument("--candidate-stream-section-manifest", type=Path)
    parser.add_argument("--decoded-output-equivalence-report", type=Path)
    parser.add_argument("--roundtrip-decode-validation-manifest", type=Path)
    parser.add_argument("--candidate-archive-manifest", type=Path)
    parser.add_argument("--runtime-tree-parity-manifest", type=Path)
    parser.add_argument("--strict-pre-submission-compliance-json", type=Path)
    parser.add_argument("--meta-lagrangian-atom-json", type=Path)
    parser.add_argument(
        "--hdc2-stream-work-product-profile",
        type=Path,
        help=(
            "HNeRV structural-recode profile used to materialize byte-closed "
            "HDC2 stream-level source/candidate manifests."
        ),
    )
    parser.add_argument(
        "--hdc2-stream-work-product-source-archive",
        type=Path,
        help="Source HNeRV single-member archive for HDC2 stream work-product extraction.",
    )
    parser.add_argument(
        "--hdc2-stream-work-product-source-exact-eval-json",
        type=Path,
        help=(
            "Source exact-eval JSON required for runtime-tree custody when "
            "--hdc2-stream-work-product-dir is used."
        ),
    )
    parser.add_argument(
        "--hdc2-stream-work-product-dir",
        type=Path,
        help=(
            "Directory for HDC2 stream work-product JSONs and candidate stream bytes. "
            "Generated artifacts are attached to the candidate packet."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--entropy-audit-json-out",
        type=Path,
        help=(
            "Write the normalized/generated entropy-overhead audit used as "
            "packet input. This is local audit materialization only."
        ),
    )
    parser.add_argument(
        "--fail-if-missing",
        action="store_true",
        help="Exit 1 if any selected-target requirement artifact is missing.",
    )
    return parser.parse_args(argv)


def artifact_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for spec in args.artifact:
        requirement_id, path = _parse_artifact_spec(spec)
        _add_artifact(artifacts, requirement_id, path)
    for attr, requirement_id in NAMED_ARTIFACT_FLAGS.items():
        path = getattr(args, attr)
        if path is not None:
            _add_artifact(artifacts, requirement_id, path)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        artifacts = artifact_paths_from_args(args)
        if args.hdc2_stream_work_product_dir is not None:
            if args.hdc2_stream_work_product_profile is None:
                raise HnervEntropyCandidatePacketError(
                    "--hdc2-stream-work-product-dir requires --hdc2-stream-work-product-profile"
                )
            if args.hdc2_stream_work_product_source_archive is None:
                raise HnervEntropyCandidatePacketError(
                    "--hdc2-stream-work-product-dir requires "
                    "--hdc2-stream-work-product-source-archive"
                )
            if args.hdc2_stream_work_product_source_exact_eval_json is None:
                raise HnervEntropyCandidatePacketError(
                    "--hdc2-stream-work-product-dir requires "
                    "--hdc2-stream-work-product-source-exact-eval-json"
                )
            generated = _materialize_hdc2_stream_work_product(args)
            for requirement_id, path in generated.items():
                _add_artifact(artifacts, requirement_id, path)
        entropy_audit = args.entropy_audit
        if args.discovery_only or entropy_audit is None:
            discovery = discover_candidate_audit_inputs(
                search_roots=args.search_root or None,
                repo_root=REPO_ROOT,
            )
            if args.discovery_only or discovery.get("selected_entropy_audit") is None:
                discovery = attach_tool_run_manifest(
                    discovery,
                    tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
                    argv=raw_argv,
                    input_paths=discovery_report_input_paths(discovery, REPO_ROOT),
                    repo_root=REPO_ROOT,
                    output_path=args.json_out,
                )
                text = json_text(discovery)
                if args.json_out:
                    args.json_out.parent.mkdir(parents=True, exist_ok=True)
                    args.json_out.write_text(text, encoding="utf-8")
                else:
                    print(text, end="")
                if args.fail_if_missing and discovery.get("missing_source_artifacts"):
                    _print_fail_if_missing(
                        "discovery missing source artifacts",
                        discovery.get("missing_source_artifacts") or [],
                        args.json_out,
                    )
                    return 1
                return 0
            selected = discovery["selected_entropy_audit"]
            entropy_audit = REPO_ROOT / str(selected["path"])

        if args.entropy_audit_json_out is not None:
            normalized_audit, audit_source_kind = normalize_entropy_audit_payload(read_json(entropy_audit))
            normalized_audit["audit_source_kind"] = audit_source_kind
            normalized_audit = attach_tool_run_manifest(
                normalized_audit,
                tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
                argv=raw_argv,
                input_paths=[entropy_audit],
                repo_root=REPO_ROOT,
                output_path=args.entropy_audit_json_out,
            )
            args.entropy_audit_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.entropy_audit_json_out.write_text(json_text(normalized_audit), encoding="utf-8")
            entropy_audit = args.entropy_audit_json_out

        manifest = build_candidate_packet_manifest(
            entropy_audit,
            target_rank=args.target_rank,
            target_label=args.target_label,
            target_kind=args.target_kind,
            artifact_paths=artifacts,
            repo_root=REPO_ROOT,
        )
    except HnervEntropyCandidatePacketError as exc:
        print(f"FATAL: HNeRV entropy candidate packet input rejected: {exc}", file=sys.stderr)
        return 2

    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=existing_artifact_input_paths(entropy_audit, artifacts),
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_missing and manifest.get("missing_artifacts"):
        _print_fail_if_missing(
            "candidate packet missing required artifacts",
            manifest.get("missing_artifacts") or [],
            args.json_out,
        )
        return 1
    return 0


def _parse_artifact_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise HnervEntropyCandidatePacketError("--artifact must be REQUIREMENT_ID=PATH")
    requirement_id, path = value.split("=", 1)
    requirement_id = requirement_id.strip()
    if not requirement_id:
        raise HnervEntropyCandidatePacketError("--artifact requirement id must be nonempty")
    if not path:
        raise HnervEntropyCandidatePacketError("--artifact path must be nonempty")
    return requirement_id, Path(path)


def _add_artifact(artifacts: dict[str, Path], requirement_id: str, path: Path) -> None:
    if requirement_id in artifacts:
        raise HnervEntropyCandidatePacketError(f"duplicate artifact requirement id: {requirement_id}")
    artifacts[requirement_id] = path


def _print_fail_if_missing(reason: str, missing: list[object], json_out: Path | None) -> None:
    items = ", ".join(str(item) for item in missing) or "unknown"
    print(f"FATAL: HNeRV entropy candidate packet {reason}: {items}", file=sys.stderr)
    if json_out is not None:
        print(f"FATAL: details written to {json_out}", file=sys.stderr)


def _materialize_hdc2_stream_work_product(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.hdc2_stream_work_product_dir
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_stream_path = output_dir / "candidate_hdc2_global_prev_symbol_stream.bin"
    work_product = build_hdc2_stream_byte_equivalence_work_product(
        args.hdc2_stream_work_product_profile,
        args.hdc2_stream_work_product_source_archive,
        source_exact_eval_json_path=args.hdc2_stream_work_product_source_exact_eval_json,
        candidate_stream_path=candidate_stream_path,
        bounded_candidate_stream_dir=output_dir,
        repo_root=REPO_ROOT,
    )
    named_payloads = {
        "source_archive_manifest": work_product["source_archive_manifest"],
        "source_stream_section_manifest": work_product["source_stream_section_manifest"],
        "candidate_stream_section_manifest": work_product["candidate_stream_section_manifest"],
        "decoded_output_equivalence_report": work_product["decoded_output_equivalence_report"],
        "roundtrip_decode_validation_manifest": work_product[
            "roundtrip_decode_validation_manifest"
        ],
        "byte_accounted_model_overhead_reduction_manifest": work_product[
            "byte_accounted_model_overhead_reduction_manifest"
        ],
        "byte_accounted_static_model_context_reduction_manifest": work_product[
            "byte_accounted_static_model_context_reduction_manifest"
        ],
        "old_new_model_context_table_diff": work_product["old_new_model_context_table_diff"],
    }
    generated: dict[str, Path] = {}
    for name, payload in named_payloads.items():
        path = output_dir / f"{HDC2_STREAM_ARTIFACT_REQUIREMENTS[name]}.json"
        path.write_text(json_text(payload), encoding="utf-8")
        generated[HDC2_STREAM_ARTIFACT_REQUIREMENTS[name]] = path
    (output_dir / "hdc2_stream_byte_equivalence_work_product.json").write_text(
        json_text(work_product),
        encoding="utf-8",
    )
    return generated


if __name__ == "__main__":
    raise SystemExit(main())
