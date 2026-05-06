#!/usr/bin/env python3
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
    HnervEntropyCandidatePacketError,
    build_candidate_packet_manifest,
    existing_artifact_input_paths,
)
from tac.repo_io import json_text  # noqa: E402
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
    parser.add_argument("--entropy-audit", type=Path, required=True)
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
    parser.add_argument("--json-out", type=Path)
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
        manifest = build_candidate_packet_manifest(
            args.entropy_audit,
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
        input_paths=existing_artifact_input_paths(args.entropy_audit, artifacts),
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


if __name__ == "__main__":
    raise SystemExit(main())
