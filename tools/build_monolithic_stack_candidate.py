#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a byte-closed monolithic HNeRV section-replacement candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.monolithic_packet_candidate import (  # noqa: E402
    MonolithicPacketCandidateError,
    ReplacementSection,
    build_monolithic_packet_candidate,
)
from tools.export_active_lane_claim_json import build_active_lane_claim_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--target-section")
    parser.add_argument("--replacement-section", type=Path)
    parser.add_argument(
        "--replacement-manifest",
        type=Path,
        help=(
            "JSON object with a replacements list, or a raw list. Each replacement "
            "uses section_name/target_section and replacement_path/path."
        ),
    )
    parser.add_argument("--runtime-parity-json", type=Path)
    parser.add_argument("--lane-claim-json", type=Path)
    parser.add_argument("--claims-path", type=Path)
    parser.add_argument("--dispatch-lane-id")
    parser.add_argument("--dispatch-instance-job-id")
    parser.add_argument("--now-utc")
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--expected-source-archive-sha256")
    parser.add_argument("--expected-source-archive-bytes", type=int)
    parser.add_argument("--expected-old-section-sha256")
    parser.add_argument("--expected-old-section-bytes", type=int)
    parser.add_argument("--expected-new-section-sha256")
    parser.add_argument("--expected-new-section-bytes", type=int)
    args = parser.parse_args(argv)

    try:
        replacements = _load_replacements(args)
        manifest = build_monolithic_packet_candidate(
            source_archive=args.source_archive,
            output_archive=args.output_archive,
            candidate_id=args.candidate_id,
            replacements=replacements,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
            expected_source_archive_bytes=args.expected_source_archive_bytes,
            manifest_output=args.manifest_output,
            runtime_parity=_load_optional_json_object(args.runtime_parity_json),
            lane_claim=_lane_claim_payload(args),
            dispatch_lane_id=args.dispatch_lane_id,
            dispatch_instance_job_id=args.dispatch_instance_job_id,
        )
    except (MonolithicPacketCandidateError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"monolithic candidate build failed: {exc}") from None

    archive = manifest["candidate_archive"]
    replacement_manifest = manifest["replacements"][0]
    print(
        f"wrote {archive['path']} "
        f"({archive['bytes']} bytes, sha256={archive['sha256']})"
    )
    print(
        f"sections replaced: {len(manifest['replacements'])}; "
        f"first {replacement_manifest['section_name']}: "
        f"{replacement_manifest['old_bytes']} -> {replacement_manifest['new_bytes']} bytes "
        f"(delta {replacement_manifest['section_byte_delta']})"
    )
    print(
        f"manifest {args.manifest_output} "
        f"ready_for_exact_eval_dispatch={manifest['ready_for_exact_eval_dispatch']}"
    )
    return 0


def _load_replacements(args: argparse.Namespace) -> list[ReplacementSection]:
    if args.replacement_manifest is not None:
        payload = _load_json_object(args.replacement_manifest)
        raw_replacements = payload.get("replacements") if isinstance(payload, dict) else payload
        if not isinstance(raw_replacements, list):
            raise MonolithicPacketCandidateError(
                "--replacement-manifest must be a list or an object with a replacements list"
            )
        return [
            _replacement_from_mapping(item, base_dir=args.replacement_manifest.parent)
            for item in raw_replacements
        ]

    if not args.target_section or args.replacement_section is None:
        raise MonolithicPacketCandidateError(
            "either --replacement-manifest or both --target-section/--replacement-section are required"
        )
    return [
        ReplacementSection(
            section_name=args.target_section,
            replacement_path=args.replacement_section,
            expected_old_sha256=args.expected_old_section_sha256,
            expected_old_bytes=args.expected_old_section_bytes,
            expected_new_sha256=args.expected_new_section_sha256,
            expected_new_bytes=args.expected_new_section_bytes,
        )
    ]


def _replacement_from_mapping(item: Any, *, base_dir: Path) -> ReplacementSection:
    if not isinstance(item, dict):
        raise MonolithicPacketCandidateError("replacement entry must be an object")
    section_name = item.get("section_name", item.get("target_section"))
    replacement_path = item.get("replacement_path", item.get("path"))
    if not isinstance(section_name, str) or not section_name:
        raise MonolithicPacketCandidateError("replacement entry missing section_name")
    if not isinstance(replacement_path, str) or not replacement_path:
        raise MonolithicPacketCandidateError("replacement entry missing replacement_path")
    path = Path(replacement_path)
    if not path.is_absolute():
        path = base_dir / path
    return ReplacementSection(
        section_name=section_name,
        replacement_path=path,
        expected_old_sha256=_optional_str(item.get("expected_old_sha256")),
        expected_old_bytes=_optional_int(item.get("expected_old_bytes")),
        expected_new_sha256=_optional_str(item.get("expected_new_sha256")),
        expected_new_bytes=_optional_int(item.get("expected_new_bytes")),
    )


def _load_optional_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json_object(path)
    if not isinstance(payload, dict):
        raise MonolithicPacketCandidateError(f"{path} must contain a JSON object")
    return payload


def _lane_claim_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.lane_claim_json is not None and args.claims_path is not None:
        raise MonolithicPacketCandidateError("use either --lane-claim-json or --claims-path, not both")
    if args.lane_claim_json is not None:
        return _load_optional_json_object(args.lane_claim_json)
    if args.claims_path is None:
        return None
    if not args.dispatch_lane_id:
        raise MonolithicPacketCandidateError("--claims-path requires --dispatch-lane-id")
    return build_active_lane_claim_json(
        claims_path=args.claims_path,
        lane_id=args.dispatch_lane_id,
        instance_job_id=args.dispatch_instance_job_id,
        now_utc=args.now_utc,
        ttl_hours=args.ttl_hours,
    )


def _load_json_object(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise MonolithicPacketCandidateError(f"expected string or null, got {type(value).__name__}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise MonolithicPacketCandidateError(f"expected integer or null, got {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
