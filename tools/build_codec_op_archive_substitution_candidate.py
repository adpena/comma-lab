#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic CodecOp archive-substitution candidate.

This is a physical ZIP-member replacement tool. It refuses packed payload
logical-substream surgery unless the operator explicitly confirms the
replacement stream is a complete packed payload container. The emitted manifest
is archive-construction evidence only; exact CUDA auth eval is still required
for any score or promotion claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec_op_archive_substitution import (  # noqa: E402
    ArchiveSubstitutionError,
    build_archive_substitution_candidate_from_paths,
    codec_op_candidate_manifest_entry,
)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArchiveSubstitutionError(f"{label} must be a JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--expected-source-archive-bytes", type=int, required=True)
    parser.add_argument("--target-member", required=True)
    parser.add_argument("--expected-target-member-sha256", required=True)
    parser.add_argument("--expected-target-member-bytes", type=int, required=True)
    parser.add_argument("--replacement-substream", type=Path, required=True)
    parser.add_argument("--expected-replacement-sha256")
    parser.add_argument("--expected-replacement-bytes", type=int)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--codec-op-manifest",
        type=Path,
        help="Optional tools/codec_op_param_sweep_manifest.py output to cite and validate.",
    )
    parser.add_argument(
        "--codec-op-candidate-id",
        help="Candidate id inside --codec-op-manifest. Required when that manifest has multiple candidates.",
    )
    parser.add_argument(
        "--exact-runtime-parity-json",
        type=Path,
        help="Optional reviewed parity report. Must pass for ready_for_exact_eval_dispatch=true.",
    )
    parser.add_argument(
        "--lane-claim-json",
        type=Path,
        help="Optional active lane-claim proof. Must pass for ready_for_exact_eval_dispatch=true.",
    )
    parser.add_argument(
        "--allow-packed-payload-container-replacement",
        action="store_true",
        help=(
            "Permit replacing p/renderer_payload.bin/renderer_payload.bin.br. "
            "Use only when the replacement is the complete packed container, "
            "not a logical substream inside it."
        ),
    )
    args = parser.parse_args(argv)

    try:
        codec_op_manifest = None
        if args.codec_op_manifest is not None:
            replacement_bytes = args.replacement_substream.stat().st_size
            codec_op_manifest = codec_op_candidate_manifest_entry(
                args.codec_op_manifest,
                candidate_id=args.codec_op_candidate_id,
                replacement_bytes=replacement_bytes,
            )

        exact_runtime_parity = (
            _read_json_object(
                args.exact_runtime_parity_json,
                label="runtime parity report",
            )
            if args.exact_runtime_parity_json is not None
            else None
        )
        lane_claim = (
            _read_json_object(args.lane_claim_json, label="lane claim proof")
            if args.lane_claim_json is not None
            else None
        )

        manifest = build_archive_substitution_candidate_from_paths(
            source_archive=args.source_archive,
            expected_source_archive_sha256=args.expected_source_archive_sha256,
            expected_source_archive_bytes=args.expected_source_archive_bytes,
            target_member_name=args.target_member,
            expected_target_member_sha256=args.expected_target_member_sha256,
            expected_target_member_bytes=args.expected_target_member_bytes,
            replacement_substream_path=args.replacement_substream,
            output_archive=args.output_archive,
            candidate_id=args.candidate_id,
            expected_replacement_sha256=args.expected_replacement_sha256,
            expected_replacement_bytes=args.expected_replacement_bytes,
            manifest_output=args.manifest_output,
            allow_packed_payload_container_replacement=args.allow_packed_payload_container_replacement,
            codec_op_manifest=codec_op_manifest,
            exact_runtime_parity=exact_runtime_parity,
            lane_claim=lane_claim,
        )
    except (ArchiveSubstitutionError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"codec-op archive substitution failed: {exc}") from None

    archive = manifest["archive"]
    target = manifest["target_member"]
    print(
        "wrote "
        f"{archive['candidate_path']} "
        f"({archive['new_archive_bytes']} bytes, sha256={archive['new_archive_sha256']})"
    )
    print(
        f"target {target['name']}: "
        f"{target['old_bytes']} -> {target['new_bytes']} bytes "
        f"(delta {target['member_byte_delta']})"
    )
    print(
        f"manifest {args.manifest_output} "
        f"ready_for_exact_eval_dispatch={manifest['ready_for_exact_eval_dispatch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
