#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed packet-readiness manifest for A5 PR101 frame budgets."""

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

from tac.pr101_frame_conditional_packet_readiness import (  # noqa: E402
    CANDIDATE_ARCHIVE_MANIFEST,
    PACKET_RUNTIME_PATCH_MANIFEST,
    PER_PAIR_SCORE_MARGINAL_MANIFEST,
    RUNTIME_CONSUMPTION_PROOF,
    STRICT_PRE_SUBMISSION_COMPLIANCE_JSON,
    FrameConditionalPacketReadinessError,
    build_packet_readiness,
    existing_artifact_input_paths,
)
from tac.repo_io import json_text  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a5-manifest", type=Path, required=True)
    parser.add_argument("--candidate-archive-manifest", type=Path)
    parser.add_argument("--packet-runtime-patch-manifest", type=Path)
    parser.add_argument("--runtime-consumption-proof", type=Path)
    parser.add_argument("--per-pair-score-marginal-manifest", type=Path)
    parser.add_argument("--strict-pre-submission-compliance-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    artifact_paths = {
        CANDIDATE_ARCHIVE_MANIFEST: args.candidate_archive_manifest,
        PACKET_RUNTIME_PATCH_MANIFEST: args.packet_runtime_patch_manifest,
        RUNTIME_CONSUMPTION_PROOF: args.runtime_consumption_proof,
        PER_PAIR_SCORE_MARGINAL_MANIFEST: args.per_pair_score_marginal_manifest,
        STRICT_PRE_SUBMISSION_COMPLIANCE_JSON: args.strict_pre_submission_compliance_json,
    }
    try:
        payload = build_packet_readiness(
            a5_manifest_path=args.a5_manifest,
            artifact_paths=artifact_paths,
            repo_root=REPO_ROOT,
        )
    except (OSError, ValueError, FrameConditionalPacketReadinessError) as exc:
        print(f"FATAL: A5 packet-readiness input rejected: {exc}", file=sys.stderr)
        return 2

    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=existing_artifact_input_paths(args.a5_manifest, artifact_paths),
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    text = json_text(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    if args.fail_if_not_ready and payload["ready_for_exact_eval_after_lane_claim"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
