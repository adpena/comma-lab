#!/usr/bin/env python3
"""Audit HNeRV section candidate diffs for no-op-resistant byte proof."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_section_repack import (  # noqa: E402
    audit_candidate_section_diff,
    build_section_repack_plan,
    candidate_diff_from_scorecard_manifests,
)
from tac.repo_io import json_text, read_json  # noqa: E402

DEFAULT_SCORECARD = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "public_hnerv_frontier_payload_profiles_20260504_codex"
    / "scorecard.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--candidate-label")
    parser.add_argument("--candidate-diff-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit nonzero if the section diff is blocked.",
    )
    parser.add_argument(
        "--require-raw-equivalence",
        action="store_true",
        help=(
            "Require brotli/section raw-equivalence proof for decoder/latent "
            "sections before archive preflight."
        ),
    )
    parser.add_argument(
        "--require-byte-reduction",
        action="store_true",
        help="Require every audited changed section to be smaller than source.",
    )
    parser.add_argument(
        "--require-same-runtime-full-frame-parity",
        action="store_true",
        help=(
            "Require a same-runtime full-frame streaming parity proof bound to "
            "the source and candidate archive SHA-256s."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scorecard = read_json(args.scorecard)
    plan = build_section_repack_plan(scorecard, labels=[args.source_label])
    if args.candidate_diff_json is not None:
        candidate_diff = read_json(args.candidate_diff_json)
    elif args.candidate_label:
        candidate_diff = candidate_diff_from_scorecard_manifests(
            scorecard,
            source_label=args.source_label,
            candidate_label=args.candidate_label,
        )
    else:
        raise SystemExit("provide --candidate-label or --candidate-diff-json")
    audit = audit_candidate_section_diff(
        plan,
        candidate_diff,
        require_raw_equivalence=args.require_raw_equivalence,
        require_byte_reduction=args.require_byte_reduction,
        require_same_runtime_full_frame_parity=args.require_same_runtime_full_frame_parity,
    )
    text = json_text(audit)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if args.fail_if_blocked and audit["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
