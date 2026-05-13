#!/usr/bin/env python3
"""Rank HNeRV frontier byte mass against entropy-gap targets."""

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

from tac.hnerv_frontier_entropy_ranking import (  # noqa: E402
    build_frontier_entropy_gap_ranking,
    render_markdown,
)
from tac.repo_io import json_text, read_json, repo_relative  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

DEFAULT_SCORECARD = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "hnerv_frontier_scorecard_refresh_20260513_codex"
    / "scorecard.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument(
        "--frontier-mode",
        choices=("canonical", "score_lowering"),
        default="canonical",
        help=(
            "Which exact-CUDA scorecard frontier to plan against. "
            "Use score_lowering for internal optimizer routing; canonical remains the "
            "public/promotion-safe default."
        ),
    )
    parser.add_argument(
        "--entropy-audit",
        action="append",
        default=[],
        type=Path,
        help="Entropy-overhead audit JSON to join with the current frontier.",
    )
    parser.add_argument(
        "--candidate-manifest",
        action="append",
        default=[],
        type=Path,
        help="Lossless repack/candidate manifest used for section SHA lineage.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        scorecard = read_json(args.scorecard)
        entropy_audits = [read_json(path) for path in args.entropy_audit]
        candidate_manifests = []
        for path in args.candidate_manifest:
            payload = read_json(path)
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("candidate_manifest_path", repo_relative(path, REPO_ROOT))
            candidate_manifests.append(payload)
        manifest = build_frontier_entropy_gap_ranking(
            scorecard,
            entropy_audits=entropy_audits,
            candidate_manifests=candidate_manifests,
            frontier_mode=args.frontier_mode,
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: HNeRV frontier entropy ranking input rejected: {exc}", file=sys.stderr)
        return 2

    input_paths = [args.scorecard, *args.entropy_audit, *args.candidate_manifest]
    manifest = attach_tool_run_manifest(
        manifest,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(manifest), encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
