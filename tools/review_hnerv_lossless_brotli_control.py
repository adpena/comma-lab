#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Review the existing PR106x low-level Brotli exact-control evidence."""

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

from tac.hnerv_lossless_control_promotion_review import (  # noqa: E402
    build_lossless_control_promotion_review,
    inspect_single_member_archive,
    render_markdown,
)
from tac.hnerv_frontier_defaults import (  # noqa: E402
    HNERV_ACTIVE_ENTROPY_RANKING,
    HNERV_ACTIVE_SCORECARD,
)
from tac.repo_io import json_text, read_json, repo_relative  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

DEFAULT_RESULT_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_SCORECARD = HNERV_ACTIVE_SCORECARD
DEFAULT_RANKING = HNERV_ACTIVE_ENTROPY_RANKING
DEFAULT_CANDIDATE = (
    DEFAULT_RESULT_ROOT
    / "hnerv_lowlevel_repack_pr106x_20260506_codex"
    / "result.json"
)
DEFAULT_PUBLIC_PREFLIGHT = (
    DEFAULT_RESULT_ROOT
    / "hnerv_lowlevel_repack_pr106x_20260506_codex"
    / "public_replay_preflight.json"
)
DEFAULT_CANDIDATE_ARCHIVE = (
    DEFAULT_RESULT_ROOT
    / "hnerv_lowlevel_repack_pr106x_20260506_codex"
    / "pr106x_hnerv_brotli_repack_candidate.zip"
)
DEFAULT_EXACT_DIR = (
    DEFAULT_RESULT_ROOT
    / "lightning_batch"
    / "exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506"
)
DEFAULT_EXACT_EVAL = DEFAULT_EXACT_DIR / "contest_auth_eval.adjudicated.json"
DEFAULT_ADJUDICATION = DEFAULT_EXACT_DIR / "adjudication_provenance.json"
DEFAULT_EXACT_ARCHIVE = DEFAULT_EXACT_DIR / "archive.zip"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-label", default="PR106x-lowlevel-brotli")
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--entropy-ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--public-preflight", type=Path, default=DEFAULT_PUBLIC_PREFLIGHT)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--exact-eval", type=Path, default=DEFAULT_EXACT_EVAL)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--exact-archive", type=Path, default=DEFAULT_EXACT_ARCHIVE)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    input_paths = [
        args.scorecard,
        args.entropy_ranking,
        args.candidate_manifest,
        args.exact_eval,
        args.adjudication,
        args.public_preflight,
        args.candidate_archive,
        args.exact_archive,
    ]
    try:
        scorecard = read_json(args.scorecard)
        entropy_ranking = read_json(args.entropy_ranking)
        candidate_manifest = read_json(args.candidate_manifest)
        exact_eval = read_json(args.exact_eval)
        adjudication = read_json(args.adjudication)
        public_preflight = read_json(args.public_preflight)
        candidate_archive = inspect_single_member_archive(args.candidate_archive, repo_root=REPO_ROOT)
        exact_archive = inspect_single_member_archive(args.exact_archive, repo_root=REPO_ROOT)
        review = build_lossless_control_promotion_review(
            target_label=args.target_label,
            scorecard=scorecard,
            entropy_ranking=entropy_ranking,
            candidate_manifest=candidate_manifest,
            exact_eval=exact_eval,
            adjudication=adjudication,
            public_preflight=public_preflight,
            candidate_archive=candidate_archive,
            exact_eval_archive=exact_archive,
            input_paths={
                "scorecard": repo_relative(args.scorecard, REPO_ROOT),
                "entropy_ranking": repo_relative(args.entropy_ranking, REPO_ROOT),
                "candidate_manifest": repo_relative(args.candidate_manifest, REPO_ROOT),
                "exact_eval": repo_relative(args.exact_eval, REPO_ROOT),
                "adjudication": repo_relative(args.adjudication, REPO_ROOT),
                "public_preflight": repo_relative(args.public_preflight, REPO_ROOT),
                "candidate_archive": repo_relative(args.candidate_archive, REPO_ROOT),
                "exact_archive": repo_relative(args.exact_archive, REPO_ROOT),
            },
        )
    except (OSError, ValueError) as exc:
        print(f"FATAL: HNeRV lossless-control promotion review failed: {exc}", file=sys.stderr)
        return 2

    review = attach_tool_run_manifest(
        review,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=input_paths,
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(review), encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(review), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
