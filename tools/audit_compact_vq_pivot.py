#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit whether compact VQ should keep receiving score-lowering budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.compact_vq_pivot_audit import (  # noqa: E402
    COMPACT_VQ_PIVOT_AUDIT_SCHEMA,
    build_compact_vq_pivot_audit,
    write_compact_vq_pivot_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument(
        "--upstream-dir",
        default=Path("/Users/adpena/Projects/pact/upstream"),
        type=Path,
        help="Pinned upstream contest eval tree to hash and inspect.",
    )
    parser.add_argument(
        "--mlx-profile",
        action="append",
        default=[],
        type=Path,
        help="Full-video MLX section/value profile JSON. Repeatable.",
    )
    parser.add_argument("--family", default="pact_nerv_vq")
    parser.add_argument(
        "--max-mlx-score-for-local-replay",
        default=0.5,
        type=float,
        help="Above this advisory score, route to durable demotion before replay spend.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_compact_vq_pivot_audit(
        repo_root=args.repo_root,
        upstream_dir=args.upstream_dir,
        mlx_profile_paths=args.mlx_profile,
        family=args.family,
        max_mlx_score_for_local_replay=args.max_mlx_score_for_local_replay,
    )
    out = _resolve(args.output, base=args.repo_root)
    write_compact_vq_pivot_audit(
        output_path=out,
        audit=audit,
        allow_overwrite=args.force,
    )
    print(
        json.dumps(
            {
                "schema": COMPACT_VQ_PIVOT_AUDIT_SCHEMA,
                "output": out.as_posix(),
                "family": audit["family"],
                "verdict": audit["verdict"],
                "best_full_video_mlx_score": audit["profile_signal"][
                    "best_full_video_mlx_score"
                ],
                "blockers": audit["blockers"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit_compact_vq_pivot failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
