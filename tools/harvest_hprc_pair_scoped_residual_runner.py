#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest one executed HPRC pair-scoped residual runner row."""

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

from tac.substrates.hprc.pair_scoped_residual_harvest import (  # noqa: E402
    build_pair_scoped_residual_runner_harvest,
    write_pair_scoped_residual_runner_harvest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--proof-root", action="append", type=Path, default=[])
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    output = _resolve(args.output, base=repo_root)
    harvest = build_pair_scoped_residual_runner_harvest(
        runner_plan_path=args.runner_plan,
        candidate_id=str(args.candidate_id),
        proof_roots=list(args.proof_root),
        repo_root=repo_root,
    )
    write_pair_scoped_residual_runner_harvest(
        output_path=output,
        harvest=harvest,
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "archive_sha256": harvest["archive"]["sha256"],
                "receiver_proof_binding": harvest["receiver_proof_binding"]["status"],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
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
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
