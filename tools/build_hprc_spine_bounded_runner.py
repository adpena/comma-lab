#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the bounded runner plan for compact HPRC representation spine rows."""

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

from tac.substrates.hprc.spine_bounded_runner import (  # noqa: E402
    HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
    build_spine_bounded_runner_plan,
    write_spine_bounded_runner_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-report", required=True, type=Path)
    parser.add_argument(
        "--mlx-profile",
        action="append",
        default=[],
        type=Path,
        help="hprc_mlx_component_neutralization_profile.json. Repeatable.",
    )
    parser.add_argument(
        "--exact-gate-report",
        action="append",
        default=[],
        type=Path,
        help="Exact gate bridge/report JSON to attach when available. Repeatable.",
    )
    parser.add_argument(
        "--receiver-proof",
        action="append",
        default=[],
        type=Path,
        help="Generated inflate receiver proof JSON to attach when available. Repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_spine_bounded_runner_plan(
        acquisition_report_path=args.acquisition_report,
        repo_root=args.repo_root,
        mlx_profile_paths=args.mlx_profile,
        receiver_proof_report_paths=args.receiver_proof,
        exact_gate_report_paths=args.exact_gate_report,
    )
    out = _resolve(args.output, base=args.repo_root)
    write_spine_bounded_runner_plan(
        output_path=out,
        plan=plan,
        allow_overwrite=args.force,
    )
    print(
        json.dumps(
            {
                "schema": HPRC_SPINE_BOUNDED_RUNNER_PLAN_SCHEMA,
                "output": out.as_posix(),
                "compact_base_sweep_rows": len(plan["compact_base_sweep_rows"]),
                "section_value_rows": len(plan["section_value_rows"]),
                "selected_runner_rows": len(plan["selected_runner_rows"]),
                "blockers": plan["blockers"],
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
        print(f"build_hprc_spine_bounded_runner failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
