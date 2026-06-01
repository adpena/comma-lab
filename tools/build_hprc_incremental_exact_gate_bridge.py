#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an exact-gate bridge from an HPRC incremental execution report."""

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

from tac.substrates.hprc.exact_gate import (  # noqa: E402
    build_hprc_incremental_exact_gate_bridge,
    write_hprc_incremental_exact_gate_bridge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    output = _resolve(args.output, base=repo_root)
    bridge = build_hprc_incremental_exact_gate_bridge(
        execution_report_path=args.execution_report,
        repo_root=repo_root,
    )
    write_hprc_incremental_exact_gate_bridge(
        output_path=output,
        bridge=bridge,
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "candidate_id": bridge["candidate_id"],
                "archive_sha256": bridge["archive"]["sha256"],
                "dispatchable_after_lane_claim": bridge["exact_dispatch_plan"][
                    "dispatchable_after_lane_claim"
                ],
                "preclaim_blockers": bridge["exact_dispatch_plan"][
                    "preclaim_blockers"
                ],
                "ready_for_exact_eval_dispatch": bridge[
                    "ready_for_exact_eval_dispatch"
                ],
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
