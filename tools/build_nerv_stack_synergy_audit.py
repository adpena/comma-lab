#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the HiNeRV/SNeRV full-stack synergy audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_stack_synergy_audit import (  # noqa: E402
    build_nerv_stack_synergy_audit,
)


def _default_output() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / ".omx" / "research" / f"nerv_stack_synergy_audit_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument(
        "--hard-byte-ceiling",
        type=int,
        action="append",
        default=None,
        help="Repeatable archive byte ceiling. Defaults to 178000 and 216000.",
    )
    parser.add_argument("--memo-limit-per-stack", type=int, default=40)
    parser.add_argument("--marker-limit-per-stack", type=int, default=80)
    parser.add_argument(
        "--hinerv-official-source-audit",
        type=Path,
        help=(
            "Optional hinerv_official_source_parity_audit.v1 JSON. The stack "
            "audit consumes it as false-authority source-forward evidence only."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = args.output or _default_output()
    if not out.is_absolute():
        out = Path(args.repo_root) / out
    if out.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ceilings = tuple(args.hard_byte_ceiling or [178_000, 216_000])
    hinerv_official_source_audit = (
        None
        if args.hinerv_official_source_audit is None
        else json.loads(
            Path(args.hinerv_official_source_audit).read_text(encoding="utf-8")
        )
    )
    audit = build_nerv_stack_synergy_audit(
        repo_root=args.repo_root,
        hard_byte_ceilings=ceilings,
        num_pairs=args.num_pairs,
        memo_limit_per_stack=args.memo_limit_per_stack,
        marker_limit_per_stack=args.marker_limit_per_stack,
        hinerv_official_source_audit=hinerv_official_source_audit,
    )
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "schema": audit["schema"],
                "output": out.as_posix(),
                "score_claim": audit["score_claim"],
                "ready_for_exact_eval_dispatch": audit[
                    "ready_for_exact_eval_dispatch"
                ],
                "blocker_count": len(audit["blockers"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
