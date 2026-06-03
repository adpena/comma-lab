#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the false-authority HiNeRV/SNeRV source-parity contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.analysis.nerv_source_parity_contract import (
    build_nerv_source_parity_contract,
    render_nerv_source_parity_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--snerv-official-source-audit",
        type=Path,
        help="Optional snerv_official_source_parity_audit.v1 JSON to embed.",
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=("hi_nerv", "snerv"),
        help="Family to include. Repeatable; defaults to both.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    families = tuple(args.family or ("hi_nerv", "snerv"))
    snerv_official_source_audit = (
        None
        if args.snerv_official_source_audit is None
        else json.loads(args.snerv_official_source_audit.read_text(encoding="utf-8"))
    )
    report = build_nerv_source_parity_contract(
        repo_root=args.repo_root,
        families=families,
        snerv_official_source_audit=snerv_official_source_audit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_nerv_source_parity_markdown(report),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "authority": report["authority"],
                "families": report["families"],
                "required_for_long_training_ready": report["required_for_long_training_ready"],
                "blocker_count": len(report["blockers"]),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
                "output_json": args.output_json.as_posix(),
                "output_md": args.output_md.as_posix() if args.output_md else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
