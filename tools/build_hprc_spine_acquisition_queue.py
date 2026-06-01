#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build value-per-byte acquisition rows from HPRC representation spine manifests."""

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

from tac.substrates.hprc.spine_acquisition import (  # noqa: E402
    DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
    HPRC_SPINE_ACQUISITION_REPORT_SCHEMA,
    build_spine_acquisition_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection-manifest",
        action="append",
        required=True,
        type=Path,
        help="hprc_representation_spine_manifest.json path. Repeatable.",
    )
    parser.add_argument(
        "--hard-byte-ceiling",
        action="append",
        type=int,
        help=(
            "Charged archive-byte ceiling for base renderers. Repeatable. "
            f"Defaults to {','.join(str(v) for v in DEFAULT_BASE_RENDERER_BYTE_CEILINGS)}."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    manifests = [_resolve(path, base=repo_root) for path in args.projection_manifest]
    ceilings = tuple(args.hard_byte_ceiling or DEFAULT_BASE_RENDERER_BYTE_CEILINGS)
    output = _resolve(args.output, base=repo_root)
    if output.exists() and not args.force:
        raise ValueError(f"output already exists: {output}")
    report = build_spine_acquisition_report(
        projection_manifest_paths=manifests,
        hard_byte_ceilings=ceilings,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": HPRC_SPINE_ACQUISITION_REPORT_SCHEMA,
                "output": output.as_posix(),
                "row_count": report["row_count"],
                "best_under_each_ceiling": report["best_under_each_ceiling"],
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
        print(f"build_hprc_spine_acquisition_queue failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
