#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add missing archive-embedded runtime files with deterministic ZIP custody."""

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

from tac.substrates.hprc.runtime_closure_repair import (  # noqa: E402
    HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA,
    repair_embedded_runtime_zip_closure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output-archive", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--add-member",
        action="append",
        default=[],
        metavar="ZIP_MEMBER=SOURCE_PATH",
    )
    parser.add_argument(
        "--replace-member",
        action="append",
        default=[],
        metavar="ZIP_MEMBER=SOURCE_PATH",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    members = {
        member: _resolve(Path(path), repo_root=repo_root)
        for member, path in (_split(raw) for raw in args.add_member)
    }
    replacements = {
        member: _resolve(Path(path), repo_root=repo_root)
        for member, path in (_split(raw) for raw in args.replace_member)
    }
    if not members and not replacements:
        raise ValueError("at least one --add-member or --replace-member is required")
    report = repair_embedded_runtime_zip_closure(
        source_archive=_resolve(args.source_archive, repo_root=repo_root),
        output_archive=_resolve(args.output_archive, repo_root=repo_root),
        add_members=members,
        replace_members=replacements,
        report_path=_resolve(args.report, repo_root=repo_root),
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "schema": HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA,
                "report_path": report["report_path"],
                "output_archive": report["output_archive"],
                "added_member_count": len(report["added_members"]),
                "replaced_member_count": len(report["replaced_members"]),
                "blockers": report["blockers"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0 if not report["blockers"] else 1


def _split(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"expected ZIP_MEMBER=SOURCE_PATH: {raw!r}")
    member, path = raw.split("=", 1)
    if not member.strip() or not path.strip():
        raise ValueError(f"expected nonempty ZIP_MEMBER=SOURCE_PATH: {raw!r}")
    return member.strip(), path.strip()


def _resolve(path: Path, *, repo_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"repair_embedded_runtime_zip_closure failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
