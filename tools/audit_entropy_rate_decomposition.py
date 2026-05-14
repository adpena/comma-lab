#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only entropy-rate decomposition from stream counts."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.entropy_rate_decomposition import (  # noqa: E402
    build_entropy_rate_decomposition,
    render_markdown,
)
from tac.repo_io import json_text, read_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON object with a streams list, or a raw streams list.")
    parser.add_argument("--source-label", default="")
    parser.add_argument("--evidence-grade", default="empirical")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = read_json(args.input)
    if isinstance(payload, dict):
        streams = payload.get("streams")
        source_label = args.source_label or str(payload.get("source_label") or "")
        evidence_grade = args.evidence_grade or str(payload.get("evidence_grade") or "empirical")
    else:
        streams = payload
        source_label = args.source_label
        evidence_grade = args.evidence_grade
    manifest = build_entropy_rate_decomposition(
        streams,
        source_label=source_label,
        evidence_grade=evidence_grade,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(manifest), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    if not args.json_out and not args.md_out:
        if args.format == "markdown":
            print(render_markdown(manifest), end="")
        else:
            print(json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
