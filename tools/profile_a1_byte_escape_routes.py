#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile A1 Rule #6 byte-escape routes.

Planning-only: reads the exact A1 archive, writes deterministic JSON/Markdown
evidence, and does not emit a candidate archive or score claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.a1_byte_escape_profile import (  # noqa: E402
    build_a1_byte_escape_profile,
    render_a1_byte_escape_markdown,
)
from tac.repo_io import write_json  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT / "submissions/a1/archive.zip"
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / ".omx/research/a1_rule6_byte_escape_profile_20260517_codex.json"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT / ".omx/research/a1_rule6_byte_escape_profile_20260517_codex.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    profile = build_a1_byte_escape_profile(args.archive, repo_root=repo_root)

    write_json(args.output_json, profile)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_a1_byte_escape_markdown(profile), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
