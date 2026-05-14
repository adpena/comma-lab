#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI wrapper for comma-lab research-state tracking audits.

# no-argparse-OK: thin re-export of comma_lab.research_state.main(), which owns
# the argparse surface. --help still works because the delegate parses sys.argv.
"""

from __future__ import annotations

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.research_state import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
