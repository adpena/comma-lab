# SPDX-License-Identifier: MIT
"""Bootstrap helpers for repository-local tools.

Root-level tools should not each hand-roll repo-root discovery and `sys.path`
mutation. Keep this file dependency-light so it can be imported before the
package itself is importable.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root_from_tool(path: str | Path) -> Path:
    """Return the repository root for a file under ``tools/``."""

    return Path(path).resolve().parents[1]


def ensure_repo_imports(repo_root: str | Path) -> None:
    """Add repo root and ``src`` to ``sys.path`` if absent."""

    root = Path(repo_root).resolve()
    for candidate in (root, root / "src"):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def prepend_paths(*paths: str | Path) -> None:
    """Prepend import paths in the same order provided, without duplicates."""

    for candidate in reversed(paths):
        value = str(Path(candidate))
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


__all__ = ["ensure_repo_imports", "prepend_paths", "repo_root_from_tool"]
