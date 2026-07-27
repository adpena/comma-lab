# SPDX-License-Identifier: MIT
"""Content-addressed provenance for deterministic training producers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tac.upstream_source_closure import (
    UPSTREAM_SOURCE_CLOSURE_SCHEMA,
    compute_upstream_source_closure_identity,
)


def _git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def capture_training_source_provenance(repo_root: Path | str) -> dict[str, Any]:
    """Capture exact code and portable evaluator-source identities.

    Failure stays explicit: unavailable Git or source custody becomes
    ``"unknown"``.  The dirty bit remains independent from the portable
    evaluator closure so a host-local ``upstream/.venv`` does not erase scorer
    provenance.
    """

    root = Path(repo_root).resolve()
    git_sha = _git(root, "rev-parse", "HEAD") or "unknown"
    git_dirty = bool(_git(root, "status", "--porcelain"))
    try:
        upstream = compute_upstream_source_closure_identity(root)
        upstream_sha256 = str(upstream["closure_sha256"])
    except Exception:
        upstream_sha256 = "unknown"
    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "upstream_snapshot_schema": UPSTREAM_SOURCE_CLOSURE_SCHEMA,
        "upstream_snapshot_sha256": upstream_sha256,
    }


__all__ = ["capture_training_source_provenance"]
