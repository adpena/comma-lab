# SPDX-License-Identifier: MIT
"""Deterministic local cache for expensive preflight smoke checks.

The cache is intentionally local state under ``.omx/state``. It accelerates
repeat DX loops but never creates score evidence and never changes dispatch
readiness semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import json_text, repo_relative, sha256_file  # noqa: E402

CACHE_SCHEMA_VERSION = 1
CACHE_DIR = REPO / ".omx" / "state" / "preflight_cache"


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a stable fingerprint for one required input file."""
    return {
        "path": repo_relative(path, REPO),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_cache_key(
    *,
    name: str,
    files: list[Path],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-comparable cache key from files and config."""
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "name": name,
        "config": config or {},
        "files": [file_fingerprint(path) for path in files],
    }


def load_valid_cache(name: str, key: dict[str, Any]) -> dict[str, Any] | None:
    """Return a cached payload only when the full key and pass status match."""
    path = CACHE_DIR / f"{name}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if payload.get("key") != key:
        return None
    if payload.get("passed") is not True:
        return None
    return payload


def write_pass_cache(name: str, key: dict[str, Any], result: dict[str, Any]) -> None:
    """Persist a successful local preflight result."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "key": key,
        "passed": True,
        "result": result,
    }
    (CACHE_DIR / f"{name}.json").write_text(json_text(payload), encoding="utf-8")
