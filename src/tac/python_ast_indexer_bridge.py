"""Optional Rust bridge for preflight Python AST indexing.

The Python preflight implementation remains the oracle. This bridge only uses
``runtime-rs/crates/python-ast-indexer`` when an already-built binary is
available, and callers must tolerate fallback when it is absent or unusable.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class PythonAstIndexerBridgeError(RuntimeError):
    """Raised when the optional Rust AST indexer is present but unusable."""


def resolve_python_ast_indexer_binary(
    binary_path: str | Path | None = None,
) -> Path | None:
    """Resolve an already-built ``python-ast-indexer`` binary if present."""

    if binary_path is not None:
        candidate = Path(binary_path)
        return candidate if _is_executable_file(candidate) else None

    candidates: list[Path] = []
    env_bin = os.environ.get("TAC_PYTHON_AST_INDEXER_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "runtime-rs" / "target" / "release" / "python-ast-indexer",
        ]
    )
    if os.environ.get("TAC_PYTHON_AST_INDEXER_ALLOW_DEBUG") == "1":
        candidates.append(
            repo_root / "runtime-rs" / "target" / "debug" / "python-ast-indexer"
        )
    for candidate in candidates:
        if _is_executable_file(candidate):
            return candidate
    return None


def index_python_top_level_names_native(
    paths: list[Path],
    *,
    binary_path: str | Path | None = None,
    timeout_s: float = 30.0,
    chunk_size: int = 256,
    use_cache: bool = True,
) -> dict[Path, set[str]]:
    """Return top-level-name sets from the optional Rust AST indexer.

    Parse failures are omitted from the returned map so callers can fall back
    to Python for those files. A nonzero exit is accepted because the Rust CLI
    intentionally returns 1 when any batch member failed to parse while still
    emitting structured JSON for the successful members.
    """

    binary = resolve_python_ast_indexer_binary(binary_path)
    if binary is None:
        return {}
    if not paths:
        return {}
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    cache = _load_incremental_cache(binary) if use_cache else {}
    cache_changed = False
    out: dict[Path, set[str]] = {}
    misses: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        key = _cache_key(binary, path)
        row = cache.get(key)
        if isinstance(row, dict) and _cache_row_fresh(path, row):
            names = row.get("top_level_names")
            if isinstance(names, list):
                out[path] = {name for name in names if isinstance(name, str)}
                continue
        misses.append(path)

    for start in range(0, len(misses), chunk_size):
        chunk = misses[start:start + chunk_size]
        proc = subprocess.run(  # subprocess-no-check-OK: returncode 1 can carry parse-failure JSON.
            [str(binary), "--batch", *[str(path) for path in chunk]],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode not in (0, 1):
            raise PythonAstIndexerBridgeError(
                f"python-ast-indexer exited {proc.returncode}: {proc.stderr.strip()}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise PythonAstIndexerBridgeError(
                "python-ast-indexer did not emit valid JSON"
            ) from exc
        if not isinstance(payload, list):
            raise PythonAstIndexerBridgeError(
                "python-ast-indexer batch JSON must be a list"
            )
        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("parse_ok") is not True:
                continue
            raw_path = row.get("path")
            raw_names = row.get("top_level_names")
            if not isinstance(raw_path, str) or not isinstance(raw_names, list):
                continue
            names = {name for name in raw_names if isinstance(name, str)}
            path = Path(raw_path).resolve()
            out[path] = names
            if use_cache:
                key = _cache_key(binary, path)
                cache[key] = _cache_row(path, sorted(names))
                cache_changed = True
    if use_cache and cache_changed:
        _store_incremental_cache(binary, cache)
    return out


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _cache_path(binary: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    stem = f"python_ast_indexer_{binary.resolve().stat().st_mtime_ns:x}.json"
    return root / ".omx" / "cache" / stem


def _cache_key(binary: Path, path: Path) -> str:
    return f"{binary.resolve()}::{path.resolve()}"


def _cache_row(path: Path, names: list[str]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "top_level_names": names,
        "cached_at": time.time(),
    }


def _cache_row_fresh(path: Path, row: dict[str, Any]) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return row.get("size") == stat.st_size and row.get("mtime_ns") == stat.st_mtime_ns


def _load_incremental_cache(binary: Path) -> dict[str, Any]:
    path = _cache_path(binary)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _store_incremental_cache(binary: Path, cache: dict[str, Any]) -> None:
    path = _cache_path(binary)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, sort_keys=True))
    except OSError:
        return


__all__ = [
    "PythonAstIndexerBridgeError",
    "index_python_top_level_names_native",
    "resolve_python_ast_indexer_binary",
]
