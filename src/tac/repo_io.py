"""Small deterministic IO helpers for local tooling.

These helpers intentionally avoid provider state and process mutation. They are
for repository-local audit, readiness, profiling, and manifest tools that need
stable JSON bytes, SHA-256 digests, and repo-relative path display.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def json_text(payload: Any) -> str:
    """Return canonical pretty JSON text with a trailing newline."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def json_line(payload: Any) -> str:
    """Return canonical one-line JSON text with a trailing newline."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def read_json(path: str | Path) -> Any:
    """Read JSON from ``path`` using UTF-8."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    """Write canonical pretty JSON to ``path`` using UTF-8."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json_text(payload), encoding="utf-8")


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of one file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of an in-memory byte payload."""

    return sha256(data).hexdigest()


def repo_relative(path: str | Path, repo_root: str | Path) -> str:
    """Return a POSIX path relative to ``repo_root`` when possible."""

    candidate = Path(path)
    root = Path(repo_root)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


__all__ = [
    "json_line",
    "json_text",
    "read_json",
    "repo_relative",
    "sha256_bytes",
    "sha256_file",
    "write_json",
]
