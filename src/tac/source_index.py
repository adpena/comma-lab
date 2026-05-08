"""Shared source-tree inventory and parser cache for developer scanners.

This module is intentionally Python-first. It gives preflight-style scanners a
normal object to share file discovery, source text, and AST parses within one
process without monkeypatching ``pathlib``. Rust can replace the inventory
backend later, but the scanner contract stays small and testable here.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_CURRENT_SOURCE_INDEX: contextvars.ContextVar[SourceIndex | None] = contextvars.ContextVar(
    "tac_current_source_index",
    default=None,
)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _path_key(path: Path) -> str:
    return _safe_resolve(path).as_posix()


@dataclass
class SourceIndex:
    """Process-local source index for one repo root.

    ``SourceIndex`` is scoped by callers, usually through
    :func:`source_index_context`. It caches only read-only scanner inputs:
    recursive file lists, source text, and parsed Python ASTs. It deliberately
    does not watch for file mutations; create a fresh index for a fresh scan.
    """

    root: Path
    skip_parts: frozenset[str] = frozenset({"__pycache__", "comma_lab_public_export"})
    _file_cache: dict[tuple[tuple[str, ...], str], tuple[Path, ...]] = field(default_factory=dict, init=False)
    _text_cache: dict[tuple[str, str | None, str | None], str] = field(default_factory=dict, init=False)
    _ast_cache: dict[str, ast.AST] = field(default_factory=dict, init=False)
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "file_list_hits": 0,
            "file_list_misses": 0,
            "text_hits": 0,
            "text_misses": 0,
            "ast_hits": 0,
            "ast_misses": 0,
        },
        init=False,
    )

    def __post_init__(self) -> None:
        self.root = _safe_resolve(Path(self.root))

    def files(self, dirs: Sequence[str | Path], *, pattern: str) -> tuple[Path, ...]:
        """Return sorted files under ``dirs`` matching ``pattern``.

        Relative directories are resolved under ``root``. Paths containing any
        configured ``skip_parts`` are excluded, matching the current preflight
        OSS-mirror and ``__pycache__`` policy.
        """

        dir_keys = tuple(self._dir_key(item) for item in dirs)
        key = (dir_keys, pattern)
        cached = self._file_cache.get(key)
        if cached is not None:
            self._stats["file_list_hits"] += 1
            return cached

        self._stats["file_list_misses"] += 1
        paths: dict[str, Path] = {}
        for item in dirs:
            base = self._resolve_dir(item)
            if not base.exists():
                continue
            for path in base.rglob(pattern):
                if self._should_skip(path):
                    continue
                if not path.is_file():
                    continue
                paths[_path_key(path)] = path
        out = tuple(paths[key] for key in sorted(paths))
        self._file_cache[key] = out
        return out

    def read_text(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        """Read and cache source text for ``path``."""

        target = Path(path)
        key = (_path_key(target), encoding, errors)
        cached = self._text_cache.get(key)
        if cached is not None:
            self._stats["text_hits"] += 1
            return cached

        self._stats["text_misses"] += 1
        text = target.read_text(encoding=encoding, errors=errors)
        self._text_cache[key] = text
        return text

    def python_ast(self, path: str | Path) -> ast.AST:
        """Parse and cache a Python AST for ``path``."""

        target = Path(path)
        key = _path_key(target)
        cached = self._ast_cache.get(key)
        if cached is not None:
            self._stats["ast_hits"] += 1
            return cached

        self._stats["ast_misses"] += 1
        tree = ast.parse(self.read_text(target), filename=str(target))
        self._ast_cache[key] = tree
        return tree

    def repo_relative(self, path: str | Path) -> str:
        """Return a stable POSIX path relative to this index root when possible."""

        target = _safe_resolve(Path(path))
        try:
            return target.relative_to(self.root).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss counters and current cache sizes."""

        return {
            **self._stats,
            "file_list_cache_entries": len(self._file_cache),
            "text_cache_entries": len(self._text_cache),
            "ast_cache_entries": len(self._ast_cache),
        }

    def _dir_key(self, item: str | Path) -> str:
        path = Path(item)
        if path.is_absolute():
            return _path_key(path)
        return path.as_posix().rstrip("/")

    def _resolve_dir(self, item: str | Path) -> Path:
        path = Path(item)
        if path.is_absolute():
            return path
        return self.root / path

    def _should_skip(self, path: Path) -> bool:
        parts = set(path.parts)
        return any(part in parts for part in self.skip_parts)


def get_current_source_index(repo_root: str | Path | None = None) -> SourceIndex | None:
    """Return the active source index when it matches ``repo_root``."""

    index = _CURRENT_SOURCE_INDEX.get()
    if index is None or repo_root is None:
        return index
    if index.root != _safe_resolve(Path(repo_root)):
        return None
    return index


@contextlib.contextmanager
def source_index_context(repo_root: str | Path) -> Iterator[SourceIndex]:
    """Install a shared source index for scanners in this context."""

    existing = get_current_source_index(repo_root)
    if existing is not None:
        yield existing
        return

    index = SourceIndex(Path(repo_root))
    token = _CURRENT_SOURCE_INDEX.set(index)
    try:
        yield index
    finally:
        _CURRENT_SOURCE_INDEX.reset(token)


__all__ = ["SourceIndex", "get_current_source_index", "source_index_context"]
