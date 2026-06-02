# SPDX-License-Identifier: MIT
"""Code-aware marker scanning for implementation/parity contracts.

Marker strings are useful for lightweight source audits, but plain substring
grep is a false-green trap: comments and docstrings can satisfy a gate without
runtime behavior.  This helper strips Python comments and docstrings while
preserving executable identifiers and string constants that are part of data or
receiver grammar.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterable
from pathlib import Path


def read_python_source_for_marker_scan(
    path: str | Path,
    *,
    exclude_names: Iterable[str] = (),
) -> str:
    """Read Python source under ``path`` for marker scans without prose noise."""

    source_path = Path(path)
    excluded = {str(name) for name in exclude_names}
    if source_path.is_dir():
        return "\n".join(
            strip_python_comments_and_docstrings(
                child.read_text(encoding="utf-8", errors="replace")
            )
            for child in sorted(source_path.rglob("*.py"))
            if child.name not in excluded
        )
    if source_path.is_file():
        return strip_python_comments_and_docstrings(
            source_path.read_text(encoding="utf-8", errors="replace")
        )
    return ""


def strip_python_comments_and_docstrings(text: str) -> str:
    """Return source text with comments and Python docstrings removed."""

    docstring_lines = _docstring_lines(text)
    try:
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                continue
            if token.type == tokenize.STRING and token.start[0] in docstring_lines:
                continue
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except tokenize.TokenError:
        return "\n".join(line.split("#", maxsplit=1)[0] for line in text.splitlines())


def _docstring_lines(text: str) -> set[int]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    node_types = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, node_types) or not getattr(node, "body", None):
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start = int(getattr(first, "lineno", 0) or 0)
            end = int(getattr(first, "end_lineno", start) or start)
            lines.update(range(start, end + 1))
    return lines


__all__ = [
    "read_python_source_for_marker_scan",
    "strip_python_comments_and_docstrings",
]
