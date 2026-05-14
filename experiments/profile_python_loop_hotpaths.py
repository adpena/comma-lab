#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile Python loop hotpaths for vectorization triage.

This is an engineering profiler, not score evidence. It statically scans
project Python files, ranks loops that look likely to touch large mask/frame/
atom tensors, and emits deterministic JSON for the optimization control plane.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "python_loop_hotpath_profile_v1"
TOOL = "experiments/profile_python_loop_hotpaths.py"
DEFAULT_ROOTS = ("experiments", "src/tac", "submissions/robust_current")
EXCLUDED_PART_NAMES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}
EXCLUDED_RELATIVE_PREFIXES = (
    Path("experiments/results"),
    Path("reports/raw"),
    Path("src/tac/tests"),
)
LOW_PRIORITY_PATH_HINTS = (
    "preflight.py",
    "training.py",
)
TRAINING_LOOP_HEADERS = (
    "for epoch in range",
    "for batch",
    "for step",
)
HOT_KEYWORDS = (
    "pixel",
    "mask",
    "frame",
    "row",
    "col",
    "atom",
    "candidate",
    "policy",
    "pair",
    "height",
    "width",
    "class",
    "span",
    "run",
    "tensor",
)
VECTORIZED_HINTS = (
    "np.",
    "numpy.",
    "torch.",
    "bincount",
    "searchsorted",
    "argmax",
    "where",
    "stack",
    "einsum",
    "unique",
)


@dataclass(frozen=True)
class LoopRecord:
    path: Path
    line: int
    end_line: int
    loop_kind: str
    header: str
    body_text: str
    nested_loop_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _iter_python_files(roots: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            resolved = root.resolve()
            if not _is_excluded_path(resolved):
                paths.add(resolved)
        elif root.is_dir():
            for path in root.rglob("*.py"):
                resolved = path.resolve()
                if _is_excluded_path(resolved):
                    continue
                paths.add(resolved)
    return sorted(paths, key=lambda item: str(item))


def _is_excluded_path(path: Path) -> bool:
    if set(path.parts) & EXCLUDED_PART_NAMES:
        return True
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return False
    return any(rel == prefix or prefix in rel.parents for prefix in EXCLUDED_RELATIVE_PREFIXES)


def _node_text(lines: list[str], node: ast.AST) -> str:
    start = max(int(getattr(node, "lineno", 1)) - 1, 0)
    end = min(int(getattr(node, "end_lineno", start + 1)), len(lines))
    return "\n".join(lines[start:end])


def _loop_header(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:240]


def _loop_records_for_file(path: Path) -> list[LoopRecord]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    records: list[LoopRecord] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            continue
        body_text = _node_text(lines, node)
        nested = sum(
            1
            for child in ast.walk(node)
            if child is not node and isinstance(child, (ast.For, ast.AsyncFor, ast.While))
        )
        records.append(
            LoopRecord(
                path=path,
                line=int(getattr(node, "lineno", 0)),
                end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
                loop_kind=type(node).__name__,
                header=_loop_header(body_text),
                body_text=body_text,
                nested_loop_count=nested,
            )
        )
    return records


def _score_loop(record: LoopRecord) -> dict[str, Any]:
    haystack = f"{record.path} {record.header} {record.body_text}".lower()
    keyword_hits = [keyword for keyword in HOT_KEYWORDS if keyword in haystack]
    vector_hints = [hint for hint in VECTORIZED_HINTS if hint.lower() in haystack]
    line_count = max(1, record.end_line - record.line + 1)
    score = 0
    score += 3 * len(keyword_hits)
    score += 6 * int(record.nested_loop_count > 0)
    score += 3 * record.nested_loop_count
    score += 2 * int("range(" in haystack)
    score += 2 * int("enumerate(" in haystack)
    score += 2 * int(line_count >= 12)
    score -= min(5, len(vector_hints))
    if "experiments/results" in str(record.path):
        score -= 10
    if "archive/" in str(record.path):
        score -= 4
    if any(hint in str(record.path) for hint in LOW_PRIORITY_PATH_HINTS):
        score -= 30
    if any(record.header.lower().startswith(header) for header in TRAINING_LOOP_HEADERS):
        score -= 12
    return {
        "body_line_count": line_count,
        "header": record.header,
        "hot_keywords": keyword_hits,
        "line": record.line,
        "end_line": record.end_line,
        "loop_kind": record.loop_kind,
        "nested_loop_count": record.nested_loop_count,
        "path": str(record.path),
        "static_hotpath_score": int(score),
        "vectorized_hint_tokens": vector_hints,
        "vectorization_recommendation": _recommendation(record, keyword_hits, vector_hints),
    }


def _recommendation(record: LoopRecord, keyword_hits: list[str], vector_hints: list[str]) -> str:
    header = record.header.lower()
    body = record.body_text.lower()
    if any(hint in str(record.path) for hint in LOW_PRIORITY_PATH_HINTS):
        return "guard/training infrastructure; optimize only with measured wall-clock evidence"
    if any(header.startswith(prefix) for prefix in TRAINING_LOOP_HEADERS):
        return "training-control loop; use profiler traces before static rewrites"
    if record.nested_loop_count and {"pixel", "row", "col", "width", "height"} & set(keyword_hits):
        return "inspect for NumPy/Torch vectorization or chunked array kernels before full-resolution runs"
    if "for start in range" in header and ("batch" in header or "chunk" in header):
        return "likely intentional batching loop; optimize only if profiling shows overhead"
    if "struct." in body or "pack" in body or "unpack" in body:
        return "wire-format loop; prefer correctness tests before micro-optimization"
    if vector_hints:
        return "already uses vectorized primitives; check memory layout/chunk size before rewriting"
    return "review if this file is on an active dispatch/build path"


def build_profile(
    *,
    roots: Iterable[Path],
    output_json: Path,
    limit: int = 80,
) -> dict[str, Any]:
    paths = _iter_python_files(roots)
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            records = _loop_records_for_file(path)
        except SyntaxError as exc:
            rows.append(
                {
                    "path": str(path),
                    "parse_error": f"{type(exc).__name__}: {exc}",
                    "static_hotpath_score": -999,
                }
            )
            continue
        for record in records:
            row = _score_loop(record)
            row["file_sha256"] = _sha256_file(path)
            rows.append(row)
    rows.sort(
        key=lambda item: (
            -int(item.get("static_hotpath_score", -999)),
            str(item.get("path", "")),
            int(item.get("line", 0)),
        )
    )
    payload = {
        "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "evidence_grade": "engineering_profile",
        "input_roots": [str(Path(root).resolve()) for root in roots],
        "limit": int(limit),
        "loop_count": len(rows),
        "no_score_claim": True,
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "top_hotpaths": rows[:limit],
        "tool": TOOL,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(_json_bytes(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args(argv)
    roots = args.root or [Path(root) for root in DEFAULT_ROOTS]
    payload = build_profile(roots=roots, output_json=args.output_json, limit=args.limit)
    print(
        json.dumps(
            {
                "loop_count": payload["loop_count"],
                "output_json": str(args.output_json),
                "schema": payload["schema"],
                "score_claim": payload["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
