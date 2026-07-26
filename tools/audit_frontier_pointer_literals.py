#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory hardcoded frontier-pointer literals without rewriting history.

The canonical live target is
``.omx/state/canonical_frontier_pointer.json::effective_frontier``.  This
audit separates executable assignments whose names imply frontier/pointer
semantics from stale textual references in old diagnostics and prose.  It is
an inventory and a scoped strict gate; historical files are never edited by
the audit itself.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (Path("src"), Path("tools"), Path("experiments"))
DEFAULT_POINTER = Path(".omx/state/canonical_frontier_pointer.json")
_POINTER_NAME = re.compile(
    r"(?:frontier|pointer)|(?:^|_)(?:target_?score|score_?target|score_?to_?beat|competitive_?score)$",
    re.IGNORECASE,
)
_COMPETITIVE_POINTER_NAME = re.compile(
    r"^(?:(?:current|best|local|our|public|upstream|effective|canonical)_)*"
    r"(?:frontier(?:_score|_pointer)?|pointer(?:_score)?|target_?score|score_?target|"
    r"score_?to_?beat|competitive_?score)$",
    re.IGNORECASE,
)
_SCORE_VALUE_NAME = re.compile(r"^(?:candidate_|final_|current_|projected_)?score$", re.IGNORECASE)
_RETIRED_TEXT = re.compile(r"(?<![0-9])0\.191(?:0|08|09|10|[0-9]*)?(?![0-9])")
_EXCLUDED_TREE_PARTS = frozenset({".git", ".venv", "__pycache__", "build", "dist", "node_modules"})
_POINTER_SCHEMA_VERSION = "canonical_frontier_pointer_v1_20260519"
_SELECTION_RULE = (
    "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, "
    "upstream_official_leaderboard.best_entry)"
)


class PointerLiteralAuditError(RuntimeError):
    """Raised when inputs are missing, ambiguous, or syntactically invalid."""


def _validate_local_anchor(value: Any, *, field: str, axis: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PointerLiteralAuditError(f"canonical pointer {field} must be an object or null")
    score = _finite_positive(value.get("score"), f"canonical pointer {field}.score")
    if value.get("axis") != axis:
        raise PointerLiteralAuditError(f"canonical pointer {field} has the wrong exact axis")
    expected_grade = "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]"
    if value.get("evidence_grade") != expected_grade:
        raise PointerLiteralAuditError(f"canonical pointer {field} lacks exact-axis evidence grade")
    archive_sha256 = value.get("archive_sha256")
    if not isinstance(archive_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None:
        raise PointerLiteralAuditError(f"canonical pointer {field} lacks a canonical archive SHA-256")
    hardware = value.get("hardware_substrate")
    if not isinstance(hardware, str) or not hardware:
        raise PointerLiteralAuditError(f"canonical pointer {field} lacks hardware custody")
    for optional in ("lane_id", "measured_at_utc", "source_path"):
        if value.get(optional) is not None and not isinstance(value.get(optional), str):
            raise PointerLiteralAuditError(f"canonical pointer {field}.{optional} has the wrong type")
    if not isinstance(value.get("extra"), Mapping):
        raise PointerLiteralAuditError(f"canonical pointer {field}.extra must be an object")
    return {**dict(value), "score": score}


def _validate_public_entry(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PointerLiteralAuditError(f"{label} must be an object")
    score = _finite_positive(value.get("score"), f"{label}.score")
    rank = value.get("rank")
    if type(rank) is not int or rank < 1:
        raise PointerLiteralAuditError(f"{label}.rank must be a positive exact integer")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise PointerLiteralAuditError(f"{label}.name must be nonempty")
    pr_number = value.get("pr_number")
    if pr_number is not None and (type(pr_number) is not int or pr_number < 1):
        raise PointerLiteralAuditError(f"{label}.pr_number has the wrong type")
    pr_url = value.get("pr_url")
    if pr_url is not None and not isinstance(pr_url, str):
        raise PointerLiteralAuditError(f"{label}.pr_url has the wrong type")
    return {**dict(value), "score": score}


def _validated_public_best(snapshot: Any, *, label: str) -> tuple[dict[str, Any], str | None] | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise PointerLiteralAuditError(f"{label} must be an object or null")
    status = snapshot.get("fetch_status")
    if status != "ok":
        if status not in {"network_failure", "parse_failure"}:
            raise PointerLiteralAuditError(f"{label}.fetch_status is not canonical")
        entries = snapshot.get("entries")
        if entries != []:
            raise PointerLiteralAuditError(f"{label} failure snapshot must not carry current entries")
        cached = _validated_public_best(snapshot.get("cached_snapshot"), label=f"{label}.cached_snapshot")
        if cached is None:
            return None
        cached_at = snapshot.get("cached_snapshot_at_utc")
        if not isinstance(cached_at, str) or cached_at != cached[1]:
            raise PointerLiteralAuditError(f"{label}.cached_snapshot_at_utc does not bind the cached snapshot")
        return cached[0], cached_at
    if snapshot.get("source") != "official_leaderboard":
        raise PointerLiteralAuditError(f"{label}.source is not the official leaderboard")
    entries_raw = snapshot.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise PointerLiteralAuditError(f"{label}.entries must be a nonempty list")
    entries = [
        _validate_public_entry(row, label=f"{label}.entries[{index}]")
        for index, row in enumerate(entries_raw)
    ]
    if len({row["rank"] for row in entries}) != len(entries):
        raise PointerLiteralAuditError(f"{label}.entries contains duplicate ranks")
    best = min(entries, key=lambda row: (float(row["score"]), int(row["rank"])))
    stated = _validate_public_entry(snapshot.get("best_entry"), label=f"{label}.best_entry")
    for key in ("score", "rank", "name", "pr_number", "pr_url"):
        if stated.get(key) != best.get(key):
            raise PointerLiteralAuditError(f"{label}.best_entry is not the semantic minimum")
    if snapshot.get("entry_count") != len(entries) or snapshot.get("score_precision") != "official_display":
        raise PointerLiteralAuditError(f"{label} leaderboard metadata is inconsistent")
    fetched_at = snapshot.get("fetched_at_utc")
    if not isinstance(fetched_at, str) or not fetched_at:
        raise PointerLiteralAuditError(f"{label}.fetched_at_utc is absent")
    return best, fetched_at


def validate_canonical_pointer_payload(pointer: Any) -> dict[str, Any]:
    """Strictly rederive and validate the competitive effective-frontier row."""

    if not isinstance(pointer, Mapping):
        raise PointerLiteralAuditError("canonical pointer root must be an object")
    required = {
        "schema_version",
        "our_local_frontier_contest_cpu",
        "our_local_frontier_contest_cuda",
        "submitted_pr_number_for_current_frontier",
        "upstream_leaderboard_snapshot",
        "upstream_leaderboard_snapshot_at_utc",
        "last_refreshed_utc",
        "auto_update_on_dispatch_completion",
        "pointer_refresh_command",
        "refresh_provenance",
        "effective_frontier",
    }
    if not required.issubset(pointer):
        raise PointerLiteralAuditError("canonical pointer is missing required schema fields")
    if pointer.get("schema_version") != _POINTER_SCHEMA_VERSION:
        raise PointerLiteralAuditError("canonical pointer schema version is unsupported")
    if not isinstance(pointer.get("last_refreshed_utc"), str) or not pointer.get("last_refreshed_utc"):
        raise PointerLiteralAuditError("canonical pointer refresh timestamp is absent")
    if type(pointer.get("auto_update_on_dispatch_completion")) is not bool:
        raise PointerLiteralAuditError("canonical pointer auto-update flag has the wrong type")
    if not isinstance(pointer.get("pointer_refresh_command"), str) or not pointer.get("pointer_refresh_command"):
        raise PointerLiteralAuditError("canonical pointer refresh command is absent")
    if not isinstance(pointer.get("refresh_provenance"), Mapping):
        raise PointerLiteralAuditError("canonical pointer refresh provenance must be an object")
    pr_number = pointer.get("submitted_pr_number_for_current_frontier")
    if pr_number is not None and (type(pr_number) is not int or pr_number < 1):
        raise PointerLiteralAuditError("canonical pointer submitted PR number has the wrong type")

    cpu = _validate_local_anchor(
        pointer.get("our_local_frontier_contest_cpu"),
        field="our_local_frontier_contest_cpu",
        axis="contest_cpu",
    )
    cuda = _validate_local_anchor(
        pointer.get("our_local_frontier_contest_cuda"),
        field="our_local_frontier_contest_cuda",
        axis="contest_cuda",
    )
    public_result = _validated_public_best(
        pointer.get("upstream_leaderboard_snapshot"),
        label="canonical pointer upstream_leaderboard_snapshot",
    )
    top_snapshot_at = pointer.get("upstream_leaderboard_snapshot_at_utc")
    snapshot = pointer.get("upstream_leaderboard_snapshot")
    if snapshot is not None:
        if not isinstance(top_snapshot_at, str) or top_snapshot_at != snapshot.get("fetched_at_utc"):
            raise PointerLiteralAuditError("canonical pointer upstream snapshot timestamp is inconsistent")
    elif top_snapshot_at is not None:
        raise PointerLiteralAuditError("canonical pointer has a snapshot timestamp without a snapshot")

    candidates: list[dict[str, Any]] = []
    for source, anchor in (
        ("our_local_frontier_contest_cpu", cpu),
        ("our_local_frontier_contest_cuda", cuda),
    ):
        if anchor is not None:
            candidates.append(
                {
                    "score": anchor["score"],
                    "source": source,
                    "source_kind": "owned_or_banked_local_exact_anchor",
                    "axis": anchor["axis"],
                    "archive_sha256": anchor["archive_sha256"],
                    "lane_id": anchor.get("lane_id"),
                    "hardware_substrate": anchor["hardware_substrate"],
                    "measured_at_utc": anchor.get("measured_at_utc"),
                    "evidence_grade": anchor["evidence_grade"],
                    "custody": "local_anchor_record; inspect lane policy before submission",
                }
            )
    if public_result is not None:
        public, public_at = public_result
        candidates.append(
            {
                "score": public["score"],
                "source": "upstream_official_leaderboard",
                "source_kind": "external_public_leaderboard_target",
                "axis": "official_leaderboard",
                "leaderboard_rank": public["rank"],
                "submission_name": public["name"],
                "pr_number": public.get("pr_number"),
                "pr_url": public.get("pr_url"),
                "snapshot_at_utc": public_at,
                "evidence_grade": "[official-leaderboard display]",
                "score_precision": "official_display",
                "custody": "external target only; no local archive authority implied",
            }
        )
    if not candidates:
        raise PointerLiteralAuditError("canonical pointer has no qualifying competitive candidates")
    expected = min(
        candidates,
        key=lambda row: (float(row["score"]), 0 if str(row["source"]).startswith("our_local_") else 1),
    )
    expected |= {"selection_rule": _SELECTION_RULE, "role": "competitive_score_to_beat"}
    effective = pointer.get("effective_frontier")
    if not isinstance(effective, Mapping):
        raise PointerLiteralAuditError("canonical pointer effective frontier must be an object")
    if set(effective) != set(expected):
        raise PointerLiteralAuditError("canonical pointer effective frontier has fabricated or missing fields")
    for key, expected_value in expected.items():
        if effective.get(key) != expected_value:
            raise PointerLiteralAuditError(f"canonical pointer effective frontier has fabricated {key}")
    return {"pointer": dict(pointer), "effective_frontier": expected, "effective_score": expected["score"]}


def load_validated_canonical_pointer(path: Path) -> dict[str, Any]:
    """Load exact bytes and reject schema-valid-looking but semantically false pointers."""

    try:
        pointer_bytes = path.read_bytes()
        pointer = json.loads(pointer_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PointerLiteralAuditError(f"canonical pointer is unavailable: {path}") from exc
    validated = validate_canonical_pointer_payload(pointer)
    return {**validated, "pointer_bytes": pointer_bytes}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def _target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Attribute):
        return (target.attr,)
    if isinstance(target, ast.Subscript):
        key = _literal_string(target.slice)
        return (key,) if key is not None else ()
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for child in target.elts for name in _target_names(child))
    return ()


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _numeric_constant(node: ast.AST | None) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _numeric_constant(node.operand)
        return -value if value is not None else None
    return None


def _semantic_names(node: ast.AST | None) -> tuple[str, ...]:
    """Return identifier/attribute/subscript-key names carried by an expression."""

    if node is None:
        return ()
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_semantic_names(node.value), node.attr)
    if isinstance(node, ast.Subscript):
        key = _literal_string(node.slice)
        return (*_semantic_names(node.value), *((key,) if key is not None else ()))
    return ()


def _pointer_values(node: ast.AST | None, *, inherited_name: str | None = None) -> list[tuple[str, float]]:
    """Find numeric values in pointer-named mapping slots without treating all mappings as pointers."""

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        rows: list[tuple[str, float]] = []
        for keyword in node.keywords:
            value = _numeric_constant(keyword.value)
            if keyword.arg is not None and value is not None and _POINTER_NAME.search(keyword.arg):
                rows.append((keyword.arg, value))
        return rows
    if not isinstance(node, ast.Dict):
        value = _numeric_constant(node)
        if value is not None and inherited_name is not None and _POINTER_NAME.search(inherited_name):
            return [(inherited_name, value)]
        return []
    rows: list[tuple[str, float]] = []
    inherited_pointer = inherited_name is not None and _POINTER_NAME.search(inherited_name)
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        key = _literal_string(key_node)
        value = _numeric_constant(value_node)
        if value is not None and key is not None and (
            _POINTER_NAME.search(key) or (inherited_pointer and key.casefold() == "score")
        ):
            rows.append((key if _POINTER_NAME.search(key) else str(inherited_name), value))
        elif isinstance(value_node, ast.Dict):
            rows.extend(_pointer_values(value_node, inherited_name=key or inherited_name))
    return rows


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PointerLiteralAuditError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PointerLiteralAuditError(f"{label} must be finite and positive")
    return result


def _is_test_path(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    return "tests" in parts or Path(relative_path).name.startswith("test_")


def _is_repo_source(path: Path) -> bool:
    """Exclude vendored runtimes, caches, and build products from source audit."""

    return not any(part in _EXCLUDED_TREE_PARTS for part in path.parts)


def _tracked_and_untracked_repo_python() -> tuple[Path, ...]:
    """Return the Git-owned work surface, excluding ignored evidence/vendor trees."""

    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PointerLiteralAuditError("git ls-files failed while resolving the repository work surface")
    return tuple(
        REPO / line
        for line in completed.stdout.splitlines()
        if line and _is_repo_source(Path(line))
    )


def audit_python_file(path: Path) -> dict[str, Any]:
    """Return syntax-aware decision literals and line-aware stale text."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PointerLiteralAuditError(f"cannot audit {path}: {exc}") from exc
    relative = _relative(path)
    assignments: list[dict[str, Any]] = []

    def add(node: ast.AST, name: str, value: float, kind: str) -> None:
        category = (
            "dynamic_competitive_pointer_candidate"
            if _COMPETITIVE_POINTER_NAME.fullmatch(name)
            else "pointer_related_numeric_symbol"
        )
        assignments.append(
            {
                "line": int(node.lineno),
                "name": name,
                "value": value,
                "kind": kind,
                "classification": (
                    "test_fixture_pointer_literal" if _is_test_path(relative) else "executable_pointer_literal"
                ),
                "semantic_category": category,
            }
        )

    for node in ast.walk(tree):
        targets: Iterable[ast.expr]
        value_node: ast.AST | None
        if isinstance(node, ast.Assign):
            targets, value_node = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value_node = (node.target,), node.value
        else:
            targets, value_node = (), None
        if targets:
            value = _numeric_constant(value_node)
            for target in targets:
                target_names = _target_names(target)
                if value is not None:
                    for name in target_names:
                        if _POINTER_NAME.search(name):
                            add(node, name, value, "assignment")
                if isinstance(value_node, (ast.Dict, ast.Call)):
                    for inherited_name in target_names or (None,):
                        for name, mapping_value in _pointer_values(value_node, inherited_name=inherited_name):
                            add(node, name, mapping_value, "mapping_value")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            positional = [*node.args.posonlyargs, *node.args.args]
            positional_with_defaults = positional[len(positional) - len(node.args.defaults) :]
            for argument, default in zip(positional_with_defaults, node.args.defaults, strict=True):
                value = _numeric_constant(default)
                if value is not None and _POINTER_NAME.search(argument.arg):
                    add(default, argument.arg, value, "function_default")
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                value = _numeric_constant(default)
                if value is not None and _POINTER_NAME.search(argument.arg):
                    add(default, argument.arg, value, "keyword_default")
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for index, operand in enumerate(operands):
                value = _numeric_constant(operand)
                if value is None or not 0.0 < value < 1.0:
                    continue
                neighbors = operands[max(0, index - 1) : index] + operands[index + 1 : index + 2]
                names = tuple(name for neighbor in neighbors for name in _semantic_names(neighbor))
                semantic = next(
                    (name for name in names if _POINTER_NAME.search(name) or _SCORE_VALUE_NAME.search(name)),
                    None,
                )
                if semantic is not None:
                    add(node, semantic, value, "comparison_threshold")
    # Nested AST walks can surface the same literal through an assignment and
    # its mapping child; retain one stable semantic finding per location/form.
    assignments = [
        dict(row)
        for row in {
            (row["line"], row["name"], row["value"], row["kind"], row["classification"]): row
            for row in assignments
        }.values()
    ]
    assignments.sort(key=lambda row: (row["line"], row["name"], row["kind"]))
    retired_text = [
        {"line": line_number, "text": line.strip()[:240]}
        for line_number, line in enumerate(source.splitlines(), start=1)
        if _RETIRED_TEXT.search(line)
    ]
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "pointer_assignments": assignments,
        "retired_pointer_text": retired_text,
    }


def build_audit(
    *,
    roots: Iterable[Path],
    pointer_path: Path,
    strict_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    resolved_pointer = pointer_path if pointer_path.is_absolute() else REPO / pointer_path
    validated_pointer = load_validated_canonical_pointer(resolved_pointer)
    effective = validated_pointer["effective_frontier"]
    effective_score = validated_pointer["effective_score"]

    files: list[Path] = []
    repo_python = _tracked_and_untracked_repo_python()
    for root in roots:
        resolved = root if root.is_absolute() else REPO / root
        if resolved.is_file() and resolved.suffix == ".py":
            files.append(resolved)
        elif resolved.is_dir():
            try:
                resolved.relative_to(REPO)
            except ValueError:
                files.extend(path for path in resolved.rglob("*.py") if path.is_file() and _is_repo_source(path))
            else:
                files.extend(path for path in repo_python if path.is_relative_to(resolved))
        else:
            raise PointerLiteralAuditError(f"audit root is absent: {resolved}")
    rows = [audit_python_file(path) for path in sorted(set(files))]
    rows = [row for row in rows if row["pointer_assignments"] or row["retired_pointer_text"]]
    executable = [
        {"path": row["path"], **assignment}
        for row in rows
        for assignment in row["pointer_assignments"]
        if assignment["classification"] == "executable_pointer_literal"
    ]
    strict = {_relative(path if path.is_absolute() else REPO / path) for path in strict_paths}
    strict_violations = []
    for row in rows:
        if row["path"] not in strict:
            continue
        competitive = [
            assignment
            for assignment in row["pointer_assignments"]
            if assignment["semantic_category"] == "dynamic_competitive_pointer_candidate"
        ]
        if competitive or row["retired_pointer_text"]:
            strict_violations.append(
                {
                    **row,
                    "pointer_assignments": competitive,
                }
            )
    competitive_executable = [
        row
        for row in executable
        if row["semantic_category"] == "dynamic_competitive_pointer_candidate"
    ]
    result = {
        "schema": "frontier_pointer_literal_audit.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "canonical_pointer": {
            "path": _relative(resolved_pointer),
            "sha256": _sha256(resolved_pointer),
            "effective_score": effective_score,
            "effective_axis": effective.get("axis"),
            "effective_source": effective.get("source"),
            "selection_rule": effective.get("selection_rule"),
        },
        "scope": [_relative(root if root.is_absolute() else REPO / root) for root in roots],
        "classification_rule": (
            "AST numeric pointer-related symbols are broad inventory; only exact dynamic competitive-pointer names "
            "and 0.191-family text trigger the scoped strict gate. Component, mission, topology, and other broad "
            "frontier-named symbols require triage and are not counted as competitive-pointer debt"
        ),
        "python_files_scanned": len(files),
        "files_with_findings": len(rows),
        "executable_pointer_literal_count": len(executable),
        "competitive_pointer_literal_count": len(competitive_executable),
        "retired_pointer_text_count": sum(len(row["retired_pointer_text"]) for row in rows),
        "strict_paths": sorted(strict),
        "strict_violation_count": len(strict_violations),
        "strict_violations": strict_violations,
        "executable_pointer_literals": executable,
        "competitive_pointer_literals": competitive_executable,
        "findings": rows,
        "verdict_scope": "inventory and scoped preflight only; historical score evidence is not rewritten",
        "pointer_moved": False,
        "score_claim": False,
    }
    result["receipt_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return result


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise PointerLiteralAuditError(f"write-once output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != encoded:
                raise PointerLiteralAuditError(f"concurrent write-once output differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", dest="roots")
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--strict-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_audit(
        roots=args.roots or DEFAULT_ROOTS,
        pointer_path=args.pointer,
        strict_paths=args.strict_path,
    )
    if args.output is not None:
        output = args.output if args.output.is_absolute() else REPO / args.output
        _write_once_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 1 if payload["strict_violation_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
