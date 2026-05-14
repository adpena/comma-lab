# SPDX-License-Identifier: MIT
"""Deterministic readiness audit for hidden-gem registry entries.

The hidden-gem registry is a planning surface, not score evidence. This module
adds a cheap local audit that checks whether each registry row points at live
repo evidence and integration targets without reading provider state or
launching GPU work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.hidden_gems import all_hidden_gems
from tac.repo_io import sha256_file

if TYPE_CHECKING:
    from tac.hidden_gems import HiddenGemEntry

SCHEMA_VERSION = 1
AUDIT_NAME = "hidden_gem_readiness"
DISPATCH_BLOCKER_REGISTRY_ONLY = "hidden_gem_registry_not_exact_eval_dispatch_evidence"


@dataclass(frozen=True)
class PathReadiness:
    """Stable probe result for one repo-relative path."""

    path: str
    exists: bool
    kind: str
    bytes: int | None
    sha256: str | None


@dataclass(frozen=True)
class HiddenGemReadiness:
    """Readiness row for one hidden-gem registry entry."""

    key: str
    category: str
    registry_status: str
    readiness_status: str
    eligible_for_local_patch: bool
    ready_for_exact_eval_dispatch: bool
    dispatch_blockers: tuple[str, ...]
    missing_evidence_paths: tuple[str, ...]
    missing_integration_targets: tuple[str, ...]
    evidence: tuple[PathReadiness, ...]
    integration_targets: tuple[PathReadiness, ...]
    next_patch: str


def audit_hidden_gems(
    *,
    repo_root: Path | str,
    entries: Iterable[HiddenGemEntry] | None = None,
) -> tuple[HiddenGemReadiness, ...]:
    """Audit hidden-gem rows against the current checkout."""
    root = Path(repo_root)
    rows = tuple(all_hidden_gems() if entries is None else entries)
    return tuple(_audit_entry(root, entry) for entry in rows)


def readiness_payload(
    *,
    repo_root: Path | str,
    entries: Iterable[HiddenGemEntry] | None = None,
) -> dict[str, Any]:
    """Return a JSON-stable readiness report."""
    rows = audit_hidden_gems(repo_root=repo_root, entries=entries)
    status_counts = Counter(row.readiness_status for row in rows)
    payload_rows = [hidden_gem_readiness_to_dict(row) for row in rows]
    return {
        "audit": AUDIT_NAME,
        "entries": payload_rows,
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "eligible_for_local_patch_count": sum(row.eligible_for_local_patch for row in rows),
            "entry_count": len(rows),
            "missing_evidence_path_count": sum(len(row.missing_evidence_paths) for row in rows),
            "missing_integration_target_count": sum(len(row.missing_integration_targets) for row in rows),
            "readiness_status_counts": dict(sorted(status_counts.items())),
            "ready_for_exact_eval_dispatch_count": sum(row.ready_for_exact_eval_dispatch for row in rows),
        },
    }


def hidden_gem_readiness_to_dict(row: HiddenGemReadiness) -> dict[str, Any]:
    """Convert one readiness row to a deterministic JSON mapping."""
    return {
        "category": row.category,
        "dispatch_blockers": list(row.dispatch_blockers),
        "eligible_for_local_patch": row.eligible_for_local_patch,
        "evidence": [path_readiness_to_dict(path) for path in row.evidence],
        "integration_targets": [path_readiness_to_dict(path) for path in row.integration_targets],
        "key": row.key,
        "missing_evidence_paths": list(row.missing_evidence_paths),
        "missing_integration_targets": list(row.missing_integration_targets),
        "next_patch": row.next_patch,
        "readiness_status": row.readiness_status,
        "ready_for_exact_eval_dispatch": row.ready_for_exact_eval_dispatch,
        "registry_status": row.registry_status,
    }


def path_readiness_to_dict(row: PathReadiness) -> dict[str, Any]:
    """Convert one path probe to a deterministic JSON mapping."""
    return {
        "bytes": row.bytes,
        "exists": row.exists,
        "kind": row.kind,
        "path": row.path,
        "sha256": row.sha256,
    }


def render_markdown(rows: Iterable[HiddenGemReadiness]) -> str:
    """Render readiness rows as deterministic markdown."""
    audited = tuple(rows)
    lines = [
        "# Hidden-Gem Readiness Audit",
        "",
        "This audit checks live repo evidence and integration targets. It never unlocks exact-eval dispatch.",
        "",
        "| key | registry status | readiness | missing evidence | missing targets | next patch |",
        "|---|---|---|---|---|---|",
    ]
    if not audited:
        lines.append("| _none_ | _none_ | _none_ | _none_ | _none_ | _none_ |")
    for row in audited:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_cell(row.key)}`",
                    f"`{_markdown_cell(row.registry_status)}`",
                    f"`{_markdown_cell(row.readiness_status)}`",
                    _markdown_path_list(row.missing_evidence_paths),
                    _markdown_path_list(row.missing_integration_targets),
                    _markdown_cell(row.next_patch),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _audit_entry(root: Path, entry: HiddenGemEntry) -> HiddenGemReadiness:
    evidence = tuple(_probe_path(root, path) for path in entry.evidence_paths)
    integration_targets = tuple(_probe_path(root, path) for path in entry.integration_targets)
    missing_evidence = tuple(row.path for row in evidence if not row.exists)
    missing_targets = tuple(row.path for row in integration_targets if not row.exists)
    readiness_status = _readiness_status(entry, missing_evidence, missing_targets)
    eligible_for_local_patch = readiness_status == "ready_for_local_patch"
    dispatch_blockers = _dispatch_blockers(entry, missing_evidence, missing_targets)
    return HiddenGemReadiness(
        key=entry.key,
        category=entry.category,
        registry_status=entry.status,
        readiness_status=readiness_status,
        eligible_for_local_patch=eligible_for_local_patch,
        ready_for_exact_eval_dispatch=False,
        dispatch_blockers=dispatch_blockers,
        missing_evidence_paths=missing_evidence,
        missing_integration_targets=missing_targets,
        evidence=evidence,
        integration_targets=integration_targets,
        next_patch=entry.next_patch,
    )


def _probe_path(root: Path, relpath: str) -> PathReadiness:
    path = root / relpath
    if path.is_file():
        return PathReadiness(
            path=relpath,
            exists=True,
            kind="file",
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
    if path.is_dir():
        return PathReadiness(path=relpath, exists=True, kind="dir", bytes=None, sha256=None)
    if path.exists():
        return PathReadiness(path=relpath, exists=True, kind="other", bytes=None, sha256=None)
    return PathReadiness(path=relpath, exists=False, kind="missing", bytes=None, sha256=None)


def _readiness_status(
    entry: HiddenGemEntry,
    missing_evidence: tuple[str, ...],
    missing_targets: tuple[str, ...],
) -> str:
    if missing_targets:
        return "blocked_missing_integration_targets"
    if missing_evidence:
        return "blocked_missing_evidence"
    if entry.status == "ready_for_patch":
        return "ready_for_local_patch"
    return f"{entry.status}_tracked"


def _dispatch_blockers(
    entry: HiddenGemEntry,
    missing_evidence: tuple[str, ...],
    missing_targets: tuple[str, ...],
) -> tuple[str, ...]:
    blockers = [DISPATCH_BLOCKER_REGISTRY_ONLY]
    if entry.status != "ready_for_patch":
        blockers.append(f"registry_status_{entry.status}")
    if missing_evidence:
        blockers.append("missing_evidence_paths")
    if missing_targets:
        blockers.append("missing_integration_targets")
    return tuple(blockers)


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _markdown_path_list(values: Iterable[str]) -> str:
    rows = tuple(values)
    if not rows:
        return "_none_"
    return "<br>".join(f"`{_markdown_cell(value)}`" for value in rows)


__all__ = [
    "AUDIT_NAME",
    "DISPATCH_BLOCKER_REGISTRY_ONLY",
    "SCHEMA_VERSION",
    "HiddenGemReadiness",
    "PathReadiness",
    "audit_hidden_gems",
    "hidden_gem_readiness_to_dict",
    "path_readiness_to_dict",
    "readiness_payload",
    "render_markdown",
]
