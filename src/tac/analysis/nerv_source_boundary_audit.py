# SPDX-License-Identifier: MIT
"""Source-boundary audit for evaluator-witness NeRV artifacts.

Offline search may use the original video, scorer models, giant teachers, and
oracle caches.  The eval-time runtime must not smuggle learned payload through
source constants or external sidecars; score-affecting learned state belongs in
``archive.zip``.  This module implements the Phase-0 audit that separates those
two worlds before HiNeRV/SNeRV long runs can be trusted.
"""

from __future__ import annotations

import base64
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
)
from tac.repo_io import sha256_bytes, sha256_file

NERV_SOURCE_BOUNDARY_AUDIT_SCHEMA = "nerv_source_boundary_audit.v1"
DEFAULT_LARGE_SOURCE_BYTES = 64_000
DEFAULT_LARGE_LITERAL_BYTES = 16_384

BoundaryMode = Literal["conservative", "aggressive"]

_BASE64_RUN_RE = re.compile(
    rb"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{128,}={0,2})(?![A-Za-z0-9+/=])"
)
_HEX_RUN_RE = re.compile(rb"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{512,})(?![0-9a-fA-F])")
_NUMPY_ARRAY_RE = re.compile(
    rb"\b(?:np|numpy)\.(?:array|asarray|frombuffer)\s*\(", re.MULTILINE
)
_TORCH_TENSOR_RE = re.compile(
    rb"\btorch\.(?:tensor|as_tensor|frombuffer)\s*\(", re.MULTILINE
)
_MLX_ARRAY_RE = re.compile(rb"\bmx\.array\s*\(", re.MULTILINE)
_EXTERNAL_ARTIFACT_RE = re.compile(
    r"(?P<path>(?:/Volumes|/Users|/tmp|/var/folders|experiments/results|\.omx|runs|outputs)"
    r"[^'\"\s)]*?\.(?:npy|npz|pt|pth|safetensors|pkl|pickle|bin|zip|zst|br|xz|mkv|raw|png))"
)
_LEARNED_FILE_SUFFIXES = {
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".bin",
}


class NervSourceBoundaryAuditError(ValueError):
    """Raised when source-boundary audit inputs are invalid."""


def audit_nerv_source_boundary(
    *,
    source_paths: Sequence[str | Path],
    archive_zip: str | Path | None = None,
    mode: BoundaryMode = "conservative",
    large_source_bytes: int = DEFAULT_LARGE_SOURCE_BYTES,
    large_literal_bytes: int = DEFAULT_LARGE_LITERAL_BYTES,
) -> dict[str, Any]:
    """Audit eval-time source for uncharged learned payload leakage.

    Conservative mode blocks both large source files and learned-looking source
    literals.  Aggressive mode permits compact generic algorithms but still
    blocks large constants and external learned sidecar references.
    """

    if mode not in {"conservative", "aggressive"}:
        raise NervSourceBoundaryAuditError(f"unknown source-boundary mode: {mode}")
    if not source_paths:
        raise NervSourceBoundaryAuditError("source_paths must not be empty")
    if large_source_bytes <= 0 or large_literal_bytes <= 0:
        raise NervSourceBoundaryAuditError("byte thresholds must be positive")

    source_reports = [
        _audit_source_path(
            Path(path).expanduser().resolve(strict=False),
            large_source_bytes=large_source_bytes,
            large_literal_bytes=large_literal_bytes,
        )
        for path in source_paths
    ]
    archive_report = _archive_report(archive_zip)
    issues = _collect_issues(source_reports, mode=mode)
    blockers = _blockers_from_issues(issues, archive_report=archive_report)
    clean = not blockers
    payload = {
        "schema": NERV_SOURCE_BOUNDARY_AUDIT_SCHEMA,
        "mode": mode,
        "source_paths": [str(Path(path).expanduser().resolve(strict=False)) for path in source_paths],
        "archive_zip": archive_report,
        "thresholds": {
            "large_source_bytes": int(large_source_bytes),
            "large_literal_bytes": int(large_literal_bytes),
        },
        "source_reports": source_reports,
        "issues": issues,
        "blockers": blockers,
        "source_boundary_clean": clean,
        "ready_for_witness_compile": clean,
        "long_training_gate_satisfied": clean,
        "authority": {
            "source_boundary": "static_eval_time_payload_boundary_audit",
            "score_claim": False,
            "exact_eval_authority": False,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    return apply_proxy_evidence_boundary(
        payload,
        dispatch_blockers=(
            []
            if clean
            else ["nerv_source_boundary_audit_blocked_by_uncharged_payload_risk"]
        ),
    )


def _audit_source_path(
    path: Path,
    *,
    large_source_bytes: int,
    large_literal_bytes: int,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": path.as_posix(),
            "exists": False,
            "kind": "missing",
            "bytes": 0,
            "sha256": None,
            "issues": [{"kind": "missing_source_path", "severity": "blocker"}],
        }
    if path.is_dir():
        children = [
            _audit_source_file(
                child,
                large_source_bytes=large_source_bytes,
                large_literal_bytes=large_literal_bytes,
            )
            for child in sorted(path.rglob("*"))
            if child.is_file() and not _is_ignored_source_child(child)
        ]
        issues = [issue for child in children for issue in child.get("issues", [])]
        total_bytes = sum(int(child.get("bytes") or 0) for child in children)
        tree_hash = sha256_bytes(
            "\n".join(
                f"{child['path']}:{child.get('sha256') or ''}:{child.get('bytes') or 0}"
                for child in children
            ).encode("utf-8")
        )
        return {
            "path": path.as_posix(),
            "exists": True,
            "kind": "directory",
            "bytes": total_bytes,
            "sha256": tree_hash,
            "file_count": len(children),
            "files": children,
            "issues": issues,
        }
    return _audit_source_file(
        path,
        large_source_bytes=large_source_bytes,
        large_literal_bytes=large_literal_bytes,
    )


def _audit_source_file(
    path: Path,
    *,
    large_source_bytes: int,
    large_literal_bytes: int,
) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {
            "path": path.as_posix(),
            "exists": False,
            "kind": "unreadable",
            "bytes": 0,
            "sha256": None,
            "issues": [
                {
                    "kind": "unreadable_source_file",
                    "severity": "blocker",
                    "detail": str(exc),
                }
            ],
        }
    issues: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if len(data) > large_source_bytes and suffix in {".py", ".sh", ".txt", ".json"}:
        issues.append(
            {
                "kind": "large_eval_time_source_file",
                "severity": "review",
                "path": path.as_posix(),
                "bytes": len(data),
                "threshold": large_source_bytes,
            }
        )
    if suffix in _LEARNED_FILE_SUFFIXES:
        issues.append(
            {
                "kind": "learned_payload_file_in_source_paths",
                "severity": "blocker",
                "path": path.as_posix(),
                "bytes": len(data),
            }
        )
    issues.extend(_literal_issues(path, data, large_literal_bytes=large_literal_bytes))
    issues.extend(_external_reference_issues(path, data))
    return {
        "path": path.as_posix(),
        "exists": True,
        "kind": "file",
        "bytes": len(data),
        "sha256": sha256_file(path),
        "issues": issues,
    }


def _literal_issues(
    path: Path,
    data: bytes,
    *,
    large_literal_bytes: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for regex, kind in (
        (_BASE64_RUN_RE, "large_base64_literal_in_eval_source"),
        (_HEX_RUN_RE, "large_hex_literal_in_eval_source"),
    ):
        for match in regex.finditer(data):
            literal = match.group(0)
            decoded_bytes = _decoded_literal_size(literal, kind=kind)
            if decoded_bytes >= large_literal_bytes:
                issues.append(
                    {
                        "kind": kind,
                        "severity": "blocker",
                        "path": path.as_posix(),
                        "literal_bytes": len(literal),
                        "decoded_bytes_estimate": decoded_bytes,
                        "threshold": large_literal_bytes,
                    }
                )
    for regex, kind in (
        (_NUMPY_ARRAY_RE, "inline_numpy_array_constructor_in_eval_source"),
        (_TORCH_TENSOR_RE, "inline_torch_tensor_constructor_in_eval_source"),
        (_MLX_ARRAY_RE, "inline_mlx_array_constructor_in_eval_source"),
    ):
        if regex.search(data):
            issues.append(
                {
                    "kind": kind,
                    "severity": "review",
                    "path": path.as_posix(),
                }
            )
    return issues


def _decoded_literal_size(literal: bytes, *, kind: str) -> int:
    if kind == "large_hex_literal_in_eval_source":
        return len(literal) // 2
    try:
        return len(base64.b64decode(literal, validate=True))
    except Exception:
        return 0


def _external_reference_issues(path: Path, data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="ignore")
    issues: list[dict[str, Any]] = []
    for match in _EXTERNAL_ARTIFACT_RE.finditer(text):
        ref = match.group("path")
        issues.append(
            {
                "kind": "external_artifact_reference_in_eval_source",
                "severity": "blocker",
                "path": path.as_posix(),
                "reference": ref,
            }
        )
    return issues


def _archive_report(archive_zip: str | Path | None) -> dict[str, Any]:
    if archive_zip is None:
        return {
            "path": None,
            "present": False,
            "bytes": 0,
            "sha256": None,
            "members": [],
        }
    path = Path(archive_zip).expanduser().resolve(strict=False)
    if not path.is_file():
        return {
            "path": path.as_posix(),
            "present": False,
            "bytes": 0,
            "sha256": None,
            "members": [],
            "issues": [{"kind": "archive_zip_missing", "severity": "blocker"}],
        }
    members: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                members.append(
                    {
                        "name": info.filename,
                        "compress_type": int(info.compress_type),
                        "compressed_bytes": int(info.compress_size),
                        "uncompressed_bytes": int(info.file_size),
                        "crc": int(info.CRC),
                    }
                )
    except zipfile.BadZipFile:
        return {
            "path": path.as_posix(),
            "present": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "members": [],
            "issues": [{"kind": "archive_zip_invalid", "severity": "blocker"}],
        }
    return {
        "path": path.as_posix(),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "members": members,
        "issues": [],
    }


def _collect_issues(
    source_reports: Sequence[Mapping[str, Any]],
    *,
    mode: BoundaryMode,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for report in source_reports:
        issues.extend(_iter_issues(report))
    if mode == "conservative":
        for issue in issues:
            if issue.get("kind") == "large_eval_time_source_file":
                issue["severity"] = "blocker"
    return issues


def _blockers_from_issues(
    issues: Sequence[Mapping[str, Any]],
    *,
    archive_report: Mapping[str, Any],
) -> list[str]:
    blockers = [
        f"{issue.get('kind')}:{issue.get('path') or issue.get('reference') or 'unknown'}"
        for issue in issues
        if str(issue.get("severity")) == "blocker"
    ]
    blockers.extend(
        f"{issue.get('kind')}:{archive_report.get('path') or 'archive_zip'}"
        for issue in archive_report.get("issues") or []
        if isinstance(issue, Mapping) and str(issue.get("severity")) == "blocker"
    )
    return _ordered_unique(blockers)


def _iter_issues(report: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for issue in report.get("issues") or []:
        if isinstance(issue, Mapping):
            yield dict(issue)
    for child in report.get("files") or []:
        if isinstance(child, Mapping):
            yield from _iter_issues(child)


def _is_ignored_source_child(path: Path) -> bool:
    parts = set(path.parts)
    return bool(
        parts.intersection(
            {
                "__pycache__",
                ".git",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                ".venv",
                "node_modules",
            }
        )
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "DEFAULT_LARGE_LITERAL_BYTES",
    "DEFAULT_LARGE_SOURCE_BYTES",
    "NERV_SOURCE_BOUNDARY_AUDIT_SCHEMA",
    "NervSourceBoundaryAuditError",
    "audit_nerv_source_boundary",
]
