# SPDX-License-Identifier: MIT
"""Implementation-readiness gate for compact carrier candidates.

The HPRC spine normalizes archive custody and byte accounting, but a row can
still come from a smoke, mock, proxy, scaffold, or otherwise non-real
implementation.  This module turns those signals into typed runner blockers so
budget routing does not treat fake/minimal implementations as score-lowering
work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT

HPRC_IMPLEMENTATION_READINESS_SCHEMA = "hprc_implementation_readiness.v1"
HPRC_IMPLEMENTATION_READINESS_EVIDENCE_SCHEMA = (
    "hprc_implementation_readiness_evidence.v1"
)

_BLOCKING_TRUE_KEYS = frozenset(
    {
        "allow_mock_scorer_teacher",
        "archive_bytes_proxy_only",
        "decoder_stubbed",
        "fake_implementation",
        "full_main_not_implemented",
        "interpolate_stubbed_to_tiny_tensor",
        "is_smoke",
        "mock_scorer_teacher",
        "mock_teacher",
        "parse_archive_stubbed",
        "proxy_only",
        "scaffold_only",
        "smoke",
        "smoke_only",
        "synthetic_non_smoke",
        "synthetic_targets",
        "uses_mock_scorer",
        "uses_synthetic_targets",
    }
)

_BLOCKING_STRING_PATTERNS = (
    "archive_bytes_proxy",
    "fake implementation",
    "full_main_not_implemented",
    "not implemented",
    "notimplementederror",
    "placeholder",
    "proxy-only",
    "proxy_only",
    "scaffold-only",
    "scaffold_only",
    "smoke-only",
    "smoke_only",
    "stubbed",
    "synthetic data",
    "synthetic non-smoke",
    "synthetic_non_smoke",
)

_MOCK_STRING_PATTERNS = (
    "allow-mock-scorer-teacher",
    "mock scorer",
    "mock-scorer",
    "mock_scorer",
    "mock teacher",
    "mock-teacher",
    "mock_teacher",
)


def build_implementation_readiness(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    require_full_video_coverage: bool = True,
    require_archive_custody: bool = False,
) -> dict[str, Any]:
    """Classify whether a candidate row is real enough for runner budget.

    The gate is evidence-driven: it blocks only on explicit machine-readable or
    text-level signals that a row is smoke/mock/proxy/scaffold/stubbed, on
    partial-video coverage when full coverage is required, or on missing archive
    custody when the caller asks for a materialization-ready row.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    evidence: list[dict[str, Any]] = []
    _collect_blocking_evidence(payload, path="$", evidence=evidence)
    if require_full_video_coverage:
        _collect_coverage_evidence(payload, evidence=evidence)
    if require_archive_custody:
        _collect_archive_custody_evidence(payload, evidence=evidence)

    evidence = _dedupe_evidence(evidence)
    blockers = _dedupe([str(item["blocker"]) for item in evidence])
    status = (
        "blocked_by_fake_or_incomplete_implementation"
        if blockers
        else "implementation_ready_for_budget_routing"
    )
    source = (
        None
        if source_path is None
        else Path(source_path).expanduser().resolve(strict=False)
    )
    payload_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "schema": HPRC_IMPLEMENTATION_READINESS_SCHEMA,
        "source_path": None if source is None else source.as_posix(),
        "projection_manifest_path": None if source is None else source.as_posix(),
        "payload_fingerprint_sha256": payload_fingerprint,
        "require_full_video_coverage": bool(require_full_video_coverage),
        "require_archive_custody": bool(require_archive_custody),
        "status": status,
        "ready_for_budget_routing": not blockers,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "blocking_evidence_count": len(evidence),
        "blocking_evidence": evidence,
        "blockers": blockers,
    }


def implementation_readiness_blockers(readiness: Mapping[str, Any]) -> list[str]:
    """Return blocker codes from a readiness payload."""

    values = readiness.get("blockers")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    return [str(value) for value in values if str(value)]


def _collect_blocking_evidence(
    value: Any,
    *,
    path: str,
    evidence: list[dict[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            normalized_key = _normalize_token(key)
            if normalized_key in _BLOCKING_TRUE_KEYS and child is True:
                evidence.append(
                    _evidence(
                        path=child_path,
                        kind="blocking_true_flag",
                        blocker=f"{normalized_key}_blocks_runner_budget",
                        value=True,
                    )
                )
            _collect_blocking_evidence(child, path=child_path, evidence=evidence)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, child in enumerate(value):
            _collect_blocking_evidence(child, path=f"{path}[{idx}]", evidence=evidence)
        return
    if isinstance(value, str):
        lowered = _normalize_text(value)
        for pattern in _BLOCKING_STRING_PATTERNS:
            if pattern in lowered:
                evidence.append(
                    _evidence(
                        path=path,
                        kind="blocking_text_signal",
                        blocker=f"{_normalize_token(pattern)}_blocks_runner_budget",
                        value=value,
                    )
                )
        for pattern in _MOCK_STRING_PATTERNS:
            if pattern in lowered:
                evidence.append(
                    _evidence(
                        path=path,
                        kind="mock_or_teacher_text_signal",
                        blocker="mock_scorer_or_teacher_blocks_runner_budget",
                        value=value,
                    )
                )


def _collect_coverage_evidence(
    payload: Mapping[str, Any],
    *,
    evidence: list[dict[str, Any]],
) -> None:
    declared = _find_first_positive_int(
        payload,
        key_names=("declared_pairs", "num_pairs", "max_pairs"),
    )
    if declared is None:
        return
    if declared < CONTEST_PAIR_COUNT:
        evidence.append(
            _evidence(
                path="$",
                kind="partial_coverage",
                blocker="declared_pair_coverage_below_full_video",
                value={
                    "declared_pairs": declared,
                    "required_pairs": CONTEST_PAIR_COUNT,
                },
            )
        )


def _collect_archive_custody_evidence(
    payload: Mapping[str, Any],
    *,
    evidence: list[dict[str, Any]],
) -> None:
    path = _find_first_string(
        payload,
        key_names=("archive_zip_path", "archive_path", "path"),
    )
    sha = _find_first_string(payload, key_names=("archive_zip_sha256", "sha256"))
    byte_count = _find_first_positive_int(
        payload,
        key_names=("archive_zip_bytes", "archive_bytes", "bytes"),
    )
    if not path:
        evidence.append(
            _evidence(
                path="$",
                kind="missing_archive_custody",
                blocker="archive_zip_path_missing_for_budget_routing",
                value=None,
            )
        )
    if not sha:
        evidence.append(
            _evidence(
                path="$",
                kind="missing_archive_custody",
                blocker="archive_sha256_missing_for_budget_routing",
                value=None,
            )
        )
    if byte_count is None:
        evidence.append(
            _evidence(
                path="$",
                kind="missing_archive_custody",
                blocker="archive_bytes_missing_for_budget_routing",
                value=None,
            )
        )


def _find_first_positive_int(
    value: Any,
    *,
    key_names: tuple[str, ...],
) -> int | None:
    wanted = {_normalize_token(key) for key in key_names}
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if _normalize_token(str(raw_key)) in wanted:
                parsed = _positive_int(child)
                if parsed is not None:
                    return parsed
            found = _find_first_positive_int(child, key_names=key_names)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_first_positive_int(child, key_names=key_names)
            if found is not None:
                return found
    return None


def _find_first_string(
    value: Any,
    *,
    key_names: tuple[str, ...],
) -> str | None:
    wanted = {_normalize_token(key) for key in key_names}
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if _normalize_token(str(raw_key)) in wanted and isinstance(child, str) and child:
                return child
            found = _find_first_string(child, key_names=key_names)
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_first_string(child, key_names=key_names)
            if found:
                return found
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _evidence(*, path: str, kind: str, blocker: str, value: Any) -> dict[str, Any]:
    return {
        "schema": HPRC_IMPLEMENTATION_READINESS_EVIDENCE_SCHEMA,
        "path": path,
        "kind": kind,
        "blocker": blocker,
        "value": value,
    }


def _dedupe_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_token(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _normalize_text(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


__all__ = [
    "HPRC_IMPLEMENTATION_READINESS_SCHEMA",
    "build_implementation_readiness",
    "implementation_readiness_blockers",
]
