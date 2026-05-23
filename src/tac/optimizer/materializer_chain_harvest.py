# SPDX-License-Identifier: MIT
"""Harvest completed materializer chains into optimizer candidate rows.

Materializer chain manifests are custody evidence, not dispatch authority. This
adapter validates the live archive/artifact surface and emits the planning-row
shape consumed by ``tools/build_optimizer_candidate_queue.py``.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimizer.exact_readiness import validate_serialized_archive_delta_contract

SUPPORTED_CHAIN_SCHEMAS = frozenset(
    {
        "byte_range_entropy_recode_chain_v1",
        "inverse_scorer_cell_candidate_chain_v1",
    }
)
TOOL_NAME = "tools/build_optimizer_candidate_queue.py"
LOCAL_ADVISORY_AXIS_TOKENS = (
    "macos-cpu-advisory",
    "macos-cpu",
    "cpu-advisory",
    "macos-mlx",
    "mlx-research-signal",
    "locality-control",
)


class MaterializerChainHarvestError(ValueError):
    """Raised when a materializer chain cannot be harvested safely."""


def adapt_materializer_chain_manifest_to_candidate(
    chain: Mapping[str, Any],
    *,
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Return one non-authoritative optimizer candidate row for ``chain``."""

    schema = str(chain.get("schema") or "")
    if schema not in SUPPORTED_CHAIN_SCHEMAS:
        raise MaterializerChainHarvestError(f"unsupported_chain_schema:{schema!r}")
    _require_false_authority(chain, label="chain")
    _require_chain_complete(chain)
    _require_serialized_archive_delta(chain)

    candidate_archive = _archive_record(chain, "candidate_archive", repo_root=repo_root)
    source_archive = _archive_record(chain, "source_archive", repo_root=repo_root)
    delta_blockers, delta_facts = validate_serialized_archive_delta_contract(
        chain,
        actual_candidate_archive_bytes=candidate_archive["bytes"],
    )
    delta_blockers.extend(
        _serialized_delta_archive_custody_blockers(
            delta_facts,
            source_archive=source_archive,
            candidate_archive=candidate_archive,
        )
    )
    if delta_blockers:
        raise MaterializerChainHarvestError(
            "serialized_archive_delta_blocked:" + ",".join(delta_blockers)
        )
    _validate_chain_artifacts(chain, repo_root=repo_root)

    source_sha = _string_or_none(source_archive.get("sha256"))
    source_bytes = _positive_int(source_archive.get("bytes"))
    archive_changed = (
        source_sha is not None and candidate_archive["sha256"] != source_sha
    )
    byte_changed = (
        source_bytes is not None and candidate_archive["bytes"] != source_bytes
    )
    candidate_id = _candidate_id(
        chain, schema=schema, archive_sha=candidate_archive["sha256"]
    )
    row = {
        "candidate_id": candidate_id,
        "lane_id": str(chain.get("lane_id") or f"materializer_harvest::{schema}"),
        "lane_class": "materializer_chain_harvest",
        "candidate_family": _candidate_family(schema),
        "optimizer_tool": TOOL_NAME,
        "schema": schema,
        "source_manifest_path": _repo_rel(source_path, repo_root),
        "source_paths": [_repo_rel(source_path, repo_root)],
        "candidate_archive_path": candidate_archive["path"],
        "archive_path": candidate_archive["path"],
        "candidate_archive_sha256": candidate_archive["sha256"],
        "archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "archive_bytes": candidate_archive["bytes"],
        "source_archive_sha256": source_sha,
        "source_archive_bytes": source_bytes,
        "source_archive_path": source_archive.get("path"),
        "serialized_archive_delta": dict(chain.get("serialized_archive_delta") or {}),
        "serialized_archive_delta_validated": delta_facts,
        "score_affecting_payload_changed": archive_changed,
        "charged_bits_changed": byte_changed,
        "score_affecting_change_proof": _score_affecting_change_proof(
            source_sha=source_sha,
            source_bytes=source_bytes,
            candidate_archive=candidate_archive,
            archive_changed=archive_changed,
            byte_changed=byte_changed,
        ),
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "readiness_blockers": _string_list(chain.get("readiness_blockers")),
        "next_required_gates": _string_list(chain.get("next_required_gates")),
        "chain_artifact_count": len(chain.get("artifacts") or {}),
        "chain_step_count": len(chain.get("chain_steps") or []),
        "local_advisory_axes": _local_advisory_axes(chain),
        "local_advisory_axes_semantics": (
            "non_authoritative_planning_signal_only_not_score_claim"
        ),
        "evidence_semantics": (
            "materializer_chain_harvest_candidate_pending_exact_readiness"
        ),
        "evidence_grade": "[materializer-chain-harvest-no-score]",
        "harvested_at_utc": _utc_now(),
    }
    out = apply_proxy_evidence_boundary(
        row,
        dispatch_blockers=[
            "materializer_chain_is_not_dispatch_authorization",
            "materialized_archive_runtime_custody_required",
            "exact_readiness_promotion_required",
            "exact_auth_eval_result_required_before_score_claim",
            *_string_list(chain.get("readiness_blockers")),
            *_string_list(chain.get("dispatch_blockers")),
        ],
    )
    out["score_affecting_payload_changed"] = archive_changed
    out["charged_bits_changed"] = byte_changed
    return out


def _candidate_id(chain: Mapping[str, Any], *, schema: str, archive_sha: str) -> str:
    value = chain.get("candidate_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"{_candidate_family(schema)}_{archive_sha[:12]}"


def _candidate_family(schema: str) -> str:
    if schema == "byte_range_entropy_recode_chain_v1":
        return "byte_range_entropy_recode"
    if schema == "inverse_scorer_cell_candidate_chain_v1":
        return "inverse_scorer_cell"
    return "materializer_chain"


def _require_chain_complete(chain: Mapping[str, Any]) -> None:
    required_true = (
        "byte_closed_candidate_emitted",
        "runtime_adapter_ready",
        "receiver_contract_satisfied",
        "candidate_runtime_adapter_blocker_cleared",
    )
    for key in required_true:
        if chain.get(key) is not True:
            raise MaterializerChainHarvestError(f"{key}_not_true")


def _require_serialized_archive_delta(chain: Mapping[str, Any]) -> None:
    raw = chain.get("serialized_archive_delta")
    if raw is None:
        raise MaterializerChainHarvestError("serialized_archive_delta_missing")
    if not isinstance(raw, Mapping):
        raise MaterializerChainHarvestError("serialized_archive_delta_not_object")


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise MaterializerChainHarvestError(str(exc)) from exc


def _archive_record(
    chain: Mapping[str, Any],
    key: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    raw = chain.get(key)
    if not isinstance(raw, Mapping):
        raise MaterializerChainHarvestError(f"{key}_missing")
    path = _record_path(raw, repo_root=repo_root, label=key)
    observed_sha = _sha256_file(path)
    observed_bytes = path.stat().st_size
    declared_sha = _string_or_none(raw.get("sha256") or raw.get("archive_sha256"))
    declared_bytes = _positive_int(raw.get("bytes") or raw.get("archive_bytes"))
    if declared_sha is None:
        raise MaterializerChainHarvestError(f"{key}_sha256_missing")
    if declared_sha != observed_sha:
        raise MaterializerChainHarvestError(f"{key}_sha256_mismatch")
    if declared_bytes is None:
        raise MaterializerChainHarvestError(f"{key}_bytes_missing")
    if declared_bytes != observed_bytes:
        raise MaterializerChainHarvestError(f"{key}_bytes_mismatch")
    return {
        **dict(raw),
        "path": _repo_rel(path, repo_root),
        "sha256": observed_sha,
        "bytes": observed_bytes,
    }


def _serialized_delta_archive_custody_blockers(
    delta_facts: Mapping[str, Any],
    *,
    source_archive: Mapping[str, Any],
    candidate_archive: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    source_bytes = _positive_int(delta_facts.get("source_archive_bytes"))
    candidate_bytes = _positive_int(delta_facts.get("candidate_archive_bytes"))
    if source_bytes != source_archive["bytes"]:
        blockers.append(
            "serialized_archive_delta_source_bytes_mismatch:"
            f"{source_bytes}!={source_archive['bytes']}"
        )
    if candidate_bytes != candidate_archive["bytes"]:
        blockers.append(
            "serialized_archive_delta_candidate_bytes_mismatch:"
            f"{candidate_bytes}!={candidate_archive['bytes']}"
        )
    return blockers


def _record_path(record: Mapping[str, Any], *, repo_root: Path, label: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value.strip():
        raise MaterializerChainHarvestError(f"{label}_path_missing")
    path = Path(value)
    if path.is_absolute():
        raise MaterializerChainHarvestError(f"{label}_path_must_be_repo_relative")
    raw_path = repo_root / path
    if raw_path.is_symlink():
        raise MaterializerChainHarvestError(f"{label}_file_is_symlink:{path}")
    resolved = raw_path.resolve(strict=False)
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        raise MaterializerChainHarvestError(f"{label}_path_outside_repo") from None
    if not resolved.is_file():
        raise MaterializerChainHarvestError(f"{label}_file_missing:{path}")
    return resolved


def _validate_chain_artifacts(chain: Mapping[str, Any], *, repo_root: Path) -> None:
    artifacts = chain.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise MaterializerChainHarvestError("chain_artifacts_missing")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise MaterializerChainHarvestError(f"artifact_record_not_object:{name}")
        _validate_artifact_record(record, repo_root=repo_root, label=f"artifact:{name}")
    steps = chain.get("chain_steps")
    if not isinstance(steps, list) or not steps:
        raise MaterializerChainHarvestError("chain_steps_missing")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise MaterializerChainHarvestError(f"chain_step_not_object:{index}")
        if step.get("status") != "succeeded":
            raise MaterializerChainHarvestError(f"chain_step_not_succeeded:{index}")
        artifact = step.get("artifact")
        if isinstance(artifact, Mapping):
            _validate_artifact_record(
                artifact,
                repo_root=repo_root,
                label=f"chain_step_artifact:{index}",
            )


def _validate_artifact_record(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> None:
    path = _record_path(record, repo_root=repo_root, label=label)
    declared_sha = _string_or_none(record.get("sha256"))
    if declared_sha is None:
        raise MaterializerChainHarvestError(f"{label}_sha256_missing")
    if _sha256_file(path) != declared_sha:
        raise MaterializerChainHarvestError(f"{label}_sha256_mismatch")
    declared_bytes = _positive_int(record.get("bytes"))
    if declared_bytes is None:
        raise MaterializerChainHarvestError(f"{label}_bytes_missing")
    if path.stat().st_size != declared_bytes:
        raise MaterializerChainHarvestError(f"{label}_bytes_mismatch")


def _score_affecting_change_proof(
    *,
    source_sha: str | None,
    source_bytes: int | None,
    candidate_archive: Mapping[str, Any],
    archive_changed: bool,
    byte_changed: bool,
) -> dict[str, Any]:
    return {
        "source_archive_sha256": source_sha,
        "candidate_archive_sha256": candidate_archive["sha256"],
        "source_archive_bytes": source_bytes,
        "candidate_archive_bytes": candidate_archive["bytes"],
        "archive_changed": archive_changed,
        "byte_different": byte_changed,
    }


def _local_advisory_axes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mapping in _iter_mappings(payload):
        axis = mapping.get("score_axis") or mapping.get("axis")
        if not isinstance(axis, str):
            continue
        token = _axis_token(axis)
        if not any(item in token for item in LOCAL_ADVISORY_AXIS_TOKENS):
            continue
        out.append(
            {
                "score_axis": axis,
                "score": mapping.get("score") or mapping.get("score_recomputed"),
                "evidence_grade": mapping.get("evidence_grade"),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
            }
        )
    return _dedupe_advisory_axes(out)


def _dedupe_advisory_axes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = repr(sorted(row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for inner in value.values():
            yield from _iter_mappings(inner)
    elif isinstance(value, list | tuple):
        for inner in value:
            yield from _iter_mappings(inner)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable) and not isinstance(
        value, Mapping | bytes | bytearray
    ):
        return ordered_unique(str(item) for item in value if str(item))
    return [str(value)]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    return None


def _axis_token(value: str) -> str:
    return value.strip().strip("[]").lower().replace("_", "-").replace(" ", "-")


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        raise MaterializerChainHarvestError(f"path_outside_repo:{path}") from None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "SUPPORTED_CHAIN_SCHEMAS",
    "MaterializerChainHarvestError",
    "adapt_materializer_chain_manifest_to_candidate",
]
