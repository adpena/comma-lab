# SPDX-License-Identifier: MIT
"""Operator storage waterfall policy for bulky local experiment work.

This module is the policy sidecar. Storage probes and queue builders can import
it without learning the concrete external-volume order or cold-store layout.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

POLICY_ID = "operator_storage_waterfall.v1"
POLICY_SCHEMA = "comma_lab.operator_storage_waterfall.v1"
ARTIFACT_CATALOG_METADATA_SCHEMA = (
    "comma_lab.operator_storage_waterfall_artifact_catalog_metadata.v1"
)
DEFAULT_COLD_STORE_SUBDIR = "cold_store"
DEFAULT_WORK_TIER_ORDER: tuple[tuple[str, str], ...] = (
    ("vertigo", "/Volumes/VertigoDataTier/pact"),
    ("apdatastore", "/Volumes/APDataStore/pact"),
)

FALSE_AUTHORITY_FIELDS: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
    "field_selection_ready_for_exact_eval_dispatch": False,
}

GITIGNORED_STORAGE_PREFLIGHT_ARTIFACT_PATTERNS: tuple[str, ...] = (
    ".omx/research/dqs1_local_first_storage_plan_*.json",
    ".omx/research/dqs1_local_first_proactive_cleanup_*.json",
    ".omx/research/dqs1_local_first_proactive_cleanup_*.json.journal.jsonl",
    ".omx/research/byte_shaving_materializer_storage_plan_*.json",
    ".omx/research/byte_shaving_materializer_proactive_cleanup_*.json",
    ".omx/research/byte_shaving_materializer_proactive_cleanup_*.json.journal.jsonl",
    ".omx/research/*_artifact_retention_*.json",
    ".omx/research/*_artifact_retention_*.json.journal.jsonl",
)


@dataclass(frozen=True)
class OperatorStorageTier:
    """One policy or explicit-override tier in priority order."""

    name: str
    root: str
    priority: int
    source: str
    allow_local_disk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def with_false_authority(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the standard false-authority markers to a policy payload."""

    return {**payload, **FALSE_AUTHORITY_FIELDS}


def operator_work_tiers(
    storage_tier_overrides: Sequence[str] | None = None,
    *,
    allow_local_disk: bool = False,
) -> tuple[OperatorStorageTier, ...]:
    """Return policy tiers, or explicit ``name=/path`` overrides."""

    overrides = tuple(storage_tier_overrides or ())
    if overrides:
        return tuple(
            _parse_tier_override(
                value,
                priority=index,
                allow_local_disk=allow_local_disk,
            )
            for index, value in enumerate(overrides)
        )
    return tuple(
        OperatorStorageTier(
            name=name,
            root=root,
            priority=index,
            source="operator_policy",
            allow_local_disk=False,
        )
        for index, (name, root) in enumerate(DEFAULT_WORK_TIER_ORDER)
    )


def operator_storage_tier_cli_specs(
    storage_tier_overrides: Sequence[str] | None = None,
    *,
    allow_local_disk: bool = False,
) -> tuple[str, ...]:
    """Return ``name=/path`` specs in the effective policy order."""

    return tuple(
        f"{tier.name}={tier.root}"
        for tier in operator_work_tiers(
            storage_tier_overrides,
            allow_local_disk=allow_local_disk,
        )
    )


def operator_cold_store_roots(
    *,
    storage_tier_overrides: Sequence[str] | None = None,
    cold_store_root_overrides: Sequence[str] | None = None,
    cold_store_subdir: str = DEFAULT_COLD_STORE_SUBDIR,
    allow_local_disk: bool = False,
) -> tuple[str, ...]:
    """Return cold-store roots derived from the effective tier order."""

    explicit = tuple(str(root) for root in (cold_store_root_overrides or ()) if str(root))
    if explicit:
        return explicit
    subdir = _clean_subdir(cold_store_subdir)
    return tuple(
        str(Path(tier.root).expanduser() / subdir)
        for tier in operator_work_tiers(
            storage_tier_overrides,
            allow_local_disk=allow_local_disk,
        )
    )


def operator_storage_policy_payload(
    *,
    storage_tier_overrides: Sequence[str] | None = None,
    cold_store_root_overrides: Sequence[str] | None = None,
    cold_store_subdir: str = DEFAULT_COLD_STORE_SUBDIR,
    allow_local_disk: bool = False,
    policy_id: str = POLICY_ID,
    policy_schema: str = POLICY_SCHEMA,
) -> dict[str, Any]:
    """Emit the machine-readable operator policy payload."""

    tiers = operator_work_tiers(
        storage_tier_overrides,
        allow_local_disk=allow_local_disk,
    )
    cold_store_roots = operator_cold_store_roots(
        storage_tier_overrides=storage_tier_overrides,
        cold_store_root_overrides=cold_store_root_overrides,
        cold_store_subdir=cold_store_subdir,
        allow_local_disk=allow_local_disk,
    )
    return with_false_authority(
        {
            "schema": policy_schema,
            "policy_id": policy_id,
            "work_tier_order": [tier.to_dict() for tier in tiers],
            "storage_tier_cli_specs": [f"{tier.name}={tier.root}" for tier in tiers],
            "cold_store_roots": list(cold_store_roots),
            "cold_store_subdir": _clean_subdir(cold_store_subdir),
            "local_disk_enabled_by_default": False,
            "local_disk_enabled": bool(allow_local_disk),
            "local_disk_requires_explicit_opt_in": True,
            "explicit_storage_tier_override": bool(storage_tier_overrides),
            "explicit_cold_store_override": bool(cold_store_root_overrides),
        }
    )


def storage_preflight_artifact_catalog_metadata(
    *,
    storage_plan_path: str | Path | None,
    cleanup_plan_path: str | Path | None,
    journal_path: str | Path | None,
    lifecycle_kind: str = "HISTORICAL_PROVENANCE",
    policy_id: str = POLICY_ID,
    policy_schema: str = POLICY_SCHEMA,
) -> dict[str, Any]:
    """Describe storage/cleanup artifacts for queue and catalog consumers."""

    storage_path = _string_or_none(storage_plan_path)
    cleanup_path = _string_or_none(cleanup_plan_path)
    journal = _string_or_none(journal_path)
    artifact_paths = [path for path in (storage_path, cleanup_path, journal) if path]
    return with_false_authority(
        {
            "schema": ARTIFACT_CATALOG_METADATA_SCHEMA,
            "policy_id": policy_id,
            "policy_schema": policy_schema,
            "storage_plan_path": storage_path,
            "cleanup_plan_path": cleanup_path,
            "journal_path": journal,
            "artifact_paths": artifact_paths,
            "lifecycle_kind": lifecycle_kind,
            "tracked_local_boundary": {
                path: _tracked_local_boundary_for_path(path)
                for path in artifact_paths
            },
        }
    )


def _parse_tier_override(
    value: str,
    *,
    priority: int,
    allow_local_disk: bool,
) -> OperatorStorageTier:
    if "=" not in value:
        raise ValueError(f"storage tier must be name=/path, got {value!r}")
    name, raw_root = value.split("=", 1)
    if not name.strip() or not raw_root.strip():
        raise ValueError(f"storage tier must be name=/path, got {value!r}")
    return OperatorStorageTier(
        name=name.strip(),
        root=str(Path(raw_root.strip()).expanduser()),
        priority=priority,
        source="explicit_override",
        allow_local_disk=bool(allow_local_disk),
    )


def _clean_subdir(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("cold_store_subdir must be relative")
    parts = [part for part in path.parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError("cold_store_subdir may not contain '..'")
    return Path(*parts).as_posix() if parts else DEFAULT_COLD_STORE_SUBDIR


def _string_or_none(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _tracked_local_boundary_for_path(path: str) -> dict[str, Any]:
    ignored_pattern = _gitignored_storage_preflight_pattern(path)
    if ignored_pattern is not None:
        return {
            "artifact_kind_registry_pattern": ".omx/research/**/*.json*",
            "gitignore_pattern": ignored_pattern,
            "lifecycle_kind": "HISTORICAL_PROVENANCE",
            "git_boundary": "gitignored_local_control_artifact",
            "local_only": True,
        }
    if path.startswith(".omx/research/"):
        return {
            "artifact_kind_registry_pattern": ".omx/research/**/*.json*",
            "lifecycle_kind": "HISTORICAL_PROVENANCE",
            "git_boundary": "trackable_when_promoted",
            "local_only": False,
        }
    if path.startswith(".omx/state/"):
        return {
            "artifact_kind_registry_pattern": ".omx/state/**",
            "lifecycle_kind": "LIVE_STATE",
            "git_boundary": "stateful_append_or_versioned",
            "local_only": False,
        }
    if path.startswith("/Volumes/"):
        return {
            "artifact_kind_registry_pattern": None,
            "lifecycle_kind": "BULK_LOCAL_ARTIFACT",
            "git_boundary": "local_only_external_volume",
            "local_only": True,
        }
    return {
        "artifact_kind_registry_pattern": None,
        "lifecycle_kind": "LOCAL_OR_REPO_RELATIVE_ARTIFACT",
        "git_boundary": "caller_must_classify_before_commit",
        "local_only": True,
    }


def _gitignored_storage_preflight_pattern(path: str) -> str | None:
    for pattern in GITIGNORED_STORAGE_PREFLIGHT_ARTIFACT_PATTERNS:
        if fnmatch.fnmatch(path, pattern):
            return pattern
    return None


__all__ = [
    "ARTIFACT_CATALOG_METADATA_SCHEMA",
    "DEFAULT_COLD_STORE_SUBDIR",
    "DEFAULT_WORK_TIER_ORDER",
    "FALSE_AUTHORITY_FIELDS",
    "GITIGNORED_STORAGE_PREFLIGHT_ARTIFACT_PATTERNS",
    "POLICY_ID",
    "POLICY_SCHEMA",
    "OperatorStorageTier",
    "operator_cold_store_roots",
    "operator_storage_policy_payload",
    "operator_storage_tier_cli_specs",
    "operator_work_tiers",
    "storage_preflight_artifact_catalog_metadata",
    "with_false_authority",
]
