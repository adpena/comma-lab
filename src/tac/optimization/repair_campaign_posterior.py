# SPDX-License-Identifier: MIT
"""Append-only repair campaign posterior for local stackability learning signals."""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
)
from tac.repo_io import json_line, json_text, sha256_file

_REPO_ROOT = Path(__file__).resolve().parents[3]

REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_ROW_SCHEMA = (
    "repair_campaign_stackability_posterior_row.v1"
)
REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA = (
    "repair_campaign_stackability_posterior_append_report.v1"
)
DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH = (
    _REPO_ROOT / ".omx" / "state" / "repair_campaign_stackability_posterior.jsonl"
)
DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH = (
    _REPO_ROOT / ".omx" / "state" / ".repair_campaign_stackability_posterior.lock"
)


class RepairCampaignPosteriorError(ValueError):
    """Raised when a repair campaign posterior row cannot be appended."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path: str | Path, *, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, *, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json_text(payload).encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _source_signal_record(
    signal_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    resolved = _resolve(signal_path, repo_root=repo_root)
    if not resolved.is_file():
        raise RepairCampaignPosteriorError(
            f"required learning signal artifact missing: {signal_path}"
        )
    return {
        "label": "repair_campaign_learning_signal",
        "path": _repo_rel(resolved, repo_root=repo_root),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def build_repair_campaign_stackability_posterior_row(
    *,
    learning_signal_path: str | Path,
    learning_signal: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build one deterministic false-authority posterior row."""

    if learning_signal.get("schema") != REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA:
        raise RepairCampaignPosteriorError(
            "learning signal must be repair_campaign_learning_signal.v1"
        )
    require_no_truthy_authority_fields(
        learning_signal,
        context="repair_campaign_stackability_posterior_learning_signal",
    )
    replay_identity = _mapping(learning_signal.get("replay_identity"))
    hash_manifest_sha = str(replay_identity.get("hash_manifest_sha256") or "").strip()
    typed_response_id = str(learning_signal.get("typed_response_id") or "").strip()
    if not typed_response_id:
        raise RepairCampaignPosteriorError("learning signal missing typed_response_id")
    if not hash_manifest_sha:
        raise RepairCampaignPosteriorError(
            "learning signal missing replay hash_manifest_sha256"
        )
    local_update = _mapping(learning_signal.get("local_planning_update"))
    feature_vector = _mapping(local_update.get("planner_feature_vector"))
    acquisition_policy = str(local_update.get("recommended_acquisition_policy") or "")
    row_identity = {
        "schema": "repair_campaign_stackability_posterior_row_identity.v1",
        "typed_response_id": typed_response_id,
        "candidate_id": learning_signal.get("candidate_id"),
        "family_id": learning_signal.get("family_id"),
        "hash_manifest_sha256": hash_manifest_sha,
        "source_records_sha256": replay_identity.get("source_records_sha256"),
        "replay_argv_sha256": replay_identity.get("replay_argv_sha256"),
    }
    row = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_ROW_SCHEMA,
        "row_id": _stable_sha256(row_identity),
        "row_identity": row_identity,
        "ingested_at_utc": _utc_now(),
        "typed_response_id": typed_response_id,
        "candidate_id": learning_signal.get("candidate_id"),
        "family_id": learning_signal.get("family_id"),
        "component_response_axis": learning_signal.get("component_response_axis"),
        "evidence_grade": learning_signal.get("evidence_grade"),
        "source_signal": _source_signal_record(
            learning_signal_path,
            repo_root=repo_root,
        ),
        "replay_identity": dict(replay_identity),
        "local_planning_update": dict(local_update),
        "planner_feature_vector": dict(feature_vector),
        "acquisition_policy_delta": {
            "schema": "repair_campaign_acquisition_policy_delta.v1",
            "recommended_acquisition_policy": acquisition_policy,
            "family_priority_direction": (
                "increase"
                if acquisition_policy
                == "increase_priority_for_exact_axis_component_response_replay"
                else "hold"
            ),
            "expected_local_improvement_score_units": _safe_float(
                feature_vector.get("expected_local_improvement_score_units")
            ),
            "improvement_per_allocated_byte": _safe_float(
                feature_vector.get("improvement_per_allocated_byte")
            ),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "blockers": list(learning_signal.get("blockers") or []),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_campaign_acquisition_prior_update_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context="repair_campaign_stackability_posterior_row",
    )
    return row


def load_repair_campaign_stackability_posterior_rows(
    posterior_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load existing posterior rows from JSONL."""

    path = Path(posterior_path or DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RepairCampaignPosteriorError(
                f"{path}: invalid JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RepairCampaignPosteriorError(
                f"{path}: posterior row {line_number} must be a JSON object"
            )
        rows.append(payload)
    return rows


@contextmanager
def _posterior_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def append_repair_campaign_stackability_posterior_signal(
    *,
    learning_signal_path: str | Path,
    learning_signal: Mapping[str, Any],
    posterior_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    repo_root: str | Path = _REPO_ROOT,
) -> dict[str, Any]:
    """Append one learning signal to the repair posterior with duplicate suppression."""

    row = build_repair_campaign_stackability_posterior_row(
        learning_signal_path=learning_signal_path,
        learning_signal=learning_signal,
        repo_root=repo_root,
    )
    path = Path(posterior_path or DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH)
    lock = Path(lock_path or DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH)
    with _posterior_lock(lock):
        rows = load_repair_campaign_stackability_posterior_rows(path)
        existing = next(
            (
                item
                for item in rows
                if isinstance(item, Mapping) and item.get("row_id") == row["row_id"]
            ),
            None,
        )
        appended = existing is None
        if appended:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json_line(row))
    report = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA,
        "posterior_path": _repo_rel(path, repo_root=repo_root),
        "lock_path": _repo_rel(lock, repo_root=repo_root),
        "row_id": row["row_id"],
        "typed_response_id": row["typed_response_id"],
        "candidate_id": row.get("candidate_id"),
        "family_id": row.get("family_id"),
        "appended": appended,
        "skipped_duplicate": not appended,
        "existing_row_count": len(rows),
        "final_row_count": len(rows) + (1 if appended else 0),
        "source_signal": row["source_signal"],
        "acquisition_policy_delta": row["acquisition_policy_delta"],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_campaign_stackability_posterior_append_audit",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_stackability_posterior_append_report",
    )
    return report


__all__ = [
    "DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH",
    "DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH",
    "REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA",
    "REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_ROW_SCHEMA",
    "RepairCampaignPosteriorError",
    "append_repair_campaign_stackability_posterior_signal",
    "build_repair_campaign_stackability_posterior_row",
    "load_repair_campaign_stackability_posterior_rows",
]
