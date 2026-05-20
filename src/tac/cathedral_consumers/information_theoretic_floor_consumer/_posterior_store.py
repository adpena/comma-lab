# SPDX-License-Identifier: MIT
"""Canonical fcntl-locked APPEND-ONLY JSONL posterior store for the Tier B
paired-comparison observability surface.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5 + Catalog #131 (no bare
writes to shared state) + Catalog #110/#113 (HISTORICAL_PROVENANCE
APPEND-ONLY) + Catalog #245 (canonical 4-layer ledger pattern) + sister
DIM-1-PHASE-2-START phase_2_ablation_posterior pattern (commit
``e733c3dd4`` per `feedback_wave_3_dim_1_phase_2_start_per_adjuster_ablation_landed_20260520.md`).

Store path: ``.omx/state/consumer_tier_b_promotion_posterior.jsonl``
Lock path: ``.omx/state/consumer_tier_b_promotion_posterior.lock``

Schema (canonical row):
    consumer_name                  str
    consumer_version               str
    candidate_hint                 dict-form (best-effort serializable snapshot)
    tier_a_payload                 dict-form (full Tier A consume_candidate output)
    tier_b_payload                 dict-form (full Tier B consume_candidate output)
    divergence                     dict-form {
                                      "predicted_delta_adjustment_diff": float,
                                      "axis_tag_a": str,
                                      "axis_tag_b": str,
                                      "branch_kind_a": str,
                                      "branch_kind_b": str,
                                    }
    written_at_utc                 ISO 8601 UTC
    written_pid                    int
    written_host                   str
    row_uuid                       str (uuid4 for forensic dedupe)
    schema_version                 str (= SCHEMA_VERSION)
"""
from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "consumer_tier_b_promotion_posterior_v1_20260520"

# Resolve canonical store path relative to repo root via env var fallback.
# Mirrors sister tac.cost_band_calibration / tac.deploy.modal.call_id_ledger
# pattern.
def _resolve_state_dir() -> Path:
    """Return canonical .omx/state directory path."""
    env_repo_root = os.environ.get("PACT_REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root) / ".omx" / "state"
    # Walk upward from this file's location to find repo root.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".omx").is_dir() and (parent / "src").is_dir():
            return parent / ".omx" / "state"
    # Defensive fallback: relative to CWD.
    return Path.cwd() / ".omx" / "state"


def _resolve_store_path() -> Path:
    """Return canonical store JSONL path."""
    return _resolve_state_dir() / "consumer_tier_b_promotion_posterior.jsonl"


def _resolve_lock_path() -> Path:
    """Return canonical lock file path (sibling of store; Catalog #131
    sister discipline)."""
    return _resolve_state_dir() / "consumer_tier_b_promotion_posterior.lock"


CONSUMER_TIER_B_PROMOTION_POSTERIOR_PATH = _resolve_store_path()
CONSUMER_TIER_B_PROMOTION_POSTERIOR_LOCK_PATH = _resolve_lock_path()


def _utc_now_iso() -> str:
    """Canonical UTC ISO 8601 timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z").replace(
        "+0000", "+00:00"
    )


def _candidate_hint(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort serializable snapshot of candidate for forensic linking.

    Strips heavyweight tensor fields (M_contest / M_inflated) which would
    bloat the JSONL store; preserves identifying fields (substrate_name /
    m_contest_array_sha256 / archive_sha256 / candidate_id).
    """
    hint: dict[str, Any] = {}
    for key in (
        "substrate_name",
        "candidate_id",
        "archive_sha256",
        "m_contest_array_sha256",
        "lane_id",
        "information_theoretic_floor",
        "current_best_empirical_score",
        "floor_estimate_mode",
        "per_axis_floor",
        "per_axis_residuals",
        "substrate_fit_scores",
    ):
        if key in candidate:
            value = candidate[key]
            # JSON-safe coercion
            try:
                json.dumps(value)
                hint[key] = value
            except (TypeError, ValueError):
                hint[key] = repr(value)[:200]
    return hint


def _compute_divergence(
    tier_a_payload: Mapping[str, Any], tier_b_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute paired-comparison divergence summary."""
    a_delta = float(tier_a_payload.get("predicted_delta_adjustment", 0.0))
    b_delta = float(tier_b_payload.get("predicted_delta_adjustment", 0.0))
    return {
        "predicted_delta_adjustment_diff": float(b_delta - a_delta),
        "axis_tag_a": str(tier_a_payload.get("axis_tag", "")),
        "axis_tag_b": str(tier_b_payload.get("axis_tag", "")),
        "branch_kind_a": str(tier_a_payload.get("consumer_branch_kind", "")),
        "branch_kind_b": str(tier_b_payload.get("consumer_branch_kind", "")),
        "sign_flip": (a_delta * b_delta < 0.0),
        "abs_diff": abs(float(b_delta - a_delta)),
    }


def append_paired_comparison_row(
    *,
    consumer_name: str,
    consumer_version: str,
    tier_a_payload: Mapping[str, Any],
    tier_b_payload: Mapping[str, Any],
    candidate_hint: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one paired-comparison row to the canonical store under fcntl
    LOCK_EX.

    Returns the appended row (for caller logging / test verification).

    Per Catalog #131 + #110/#113 + #245: write is APPEND-ONLY; mutation of
    persisted rows is FORBIDDEN. Per Catalog #138 sister: corrupt JSONL
    rows are quarantined on read, never silently overwritten.

    The function is defensive: missing parent dir is auto-created; fcntl
    unavailability (e.g. Windows) is silently swallowed (caller handles).

    Args:
        consumer_name: canonical consumer name (e.g. "information_theoretic_floor_consumer").
        consumer_version: canonical consumer version (e.g. "2.0.0").
        tier_a_payload: full Tier A consume_candidate output dict.
        tier_b_payload: full Tier B consume_candidate output dict.
        candidate_hint: dict of candidate-identifying fields (will be
            JSON-coerced + truncated via :func:`_candidate_hint`).

    Returns:
        The full row that was appended (post-coercion).
    """
    import fcntl

    store_path = _resolve_store_path()
    lock_path = _resolve_lock_path()
    state_dir = store_path.parent
    state_dir.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "consumer_name": str(consumer_name),
        "consumer_version": str(consumer_version),
        "candidate_hint": _candidate_hint(candidate_hint),
        "tier_a_payload": _json_safe(tier_a_payload),
        "tier_b_payload": _json_safe(tier_b_payload),
        "divergence": _compute_divergence(tier_a_payload, tier_b_payload),
        "written_at_utc": _utc_now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
        "row_uuid": str(uuid.uuid4()),
    }

    serialized = json.dumps(row, sort_keys=True) + "\n"

    # Atomic fcntl-locked append per Catalog #131. The lock file is held
    # only for the duration of the write; readers do not need the lock
    # because JSONL append is atomic at the OS level for small rows.
    with open(lock_path, "a") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with open(store_path, "a", encoding="utf-8") as store_fh:
                store_fh.write(serialized)
                store_fh.flush()
                os.fsync(store_fh.fileno())
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    return row


def _json_safe(payload: Any) -> Any:
    """Best-effort JSON-safe coercion.

    Mappings and lists are recursively coerced; non-JSON-safe leaf values
    are repr()'d (truncated to 500 chars). This avoids canonical-payload
    persistence failures from numpy scalars, custom objects, etc.
    """
    if isinstance(payload, Mapping):
        return {str(k): _json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(v) for v in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    try:
        json.dumps(payload)
        return payload
    except (TypeError, ValueError):
        return repr(payload)[:500]


def load_paired_comparison_rows_lenient(
    store_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all paired-comparison rows lenient-mode (skip malformed).

    Per Catalog #138 sister: corrupt rows are SKIPPED (lenient); use
    :func:`load_paired_comparison_rows_strict` for fail-closed semantics.

    Returns empty list when store file does not exist (canonical empty
    posterior).
    """
    path = store_path or _resolve_store_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows


class PairedComparisonPosteriorCorruptError(Exception):
    """Raised by :func:`load_paired_comparison_rows_strict` on JSON parse
    failure per Catalog #138 sister discipline."""


def load_paired_comparison_rows_strict(
    store_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all paired-comparison rows STRICT (raise on malformed).

    Per Catalog #138 sister discipline: corrupt JSONL surfaces via
    PairedComparisonPosteriorCorruptError so the operator can quarantine
    the corrupt file via the canonical pattern.

    Returns empty list when store file does not exist.
    """
    path = store_path or _resolve_store_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise PairedComparisonPosteriorCorruptError(
                        f"line {line_no} of {path} failed JSON parse: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise PairedComparisonPosteriorCorruptError(
                        f"line {line_no} of {path} is not a JSON object: "
                        f"{type(row).__name__}"
                    )
                rows.append(row)
    except OSError as exc:
        raise PairedComparisonPosteriorCorruptError(
            f"failed to read {path}: {exc}"
        ) from exc
    return rows


__all__ = [
    "SCHEMA_VERSION",
    "CONSUMER_TIER_B_PROMOTION_POSTERIOR_PATH",
    "CONSUMER_TIER_B_PROMOTION_POSTERIOR_LOCK_PATH",
    "PairedComparisonPosteriorCorruptError",
    "append_paired_comparison_row",
    "load_paired_comparison_rows_lenient",
    "load_paired_comparison_rows_strict",
]
