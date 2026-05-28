# SPDX-License-Identifier: MIT
"""Canonical anti-patterns registry — fcntl-locked JSONL APPEND-ONLY ledger.

Sister of ``tac.canonical_equations.registry`` at the NEGATIVE-registry
surface. Mirrors the Catalog #245 Modal call_id ledger + Catalog #313
probe-outcomes ledger pattern: every event is a new row referencing the
same ``anti_pattern_id`` so the historical audit trail of registrations +
falsification appends + recalibration events is reconstructable via
``query_*`` helpers.

Schema (one event per JSONL row):

    {
        "schema_version": "canonical_anti_pattern_v1_20260528",
        "event_type": "anti_pattern_registered" | "falsification_appended"
                      | "anti_pattern_recalibrated" | "unwind_path_ratified",
        "anti_pattern_id": "lzma_on_already_brotli_saturated_compounding_v1",
        "anti_pattern_payload": { ... AntiPattern.to_dict() ... },
        "written_at_utc": "...",
        "written_pid": 12345,
        "written_host": "macbook-air",
        "agent": "claude" | "codex" | "operator",
        "subagent_id": "...",
        "notes": "free-form context",
    }

Path discipline:
  * Ledger: ``.omx/state/canonical_anti_patterns_registry.jsonl`` (committed
    per HISTORICAL_PROVENANCE per Catalog #110/#113).
  * Lock file: ``.omx/state/canonical_anti_patterns_registry.jsonl.lock``
    (gitignored LIVE_STATE).
  * Bare writes to the path SHOULD be refused by Catalog #131 sister gate
    (path registration in ``_SHARED_STATE_PATH_MARKERS`` queued for follow-on).

Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132): APPEND-ONLY
— the prior payload is preserved verbatim; only a NEW event row is added.
Mirrors ``tac.canonical_equations.registry`` semantics exactly.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from tac.canonical_anti_patterns.anti_pattern import (
    CANONICAL_ANTI_PATTERN_SCHEMA_VERSION,
    RECALIBRATE_ON_NEW_FALSIFICATIONS,
    AntiPattern,
    EmpiricalFalsification,
    InvalidAntiPatternError,
    _utc_now_iso,
)
from tac.provenance.contract import (
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ANTI_PATTERNS_REGISTRY_PATH = (
    REPO_ROOT / ".omx" / "state" / "canonical_anti_patterns_registry.jsonl"
)
CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK = (
    CANONICAL_ANTI_PATTERNS_REGISTRY_PATH.with_suffix(
        CANONICAL_ANTI_PATTERNS_REGISTRY_PATH.suffix + ".lock"
    )
)

# Lock acquisition timeout (seconds). Mirrors Catalog #245/#313/#344 default.
LOCK_TIMEOUT_SECONDS = 30

# Canonical event taxonomy.
EVENT_ANTI_PATTERN_REGISTERED = "anti_pattern_registered"
EVENT_FALSIFICATION_APPENDED = "falsification_appended"
EVENT_ANTI_PATTERN_RECALIBRATED = "anti_pattern_recalibrated"
EVENT_UNWIND_PATH_RATIFIED = "unwind_path_ratified"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_ANTI_PATTERN_REGISTERED,
        EVENT_FALSIFICATION_APPENDED,
        EVENT_ANTI_PATTERN_RECALIBRATED,
        EVENT_UNWIND_PATH_RATIFIED,
    }
)


class AntiPatternRegistryCorruptError(RuntimeError):
    """Raised when the registry ledger is corrupt; mirrors Catalog #245 sister."""


_registry_lock_depth_tls = threading.local()


def _get_lock_depth() -> int:
    return int(getattr(_registry_lock_depth_tls, "depth", 0))


def _set_lock_depth(value: int) -> None:
    _registry_lock_depth_tls.depth = int(value)


def _registry_lock_held() -> bool:
    return _get_lock_depth() > 0


@contextlib.contextmanager
def _registry_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the registry lock file.

    Re-entry is counted (depth > 1); fcntl is only re-acquired on 0->1.
    """
    p = lock_path or CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    depth = _get_lock_depth()
    if depth > 0:
        _set_lock_depth(depth + 1)
        try:
            yield None
        finally:
            _set_lock_depth(_get_lock_depth() - 1)
        return
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not acquire {p} within {LOCK_TIMEOUT_SECONDS}s"
                    ) from None
                time.sleep(0.05)
        _set_lock_depth(_get_lock_depth() + 1)
        try:
            yield fd
        finally:
            _set_lock_depth(_get_lock_depth() - 1)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _quarantine_corrupt_file(path: Path) -> Path:
    if not path.exists():
        return path
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}")
    counter = 0
    while quarantine.exists():
        counter += 1
        quarantine = path.with_suffix(path.suffix + f".corrupt.{ts}.{counter}")
    os.rename(path, quarantine)
    return quarantine


def load_anti_patterns_events_lenient(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all registry events; skip malformed lines silently (read-only callers)."""
    p = path or CANONICAL_ANTI_PATTERNS_REGISTRY_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def load_anti_patterns_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Strict load for mutating callers; raises on corrupt state (Catalog #138)."""
    p = path or CANONICAL_ANTI_PATTERNS_REGISTRY_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise AntiPatternRegistryCorruptError(
            f"canonical anti-patterns registry at {p} could not be read: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError as exc:
            raise AntiPatternRegistryCorruptError(
                f"canonical anti-patterns registry at {p} line {lineno}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(r, dict):
            raise AntiPatternRegistryCorruptError(
                f"canonical anti-patterns registry at {p} line {lineno}: non-dict root "
                f"(type={type(r).__name__})"
            )
        rows.append(r)
    return rows


def _validate_event_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CANONICAL_ANTI_PATTERN_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CANONICAL_ANTI_PATTERN_SCHEMA_VERSION!r}"
        )
    if record.get("event_type") not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}"
        )
    ap_id = record.get("anti_pattern_id")
    if not isinstance(ap_id, str) or not ap_id.strip():
        raise ValueError("anti_pattern_id must be a non-empty string")
    if not isinstance(record.get("anti_pattern_payload"), dict):
        raise ValueError("anti_pattern_payload must be a dict")


def _save_ledger(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write under lock — tmp + fsync + os.replace."""
    if not _registry_lock_held():
        raise RuntimeError(
            "_save_ledger called WITHOUT holding _registry_lock; this is a "
            "CONCURRENCY BUG per Catalog #140 (state writers own their lock end-to-end)."
        )
    p = path or CANONICAL_ANTI_PATTERNS_REGISTRY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _append_event_locked(
    event_type: str,
    anti_pattern: AntiPattern,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Append a single registry event under fcntl lock."""
    record = {
        "schema_version": CANONICAL_ANTI_PATTERN_SCHEMA_VERSION,
        "event_type": event_type,
        "anti_pattern_id": anti_pattern.anti_pattern_id,
        "anti_pattern_payload": anti_pattern.to_dict(),
        "written_at_utc": _utc_now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
        "agent": agent or "claude",
        "subagent_id": subagent_id,
        "notes": notes,
    }
    _validate_event_record(record)
    p_path = path or CANONICAL_ANTI_PATTERNS_REGISTRY_PATH
    l_path = lock_path or CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK

    with _registry_lock(l_path):
        try:
            existing = load_anti_patterns_strict(p_path)
        except AntiPatternRegistryCorruptError:
            _quarantine_corrupt_file(p_path)
            existing = []
        existing.append(record)
        _save_ledger(existing, p_path)
    return record


def register_anti_pattern(
    anti_pattern: AntiPattern,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> AntiPattern:
    """Append an 'anti_pattern_registered' event; returns the anti-pattern echo."""
    if not isinstance(anti_pattern, AntiPattern):
        raise InvalidAntiPatternError(
            f"register_anti_pattern expected AntiPattern, got {type(anti_pattern).__name__}"
        )
    _append_event_locked(
        EVENT_ANTI_PATTERN_REGISTERED,
        anti_pattern,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=notes,
    )
    return anti_pattern


def append_empirical_falsification(
    falsification: EmpiricalFalsification,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> AntiPattern:
    """Append a new EmpiricalFalsification; emits 'falsification_appended' event.

    Loads latest payload for the parent ``anti_pattern_id``, calls
    ``with_new_falsification``, persists the updated anti-pattern as a new
    event. Per CLAUDE.md "Locked writes preserve deletions" (Catalog #132):
    APPEND-ONLY — the prior payload is preserved verbatim; only a NEW
    event row is added. Mirrors
    ``tac.canonical_equations.registry.update_equation_with_empirical_anchor``.
    """
    if not isinstance(falsification, EmpiricalFalsification):
        raise InvalidAntiPatternError(
            f"append_empirical_falsification expected EmpiricalFalsification, "
            f"got {type(falsification).__name__}"
        )
    p_path = path or CANONICAL_ANTI_PATTERNS_REGISTRY_PATH
    l_path = lock_path or CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK

    with _registry_lock(l_path):
        try:
            existing = load_anti_patterns_strict(p_path)
        except AntiPatternRegistryCorruptError:
            _quarantine_corrupt_file(p_path)
            existing = []
        latest_payload = None
        for row in existing:
            if row.get("anti_pattern_id") == falsification.anti_pattern_id:
                latest_payload = row.get("anti_pattern_payload")
        if latest_payload is None:
            raise InvalidAntiPatternError(
                f"anti_pattern_id={falsification.anti_pattern_id!r} not found "
                "in registry; call register_anti_pattern first"
            )
        anti_pattern = _anti_pattern_from_dict(latest_payload)
        updated = anti_pattern.with_new_falsification(falsification)
        _append_event_locked(
            EVENT_FALSIFICATION_APPENDED,
            updated,
            path=p_path,
            lock_path=l_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=notes,
        )
    return updated


def _prov_from_dict(d: Mapping[str, Any]) -> Provenance:
    """Reconstruct a Provenance from its dict form; mirrors sister registry helper."""
    kind = ProvenanceKind(d["artifact_kind"])
    grade = ProvenanceEvidenceGrade(d["evidence_grade"])
    composed_raw = d.get("composed_from", ()) or ()
    composed: tuple[Provenance, ...] = tuple(
        _prov_from_dict(c) for c in composed_raw if isinstance(c, Mapping)
    )
    return Provenance(
        artifact_kind=kind,
        source_path=d["source_path"],
        source_sha256=d["source_sha256"],
        measurement_axis=d["measurement_axis"],
        hardware_substrate=d["hardware_substrate"],
        evidence_grade=grade,
        promotion_eligible=bool(d.get("promotion_eligible", False)),
        score_claim_valid=bool(d.get("score_claim_valid", False)),
        captured_at_utc=d["captured_at_utc"],
        canonical_helper_invocation=d.get("canonical_helper_invocation", "unknown"),
        contest_archive_zip_path=d.get("contest_archive_zip_path", ""),
        contest_archive_member_name=d.get("contest_archive_member_name", ""),
        composed_from=composed,
        rejection_reason=d.get("rejection_reason", ""),
    )


def _anti_pattern_from_dict(payload: Mapping[str, Any]) -> AntiPattern:
    """Reconstruct an AntiPattern from a serialized payload dict."""
    falsifications = tuple(
        EmpiricalFalsification(
            anti_pattern_id=f["anti_pattern_id"],
            falsification_id=f["falsification_id"],
            measurement_method=f["measurement_method"],
            empirical_artifact_path=f["empirical_artifact_path"],
            empirical_output=f["empirical_output"],
            falsification_residual=(
                float(f["falsification_residual"])
                if f.get("falsification_residual") is not None
                else None
            ),
            captured_at_utc=f["captured_at_utc"],
            canonical_provenance=_prov_from_dict(f["canonical_provenance"]),
            incident_classification=f["incident_classification"],
            severity_observed=f["severity_observed"],
            operator_routable_unwind_path=f["operator_routable_unwind_path"],
        )
        for f in payload.get("empirical_falsifications", [])
    )
    return AntiPattern(
        anti_pattern_id=payload["anti_pattern_id"],
        description=payload["description"],
        forbidden_pattern_predicate=payload["forbidden_pattern_predicate"],
        falsification_band=payload["falsification_band"],
        recurrence_conditions=tuple(payload.get("recurrence_conditions", [])),
        canonical_source_anchor=payload["canonical_source_anchor"],
        canonical_unwind_path=payload["canonical_unwind_path"],
        canonical_producers=tuple(payload.get("canonical_producers", [])),
        canonical_consumers=tuple(payload.get("canonical_consumers", [])),
        paradigm_class=payload["paradigm_class"],
        severity=payload["severity"],
        provenance=_prov_from_dict(payload["provenance"]),
        empirical_falsifications=falsifications,
        last_recalibration_utc=payload["last_recalibration_utc"],
        next_recalibration_trigger=payload["next_recalibration_trigger"],
        schema_version=payload.get(
            "schema_version", CANONICAL_ANTI_PATTERN_SCHEMA_VERSION
        ),
    )


def query_anti_patterns(
    *,
    path: Path | None = None,
) -> list[AntiPattern]:
    """Return latest payload per anti_pattern_id as reconstructed AntiPattern."""
    rows = load_anti_patterns_events_lenient(path)
    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        ap_id = row.get("anti_pattern_id")
        if isinstance(ap_id, str):
            latest_by_id[ap_id] = row.get("anti_pattern_payload", {})
    out: list[AntiPattern] = []
    for payload in latest_by_id.values():
        try:
            out.append(_anti_pattern_from_dict(payload))
        except (KeyError, InvalidAntiPatternError):
            # Skip historical rows that fail current invariants (forward-compat).
            continue
    return out


def query_anti_patterns_by_substrate(
    substrate_id: str,
    *,
    path: Path | None = None,
) -> list[AntiPattern]:
    """Return anti-patterns whose canonical_producers / consumers / description / forbidden predicate mentions the substrate.

    Best-effort substring match across multiple fields so e.g. ``"pact_nerv"``
    surfaces anti-patterns whose producers list ``"experiments/train_substrate_pact_nerv_*"``
    OR whose recurrence conditions mention ``"pact_nerv selector"``.
    """
    token = (substrate_id or "").lower().strip()
    if not token:
        return []
    out: list[AntiPattern] = []
    for ap in query_anti_patterns(path=path):
        haystack_parts = [
            ap.description.lower(),
            ap.forbidden_pattern_predicate.lower(),
            ap.canonical_unwind_path.lower(),
            " ".join(c.lower() for c in ap.recurrence_conditions),
            " ".join(p.lower() for p in ap.canonical_producers),
            " ".join(c.lower() for c in ap.canonical_consumers),
        ]
        haystack = " ".join(haystack_parts)
        if token in haystack:
            out.append(ap)
    return out


def query_falsifications_by_paradigm_class(
    paradigm_class: str,
    *,
    path: Path | None = None,
) -> list[EmpiricalFalsification]:
    """Return all empirical falsifications whose parent anti-pattern has the given paradigm_class."""
    if not paradigm_class:
        return []
    out: list[EmpiricalFalsification] = []
    for ap in query_anti_patterns(path=path):
        if ap.paradigm_class == paradigm_class:
            out.extend(ap.empirical_falsifications)
    return out


def query_recurrence_rate_by_severity(
    *,
    path: Path | None = None,
) -> dict[str, float]:
    """Compute per-severity recurrence rate.

    Returns a dict mapping severity → proportion of anti-patterns at that
    severity whose ``is_actively_recurring`` is True (i.e. >= 2 empirical
    falsifications recorded). Operator-facing summary so a single CLI
    query shows whether CRITICAL anti-patterns are recurring more
    frequently than LOW ones (cadence audit signal).

    Returns empty dict if no anti-patterns registered.
    """
    counts_by_severity: dict[str, list[int]] = {}
    for ap in query_anti_patterns(path=path):
        bucket = counts_by_severity.setdefault(ap.severity, [])
        bucket.append(1 if ap.is_actively_recurring else 0)
    return {
        sev: (sum(bucket) / len(bucket)) if bucket else 0.0
        for sev, bucket in counts_by_severity.items()
    }


def get_anti_pattern_by_id(
    anti_pattern_id: str,
    *,
    path: Path | None = None,
) -> AntiPattern | None:
    """Lookup an anti-pattern by id; returns None if absent."""
    for ap in query_anti_patterns(path=path):
        if ap.anti_pattern_id == anti_pattern_id:
            return ap
    return None


@dataclass(frozen=True)
class AntiPatternRecalibrationReport:
    """Result of auto_recalibrate_from_continual_learning_posterior.

    Mirrors ``RecalibrationReport`` from sister canonical_equations registry
    at the NEGATIVE-registry surface.

    Reports which anti-patterns had falsification_band refreshed + how many
    new falsifications were absorbed. Operator-facing summary so the
    severity drift is visible in CLI output.
    """

    anti_patterns_evaluated: int
    anti_patterns_recalibrated: int
    anti_patterns_eligible_but_unchanged: int
    per_anti_pattern_summary: dict[str, dict[str, Any]] = field(default_factory=dict)


# Canonical trigger threshold (mirrors RECALIBRATE_ON_NEW_FALSIFICATIONS) at
# which the 3+-falsification auto-refit fires.
_AUTO_REFIT_MIN_FALSIFICATIONS = 3


def _refit_band_from_falsifications(
    anti_pattern: AntiPattern,
) -> dict[str, float]:
    """Recompute the falsification_band directly from landed falsifications.

    Evidence-faithful refit: each landed :class:`EmpiricalFalsification`
    carries a non-negative ``falsification_residual`` (or None if no
    numeric residual is meaningful). The refit derives a per-axis (keyed
    by ``measurement_method``) band:

      ``<method>_residual_lo`` = min observed non-None residual on that axis
      ``<method>_residual_hi`` = max observed non-None residual on that axis
      ``<method>_falsification_count`` = number of falsifications on axis

    Per the canonical_equations sister at Catalog #371: this does NOT
    synthesize new falsifications and does NOT fabricate measurements
    (Catalog #287 / #323). It only re-derives the band SUMMARY from
    falsifications that already landed via signed
    ``append_empirical_falsification`` calls. The stale-prior bug class
    (an anti-pattern whose stored falsification_band no longer matches its
    own landed falsifications because a sister appended falsifications but
    never re-summarized) is what this extincts.
    """
    refit: dict[str, float] = {}
    by_method: dict[str, list[float]] = {}
    for fals in anti_pattern.empirical_falsifications:
        method = fals.measurement_method
        if fals.falsification_residual is None:
            # Track presence even without numeric residual so count is honest.
            by_method.setdefault(method, [])
            continue
        by_method.setdefault(method, []).append(float(fals.falsification_residual))
    for method, residuals in by_method.items():
        count_key = f"{method}_falsification_count"
        refit[count_key] = float(len(by_method[method]))
        # Only emit lo/hi when at least one numeric residual exists for the axis.
        if residuals:
            refit[f"{method}_residual_lo"] = float(min(residuals))
            refit[f"{method}_residual_hi"] = float(max(residuals))
    return refit


def auto_recalibrate_from_continual_learning_posterior(
    anti_pattern_id: str | None = None,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> AntiPatternRecalibrationReport:
    """Periodic auto-recalibration of canonical anti-pattern falsification_band.

    For every anti-pattern whose ``next_recalibration_trigger`` is
    ``when_3+_new_empirical_falsifications_in_domain`` AND whose
    falsification count is >= ``_AUTO_REFIT_MIN_FALSIFICATIONS`` (3), this
    helper re-derives ``falsification_band`` directly from the anti-pattern's
    own landed :class:`EmpiricalFalsification` rows and, if that recomputed
    band differs from the stored band, appends a canonical
    ``anti_pattern_recalibrated`` event (bumping ``last_recalibration_utc``).
    The original payload is preserved verbatim per Catalog #110/#113
    APPEND-ONLY; only a NEW event row is written.

    Why this is NOT a Catalog #287 violation: no falsification is synthesized
    and no empirical_output is fabricated. The refit only re-SUMMARIZES
    residuals from falsifications that already landed via signed
    ``append_empirical_falsification`` calls.

    Per the lesson encoded by Catalog #371 (sister at canonical_equations
    surface): this is NOT a stub. It actually re-derives the band from
    landed falsifications and is idempotent (2nd run with no new
    falsifications recalibrates 0 anti-patterns).

    Anti-patterns on other triggers (``when_severity_drift_exceeds_2x`` /
    ``when_operator_invokes_recalibrate_anti_pattern`` /
    ``never_auto_operator_only``) are reported but NOT auto-refit here —
    those require an operator invocation or a drift-detection path, by
    design.

    Returns an :class:`AntiPatternRecalibrationReport`.
    """
    anti_patterns = query_anti_patterns(path=path)
    if anti_pattern_id is not None:
        anti_patterns = [
            ap for ap in anti_patterns if ap.anti_pattern_id == anti_pattern_id
        ]

    summary: dict[str, dict[str, Any]] = {}
    recalibrated_count = 0
    eligible_unchanged = 0

    for ap in anti_patterns:
        trigger = ap.next_recalibration_trigger
        fals_count = len(ap.empirical_falsifications)
        refit_eligible = (
            trigger == RECALIBRATE_ON_NEW_FALSIFICATIONS
            and fals_count >= _AUTO_REFIT_MIN_FALSIFICATIONS
        )
        refit_band = _refit_band_from_falsifications(ap) if refit_eligible else {}
        stored_band = dict(ap.falsification_band)
        # Drift detection: did the recomputed summary diverge from the stored
        # summary?
        changed = refit_eligible and refit_band != stored_band
        did_recalibrate = False

        if changed:
            updated = replace(
                ap,
                falsification_band=refit_band,
                last_recalibration_utc=_utc_now_iso(),
            )
            n_keys_changed = sum(
                1 for k, v in refit_band.items() if stored_band.get(k) != v
            ) + sum(1 for k in stored_band if k not in refit_band)
            _append_event_locked(
                EVENT_ANTI_PATTERN_RECALIBRATED,
                updated,
                path=path,
                lock_path=lock_path,
                agent=agent or "claude",
                subagent_id=subagent_id,
                notes=(
                    f"auto_recalibrate: refit falsification_band from "
                    f"{fals_count} landed falsifications; {n_keys_changed} band "
                    f"key(s) changed (stale-prior orphan extinction)"
                ),
            )
            recalibrated_count += 1
            did_recalibrate = True
            ap = updated  # report the post-refit state
        elif refit_eligible:
            eligible_unchanged += 1

        summary[ap.anti_pattern_id] = {
            "falsification_count": fals_count,
            "current_band": dict(ap.falsification_band),
            "is_actively_recurring": ap.is_actively_recurring,
            "trigger": trigger,
            "refit_eligible": refit_eligible,
            "recalibrated": did_recalibrate,
        }

    return AntiPatternRecalibrationReport(
        anti_patterns_evaluated=len(anti_patterns),
        anti_patterns_recalibrated=recalibrated_count,
        anti_patterns_eligible_but_unchanged=eligible_unchanged,
        per_anti_pattern_summary=summary,
    )


__all__ = [
    "CANONICAL_ANTI_PATTERNS_REGISTRY_PATH",
    "CANONICAL_ANTI_PATTERNS_REGISTRY_LOCK",
    "LOCK_TIMEOUT_SECONDS",
    "EVENT_ANTI_PATTERN_REGISTERED",
    "EVENT_FALSIFICATION_APPENDED",
    "EVENT_ANTI_PATTERN_RECALIBRATED",
    "EVENT_UNWIND_PATH_RATIFIED",
    "VALID_EVENT_TYPES",
    "AntiPatternRegistryCorruptError",
    "AntiPatternRecalibrationReport",
    "load_anti_patterns_events_lenient",
    "load_anti_patterns_strict",
    "register_anti_pattern",
    "append_empirical_falsification",
    "query_anti_patterns",
    "query_anti_patterns_by_substrate",
    "query_falsifications_by_paradigm_class",
    "query_recurrence_rate_by_severity",
    "get_anti_pattern_by_id",
    "auto_recalibrate_from_continual_learning_posterior",
]
