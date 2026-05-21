# SPDX-License-Identifier: MIT
"""Canonical equations registry — fcntl-locked JSONL APPEND-ONLY ledger.

Per operator NON-NEGOTIABLE 2026-05-19. Mirrors the Catalog #245 Modal
call_id ledger + Catalog #313 probe-outcomes ledger pattern: every event
is a new row referencing the same ``equation_id`` so the historical audit
trail of registrations + anchor updates + recalibration events is
reconstructable via ``query_equations_by_*`` helpers.

Schema (one event per JSONL row):

    {
        "schema_version": "canonical_equation_v1_20260519",
        "event_type": "registered" | "anchor_appended" | "recalibrated" | "deprecated",
        "equation_id": "mps_drift_architecture_class_dependent_v1",
        "equation_payload": { ... CanonicalEquation.to_dict() ... },
        "written_at_utc": "...",
        "written_pid": 12345,
        "written_host": "macbook-air",
        "agent": "claude" | "codex" | "operator",
        "subagent_id": "...",
        "notes": "free-form context",
    }

The latest event per ``equation_id`` is the current state; query helpers
reduce-by-equation-id to surface only the most recent payload.

Path discipline:
  * Ledger: ``.omx/state/canonical_equations_registry.jsonl`` (committed
    per HISTORICAL_PROVENANCE per Catalog #110/#113).
  * Lock file: ``.omx/state/canonical_equations_registry.jsonl.lock``
    (gitignored LIVE_STATE).
  * Bare writes to the path are refused by Catalog #131 sister gate
    (path is registered in ``_SHARED_STATE_PATH_MARKERS``).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tac.canonical_equations.equation import (
    CANONICAL_EQUATION_SCHEMA_VERSION,
    CanonicalEquation,
    EmpiricalAnchor,
    InvalidEquationError,
    _utc_now_iso,
)
from tac.provenance.contract import (
    Provenance,
    ProvenanceEvidenceGrade,
    ProvenanceKind,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_EQUATIONS_REGISTRY_PATH = (
    REPO_ROOT / ".omx" / "state" / "canonical_equations_registry.jsonl"
)
CANONICAL_EQUATIONS_REGISTRY_LOCK = CANONICAL_EQUATIONS_REGISTRY_PATH.with_suffix(
    CANONICAL_EQUATIONS_REGISTRY_PATH.suffix + ".lock"
)

# Lock acquisition timeout (seconds). Mirrors Catalog #245/#313 default.
LOCK_TIMEOUT_SECONDS = 30

# Canonical event taxonomy.
EVENT_REGISTERED = "registered"
EVENT_ANCHOR_APPENDED = "anchor_appended"
EVENT_RECALIBRATED = "recalibrated"
EVENT_DEPRECATED = "deprecated"
# WAVE-3-CANONICAL-EQUATION-26-DOMAIN-REFINEMENT 2026-05-20: NEW event type for
# evidence-driven domain_of_validity narrowing. Per Catalog #110/#113 APPEND-ONLY
# HISTORICAL_PROVENANCE: a domain_refined event NEVER mutates the prior
# registered/anchor_appended rows; it appends a NEW payload row with extended
# domain_of_validity fields (_included + _excluded) so downstream consumers
# (cathedral autopilot reranker / canonical_equation_lookup_consumer / future
# canonical-equation lookups) cannot silently re-make the cargo-culted assumption
# the empirical anchor falsified. Sister anchor: DWT-DETAIL-SUBBAND CPU SMOKE
# (commit f25f8cc1b; KL=1.638 nats / 3.28σ) proving direct procedural-codebook
# substitution on DWT detail subbands corrupts inverse DWT.
EVENT_DOMAIN_REFINED = "domain_refined"

VALID_EVENT_TYPES = frozenset(
    {
        EVENT_REGISTERED,
        EVENT_ANCHOR_APPENDED,
        EVENT_RECALIBRATED,
        EVENT_DEPRECATED,
        EVENT_DOMAIN_REFINED,
    }
)


class CanonicalEquationsRegistryCorruptError(RuntimeError):
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
    p = lock_path or CANONICAL_EQUATIONS_REGISTRY_LOCK
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


def load_registry_events_lenient(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all registry events; skip malformed lines silently (read-only callers)."""
    p = path or CANONICAL_EQUATIONS_REGISTRY_PATH
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


def load_equation_registry_strict(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Strict load for mutating callers; raises on corrupt state (Catalog #138)."""
    p = path or CANONICAL_EQUATIONS_REGISTRY_PATH
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise CanonicalEquationsRegistryCorruptError(
            f"canonical equations registry at {p} could not be read: {exc}"
        ) from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError as exc:
            raise CanonicalEquationsRegistryCorruptError(
                f"canonical equations registry at {p} line {lineno}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(r, dict):
            raise CanonicalEquationsRegistryCorruptError(
                f"canonical equations registry at {p} line {lineno}: non-dict root "
                f"(type={type(r).__name__})"
            )
        rows.append(r)
    return rows


def _validate_event_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CANONICAL_EQUATION_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CANONICAL_EQUATION_SCHEMA_VERSION!r}"
        )
    if record.get("event_type") not in VALID_EVENT_TYPES:
        raise ValueError(
            f"event_type must be one of {sorted(VALID_EVENT_TYPES)!r}"
        )
    eq_id = record.get("equation_id")
    if not isinstance(eq_id, str) or not eq_id.strip():
        raise ValueError("equation_id must be a non-empty string")
    if not isinstance(record.get("equation_payload"), dict):
        raise ValueError("equation_payload must be a dict")


def _save_ledger(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    """Atomic write under lock — tmp + fsync + os.replace."""
    if not _registry_lock_held():
        raise RuntimeError(
            "_save_ledger called WITHOUT holding _registry_lock; this is a "
            "CONCURRENCY BUG per Catalog #140 (state writers own their lock end-to-end)."
        )
    p = path or CANONICAL_EQUATIONS_REGISTRY_PATH
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
    equation: CanonicalEquation,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Append a single registry event under fcntl lock."""
    record = {
        "schema_version": CANONICAL_EQUATION_SCHEMA_VERSION,
        "event_type": event_type,
        "equation_id": equation.equation_id,
        "equation_payload": equation.to_dict(),
        "written_at_utc": _utc_now_iso(),
        "written_pid": os.getpid(),
        "written_host": socket.gethostname(),
        "agent": agent or "claude",
        "subagent_id": subagent_id,
        "notes": notes,
    }
    _validate_event_record(record)
    p_path = path or CANONICAL_EQUATIONS_REGISTRY_PATH
    l_path = lock_path or CANONICAL_EQUATIONS_REGISTRY_LOCK

    with _registry_lock(l_path):
        try:
            existing = load_equation_registry_strict(p_path)
        except CanonicalEquationsRegistryCorruptError:
            _quarantine_corrupt_file(p_path)
            existing = []
        existing.append(record)
        _save_ledger(existing, p_path)
    return record


def register_canonical_equation(
    equation: CanonicalEquation,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> CanonicalEquation:
    """Append a 'registered' event; returns the equation echo."""
    if not isinstance(equation, CanonicalEquation):
        raise InvalidEquationError(
            f"register_canonical_equation expected CanonicalEquation, got {type(equation).__name__}"
        )
    _append_event_locked(
        EVENT_REGISTERED,
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=notes,
    )
    return equation


def update_equation_with_empirical_anchor(
    equation_id: str,
    anchor: EmpiricalAnchor,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> CanonicalEquation:
    """Append a new EmpiricalAnchor to the equation; emits 'anchor_appended' event.

    Loads latest payload for ``equation_id``, calls ``with_new_anchor``,
    persists the updated equation as a new event. Per CLAUDE.md
    "Locked writes preserve deletions" (Catalog #132): APPEND-ONLY — the
    prior payload is preserved verbatim; only a NEW event row is added.
    """
    if not isinstance(anchor, EmpiricalAnchor):
        raise InvalidEquationError(
            f"update_equation_with_empirical_anchor expected EmpiricalAnchor, "
            f"got {type(anchor).__name__}"
        )
    p_path = path or CANONICAL_EQUATIONS_REGISTRY_PATH
    l_path = lock_path or CANONICAL_EQUATIONS_REGISTRY_LOCK

    with _registry_lock(l_path):
        try:
            existing = load_equation_registry_strict(p_path)
        except CanonicalEquationsRegistryCorruptError:
            _quarantine_corrupt_file(p_path)
            existing = []
        latest_payload = None
        for row in existing:
            if row.get("equation_id") == equation_id:
                latest_payload = row.get("equation_payload")
        if latest_payload is None:
            raise InvalidEquationError(
                f"equation_id={equation_id!r} not found in registry; "
                "call register_canonical_equation first"
            )
        equation = _equation_from_dict(latest_payload)
        updated = equation.with_new_anchor(anchor)
        _append_event_locked(
            EVENT_ANCHOR_APPENDED,
            updated,
            path=p_path,
            lock_path=l_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=notes,
        )
    return updated


def update_equation_with_domain_refinement(
    equation_id: str,
    *,
    domain_of_validity_extension: Mapping[str, Any],
    rationale: str,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
    notes: str | None = None,
) -> CanonicalEquation:
    """Append a ``domain_refined`` event narrowing the equation's domain_of_validity.

    Per WAVE-3-CANONICAL-EQUATION-26-DOMAIN-REFINEMENT 2026-05-20: the
    sister DWT-DETAIL-SUBBAND CPU smoke (commit ``f25f8cc1b``; KL=1.638
    nats / 3.28σ) empirically vindicated the T3 DWT BIND symposium
    Assumption-Adversary verdict #1, proving direct procedural-codebook
    substitution on DWT detail subbands corrupts inverse DWT. Equation
    ``procedural_codebook_from_seed_compression_savings_v1`` remains valid
    for INTERMEDIATE-TRANSFORM contexts (NSCS06 v8 chroma LUT / DP1 OOD-
    derived basis / similar quantizer paths) but MUST exclude direct
    detail-subband byte substitution. This helper codifies the lesson
    permanently in the canonical surface.

    The ``domain_of_validity_extension`` is merged into the equation's
    existing ``domain_of_validity`` mapping. The two CANONICAL refinement
    fields are:

      * ``domain_of_validity_included`` (tuple[str, ...]) — contexts the
        equation is valid for.
      * ``domain_of_validity_excluded`` (tuple[str, ...]) — contexts the
        equation is EXPLICITLY invalid for. Callers in these contexts
        MUST be refused via :class:`DomainOfValidityViolation`.

    The ``rationale`` MUST be a substantive non-placeholder string (>= 4
    chars; not ``<rationale>`` / ``<reason>`` literal) per Catalog #287
    sister discipline so the helper's own docstring example cannot
    self-waive.

    Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: the prior
    registered/anchor_appended rows are preserved verbatim; only a NEW
    ``domain_refined`` event row is added. Sister of
    :func:`update_equation_with_empirical_anchor` (same append-only
    pattern at the anchor surface; this helper is the domain-of-validity
    surface).

    Returns the updated CanonicalEquation (frozen-safe; original is not
    mutated).
    """
    from dataclasses import replace

    if not isinstance(domain_of_validity_extension, Mapping):
        raise InvalidEquationError(
            "domain_of_validity_extension must be a Mapping"
        )
    if not isinstance(rationale, str):
        raise InvalidEquationError("rationale must be a string")
    cleaned = rationale.strip()
    if len(cleaned) < 4:
        raise InvalidEquationError(
            f"rationale must be a substantive non-placeholder string (>= 4 chars); "
            f"got {rationale!r}"
        )
    placeholder_tokens = ("<rationale>", "<reason>", "<rationale_here>", "<reason_here>")
    if cleaned in placeholder_tokens:
        raise InvalidEquationError(
            f"rationale {cleaned!r} is a placeholder literal; provide substantive "
            "rationale per Catalog #287 sister discipline"
        )

    p_path = path or CANONICAL_EQUATIONS_REGISTRY_PATH
    l_path = lock_path or CANONICAL_EQUATIONS_REGISTRY_LOCK

    with _registry_lock(l_path):
        try:
            existing = load_equation_registry_strict(p_path)
        except CanonicalEquationsRegistryCorruptError:
            _quarantine_corrupt_file(p_path)
            existing = []
        latest_payload = None
        for row in existing:
            if row.get("equation_id") == equation_id:
                latest_payload = row.get("equation_payload")
        if latest_payload is None:
            raise InvalidEquationError(
                f"equation_id={equation_id!r} not found in registry; "
                "call register_canonical_equation first"
            )
        equation = _equation_from_dict(latest_payload)
        merged_domain: dict[str, Any] = dict(equation.domain_of_validity)
        merged_domain.update(domain_of_validity_extension)
        merged_domain["last_domain_refinement_utc"] = _utc_now_iso()
        merged_domain["last_domain_refinement_rationale"] = cleaned
        updated = replace(equation, domain_of_validity=merged_domain)
        _append_event_locked(
            EVENT_DOMAIN_REFINED,
            updated,
            path=p_path,
            lock_path=l_path,
            agent=agent,
            subagent_id=subagent_id,
            notes=notes or f"domain_refined: {cleaned[:120]}",
        )
    return updated


def _equation_from_dict(payload: Mapping[str, Any]) -> CanonicalEquation:
    """Reconstruct a CanonicalEquation from a serialized payload dict."""
    from tac.provenance.contract import (
        InvalidProvenanceError,
        Provenance,
        ProvenanceEvidenceGrade,
        ProvenanceKind,
    )

    def _prov_from_dict(d: Mapping[str, Any]) -> Provenance:
        kind = ProvenanceKind(d["artifact_kind"])
        grade = ProvenanceEvidenceGrade(d["evidence_grade"])
        # Per Provenance __post_init__ invariants, CONTEST_ARCHIVE_MEMBER
        # requires both contest_archive_zip_path AND contest_archive_member_name;
        # AGGREGATE_OF_PROVENANCES requires composed_from. The previous
        # implementation dropped these fields silently on registry round-trip,
        # which broke paired-archive-member anchor appends for any equation
        # carrying a contest-CUDA or contest-CPU empirical anchor (the
        # CUDA-first-then-CPU paired-append path hit InvalidProvenanceError
        # on the round-trip read of the prior payload). Pass-through fixes
        # the bug class structurally. Empirical anchor: OVERNIGHT-P 2026-05-21
        # HFV2 sparse pair sidecar paired CUDA+CPU equation registration.
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

    anchors = tuple(
        EmpiricalAnchor(
            anchor_id=a["anchor_id"],
            measurement_utc=a["measurement_utc"],
            inputs=a["inputs"],
            predicted_output=a["predicted_output"],
            empirical_output=a["empirical_output"],
            residual=float(a["residual"]),
            source_artifact=a["source_artifact"],
            measurement_method=a["measurement_method"],
            provenance=_prov_from_dict(a["provenance"]),
        )
        for a in payload.get("empirical_anchors", [])
    )
    return CanonicalEquation(
        equation_id=payload["equation_id"],
        name=payload["name"],
        one_line_summary=payload["one_line_summary"],
        latex_form=payload["latex_form"],
        python_callable_module_path=payload["python_callable_module_path"],
        domain_of_validity=payload["domain_of_validity"],
        units_in=payload["units_in"],
        units_out=payload["units_out"],
        empirical_anchors=anchors,
        predicted_vs_empirical_residual=payload["predicted_vs_empirical_residual"],
        last_calibration_utc=payload["last_calibration_utc"],
        next_recalibration_trigger=payload["next_recalibration_trigger"],
        canonical_consumers=tuple(payload.get("canonical_consumers", [])),
        canonical_producers=tuple(payload.get("canonical_producers", [])),
        provenance=_prov_from_dict(payload["provenance"]),
        schema_version=payload.get("schema_version", CANONICAL_EQUATION_SCHEMA_VERSION),
    )


def query_equations(
    *,
    path: Path | None = None,
) -> list[CanonicalEquation]:
    """Return latest payload per equation_id as reconstructed CanonicalEquation."""
    rows = load_registry_events_lenient(path)
    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        eq_id = row.get("equation_id")
        if isinstance(eq_id, str):
            latest_by_id[eq_id] = row.get("equation_payload", {})
    out: list[CanonicalEquation] = []
    for payload in latest_by_id.values():
        try:
            out.append(_equation_from_dict(payload))
        except (KeyError, InvalidEquationError):
            # Skip historical rows that fail current invariants (forward-compat).
            continue
    return out


def query_equations_by_domain(
    domain_token: str,
    *,
    path: Path | None = None,
) -> list[CanonicalEquation]:
    """Return equations whose domain_of_validity mentions the token."""
    token = (domain_token or "").lower()
    if not token:
        return []
    out = []
    for eq in query_equations(path=path):
        as_str = json.dumps(dict(eq.domain_of_validity)).lower()
        if token in as_str:
            out.append(eq)
    return out


def query_equations_by_consumer(
    consumer_module_path: str,
    *,
    path: Path | None = None,
) -> list[CanonicalEquation]:
    """Return equations declaring this dotted module path as a consumer."""
    if not consumer_module_path:
        return []
    out = []
    for eq in query_equations(path=path):
        for c in eq.canonical_consumers:
            if consumer_module_path in c or c in consumer_module_path:
                out.append(eq)
                break
    return out


def query_equations_by_producer(
    producer_module_path: str,
    *,
    path: Path | None = None,
) -> list[CanonicalEquation]:
    """Return equations declaring this dotted module path as a producer."""
    if not producer_module_path:
        return []
    out = []
    for eq in query_equations(path=path):
        for p in eq.canonical_producers:
            if producer_module_path in p or p in producer_module_path:
                out.append(eq)
                break
    return out


def get_equation_by_id(
    equation_id: str,
    *,
    path: Path | None = None,
) -> CanonicalEquation | None:
    """Lookup an equation by id; returns None if absent."""
    for eq in query_equations(path=path):
        if eq.equation_id == equation_id:
            return eq
    return None


@dataclass(frozen=True)
class RecalibrationReport:
    """Result of auto_recalibrate_from_continual_learning_posterior.

    Reports which equations had residuals refreshed + how many new anchors
    were absorbed. Operator-facing summary so the calibration drift is
    visible in CLI output.
    """

    equations_checked: int
    equations_recalibrated: int
    new_anchors_absorbed: int
    per_equation_summary: dict[str, dict[str, Any]] = field(default_factory=dict)


def auto_recalibrate_from_continual_learning_posterior(
    equation_id: str | None = None,
    *,
    path: Path | None = None,
) -> RecalibrationReport:
    """Stub for periodic auto-recalibration.

    The full continual-learning posterior consumption is implemented as a
    follow-on cathedral consumer (see ``canonical_equation_lookup_consumer``).
    This helper exists so the operator-facing CLI
    ``tools/recalibrate_equation.py`` has a callable entry point; it
    iterates current equations + emits a no-op report when no new anchors
    are present (the canonical "the system already knows" path).

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": no
    automatic anchor synthesis here — actual recalibration requires an
    explicit ``update_equation_with_empirical_anchor`` call backed by a
    measured artifact path.
    """
    equations = query_equations(path=path)
    if equation_id is not None:
        equations = [e for e in equations if e.equation_id == equation_id]
    summary: dict[str, dict[str, Any]] = {}
    for eq in equations:
        summary[eq.equation_id] = {
            "anchor_count": len(eq.empirical_anchors),
            "current_residuals": dict(eq.predicted_vs_empirical_residual),
            "well_calibrated": eq.is_well_calibrated,
            "trigger": eq.next_recalibration_trigger,
        }
    return RecalibrationReport(
        equations_checked=len(equations),
        equations_recalibrated=0,  # stub; auto-refit comes in a follow-on landing
        new_anchors_absorbed=0,
        per_equation_summary=summary,
    )


__all__ = [
    "CANONICAL_EQUATIONS_REGISTRY_PATH",
    "CANONICAL_EQUATIONS_REGISTRY_LOCK",
    "LOCK_TIMEOUT_SECONDS",
    "EVENT_REGISTERED",
    "EVENT_ANCHOR_APPENDED",
    "EVENT_RECALIBRATED",
    "EVENT_DEPRECATED",
    "EVENT_DOMAIN_REFINED",
    "VALID_EVENT_TYPES",
    "CanonicalEquationsRegistryCorruptError",
    "RecalibrationReport",
    "load_registry_events_lenient",
    "load_equation_registry_strict",
    "register_canonical_equation",
    "update_equation_with_empirical_anchor",
    "update_equation_with_domain_refinement",
    "query_equations",
    "query_equations_by_domain",
    "query_equations_by_consumer",
    "query_equations_by_producer",
    "get_equation_by_id",
    "auto_recalibrate_from_continual_learning_posterior",
]
