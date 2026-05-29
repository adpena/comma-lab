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
    RECALIBRATE_ON_NEW_ANCHORS,
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


# Canonical trigger token (mirrors equation.RECALIBRATE_ON_NEW_ANCHORS) at which
# the 3+-anchor auto-refit fires. Imported lazily inside the function to avoid a
# top-level cycle, but pinned here so the threshold is auditable in one place.
_AUTO_REFIT_MIN_ANCHORS = 3


def _anchor_in_domain_context(anchor: EmpiricalAnchor) -> str | None:
    """Return the canonical ``in_domain_context`` token attached to an anchor.

    Anchors typically declare ``inputs.in_domain_context`` (string) per the
    canonical anchor schema. This helper centralizes lookup so the recalibrator
    + sister gate Catalog #359 (residual-hybrid misapplication) read the same
    field consistently. Returns ``None`` when absent / non-string.
    """
    in_domain = anchor.inputs.get("in_domain_context") if isinstance(anchor.inputs, Mapping) else None
    if isinstance(in_domain, str) and in_domain.strip():
        return in_domain.strip()
    return None


def _anchor_is_in_excluded_context(
    anchor: EmpiricalAnchor,
    excluded_contexts: tuple[str, ...] | list[str] | None,
) -> bool:
    """Return True when the anchor's ``in_domain_context`` matches an excluded one.

    Per Slot A NEGATIVE-RESULTS-AUDIT-V2 finding F05 + Catalog #359 sister
    discipline: an anchor whose ``inputs.in_domain_context`` is enumerated in
    the equation's ``domain_of_validity.excluded_contexts`` was landed for a
    context the equation EXPLICITLY DOES NOT PREDICT FOR. The anchor remains
    in the registry per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE
    (the empirical measurement IS real evidence), but it MUST NOT pollute the
    equation's residual SUMMARY because the residual would be paid against a
    prediction the equation never claimed.

    Empirical anchor: ``pose_axis_score_direction_matching_paradigm_savings_v1``
    landed an anchor with ``in_domain_context=
    'mlx_native_pose_axis_score_direction_matching_standalone_replaces_segnet'``
    whose residual=30.68 — this context is in ``excluded_contexts`` because
    standalone replacement (no SegNet) is a separate equation. Letting the
    30.68 residual into the summary makes the equation look ~10x worse
    calibrated than it actually is in its declared domain.
    """
    if not excluded_contexts:
        return False
    in_domain = _anchor_in_domain_context(anchor)
    if in_domain is None:
        return False
    return in_domain in tuple(excluded_contexts)


def _anchor_residual_is_nan_or_infinite(anchor: EmpiricalAnchor) -> bool:
    """Return True when the anchor's stored residual is NaN / infinite.

    Per Slot A NEGATIVE-RESULTS-AUDIT-V2 finding F08 + F09: 2 equations carry
    pre-rejection NaN-sentinel anchors (``EmpiricalAnchor.__post_init__`` now
    refuses NaN at construction per ``equation.py`` lines 146-147, but a
    handful of historical payloads predated that invariant). The lenient
    registry loader cannot detect this because the legacy NaN residuals were
    embedded inside the residual SUMMARY map (``predicted_vs_empirical_residual``)
    rather than inside an EmpiricalAnchor itself, but defense-in-depth: if any
    future loader path slips a NaN anchor through, the refit MUST skip it
    rather than propagating NaN into the summary (NaN poisons every downstream
    consumer: ``is_well_calibrated`` returns False forever; cathedral autopilot
    Catalog #335 lookup consumer sees a None-equivalent; arithmetic comparisons
    silently fail because ``NaN != NaN``).
    """
    r = float(anchor.residual)
    return r != r or r == float("inf") or r == float("-inf")


def _refit_residual_map_from_anchors(
    equation: CanonicalEquation,
) -> dict[str, float]:
    """Recompute the per-axis residual map directly from the equation's anchors.

    Evidence-faithful refit: each landed :class:`EmpiricalAnchor` carries a
    ``measurement_method`` axis token + a normalized ``residual`` magnitude
    (NaN already refused at construction per ``EmpiricalAnchor.__post_init__``).
    The latest anchor per axis wins (mirrors ``with_new_anchor`` semantics,
    which overwrite ``predicted_vs_empirical_residual[method]`` on append).

    Two evidence-faithful filters applied per Slot A NEGATIVE-RESULTS-AUDIT-V2
    FIX O1 + O2 (2026-05-28):

    1. **Excluded-context filter (FIX O1)** — anchors whose
       ``inputs.in_domain_context`` matches the equation's
       ``domain_of_validity.excluded_contexts`` are SKIPPED. The anchor row
       itself remains in the registry per Catalog #110/#113 APPEND-ONLY (it IS
       real empirical evidence for an OUT-OF-DOMAIN context); only the residual
       SUMMARY excludes it. Sister of Catalog #359
       (``check_no_canonical_equation_misapplication_to_residual_hybrid_contexts``)
       at the per-anchor recalibration surface.
    2. **NaN-sentinel skip (FIX O2)** — anchors whose stored residual is NaN
       or infinite are SKIPPED. Defense-in-depth: ``EmpiricalAnchor.__post_init__``
       already refuses NaN at construction, but defensive coding here protects
       against future loader-path regressions that might slip a NaN through.

    This does NOT synthesize new anchors and does NOT fabricate a measurement
    (Catalog #287 / #323): it only re-derives the residual *summary* from
    anchors that already landed via signed ``update_equation_with_empirical_anchor``
    calls. The stale-prior bug class (an equation whose stored residual map no
    longer matches its own anchors because a sister appended anchors but never
    re-summarized) is what this extincts.
    """
    excluded_contexts = equation.domain_of_validity.get("excluded_contexts") if isinstance(equation.domain_of_validity, Mapping) else None
    refit: dict[str, float] = {}
    for anchor in equation.empirical_anchors:
        if _anchor_is_in_excluded_context(anchor, excluded_contexts):
            # FIX O1: anchor measured in EXCLUDED context; the equation does
            # not predict for this context, so its residual must not pollute
            # the in-domain summary. Anchor row preserved per APPEND-ONLY.
            continue
        if _anchor_residual_is_nan_or_infinite(anchor):
            # FIX O2: NaN/inf residual would poison every downstream consumer
            # (is_well_calibrated; cathedral autopilot lookup; arithmetic).
            # Defense-in-depth: EmpiricalAnchor.__post_init__ also refuses NaN
            # at construction (equation.py L146-147).
            continue
        # Latest-wins per axis (anchors are stored in append order).
        refit[anchor.measurement_method] = float(anchor.residual)
    return refit


def _stored_map_has_corrupt_residual_keys(stored_map: Mapping[str, float]) -> bool:
    """Return True when the stored residual map carries NaN/inf residuals.

    Per Slot A NEGATIVE-RESULTS-AUDIT-V2 FIX O2: a handful of historical
    equation payloads predate the EmpiricalAnchor NaN-rejection invariant
    and now have NaN entries stuck in their residual SUMMARY map even though
    no current anchor would justify a NaN. The recalibrator's primary
    eligibility path (``RECALIBRATE_ON_NEW_ANCHORS`` + 3+ anchors) may not
    cover these (1 or 2 anchors only); this helper enables the
    NaN-cleanup-eligibility secondary path to fire whenever the stored map
    contains a residual that the canonical EmpiricalAnchor invariants now
    refuse at construction.
    """
    for v in stored_map.values():
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if fv != fv or fv == float("inf") or fv == float("-inf"):
            return True
    return False


def auto_recalibrate_from_continual_learning_posterior(
    equation_id: str | None = None,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> RecalibrationReport:
    """Periodic auto-recalibration of canonical-equation residual summaries.

    For every equation whose ``next_recalibration_trigger`` is
    ``when_3+_new_empirical_anchors_in_domain`` AND whose anchor count is
    >= ``_AUTO_REFIT_MIN_ANCHORS`` (3), this helper re-derives
    ``predicted_vs_empirical_residual`` directly from the equation's own
    landed :class:`EmpiricalAnchor` rows and, if that recomputed map differs
    from the stored map, appends a canonical ``recalibrated`` event (bumping
    ``last_calibration_utc``). The original payload is preserved verbatim per
    Catalog #110/#113 APPEND-ONLY; only a NEW event row is written.

    Why this is NOT a Catalog #287 violation: no anchor is synthesized and no
    empirical_output is fabricated. The refit only re-SUMMARIZES residuals
    from anchors that already landed via signed
    ``update_equation_with_empirical_anchor`` calls. The bug class this
    extincts is the stale-prior orphan: an equation accumulates 3+ disagreeing
    anchors (e.g. ``hinton_kl_distill_enables_qat_catalyst_composition_savings_v1``
    whose anchors falsify the closed-form alpha=0.15 lift toward an empirical
    ~0) yet its stored residual summary + ``last_calibration_utc`` never move
    because the previous recalibrator no-op'd even when its own trigger
    condition was already satisfied.

    Equations on other triggers (``when_residual_drift_exceeds_2x`` /
    ``when_operator_invokes_recalibrate_equation`` / ``never_auto_operator_only``)
    are reported but NOT auto-refit here — those require an operator invocation
    or a drift-detection path, by design.

    **Slot A NEGATIVE-RESULTS-AUDIT-V2 FIX O1 + O2 (2026-05-28)** ext:
    every refit invocation now passes anchors through two evidence-faithful
    filters in :func:`_refit_residual_map_from_anchors`: (FIX O1) skip
    anchors whose ``inputs.in_domain_context`` matches the equation's
    ``domain_of_validity.excluded_contexts``; (FIX O2) skip anchors with
    NaN/inf residuals. Additionally, equations with NaN/inf in their stored
    residual map (legacy payloads that predate the construction invariant)
    qualify for refit even when below the 3-anchor threshold, via the
    secondary NaN-cleanup-eligibility path.

    Returns a :class:`RecalibrationReport`. ``new_anchors_absorbed`` counts
    the (axis, residual) summary rows that were updated/added by the refit
    (the system "absorbing" already-landed evidence into the summary), NOT
    new measurements.
    """
    from dataclasses import replace

    equations = query_equations(path=path)
    if equation_id is not None:
        equations = [e for e in equations if e.equation_id == equation_id]

    summary: dict[str, dict[str, Any]] = {}
    recalibrated_count = 0
    absorbed_count = 0

    for eq in equations:
        trigger = eq.next_recalibration_trigger
        anchor_count = len(eq.empirical_anchors)
        # Primary refit eligibility: canonical RECALIBRATE_ON_NEW_ANCHORS trigger
        # + 3+ anchors. Mirrors original intent.
        refit_eligible = (
            trigger == RECALIBRATE_ON_NEW_ANCHORS
            and anchor_count >= _AUTO_REFIT_MIN_ANCHORS
        )
        # Secondary NaN-cleanup eligibility per FIX O2: even when below the
        # 3-anchor threshold, if the stored residual map contains NaN/inf
        # residuals that the current EmpiricalAnchor invariants now refuse
        # at construction, the refit MUST run to drop those poison values
        # from the summary. The refit IS still evidence-faithful (driven from
        # the equation's own anchors only); it just additionally fires when
        # the stored map carries values no anchor would now justify.
        stored_map = dict(eq.predicted_vs_empirical_residual)
        nan_cleanup_eligible = (
            trigger == RECALIBRATE_ON_NEW_ANCHORS
            and not refit_eligible
            and anchor_count >= 1
            and _stored_map_has_corrupt_residual_keys(stored_map)
        )
        do_refit = refit_eligible or nan_cleanup_eligible
        refit_map = _refit_residual_map_from_anchors(eq) if do_refit else {}
        # Drift detection: did the recomputed summary diverge from the stored
        # summary? (New axis keys, changed residual magnitudes, or stale keys
        # whose anchors were superseded.)
        changed = do_refit and refit_map != stored_map
        did_recalibrate = False

        if changed:
            updated = replace(
                eq,
                predicted_vs_empirical_residual=refit_map,
                last_calibration_utc=_utc_now_iso(),
            )
            n_axes_changed = sum(
                1
                for k, v in refit_map.items()
                if stored_map.get(k) != v
            ) + sum(1 for k in stored_map if k not in refit_map)
            cleanup_reason = (
                "stale-prior orphan extinction"
                if refit_eligible
                else "nan-cleanup eligibility (FIX O2 secondary path)"
            )
            _append_event_locked(
                EVENT_RECALIBRATED,
                updated,
                path=path,
                lock_path=lock_path,
                agent=agent or "claude",
                subagent_id=subagent_id,
                notes=(
                    f"auto_recalibrate: refit residual summary from "
                    f"{anchor_count} landed anchors (excluded_contexts + NaN "
                    f"filters per Slot A FIX O1+O2 2026-05-28); "
                    f"{n_axes_changed} axis row(s) changed ({cleanup_reason})"
                ),
            )
            recalibrated_count += 1
            absorbed_count += n_axes_changed
            did_recalibrate = True
            eq = updated  # report the post-refit state

        summary[eq.equation_id] = {
            "anchor_count": anchor_count,
            "current_residuals": dict(eq.predicted_vs_empirical_residual),
            "well_calibrated": eq.is_well_calibrated,
            "trigger": trigger,
            "refit_eligible": refit_eligible,
            "nan_cleanup_eligible": nan_cleanup_eligible,
            "recalibrated": did_recalibrate,
        }

    return RecalibrationReport(
        equations_checked=len(equations),
        equations_recalibrated=recalibrated_count,
        new_anchors_absorbed=absorbed_count,
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
