"""Curriculum-candidate POOL — the P0 tracked costate class (task #403).

Per the CLAUDE.md NON-NEGOTIABLE "'Off' is a tracked queue, never a forgotten default" +
the OPERATOR-PRIORITY P0 binding 2026-07-10 (memory
``curriculum_candidate_pool_p0_orphan_class_20260710`` + design memo
``.omx/research/curriculum_candidate_pool_p0_20260710.md``): a CURRICULUM candidate in ANY form
(stages / losses / init / preconditioning / data-order / averaging / solve-interleave /
state-evolution) that is designed-or-built-but-never-fired is the SAME orphaned-signal class the
lever activation ledger tracks — but a curriculum candidate is often a TOOL, a stage, a data-order,
or a whole vehicle DOF that is NOT a single ``--flag`` DSL ``Lever``, so the lever-scoped
``activation_ledger`` event vocabulary cannot hold it without overloading its schema (the #400
agent's finding). This module is the SIBLING store that holds those candidates WITHOUT that overload.

A chat-only inventory of built-never-fired curriculum candidates IS the orphan bug — so the inventory
lives here as a durable, ranked, controller-held queue that ``tools/costate_digest.py`` §curriculum-pool
reads. The controller remembers and surfaces the next-fireable rows; the operator never has to.

Store: canonical, APPEND-ONLY, fcntl-locked JSONL at ``.omx/state/curriculum_candidate_pool.jsonl``
(mirrors the ``.omx/state/*.jsonl`` canonical stores; latest-row-wins per ``candidate`` key on read).

NO-FAKE discipline (per CLAUDE.md supreme rule): every row carries a ``source_anchor`` (the commit /
memo / DAG-FEED / task-# that BUILT or DESIGNED it) and its HONEST ``status`` — a designed-but-unbuilt
candidate is ``needs-build`` (NOT ``measured``); a built-but-never-launched candidate is
``built-never-fired`` (NOT ``measured``). Production ``measured`` requires a byte-closed verdict plus
revalidated durable receipt bytes and exact SHA custody; research-only compute findings remain
explicitly non-promotable. This store MEASURES nothing and mints no law (equations leg
N/A-with-rationale) — it CITES existing registered equations. It moves the pointer by ZERO: it is
apparatus (means), not a score.

State (``status``) machine, latest-row-wins per candidate:
  needs-build / built-never-fired / reformulation-queue / armed  --measured-->  measured
  (and --retired--> retired-with-reason from any state; terminal, dormant-with-reactivation).

The #247 costate SENSE layer consumes :func:`duty_to_measure_pool` to rank production-fireable
curriculum candidates into its DECIDE queue beside the lever duty-to-measure line. Research-only
rows stay visible through ``pool_summary()['research_signals']`` and never enter DECIDE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tac.jsonl_store import append_locked_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PATH = _REPO_ROOT / ".omx" / "state" / "curriculum_candidate_pool.jsonl"

# ── canonical status vocabulary (latest-row-wins per candidate) ──────────────────────────────────
STATUS_ARMED = "armed"  # already flying in the sealed launch config
STATUS_BUILT_NEVER_FIRED = "built-never-fired"  # code exists (default-off byte-identical), never launched
STATUS_NEEDS_BUILD = "needs-build"  # design/task exists, no code on THIS vehicle
STATUS_REFORMULATION_QUEUE = "reformulation-queue"  # prior formulation measured NO-GO; named reformulation owed
STATUS_MEASURED = "measured"  # typed byte-closed or research-diagnostic evidence
STATUS_RETIRED = "retired-with-reason"  # retired with a recorded MEASURED law (dormant-w-reactivation)
VALID_STATUSES = frozenset(
    {
        STATUS_ARMED,
        STATUS_BUILT_NEVER_FIRED,
        STATUS_NEEDS_BUILD,
        STATUS_REFORMULATION_QUEUE,
        STATUS_MEASURED,
        STATUS_RETIRED,
    }
)

# ── canonical form-class vocabulary (the "ANY form" the operator P0 binding enumerated) ──────────
VALID_FORM_CLASSES = frozenset(
    {
        "loss-geometry",
        "optimizer-stage",
        "regularizer-schedule",
        "data-curriculum",
        "architecture-growth",
        "averaging",
        "preconditioning",
        "discrete-solve-interleave",
        "init-warm-start",
        "state-evolution",
        "pose-carrier",
    }
)

VALID_AXES = frozenset({"d_seg", "d_pose", "rate"})
EVIDENCE_BYTE_CLOSED = "byte_closed"
EVIDENCE_RESEARCH_DIAGNOSTIC = "research_diagnostic"
VALID_EVIDENCE_KINDS = frozenset({EVIDENCE_BYTE_CLOSED, EVIDENCE_RESEARCH_DIAGNOSTIC})

# Production ``measured`` is deliberately narrower than "a file with a matching hash".  This
# normalized wrapper is the only receipt type this pool currently knows how to adjudicate.  New
# receipt families must add an explicit validator here; there is no permissive fallback.
PRODUCTION_RECEIPT_SCHEMA = "curriculum_candidate_production_admission.v1"
PRODUCTION_RECEIPT_TYPE = "curriculum_candidate_production_admission"

# ranking order: the duty queue the controller drains (built-never-fired highest readiness) first.
_STATUS_ORDER = {
    STATUS_BUILT_NEVER_FIRED: 0,
    STATUS_NEEDS_BUILD: 1,
    STATUS_REFORMULATION_QUEUE: 2,
    STATUS_ARMED: 3,
    STATUS_MEASURED: 4,
    STATUS_RETIRED: 5,
}
# the statuses that are OWED a measurement (the controller's duty-to-measure queue for curricula).
_DUTY_STATUSES = frozenset({STATUS_BUILT_NEVER_FIRED, STATUS_NEEDS_BUILD, STATUS_REFORMULATION_QUEUE})


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_nonempty_string(value: object) -> bool:
    return value is None or _nonempty_string(value)


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _production_admission_receipt_error(payload: object, *, candidate: str) -> str | None:
    """Validate the one explicitly supported production-admission receipt schema."""

    if not isinstance(payload, dict):
        return "production receipt must contain a JSON object"
    if payload.get("receipt_type") != PRODUCTION_RECEIPT_TYPE:
        return f"receipt_type must equal {PRODUCTION_RECEIPT_TYPE!r}"
    if payload.get("candidate") != candidate:
        return f"receipt candidate must equal {candidate!r}"

    verdict = payload.get("verdict")
    if not isinstance(verdict, dict):
        return "verdict must be an object"
    if verdict.get("status") != "ADMITTED" or verdict.get("passed") is not True:
        return "verdict must record status='ADMITTED' and passed=true"
    if verdict.get("byte_closed") is not True:
        return "verdict.byte_closed must be true"

    authority = payload.get("authority")
    if not isinstance(authority, dict):
        return "authority must be an object"
    if authority.get("outcome") != "ACCEPTED":
        return "authority.outcome must equal 'ACCEPTED'"
    if authority.get("research_only") is not False:
        return "authority.research_only must be false"
    if authority.get("promotion_eligible") is not True:
        return "authority.promotion_eligible must be true"

    custody = payload.get("custody")
    if not isinstance(custody, dict):
        return "custody must be an object"
    if custody.get("outcome") != "VALID":
        return "custody.outcome must equal 'VALID'"
    archive_sha256 = custody.get("archive_sha256")
    runtime_tree_sha256 = custody.get("runtime_tree_sha256")
    upstream_snapshot_sha256 = custody.get("upstream_snapshot_sha256")
    if not _valid_sha256(archive_sha256):
        return "custody.archive_sha256 must be lowercase SHA-256 hex"
    if not _valid_sha256(runtime_tree_sha256):
        return "custody.runtime_tree_sha256 must be lowercase SHA-256 hex"
    if not _valid_sha256(upstream_snapshot_sha256):
        return "custody.upstream_snapshot_sha256 must be lowercase SHA-256 hex"
    archive_bytes = custody.get("archive_bytes")
    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0:
        return "custody.archive_bytes must be a positive integer"

    measurement = payload.get("measurement")
    if not isinstance(measurement, dict):
        return "measurement must be an object"
    if measurement.get("n_samples") != 600:
        return "measurement.n_samples must equal 600"
    d_seg = measurement.get("d_seg")
    d_pose = measurement.get("d_pose")
    score = measurement.get("score")
    for name, value in (("d_seg", d_seg), ("d_pose", d_pose), ("score", score)):
        if not _finite_number(value) or float(value) < 0.0:
            return f"measurement.{name} must be a finite non-negative number"

    # Route axis/hardware acceptance through the existing Catalog #127 canonical validator instead
    # of duplicating its allow-list here.
    from tac.continual_learning import ContestResult

    custody_verdict = ContestResult(
        axis=authority.get("axis", ""),
        hardware_substrate=authority.get("hardware_substrate", ""),
        architecture_class=f"curriculum_candidate:{candidate}",
        score_value=float(score),
        evidence_tag=authority.get("evidence_tag", ""),
        archive_sha256=str(archive_sha256),
        archive_bytes=archive_bytes,
    ).validate_custody_verdict()
    if not custody_verdict.accepted:
        return f"canonical authority custody refused: {custody_verdict.reason}"

    from tac.auth_eval_schema import contest_formula_score

    recomputed = contest_formula_score(
        seg_dist=float(d_seg),
        pose_dist=float(d_pose),
        archive_bytes=archive_bytes,
    )
    if not math.isclose(float(score), recomputed, rel_tol=0.0, abs_tol=1e-9):
        return f"measurement.score mismatch: receipt {float(score)}, recomputed {recomputed}"
    return None


_PRODUCTION_RECEIPT_VALIDATORS = {
    PRODUCTION_RECEIPT_SCHEMA: _production_admission_receipt_error,
}

# Code-reviewed trust root.  A caller cannot mint authority by writing a syntactically valid JSON
# object and hashing it: candidate, path, and exact receipt bytes must first be pinned here.  Keep the
# default fail-closed; tests may monkeypatch a scoped entry to exercise a supported validator.
_TRUSTED_PRODUCTION_RECEIPTS: dict[tuple[str, str], str] = {}


def _production_receipt_semantic_error(receipt_bytes: bytes, *, candidate: str) -> str | None:
    try:
        payload = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return f"production receipt is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return "production receipt must contain a JSON object"
    schema = payload.get("schema")
    validator = _PRODUCTION_RECEIPT_VALIDATORS.get(schema)
    if validator is None:
        return f"unsupported production receipt schema {schema!r}"
    return validator(payload, candidate=candidate)


def _receipt_custody_error(
    verdict_ref: object,
    trusted_sha256: object,
    *,
    production_candidate: str | None = None,
) -> str | None:
    """Return why a receipt lacks durable byte custody, or ``None`` when verified.

    Production ``measured`` authority always calls this gate. A research diagnostic calls it when
    it elects to bind a reviewed receipt hash, without gaining production authority. The receipt
    must be a relative path below :data:`_REPO_ROOT`; every existing path component must be
    non-symlink; the leaf must be a regular file; and bytes read through a no-follow descriptor must
    match the pinned SHA. Read-time revalidation makes later tampering de-authorize the stored row.
    """

    if not _nonempty_string(verdict_ref):
        return "verdict_ref is required"
    if not _valid_sha256(trusted_sha256):
        return "trusted_receipt_sha256 is required and must be lowercase SHA-256 hex"

    if production_candidate is not None:
        trusted_registry_sha = _TRUSTED_PRODUCTION_RECEIPTS.get((production_candidate, str(verdict_ref)))
        if trusted_registry_sha is None:
            return "production receipt candidate/path is not in the code-reviewed trust registry"
        if trusted_registry_sha != trusted_sha256:
            return (
                "production receipt SHA-256 does not match the code-reviewed trust registry: "
                f"expected {trusted_registry_sha}, got {trusted_sha256}"
            )

    ref = Path(verdict_ref)
    if ref.is_absolute():
        return "verdict_ref must be repo-relative"
    if not ref.parts or any(part == ".." for part in ref.parts):
        return "verdict_ref must remain below the repository root"

    try:
        root = _REPO_ROOT.resolve(strict=True)
    except OSError as exc:
        return f"repository root is unavailable: {exc}"

    current = root
    leaf_stat: os.stat_result | None = None
    for part in ref.parts:
        if part in ("", "."):
            continue
        current = current / part
        try:
            leaf_stat = current.lstat()
        except OSError as exc:
            return f"receipt path is unavailable: {exc}"
        if stat.S_ISLNK(leaf_stat.st_mode):
            return f"receipt path contains a symlink: {current.relative_to(root)}"

    if leaf_stat is None or not stat.S_ISREG(leaf_stat.st_mode):
        return "verdict_ref must name a regular file"

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(current, flags)
    except OSError as exc:
        return f"receipt cannot be opened without following symlinks: {exc}"

    digest = hashlib.sha256()
    receipt_bytes = bytearray()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            return "opened verdict_ref is not a regular file"
        if (before.st_dev, before.st_ino) != (leaf_stat.st_dev, leaf_stat.st_ino):
            return "receipt changed while custody was being established"
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
            if production_candidate is not None:
                receipt_bytes.extend(chunk)
        after = os.fstat(fd)
    except OSError as exc:
        return f"receipt bytes could not be read: {exc}"
    finally:
        os.close(fd)

    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_identity != after_identity:
        return "receipt changed while its bytes were being hashed"
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != trusted_sha256:
        return f"receipt SHA-256 mismatch: expected {trusted_sha256}, got {actual_sha256}"
    if production_candidate is not None:
        semantic_error = _production_receipt_semantic_error(bytes(receipt_bytes), candidate=production_candidate)
        if semantic_error is not None:
            return semantic_error
    return None


def _valid_stored_row(row: object) -> bool:
    """Validate one decoded JSONL row before it may participate in latest-wins overlay."""

    if not isinstance(row, dict):
        return False
    if not _nonempty_string(row.get("candidate")):
        return False
    if row.get("status") not in VALID_STATUSES:
        return False
    if row.get("form_class") not in VALID_FORM_CLASSES:
        return False
    if not _nonempty_string(row.get("source_anchor")) or not _nonempty_string(row.get("gate")):
        return False

    dsl_lever = row.get("dsl_lever")
    dsl_na_reason = row.get("dsl_na_reason")
    if not _optional_nonempty_string(dsl_lever) or not _optional_nonempty_string(dsl_na_reason):
        return False
    if _nonempty_string(dsl_lever) == _nonempty_string(dsl_na_reason):
        return False

    for key in ("justification", "slot", "owner"):
        if key in row and not isinstance(row[key], str):
            return False
    for key in ("authority_axis", "verdict_scope", "activation_status", "verdict_ref"):
        if key in row and not _optional_nonempty_string(row[key]):
            return False

    research_only = row.get("research_only", False)
    if not isinstance(research_only, bool):
        return False
    evidence_kind = row.get("evidence_kind")
    if evidence_kind is not None and evidence_kind not in VALID_EVIDENCE_KINDS:
        return False
    if row["status"] == STATUS_MEASURED:
        required_kind = EVIDENCE_RESEARCH_DIAGNOSTIC if research_only else EVIDENCE_BYTE_CLOSED
        if evidence_kind != required_kind or not _nonempty_string(row.get("verdict_ref")):
            return False
        receipt_sha = row.get("trusted_receipt_sha256")
        if (not research_only or receipt_sha is not None) and _receipt_custody_error(
            row.get("verdict_ref"),
            receipt_sha,
            production_candidate=row["candidate"] if not research_only else None,
        ):
            return False
    elif evidence_kind is not None:
        return False

    est_delta_s = row.get("est_delta_s")
    axis = row.get("axis")
    if est_delta_s is not None:
        if not _finite_number(est_delta_s) or float(est_delta_s) < 0.0 or axis not in VALID_AXES:
            return False
    elif axis is not None and axis not in VALID_AXES:
        return False
    speedup = row.get("realized_speedup_factor")
    if speedup is not None and (not _finite_number(speedup) or float(speedup) < 0.0):
        return False
    reduction = row.get("derived_cost_reduction_fraction")
    if reduction is not None and (not _finite_number(reduction) or not 0.0 <= float(reduction) <= 1.0):
        return False
    receipt_sha = row.get("trusted_receipt_sha256")
    if receipt_sha is not None and not _valid_sha256(receipt_sha):
        return False
    blockers = row.get("blockers", [])
    return not (
        blockers is None
        or isinstance(blockers, (str, bytes))
        or not isinstance(blockers, Sequence)
        or any(not _nonempty_string(item) for item in blockers)
    )


def _blocked_research_seed_for_missing_custody(row: object) -> dict | None:
    """Keep a pinned research signal visible when its receipt bytes are unavailable.

    A missing/changed research receipt must remove ``measured`` authority, but silently deleting the
    candidate recreates the orphan class this pool exists to prevent.  Only otherwise-valid,
    explicitly research-only measured seeds are downgraded; malformed or production seeds still
    disappear fail-closed.
    """

    if not isinstance(row, dict):
        return None
    if (
        row.get("status") != STATUS_MEASURED
        or row.get("research_only") is not True
        or row.get("evidence_kind") != EVIDENCE_RESEARCH_DIAGNOSTIC
        or not _nonempty_string(row.get("candidate"))
    ):
        return None
    receipt_sha = row.get("trusted_receipt_sha256")
    if receipt_sha is None:
        return None
    custody_error = _receipt_custody_error(row.get("verdict_ref"), receipt_sha)
    if custody_error is None:
        return None

    blocked = dict(row)
    blocked["status"] = STATUS_REFORMULATION_QUEUE
    blocked["evidence_kind"] = None
    blocked["gate"] = f"RECEIPT_CUSTODY_BLOCKED: {custody_error}; {row.get('gate', '')}"
    blocked["justification"] = f"UNVERIFIED_CUSTODY_RESEARCH_SIGNAL: {row.get('justification', '')}"
    blocked["activation_status"] = "RECEIPT_CUSTODY_BLOCKED_NO_PRODUCTION_AUTHORITY"
    blocked["blockers"] = [
        *row.get("blockers", ()),
        f"RECEIPT_CUSTODY_BLOCKED: {custody_error}",
    ]
    return blocked if _valid_stored_row(blocked) else None


# _append_locked_jsonl canonicalized to tac.jsonl_store.append_locked_jsonl (audit finding #4,
# .omx/research/hardcode_duplication_audit_witness_stack_20260710.md) — was byte-identical to the
# copy in activation_ledger.py; now a single shared helper.


def record_candidate(
    candidate: str,
    status: str,
    *,
    form_class: str,
    source_anchor: str,
    gate: str,
    justification: str = "",
    dsl_lever: str | None = None,
    dsl_na_reason: str | None = None,
    slot: str = "",
    owner: str = "",
    est_delta_s: float | None = None,
    axis: str | None = None,
    verdict_ref: str | None = None,
    evidence_kind: str | None = None,
    research_only: bool = False,
    authority_axis: str | None = None,
    verdict_scope: str | None = None,
    activation_status: str | None = None,
    realized_speedup_factor: float | None = None,
    derived_cost_reduction_fraction: float | None = None,
    trusted_receipt_sha256: str | None = None,
    blockers: Sequence[str] = (),
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append ONE curriculum-candidate row (APPEND-ONLY, fcntl-locked). Returns the written row.

    NO-FAKE contract, enforced here:
      * ``status`` ∈ :data:`VALID_STATUSES`; ``form_class`` ∈ :data:`VALID_FORM_CLASSES`.
      * ``source_anchor`` REQUIRED — every row cites the commit / memo / DAG-FEED / task-# that
        BUILT or DESIGNED it (never a candidate asserted from thin air).
      * EXACTLY-ONE-OF ``dsl_lever`` (the held ``curriculum_dsl`` factory name) OR ``dsl_na_reason``
        (why this candidate is NOT a single-flag DSL lever — tool-side / vehicle-level / not-yet-built).
        This is the per-row DSL-leg discipline (the triality DSL leg for the pool).
      * ``status == 'measured'`` REQUIRES a ``verdict_ref`` and authority-consistent
        ``evidence_kind``: production rows require ``byte_closed`` plus a durable repo-local regular
        non-symlink receipt whose bytes match the REQUIRED ``trusted_receipt_sha256``; research-only
        rows require ``research_diagnostic`` and cannot mint production authority.
      * ``est_delta_s`` (optional) is the POSITIVE ΔS magnitude the candidate buys, joining the same
        relative-significance math the lever queue uses; requires ``axis`` ∈ :data:`VALID_AXES`.
      * Research-only findings preserve authority, verdict scope, activation disposition, measured
        saving, trusted receipt identity, and blockers in this row.  This module never appends an
        activation-ledger event, so a research declaration cannot masquerade as a live fire.
    """
    if not _nonempty_string(candidate):
        raise ValueError(f"candidate must be a non-empty str, got {candidate!r}")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; must be one of {sorted(VALID_STATUSES)}")
    if form_class not in VALID_FORM_CLASSES:
        raise ValueError(f"invalid form_class {form_class!r}; must be one of {sorted(VALID_FORM_CLASSES)}")
    if not _nonempty_string(source_anchor):
        raise ValueError("source_anchor is required (NO-FAKE: every candidate cites its build/design anchor)")
    if not _nonempty_string(gate):
        raise ValueError("gate is required and must be a non-empty str")
    if any(not isinstance(value, str) for value in (justification, slot, owner)):
        raise ValueError("justification, slot, and owner must be strings")
    # exactly-one-of {dsl_lever, dsl_na_reason}
    if not _optional_nonempty_string(dsl_lever) or not _optional_nonempty_string(dsl_na_reason):
        raise ValueError("dsl_lever and dsl_na_reason must be None or non-empty strings")
    if _nonempty_string(dsl_lever) == _nonempty_string(dsl_na_reason):
        raise ValueError(
            "exactly one of dsl_lever / dsl_na_reason is required (the per-row DSL-leg discipline): "
            f"got dsl_lever={dsl_lever!r} dsl_na_reason={dsl_na_reason!r}"
        )
    if not _optional_nonempty_string(verdict_ref):
        raise ValueError("verdict_ref must be None or a non-empty string")
    if not isinstance(research_only, bool):
        raise ValueError("research_only must be a bool")
    if status == STATUS_MEASURED:
        if not _nonempty_string(verdict_ref):
            raise ValueError("status 'measured' requires a verdict_ref (custody-complete verdict artifact)")
        required_kind = EVIDENCE_RESEARCH_DIAGNOSTIC if research_only else EVIDENCE_BYTE_CLOSED
        if evidence_kind != required_kind:
            raise ValueError(f"measured research_only={research_only} requires evidence_kind={required_kind!r}")
    elif evidence_kind is not None:
        raise ValueError("evidence_kind is only valid for status 'measured'")
    if est_delta_s is not None:
        if not _finite_number(est_delta_s):
            raise ValueError(f"est_delta_s must be a finite number or None, got {est_delta_s!r}")
        est_delta_s = float(est_delta_s)
        if est_delta_s < 0:
            raise ValueError(f"est_delta_s must be a positive ΔS magnitude or None, got {est_delta_s!r}")
        if axis not in VALID_AXES:
            raise ValueError(f"est_delta_s requires axis ∈ {sorted(VALID_AXES)}, got {axis!r}")
    elif axis is not None and axis not in VALID_AXES:
        raise ValueError(f"axis must be None or one of {sorted(VALID_AXES)}, got {axis!r}")
    for name, value in (
        ("authority_axis", authority_axis),
        ("verdict_scope", verdict_scope),
        ("activation_status", activation_status),
    ):
        if not _optional_nonempty_string(value):
            raise ValueError(f"{name} must be None or a non-empty string")
    if realized_speedup_factor is not None:
        if not _finite_number(realized_speedup_factor):
            raise ValueError("realized_speedup_factor must be a finite number")
        realized_speedup_factor = float(realized_speedup_factor)
        if realized_speedup_factor < 0.0:
            raise ValueError("realized_speedup_factor must be non-negative")
    if derived_cost_reduction_fraction is not None:
        if not _finite_number(derived_cost_reduction_fraction):
            raise ValueError("derived_cost_reduction_fraction must be a finite number")
        derived_cost_reduction_fraction = float(derived_cost_reduction_fraction)
        if not 0.0 <= derived_cost_reduction_fraction <= 1.0:
            raise ValueError("derived_cost_reduction_fraction must be in [0, 1]")
    if trusted_receipt_sha256 is not None and not _valid_sha256(trusted_receipt_sha256):
        raise ValueError("trusted_receipt_sha256 must be a lowercase SHA-256 hex digest")
    if status == STATUS_MEASURED and (not research_only or trusted_receipt_sha256 is not None):
        custody_error = _receipt_custody_error(
            verdict_ref,
            trusted_receipt_sha256,
            production_candidate=candidate if not research_only else None,
        )
        if custody_error is not None:
            authority = "production measured" if not research_only else "research diagnostic pinned"
            raise ValueError(f"{authority} receipt custody invalid: {custody_error}")
    if blockers is None or isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
        raise ValueError("blockers must be a non-string sequence of non-empty strings")
    blocker_list = list(blockers)
    if any(not _nonempty_string(item) for item in blocker_list):
        raise ValueError("blockers must contain only non-empty strings")
    row = {
        "candidate": candidate,
        "status": status,
        "form_class": form_class,
        "source_anchor": source_anchor,
        "gate": gate,
        "justification": justification,
        "dsl_lever": dsl_lever,
        "dsl_na_reason": dsl_na_reason,
        "slot": slot,
        "owner": owner,
        "est_delta_s": est_delta_s,
        "axis": axis,
        "verdict_ref": verdict_ref,
        "evidence_kind": evidence_kind,
        "research_only": research_only,
        "authority_axis": authority_axis,
        "verdict_scope": verdict_scope,
        "activation_status": activation_status,
        "realized_speedup_factor": realized_speedup_factor,
        "derived_cost_reduction_fraction": derived_cost_reduction_fraction,
        "trusted_receipt_sha256": trusted_receipt_sha256,
        "blockers": blocker_list,
        "agent": agent,
        "ts": _utc(),
    }
    append_locked_jsonl(Path(path) if path is not None else POOL_PATH, row)
    return row


def _read_pool(path: Path | None = None, *, include_seed: bool | None = None) -> dict[str, dict]:
    """Validated latest-row-wins map ``candidate -> row``; malformed overlays are skipped.

    On the DEFAULT store path (``path is None``) the committed :data:`_SEED` inventory is the READ
    BASELINE (``include_seed`` defaults True) — so a fresh checkout is never empty WITHOUT tracking a
    runtime ``.jsonl`` (the store is gitignored, matching the sibling ledgers; the durable inventory
    lives in code). Real recorded rows (fires / measures / retires appended via
    :func:`record_candidate`) OVERLAY the seed baseline (latest-wins). An EXPLICIT ``path`` (tests /
    tooling) reads that file PURELY (``include_seed`` defaults False) for isolation.
    """
    if include_seed is None:
        include_seed = path is None
    out: dict[str, dict] = {}
    if include_seed:
        # A committed seed is an evidence path too, not a validation bypass.  Invalid seeds fail
        # closed exactly like invalid JSONL overlays; production measured seeds must prove live byte
        # custody before they can appear in the typed pool.
        for candidate, row in _seed_map().items():
            if _valid_stored_row(row):
                out[candidate] = row
                continue
            blocked = _blocked_research_seed_for_missing_custody(row)
            if blocked is not None:
                out[candidate] = blocked
    p = Path(path) if path is not None else POOL_PATH
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if not _valid_stored_row(row):
            continue
        candidate = row["candidate"]
        previous = out.get(candidate)
        if (
            previous is not None
            and bool(previous.get("research_only", False))
            and not bool(row.get("research_only", False))
        ):
            # This landing has no explicit promotion/declassification API. Missing/False must never
            # silently turn a research-only SENSE row into production DECIDE authority.
            continue
        out[candidate] = row  # later fully-valid row overwrites earlier -> latest-wins (over seed too)
    return out


@dataclass(frozen=True)
class CandidateStatus:
    candidate: str
    status: str
    form_class: str
    dsl_lever: str | None
    dsl_na_reason: str | None
    gate: str
    source_anchor: str
    slot: str
    owner: str
    est_delta_s: float | None
    axis: str | None
    evidence_kind: str | None
    research_only: bool
    authority_axis: str | None
    verdict_scope: str | None
    activation_status: str | None
    realized_speedup_factor: float | None
    derived_cost_reduction_fraction: float | None
    trusted_receipt_sha256: str | None
    blockers: tuple[str, ...]
    in_duty_queue: bool


def candidate_status(candidate: str, path: Path | None = None) -> CandidateStatus | None:
    """The latest recorded status of ONE candidate, or ``None`` if it is not in the pool."""
    row = _read_pool(path).get(candidate)
    if row is None:
        return None
    return _row_to_status(row)


def _row_to_status(row: dict) -> CandidateStatus:
    return CandidateStatus(
        candidate=row["candidate"],
        status=row["status"],
        form_class=row.get("form_class", ""),
        dsl_lever=row.get("dsl_lever"),
        dsl_na_reason=row.get("dsl_na_reason"),
        gate=row.get("gate", ""),
        source_anchor=row.get("source_anchor", ""),
        slot=row.get("slot", ""),
        owner=row.get("owner", ""),
        est_delta_s=row.get("est_delta_s"),
        axis=row.get("axis"),
        evidence_kind=row.get("evidence_kind"),
        research_only=bool(row.get("research_only", False)),
        authority_axis=row.get("authority_axis"),
        verdict_scope=row.get("verdict_scope"),
        activation_status=row.get("activation_status"),
        realized_speedup_factor=row.get("realized_speedup_factor"),
        derived_cost_reduction_fraction=row.get("derived_cost_reduction_fraction"),
        trusted_receipt_sha256=row.get("trusted_receipt_sha256"),
        blockers=tuple(row.get("blockers", ())),
        in_duty_queue=(row["status"] in _DUTY_STATUSES and not bool(row.get("research_only", False))),
    )


def _sort_key(row: dict) -> tuple:
    """Rank: status order (built-never-fired first) -> est_delta_s desc (None last) -> candidate."""
    est = row.get("est_delta_s")
    return (
        _STATUS_ORDER.get(row.get("status"), 9),
        0 if est is not None else 1,
        -(est or 0.0),
        row.get("candidate", ""),
    )


def pool_report(path: Path | None = None) -> list[dict]:
    """The operator-facing pool: EVERY tracked candidate, ranked built-never-fired -> needs-build ->
    reformulation-queue -> armed -> measured -> retired. Each row carries its DSL-leg disposition +
    gate + source anchor, so a reviewer scans one table and sees what is owed a fire vs already flying."""
    rows = sorted(_read_pool(path).values(), key=_sort_key)
    return rows


def duty_to_measure_pool(path: Path | None = None) -> list[dict]:
    """Curriculum candidates OWED a measurement (built-never-fired / needs-build / reformulation-queue),
    ranked. The costate SENSE reads these into its DECIDE queue beside the lever duty-to-measure line —
    the CONTROLLER holds the curriculum queue, the operator never has to remember it. Research-only
    rows are SENSE signals, never DECIDE/fireable duty, and are excluded here fail-closed."""
    return [
        row
        for row in pool_report(path)
        if row.get("status") in _DUTY_STATUSES and not bool(row.get("research_only", False))
    ]


def pool_summary(path: Path | None = None) -> dict:
    """Machine-readable counts, production-fireable duty, and separate research-only signals."""
    pool = _read_pool(path)
    counts: dict[str, int] = dict.fromkeys(VALID_STATUSES, 0)
    for r in pool.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    duty = duty_to_measure_pool(path)
    research_signals = [row for row in pool_report(path) if bool(row.get("research_only", False))]
    return {
        "total": len(pool),
        "counts": counts,
        "owed": len(duty),
        "top_fireable": duty[:6],
        "research_signals": research_signals,
    }


# ── CANONICAL SEED INVENTORY (from the #403 design memo, re-derived from primary artifacts) ──────
# Each tuple is a candidate row. This is the durable inventory the memo re-derived (labels MEASURED /
# DERIVED / ESTIMATED per row); it is the SEED for the store so a fresh checkout is never empty. Rows
# with a held DSL factory carry ``dsl_lever``; tool-side / vehicle-level / not-yet-built rows carry
# ``dsl_na_reason`` (the per-row DSL-leg discipline — exactly one). NO-FAKE: unbuilt rows are
# ``needs-build`` and built-never-launched rows are ``built-never-fired``; measured rows carry typed
# evidence, and research diagnostics are explicitly non-promotable.
_SEED: tuple[dict, ...] = (
    # ── §1 ARMED (already flying in the sealed v7.5.2 launch config) ──────────────────────────────
    dict(
        candidate="seg_form_unify_tau",
        status=STATUS_ARMED,
        form_class="loss-geometry",
        dsl_lever="SegFormUnifyTau",
        slot="in-run-stage",
        owner="#302",
        gate="composed in v752 launch-1 (DERIVED blinded, #302 §1: CE stage IS the τ≈1 arc)",
        justification="witness-native schedule derivation (dissolves the PR95 CE→tau switch)",
        source_anchor=".omx/research/witness_native_schedule_derivation_20260709.md §1.2",
    ),
    dict(
        candidate="tail_k_warm_restart",
        status=STATUS_ARMED,
        form_class="optimizer-stage",
        dsl_lever="TailCycles",
        slot="in-run-stage",
        owner="#302",
        gate="composed in v752 (DERIVED #302 §2 finite-τ turnpike)",
        justification="TAIL turnpike cycles",
        source_anchor="witness_native_schedule_derivation_20260709.md §2",
    ),
    dict(
        candidate="n323_ladder_island_homotopy",
        status=STATUS_ARMED,
        form_class="loss-geometry",
        dsl_lever="LadderIslandHomotopy",
        slot="in-run-stage",
        owner="#302",
        gate="composed in v752 (MEASURED-vindicated Phase-2 element-4 STRONG MATCH)",
        justification="LADDER island-birth per-class-λ homotopy",
        source_anchor="witness_native_schedule_derivation_20260709.md Phase-2 element-4",
    ),
    dict(
        candidate="r7_polyak_finisher",
        status=STATUS_ARMED,
        form_class="averaging",
        dsl_lever="PolyakFinisher",
        slot="terminal-band",
        owner="#302",
        gate="composed in v752 (extra ckpt candidate; EMA shadow NEVER replaced)",
        justification="turnpike orbit → uniform tail mean O(1/√n) beats phase-carrying EMA",
        source_anchor="witness_native_schedule_derivation_20260709.md (turnpike averaging)",
    ),
    dict(
        candidate="dseg_aware_taper",
        status=STATUS_ARMED,
        form_class="regularizer-schedule",
        dsl_lever="DsegAwareTaper",
        slot="in-run-stage",
        owner="ladder",
        gate="v752 delta (INSTANCE-grade; owed-15 n600 fresh-arm isolation converts INSTANCE→MEASURED-or-rollback)",
        justification="byte-neutral spectral reallocation by GT margin saliency (#121)",
        source_anchor="canonical_equations.dseg_aware_fourier_taper_20260709",
        est_delta_s=None,
        axis=None,
    ),
    # ── §2 BUILT-NEVER-FIRED (the duty-to-measure core; highest readiness) ────────────────────────
    dict(
        candidate="dig_s1_query_real_disagreement_audit_policy",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="state-evolution",
        dsl_lever="witness_dsl.replace_round5_deeper_nonlinear_policy.ReplaceRound5DeeperNonlinearPolicy",
        slot="DIG-S1-QUERY-REAL-CALIBRATION",
        owner="lane_replace_round5_deeper_nonlinear_20260713",
        gate="live admission requires passing localizer, on-policy transition custody, preserved 4% targeted plus 1% randomized-audit propensities, and explicit probability calibration",
        justification="MEASURED research-only error-ranking gate passes (189.813x high/low error; Spearman 0.865610; positive audit propensity), but ensemble ECE 0.186204 and the localizer primary gate fails, so live remains REFUSE",
        source_anchor=".omx/research/replace_round5_deeper_nonlinear_20260713.md",
        verdict_ref="experiments/results/replace_round5_deeper_nonlinear_20260713/receipt.json",
        research_only=True,
        activation_status="REFUSE_LOCALIZER_AND_ECE_GATES_FAILED",
        blockers=("localizer primary gate fails", "ensemble ECE 0.186204 fails explicit probability calibration"),
    ),
    dict(
        candidate="hardness_oversample_lever5",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="data-curriculum",
        dsl_lever="HardnessOversample",
        slot="in-run-stage",
        owner="#403",
        gate="fair A/B = --hardness-weighted on/off at fixed oversample (same total steps, different "
        "allocation); source=realized (trainer-recommended)",
        justification="MEASURED anchors: 44%-of-CE-residual-spikes-are-LANE (#205 CE-floor L67) + "
        "margin-saliency #141; trainer L11303-11324 default-off byte-identical; DSL leg folded this landing",
        source_anchor="experiments/train_levelset_witness_realized_through_R_mlx.py L11303-11324 (LEVER-5)",
    ),
    dict(
        candidate="step_native_finer",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="architecture-growth",
        dsl_lever="StepNativeActivation",
        slot="in-run-stage",
        owner="#310",
        gate="pinned v7.5.2 ladder rung; byte-close A/B annealed-β+FINER vs OFF; surviving flips must move to no-ring survival",
        justification="ESTIMATED 31.6% of remaining descent (FEED-stepnative); negcure RANK-4 second-exemplar",
        source_anchor="DAG FEED-stepnative (factory+flags+equation 07-07; guard-hardened 07-09)",
        est_delta_s=None,
        axis="d_seg",
    ),
    dict(
        candidate="focal_boundary_distance_301",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        dsl_lever="SegFocalGamma",
        slot="in-run-stage",
        owner="#301",
        gate="PRE-REGISTERED (#301 item 3): ep50→100 witness-alone slope flattens (|Δd_seg|<0.02/25ep, "
        "islands>50% of residual) → deploy focal at γ*; steep → HOLD. CAVEAT: re-check γ* on live ckpt (C17 γ*=0)",
        justification="MEASURED build (task #301 completed: default-OFF byte-identical, tested, READY-not-deployed)",
        source_anchor="task #301 (completed); BoundaryDistance sister lever",
    ),
    dict(
        candidate="head_geometry_218_etf_am",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        dsl_lever="HeadGeometry",
        slot="in-run-stage",
        owner="#218",
        gate="fire ETF first (byte-free + rate-win, neural-collapse minority-norm fix); AM-hinge needs "
        "--margin-field-head-weight>0",
        justification="DERIVED (#218: rare-class lane margin fix; trainer L10814-10824 built default-off); "
        "DSL leg folded this landing",
        source_anchor="experiments/train_levelset_witness_realized_through_R_mlx.py L10814-10824 (#218 facet-1)",
    ),
    dict(
        candidate="persistence_topology_218_224",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        dsl_lever="PersistenceTopology",
        slot="in-run-stage",
        owner="#218",
        gate="warm-up epochs param; fire with the #218 rung; --persistence-classes auto self-detects erasure-tail classes",
        justification="MEASURED build (#224 wired; dash erasure law dash_erasure_homogenization_v1 is the target, L65)",
        source_anchor="canonical_equations.dash_erasure_homogenization_v1",
    ),
    dict(
        candidate="msal_uni_sr_reachability_268",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        dsl_lever="MarginSaliencyReachability",
        slot="in-run-stage",
        owner="#268",
        gate="ZERO build: gt_n600_sR.npz READY; A/B on the fragile annulus band",
        justification="MEASURED: texture proxy AT CHANCE vs through-R reachability (L76); negcure RANK-2 second-exemplar-grade",
        source_anchor="L76 (LEVER-4 msal_uni through-R reachability)",
    ),
    dict(
        candidate="weight_entropy_penalty_balle",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="regularizer-schedule",
        dsl_lever="WeightEntropyPenaltyMLX",
        slot="in-run-stage",
        owner="#397",
        gate="fire on a rate-attack arm; MEASURED −19.6% bytes (torch ancestor — ancestor-rule: number does NOT transfer, mechanism does)",
        justification="FEED-reactivation-397 T-397-1 (built-never-fired factory)",
        source_anchor="DAG FEED-reactivation-397 T-397-1",
        est_delta_s=None,
        axis="rate",
    ),
    dict(
        candidate="laguerre_ot_head_offset",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="discrete-solve-interleave",
        dsl_lever="HeadOffsetSolver",
        slot="terminal-band",
        owner="#397",
        gate="fire as a terminal-band solve rung after gradient descent bottoms",
        justification="deep-math #284 (argmax = Laguerre power diagram, L-v8)",
        source_anchor="DAG FEED-reactivation-397 T-397-1 + #284",
    ),
    dict(
        candidate="steik_normalized_316",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="regularizer-schedule",
        dsl_na_reason="eikonal sub-knobs compiled via EikonalViscosity-family config, not a bare-name lever (fold owed if promoted)",
        slot="in-run-stage",
        owner="#316",
        gate="#316 FAIR eikonal-viscosity test first (operator-GO flagged; D1 known-tainted era)",
        justification="MEASURED: RAW StEik NO-GO n24 (self-amplifying 575×-1431×, verdict_scope: formulation); normalized = named follow-up (FEED-05v)",
        source_anchor="DAG FEED-05v",
    ),
    dict(
        candidate="viscoreg_eps_continuation_316",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="regularizer-schedule",
        dsl_lever="EikonalViscosity",
        slot="in-run-stage",
        owner="#316",
        gate="#316 fair test (first non-confounded measurement; re-grades D1/D2/D7)",
        justification="MEASURED n24: ε=0.3 STABLE + d_seg 2.3× better than control; ε=1.0 explodes (two-sided window; FEED-05v)",
        source_anchor="DAG FEED-05v",
        est_delta_s=None,
        axis="d_seg",
    ),
    dict(
        candidate="per_param_grad_normalize",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="optimizer-stage",
        dsl_na_reason="stability-preset surface (autoconfig resolve_stability_config), not a swept DSL lever yet",
        slot="in-run-stage",
        owner="—",
        gate="OWED trajectory A/B (alters seg-vs-pose gradient scale ratio — documented caveat)",
        justification="MEASURED-built FEED-collapsefix (byte-identical default; candidate better PRIMARY than global clip for batch=1)",
        source_anchor="DAG FEED-collapsefix",
    ),
    dict(
        candidate="muon_event_derived_switch",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="optimizer-stage",
        dsl_na_reason="--muon-start-event flag exists (unmapped sub-knob of the event scaffold); fold owed if promoted",
        slot="in-run-stage",
        owner="#302",
        gate="fire when conditioning event (σ_min/curvature) trips AND nucleation done (#302 §3c)",
        justification="DERIVED (#302: fixed-726 = un-derived knee-transfer residual; event scaffold present)",
        source_anchor="witness_native_schedule_derivation_20260709.md §3c",
    ),
    dict(
        candidate="length_sigma_matrix_rung1b",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        dsl_lever="LengthSigma",
        slot="in-run-stage",
        owner="ladder",
        gate="pinned rung 1b: fires at the tau boundary BEFORE any other length-touching lever",
        justification="MEASURED build (fitted-20260707 inherited then deliberately reverted to σ≡1 to keep Class-A attribution clean)",
        source_anchor="SPEC_v752 §A.8/§1b",
    ),
    dict(
        candidate="chroma_rung_752",
        status=STATUS_BUILT_NEVER_FIRED,
        form_class="loss-geometry",
        dsl_lever="SegChromaBoundary",
        slot="in-run-stage",
        owner="ladder",
        gate="rung fires per FEED-chroma-rung registration (its own increment A/B); SegNet reads RGB → chroma is a d_seg actuator",
        justification="registered-off rung (CLAUDE.md §Chroma: any verdict ignoring chroma is provisional)",
        source_anchor="DAG FEED-chroma-rung",
        est_delta_s=None,
        axis="d_seg",
    ),
    # ── §3 NEEDS-BUILD (design/task exists, no code on THIS vehicle) ──────────────────────────────
    dict(
        candidate="swa_tail_soup",
        status=STATUS_NEEDS_BUILD,
        form_class="averaging",
        dsl_na_reason="checkpoint-space op (byte-close-side tool + trainer export hook), not a flag; unbuilt",
        slot="terminal-band",
        owner="NEW (follow-up task)",
        gate="design-memo-first; A/B vs PolyakFinisher (built sibling) + EMA; EMA never replaced",
        justification="DERIVED: turnpike cycles produce K basin-endpoint iterates; cross-cycle soup is the unbuilt complement (grep: zero swa/soup code in tree)",
        source_anchor="pool row §3.1 (grep MEASURED: no swa/soup code)",
    ),
    dict(
        candidate="fisher_gn_head_full_p_solve",
        status=STATUS_NEEDS_BUILD,
        form_class="preconditioning",
        dsl_na_reason="in-trainer CG solve block, not a flag; unbuilt",
        slot="in-run-stage",
        owner="#341-adjacent (follow-up task)",
        gate="JOINS #341: quadratic head chart CONFIRMED (LM ρ 0.847/0.868) but K=8 subset-solve overfits (+5.1% net) → ONLY full-P in-trainer GPU solve admissible (~11min/CG-iter@17×, L77)",
        justification="MEASURED foundation: margin=Fisher ρ0.978 (L1) + eq quadratic_head_chart_subset_solve_gap_v1",
        source_anchor="canonical_equations.quadratic_head_chart_subset_solve_gap_v1",
    ),
    dict(
        candidate="in_run_click_interleave",
        status=STATUS_NEEDS_BUILD,
        form_class="discrete-solve-interleave",
        dsl_na_reason="mc_finisher is a byte-close-side tool (#400 diagonal); in-run interleave needs a trainer-side design; unbuilt",
        slot="terminal-band → in-run-stage",
        owner="#400 (+ follow-up task)",
        gate="promotion gate: #400 diagonal MEASUREMENT first (terminal-band); only then design the in-run interleave",
        justification="MEASURED: n8 click row MOVED the pointer −1.7e-5 (FEED-pointer-move-n8click); n600 sweep in-flight",
        source_anchor="DAG FEED-pointer-move-n8click",
    ),
    dict(
        candidate="iga_boundary_tangent_ntk_309",
        status=STATUS_NEEDS_BUILD,
        form_class="preconditioning",
        dsl_na_reason="NTK-level preconditioning, not a flag; unbuilt",
        slot="in-run-stage",
        owner="#309",
        gate="HELD-WEAKENED: owed-16 v1 basis ≈0 zero-shot AND owed16v2 REBALANCE no-benefit (FEED-legdisposition-owed16v2) → re-derive whether the NTK mechanism survives before building",
        justification="MEASURED 3.2× along-tangent deficit (L65) stands, but two basis-level cures measured ≈0 — negative↔cure join points AWAY",
        source_anchor="DAG FEED-legdisposition-owed16v2",
    ),
    dict(
        candidate="sam_flat_minima_mdl_242",
        status=STATUS_NEEDS_BUILD,
        form_class="regularizer-schedule",
        dsl_na_reason="SAM pre-quant stage, not a flag; unbuilt",
        slot="in-run-stage",
        owner="#242",
        gate="design-memo-first; adjacency: WeightEntropyPenaltyMLX (§2.7) covers the rate-in-loss half — fire that FIRST, build SAM only if the entropy penalty measures insufficient",
        justification="DERIVED (compression-as-intelligence; task #242 pending)",
        source_anchor="task #242 (pending)",
    ),
    dict(
        candidate="grids_bulk_inr_annulus_308",
        status=STATUS_NEEDS_BUILD,
        form_class="architecture-growth",
        dsl_na_reason="vehicle-level (grids+INR hybrid), not a flag; unbuilt",
        slot="next-vehicle",
        owner="#308",
        gate="negcure HELD (no matched MEASURED violated fact); v8 per-class carriers partially embody it — re-scope AGAINST v8 increment-1 before building",
        justification="DERIVED (NeurIPS'25 grids-beat-INRs-except-boundary; matched-bytes protocol pre-registered)",
        source_anchor="task #308 (pending)",
    ),
    dict(
        candidate="post_muon_sgld_217",
        status=STATUS_NEEDS_BUILD,
        form_class="optimizer-stage",
        dsl_na_reason="SGLD sub-stage, not a flag; unbuilt (grep MEASURED: zero SGLD code in trainer)",
        slot="terminal-band",
        owner="#217/#216",
        gate="GATED on #216 saddle-to-saddle signature test ($0, run first); components (i) Damian reweight ≈ hardness/margin-saliency (§2.1/§2.6 partially cover)",
        justification="DERIVED (MFLD multi-index leap theory; Muon≈Stiefel so it applies exactly)",
        source_anchor="pool row §3.7 (grep: zero SGLD code)",
    ),
    dict(
        candidate="kd_warm_start_129",
        status=STATUS_NEEDS_BUILD,
        form_class="init-warm-start",
        dsl_na_reason="production actuator spanning export, not a flag; PARTIALLY built (FiLM-v2 trunk-decoupling DONE 867ff3af5; actuator pending)",
        slot="next-vehicle",
        owner="#129",
        gate="bind-all spec production_readiness_bind_all_ingredients_20260616.md; #301 banked it as the 3rd rung (KD-from-#205-teacher on island band)",
        justification="DERIVED (Hinton KD; production linchpin)",
        source_anchor="commit 867ff3af5 (FiLM-v2 trunk-decoupling; ∂d_seg/∂pose=0 proven)",
    ),
    dict(
        candidate="simpletes_k_gt1_319",
        status=STATUS_NEEDS_BUILD,
        form_class="discrete-solve-interleave",
        dsl_na_reason="shadow-controller shape (campaign layer), not a trainer flag; unbuilt (grep: not in witness_control)",
        slot="in-run-stage (advisory layer)",
        owner="#319/#315",
        gate="GATED behind #315 + BINDING backtest against v1-v5+#205 logs before adoption; fire when through-R band spans 0 at n=3",
        justification="DERIVED (SimpleTES DF-1 split-verdict; costate core NOT-RELEVANT refused)",
        source_anchor="pool row §3.9 (grep: not in witness_control)",
    ),
    dict(
        candidate="md_decoupling_195",
        status=STATUS_NEEDS_BUILD,
        form_class="optimizer-stage",
        dsl_na_reason="no --optimizer/--md-base in the levelset trainer = TRAINER-GAP; unbuilt on this vehicle",
        slot="in-run-stage",
        owner="#195",
        gate="build the optimizer flag, then LR-transfer A/B per #195",
        justification="DERIVED (MD stable-by-construction claim is UNVERIFIED on this vehicle — that is the point of the A/B); 5-LENS CONTRADICT row",
        source_anchor="pool row §3.10 (5-LENS trainer-gap)",
    ),
    dict(
        candidate="pose_inverse_carrier_distill",
        status=STATUS_NEEDS_BUILD,
        form_class="pose-carrier",
        dsl_na_reason="decoder-native generator + offline distillation, not a flag; unbuilt (research_only)",
        slot="terminal-band / next-vehicle",
        owner="#248/#366",
        gate="ADVISORY only (research_only=true): offline PoseNet discovery/distill of frame-0 corrections; archive must decode WITHOUT PoseNet/scorer/GT tables; gate = #248 P-B FiLM read-back decisive first",
        justification="DERIVED (ADVISORY_sdf_pose_inverse_carrier_20260710: unconstrained frame-0 inverse proves evaluator admits accurate witnesses)",
        source_anchor=".omx/research/ADVISORY_sdf_pose_inverse_carrier_20260710.md",
        research_only=True,
        activation_status="RESEARCH_ONLY_ADVISORY_NO_PRODUCTION_ACTIVATION",
        blockers=(
            "offline decoder-native generator and distillation remain unbuilt",
            "archive must decode without PoseNet, scorer, or GT tables",
            "#248 P-B FiLM read-back gate unresolved",
        ),
    ),
    dict(
        candidate="som_organized_codebooks",
        status=STATUS_NEEDS_BUILD,
        form_class="state-evolution",
        dsl_na_reason="design-banked representation-side codebook, not a flag; unbuilt",
        slot="terminal-band / next-vehicle",
        owner="NEW (design-banked)",
        gate="EXACT-GATED A/B ONLY (representation-side chart levers repeatedly MEASURED ≈0 realized through R); pays twice IF measured: click-polish ±1/±2 locality + temporal-delta rate",
        justification="DERIVED: SOM magnification law under-allocates rare regions = the measured lane starvation; the conscience CURE already embodied by LADDER per-class λ + #218 — codebook is the only NEW piece",
        source_anchor=".omx/research/papers_checked_kohonen_som_20260710.md item 4",
    ),
    # ── §4 REFORMULATION-QUEUE (prior formulation measured NO-GO; named reformulation owed) ───────
    dict(
        candidate="nca_lppn_state_146",
        status=STATUS_REFORMULATION_QUEUE,
        form_class="state-evolution",
        dsl_na_reason="vehicle-level NCA/LPPN stage, not a flag; unbuilt",
        slot="next-vehicle",
        owner="#146/P10",
        gate="fires ONLY on AMBER/P10 reactivation; reformulation = coarse-NCA-grid + our trunk as the LPPN (the split #146's arm lacked)",
        justification="MEASURED wall stands (#146 33K-rule generalization gate); reformulation named in papers_checked",
        source_anchor=".omx/research/papers_checked_cells2pixels_nca_lppn_20260710.md",
    ),
    dict(
        candidate="curvelet_from_scratch_trajectory",
        status=STATUS_REFORMULATION_QUEUE,
        form_class="init-warm-start",
        dsl_na_reason="trajectory-shaping curvelet init, not a flag; re-derivation owed before build",
        slot="in-run-stage",
        owner="#397",
        gate="owed16v2 measured no-benefit ⇒ confound resolved AGAINST the axis; gate = a fresh derivation must name a mechanism that survives BOTH measured negatives before any build",
        justification="MEASURED: naive-palette realized ceiling F=0.0337 ≫ directional-direct 0.0037; trained trunk redundant with basis |Δ|≤1.4% (FEED-reactivation-397 FIRE-1)",
        source_anchor="DAG FEED-reactivation-397 FIRE-1",
    ),
    # ── §5 EXPLICITLY EXCLUDED (retired-with-reason — do not re-propose) ──────────────────────────
    # The first rows in this section are measured research signals: visible in SENSE, never
    # production-activated. Explicit retired-with-reason exclusions follow them below.
    {
        "candidate": "p0_guarded_exact_costate_reuse_k2",
        "status": STATUS_MEASURED,
        "form_class": "state-evolution",
        "dsl_lever": "exact_costate_reuse_k2_lever",
        "slot": "in-run-stage",
        "owner": "lane_p0_backward_closer_20260713",
        "gate": "NOT_ADMITTED corrected n600 gate: accepted stale-minus-exact d_seg regret <= 0 only 308/456 (requires 456/456); renderer-gradient relL2 < 1 passed 456/456; default OFF",
        "justification": "MEASURED research diagnostic: 456/600 behavioral full-facet accepts (p=0.76), but the corrected fidelity gate failed; admitted realized bulk saving is 1.0x / reduction 0.0",
        "source_anchor": ".omx/research/p0_costate_reuse_k2_corrected_adjudication_receipt_20260714.json",
        "verdict_ref": ".omx/research/p0_costate_reuse_k2_corrected_adjudication_receipt_20260714.json",
        "evidence_kind": EVIDENCE_RESEARCH_DIAGNOSTIC,
        "research_only": True,
        "authority_axis": "[macOS-CPU advisory; cached n600 Torch/NumPy-fp32 training-gradient MEANS; teacher-backward slice only; NON-GLOBAL]",
        "verdict_scope": "corrected source-bound n600 K=2 teacher-backward diagnostic only; NOT a global throughput win; HEAD e59f69a79c dominant 95%-kill forward-only frozen authority verdict remains controlling; pointer_moved=false; score_claim=false; FIDELITY_BLOCKED_PENDING_NEW_FORMULATION",
        "activation_status": "NOT_ADMITTED_DEFAULT_OFF_NO_LIVE_PROVIDER_OR_RESUME_REGISTRATION",
        "realized_speedup_factor": 1.0,
        "derived_cost_reduction_fraction": 0.0,
        "trusted_receipt_sha256": "30ce7e5e23b10cb15c52a89debc57b0bf5349be16ed9cb0e97c3974579465ff7",
        "blockers": (
            "accepted stale-minus-exact d_seg regret <= 0 only 308/456; corrected fidelity gate requires 456/456",
            "DERIVED_COUNTERFACTUAL_BEHIND_FAILED_FIDELITY_GATE: exact-backward-call amortization 1.6129032258064517x and reduction 0.38; never achieved, admitted, global, or wall-clock",
            "live exact-costate provider absent",
            "canonical resume-registry integration absent",
            "in-loop timer is owed only after a fresh formulation passes fidelity admission; no timer is owed for this rejected formulation",
        ),
    },
    {
        "candidate": "p0_sparse_adjoint_dense_fullrank",
        "status": STATUS_MEASURED,
        "form_class": "preconditioning",
        "dsl_na_reason": "measured diagnostic formulation; NO_GO_DENSE_FULLRANK has no production activation",
        "slot": "teacher backward",
        "owner": "p0_sparse_adjoint",
        "gate": "NO_GO_DENSE_FULLRANK; realized exact/dense saving 1.0x; do not dispatch the tested arm",
        "justification": "MEASURED n600 source-bound costate replay; ideal custom spatial ceiling is not realized exact/dense saving",
        "source_anchor": ".omx/research/p0_sparse_adjoint_costate_vjp_20260713.md",
        "verdict_ref": ".omx/research/p0_sparse_adjoint_costate_vjp_20260713.md",
        "evidence_kind": EVIDENCE_RESEARCH_DIAGNOSTIC,
        "research_only": True,
        "authority_axis": "[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]",
        "verdict_scope": "source-bound task455 n600 replay; frozen EfficientNet-B2 U-Net SegNet CE input costate; 4.7366% output masks and high-fidelity cross-state low-rank basis",
        "activation_status": "NO_PRODUCTION_ACTIVATION_NO_GO_DENSE_FULLRANK",
        "realized_speedup_factor": 1.0,
        "trusted_receipt_sha256": "bc3e68c139f8472cd43badeb6ce70d3270f2a30945c714a0b2c1d8da57eeb771",
        "blockers": ("dense framework realizes no sparse saving", "tested input costate is nonzero and high-rank"),
    },
    {
        "candidate": "p0_terminal_exact_metric_396_costate_skip",
        "status": STATUS_MEASURED,
        "form_class": "discrete-solve-interleave",
        "dsl_lever": "terminal_exact_metric_costate_skip_lever",
        "slot": "terminal-band",
        "owner": "lane_p0_backward_closer_20260713",
        "gate": "route-local exact-metric #396 accept/reject only; SPSA/ES and live trainer activation unadmitted",
        "justification": "MEASURED n600-composed exact objective; DERIVED zero costate calls and 1.0 route-local teacher-cost reduction",
        "source_anchor": ".omx/research/p0_terminal_costate_skip_handoff_20260713.json",
        "verdict_ref": ".omx/research/p0_terminal_costate_skip_handoff_20260713.json",
        "evidence_kind": EVIDENCE_RESEARCH_DIAGNOSTIC,
        "research_only": True,
        "authority_axis": "[macOS-CPU advisory; frozen CPU-torch exact cells; NON-PROMOTABLE]",
        "verdict_scope": "admitted only for the post-training #396 exact-metric accept/reject terminal route on the pinned n600 objective; bulk SPSA/ES, training-loop teacher reduction, contest score, and runtime claims are not admitted",
        "activation_status": "ROUTE_LOCAL_ONLY_NO_LIVE_TRAINER_ACTIVATION",
        "derived_cost_reduction_fraction": 1.0,
        "trusted_receipt_sha256": "17574857da5ff862e520140977e988197962f009d6870d23fe3071c398112a9c",
        "blockers": (
            "SPSA/ES effective-dimension certificate unadmitted",
            "live trainer activation absent",
            "bulk training reduction unquantified",
        ),
    },
    # Explicitly excluded candidates (retired-with-reason; do not re-propose).
    dict(
        candidate="gradnorm_per_step_balancing",
        status=STATUS_RETIRED,
        form_class="optimizer-stage",
        dsl_na_reason="per-step loss balancing FORBIDDEN by law (not a lever)",
        slot="—",
        owner="#312",
        gate="RETIRED: #312 law — naive GradNorm would DOWN-weight the eikonal CANARY mid-runaway; loss weights move at STAGE BOUNDARIES ONLY (assert_loss_weights_stage_boundary_only)",
        justification="MEASURED law (FEED-05r §4); sanctioned form = the #312 stage-boundary gradient-share checkpoint probe (task completed)",
        source_anchor="DAG FEED-05r §4 + SPEC_v75 §8-C",
    ),
    dict(
        candidate="mod_dim_as_capacity_reopen",
        status=STATUS_RETIRED,
        form_class="architecture-growth",
        dsl_na_reason="mod-dim-as-capacity re-open (not a lever); CLOSED",
        slot="—",
        owner="#299",
        gate="RETIRED: #299 CLOSED (verdict_scope: formulation) — refuted by #300's measured island-gradient-starvation mechanism",
        justification="MEASURED (FEED-reactivation-397)",
        source_anchor="DAG FEED-reactivation-397 (#299/#300)",
    ),
    dict(
        candidate="raw_gradient_steik",
        status=STATUS_RETIRED,
        form_class="regularizer-schedule",
        dsl_na_reason="raw-gradient StEik (not a lever); fire the NORMALIZED variant (§2.9) instead",
        slot="—",
        owner="#316",
        gate="RETIRED: MEASURED NO-GO n24 (self-amplifying 575×–1431×); the normalized variant is steik_normalized_316 — fire THAT, never the raw form",
        justification="MEASURED (FEED-05v)",
        source_anchor="DAG FEED-05v",
    ),
)


def _seed_map() -> dict[str, dict]:
    """``{candidate: full-row}`` from the committed :data:`_SEED` inventory — the READ BASELINE for the
    default store path. Each seed dict is normalized to the full row shape (missing optional fields
    defaulted) so consumers read it identically to a recorded row. Pure; deterministic."""
    out: dict[str, dict] = {}
    for row in _SEED:
        out[row["candidate"]] = {
            "candidate": row["candidate"],
            "status": row["status"],
            "form_class": row.get("form_class", ""),
            "source_anchor": row.get("source_anchor", ""),
            "gate": row.get("gate", ""),
            "justification": row.get("justification", ""),
            "dsl_lever": row.get("dsl_lever"),
            "dsl_na_reason": row.get("dsl_na_reason"),
            "slot": row.get("slot", ""),
            "owner": row.get("owner", ""),
            "est_delta_s": row.get("est_delta_s"),
            "axis": row.get("axis"),
            "verdict_ref": row.get("verdict_ref"),
            "evidence_kind": row.get("evidence_kind"),
            "research_only": bool(row.get("research_only", False)),
            "authority_axis": row.get("authority_axis"),
            "verdict_scope": row.get("verdict_scope"),
            "activation_status": row.get("activation_status"),
            "realized_speedup_factor": row.get("realized_speedup_factor"),
            "derived_cost_reduction_fraction": row.get("derived_cost_reduction_fraction"),
            "trusted_receipt_sha256": row.get("trusted_receipt_sha256"),
            "blockers": list(row.get("blockers", ())),
            "agent": "seed",
            "ts": "seed",
        }
    return out


def seed_default_pool(path: Path | None = None, *, agent: str = "task_403_seed") -> int:
    """Seed the store from :data:`_SEED` — writes ONLY candidates not already present (idempotent by
    ``candidate`` key, so re-running never duplicates a row). Returns the number of rows written.

    This is what makes a fresh checkout's pool non-empty (the durable inventory the memo re-derived);
    a later real event (a fire/measure/retire) is recorded via :func:`record_candidate` and wins on
    read (latest-wins). NO-FAKE: seeds carry the HONEST status and evidence kind per row; measured
    research diagnostics remain research-only and never enter production duty."""
    existing = set(_read_pool(path).keys())
    written = 0
    for row in _SEED:
        if row["candidate"] in existing:
            continue
        record_candidate(**{**row, "agent": agent}, path=path)
        written += 1
    return written


__all__ = [
    "EVIDENCE_BYTE_CLOSED",
    "EVIDENCE_RESEARCH_DIAGNOSTIC",
    "POOL_PATH",
    "PRODUCTION_RECEIPT_SCHEMA",
    "PRODUCTION_RECEIPT_TYPE",
    "STATUS_ARMED",
    "STATUS_BUILT_NEVER_FIRED",
    "STATUS_MEASURED",
    "STATUS_NEEDS_BUILD",
    "STATUS_REFORMULATION_QUEUE",
    "STATUS_RETIRED",
    "VALID_AXES",
    "VALID_EVIDENCE_KINDS",
    "VALID_FORM_CLASSES",
    "VALID_STATUSES",
    "CandidateStatus",
    "candidate_status",
    "duty_to_measure_pool",
    "pool_report",
    "pool_summary",
    "record_candidate",
    "seed_default_pool",
]
