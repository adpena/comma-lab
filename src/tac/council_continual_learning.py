# SPDX-License-Identifier: MIT
"""Council deliberation continual-learning canonical helper (Catalog #300 sister).

Per the COUNCIL-HIERARCHY-V2 landing 2026-05-16 (operator-approved 4-tier
protocol with maximum-signal preservation + continual-learning wire-in
meta-principles): every T2+ council deliberation (and every T1 working group
whose finding crosses an elevation trigger) MUST emit a continual-learning
anchor via :func:`append_council_anchor`. The persisted council verdict +
dissent + assumption classification become signal that future deliberations,
the autopilot ranker, and the Rashomon ensemble can consume.

This file is the **canonical** entry point. Any caller persisting council
state outside :func:`append_council_anchor` re-introduces the orphaned-work
bug class per CLAUDE.md "Subagent coherence-by-default" non-negotiable: a
council finding the planner cannot see is orphaned signal.

Schema and on-disk format mirror :mod:`tac.cost_band_calibration` (the
canonical fcntl-locked JSONL append-only state writer per Catalog #128 +
Catalog #131 + Catalog #175 / #177 discipline). Read-side strict-load
mirrors :mod:`tac.deploy.lightning.active_jobs_state` (Catalog #138 strict-
load fail-closed pattern).

Persisted at ``.omx/state/council_deliberation_posterior.jsonl`` (HISTORICAL_
PROVENANCE per Catalog #110 / #113 / #132 — APPEND-ONLY; outcomes are NEW
rows referencing the same ``deliberation_id``). Locked via the sibling
``.omx/state/.council_deliberation_posterior.lock`` per Catalog #131
bare-write discipline.

Public API:

* :class:`CouncilTier` — enum-like sentinel constants (T1/T2/T3/T4).
* :class:`CouncilDeliberationRecord` — frozen dataclass mirroring the v2
  frontmatter (council_tier / attendees / verdict / dissent /
  assumption_adversary_verdict / decisions_recorded / etc.).
* :func:`append_council_anchor` — fcntl-locked JSONL append-only writer
  per Catalog #128 / #131 discipline.
* :func:`load_council_anchors_strict` — raises
  :class:`CouncilPosteriorCorruptError` on JSON parse failure (Catalog
  #138 pattern).
* :func:`load_council_anchors` — lenient loader; skips malformed lines.
* :func:`query_anchors_by_topic` — cite-chain helper.
* :func:`query_dissent_history` — surface minority opinions by member.
* :func:`query_assumption_classification_history` — surface HARD-EARNED-
  vs-CARGO-CULTED verdicts for a given assumption.

Verified against:

* Catalog #128 / #131 (fcntl-locked shared state writes).
* Catalog #138 (strict-load fail-closed for mutating paths).
* Catalog #245 (Modal call_id ledger — sister APPEND-ONLY JSONL pattern).
* CLAUDE.md "Council conduct" Fix-7 amendment (per-round assumption
  surfacing).
* CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" (HARD-EARNED-vs-CARGO-
  CULTED classification).

Memory: ``feedback_council_hierarchy_v2_landed_20260516.md`` [verified-
against: Catalog #128, #131, #138, #245, #292].
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "CouncilTier",
    "VALID_TIERS",
    "VALID_VERDICTS",
    "CouncilDeliberationRecord",
    "CouncilPosteriorCorruptError",
    "DEFAULT_COUNCIL_POSTERIOR_PATH",
    "DEFAULT_COUNCIL_POSTERIOR_LOCK_PATH",
    "SCHEMA_VERSION",
    "append_council_anchor",
    "load_council_anchors",
    "load_council_anchors_strict",
    "query_anchors_by_topic",
    "query_dissent_history",
    "query_assumption_classification_history",
    "update_from_anchor",
]


SCHEMA_VERSION = "council_deliberation_posterior_v1"


# Repo root resolution mirrors the rest of tac.* (this module sits in
# ``src/tac/``; the repo root is two parents up).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


DEFAULT_COUNCIL_POSTERIOR_PATH = (
    _REPO_ROOT / ".omx" / "state" / "council_deliberation_posterior.jsonl"
)
DEFAULT_COUNCIL_POSTERIOR_LOCK_PATH = (
    _REPO_ROOT / ".omx" / "state" / ".council_deliberation_posterior.lock"
)


class CouncilTier:
    """4-tier council hierarchy v2 sentinel constants.

    Per the COUNCIL-HIERARCHY-V2 spec (operator-approved 2026-05-16):

    * **T1** — Working Group (1-3 named members; bounded scope; ≤30 min
      deliberation; no veto power; output is a recommendation feeding a
      T2/T3 deliberation).
    * **T2** — Inner-Skunkworks (5-of-6 sextet pact: Shannon LEAD /
      Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-
      Adversary; binding decision authority for in-flight engineering
      tradeoffs; Shannon tie-break).
    * **T3** — Full Grand Council (≥12-of-20 grand council + 5-of-6
      sextet; binding decisions touching CLAUDE.md non-negotiables /
      cross-cutting wire-ins / strategic redirection; Shannon tie-break
      with Dykstra fallback).
    * **T4** — Symposium (≥16-of-20 + 6-of-6 sextet + ≥1 specialist per
      affected paradigm; required for kill-and-replace decisions /
      multi-month directional shifts / operator-pre-attention escalation;
      operator-resolves on tie).
    """

    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


VALID_TIERS = frozenset({CouncilTier.T1, CouncilTier.T2, CouncilTier.T3, CouncilTier.T4})


# Verdict enum-like; deliberation outcomes that the autopilot ranker /
# Rashomon ensemble can consume.
VALID_VERDICTS = frozenset({
    "PROCEED",
    "PROCEED_WITH_REVISIONS",
    "DEFER_PENDING_EVIDENCE",
    "REFUSE",
    "ESCALATE_TO_OPERATOR",
    "ESCALATE_TO_HIGHER_TIER",
})


class CouncilPosteriorCorruptError(RuntimeError):
    """Raised by :func:`load_council_anchors_strict` on JSON parse failure.

    Mirrors :class:`ActiveJobsCorruptError` / :class:`CallIdLedgerCorruptError`
    per Catalog #138 strict-load discipline. The lenient counterpart
    :func:`load_council_anchors` silently skips malformed rows so an
    audit reader can survive partial corruption; the strict variant
    raises so a writer cannot silently reset corrupt state.
    """


@dataclass(frozen=True)
class CouncilDeliberationRecord:
    """One council deliberation event: the v2 frontmatter persisted to JSONL.

    Schema is intentionally append-only; fields added in future v2.N
    revisions MUST be optional (default to ``None`` or empty collection)
    so legacy rows continue loading.

    Field semantics:

    * ``deliberation_id`` — slugged identifier (e.g.
      ``council_hierarchy_v2_landing_20260516``); same id appears across
      ``dispatched`` / ``outcome`` rows per CLAUDE.md Modal call_id
      ledger sister pattern.
    * ``topic`` — short human-readable subject line.
    * ``council_tier`` — one of :attr:`CouncilTier.T1` .. ``T4``.
    * ``council_attendees`` — list of named members (e.g.
      ``["Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian",
      "Assumption-Adversary"]``).
    * ``council_quorum_met`` — boolean; per the v2 quorum rules.
    * ``council_verdict`` — one of :data:`VALID_VERDICTS`.
    * ``council_dissent`` — list of ``{"member": ..., "verbatim": ...}``
      dicts; verbatim minority opinions preserved per CLAUDE.md
      "maximum signal preservation" rule.
    * ``council_assumption_adversary_verdict`` — per-assumption
      classifications: list of ``{"assumption": ..., "classification":
      "HARD-EARNED" | "CARGO-CULTED" | "UNCLEAR", "rationale": ...}``.
      Required at T2+ per Catalog #292.
    * ``council_decisions_recorded`` — list of decisions / op-routables
      emitted by the deliberation (the actuator surface; downstream
      sister subagents / autopilot consume these).
    * ``related_deliberation_ids`` — cite-chain pointers to prior
      deliberations on the same topic so future Rashomon ensemble
      members can trace position evolution.
    * ``event_type`` — ``"dispatched"`` for the deliberation landing;
      ``"outcome"`` for downstream empirical verdicts referencing the
      same ``deliberation_id``.
    * ``parent_id_or_session`` — Claude subagent/session id that
      authored the record (continual learning provenance).
    """

    deliberation_id: str
    topic: str
    council_tier: str
    council_attendees: tuple[str, ...]
    council_quorum_met: bool
    council_verdict: str
    council_dissent: tuple[dict[str, str], ...] = field(default_factory=tuple)
    council_assumption_adversary_verdict: tuple[dict[str, str], ...] = field(
        default_factory=tuple
    )
    council_decisions_recorded: tuple[str, ...] = field(default_factory=tuple)
    related_deliberation_ids: tuple[str, ...] = field(default_factory=tuple)
    event_type: str = "dispatched"
    parent_id_or_session: str | None = None
    memory_path: str | None = None
    notes: str = ""
    written_at_utc: str = ""
    written_pid: int | None = None
    written_host: str | None = None
    schema: str = SCHEMA_VERSION


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_record(record: CouncilDeliberationRecord) -> None:
    if not isinstance(record.deliberation_id, str) or not record.deliberation_id.strip():
        raise ValueError("deliberation_id must be a non-empty string")
    if "\n" in record.deliberation_id or "\r" in record.deliberation_id:
        raise ValueError("deliberation_id must not contain newlines")
    if not isinstance(record.topic, str) or not record.topic.strip():
        raise ValueError("topic must be a non-empty string")
    if record.council_tier not in VALID_TIERS:
        raise ValueError(
            f"council_tier={record.council_tier!r} not in {sorted(VALID_TIERS)}"
        )
    if not isinstance(record.council_attendees, (list, tuple)) or len(record.council_attendees) == 0:
        raise ValueError("council_attendees must be a non-empty sequence")
    for member in record.council_attendees:
        if not isinstance(member, str) or not member.strip():
            raise ValueError(f"council_attendees entries must be non-empty strings; got {member!r}")
    if record.council_verdict not in VALID_VERDICTS:
        raise ValueError(
            f"council_verdict={record.council_verdict!r} not in {sorted(VALID_VERDICTS)}"
        )
    # T2+ deliberations MUST have at least one assumption-adversary verdict
    # entry per CLAUDE.md Catalog #292 + "Council conduct" Fix-7 amendment.
    if record.council_tier in (CouncilTier.T2, CouncilTier.T3, CouncilTier.T4):
        if len(record.council_assumption_adversary_verdict) == 0:
            raise ValueError(
                f"council_tier={record.council_tier} requires at least one "
                "council_assumption_adversary_verdict entry (per Catalog #292 "
                "+ CLAUDE.md 'Council conduct' Fix-7 amendment)"
            )


def _record_to_dict(record: CouncilDeliberationRecord) -> dict[str, Any]:
    return {
        "schema": record.schema,
        "deliberation_id": record.deliberation_id,
        "topic": record.topic,
        "council_tier": record.council_tier,
        "council_attendees": list(record.council_attendees),
        "council_quorum_met": bool(record.council_quorum_met),
        "council_verdict": record.council_verdict,
        "council_dissent": [dict(d) for d in record.council_dissent],
        "council_assumption_adversary_verdict": [
            dict(d) for d in record.council_assumption_adversary_verdict
        ],
        "council_decisions_recorded": list(record.council_decisions_recorded),
        "related_deliberation_ids": list(record.related_deliberation_ids),
        "event_type": record.event_type,
        "parent_id_or_session": record.parent_id_or_session,
        "memory_path": record.memory_path,
        "notes": record.notes,
        "written_at_utc": record.written_at_utc or _now_utc_iso(),
        "written_pid": record.written_pid if record.written_pid is not None else os.getpid(),
        "written_host": record.written_host or socket.gethostname(),
    }


def _dict_to_record(payload: dict[str, Any]) -> CouncilDeliberationRecord:
    attendees = payload.get("council_attendees") or []
    dissent_raw = payload.get("council_dissent") or []
    adv_raw = payload.get("council_assumption_adversary_verdict") or []
    decisions_raw = payload.get("council_decisions_recorded") or []
    related_raw = payload.get("related_deliberation_ids") or []
    return CouncilDeliberationRecord(
        deliberation_id=str(payload["deliberation_id"]),
        topic=str(payload["topic"]),
        council_tier=str(payload["council_tier"]),
        council_attendees=tuple(str(m) for m in attendees),
        council_quorum_met=bool(payload.get("council_quorum_met", False)),
        council_verdict=str(payload["council_verdict"]),
        council_dissent=tuple(
            {str(k): str(v) for k, v in (d or {}).items()} for d in dissent_raw
        ),
        council_assumption_adversary_verdict=tuple(
            {str(k): str(v) for k, v in (d or {}).items()} for d in adv_raw
        ),
        council_decisions_recorded=tuple(str(s) for s in decisions_raw),
        related_deliberation_ids=tuple(str(s) for s in related_raw),
        event_type=str(payload.get("event_type", "dispatched")),
        parent_id_or_session=payload.get("parent_id_or_session"),
        memory_path=payload.get("memory_path"),
        notes=str(payload.get("notes", "")),
        written_at_utc=str(payload.get("written_at_utc", "")),
        written_pid=(
            int(payload["written_pid"]) if payload.get("written_pid") is not None else None
        ),
        written_host=payload.get("written_host"),
        schema=str(payload.get("schema", SCHEMA_VERSION)),
    )


def append_council_anchor(
    record: CouncilDeliberationRecord,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Append a council deliberation record to the JSONL posterior under
    fcntl LOCK_EX (Catalog #128 + #131 + #245 sister discipline).

    Concurrent appenders serialize at the lock; each writes exactly one
    JSON line followed by newline. Per Catalog #110 / #113 HISTORICAL_
    PROVENANCE: the JSONL is APPEND-ONLY — outcome rows for a prior
    deliberation reference the same ``deliberation_id`` rather than
    mutating the original row.

    Raises :class:`ValueError` on missing/invalid required fields. T2+
    records without ``council_assumption_adversary_verdict`` are
    refused per Catalog #292 + CLAUDE.md "Council conduct" Fix-7
    amendment.
    """
    _validate_record(record)
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    lock = lock_path or DEFAULT_COUNCIL_POSTERIOR_LOCK_PATH
    _ensure_state_dir(posterior)
    _ensure_state_dir(lock)
    payload = _record_to_dict(record)
    line = json.dumps(payload, sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with posterior.open("a", encoding="utf-8") as pf:
                pf.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def load_council_anchors(
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Lenient loader; skips malformed lines + wrong-schema rows.

    Mirrors :func:`tac.cost_band_calibration.load_anchors` semantics.
    Use the strict variant when a writer is about to MUTATE / REPLACE
    the file (so a partial corruption doesn't silently reset state).
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[CouncilDeliberationRecord] = []
    for raw in posterior.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") != SCHEMA_VERSION:
            continue
        try:
            out.append(_dict_to_record(payload))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def load_council_anchors_strict(
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Strict loader; raises :class:`CouncilPosteriorCorruptError` on parse
    failure. Mirror of :func:`tac.deploy.lightning.active_jobs_state.
    load_active_jobs_strict` per Catalog #138 fail-closed discipline.

    Writers about to MUTATE the file MUST use the strict variant so
    a partial corruption raises instead of silently dropping rows on
    next replace.
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[CouncilDeliberationRecord] = []
    for lineno, raw in enumerate(
        posterior.read_text(encoding="utf-8").splitlines(), start=1
    ):
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CouncilPosteriorCorruptError(
                f"line {lineno} of {posterior} failed JSON parse: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise CouncilPosteriorCorruptError(
                f"line {lineno} of {posterior} is not a dict"
            )
        if payload.get("schema") != SCHEMA_VERSION:
            # Unknown schema: skip (forward compat); strict-load refuses
            # only malformed rows, not legitimate schema-version skew.
            continue
        try:
            out.append(_dict_to_record(payload))
        except (KeyError, ValueError, TypeError) as exc:
            raise CouncilPosteriorCorruptError(
                f"line {lineno} of {posterior} failed schema construction: {exc}"
            ) from exc
    return out


def query_anchors_by_topic(
    topic_substring: str,
    *,
    posterior_path: Path | None = None,
) -> list[CouncilDeliberationRecord]:
    """Case-insensitive substring match on the ``topic`` field.

    Cite-chain helper sister of :func:`tac.deploy.modal.call_id_ledger.
    query_by_call_id`. Future Rashomon ensemble members and the
    Assumption-Adversary use this to trace position evolution on a
    given topic across prior deliberations.
    """
    if not topic_substring:
        return []
    needle = topic_substring.lower()
    anchors = load_council_anchors(posterior_path=posterior_path)
    return [a for a in anchors if needle in a.topic.lower()]


def query_dissent_history(
    member_name: str,
    *,
    posterior_path: Path | None = None,
) -> list[tuple[CouncilDeliberationRecord, dict[str, str]]]:
    """Surface every prior minority opinion by ``member_name``.

    Returns list of ``(record, dissent_entry)`` pairs. Maximum-signal
    preservation surface per CLAUDE.md "Council conduct" maximum-signal
    rule: future deliberations can trace which members have consistently
    flagged X as cargo-culted vs hard-earned.
    """
    if not member_name:
        return []
    needle = member_name.lower()
    anchors = load_council_anchors(posterior_path=posterior_path)
    matches: list[tuple[CouncilDeliberationRecord, dict[str, str]]] = []
    for anchor in anchors:
        for dissent in anchor.council_dissent:
            member_field = str(dissent.get("member", "")).lower()
            if needle in member_field:
                matches.append((anchor, dict(dissent)))
    return matches


def query_assumption_classification_history(
    assumption_substring: str,
    *,
    posterior_path: Path | None = None,
) -> list[tuple[CouncilDeliberationRecord, dict[str, str]]]:
    """Surface every prior HARD-EARNED-vs-CARGO-CULTED verdict for a
    given assumption substring.

    Returns list of ``(record, assumption_entry)`` pairs. Sister of
    :func:`query_dissent_history` at the assumption-classification
    surface. Future Assumption-Adversary deliberations consume this
    to trace classification stability (an assumption repeatedly
    classified HARD-EARNED across 5 deliberations is unlikely to be
    cargo-culted; an assumption flipping HARD-EARNED ↔ CARGO-CULTED is
    a red flag for the Assumption-Adversary to surface explicitly).
    """
    if not assumption_substring:
        return []
    needle = assumption_substring.lower()
    anchors = load_council_anchors(posterior_path=posterior_path)
    matches: list[tuple[CouncilDeliberationRecord, dict[str, str]]] = []
    for anchor in anchors:
        for entry in anchor.council_assumption_adversary_verdict:
            assumption_field = str(entry.get("assumption", "")).lower()
            if needle in assumption_field:
                matches.append((anchor, dict(entry)))
    return matches


# Canonical name for the continual-learning posterior update hook
# (Catalog #265 canonical-contract token + Catalog #125 hook 5).
def update_from_anchor(
    record: CouncilDeliberationRecord,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Alias of :func:`append_council_anchor` for the canonical contract.

    Catalog #265 (`check_symposium_impls_canonical_contract`) requires
    each module to expose ``update_from_anchor`` as the continual-
    learning hook surface (Catalog #125 hook 5). This module is not in
    ``src/tac/symposium_impls/`` but mirrors the same canonical contract
    so future Rashomon ensemble / autopilot consumers route through one
    well-known name regardless of which canonical helper they're hooking.
    """
    append_council_anchor(
        record,
        posterior_path=posterior_path,
        lock_path=lock_path,
    )
