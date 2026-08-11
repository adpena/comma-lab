# SPDX-License-Identifier: MIT
"""tac.verdicts.emit — the canonical verdict-emission helper.

:func:`emit_verdict` is the ONE way to write a verdict/decision JSON. It REFUSES a
verdict that omits a required honesty field, so the standing hooks never have to
fire for a hand-rolled JSON again:

  * ``scope`` (a :class:`VerdictScope`)          — the verdict-scope ladder
        INSTANCE < FORMULATION < FAMILY < PARADIGM + a ``scoped_to`` string
        (operator 2026-07-08: "one failure does not mean family dead"). Missing ⇒
        refuse.
  * ``rows`` (>= 1 :class:`~tac.verdicts.MeasurementRow`) for a MEASURED claim —
        a measured verdict with zero measurement rows is a claim without evidence.
  * ``composition`` (a :class:`Composition`)      — the interaction sign with the
        active lever set, OR an explicit ``deferred_to_ab_protocol`` (P12, operator
        2026-07-09: "synergies and antagonisms are first-class"). Missing ⇒ refuse.
  * ``constraint_carved``                          — what region of design space this
        result removes/pins (P10, operator 2026-07-09: "negatives carve the
        optimum"). Missing ⇒ refuse.
  * a NEGATIVE verdict additionally requires a ``reformulation_queue`` — the
        untested alternatives — which may be explicitly EMPTY with a reason (a
        genuine family-level exhaustion), never silently absent.

Write is atomic (tmp + ``os.replace``) so a crash mid-write never leaves a
truncated verdict JSON.

:func:`verdict_payload` is the EMBEDDABLE half — the same validation, returning the
payload dict instead of writing a file — for producers that must carry the verdict
INSIDE a larger artifact. It exists because the write-only API was measured (task #936,
2026-08-03) to be the reason adoption failed: 474 of 10,728 tracked ``.py`` files
hand-roll a ``"verdict_scope"`` key while ``emit_verdict`` had 0 production callers.
``emit_verdict`` is a thin write-wrapper around it, so the two cannot drift.

#389 WIRED: on a successful atomic write, :func:`emit_verdict` posts a
``verdict_landed`` bulletin event to the session_bus (fail-open, like
``tac.review_counter.record_round``) so sibling agents learn a verdict LANDED
without re-grepping the repo. The session_bus import is LAZY (inside the
fail-open helper), so this module's IMPORT-TIME dependencies stay stdlib +
tac.verdicts only — the emitted JSON write is unchanged and byte-identical.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from numbers import Real
from pathlib import Path

from tac.verdicts.measurement_row import MeasurementRow

__all__ = [
    "SCHEMA_VERSION",
    "Composition",
    "InstrumentClaimScope",
    "InteractionSign",
    "ScopeLevel",
    "VerdictEmitError",
    "VerdictScope",
    "emit_verdict",
    "verdict_payload",
]

# Bump when the on-disk JSON shape changes in a consumer-breaking way.
SCHEMA_VERSION = "verdict.v1"


class VerdictEmitError(ValueError):
    """Raised when :func:`emit_verdict` is called with a claim that omits a
    required honesty field. Subclasses ``ValueError``."""


@dataclass(frozen=True, slots=True)
class InstrumentClaimScope:
    """Capacity cross-check for an object-level claim.

    Both capacities are expressed in ``claim_units``.  Construction refuses
    when the claimed object is richer than the instrument that produced the
    verdict, so no caller can accidentally serialize an object-level closure
    from an instrument-scoped probe.
    """

    instrument_capacity: int | float
    object_capacity: int | float
    claim_units: str
    scope_ok: bool = field(init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("instrument_capacity", self.instrument_capacity),
            ("object_capacity", self.object_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise VerdictEmitError(f"{label} must be a finite non-negative number")
            if not math.isfinite(float(value)) or value < 0:
                raise VerdictEmitError(f"{label} must be a finite non-negative number")
        if not isinstance(self.claim_units, str) or not self.claim_units.strip():
            raise VerdictEmitError("claim_units must be a non-empty string")
        expected = self.object_capacity <= self.instrument_capacity
        object.__setattr__(self, "scope_ok", expected)
        if not expected:
            raise VerdictEmitError(
                "REFUSE verdict emission: object claim exceeds instrument capacity "
                f"in {self.claim_units}"
            )

    def to_json_dict(self) -> dict:
        return {
            "instrument_capacity": self.instrument_capacity,
            "object_capacity": self.object_capacity,
            "claim_units": self.claim_units,
            "scope_ok": self.scope_ok,
        }


class ScopeLevel(StrEnum):
    """The verdict-scope ladder — default to the NARROWEST the evidence supports.

    A naive/toy/binary measurement can only produce an ``INSTANCE`` negative; a
    ``FAMILY`` kill needs a theorem/citation OR failures across >= 2 structurally
    distinct formulations (enforced by the emit-time family-evidence rule).
    """

    INSTANCE = "instance"
    FORMULATION = "formulation"
    FAMILY = "family"
    PARADIGM = "paradigm"

    @property
    def rank(self) -> int:
        return {"instance": 0, "formulation": 1, "family": 2, "paradigm": 3}[self.value]

    @classmethod
    def coerce(cls, value: ScopeLevel | str) -> ScopeLevel:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        raise VerdictEmitError(
            f"scope level must be a ScopeLevel or one of "
            f"{[m.value for m in cls]!r}; got {value!r}"
        )


@dataclass(frozen=True)
class VerdictScope:
    """A verdict-scope declaration: a ladder ``level`` + a ``scoped_to`` string.

    ``scoped_to`` names the SPECIFIC thing the verdict binds to at that level (the
    one config / the one formulation / the family + its evidence). For a FAMILY (or
    PARADIGM) scope, ``family_evidence`` MUST be supplied — a citation or an
    explicit ">= 2 distinct formulations" statement — else construction refuses.
    """

    level: ScopeLevel
    scoped_to: str
    family_evidence: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", ScopeLevel.coerce(self.level))
        if not isinstance(self.scoped_to, str) or not self.scoped_to.strip():
            raise VerdictEmitError(
                "VerdictScope.scoped_to must name the specific thing the verdict "
                "binds to (non-empty string)"
            )
        if self.level in (ScopeLevel.FAMILY, ScopeLevel.PARADIGM):
            if not (isinstance(self.family_evidence, str) and self.family_evidence.strip()):
                raise VerdictEmitError(
                    f"VerdictScope level '{self.level.value}' REQUIRES family_evidence "
                    "— a citation (arXiv/DOI/theorem/impossibility bound) OR an "
                    "explicit '>= 2 structurally distinct formulations' statement "
                    "(a family kill needs a theorem or kills across >= 2 formulations)"
                )
        elif self.family_evidence is not None and not isinstance(self.family_evidence, str):
            raise VerdictEmitError("VerdictScope.family_evidence must be a string or None")

    def to_json_dict(self) -> dict:
        return {
            "level": self.level.value,
            "scoped_to": self.scoped_to,
            "family_evidence": self.family_evidence,
        }


class InteractionSign(StrEnum):
    """The measured/expected composition sign with the active lever set (P12)."""

    ADDITIVE = "additive"
    SYNERGISTIC = "synergistic"
    ANTAGONISTIC = "antagonistic"
    ORTHOGONAL = "orthogonal"
    REDUNDANT = "redundant"

    @classmethod
    def coerce(cls, value: InteractionSign | str) -> InteractionSign:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        raise VerdictEmitError(
            f"interaction sign must be an InteractionSign or one of "
            f"{[m.value for m in cls]!r}; got {value!r}"
        )


@dataclass(frozen=True)
class Composition:
    """The P12 composition row: how this lever interacts with the ACTIVE set.

    Two legal forms:
      * a stated interaction ``sign`` + the ``active_levers`` it composes with
        (empty tuple == ships standalone — a legitimate declaration), plus whether
        the sign was ``measured`` (paired arms) or expected;
      * ``deferred_to_ab_protocol=True`` — the interaction is genuinely not yet
        measured and is queued to the A/B protocol (an honest deferral, not a
        silent omission).

    Use :meth:`deferred` for the deferral form; the default constructor is the
    stated-sign form.
    """

    sign: InteractionSign | None = None
    active_levers: tuple[str, ...] = ()
    measured: bool = False
    deferred_to_ab_protocol: bool = False

    def __post_init__(self) -> None:
        if self.deferred_to_ab_protocol:
            if self.sign is not None:
                raise VerdictEmitError(
                    "Composition: deferred_to_ab_protocol=True cannot also carry a "
                    "sign — a deferral means the sign is NOT yet known"
                )
            return
        if self.sign is None:
            raise VerdictEmitError(
                "Composition REQUIRES either a sign (with active_levers) OR "
                "deferred_to_ab_protocol=True (P12: no lever verdict is complete "
                "without its composition row)"
            )
        object.__setattr__(self, "sign", InteractionSign.coerce(self.sign))
        if not isinstance(self.active_levers, tuple):
            object.__setattr__(self, "active_levers", tuple(self.active_levers))
        if any(not (isinstance(x, str) and x.strip()) for x in self.active_levers):
            raise VerdictEmitError(
                "Composition.active_levers must be non-empty strings"
            )

    @classmethod
    def deferred(cls) -> Composition:
        """The honest 'interaction not yet measured — queued to A/B' form."""
        return cls(deferred_to_ab_protocol=True)

    def to_json_dict(self) -> dict:
        return {
            "sign": self.sign.value if self.sign is not None else None,
            "active_levers": list(self.active_levers),
            "measured": self.measured,
            "deferred_to_ab_protocol": self.deferred_to_ab_protocol,
        }


def _coerce_composition(composition: Composition | str | None) -> Composition:
    if isinstance(composition, Composition):
        return composition
    if composition == "deferred_to_ab_protocol":
        return Composition.deferred()
    raise VerdictEmitError(
        "composition is REQUIRED (P12): pass a tac.verdicts.Composition, or the "
        "literal 'deferred_to_ab_protocol'. A lever verdict without its "
        f"interaction sign is incomplete. Got {composition!r}"
    )


def verdict_payload(
    *,
    verdict: str,
    scope: VerdictScope,
    rows: Sequence[MeasurementRow] = (),
    composition: Composition | str | None = None,
    constraint_carved: str,
    is_negative: bool = False,
    reformulation_queue: Sequence[str] | None = None,
    reformulation_empty_reason: str | None = None,
    measured: bool = True,
    stores_consulted: Sequence[str] | None = None,
    instrument_claim_scope: InstrumentClaimScope | None = None,
    extra: dict | None = None,
) -> dict:
    """Validate a verdict and return its canonical payload dict — WITHOUT writing.

    This is the EMBEDDABLE half of the canonical surface, and it exists because of a
    measured adoption failure (task #936, 2026-08-03): ``"verdict_scope"`` is hand-rolled
    as a raw dict key in **474 of 10,728 tracked .py files**, while :func:`emit_verdict`
    has **0 production call sites**. The discipline was adopted almost universally; the
    producer was bypassed almost universally. The mechanism is not ignorance — e.g.
    ``canonical_equations/v8_geometric_rate_decomposition_20260709.py`` hand-rolls a
    correctly-scoped FORMULATION verdict WITH a populated reformulation queue — it is
    that ``emit_verdict`` offered only a whole-file atomic write, so a producer needing
    the verdict EMBEDDED inside a larger artifact (an equation payload, a batch summary,
    a receipt) had no path to the typed validation at all.

    Same refusal semantics as :func:`emit_verdict` (that function is now a thin
    write-wrapper around this one, so the two can never drift): a FAMILY/PARADIGM scope
    without ``family_evidence`` refuses, a MEASURED claim with no rows refuses, and a
    negative without a reformulation queue refuses.

    Returns the payload dict; embed it under whatever key the host artifact uses.
    """
    return _build_verdict_payload(
        verdict=verdict,
        scope=scope,
        rows=rows,
        composition=composition,
        constraint_carved=constraint_carved,
        is_negative=is_negative,
        reformulation_queue=reformulation_queue,
        reformulation_empty_reason=reformulation_empty_reason,
        measured=measured,
        stores_consulted=stores_consulted,
        instrument_claim_scope=instrument_claim_scope,
        extra=extra,
    )


def emit_verdict(
    path: str | os.PathLike,
    *,
    verdict: str,
    scope: VerdictScope,
    rows: Sequence[MeasurementRow] = (),
    composition: Composition | str | None = None,
    constraint_carved: str,
    is_negative: bool = False,
    reformulation_queue: Sequence[str] | None = None,
    reformulation_empty_reason: str | None = None,
    measured: bool = True,
    stores_consulted: Sequence[str] | None = None,
    instrument_claim_scope: InstrumentClaimScope | None = None,
    extra: dict | None = None,
) -> Path:
    """Atomically write a validated verdict JSON to ``path``; return the ``Path``.

    Use :func:`verdict_payload` instead when the verdict must be EMBEDDED in a larger
    artifact rather than written as its own file — same validation, no write.

    Parameters
    ----------
    verdict:
        The disposition text (e.g. ``"NO-GO"``, ``"PROCEED"``, ``"DEFER"``). Non-empty.
    scope:
        A :class:`VerdictScope` (ladder level + ``scoped_to``). REQUIRED — a verdict
        without a declared scope is refused.
    rows:
        The :class:`~tac.verdicts.MeasurementRow` evidence. A ``measured`` claim
        (the default) requires >= 1 row.
    composition:
        REQUIRED (P12). A :class:`Composition` (interaction sign + active set) or
        the literal ``"deferred_to_ab_protocol"``.
    constraint_carved:
        REQUIRED (P10). What region of design space this result removes / pins.
    is_negative:
        Set for a KILL/NO-GO/negative verdict; it additionally requires a
        ``reformulation_queue`` (possibly explicitly empty-with-reason).
    reformulation_queue:
        The untested alternative formulations. Required when ``is_negative``. Pass
        an empty sequence together with ``reformulation_empty_reason`` for a genuine
        exhaustion (never a silent absence).
    measured:
        Whether this verdict rests on measurement (default True). Set False for a
        pure design/DEFER verdict that legitimately carries no measurement rows.

    Raises
    ------
    VerdictEmitError:
        If any required honesty field is missing/malformed.
    """
    payload = _build_verdict_payload(
        verdict=verdict,
        scope=scope,
        rows=rows,
        composition=composition,
        constraint_carved=constraint_carved,
        is_negative=is_negative,
        reformulation_queue=reformulation_queue,
        reformulation_empty_reason=reformulation_empty_reason,
        measured=measured,
        stores_consulted=stores_consulted,
        instrument_claim_scope=instrument_claim_scope,
        extra=extra,
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        tmp.write_text(text)
        os.replace(tmp, out)
    finally:
        # If os.replace already consumed tmp this is a no-op; on failure it cleans up.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    # #389: broadcast the landing AFTER the durable write (fail-open). Read back from the
    # validated payload so the bulletin can never describe a different verdict than the
    # one on disk.
    _post_bulletin_verdict_landed(
        out,
        payload["verdict"],
        payload["scope"]["scoped_to"],
        payload["scope"]["level"],
        payload["composition"]["sign"],
        payload["is_negative"],
        len(payload["rows"]),
    )
    return out


def _build_verdict_payload(
    *,
    verdict: str,
    scope: VerdictScope,
    rows: Sequence[MeasurementRow],
    composition: Composition | str | None,
    constraint_carved: str,
    is_negative: bool,
    reformulation_queue: Sequence[str] | None,
    reformulation_empty_reason: str | None,
    measured: bool,
    stores_consulted: Sequence[str] | None,
    instrument_claim_scope: InstrumentClaimScope | None,
    extra: dict | None,
) -> dict:
    """The ONE validation + payload path shared by :func:`verdict_payload` and
    :func:`emit_verdict`, so the embeddable and file-writing surfaces cannot drift."""
    # --- verdict text ---
    if not isinstance(verdict, str) or not verdict.strip():
        raise VerdictEmitError("verdict must be a non-empty string")

    # --- scope (verdict-scope ladder) ---
    if not isinstance(scope, VerdictScope):
        raise VerdictEmitError(
            "scope is REQUIRED and must be a tac.verdicts.VerdictScope "
            "(ladder level INSTANCE<FORMULATION<FAMILY<PARADIGM + a scoped_to "
            f"string). Got {type(scope).__name__}"
        )

    # --- constraint_carved (P10) ---
    if not (isinstance(constraint_carved, str) and constraint_carved.strip()):
        raise VerdictEmitError(
            "constraint_carved is REQUIRED (P10 'negatives carve the optimum'): "
            "state what region of design space this result removes or pins"
        )

    # --- composition (P12) ---
    comp = _coerce_composition(composition)

    # --- rows: a measured claim needs evidence ---
    rows = tuple(rows)
    for r in rows:
        if not isinstance(r, MeasurementRow):
            raise VerdictEmitError(
                "every element of rows must be a tac.verdicts.MeasurementRow; "
                f"got {type(r).__name__}"
            )
    if measured and not rows:
        raise VerdictEmitError(
            "a MEASURED verdict requires >= 1 MeasurementRow (a measured claim "
            "with no measurement rows is a claim without evidence). For a pure "
            "design/DEFER verdict with no measurement, pass measured=False."
        )

    # --- negatives require a reformulation queue (may be explicitly empty) ---
    reformulation_list: list[str] | None = None
    if is_negative:
        if reformulation_queue is None:
            raise VerdictEmitError(
                "a NEGATIVE verdict REQUIRES a reformulation_queue — the untested "
                "alternative formulations (one failed formulation is NOT a dead "
                "family). Pass an empty list WITH reformulation_empty_reason for a "
                "genuine exhaustion; never omit it."
            )
        reformulation_list = [str(x) for x in reformulation_queue]
        if not reformulation_list and not (
            isinstance(reformulation_empty_reason, str) and reformulation_empty_reason.strip()
        ):
            raise VerdictEmitError(
                "reformulation_queue is empty but reformulation_empty_reason is not "
                "given — an empty queue must state WHY the alternatives are "
                "exhausted (typically requires FAMILY-scope evidence)."
            )

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "emitted_at_utc": now,
        "verdict": verdict,
        "measured": bool(measured),
        "scope": scope.to_json_dict(),
        "composition": comp.to_json_dict(),
        "constraint_carved": constraint_carved,
        "is_negative": bool(is_negative),
        "reformulation_queue": reformulation_list,
        "reformulation_empty_reason": reformulation_empty_reason if is_negative else None,
        "rows": [r.to_json_dict() for r in rows],
        "stores_consulted": list(stores_consulted) if stores_consulted else [],
        "extra": dict(extra) if extra else {},
    }
    if instrument_claim_scope is not None:
        if not isinstance(instrument_claim_scope, InstrumentClaimScope):
            raise VerdictEmitError(
                "instrument_claim_scope must be an InstrumentClaimScope or None"
            )
        payload.update(instrument_claim_scope.to_json_dict())
    return payload


def _post_bulletin_verdict_landed(
    out: Path,
    verdict: str,
    scoped_to: str,
    scope_level: str,
    composition_sign: str | None,
    is_negative: bool,
    n_rows: int,
) -> None:
    """Broadcast a ``verdict_landed`` bulletin event so sibling agents see this verdict.

    FAIL-OPEN by design (mirrors ``tac.review_counter._post_bulletin_verdict_landed``):
    the verdict JSON on disk is the authority; the bulletin is score-neutral
    observability. A bulletin failure — missing package, disk full, lock contention —
    must NEVER break the (already-completed) verdict write, so any exception is
    swallowed. This is an intentional silent guard on a non-authoritative notification
    path AFTER the durable write, not on a safety/score surface (operating manual §8.9).
    The ``subject`` is the payload's ``scope.scoped_to`` — the thing the verdict binds to — so a
    seal-round ``staleness_check`` on that surface picks the landing up.
    """
    try:
        from tac.session_bus.bulletin import EVENT_VERDICT_LANDED, post_event

        post_event(
            kind=EVENT_VERDICT_LANDED,
            subject=scoped_to,
            payload={
                "verdict": verdict,
                "scope_level": scope_level,
                "is_negative": bool(is_negative),
                "n_rows": int(n_rows),
                "composition_sign": composition_sign,
                "path": str(out),
            },
            agent_label="emit_verdict",
        )
    except Exception:
        pass
