# SPDX-License-Identifier: MIT
"""Rivalry rows on race receipts — the RECORDING side of the built-elsewhere-unwired class.

ddm_gd5 (task #864) built the auto-derived grade-5 detector, measured three formulations against
real controls, and refuted all three. Its blocking finding, restated exactly:

    The "measured-better successor" relation — the conjunct that makes the class harmful — exists
    only in memos and receipts. It has NO representation in code. A component that has never been
    wired anywhere leaves no trace of being a rival to anything, so there is no edge, no slot
    value, and no registry row for a static analyser to walk.

So the cure is on the PRODUCING side, not the detecting side. We already run races and already
write receipts; what they do not carry is the rival relation. This module is the field-set gd5
named, recorded at the moment a race is decided — while the information is in hand and free:

    role          the slot/capability being filled, as a stable id (e.g. "token_scan_order")
    incumbent     the identifier the LIVE path currently uses
    challenger    the identifier that was raced against it
    metric        the axis compared, with its surface/axis label
    delta         the measured comparison, both values, with units
    adopted       true | false — and if false, WHY

Then the detector is ONE JOIN, :func:`unadopted_better_challengers`: rows where the challenger
beats the incumbent AND was not adopted. No orphan registry, no semantic judgement, and the harm
is QUANTIFIED at detection time — the thing neither of gd5's refuted formulations could supply.

TWO honest limits, stated here so no reader has to rediscover them:

1. This only ever sees races we RUN AND RECORD. A better successor nobody raced stays invisible,
   exactly as today. It converts "invisible to everything" into "visible iff raced" — a real
   improvement, not a solution, and NOT a closure of the #864 P0.
2. A rivalry row is EVIDENCE, never an adoption. Recording that a challenger won does not wire it,
   and :func:`unadopted_better_challengers` returning a row is a QUESTION for an owner, not a
   verdict. Per CLAUDE.md's forbidden-premature-KILL rule, ``not_adopted_reason`` carries the
   status (queued / refused-because / blocked-by-<named>), never a silent drop.

Direction is EXPLICIT (``lower_is_better``) because "better" is not derivable from a number: bytes
and distortion fall, throughput rises, and a gate that guessed would invert half its verdicts.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RivalryRow",
    "read_rivalry_rows",
    "rivalry_rows_from_arms",
    "unadopted_better_challengers",
]

_PLACEHOLDER = {"", "<reason>", "<rationale>", "tbd", "todo", "placeholder", "n/a", "none"}


@dataclass(frozen=True)
class RivalryRow:
    """One decided race: an incumbent, a challenger, a measured comparison, and a disposition.

    Every field is mandatory-by-refusal except ``not_adopted_reason``, which is mandatory exactly
    when ``adopted`` is False. A row missing any of them cannot answer the question the row exists
    to answer, which is the orphan this schema extincts.
    """

    role: str
    incumbent: str
    challenger: str
    metric: str
    axis: str
    unit: str
    incumbent_value: float
    challenger_value: float
    lower_is_better: bool = True
    adopted: bool = False
    not_adopted_reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("role", "incumbent", "challenger", "metric", "axis", "unit"):
            val = getattr(self, name)
            if not isinstance(val, str) or len(val.strip()) < 1:
                raise ValueError(
                    f"RivalryRow.{name} is required and must be a non-empty str; got {val!r} — a "
                    "rivalry row without it cannot be joined against the live path, which is the "
                    "whole reason this schema exists")
        for name in ("incumbent_value", "challenger_value"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"RivalryRow.{name} must be a real number; got {val!r}")
        if self.incumbent == self.challenger:
            raise ValueError(
                f"incumbent and challenger are the same identifier ({self.incumbent!r}); a row "
                "comparing something to itself records no rivalry")
        if not self.adopted and self.not_adopted_reason.strip().lower() in _PLACEHOLDER:
            raise ValueError(
                "adopted=False requires a substantive not_adopted_reason (queued / refused-because"
                "-<measured fact> / blocked-by-<named blocker>). An unadopted winner with no reason "
                "is precisely the silently-orphaned successor this schema exists to make visible.")

    @property
    def delta(self) -> float:
        """challenger - incumbent, in ``unit``. Derived, never stored: one fact, one key."""
        return float(self.challenger_value) - float(self.incumbent_value)

    @property
    def challenger_wins(self) -> bool:
        """Did the challenger BEAT the live incumbent on this metric? Ties are not wins."""
        return self.delta < 0 if self.lower_is_better else self.delta > 0

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "incumbent": self.incumbent,
            "challenger": self.challenger,
            "metric": self.metric,
            "axis": self.axis,
            "unit": self.unit,
            "incumbent_value": self.incumbent_value,
            "challenger_value": self.challenger_value,
            "delta": self.delta,
            "lower_is_better": self.lower_is_better,
            "challenger_wins": self.challenger_wins,
            "adopted": self.adopted,
            "not_adopted_reason": self.not_adopted_reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, row: dict) -> RivalryRow:
        """Rebuild from a receipt. ``delta``/``challenger_wins`` are RECOMPUTED, never trusted.

        A stored derived value that disagrees with its inputs is a stale-artifact confound; the
        reader recomputes so a hand-edited receipt cannot lie to the join.
        """
        return cls(
            role=row["role"], incumbent=row["incumbent"], challenger=row["challenger"],
            metric=row["metric"], axis=row["axis"], unit=row["unit"],
            incumbent_value=row["incumbent_value"], challenger_value=row["challenger_value"],
            lower_is_better=bool(row.get("lower_is_better", True)),
            adopted=bool(row.get("adopted", False)),
            not_adopted_reason=row.get("not_adopted_reason", "") or "",
            notes=row.get("notes", "") or "",
        )


def rivalry_rows_from_arms(
    *,
    role: str,
    incumbent_arm: str,
    arms: dict[str, dict],
    value_key: str,
    metric: str,
    axis: str,
    unit: str,
    not_adopted_reason: str,
    adopted_arm: str | None = None,
    lower_is_better: bool = True,
    notes: str = "",
) -> list[RivalryRow]:
    """Build one row per challenger arm of a same-payload race against its incumbent arm.

    The common shape of every race harness we run: a dict of ``{arm_name: {value_key: number}}``
    where exactly one arm IS the live path. Deriving the rows here rather than hand-writing them
    per tool keeps the relation machine-readable and stops each race from inventing its own words
    for "who won".
    """
    if incumbent_arm not in arms:
        raise ValueError(
            f"incumbent arm {incumbent_arm!r} is not among the raced arms {sorted(arms)}; the "
            "incumbent must itself be raced or the comparison has no shared control")
    base = arms[incumbent_arm][value_key]
    rows: list[RivalryRow] = []
    for name in sorted(arms):
        if name == incumbent_arm:
            continue
        adopted = adopted_arm == name
        rows.append(RivalryRow(
            role=role, incumbent=incumbent_arm, challenger=name, metric=metric, axis=axis,
            unit=unit, incumbent_value=base, challenger_value=arms[name][value_key],
            lower_is_better=lower_is_better, adopted=adopted,
            not_adopted_reason="" if adopted else not_adopted_reason, notes=notes,
        ))
    return rows


def unadopted_better_challengers(rows: list[RivalryRow]) -> list[RivalryRow]:
    """THE JOIN: challengers that BEAT the live incumbent and were not adopted.

    Sorted by margin, largest first, so the queue is ranked by measured harm rather than by guess.
    An empty result is a real and common answer — it means every race we recorded was won by the
    thing already live — and it must be reported with its denominator, never as a bare PASS.
    """
    hits = [r for r in rows if r.challenger_wins and not r.adopted]
    return sorted(hits, key=lambda r: abs(r.delta), reverse=True)


def read_rivalry_rows(receipt: str | Path) -> list[RivalryRow]:
    """Read the ``rivalry`` block of a race receipt. Missing block -> empty list, not an error.

    Most receipts predate this schema; treating their absence as a failure would make the join
    unrunnable instead of merely incomplete. The honest denominator is reported by the caller.

    READ TOTAL-BUT-LOUD. A malformed row is SKIPPED with a warning, never allowed to take down the
    read. This is not defensive habit — it is a named recurrence in this campaign: one row of 395
    that violated its own field contract raised inside the graph-memory reader and took out ALL
    fused-recall and graph queries campaign-wide, losing 100% of recall to 0.25% bad data. A join
    that dies on one bad row is a join nobody can rely on. Skipping SILENTLY would be the opposite
    error (a receipt of entirely malformed rows would read as "no rivalry", indistinguishable from
    "no race ever run"), so every skip warns.
    """
    p = Path(receipt)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        warnings.warn(f"race receipt {p} is unreadable, treating as no rivalry rows: {exc}",
                      RuntimeWarning, stacklevel=2)
        return []
    if not isinstance(payload, dict):
        warnings.warn(f"race receipt {p} is not a JSON object; no rivalry rows read",
                      RuntimeWarning, stacklevel=2)
        return []
    rows: list[RivalryRow] = []
    for i, raw in enumerate(payload.get("rivalry", [])):
        if not isinstance(raw, dict):
            warnings.warn(f"{p}: rivalry[{i}] is not an object; skipped",
                          RuntimeWarning, stacklevel=2)
            continue
        try:
            rows.append(RivalryRow.from_dict(raw))
        except (KeyError, TypeError, ValueError) as exc:
            warnings.warn(f"{p}: rivalry[{i}] is malformed and was SKIPPED ({exc}); the join is "
                          "reading fewer rows than the receipt claims",
                          RuntimeWarning, stacklevel=2)
    return rows
