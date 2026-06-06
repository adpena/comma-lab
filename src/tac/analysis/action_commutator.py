# SPDX-License-Identifier: MIT
"""Pairwise non-commutativity (commutator) ledger over ActionEffect v1 rows.

PR110-style score programs compose many evaluator actions (HiNeRV target-region
births, pair-local servo lifts, frame-0 selector perturbations).  Whether two
actions COMPOSE is a measured property, not an assumption: applying ``a`` then
``b`` to the same base archive need not move the contest score by
``delta(a) + delta(b)``.  The signed gap

    comm = delta_total(ab) - delta_total(a) - delta_total(b)

is the action commutator.  It is < 0 when the two actions are SYNERGISTIC (the
composite saved MORE score than the sum of parts → a macro-action candidate) and
> 0 when they CONFLICT (the composite gave back score the parts promised
independently → a conflict pair).  ``comm == 0`` (within ``eps``) is additive.

This module is THIN and analysis-only.  It does not orchestrate anything, does
not patch scoring, and does not promote: every ``comm`` value it emits is
derived from three real, measured :class:`tac.analysis.action_effect.ActionEffect`
rows (or, where a measured composite is absent, it emits a typed
needs-measurement row — it NEVER fabricates a commutator value).  It reuses the
single shared scoring computation already landed on ``ActionEffect`` (whose
``delta_score_total`` / ``delta_score_nonrate`` come from
``tac.score_geometry.contest_score``); it adds NO second drifting objective.

Two invariants make the arithmetic trustworthy:

* **Basis consistency** — a row's ``delta_score_total`` is ``None`` when archive
  bytes are unknown (a distortion-only observation).  We MUST NOT subtract a
  total from a nonrate (different units, different terms).  So if ANY of the
  three rows lacks ``delta_score_total`` we fall back to ``delta_score_nonrate``
  for ALL THREE and label ``basis="nonrate"``.  If a row also lacks the nonrate
  delta (no distortion endpoints AND no bytes) the commutator is undefined and
  :func:`commutator_value` raises.
* **Authority is a type** — composing actions across different authority
  surfaces (e.g. a ``fakequant_mlx`` birth with an ``inflated_torch_cpu`` servo)
  is a category error: the deltas were measured on different evaluator surfaces.
  All three rows must share one ``authority`` or :func:`commutator_value`
  raises.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tac.analysis.action_effect import ActionEffect
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

ACTION_COMMUTATOR_SCHEMA = "tac.action_commutator.v1"
ACTION_COMMUTATOR_LEDGER_SCHEMA = "tac.action_commutator_ledger.v1"
ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA = "tac.action_commutator_needs_measurement.v1"

#: Default magnitude band (score units) inside which a commutator is treated as
#: additive (neither synergistic nor conflicting).  Small but non-zero so exact
#: floating-point cancellation noise on identical replays reads as additive.
DEFAULT_COMMUTATOR_EPS = 1e-9

# Classifications.
CLASSIFICATION_SYNERGISTIC = "synergistic"
CLASSIFICATION_CONFLICTING = "conflicting"
CLASSIFICATION_ADDITIVE = "additive"

# Composition bases.
BASIS_TOTAL = "total"
BASIS_NONRATE = "nonrate"


class ActionCommutatorError(ValueError):
    """Raised when commutator inputs are structurally incompatible.

    The two structural failures are (a) an authority mismatch across the three
    rows (authority is a type) and (b) an undefined delta in the chosen basis
    (no total AND no nonrate to subtract).
    """


@dataclass(frozen=True)
class _Triple:
    """Resolved (a, b, ab) deltas in one consistent basis."""

    basis: str
    delta_a: float
    delta_b: float
    delta_ab: float


def _effect_action_id(effect: ActionEffect) -> str:
    if not isinstance(effect, ActionEffect):
        raise TypeError("effect must be an ActionEffect")
    return effect.action_id


def _resolve_authority(
    effect_a: ActionEffect,
    effect_b: ActionEffect,
    effect_ab: ActionEffect,
) -> str:
    """Return the single shared authority, or raise on mismatch.

    Authority is a type: the three rows must have been measured on the SAME
    evaluator surface for their deltas to be comparable.  Empty/whitespace
    authority is structurally rejected (the ``ActionEffect`` constructor already
    forbids it, but a defensive check keeps the error specific here).
    """

    authorities = tuple(str(e.authority).strip() for e in (effect_a, effect_b, effect_ab))
    if any(not a for a in authorities):
        raise ActionCommutatorError("each ActionEffect must carry a non-empty authority")
    if len(set(authorities)) != 1:
        raise ActionCommutatorError(
            "authority mismatch across commutator rows (authority is a type): "
            f"a={authorities[0]!r} b={authorities[1]!r} ab={authorities[2]!r}"
        )
    return authorities[0]


def _resolve_triple(
    effect_a: ActionEffect,
    effect_b: ActionEffect,
    effect_ab: ActionEffect,
) -> _Triple:
    """Resolve the three deltas in ONE consistent basis.

    Prefer ``delta_score_total`` (full nonrate+rate movement).  If ANY of the
    three rows lacks a total (bytes unknown ⇒ distortion-only row), fall back to
    ``delta_score_nonrate`` for ALL THREE so units never mix.  If a row lacks the
    chosen-basis delta entirely the commutator is undefined.
    """

    totals = tuple(e.delta_score_total for e in (effect_a, effect_b, effect_ab))
    if all(t is not None and math.isfinite(float(t)) for t in totals):
        return _Triple(BASIS_TOTAL, float(totals[0]), float(totals[1]), float(totals[2]))

    nonrates = tuple(e.delta_score_nonrate for e in (effect_a, effect_b, effect_ab))
    if all(n is not None and math.isfinite(float(n)) for n in nonrates):
        return _Triple(BASIS_NONRATE, float(nonrates[0]), float(nonrates[1]), float(nonrates[2]))

    # Neither basis is fully defined across the three rows.  Report which basis
    # was attempted and which row(s) were undefined so the failure is actionable.
    labels = ("a", "b", "ab")
    missing_total = [labels[i] for i, t in enumerate(totals) if t is None or not math.isfinite(float(t))]
    missing_nonrate = [labels[i] for i, n in enumerate(nonrates) if n is None or not math.isfinite(float(n))]
    raise ActionCommutatorError(
        "commutator undefined: no consistent basis. "
        f"delta_score_total missing/non-finite for {missing_total}; "
        f"delta_score_nonrate missing/non-finite for {missing_nonrate}"
    )


def _classify(comm: float, eps: float) -> str:
    if comm < -eps:
        return CLASSIFICATION_SYNERGISTIC
    if comm > eps:
        return CLASSIFICATION_CONFLICTING
    return CLASSIFICATION_ADDITIVE


def commutator_value(
    effect_a: ActionEffect,
    effect_b: ActionEffect,
    effect_ab: ActionEffect,
    *,
    eps: float = DEFAULT_COMMUTATOR_EPS,
) -> dict[str, Any]:
    """Measure the non-additivity of composing ``effect_a`` and ``effect_b``.

    ``effect_ab`` is the MEASURED composite (``b`` applied after ``a`` on the
    same base archive).  Returns a typed mapping::

        comm = delta(ab) - delta(a) - delta(b)

    where ``delta`` is ``delta_score_total`` when all three rows are byte-priced,
    else ``delta_score_nonrate`` for all three (``basis`` records which).

    The returned dict carries:

    * ``comm`` — the signed commutator (score units; negative ⇒ synergistic).
    * ``synergy_score_units`` — ``-comm`` (positive ⇒ the composite beat the sum
      of parts; convenient for "rank by synergy").
    * ``classification`` — ``synergistic`` (``comm < -eps``) /
      ``conflicting`` (``comm > +eps``) / ``additive`` (otherwise).
    * ``basis`` — ``total`` or ``nonrate`` (the consistent unit used).
    * ``authority`` — the single shared authority (the deltas' measurement
      surface).
    * ``macro_action_recommended`` — ``classification == synergistic``.
    * the canonical false-authority markers (this is a planning row).

    Raises :class:`ActionCommutatorError` on authority mismatch (authority is a
    type) or when no consistent basis is defined across the three rows.
    """

    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    for label, effect in (("effect_a", effect_a), ("effect_b", effect_b), ("effect_ab", effect_ab)):
        if not isinstance(effect, ActionEffect):
            raise TypeError(f"{label} must be an ActionEffect; got {type(effect)!r}")

    authority = _resolve_authority(effect_a, effect_b, effect_ab)
    triple = _resolve_triple(effect_a, effect_b, effect_ab)
    comm = triple.delta_ab - triple.delta_a - triple.delta_b
    classification = _classify(comm, eps)
    row: dict[str, Any] = {
        "schema": ACTION_COMMUTATOR_SCHEMA,
        "first_action_id": _effect_action_id(effect_a),
        "second_action_id": _effect_action_id(effect_b),
        "composed_action_id": _effect_action_id(effect_ab),
        "authority": authority,
        "basis": triple.basis,
        "delta_a": triple.delta_a,
        "delta_b": triple.delta_b,
        "delta_ab": triple.delta_ab,
        "comm": comm,
        "synergy_score_units": -comm,
        "classification": classification,
        "eps": float(eps),
        "macro_action_recommended": classification == CLASSIFICATION_SYNERGISTIC,
        "policy": {
            "composition_value_is_measured_against_replayed_composition": True,
            "negative_commutator_means_superadditive_score_improvement": True,
            "basis_is_consistent_across_all_three_rows": True,
            "authority_must_match_across_all_three_rows": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    return row


# ── ledger ────────────────────────────────────────────────────────────────


def _pair_lookup_key(first_id: str, second_id: str) -> tuple[str, str]:
    """Return the ORDER-PRESERVING key for a composite of (first, second).

    Composition is order-sensitive (``b∘a`` is not ``a∘b``).  The lookup key is
    the ordered pair so a measured ``ab`` row is matched to the request that
    declares the SAME order.  Callers that also measured ``ba`` register a
    second pair row; both are independent commutators.
    """

    return (str(first_id), str(second_id))


def _composite_match(
    composite: ActionEffect,
    first_id: str,
    second_id: str,
) -> bool:
    """Decide whether a measured composite row is the ``(first, second)`` pair.

    Match is by the composite's ``action_id`` carrying BOTH single ids as
    substrings in order, OR an explicit convention id ``"<first>+<second>"`` /
    ``"<first>__then__<second>"``.  We never guess across order: a composite that
    only mentions the ids in the opposite order does not match.
    """

    composed_id = str(composite.action_id)
    explicit = {
        f"{first_id}+{second_id}",
        f"{first_id}__then__{second_id}",
        f"{first_id}__{second_id}",
    }
    if composed_id in explicit:
        return True
    # Ordered substring containment (first appears before second).
    i = composed_id.find(str(first_id))
    if i < 0:
        return False
    j = composed_id.find(str(second_id), i + len(str(first_id)))
    return j >= 0


def _needs_measurement_row(
    first: ActionEffect,
    second: ActionEffect,
    *,
    reason: str,
    first_measurement_command: str | None,
) -> dict[str, Any]:
    """Emit a typed needs-measurement row (NO fabricated comm value).

    The proposed composite id is the canonical ``"<first>__then__<second>"`` so a
    measurement actuator knows exactly what archive to build (apply ``first``,
    then ``second``).  The row carries the union of pair/region scopes so the
    measurement can be located, the per-part deltas as side-info, and an explicit
    ``authority_compatible`` flag (a pair whose parts disagree on authority can
    never be a valid commutator and should not be queued for measurement under a
    single surface).
    """

    parts_share_authority = str(first.authority).strip() == str(second.authority).strip()
    proposed_id = f"{first.action_id}__then__{second.action_id}"
    additive_delta_total = _add_optional_floats(first.delta_score_total, second.delta_score_total)
    additive_delta_nonrate = _add_optional_floats(first.delta_score_nonrate, second.delta_score_nonrate)
    additive_delta_bytes = _add_optional_ints(first.delta_bytes, second.delta_bytes)
    byte_cost = None if additive_delta_bytes is None else abs(additive_delta_bytes)
    additive_score_improvement_total = _score_improvement(additive_delta_total)
    additive_score_improvement_nonrate = _score_improvement(additive_delta_nonrate)
    command = first_measurement_command or (
        "uv run python tools/run_pr110_commutator_ledger.py "
        "--action-effects <single_action_effects.jsonl> "
        "--pair-effects <composite_action_effects.jsonl> "
        "--output <commutator_output_dir>"
    )
    return {
        "schema": ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA,
        "first_action_id": first.action_id,
        "second_action_id": second.action_id,
        "proposed_composite_action_id": proposed_id,
        "reason": reason,
        "first_authority": first.authority,
        "second_authority": second.authority,
        "authority_compatible": parts_share_authority,
        "first_delta_score_total": first.delta_score_total,
        "second_delta_score_total": second.delta_score_total,
        "first_delta_score_nonrate": first.delta_score_nonrate,
        "second_delta_score_nonrate": second.delta_score_nonrate,
        "additive_delta_score_total": additive_delta_total,
        "additive_delta_score_nonrate": additive_delta_nonrate,
        "additive_score_improvement_total": additive_score_improvement_total,
        "additive_score_improvement_nonrate": additive_score_improvement_nonrate,
        "additive_delta_bytes": additive_delta_bytes,
        "byte_cost": byte_cost,
        "additive_value_per_byte_total": _value_per_byte(additive_score_improvement_total, byte_cost),
        "additive_value_per_byte_nonrate": _value_per_byte(additive_score_improvement_nonrate, byte_cost),
        "measurement_priority_rank": None,
        "measurement_priority_basis": None,
        "pair_ids": sorted(set(first.pair_ids) | set(second.pair_ids)),
        "region_ids": sorted(set(first.region_ids) | set(second.region_ids)),
        "comm": None,
        "classification": None,
        "first_measurement_command": command,
        "measurement_command_blockers": ["composite_action_effect_row_missing"],
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_commutator_ledger(
    single_effects: Sequence[ActionEffect],
    pair_effects: Sequence[ActionEffect] = (),
    *,
    eps: float = DEFAULT_COMMUTATOR_EPS,
    macro_action_limit: int = 16,
    conflict_pair_limit: int = 16,
    first_measurement_command: str | None = None,
) -> dict[str, Any]:
    """Build the pairwise commutator ledger over ActionEffect rows.

    For every ORDERED pair ``(first, second)`` of distinct single-action effects:

    * if a measured composite (``ab``) is present in ``pair_effects`` and the
      three rows are commutator-compatible, emit a measured commutator row;
    * if no measured composite is present (or measuring it would be invalid),
      emit a typed needs-measurement row — the commutator value is NEVER
      invented.

    Returns a mapping with:

    * ``rows`` — every measured commutator row (computed via
      :func:`commutator_value`).
    * ``macro_action_candidates`` — the most synergistic measured rows
      (``comm`` ascending, i.e. most negative first), capped at
      ``macro_action_limit``.
    * ``conflict_pairs`` — the most conflicting measured rows (``comm``
      descending), capped at ``conflict_pair_limit``.
    * ``measurement_queue`` — typed needs-measurement rows for unmeasured pairs.
    * counts + the canonical false-authority markers.

    A pair whose two single actions disagree on authority is queued with
    ``authority_compatible=False`` (it cannot be a single-surface commutator);
    it never produces a fabricated measured row.
    """

    singles = _coerce_effects(single_effects, "single_effects")
    pairs = _coerce_effects(pair_effects, "pair_effects")

    rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []

    for i, first in enumerate(singles):
        for j, second in enumerate(singles):
            if i == j:
                continue
            if first.action_id == second.action_id:
                # Self-composition / duplicate id: not a distinct pair.
                continue
            composite = _find_composite(pairs, first.action_id, second.action_id)
            if composite is None:
                queue.append(
                    _needs_measurement_row(
                        first,
                        second,
                        reason="no_measured_composite_for_ordered_pair",
                        first_measurement_command=first_measurement_command,
                    )
                )
                continue
            try:
                rows.append(commutator_value(first, second, composite, eps=eps))
            except ActionCommutatorError as exc:
                # The composite exists but is structurally incompatible (authority
                # mismatch or undefined basis).  Record it as a needs-measurement
                # row carrying the failure reason rather than inventing a value.
                queue.append(
                    _needs_measurement_row(
                        first,
                        second,
                        reason=f"measured_composite_incompatible:{exc.args[0] if exc.args else exc}",
                        first_measurement_command=first_measurement_command,
                    )
                )

    synergistic = sorted(
        (r for r in rows if r["classification"] == CLASSIFICATION_SYNERGISTIC),
        key=lambda r: r["comm"],
    )
    conflicting = sorted(
        (r for r in rows if r["classification"] == CLASSIFICATION_CONFLICTING),
        key=lambda r: r["comm"],
        reverse=True,
    )
    ranked_queue = _rank_measurement_queue(queue)
    return {
        "schema": ACTION_COMMUTATOR_LEDGER_SCHEMA,
        "single_effect_count": len(singles),
        "pair_effect_count": len(pairs),
        "ordered_pair_count": _ordered_pair_count(singles),
        "measured_commutator_count": len(rows),
        "synergistic_count": len(synergistic),
        "conflicting_count": len(conflicting),
        "additive_count": sum(1 for r in rows if r["classification"] == CLASSIFICATION_ADDITIVE),
        "needs_measurement_count": len(queue),
        "rows": rows,
        "macro_action_candidates": synergistic[: max(0, int(macro_action_limit))],
        "conflict_pairs": conflicting[: max(0, int(conflict_pair_limit))],
        "measurement_queue": ranked_queue,
        "eps": float(eps),
        "policy": {
            "commutator_values_are_measured_never_invented": True,
            "unmeasured_pairs_emit_needs_measurement_rows": True,
            "composition_is_order_sensitive": True,
            "measurement_queue_carries_first_command": True,
            "measurement_queue_ranked_by_expected_additive_score_authority_and_byte_cost": True,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _add_optional_floats(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    left_f = float(left)
    right_f = float(right)
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return None
    return left_f + right_f


def _add_optional_ints(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return int(left) + int(right)


def _score_improvement(delta: float | None) -> float | None:
    if delta is None:
        return None
    value = float(delta)
    if not math.isfinite(value):
        return None
    return -value


def _value_per_byte(improvement: float | None, byte_cost: int | None) -> float | None:
    if improvement is None or byte_cost is None or byte_cost <= 0:
        return None
    value = float(improvement)
    if not math.isfinite(value):
        return None
    return value / float(byte_cost)


def _rank_measurement_queue(queue: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted((dict(row) for row in queue), key=_measurement_queue_sort_key)
    for index, row in enumerate(ranked, start=1):
        row["measurement_priority_rank"] = index
        row["measurement_priority_basis"] = _measurement_priority_basis(row)
    return ranked


def _measurement_queue_sort_key(row: Mapping[str, Any]) -> tuple[int, int, float, float, int, str, str]:
    authority_penalty = 0 if row.get("authority_compatible") is True else 1
    improvement = _measurement_priority_improvement(row)
    missing_score_penalty = 0 if improvement is not None else 1
    value = _measurement_priority_value_per_byte(row)
    byte_cost = row.get("byte_cost")
    byte_cost_i = int(byte_cost) if isinstance(byte_cost, int) and not isinstance(byte_cost, bool) else 10**18
    return (
        authority_penalty,
        missing_score_penalty,
        -(improvement if improvement is not None else -math.inf),
        -(value if value is not None else -math.inf),
        byte_cost_i,
        str(row.get("first_action_id") or ""),
        str(row.get("second_action_id") or ""),
    )


def _measurement_priority_improvement(row: Mapping[str, Any]) -> float | None:
    total = row.get("additive_score_improvement_total")
    if isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(float(total)):
        return float(total)
    nonrate = row.get("additive_score_improvement_nonrate")
    if isinstance(nonrate, (int, float)) and not isinstance(nonrate, bool) and math.isfinite(float(nonrate)):
        return float(nonrate)
    return None


def _measurement_priority_value_per_byte(row: Mapping[str, Any]) -> float | None:
    total = row.get("additive_value_per_byte_total")
    if isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(float(total)):
        return float(total)
    nonrate = row.get("additive_value_per_byte_nonrate")
    if isinstance(nonrate, (int, float)) and not isinstance(nonrate, bool) and math.isfinite(float(nonrate)):
        return float(nonrate)
    return None


def _measurement_priority_basis(row: Mapping[str, Any]) -> str:
    if _measurement_priority_improvement(row) is None:
        return "undefined"
    if row.get("additive_score_improvement_total") is not None:
        return "total"
    return "nonrate"


def _coerce_effects(effects: Sequence[Any], label: str) -> list[ActionEffect]:
    if isinstance(effects, (str, bytes)) or not isinstance(effects, Sequence):
        raise TypeError(f"{label} must be a sequence of ActionEffect")
    out: list[ActionEffect] = []
    for item in effects:
        if not isinstance(item, ActionEffect):
            raise TypeError(f"{label} must contain only ActionEffect instances; got {type(item)!r}")
        out.append(item)
    return out


def _find_composite(
    pairs: Sequence[ActionEffect],
    first_id: str,
    second_id: str,
) -> ActionEffect | None:
    for composite in pairs:
        if _composite_match(composite, first_id, second_id):
            return composite
    return None


def _ordered_pair_count(singles: Sequence[ActionEffect]) -> int:
    unique = len({s.action_id for s in singles})
    return unique * (unique - 1)


def ledger_from_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Identity round-trip helper for a serialized ledger mapping.

    The ledger is plain JSON-serializable already; this exists so a consumer can
    assert the schema is present and the structural keys exist after a JSONL
    round-trip without re-deriving anything.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("ledger payload must be a mapping")
    if payload.get("schema") != ACTION_COMMUTATOR_LEDGER_SCHEMA:
        raise ValueError(f"not a commutator ledger: schema={payload.get('schema')!r}")
    return dict(payload)


__all__ = [
    "ACTION_COMMUTATOR_LEDGER_SCHEMA",
    "ACTION_COMMUTATOR_NEEDS_MEASUREMENT_SCHEMA",
    "ACTION_COMMUTATOR_SCHEMA",
    "BASIS_NONRATE",
    "BASIS_TOTAL",
    "CLASSIFICATION_ADDITIVE",
    "CLASSIFICATION_CONFLICTING",
    "CLASSIFICATION_SYNERGISTIC",
    "DEFAULT_COMMUTATOR_EPS",
    "ActionCommutatorError",
    "build_commutator_ledger",
    "commutator_value",
    "ledger_from_dict",
]
