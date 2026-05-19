# SPDX-License-Identifier: MIT
"""Linguistic-feature extensions to ``tac.atom`` per design memo APPENDIX B.

ADDITIVE extension to the canonical ``tac.atom`` package landed at commit
``181fa4c1e``. Per CLAUDE.md "Subagent coherence-by-default" + Catalog
#314 absorption discipline, we extend ADDITIVELY (new module + new
public symbols) rather than mutate the canonical ``Atom`` dataclass /
``types.py`` enums. This keeps the existing 41-pub API surface intact
while operationalizing TOP-3 linguistic-enrichment proposals from
the design memo APPENDIX B:

  TOP-1: Modal-logic enum replacement for boolean ``is_arbitrary``
         (``ArbitrarinessClassification`` 4-modal).
  TOP-2: Atom-algebra (``compose`` / ``intersect`` / ``union`` /
         ``complement``) as canonical operations on Atom-like objects.
  TOP-3: Temporal-logic decorators (``@always_invariant`` /
         ``@eventually_extincted_within(days)`` / ``@valid_until(condition)``).

Citations:
  - Operator standing directive 2026-05-18 verbatim *"what verbs and
    syntax and grammar and other linguistic features and logic are we
    missing to fully realize and optimize against all implications?
    yes proceed with all"*.
  - Design memo APPENDIX B (commit ``07b24f303``) -- linguistic-feature
    inventory + TOP-3 highest-EV proposals.
  - Hughes 2009 *Software Cheaply Built* -- modal logic for software contracts.
  - Manna & Pnueli 1992 *The Temporal Logic of Reactive and Concurrent
    Systems* -- ALWAYS / EVENTUALLY / UNTIL temporal-logic operators.
  - Sister Catalog #303 cargo-cult audit (HARD-EARNED vs CARGO-CULTED
    classification) -- precedent for the 4-modal classification surface.

Catalog #125 hook 1 (sensitivity_map): N/A (this module is the meta
linguistic-extension surface; does not contribute to score sensitivity).
Catalog #125 hook 4 (cathedral_autopilot_dispatch): ACTIVE -- the
``ArbitrarinessClassification`` flows through Atom.metadata, where
the autopilot ranker can use it to discount POSSIBLY_ARBITRARY entries
vs NECESSARILY_CONTEST_FIXED ones.
Catalog #305 observability surface: cite_able, decomposable_per_signal,
queryable_post_hoc.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, TypeVar

from .atom import Atom
from .types import AtomKind, AtomValidationError, ResolutionPath

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# TOP-1: Modal-logic enum replacing boolean is_arbitrary
# ---------------------------------------------------------------------------
class ArbitrarinessClassification(StrEnum):
    """Modal-logic 4-modal replacement for boolean ``is_arbitrary``.

    Per design memo APPENDIX B TOP-1. From boolean ``{True, False}`` to
    4-modal classification capturing the actual epistemic state of an
    arbitrary-value candidate:

    NECESSARILY_CONTEST_FIXED:
        Contest defines this value; it CANNOT be changed without violating
        the contest contract (e.g. ``CONTEST_RATE_DENOM_BYTES = 37_545_489``
        per ``upstream/evaluate.py``). Sister of CLAUDE.md
        "Non-Negotiable Upstream Rule".

    POSSIBLY_ARBITRARY:
        Hand-tuned; MAY be empirically optimal but not derived. The
        canonical extinction candidate -- replacement with an analytical
        / formula / empirical-anchor value would be HIGH-EV.

    NECESSARILY_EMPIRICAL:
        Must be measured (cannot be derived analytically from first
        principles). Example: per-substrate scorer-saturation point;
        per-archive Tier C density.

    INDETERMINATE_PENDING_EVIDENCE:
        Not enough information yet. Default for newly-discovered
        arbitrary-value candidates that need triage.

    Sister of Catalog #303 HARD-EARNED-vs-CARGO-CULTED classification:
    HARD-EARNED ~= NECESSARILY_EMPIRICAL + NECESSARILY_CONTEST_FIXED
    CARGO-CULTED ~= POSSIBLY_ARBITRARY
    """

    NECESSARILY_CONTEST_FIXED = "necessarily_contest_fixed"
    POSSIBLY_ARBITRARY = "possibly_arbitrary"
    NECESSARILY_EMPIRICAL = "necessarily_empirical"
    INDETERMINATE_PENDING_EVIDENCE = "indeterminate_pending_evidence"


def classify_atom_arbitrariness(atom: Atom) -> ArbitrarinessClassification:
    """Infer ``ArbitrarinessClassification`` from an existing Atom.

    Maps the canonical ``ResolutionPath`` to the 4-modal classification:
      - ResolutionPath.CONTEST_FIXED   -> NECESSARILY_CONTEST_FIXED
      - ResolutionPath.FORMULA / ANALYTICAL_SOLVE -> NECESSARILY_CONTEST_FIXED
      - ResolutionPath.EXPERIMENTAL / LEARNED / SELF_ALIEN_TECH
                                       -> NECESSARILY_EMPIRICAL

    Atoms with explicit ``arbitrariness_classification`` field in metadata
    override the inferred classification (escape hatch for atoms that
    have been operator-classified explicitly).
    """
    explicit = atom.metadata.get("arbitrariness_classification")
    if explicit is not None:
        if isinstance(explicit, ArbitrarinessClassification):
            return explicit
        try:
            return ArbitrarinessClassification(str(explicit))
        except ValueError:
            pass  # fall through to inference

    rp = atom.resolution_path
    if rp == ResolutionPath.CONTEST_FIXED:
        return ArbitrarinessClassification.NECESSARILY_CONTEST_FIXED
    if rp in (ResolutionPath.FORMULA, ResolutionPath.ANALYTICAL_SOLVE):
        return ArbitrarinessClassification.NECESSARILY_CONTEST_FIXED
    if rp in (
        ResolutionPath.EXPERIMENTAL,
        ResolutionPath.LEARNED,
        ResolutionPath.SELF_ALIEN_TECH,
    ):
        return ArbitrarinessClassification.NECESSARILY_EMPIRICAL
    return ArbitrarinessClassification.INDETERMINATE_PENDING_EVIDENCE


# ---------------------------------------------------------------------------
# TOP-2: Atom-algebra (compose / intersect / union / complement)
# ---------------------------------------------------------------------------
class AtomAlgebraError(ValueError):
    """Raised when atom-algebra operations receive incompatible operands."""


def _merge_metadata_for_op(
    *, left: Mapping[str, Any], right: Mapping[str, Any], op: str
) -> dict[str, Any]:
    """Merge two metadata dicts under an algebra operation.

    Conflict resolution: ``right`` overrides ``left`` for any shared key.
    The operation tag is recorded in ``algebra_history`` for provenance.
    """
    merged = dict(left)
    merged.update(right)
    history = list(merged.get("algebra_history", []))
    history.append(op)
    merged["algebra_history"] = history
    return merged


def compose_atoms(
    *, left: Atom, right: Atom, op: str = "and"
) -> Atom:
    """Atom-algebra composition per design memo APPENDIX B TOP-2.

    Supported operators (composition semantics):
      - ``"and"``: both atoms must apply; predicted impact is summed.
      - ``"or"``: either atom applies; predicted impact is the max.
      - ``"xor"``: exactly one applies; predicted impact is the |diff|.
      - ``"minus"``: left atom applies but right does not;
                    predicted impact is the difference (clamped to >= 0).

    The returned atom uses the LEFT atom's identity fields (kind /
    resolution_path / provenance / wired_hooks / observability_surface /
    citation / repo_link) and merges metadata. The atom_id is suffixed
    with the operation tag for provenance.

    Args:
        left: First atom operand.
        right: Second atom operand.
        op: Composition operator ("and" / "or" / "xor" / "minus").

    Returns:
        New canonical ``Atom`` representing the composition.

    Raises:
        AtomAlgebraError: if op is not recognized or operands are incompatible.
    """
    if op not in ("and", "or", "xor", "minus"):
        raise AtomAlgebraError(
            f"op must be one of 'and', 'or', 'xor', 'minus' (got {op!r})"
        )
    if not isinstance(left, Atom) or not isinstance(right, Atom):
        raise AtomAlgebraError(
            f"both operands must be Atom (got {type(left).__name__} + "
            f"{type(right).__name__})"
        )

    if op == "and":
        lower = left.predicted_impact_delta_s_lower + right.predicted_impact_delta_s_lower
        upper = left.predicted_impact_delta_s_upper + right.predicted_impact_delta_s_upper
    elif op == "or":
        lower = min(
            left.predicted_impact_delta_s_lower, right.predicted_impact_delta_s_lower
        )
        upper = max(
            left.predicted_impact_delta_s_upper, right.predicted_impact_delta_s_upper
        )
    elif op == "xor":
        lower = abs(
            left.predicted_impact_delta_s_lower - right.predicted_impact_delta_s_lower
        )
        upper = abs(
            left.predicted_impact_delta_s_upper - right.predicted_impact_delta_s_upper
        )
        # Ensure lower <= upper after abs
        if lower > upper:
            lower, upper = upper, lower
    else:  # minus
        lower = max(
            0.0, left.predicted_impact_delta_s_lower - right.predicted_impact_delta_s_upper
        )
        upper = max(
            0.0, left.predicted_impact_delta_s_upper - right.predicted_impact_delta_s_lower
        )
        if lower > upper:
            lower, upper = upper, lower

    merged_meta = _merge_metadata_for_op(
        left=left.metadata, right=right.metadata, op=f"compose:{op}"
    )

    # New cost envelope: sum of costs (we ran both probes).
    new_cost = float(left.cost_envelope_usd + right.cost_envelope_usd)

    return Atom(
        atom_id=f"{left.atom_id}__{op}__{right.atom_id}",
        kind=left.kind,
        resolution_path=left.resolution_path,
        predicted_impact_delta_s_lower=float(lower),
        predicted_impact_delta_s_upper=float(upper),
        cost_envelope_usd=new_cost,
        provenance=left.provenance,
        wired_hooks=list(left.wired_hooks),
        observability_surface=list(left.observability_surface),
        literature_citation=left.literature_citation,
        canonical_helper_repo_link=left.canonical_helper_repo_link,
        metadata=merged_meta,
    )


def intersect_atoms(*, left: Atom, right: Atom) -> Atom:
    """Set-theoretic intersection of two atoms (canonical alias for compose+and).

    Per APPENDIX B TOP-2: returns the canonical "both atoms apply" composition.
    Predicted impact bands are SUMMED (conservative; the atoms compose).
    """
    return compose_atoms(left=left, right=right, op="and")


def union_atoms(*, left: Atom, right: Atom) -> Atom:
    """Set-theoretic union of two atoms (canonical alias for compose+or).

    Per APPENDIX B TOP-2: returns the canonical "either atom applies"
    composition. Predicted impact band uses extrema (min of lowers, max
    of uppers).
    """
    return compose_atoms(left=left, right=right, op="or")


def complement_atom(*, atom: Atom) -> Atom:
    """Atom complement (what's NOT in this atom).

    Per APPENDIX B TOP-2: returns an atom carrying the NEGATED predicted
    impact -- if the original atom predicts ``[dS_lower, dS_upper]``
    improvement, the complement predicts the FOREGONE improvement (i.e.
    the opportunity cost of NOT taking this action).

    Mathematically: ``complement(atom).predicted_impact = -atom.predicted_impact``;
    cost envelope is zero (we don't pay for inaction).

    Args:
        atom: Atom to complement.

    Returns:
        New canonical ``Atom`` representing the complement.
    """
    if not isinstance(atom, Atom):
        raise AtomAlgebraError(f"atom must be Atom (got {type(atom).__name__})")

    # Negation flips signs; ensure lower <= upper after negation.
    new_lower = -atom.predicted_impact_delta_s_upper
    new_upper = -atom.predicted_impact_delta_s_lower

    merged_meta = dict(atom.metadata)
    history = list(merged_meta.get("algebra_history", []))
    history.append("complement")
    merged_meta["algebra_history"] = history

    return Atom(
        atom_id=f"{atom.atom_id}__complement",
        kind=atom.kind,
        resolution_path=atom.resolution_path,
        predicted_impact_delta_s_lower=float(new_lower),
        predicted_impact_delta_s_upper=float(new_upper),
        cost_envelope_usd=0.0,
        provenance=atom.provenance,
        wired_hooks=list(atom.wired_hooks),
        observability_surface=list(atom.observability_surface),
        literature_citation=atom.literature_citation,
        canonical_helper_repo_link=atom.canonical_helper_repo_link,
        metadata=merged_meta,
    )


# ---------------------------------------------------------------------------
# TOP-3: Temporal-logic decorators
# ---------------------------------------------------------------------------
class TemporalLogicViolationError(RuntimeError):
    """Raised when a temporal-logic decorator's contract is violated at runtime."""


def always_invariant(predicate: Callable[..., bool]) -> Callable[[F], F]:
    """Temporal-logic ALWAYS: predicate must hold at every call.

    Per design memo APPENDIX B TOP-3. The decorated function checks
    ``predicate(*args, **kwargs)`` BEFORE executing its body; raises
    ``TemporalLogicViolationError`` if false. Sister of Catalog #229
    premise-verification + #323 Provenance audit.

    Args:
        predicate: Callable returning True if the invariant holds.

    Returns:
        Decorator that wraps the target function with the always-invariant.

    Example::

        @always_invariant(lambda *a, **kw: kw.get("d_pose", 0) >= 0)
        def my_fn(*, d_pose: float) -> float:
            return d_pose ** 2
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not predicate(*args, **kwargs):
                raise TemporalLogicViolationError(
                    f"always_invariant({predicate}) violated at call to "
                    f"{fn.__qualname__}(args={args}, kwargs={kwargs})"
                )
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def eventually_extincted_within(*, days: int) -> Callable[[F], F]:
    """Temporal-logic EVENTUALLY: extinct this atom within ``days``.

    Per design memo APPENDIX B TOP-3. The decorated function STAMPS its
    decoration moment + the deadline into the function's ``__dict__`` for
    canonical auditability. Sister of Catalog #298 30-day staleness +
    Catalog #325 14-day per-substrate symposium window.

    NOTE: This decorator does NOT enforce extinction at runtime (would
    require ambient temporal state); it RECORDS the deadline for
    downstream audit. The canonical audit tool reads
    ``fn._eventually_extincted_within_deadline_utc`` and reports
    overdue decorations.

    Args:
        days: Number of days within which the decorated function should
            be extincted (replaced by a canonical sister / removed entirely).

    Returns:
        Decorator that stamps the deadline.

    Example::

        @eventually_extincted_within(days=30)
        def temporary_hand_tuned_lambda():
            return 0.997
    """
    if days <= 0:
        raise ValueError(f"days must be > 0 (got {days})")

    def decorator(fn: F) -> F:
        decoration_moment = datetime.now(UTC)
        deadline = decoration_moment + timedelta(days=days)
        # Stamp deadline into function metadata (canonical audit surface).
        fn._eventually_extincted_within_decoration_utc = (  # type: ignore[attr-defined]
            decoration_moment.isoformat()
        )
        fn._eventually_extincted_within_deadline_utc = (  # type: ignore[attr-defined]
            deadline.isoformat()
        )
        fn._eventually_extincted_within_days = days  # type: ignore[attr-defined]
        return fn

    return decorator


def valid_until(condition: Callable[..., bool]) -> Callable[[F], F]:
    """Temporal-logic UNTIL: this atom is valid until ``condition`` fires.

    Per design memo APPENDIX B TOP-3. The decorated function calls
    ``condition()`` (no args) BEFORE executing its body; if condition
    returns True, raises ``TemporalLogicViolationError`` because the
    function is no longer valid. Sister of Catalog #313 probe-outcomes
    ledger TTL + Catalog #324 post-training validation.

    Args:
        condition: Zero-arg callable returning True when the function
            should NO LONGER be valid.

    Returns:
        Decorator that wraps the target function with the until-condition.

    Example::

        is_pre_council_verdict = lambda: not council_has_proceeded()

        @valid_until(lambda: council_has_proceeded())
        def pre_council_stub():
            return "use canonical sister after council PROCEED verdict"
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if condition():
                raise TemporalLogicViolationError(
                    f"valid_until() condition fired; {fn.__qualname__} "
                    f"is no longer valid"
                )
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    # TOP-1: modal-logic classification
    "ArbitrarinessClassification",
    "classify_atom_arbitrariness",
    # TOP-2: atom algebra
    "AtomAlgebraError",
    "complement_atom",
    "compose_atoms",
    "intersect_atoms",
    "union_atoms",
    # TOP-3: temporal-logic decorators
    "TemporalLogicViolationError",
    "always_invariant",
    "eventually_extincted_within",
    "valid_until",
]
