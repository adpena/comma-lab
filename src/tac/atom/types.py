# SPDX-License-Identifier: MIT
"""Canonical type primitives for ``tac.atom``.

Source-of-truth enums + Protocol + exception for the atom-shaped element that
subsumes seven previously-scattered atom-shaped surfaces (META_LAGRANGIAN,
CARGO_CULT_ASSUMPTION, PREMISE_VERIFICATION, PROBE_OUTCOME,
COUNCIL_DELIBERATION, DISPATCH_CLAIM, ARBITRARY_VALUE) into ONE canonical
type per operator-approved design 2026-05-18.

Citations:
  - Boyd & Vandenberghe 2004 *Convex Optimization* §5.1 — Lagrangian as
    additive-element composition; each atom is a single primal/dual
    contribution to the meta-Lagrangian search.
  - Catalog #245 ``tac.deploy.modal.call_id_ledger`` — canonical 4-layer
    fcntl-locked JSONL append-only ledger pattern this module mirrors.
  - Operator standing directive 2026-05-18: "frozen dataclass + Enum +
    Protocol pattern (NOT pydantic), matching existing canonical patterns".
  - CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: atoms are the
    typed rows the solver consumes; arbitrary-value rows must enter as
    canonical atoms or remain orphaned signal.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class AtomKind(StrEnum):
    """Seven canonical atom shapes the META layer subsumes.

    ARBITRARY_VALUE — a numeric / structural choice flagged by the
        arbitrariness-extinction audit; carries resolution_path + predicted
        replacement + EV band.
    META_LAGRANGIAN — rate/seg/pose allocation atom in the existing
        ``tac.meta_lagrangian_allocator`` format (HNeRV decoder recode +
        sister rate-only families).
    CARGO_CULT_ASSUMPTION — per-substrate design-memo assumption with
        HARD-EARNED-vs-CARGO-CULTED classification per Catalog #303 +
        the hard-earned-vs-cargo-culted addendum.
    PREMISE_VERIFICATION — a per-edit premise verifier outcome per
        Catalog #229 (premise verification before edit pattern).
    PROBE_OUTCOME — an adjudicated probe verdict (INDEPENDENT / KILL /
        DEFER / PROMOTE / PROCEED / PARTIAL / OPERATOR_REVIEW_REQUIRED)
        mirroring the existing ``tac.probe_outcomes_ledger`` taxonomy
        per Catalog #313.
    COUNCIL_DELIBERATION — a sextet / grand council deliberation
        anchor mirroring ``tac.council_continual_learning`` schema per
        Catalog #300.
    DISPATCH_CLAIM — an active or terminal lane dispatch claim row from
        ``.omx/state/active_lane_dispatch_claims.md``.
    """

    ARBITRARY_VALUE = "arbitrary_value"
    META_LAGRANGIAN = "meta_lagrangian"
    CARGO_CULT_ASSUMPTION = "cargo_cult_assumption"
    PREMISE_VERIFICATION = "premise_verification"
    PROBE_OUTCOME = "probe_outcome"
    COUNCIL_DELIBERATION = "council_deliberation"
    DISPATCH_CLAIM = "dispatch_claim"


class ResolutionPath(StrEnum):
    """Five canonical resolution paths for atomic uncertainty.

    Per the arbitrariness-extinction audit canonical schema 2026-05-18 and
    the synthesis-memo-amendment design pitch operator approved verbatim
    "yes". A sixth member CONTEST_FIXED was added to subsume audit rows
    whose ``resolution_path`` is "contest_fixed" — values mandated by the
    contest itself (e.g. ``CONTEST_RATE_DENOM_BYTES = 37_545_489`` per
    upstream ``evaluate.py``); these are NOT arbitrary, only structurally
    surface-classified to keep them visible at the atom layer.

    EXPERIMENTAL — resolved by paired-comparison empirical smoke
        (small-cost dispatch + measurement).
    ANALYTICAL_SOLVE — resolved by a closed-form solve from first
        principles (Lagrangian KKT, Dykstra alternating projection,
        Bayesian posterior, OT/Wasserstein cost matrix, etc.).
    FORMULA — resolved by a known formula or canonical literature
        anchor (e.g. ``25 * archive_bytes / 37_545_489`` for the
        contest rate term).
    LEARNED — resolved by an inner-loop learning step (e.g. cyclical-LR
        finder per Smith 2017; bilevel MAML adaptation).
    SELF_ALIEN_TECH — resolved by self-compression / weight-derived
        codebook / Wyner-Ziv side-information / hyperprior etc. —
        canonical alien-technology category for contest-frontier moves
        per PR101 + PR106 + Quantizr 0.33 lineage.
    CONTEST_FIXED — contest-mandated value; not arbitrary, surfaced
        only for symmetry with the other 5 resolution paths.
    """

    EXPERIMENTAL = "experimental"
    ANALYTICAL_SOLVE = "analytical_solve"
    FORMULA = "formula"
    LEARNED = "learned"
    SELF_ALIEN_TECH = "self_alien_tech"
    CONTEST_FIXED = "contest_fixed"


class AtomValidationError(ValueError):
    """Raised by ``Atom.__post_init__`` and canonical builders when invariants fail.

    The canonical helpers fail closed; downstream consumers should NEVER
    catch this exception and silently downgrade. Either fix the kwargs at
    the construction site or wrap the construction in an explicit waiver
    routed through the canonical Catalog #229 premise-verification surface.
    """


@runtime_checkable
class AtomProtocol(Protocol):
    """Structural-typing contract for any atom-shaped element.

    Consumers (cathedral autopilot ranker / meta-Lagrangian search /
    Rashomon ensemble / Pareto frontier solver) should depend on this
    Protocol rather than the concrete ``Atom`` dataclass; this keeps
    the META layer open to future polymorphic atoms (e.g.
    META_TENSOR_PRODUCT, META_TIME_SERIES, etc.) without inheritance
    coupling per CLAUDE.md "Beauty, simplicity, and developer experience"
    non-negotiable.
    """

    atom_id: str
    kind: AtomKind
    resolution_path: ResolutionPath
    predicted_impact_delta_s_lower: float
    predicted_impact_delta_s_upper: float
    cost_envelope_usd: float

    def to_jsonl_row(self) -> dict[str, Any]: ...

    def to_meta_lagrangian_atom(self) -> dict[str, Any]: ...


__all__ = [
    "AtomKind",
    "AtomProtocol",
    "AtomValidationError",
    "ResolutionPath",
]
