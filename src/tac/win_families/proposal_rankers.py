# SPDX-License-Identifier: MIT
"""Pluggable GEOMETRIC and NEURAL proposal rankers for the F1 realized-acceptance engine.

THE BOUNDARY THAT MAKES THIS SAFE
---------------------------------
A ranker **orders and truncates** the candidate list.  It **never accepts**.  Acceptance
stays realized-only inside
:class:`tac.win_families.realized_acceptance.RealizedAcceptanceEngine`, which re-scores
whatever survives the ranker through the family's real decode.  So the worst a bad
ranker can do is waste realizations or reach a different local optimum; it cannot put an
unmeasured move into the state.  That asymmetry is the whole reason a *learned* ranker is
admissible here at all while a learned *acceptor* would not be.

Truncation is the one place a ranker changes the guarantee.  With ``top_k`` set, the
"no improving neighbour" stop becomes "no improving neighbour **among the k the ranker
kept**", which is a weaker proof.  :class:`RankerConfig` therefore records ``top_k`` and
every report carries ``convergence_proof_weakened`` so a downstream verdict cannot quietly
inherit the strong claim.

WHY MARGIN IS THE GEOMETRIC SIGNAL
----------------------------------
d_seg lives on the codim-1 boundary of the frozen argmax partition, and in the frozen
scorer's Fisher metric the boundary is where the geometry is anisotropic while the
interior is flat.  The campaign measured Fisher curvature against ``(-margin)`` at
Pearson **0.978**, so the margin field *is* the Fisher surrogate and needs no second
model.  A proposal that touches low-margin cells is touching cells whose argmax is
closest to flipping, which is exactly where a move can pay.

The carrier's geometric signal is different in kind and is handled separately: for a
12-coefficient pose carrier the informative structure is the conditioning of
``d(pose)/d(coeff)`` (``ddm_up2.conditioning_report``), so
:class:`JacobianConditioningRanker` orders coordinate moves by how much pose one lattice
step actually buys along each singular direction.

THE LEARNED RANKER
------------------
:class:`LearnedRanker` is an INTERFACE plus a feature contract, and it ships **untrained**:
constructing one without a model refuses.  The training data is free -- every
:class:`~tac.win_families.realized_acceptance.AcceptanceEvent` a descent emits is a
labelled example (proposal features -> realized delta), for accepts and rejects alike.
:func:`training_table_from_events` is the emitter that turns a run's event log into that
table, so future runs generate the corpus at zero marginal cost.

Training itself is out of scope for ``ddm_cw1`` and is stated as such rather than stubbed:
no model is fitted here, and no ranking quality is claimed for the learned path.
:func:`ranker_quality` is the honest scoring surface for when one is fitted -- it compares
a ranker's ORDER against the realized values the frozen scorer actually returned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tac.win_families.realized_acceptance import AcceptanceEvent, Proposal

__all__ = [
    "IdentityRanker",
    "JacobianConditioningRanker",
    "LearnedRanker",
    "MarginSaliencyRanker",
    "RankerConfig",
    "RankerError",
    "RankerQuality",
    "ranker_quality",
    "training_table_from_events",
]

#: Feature names emitted for every proposal, in a fixed order.  A learned ranker trained
#: on this contract stays loadable across runs; changing the order is a breaking change.
FEATURE_NAMES: tuple[str, ...] = (
    "coordinate",
    "pass_index",
    "incumbent_value",
    "rank_score",
)


class RankerError(RuntimeError):
    """A ranker precondition failed.  Always fail closed."""


@runtime_checkable
class ProposalRanker(Protocol):
    """Order (and optionally truncate) a coordinate's proposals.  Never accepts."""

    def __call__(self, proposals: Sequence[Proposal], coordinate: int) -> Sequence[Proposal]:
        ...


@dataclass(frozen=True)
class RankerConfig:
    """Shared ranker settings, and the honesty flag truncation forces."""

    #: Keep only the best ``top_k`` proposals.  ``0`` keeps all of them.
    top_k: int = 0

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise RankerError(f"top_k must be >= 0, got {self.top_k}")

    @property
    def convergence_proof_weakened(self) -> bool:
        """True when truncation downgrades the engine's no-improving-neighbour proof."""
        return self.top_k > 0

    def apply(self, ordered: list[Proposal]) -> list[Proposal]:
        return ordered[: self.top_k] if self.top_k else ordered


def _with_rank(proposal: Proposal, score: float) -> Proposal:
    return Proposal(
        coordinate=proposal.coordinate,
        label=proposal.label,
        state=proposal.state,
        rank_score=float(score),
    )


class IdentityRanker:
    """The CONTROL: returns the proposals untouched, in generator order.

    Every ranker claim is measured against this.  A ranker that does not beat the
    identity control on realizations-to-optimum has bought nothing, and the identity
    ranker is also the only configuration under which the engine's convergence proof is
    unweakened by construction.
    """

    config = RankerConfig(top_k=0)

    def __call__(self, proposals: Sequence[Proposal], coordinate: int) -> list[Proposal]:
        return list(proposals)


@dataclass(frozen=True)
class MarginSaliencyRanker:
    """GEOMETRIC: order proposals by the scorer margin at the cells they change.

    ``margin_field`` maps a coordinate (e.g. a pair index) to a per-cell margin array on
    the same lattice as the proposal state -- the frozen scorer's top-1 minus top-2 logit.
    The margin field IS the Fisher surrogate (Pearson 0.978 against Fisher curvature), so
    ranking by it is ranking by where the partition is most nearly indifferent.

    A proposal's score is the MINIMUM margin over the cells it changes: one cell close to
    flipping is the opportunity, and averaging would let a large untouched-margin region
    hide it.  Lower score sorts first.

    Proposals that change nothing score ``+inf`` and sort last -- they cannot pay.
    """

    margin_field: dict[int, np.ndarray]
    #: Incumbent state per coordinate.  REQUIRED: a proposal is a full state, so without
    #: the incumbent there is nothing to diff it against, and every proposal would score
    #: the field minimum identically -- a constant ranking wearing a geometry's name.
    #: Refusing is the fail-closed choice; silently ranking everything equal is the
    #: vacuity-equals-pass failure (a skipped instrument reads as green).
    incumbent_field: dict[int, np.ndarray]
    config: RankerConfig = field(default_factory=RankerConfig)

    def __call__(self, proposals: Sequence[Proposal], coordinate: int) -> list[Proposal]:
        margins = self.margin_field.get(coordinate)
        if margins is None:
            raise RankerError(
                f"MarginSaliencyRanker has no margin field for coordinate {coordinate}; "
                "refusing to rank on an absent geometry rather than falling back silently"
            )
        incumbent = self.incumbent_field.get(coordinate)
        if incumbent is None:
            raise RankerError(
                f"MarginSaliencyRanker has no incumbent state for coordinate {coordinate}; "
                "without it every proposal scores the field minimum and the ranking is a "
                "constant wearing a geometry's name"
            )
        scored: list[tuple[float, int, Proposal]] = []
        for index, proposal in enumerate(proposals):
            changed = self._changed_mask(proposal, incumbent, margins)
            score = float(margins[changed].min()) if changed.any() else float("inf")
            scored.append((score, index, _with_rank(proposal, score)))
        scored.sort(key=lambda row: (row[0], row[1]))
        return self.config.apply([row[2] for row in scored])

    @staticmethod
    def _changed_mask(
        proposal: Proposal, incumbent: np.ndarray, margins: np.ndarray
    ) -> np.ndarray:
        state = np.asarray(proposal.state)
        if state.shape != margins.shape:
            raise RankerError(
                f"proposal state {state.shape} and margin field {margins.shape} are on "
                "different lattices; the ranker cannot map a move to a margin"
            )
        incumbent = np.asarray(incumbent)
        if incumbent.shape != state.shape:
            raise RankerError(
                f"incumbent {incumbent.shape} and proposal state {state.shape} disagree"
            )
        return incumbent != state


@dataclass(frozen=True)
class JacobianConditioningRanker:
    """GEOMETRIC: order carrier-coefficient moves by pose bought per lattice step.

    ``jacobians`` maps a coordinate to ``d(objective components)/d(state)`` with shape
    ``(m, n)`` -- for the pose carrier, ``(6, 12)``.  The column norm of that Jacobian is
    how much the objective moves per unit of coefficient ``j``, so a single lattice step
    on a large-column-norm coefficient is the move most likely to matter.

    This is ``ddm_up2.conditioning_report`` read as an ordering rather than as a
    diagnostic.  It explains that arm's own finding structurally: when the residual
    concentrates on the smallest singular direction, the demanded step leaves the
    lattice, and no ordering rescues a move the representation cannot make.
    """

    jacobians: dict[int, np.ndarray]
    config: RankerConfig = field(default_factory=RankerConfig)

    def __call__(self, proposals: Sequence[Proposal], coordinate: int) -> list[Proposal]:
        jacobian = self.jacobians.get(coordinate)
        if jacobian is None:
            raise RankerError(
                f"JacobianConditioningRanker has no Jacobian for coordinate {coordinate}"
            )
        jacobian = np.asarray(jacobian, dtype=np.float64)
        if jacobian.ndim != 2:
            raise RankerError(f"Jacobian must be 2-D (m, n), got shape {jacobian.shape}")
        column_norms = np.linalg.norm(jacobian, axis=0)
        scored: list[tuple[float, int, Proposal]] = []
        for index, proposal in enumerate(proposals):
            slot = self._changed_slot(proposal, jacobian.shape[1])
            # Negated so that LARGER pose-per-step sorts FIRST under an ascending sort.
            score = -float(column_norms[slot]) if slot is not None else 0.0
            scored.append((score, index, _with_rank(proposal, score)))
        scored.sort(key=lambda row: (row[0], row[1]))
        return self.config.apply([row[2] for row in scored])

    @staticmethod
    def _changed_slot(proposal: Proposal, width: int) -> int | None:
        """Recover the moved coefficient index from the generator's ``dN+M`` label."""
        label = proposal.label
        if not label.startswith("d"):
            return None
        digits = ""
        for character in label[1:]:
            if character.isdigit():
                digits += character
            else:
                break
        if not digits:
            return None
        slot = int(digits)
        return slot if 0 <= slot < width else None


@dataclass(frozen=True)
class LearnedRanker:
    """NEURAL: order proposals by a learned score.  Ships UNTRAINED and refuses.

    ``model`` must be a callable mapping a ``(n_proposals, n_features)`` float array to a
    length-``n_proposals`` score array, lower-sorts-first.  No model is fitted in this
    module -- ``ddm_cw1`` built the interface and the free-corpus emitter, and states
    plainly that training was out of its scope.  Constructing this class without a model
    raises rather than silently degrading to a heuristic, because a ranker that pretends
    to be learned while running a hand-written rule is the fake-implementation class.
    """

    model: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES
    config: RankerConfig = field(default_factory=RankerConfig)

    def __post_init__(self) -> None:
        if self.model is None:
            raise RankerError(
                "LearnedRanker requires a trained model. ddm_cw1 landed the interface and "
                "the free training-data emitter (training_table_from_events) but fitted no "
                "model; use IdentityRanker or a geometric ranker until one is trained."
            )
        if not callable(self.model):
            raise RankerError(f"model must be callable, got {type(self.model).__name__}")

    def __call__(self, proposals: Sequence[Proposal], coordinate: int) -> list[Proposal]:
        if not proposals:
            return []
        features = np.array(
            [[float(coordinate), 0.0, 0.0, float(p.rank_score or 0.0)] for p in proposals],
            dtype=np.float64,
        )
        scores = np.asarray(self.model(features), dtype=np.float64)
        if scores.shape != (len(proposals),):
            raise RankerError(
                f"model returned shape {scores.shape} for {len(proposals)} proposals"
            )
        order = np.argsort(scores, kind="stable")
        return self.config.apply(
            [_with_rank(proposals[i], float(scores[i])) for i in order]
        )


# ---------------------------------------------------------------------------
# The free corpus, and the honest scoring of a ranker.
# ---------------------------------------------------------------------------


def training_table_from_events(
    events: Sequence[AcceptanceEvent],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Turn a descent's event log into ``(features, labels, feature_names)``.

    The label is the REALIZED delta in score units (negative = the move helped), which is
    what a ranker should be predicting.  Accepts and rejects are both included: a ranker
    trained only on accepts never learns what a bad move looks like, and rejects are the
    overwhelming majority of what a descent measures.

    This is the zero-marginal-cost part: the realizations already happened to satisfy the
    acceptance rule, so the corpus is a by-product of every run rather than a new cost.
    """
    if not events:
        raise RankerError("cannot build a training table from an empty event log")
    features = np.array(
        [
            [
                float(event.coordinate),
                float(event.pass_index),
                float(event.incumbent_value),
                float(event.rank_score if event.rank_score is not None else 0.0),
            ]
            for event in events
        ],
        dtype=np.float64,
    )
    labels = np.array([float(event.delta) for event in events], dtype=np.float64)
    return features, labels, FEATURE_NAMES


@dataclass(frozen=True)
class RankerQuality:
    """How well a ranker's ORDER predicted what the frozen scorer actually said."""

    proposals: int
    top1_is_best: bool
    best_rank_position: int
    realizations_to_best: int
    spearman: float

    def to_json(self) -> dict[str, Any]:
        return {
            "proposals": self.proposals,
            "top1_is_best": self.top1_is_best,
            "best_rank_position": self.best_rank_position,
            "realizations_to_best": self.realizations_to_best,
            "spearman": self.spearman,
            "note": (
                "measured against realized values from the frozen scorer; a ranker is "
                "only worth its cost if realizations_to_best beats IdentityRanker's"
            ),
        }


def ranker_quality(
    ranked_scores: Sequence[float], realized_values: Sequence[float]
) -> RankerQuality:
    """Score a ranker's order against realized outcomes, both in the ranker's order.

    ``realized_values`` must be the values the frozen scorer returned for the SAME
    proposals in the SAME order the ranker emitted.  ``realizations_to_best`` is the
    operational number: how many real decodes a run had to spend before it saw the best
    candidate under this ordering.
    """
    scores = np.asarray(ranked_scores, dtype=np.float64)
    realized = np.asarray(realized_values, dtype=np.float64)
    if scores.shape != realized.shape:
        raise RankerError(
            f"ranked_scores {scores.shape} and realized_values {realized.shape} disagree"
        )
    if scores.size == 0:
        raise RankerError("cannot score a ranker on zero proposals")
    best_position = int(np.argmin(realized))
    if scores.size == 1:
        spearman = 1.0
    else:
        rank_a = np.argsort(np.argsort(scores, kind="stable"))
        rank_b = np.argsort(np.argsort(realized, kind="stable"))
        degenerate = np.std(rank_a) == 0 or np.std(rank_b) == 0
        spearman = 0.0 if degenerate else float(np.corrcoef(rank_a, rank_b)[0, 1])
    return RankerQuality(
        proposals=int(scores.size),
        top1_is_best=bool(best_position == 0),
        best_rank_position=best_position,
        realizations_to_best=best_position + 1,
        spearman=spearman,
    )
