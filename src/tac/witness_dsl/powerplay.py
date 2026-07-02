"""POWERPLAY campaign primitives — the contest as a Schmidhuber (arXiv:1112.5309) search.

Our contest score IS a POWERPLAY Variant-II cost. Schmidhuber's practical acceptance
cost is ``C(s, tasks) = L(s) + alpha * sum_T[time_to_solve(T) - reward(T)]``. Read in our
units:

    L(s)               = archive.zip bytes                 -> the rate term 25*bytes/N
    alpha * sum[...]    = the weighted task-solving deficit -> 100*d_seg + sqrt(10*d_pose)
    alpha              = the rate-distortion Lagrangian lambda

so ``tac.contest_score.compute_contest_score`` IS the POWERPLAY cost over a 2-task
repertoire (task 1 = match SegNet argmax, task 2 = match PoseNet-6). This module makes
three POWERPLAY mechanisms executable for the campaign meta-layer (the DAG<->DSL<->equations
triality):

  1. ``CorrectnessDemonstration`` — the executable review AXIS-9 (CLAUDE.md "Recursive
     adversarial review protocol" item 9). POWERPLAY never ACCEPTS a solver-modification
     until a Correctness Demonstration PROVES the new task is solved, no prior task
     regressed, and the predecessor did not already solve it. Here: a launch/promotion is
     accepted only with MEASURED evidence — through the REAL byte-closed decode, NEVER a
     proxy / ancestor-vehicle / MPS / training-side number — that the scored quantities
     hold, net-S does not regress the predecessor, and the config actually RUNS (peak RSS).
     This is the anti-#205 object: the #205 SEAL accepted a config on a borrowed ancestor
     d_pose with no runnability check, and it OOM-died. This class refuses that.

  2. ``variant_ii_accept`` — POWERPLAY Variant-II acceptance ``c*_pred - c_new > eps``
     (admit a modification iff aggregate cost strictly improves). This IS our
     compose-without-regression / "admit only when net-S improves" rule.

  3. ``simplest_unsolvable_rank`` — the ``K(task, solver | history)`` ordering: among
     candidate levers that close an unsolved residual, prefer the highest expected |ΔS|
     per (description + validation) cost — cheapest-to-describe-and-validate first. The
     principled ordering for the #216 automated witness instrument.

Deep cross-reference: ``.omx/research/powerplay_1112.5309_deep_crossref_20260702.md``.
CONTAINMENT: everything here is PURE / decision-only — it never launches, trains, or
spends. Reuses ``compute_contest_score`` (no hand-rolled score, per the canonical-scorer
non-negotiable).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tac.contest_score import compute_contest_score, pose_term, rate_term, seg_term


# ---------------------------------------------------------------------------
# Evidence grades — the authority ladder, fail-closed (the #205 killer set)
# ---------------------------------------------------------------------------
class EvidenceGrade(str, Enum):
    """How a scored quantity (d_seg / d_pose / archive_bytes) was obtained.

    Only the first three were measured THROUGH THE REAL BYTE-CLOSED DECODE. The rest are
    SURROGATES — the exact class of number that let the #205 config pass a full SEAL
    (an ancestor-RGB d_pose 3.4e-5 quoted for a vehicle where it was never measured). A
    surrogate is NEVER a valid Correctness-Demonstration input.
    """

    CONTEST_CUDA = "contest-cuda"                                # authority (promotion + seal)
    CONTEST_CPU = "contest-cpu"                                  # authority (promotion + seal)
    MACOS_CPU_ADVISORY_THROUGH_DECODE = "macos-cpu-advisory-through-decode"  # real inflate, advisory: seal-OK, NOT promotion
    # --- SURROGATES: never valid for a Correctness Demonstration ---
    PROXY = "proxy"                # a training/proxy loss, not the scored quantity
    ANCESTOR = "ancestor"          # borrowed from a different vehicle (the #205 pose bug)
    MPS = "mps"                    # MPS forward pass — NEVER a score authority
    TRAINING_SIDE = "training-side"  # measured in-loop, not through the byte-closed decode
    PREDICTED = "predicted"        # a derived/predicted band, not a measurement


# Grades that were measured through the real byte-closed decode.
_THROUGH_REAL_DECODE: frozenset[EvidenceGrade] = frozenset({
    EvidenceGrade.CONTEST_CUDA,
    EvidenceGrade.CONTEST_CPU,
    EvidenceGrade.MACOS_CPU_ADVISORY_THROUGH_DECODE,
})
# Grades that carry contest authority (a pointer move / promotion claim).
_PROMOTION_AUTHORITY: frozenset[EvidenceGrade] = frozenset({
    EvidenceGrade.CONTEST_CUDA,
    EvidenceGrade.CONTEST_CPU,
})


class DemonstrationLevel(str, Enum):
    """What the Correctness Demonstration is being asked to license.

    ``LOCAL_SEAL`` un-HOLDs a launch (axis-9): the scored quantities must be measured
    through the REAL decode; ``macos-cpu-advisory-through-decode`` is acceptable.
    ``PROMOTION`` moves the frontier pointer: contest-CPU / contest-CUDA authority is
    required (advisory is refused).
    """

    LOCAL_SEAL = "local_seal"
    PROMOTION = "promotion"


# ---------------------------------------------------------------------------
# 1. The POWERPLAY cost  (= the contest score S)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverCost:
    """A solver's POWERPLAY cost, decomposed into L(s) (rate) and the task deficit."""

    d_seg: float
    d_pose: float
    archive_bytes: int
    S: float                 # the total POWERPLAY cost = compute_contest_score
    description_bits_term: float   # L(s): the rate term 25*bytes/N
    task_deficit_term: float       # alpha*sum[..]: 100*d_seg + sqrt(10*d_pose)


def powerplay_cost(d_seg: float, d_pose: float, archive_bytes: int | float) -> SolverCost:
    """The contest score as a POWERPLAY Variant-II cost, decomposed.

    ``S = task_deficit_term + description_bits_term`` where ``description_bits_term`` is
    L(s) (the counted archive bytes) and ``task_deficit_term`` is the SegNet+PoseNet
    deficit. Delegates the total to the canonical ``compute_contest_score`` (no hand-rolled
    score).
    """
    rate = rate_term(int(round(float(archive_bytes))))
    deficit = seg_term(d_seg) + pose_term(d_pose)
    total = compute_contest_score(d_seg, d_pose, int(round(float(archive_bytes))))
    return SolverCost(
        d_seg=float(d_seg),
        d_pose=float(d_pose),
        archive_bytes=int(round(float(archive_bytes))),
        S=total,
        description_bits_term=rate,
        task_deficit_term=deficit,
    )


# ---------------------------------------------------------------------------
# 2. Correctness Demonstration  (the executable axis-9)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoredQuantity:
    """A single scored quantity (d_seg / d_pose / archive_bytes) + how it was obtained."""

    name: str
    value: float
    grade: EvidenceGrade


@dataclass(frozen=True)
class CorrectnessDemonstration:
    """POWERPLAY's Correctness Demonstration, in our units — the executable axis-9.

    A launch (``LOCAL_SEAL``) or a frontier promotion (``PROMOTION``) is ACCEPTED only when
    ``validate()`` returns ``[]``: every scored quantity was measured through the real
    byte-closed decode at the required authority, the config demonstrably RUNS (peak RSS
    under a control-plane-safe ceiling), and net-S does not regress the predecessor
    (Variant-II acceptance). Fail-closed: a surrogate grade (proxy / ancestor / MPS /
    training-side / predicted) is ALWAYS a violation — that is the #205 failure class.
    """

    label: str
    level: DemonstrationLevel
    d_seg: ScoredQuantity
    d_pose: ScoredQuantity
    archive_bytes: ScoredQuantity
    predecessor_S: float | None = None   # current best / pointer; None == first solver (no regression check)
    peak_rss_mb: float | None = None     # measured at the REAL config; None == not measured (a violation)
    rss_ceiling_mb: float | None = None  # control-plane-safe ceiling; None == skip the RSS check
    eps: float = 1e-5

    @property
    def new_S(self) -> float:
        return compute_contest_score(
            self.d_seg.value, self.d_pose.value, int(round(self.archive_bytes.value))
        )

    def _grade_ok(self, q: ScoredQuantity) -> bool:
        if self.level == DemonstrationLevel.PROMOTION:
            return q.grade in _PROMOTION_AUTHORITY
        return q.grade in _THROUGH_REAL_DECODE

    def validate(self) -> list[str]:
        """Return the list of fail-closed violations (empty == ACCEPTED)."""
        v: list[str] = []
        need = ("contest-CPU/CUDA authority" if self.level == DemonstrationLevel.PROMOTION
                else "measured through the real byte-closed decode (advisory OK)")
        for q in (self.d_seg, self.d_pose, self.archive_bytes):
            if not self._grade_ok(q):
                v.append(
                    f"scored quantity {q.name!r} has grade {q.grade.value!r} — a SURROGATE for "
                    f"{self.level.value}; requires {need} (the #205 borrowed-number failure)"
                )
        # Runnability (axis-9 (a)): a LOCAL_SEAL must PROVE the config runs — peak RSS
        # MEASURED at the real config (a B=8 throughput bench is a surrogate, not the
        # authority). A PROMOTION's archive already ran + was contest-scored, so runnability
        # is implied; only enforce the ceiling if a measurement was supplied.
        if self.level == DemonstrationLevel.LOCAL_SEAL and self.peak_rss_mb is None:
            v.append("runnability UNPROVEN: peak_rss_mb was not measured at the real config "
                     "(a B=8 throughput bench is a surrogate, not the authority)")
        elif (self.peak_rss_mb is not None and self.rss_ceiling_mb is not None
              and self.peak_rss_mb > self.rss_ceiling_mb):
            v.append(
                f"runnability FAILS: peak_rss_mb={self.peak_rss_mb:.0f} exceeds the "
                f"control-plane-safe ceiling {self.rss_ceiling_mb:.0f} MB (the #205 OOM class)"
            )
        # No-regression (axis-9 (b) / Variant-II acceptance): net-S must strictly improve.
        if self.predecessor_S is not None:
            ok, margin = variant_ii_accept(self.new_S, self.predecessor_S, eps=self.eps)
            if not ok:
                v.append(
                    f"REGRESSION: new S={self.new_S:.6f} does not improve on predecessor "
                    f"S={self.predecessor_S:.6f} by eps={self.eps:g} (margin={margin:+.6f})"
                )
        return v

    @property
    def accepted(self) -> bool:
        return not self.validate()


# ---------------------------------------------------------------------------
# 3. Variant-II acceptance  (c*_pred - c_new > eps)
# ---------------------------------------------------------------------------
def variant_ii_accept(cost_new: float, cost_predecessor: float, *, eps: float = 1e-5) -> tuple[bool, float]:
    """POWERPLAY Variant-II acceptance: admit a modification iff aggregate cost strictly
    improves. Returns ``(accepted, margin)`` with ``margin = cost_predecessor - cost_new``
    (positive == improvement). This is our compose-without-regression / net-S gate.
    """
    margin = float(cost_predecessor) - float(cost_new)
    return (margin > eps), margin


# ---------------------------------------------------------------------------
# 4. Simplest-still-unsolvable ordering  (the #216 instrument's next())
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LeverCandidate:
    """A candidate solver-modification (lever) for the ordering.

    ``expected_delta_S`` is the SIGNED expected change to S (negative == improvement).
    ``description_bits`` is L(q): the counted bytes the lever adds to the archive (0 for a
    byte-free / train-time-only lever). ``validation_cost`` is the relative cost to run the
    Correctness Demonstration for this lever (>= 1; e.g. a lever that couples many facets
    and needs a full re-measure is more expensive than a facet-orthogonal one).
    """

    label: str
    expected_delta_S: float
    description_bits: float = 0.0
    validation_cost: float = 1.0


def simplest_unsolvable_rank(
    candidates: list[LeverCandidate], *, bit_weight: float = 1.0
) -> list[tuple[LeverCandidate, float]]:
    """Rank candidate levers by the POWERPLAY ``K(task, solver | history)`` ordering.

    Among candidates that CLOSE an unsolved residual (``expected_delta_S < 0``), prefer the
    highest expected improvement per unit of (description + validation) cost — the
    K-complexity-biased "cheapest-to-describe-and-validate that still advances the solver"
    ordering. Non-improving candidates (``expected_delta_S >= 0``) are ranked last (score 0),
    preserving order among themselves. Pure + deterministic (no RNG / wall-clock / GPU), so
    the #216 instrument's selection is reproducible.

    Returns ``[(candidate, score), ...]`` sorted by descending score. Higher score = pick first.
    """
    scored: list[tuple[LeverCandidate, float]] = []
    for c in candidates:
        improvement = -float(c.expected_delta_S)  # positive == improves S
        if improvement <= 0.0:
            scored.append((c, 0.0))
            continue
        # cost = validation-cost + bit_weight * description-bits (both >= 0); +1 avoids /0 and
        # gives a byte-free, facet-orthogonal lever the strongest score (POWERPLAY's compact +
        # quickly-validated bias).
        cost = max(0.0, float(c.validation_cost)) + bit_weight * max(0.0, float(c.description_bits))
        score = improvement / (1.0 + cost)
        scored.append((c, score))
    # stable sort: improving candidates by descending score; ties + non-improving keep input order.
    return sorted(scored, key=lambda t: t[1], reverse=True)


__all__ = [
    "EvidenceGrade",
    "DemonstrationLevel",
    "SolverCost",
    "powerplay_cost",
    "ScoredQuantity",
    "CorrectnessDemonstration",
    "variant_ii_accept",
    "LeverCandidate",
    "simplest_unsolvable_rank",
]
