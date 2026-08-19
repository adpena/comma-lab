# SPDX-License-Identifier: MIT
"""F1 -- REALIZED-ACCEPTANCE LATTICE COORDINATE DESCENT, as one typed engine.

THE PATTERN, AND WHY IT WINS
----------------------------
Four arms built the same loop independently, and it is the engine behind the 13th
pointer move and behind ``ddm_jg1``'s seg repair::

    propose (from exact structure) -> apply -> REAL decode -> accept iff realized joint dS < 0

The discipline that makes it work is *negative*: *nothing is accepted on a prediction*.
No linearisation, no surrogate loss, no proxy metric.  A candidate is rendered through
the receiver's own forward path and scored by the frozen scorer, so an accepted step is
a step that **actually happened**.  ``ddm_jg1`` states the reason in one line: a
pre-distortion move "creates new flips as well as repairing old ones, and only the
frozen scorer knows the net."

The stopping rule is likewise a *proof*, not a budget: the descent halts when a full
sweep of the coordinate's lattice neighbours finds no improving candidate.  An iteration
cap is available but is an escape hatch, and a run that hits it is reported
``converged=False`` so no verdict can quietly rest on a truncated descent.

THE FOUR SIBLINGS THIS GENERALISES
----------------------------------
``ddm_up2``    12 int12 carrier coefficients per pair; neighbours at offsets +-1,+-2;
               objective = d_pose vs DALI targets through the real render + PoseNet.
``ddm_jg1``    a 384x512 token map per pair; proposals are GT-labelled disks of radius r;
               objective = SegNet argmax flips through the real SemanticTokenRenderer.
``ddm_me1``    micro-edits on the token stream.
``ddm_tq1``    token move engine.

All four carry ``ndarray`` state, all four evaluate in batches, and all four are
per-coordinate independent.  This engine is typed to that shape rather than to a maximal
abstraction: the state is an ``ndarray``, which makes checkpointing exact and makes the
generic path the same code the siblings already run.

WHAT IS INJECTED, AND WHAT IS FIXED
-----------------------------------
Injected per family (UNIQUE-AND-COMPLETE-PER-METHOD -- the engine standardises the
*loop*, never the family's math):

* ``coordinate_set``    -- which coordinates are swept, and in what order
* ``proposal_generator``-- the family's structural move set
* ``realization_path``  -- the family's real decode
* ``joint_objective``   -- the family's score-unit objective
* ``acceptance_rule``   -- the improvement threshold

Fixed by the engine, because these are the parts that were re-derived four times and are
where the failures live:

* acceptance is REALIZED-ONLY (a ranker may order proposals; it may never accept one)
* the stop is a no-improving-neighbour sweep (the convergence proof)
* every accept AND reject is emitted as a labelled example (free training data)
* checkpoint/resume is per-coordinate and atomic (P0: every launch resumes from disk)

THE JOINT OBJECTIVE IS IN SCORE UNITS
-------------------------------------
``ddm_up2`` descends d_pose; ``ddm_jg1`` counts seg flips.  Neither is directly
comparable, and a seg repair that costs rate is not admissible just because flips fell.
So the engine's objective is contracted to return **score units** -- the units of
``upstream/evaluate.py:92``, owned by :mod:`tac.contest_score` -- and a family adapts its
native quantity into those units.  That is what makes "accept iff realized joint dS < 0" a single rule rather
than four incompatible ones.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "AcceptanceEvent",
    "AcceptanceRule",
    "CoordinateOutcome",
    "DescentReport",
    "JointObjective",
    "Proposal",
    "ProposalGenerator",
    "RealizationPath",
    "RealizedAcceptanceEngine",
    "RealizedAcceptanceError",
    "TrainingExampleSink",
]


class RealizedAcceptanceError(RuntimeError):
    """An engine precondition failed.  Always fail closed."""


# ---------------------------------------------------------------------------
# The injected contracts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One candidate move on one coordinate.

    ``state`` is the FULL proposed state for that coordinate, not a delta -- the
    realization path must never have to reconstruct what was meant.
    """

    coordinate: int
    label: str
    state: np.ndarray
    #: Optional ordering hint from a ranker.  Advisory ONLY: a ranker may reorder or
    #: truncate the candidate list, and can therefore change which optimum is reached
    #: and how fast, but it can never cause a non-improving candidate to be accepted.
    rank_score: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, np.ndarray):
            raise RealizedAcceptanceError(
                f"proposal {self.label!r} carries state of type "
                f"{type(self.state).__name__}; the engine is typed to ndarray state"
            )


@runtime_checkable
class ProposalGenerator(Protocol):
    """Emit the family's structural moves for one coordinate."""

    def __call__(self, state: np.ndarray, coordinate: int) -> Sequence[Proposal]:
        ...


@runtime_checkable
class RealizationPath(Protocol):
    """Realize proposed states through the family's REAL decode.

    Contract: the returned object is whatever :class:`JointObjective` consumes, and it
    must be produced by the receiver's own forward path -- not a re-implementation.  The
    engine cannot verify this, so each family asserts it with a byte-exactness control
    (``ddm_up2.validate_forward_model``, ``ddm_jg1.forward_model_control``) before the
    descent is trusted at all.
    """

    def __call__(self, proposals: Sequence[Proposal], coordinate: int) -> Any:
        ...


@runtime_checkable
class JointObjective(Protocol):
    """Score realized candidates, in SCORE UNITS, lower-is-better.

    Returns one float per proposal, aligned with the input order.
    """

    def __call__(self, realized: Any, proposals: Sequence[Proposal], coordinate: int) -> np.ndarray:
        ...


@dataclass(frozen=True)
class AcceptanceRule:
    """When a realized candidate may replace the incumbent.

    ``min_improvement`` is in SCORE UNITS and defaults to ``0.0`` with strict
    comparison, which reproduces the siblings' ``values[winner] >= best -> stop``.
    Setting it above zero is how a family refuses moves too small to survive its own
    reporting bound (see ``tac.local_contest_instruments.pose_report_bound``).
    """

    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_improvement) or self.min_improvement < 0.0:
            raise RealizedAcceptanceError(
                f"min_improvement must be finite and >= 0, got {self.min_improvement}"
            )

    def accepts(self, incumbent: float, candidate: float) -> bool:
        """True iff ``candidate`` improves on ``incumbent`` by at least the threshold."""
        return (incumbent - candidate) > self.min_improvement


# ---------------------------------------------------------------------------
# The free training data.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceEvent:
    """One labelled example: a proposal, its realized value, and the verdict.

    Every sweep of the engine produces these for ACCEPTED and REJECTED alike, which is
    the point -- a learned proposal ranker needs both classes, and a realized-acceptance
    run generates them at zero marginal cost because the realization already happened.
    ``rank_score`` is recorded so a ranker's ordering can be scored after the fact
    against what the frozen scorer actually said.
    """

    coordinate: int
    label: str
    pass_index: int
    incumbent_value: float
    realized_value: float
    delta: float
    accepted: bool
    rank_score: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class TrainingExampleSink(Protocol):
    """Consume acceptance events.  Any sink: JSONL writer, list, learned-ranker trainer."""

    def __call__(self, event: AcceptanceEvent) -> None:
        ...


class JsonlEventSink:
    """Append acceptance events to a JSONL file.  The default free-training-data path."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: AcceptanceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_json(), sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Reports.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordinateOutcome:
    """The descent result for one coordinate."""

    coordinate: int
    start_value: float
    final_value: float
    passes: int
    realizations: int
    converged: bool
    accepted_moves: int
    state: np.ndarray
    history: tuple[float, ...]

    @property
    def improvement(self) -> float:
        """Score units removed by this coordinate.  Positive means better."""
        return self.start_value - self.final_value

    def to_json(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "start_value": self.start_value,
            "final_value": self.final_value,
            "improvement": self.improvement,
            "passes": self.passes,
            "realizations": self.realizations,
            "converged": self.converged,
            "accepted_moves": self.accepted_moves,
            "state": np.asarray(self.state).tolist(),
            "history": list(self.history),
        }


@dataclass(frozen=True)
class DescentReport:
    """The whole run.  ``converged`` is AND-ed: one truncated coordinate taints the run."""

    outcomes: tuple[CoordinateOutcome, ...]
    wall_clock_seconds: float
    resumed_coordinates: int = 0

    @property
    def converged(self) -> bool:
        """AND over coordinates, and FALSE on an empty run.

        ``all([])`` is ``True``, so a descent that measured nothing would otherwise
        report itself converged -- the vacuity-equals-pass failure, where a skipped
        instrument reads as green.  A run with no outcomes has proved nothing.
        """
        if not self.outcomes:
            return False
        return all(outcome.converged for outcome in self.outcomes)

    @property
    def total_improvement(self) -> float:
        """Total score units removed, summed over coordinates.

        Summing is valid only when coordinates are INDEPENDENT under the objective --
        which is the structural precondition every sibling establishes separately
        (``ddm_up2``: pairs are independent; ``ddm_jg1``: per-pair token maps).  The
        engine records the sum; the family owns the independence claim.
        """
        return sum(outcome.improvement for outcome in self.outcomes)

    @property
    def total_realizations(self) -> int:
        return sum(outcome.realizations for outcome in self.outcomes)

    def to_json(self) -> dict[str, Any]:
        return {
            "coordinates": len(self.outcomes),
            "converged": self.converged,
            "total_improvement_score_units": self.total_improvement,
            "total_realizations": self.total_realizations,
            "wall_clock_seconds": self.wall_clock_seconds,
            "resumed_coordinates": self.resumed_coordinates,
            "independence_note": (
                "total_improvement sums per-coordinate improvements; that sum is a score "
                "delta only if the family has established coordinate independence under "
                "the objective. The engine does not verify independence."
            ),
            "outcomes": [outcome.to_json() for outcome in self.outcomes],
        }


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------


class RealizedAcceptanceEngine:
    """Coordinate descent whose every acceptance is a realized measurement.

    Args:
        proposal_generator: the family's structural move set.
        realization_path: the family's REAL decode.
        joint_objective: score-unit objective, lower-is-better.
        acceptance_rule: improvement threshold.
        ranker: optional advisory proposal ordering (see
            :mod:`tac.win_families.proposal_rankers`).  It may reorder and truncate;
            it may NEVER accept.
        event_sink: optional consumer of the free labelled examples.
        max_passes: 0 means run to the convergence proof.  Any positive value is a cap,
            and a coordinate that hits it is reported ``converged=False``.
        checkpoint_path: JSONL of completed coordinates.  Present coordinates are
            skipped on resume (P0: every launch resumes from disk).
    """

    def __init__(
        self,
        *,
        proposal_generator: ProposalGenerator,
        realization_path: RealizationPath,
        joint_objective: JointObjective,
        acceptance_rule: AcceptanceRule | None = None,
        ranker: Callable[[Sequence[Proposal], int], Sequence[Proposal]] | None = None,
        event_sink: TrainingExampleSink | None = None,
        max_passes: int = 0,
        checkpoint_path: Path | None = None,
    ) -> None:
        if max_passes < 0:
            raise RealizedAcceptanceError(f"max_passes must be >= 0, got {max_passes}")
        self.proposal_generator = proposal_generator
        self.realization_path = realization_path
        self.joint_objective = joint_objective
        self.acceptance_rule = acceptance_rule or AcceptanceRule()
        self.ranker = ranker
        self.event_sink = event_sink
        self.max_passes = max_passes
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    # -- checkpointing ------------------------------------------------------

    def completed_coordinates(self) -> dict[int, dict[str, Any]]:
        """Coordinates already finished, read from the checkpoint. Empty when absent."""
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            return {}
        done: dict[int, dict[str, Any]] = {}
        for line in self.checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a kill; the coordinate simply re-runs
            if "coordinate" in row:
                done[int(row["coordinate"])] = row
        return done

    def _checkpoint(self, outcome: CoordinateOutcome) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(outcome.to_json(), sort_keys=True) + "\n")
            stream.flush()

    # -- the loop -----------------------------------------------------------

    def _realize_and_score(
        self, proposals: Sequence[Proposal], coordinate: int
    ) -> np.ndarray:
        if not proposals:
            return np.zeros(0, dtype=np.float64)
        realized = self.realization_path(proposals, coordinate)
        values = np.asarray(
            self.joint_objective(realized, proposals, coordinate), dtype=np.float64
        )
        if values.shape != (len(proposals),):
            raise RealizedAcceptanceError(
                f"joint_objective returned shape {values.shape} for {len(proposals)} "
                "proposals; the contract is one score-unit value per proposal, in order"
            )
        if not np.all(np.isfinite(values)):
            raise RealizedAcceptanceError(
                f"joint_objective returned a non-finite value on coordinate {coordinate}; "
                "a realized objective must be finite or the acceptance test is undefined"
            )
        return values

    def descend_coordinate(self, state: np.ndarray, coordinate: int) -> CoordinateOutcome:
        """Descend one coordinate to its convergence proof (or to ``max_passes``)."""
        current = np.array(state, copy=True)
        incumbent_proposal = Proposal(coordinate=coordinate, label="incumbent", state=current)
        best = float(self._realize_and_score([incumbent_proposal], coordinate)[0])
        start_value = best
        history = [best]
        realizations = 1
        accepted_moves = 0
        passes = 0
        converged = True

        while True:
            passes += 1
            proposals = list(self.proposal_generator(current, coordinate))
            if self.ranker is not None and proposals:
                proposals = list(self.ranker(proposals, coordinate))
            if not proposals:
                break
            values = self._realize_and_score(proposals, coordinate)
            realizations += len(proposals)
            winner = int(np.argmin(values))
            winning_value = float(values[winner])
            accepted = self.acceptance_rule.accepts(best, winning_value)

            if self.event_sink is not None:
                for index, proposal in enumerate(proposals):
                    self.event_sink(
                        AcceptanceEvent(
                            coordinate=coordinate,
                            label=proposal.label,
                            pass_index=passes,
                            incumbent_value=best,
                            realized_value=float(values[index]),
                            delta=float(values[index]) - best,
                            accepted=bool(index == winner and accepted),
                            rank_score=proposal.rank_score,
                        )
                    )

            if not accepted:
                break  # the convergence proof: no improving neighbour exists
            best = winning_value
            current = np.array(proposals[winner].state, copy=True)
            history.append(best)
            accepted_moves += 1
            if self.max_passes and passes >= self.max_passes:
                converged = False
                break

        return CoordinateOutcome(
            coordinate=coordinate,
            start_value=start_value,
            final_value=best,
            passes=passes,
            realizations=realizations,
            converged=converged,
            accepted_moves=accepted_moves,
            state=current,
            history=tuple(history),
        )

    def descend(
        self, states: dict[int, np.ndarray] | Sequence[tuple[int, np.ndarray]],
        *,
        coordinates: Iterable[int] | None = None,
        progress: bool = False,
    ) -> DescentReport:
        """Descend every coordinate, resuming any already in the checkpoint."""
        items = dict(states)
        order = list(coordinates) if coordinates is not None else sorted(items)
        done = self.completed_coordinates()
        outcomes: list[CoordinateOutcome] = []
        resumed = 0
        started = time.time()

        for coordinate in order:
            if coordinate in done:
                row = done[coordinate]
                outcomes.append(
                    CoordinateOutcome(
                        coordinate=coordinate,
                        start_value=float(row["start_value"]),
                        final_value=float(row["final_value"]),
                        passes=int(row["passes"]),
                        realizations=int(row["realizations"]),
                        converged=bool(row["converged"]),
                        accepted_moves=int(row.get("accepted_moves", 0)),
                        state=np.asarray(row["state"]),
                        history=tuple(float(value) for value in row.get("history", ())),
                    )
                )
                resumed += 1
                continue
            if coordinate not in items:
                raise RealizedAcceptanceError(
                    f"coordinate {coordinate} was requested but no start state was supplied"
                )
            outcome = self.descend_coordinate(items[coordinate], coordinate)
            self._checkpoint(outcome)
            outcomes.append(outcome)
            if progress:
                print(
                    f"  coord {coordinate}: {outcome.start_value:.6g} -> "
                    f"{outcome.final_value:.6g} ({outcome.accepted_moves} accepted, "
                    f"{outcome.realizations} realizations)",
                    flush=True,
                )

        return DescentReport(
            outcomes=tuple(outcomes),
            wall_clock_seconds=time.time() - started,
            resumed_coordinates=resumed,
        )


# ---------------------------------------------------------------------------
# The generic lattice move set, shared by every sibling that has one.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatticeNeighbourGenerator:
    """Single-coordinate integer-lattice neighbours, clamped to the representable box.

    This is ``ddm_up2.candidate_codes_for_pair`` generalised: for a state of ``n``
    integer codes it proposes every ``(index, offset)`` move that stays inside
    ``[low, high]``.  Moves that would leave the box are not proposed at all -- a
    clamped duplicate of the incumbent would waste a realization and, worse, would
    register as a tie rather than as an unavailable move.
    """

    offsets: tuple[int, ...] = (-2, -1, 1, 2)
    low: int = -2048
    high: int = 2047

    def __post_init__(self) -> None:
        if not self.offsets:
            raise RealizedAcceptanceError("lattice generator needs at least one offset")
        if 0 in self.offsets:
            raise RealizedAcceptanceError(
                "offset 0 is the incumbent; proposing it wastes a realization and ties"
            )
        if self.low > self.high:
            raise RealizedAcceptanceError(f"empty box: low={self.low} > high={self.high}")

    def __call__(self, state: np.ndarray, coordinate: int) -> list[Proposal]:
        proposals: list[Proposal] = []
        flat = np.asarray(state)
        for index in range(flat.size):
            for offset in self.offsets:
                value = int(flat.flat[index]) + int(offset)
                if not self.low <= value <= self.high:
                    continue
                trial = np.array(flat, copy=True)
                trial.flat[index] = value
                proposals.append(
                    Proposal(
                        coordinate=coordinate,
                        label=f"d{index}{offset:+d}",
                        state=trial,
                    )
                )
        return proposals
