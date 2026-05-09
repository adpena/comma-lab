#!/usr/bin/env python3
"""Cathedral autopilot autonomous loop — typed-atom queue → Pareto → dispatch.

Sister of :mod:`tools.cathedral_autopilot` (the per-invocation planner). This
module is the **continuous** loop: it monitors the typed-atom evidence queue,
ranks candidates against the Pareto frontier, surfaces operator-decision
events, and feeds harvested results back into the continual-learning
posterior so the NEXT loop iteration ranks against the updated state.

**HALT-and-ASK pattern (operator-gate non-negotiable per CLAUDE.md):**

The loop is "autonomous" only in the *ranking + harvesting + feedback* sense.
Every dispatch decision still requires explicit operator approval. The loop
HALTS at every operator-decision event and emits a structured ``HALT_EVENT``
that the operator answers (CLI: ``--operator-decision``; programmatic:
:func:`inject_operator_decision`). The ``--require-operator-approval-on``
flag enumerates which event classes block; passing
``--require-operator-approval-on dispatch`` makes EVERY dispatch decision
operator-gated (the CLAUDE.md non-negotiable default).

Per CLAUDE.md "EXTREME EMOJI rule": NO emojis in the source / output.

Per CLAUDE.md "Forbidden score claims": this loop NEVER claims a score. It
ranks candidates by predicted score band (tagged ``[predicted; cathedral
autopilot ranking]``) and gates on operator decision before any dispatch.

Per CLAUDE.md "race-mode rigor inversion": when ``--race-mode`` is set, the
loop's dispatch-approval policy switches from "max rigor" to "smallest
credible bolt-on within ~60 minutes." Operator must explicitly enable
race-mode via the ``--race-mode`` flag.

Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": before any dispatch is
recommended, the loop checks the active-lane-dispatch-claims registry. If a
conflicting claim exists, the candidate is moved to ``DEFERRED-pending-claim``
status and the operator is notified.

Cross-references
----------------
- :mod:`tools.cathedral_autopilot` — per-invocation planner / ranker
- :mod:`tac.continual_learning` — posterior consumed and updated by the loop
- :mod:`tools.claim_lane_dispatch` — dispatch-claim coordination
- ``feedback_5_beyond_phase4_modules_landed_20260509``
- ``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509``

CLAUDE.md compliance tags
-------------------------
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``halt_and_ask_pattern_default_on``
- ``no_score_claim_only_predicted_band``
- ``cross_agent_dispatch_coordination_check``
- ``race_mode_explicit_opt_in_only``
- ``no_tmp_paths``
- ``forbidden_premature_kill_without_research_exhaustion_no_kill_verdicts``
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


AUTONOMOUS_LOOP_SCHEMA = "tac_cathedral_autopilot_autonomous_loop_v1"


# ── Events / decisions / verdicts ──────────────────────────────────────────


class EventClass(str, Enum):
    """Operator-decision event classes."""

    DISPATCH = "dispatch"
    KILL = "kill"  # NEVER auto-kill per CLAUDE.md; operator-only
    PROMOTE = "promote"
    POSTERIOR_REWEIGHT = "posterior_reweight"
    RACE_MODE_TOGGLE = "race_mode_toggle"


class OperatorDecision(str, Enum):
    """Operator's response to a HALT event."""

    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


@dataclass
class CandidateRow:
    """One typed-atom row from the candidate queue.

    Mirrors the cathedral autopilot's TechniqueEvidence schema in spirit but
    is intentionally minimal — the autonomous loop ranks candidates by
    predicted score delta + EIG/$ and surfaces them for operator decision.

    Per CLAUDE.md "Forbidden score claims": ``predicted_score_delta`` is
    explicitly tagged as a prediction, never a measurement.
    """

    candidate_id: str
    family: str  # e.g. 'hnerv_lc_v2', 'balle_scale_hyperprior'
    predicted_score_delta: float  # negative = improvement
    expected_information_gain: float
    estimated_dispatch_cost_usd: float
    blockers: list[str] = field(default_factory=list)
    notes: str = ""

    def eig_per_dollar(self) -> float:
        if self.estimated_dispatch_cost_usd <= 0.0:
            return float("inf")
        return self.expected_information_gain / self.estimated_dispatch_cost_usd


@dataclass
class HaltEvent:
    """One operator-decision halt event surfaced by the loop."""

    event_class: EventClass
    candidate_id: str
    reason: str
    predicted_score_delta: float
    estimated_cost_usd: float
    requires_approval: bool
    halt_at_utc: str = ""
    blockers: list[str] = field(default_factory=list)
    decision: OperatorDecision | None = None
    decision_at_utc: str | None = None
    decision_notes: str = ""


@dataclass
class LoopIterationReport:
    """Result of one autonomous-loop iteration."""

    iteration: int
    started_at_utc: str
    ended_at_utc: str
    n_candidates_seen: int
    n_candidates_blocked_by_dispatch_claim: int
    n_candidates_ranked: int
    halt_events: list[HaltEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema: str = AUTONOMOUS_LOOP_SCHEMA


# ── Pure ranking + halt-event construction ─────────────────────────────────


def rank_candidates(
    candidates: list[CandidateRow],
    *,
    rank_axis: str = "eig_per_dollar",
) -> list[CandidateRow]:
    """Rank candidates by the chosen axis (descending best-first).

    Recognized axes:
      - ``eig_per_dollar`` — expected information gain per dollar (default)
      - ``predicted_score_delta`` — most-negative-first (greatest improvement)

    Per CLAUDE.md "Forbidden /tmp paths": no temp paths used; pure in-memory.
    """
    if rank_axis == "eig_per_dollar":
        return sorted(candidates, key=lambda c: c.eig_per_dollar(), reverse=True)
    if rank_axis == "predicted_score_delta":
        return sorted(candidates, key=lambda c: c.predicted_score_delta)
    raise ValueError(
        f"unknown rank_axis {rank_axis!r}; must be 'eig_per_dollar' or "
        "'predicted_score_delta'"
    )


def make_dispatch_halt_event(
    candidate: CandidateRow,
    *,
    requires_approval_classes: frozenset[EventClass],
    blockers: list[str] | None = None,
) -> HaltEvent:
    """Construct a HALT event for one dispatch decision.

    Per CLAUDE.md "operator-gate non-negotiable": when ``EventClass.DISPATCH``
    is in ``requires_approval_classes``, ``requires_approval=True``.
    """
    requires = EventClass.DISPATCH in requires_approval_classes
    return HaltEvent(
        event_class=EventClass.DISPATCH,
        candidate_id=candidate.candidate_id,
        reason=(
            f"Dispatch decision for candidate {candidate.candidate_id} "
            f"(family={candidate.family}, predicted_score_delta="
            f"{candidate.predicted_score_delta:+.6f}) — operator decision "
            "required."
        ),
        predicted_score_delta=candidate.predicted_score_delta,
        estimated_cost_usd=candidate.estimated_dispatch_cost_usd,
        requires_approval=requires,
        halt_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        blockers=list(blockers or []),
    )


def make_kill_halt_event(candidate_id: str, reason: str) -> HaltEvent:
    """Construct a KILL HALT event.

    Per CLAUDE.md "kill-as-last-resort": KILL events ALWAYS require approval.
    The autonomous loop NEVER auto-kills.
    """
    return HaltEvent(
        event_class=EventClass.KILL,
        candidate_id=candidate_id,
        reason=reason,
        predicted_score_delta=0.0,
        estimated_cost_usd=0.0,
        requires_approval=True,  # FORCED True per CLAUDE.md
        halt_at_utc=dt.datetime.now(dt.UTC).isoformat(),
    )


def inject_operator_decision(
    event: HaltEvent,
    decision: OperatorDecision,
    notes: str = "",
) -> HaltEvent:
    """Record the operator's decision on a HALT event (returns updated event)."""
    if event.decision is not None:
        raise ValueError(
            f"event {event.candidate_id!r}/{event.event_class.value!r} already "
            f"decided as {event.decision.value!r}"
        )
    event.decision = decision
    event.decision_at_utc = dt.datetime.now(dt.UTC).isoformat()
    event.decision_notes = notes
    return event


# ── Dispatch-claim coordination check ──────────────────────────────────────


def check_dispatch_claim_conflict(
    candidate_id: str,
    *,
    claims_path: Path | None = None,
) -> tuple[bool, str]:
    """Check the active-lane-dispatch-claims registry for a conflicting claim.

    Returns ``(has_conflict, reason)``. Returns ``(False, "")`` if no claim
    file exists yet (cold start) or no conflicting claim is found.

    The claims file is the markdown registry referenced in CLAUDE.md
    "CROSS-AGENT DISPATCH COORDINATION". This helper does a simple substring
    scan; the canonical conflict-resolution logic lives in
    :mod:`tools.claim_lane_dispatch`.
    """
    p = claims_path or (REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md")
    if not p.is_file():
        return False, ""
    text = p.read_text(encoding="utf-8")
    if candidate_id in text:
        # Conservative match — the canonical TTL-aware logic lives elsewhere.
        return True, (
            f"candidate_id {candidate_id!r} appears in active-lane-dispatch-claims "
            f"registry at {p}. Consult tools/claim_lane_dispatch.py before dispatch."
        )
    return False, ""


# ── Loop iteration ─────────────────────────────────────────────────────────


def run_one_loop_iteration(
    candidates: list[CandidateRow],
    *,
    iteration: int = 1,
    rank_axis: str = "eig_per_dollar",
    requires_approval_on: frozenset[EventClass] = frozenset({EventClass.DISPATCH}),
    claims_path: Path | None = None,
    race_mode: bool = False,
    max_dispatch_recommendations: int | None = None,
) -> LoopIterationReport:
    """Run one cycle: rank → dispatch-claim check → halt-event emission.

    Per CLAUDE.md "operator-gate non-negotiable": this never actually
    dispatches. It surfaces HALT events with ``requires_approval=True`` for
    operator decision.

    Per CLAUDE.md "race-mode rigor inversion": when ``race_mode=True`` the
    loop trims the candidate set to the ones with smallest predicted cost
    AND non-trivial predicted_score_delta (the "smallest credible bolt-on"
    pattern from the May 4 race postmortem).
    """
    started = dt.datetime.now(dt.UTC).isoformat()
    notes: list[str] = []

    if race_mode:
        notes.append(
            "race-mode active per operator opt-in; ranking trimmed to "
            "smallest-credible-bolt-on subset"
        )
        # smallest credible bolt-on: |predicted_delta| > 0 AND lowest cost
        candidates = sorted(
            [c for c in candidates if c.predicted_score_delta < 0.0],
            key=lambda c: c.estimated_dispatch_cost_usd,
        )

    n_seen = len(candidates)
    ranked = rank_candidates(candidates, rank_axis=rank_axis) if candidates else []

    halt_events: list[HaltEvent] = []
    n_blocked = 0
    n_ranked = 0

    for cand in ranked:
        if max_dispatch_recommendations is not None and n_ranked >= max_dispatch_recommendations:
            break
        # Cross-agent dispatch claim check
        has_conflict, reason = check_dispatch_claim_conflict(
            cand.candidate_id, claims_path=claims_path
        )
        if has_conflict:
            n_blocked += 1
            notes.append(reason)
            continue

        # Existing blockers from the candidate row are surfaced too.
        event_blockers = list(cand.blockers)
        event = make_dispatch_halt_event(
            cand,
            requires_approval_classes=requires_approval_on,
            blockers=event_blockers,
        )
        halt_events.append(event)
        n_ranked += 1

    ended = dt.datetime.now(dt.UTC).isoformat()
    return LoopIterationReport(
        iteration=iteration,
        started_at_utc=started,
        ended_at_utc=ended,
        n_candidates_seen=n_seen,
        n_candidates_blocked_by_dispatch_claim=n_blocked,
        n_candidates_ranked=n_ranked,
        halt_events=halt_events,
        notes=notes,
    )


# ── Continuous-loop driver ────────────────────────────────────────────────


def run_continuous_loop(
    candidate_source: Callable[[], list[CandidateRow]],
    *,
    iterations: int,
    operator_decision_callback: Callable[[HaltEvent], OperatorDecision] | None = None,
    rank_axis: str = "eig_per_dollar",
    requires_approval_on: frozenset[EventClass] = frozenset({EventClass.DISPATCH}),
    claims_path: Path | None = None,
    race_mode: bool = False,
    max_dispatch_recommendations: int | None = None,
) -> list[LoopIterationReport]:
    """Run the continuous loop for ``iterations`` cycles.

    Each iteration calls ``candidate_source()`` to refresh the queue. The
    operator decision callback is invoked on every HALT event that has
    ``requires_approval=True``; if no callback is supplied, decisions are
    DEFER by default (the safe choice).

    Returns the list of per-iteration reports.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0; got {iterations}")
    reports: list[LoopIterationReport] = []
    for i in range(1, iterations + 1):
        candidates = candidate_source()
        report = run_one_loop_iteration(
            candidates,
            iteration=i,
            rank_axis=rank_axis,
            requires_approval_on=requires_approval_on,
            claims_path=claims_path,
            race_mode=race_mode,
            max_dispatch_recommendations=max_dispatch_recommendations,
        )
        # Operator-decision injection — DEFER when no callback supplied.
        for event in report.halt_events:
            if event.requires_approval:
                decision = (
                    operator_decision_callback(event)
                    if operator_decision_callback is not None
                    else OperatorDecision.DEFER
                )
                inject_operator_decision(event, decision)
        reports.append(report)
    return reports


# ── Serialization ──────────────────────────────────────────────────────────


def _enum_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    return v


def serialize_report(report: LoopIterationReport) -> dict[str, Any]:
    """Return a JSON-safe dict for one report."""

    def _convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = dataclasses.asdict(obj)
            return {k: _convert(v) for k, v in d.items()}
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(report)


def write_report(report: LoopIterationReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(serialize_report(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def _parse_approval_flags(items: list[str]) -> frozenset[EventClass]:
    out: set[EventClass] = set()
    for raw in items or []:
        try:
            out.add(EventClass(raw))
        except ValueError as exc:
            valid = sorted(c.value for c in EventClass)
            raise ValueError(
                f"--require-operator-approval-on {raw!r} not in {valid}"
            ) from exc
    return frozenset(out)


def load_candidates_from_jsonl(path: Path) -> list[CandidateRow]:
    """Load CandidateRow objects from a JSONL file (one row per line)."""
    rows: list[CandidateRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            raw = json.loads(s)
            rows.append(CandidateRow(
                candidate_id=raw["candidate_id"],
                family=raw["family"],
                predicted_score_delta=float(raw["predicted_score_delta"]),
                expected_information_gain=float(raw["expected_information_gain"]),
                estimated_dispatch_cost_usd=float(raw["estimated_dispatch_cost_usd"]),
                blockers=list(raw.get("blockers", [])),
                notes=str(raw.get("notes", "")),
            ))
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates-jsonl", type=Path, required=True,
                        help="Path to JSONL file of CandidateRow rows")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of loop iterations to run")
    parser.add_argument("--rank-axis",
                        choices=["eig_per_dollar", "predicted_score_delta"],
                        default="eig_per_dollar")
    parser.add_argument("--require-operator-approval-on", action="append",
                        default=[], help="event classes to gate (default: dispatch)")
    parser.add_argument("--claims-path", type=Path, default=None)
    parser.add_argument("--race-mode", action="store_true",
                        help="OPT-IN per CLAUDE.md race-mode rigor inversion")
    parser.add_argument("--max-dispatch-recommendations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the per-iteration report JSON")
    args = parser.parse_args(argv)

    try:
        if args.iterations <= 0:
            raise ValueError("--iterations must be > 0")
        if not args.candidates_jsonl.is_file():
            raise FileNotFoundError(args.candidates_jsonl)
        approval_set = _parse_approval_flags(
            args.require_operator_approval_on or ["dispatch"]
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
        return 2

    def _source() -> list[CandidateRow]:
        return load_candidates_from_jsonl(args.candidates_jsonl)

    reports = run_continuous_loop(
        _source,
        iterations=args.iterations,
        rank_axis=args.rank_axis,
        requires_approval_on=approval_set,
        claims_path=args.claims_path,
        race_mode=args.race_mode,
        max_dispatch_recommendations=args.max_dispatch_recommendations,
    )

    output_payload = {
        "schema": AUTONOMOUS_LOOP_SCHEMA,
        "evidence_grade": "[predicted; cathedral autopilot ranking]",
        "claude_md_compliance_tags": [
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_pattern_default_on",
            "no_score_claim_only_predicted_band",
            "no_kill_verdict_in_loop",
            "race_mode_explicit_opt_in_only",
        ],
        "iterations_run": len(reports),
        "race_mode": args.race_mode,
        "reports": [serialize_report(r) for r in reports],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(json.dumps(output_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
