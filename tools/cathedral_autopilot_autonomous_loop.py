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

**Operator-authorized le-$5/individual mode (2026-05-11):**

Per operator directive 2026-05-11 ("keep pushing the autopilot and xray and
magic codec and compiler and wiring and integration and everything") plus the
council's deferral overrule, the autopilot may now self-authorize dispatches
that satisfy ALL of:

  1. ``--operator-authorized-le-5-dollar-mode`` flag is explicitly set
     (default OFF; safe HALT-and-ASK preserved when the flag is absent),
  2. ``estimated_dispatch_cost_usd`` <= ``--per-dispatch-cap-usd`` (default
     $5.00 per the operator directive),
  3. cumulative-since-activation cost <= ``--cumulative-cap-usd`` (default
     $20.00 hard envelope),
  4. ``--canonical-helper-script`` resolves to ``tools/claim_lane_dispatch.py``
     OR explicit override via the same flag (the lane-claim ledger must be
     reachable for every authorized dispatch),
  5. ``CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE`` environment variable is
     set to ``1`` (defense-in-depth runtime gate — refuses to activate if the
     CLI flag is set but the env-var is not, even if other preconditions are
     met).

When all five preconditions hold, each candidate that fits the per-dispatch
cap is tagged ``[autopilot-claude-le-5-dollar]`` in its halt event, and a
structured log row is appended to the configured journal path. Candidates
crossing either cap remain HALT-and-ASK as before. The cumulative-envelope
counter is per-process (the loop's state); the canonical persistent ledger
is :mod:`tools.claim_lane_dispatch`, which the loop continues to delegate to.

No KILL verdict is ever auto-authorized.

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

# W/I/A I-1 wire-in (2026-05-12, decision I-1): the autonomous loop's
# rank_candidates step now optionally consumes the continual-learning posterior
# AND the cost-band-calibration posterior so empirical anchors reweight
# predicted score deltas (continual-learning) and refresh cost-band envelope
# decisions (cost-band). The wire-in mirrors `tools/cathedral_autopilot.py`'s
# `_posterior_correction_for_technique` pattern: a posterior-derived factor
# multiplies predicted_score_delta to produce an `adjusted_predicted_delta`
# that drives the ranking sort. The autopilot ranks are then surfaced as
# halt events for operator decision. Per CLAUDE.md "Subagent
# coherence-by-default" the wire-in is the structural fix; the per-candidate
# correction never auto-promotes nor auto-kills.
try:  # pragma: no cover - exercised by integration tests
    from tac.continual_learning import (
        load_posterior as load_continual_learning_posterior,
        posterior_query_track_correction,
    )
    from tac.cost_band_calibration import predict as predict_cost_band
    _POSTERIOR_IMPORTS_OK = True
except Exception:  # pragma: no cover - tests can stub these
    load_continual_learning_posterior = None  # type: ignore[assignment]
    posterior_query_track_correction = None  # type: ignore[assignment]
    predict_cost_band = None  # type: ignore[assignment]
    _POSTERIOR_IMPORTS_OK = False


AUTONOMOUS_LOOP_SCHEMA = "tac_cathedral_autopilot_autonomous_loop_v1"

# Operator-authorized le-$5/individual mode (2026-05-11).
OPERATOR_AUTHORIZED_MODE_ENV_VAR = "CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE"
OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED = "1"
DEFAULT_PER_DISPATCH_CAP_USD = 5.00
DEFAULT_CUMULATIVE_CAP_USD = 20.00
AUTOPILOT_AUTHORIZED_TAG = "[autopilot-claude-le-5-dollar]"
CANONICAL_HELPER_SCRIPT_RELPATH = "tools/claim_lane_dispatch.py"


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
class OperatorAuthorizedModeConfig:
    """Configuration for the operator-authorized le-$5/individual mode.

    Per CLAUDE.md "Design decisions — non-negotiable" the activation criteria
    were operator-set; this class only carries the configuration, never picks
    its own thresholds.

    Per CLAUDE.md "Operator gates must be wired and used" the activation flag
    is dual-gated (CLI ``--operator-authorized-le-5-dollar-mode`` AND env-var
    ``CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1``) so a stray CLI default
    cannot unlock dispatch authorization on its own.
    """

    enabled: bool = False
    per_dispatch_cap_usd: float = DEFAULT_PER_DISPATCH_CAP_USD
    cumulative_cap_usd: float = DEFAULT_CUMULATIVE_CAP_USD
    canonical_helper_script: Path | None = None
    journal_path: Path | None = None
    cumulative_spent_usd: float = 0.0

    def can_authorize(self, candidate: "CandidateRow") -> tuple[bool, str]:
        """Return ``(authorized, reason)``.

        Authorization requires every precondition to hold. The first failing
        precondition's reason is returned; on success the reason is empty.
        """
        if not self.enabled:
            return False, "operator-authorized mode is OFF (default safe HALT-and-ASK)"
        if candidate.estimated_dispatch_cost_usd > self.per_dispatch_cap_usd:
            return False, (
                f"candidate cost ${candidate.estimated_dispatch_cost_usd:.4f} "
                f"exceeds per-dispatch cap ${self.per_dispatch_cap_usd:.4f}"
            )
        prospective = self.cumulative_spent_usd + candidate.estimated_dispatch_cost_usd
        if prospective > self.cumulative_cap_usd:
            return False, (
                f"cumulative spend would reach ${prospective:.4f} which exceeds "
                f"the ${self.cumulative_cap_usd:.4f} envelope; operator round-trip required"
            )
        if candidate.blockers:
            return False, (
                f"candidate has unresolved blockers {candidate.blockers!r}; "
                "operator must adjudicate before any dispatch"
            )
        if candidate.estimated_dispatch_cost_usd <= 0.0:
            return False, (
                f"candidate cost {candidate.estimated_dispatch_cost_usd:.4f} is "
                "non-positive; refuse to authorize a malformed estimate"
            )
        if not self.canonical_helper_script or not self.canonical_helper_script.is_file():
            return False, (
                f"canonical helper script {self.canonical_helper_script!r} does not "
                "exist; operator must point --canonical-helper-script at a real file"
            )
        return True, ""

    def record_authorization(self, candidate: "CandidateRow") -> None:
        """Increment the per-process cumulative-spent counter.

        The persistent ledger is :mod:`tools.claim_lane_dispatch`. This in-memory
        counter is the loop's own envelope guard so a single autopilot session
        cannot drain authorization across many small dispatches.
        """
        self.cumulative_spent_usd += candidate.estimated_dispatch_cost_usd


def _env_authorizes_mode(env: dict[str, str] | None = None) -> bool:
    """Return True iff the env-var explicitly opts into authorized mode.

    Defense-in-depth: even if ``--operator-authorized-le-5-dollar-mode`` is
    passed, the runtime env-var must ALSO be set to ``1``. This guards against
    the failure mode where someone sets the CLI flag in a script but forgets
    that the operator's machine doesn't carry the env-var (so authorized
    dispatches never actually run).
    """
    import os as _os

    src = env if env is not None else _os.environ
    return src.get(OPERATOR_AUTHORIZED_MODE_ENV_VAR, "") == OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED


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
    # Operator-authorized le-$5/individual mode fields (2026-05-11).
    autopilot_authorized: bool = False
    autopilot_tag: str = ""
    autopilot_authorized_reason: str = ""
    autopilot_refused_reason: str = ""


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


def _posterior_correction_factor(
    candidate: CandidateRow,
    posterior: Any | None,
) -> tuple[float, int, str]:
    """Return ``(correction_factor, n_observations, key)`` for a candidate.

    W/I/A I-1 wire-in (2026-05-12): query the continual-learning posterior
    using the candidate's ``family`` field as the track-correction key. The
    factor multiplies the candidate's predicted_score_delta; ``n>0`` means
    empirical anchors exist for this family.

    Returns ``(1.0, 0, "")`` when the posterior is absent or no matching
    anchors are available. Per CLAUDE.md "Forbidden score claims": the
    correction is a non-authoritative planning prior; it never auto-promotes
    nor auto-kills a candidate.
    """
    if posterior is None or posterior_query_track_correction is None:
        return 1.0, 0, ""
    family = candidate.family
    if not family:
        return 1.0, 0, ""
    try:
        factor, n = posterior_query_track_correction(posterior, family)
    except Exception:  # pragma: no cover - defensive
        return 1.0, 0, family
    import math
    if n > 0 and math.isfinite(factor) and factor > 0.0:
        return float(factor), int(n), family
    return 1.0, 0, family


def rank_candidates(
    candidates: list[CandidateRow],
    *,
    rank_axis: str = "eig_per_dollar",
    continual_posterior: Any | None = None,
) -> list[CandidateRow]:
    """Rank candidates by the chosen axis (descending best-first).

    Recognized axes:
      - ``eig_per_dollar`` — expected information gain per dollar (default)
      - ``predicted_score_delta`` — most-negative-first (greatest improvement)

    When ``continual_posterior`` is provided (W/I/A I-1 wire-in, 2026-05-12),
    each candidate's predicted_score_delta is scaled by the continual-learning
    family-keyed correction factor BEFORE sorting. The original
    predicted_score_delta on the CandidateRow is preserved (the correction
    is applied transiently for sort-key purposes only) so halt events still
    report the raw prediction. Per CLAUDE.md "Subagent coherence-by-default"
    this exercises wire-in hook 5 (continual-learning posterior).

    Per CLAUDE.md "Forbidden /tmp paths": no temp paths used; pure in-memory.
    """
    if rank_axis == "eig_per_dollar":
        if continual_posterior is None:
            return sorted(candidates, key=lambda c: c.eig_per_dollar(), reverse=True)
        # Reweight EIG/$ by posterior correction. EIG itself is unchanged; the
        # cost-effective dispatch ordering still reflects empirical bias.
        def _eig_key(c: CandidateRow) -> float:
            factor, _, _ = _posterior_correction_factor(c, continual_posterior)
            return c.eig_per_dollar() * factor
        return sorted(candidates, key=_eig_key, reverse=True)
    if rank_axis == "predicted_score_delta":
        if continual_posterior is None:
            return sorted(candidates, key=lambda c: c.predicted_score_delta)
        # Reweight predicted_score_delta by posterior correction. Most-negative
        # first ordering preserved.
        def _delta_key(c: CandidateRow) -> float:
            factor, _, _ = _posterior_correction_factor(c, continual_posterior)
            return c.predicted_score_delta * factor
        return sorted(candidates, key=_delta_key)
    raise ValueError(
        f"unknown rank_axis {rank_axis!r}; must be 'eig_per_dollar' or "
        "'predicted_score_delta'"
    )


def discover_sensitivity_map_artifacts(
    search_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Enumerate available sensitivity-map artifacts under ``experiments/results``.

    W/I/A I-3 wire-in (2026-05-12): the autopilot's planner context now
    surfaces the inventory of saved sensitivity maps. This exercises
    CLAUDE.md unified-Lagrangian wire-in hook 1 (sensitivity-map
    contribution) — the autopilot is structurally aware of per-tensor
    importance evidence even when it does not consume the maps directly.

    Returns a JSON-safe dict::

        {
            "discovered": True,
            "artifact_paths": ["experiments/results/posenet_sensitivity_v5/sensitivity_map.pt", ...],
            "count": 3,
            "search_roots": ["experiments/results"]
        }

    No file is opened or parsed; the enumerator scans for ``sensitivity_map.pt``
    files only. Per CLAUDE.md "Forbidden /tmp paths" the search is rooted at
    the repo's ``experiments/results`` tree.
    """
    if search_dirs is None:
        roots = [REPO_ROOT / "experiments" / "results"]
    else:
        roots = list(search_dirs)
    artifact_paths: list[str] = []
    search_roots: list[str] = []
    for root in roots:
        search_roots.append(str(root.relative_to(REPO_ROOT)) if root.is_relative_to(REPO_ROOT) else str(root))
        if not root.is_dir():
            continue
        try:
            for p in root.rglob("sensitivity_map.pt"):
                rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
                artifact_paths.append(str(rel))
        except Exception:  # pragma: no cover - defensive
            continue
    return {
        "discovered": True,
        "artifact_paths": sorted(artifact_paths),
        "count": len(artifact_paths),
        "search_roots": search_roots,
    }


def load_planner_posterior_for_loop(
    continual_posterior_path: Path | None = None,
    *,
    include_sensitivity_map_inventory: bool = True,
) -> tuple[Any | None, dict[str, Any]]:
    """Load read-only continual-learning posterior context for the loop.

    Returns ``(posterior_or_none, context_payload)``. The payload is a small
    JSON-serializable dict reporting load status / anchor counts so iteration
    notes can surface ``loaded N=X anchors`` for operator visibility.

    W/I/A I-3 wire-in (2026-05-12): when
    ``include_sensitivity_map_inventory=True`` (default), the context payload
    ALSO includes a ``sensitivity_map_inventory`` key with the enumerated
    artifact paths (CLAUDE.md unified-Lagrangian hook 1 — sensitivity-map
    contribution).

    Per CLAUDE.md "Operator gates must be wired and used": failure to load
    falls back to ``(None, {"loaded": False, "reason": ...})`` — the loop
    keeps ranking without posterior context rather than crashing. The
    operator sees the load_error in iteration notes.
    """
    if not _POSTERIOR_IMPORTS_OK or load_continual_learning_posterior is None:
        ctx: dict[str, Any] = {"loaded": False, "reason": "tac.continual_learning import unavailable"}
        if include_sensitivity_map_inventory:
            ctx["sensitivity_map_inventory"] = discover_sensitivity_map_artifacts()
        return None, ctx
    try:
        posterior = load_continual_learning_posterior(continual_posterior_path)
    except Exception as exc:  # pragma: no cover - exercised by load_error test
        ctx = {"loaded": False, "reason": f"load_error:{type(exc).__name__}", "message": str(exc)}
        if include_sensitivity_map_inventory:
            ctx["sensitivity_map_inventory"] = discover_sensitivity_map_artifacts()
        return None, ctx
    payload: dict[str, Any] = {
        "loaded": True,
        "schema": getattr(posterior, "schema", "unknown"),
        "accepted_anchor_count": getattr(posterior, "accepted_anchor_count", 0),
        "refused_anchor_count": getattr(posterior, "refused_anchor_count", 0),
        "track_correction_count": len(getattr(posterior, "track_correction_posteriors", {})),
    }
    if include_sensitivity_map_inventory:
        payload["sensitivity_map_inventory"] = discover_sensitivity_map_artifacts()
    return posterior, payload


def cost_band_envelope_check(
    candidate: CandidateRow,
    *,
    platform: str | None = None,
    gpu: str | None = None,
    epochs: int | None = None,
    all_flags_on: bool = True,
    posterior_path: Path | None = None,
) -> tuple[float | None, str, dict[str, Any]]:
    """Query the cost-band posterior for an envelope-vs-estimate sanity check.

    W/I/A I-1 wire-in (2026-05-12, sister of continual-learning wire-in).
    Returns ``(p50_cost_usd_or_none, confidence_tag, payload)``. The payload
    captures p10/p50/p90 + anchor count so loop notes can surface a
    "candidate cost $X vs posterior p50 $Y (n=Z, tag=...)" comparison.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" the cost band
    is itself non-authoritative; it is a planning prior derived from prior
    dispatches' invoice-actuals. Returns ``(None, "unavailable", {})`` when
    inputs are missing or the predict() call raises.
    """
    if predict_cost_band is None or not platform or not gpu or epochs is None:
        return None, "unavailable", {
            "cost_band_available": False,
            "reason": (
                "predict_cost_band unavailable"
                if predict_cost_band is None
                else "platform/gpu/epochs not provided by candidate"
            ),
        }
    try:
        prediction = predict_cost_band(
            str(platform), str(gpu), int(epochs),
            all_flags_on=bool(all_flags_on), posterior_path=posterior_path,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"load_error:{type(exc).__name__}", {
            "cost_band_available": False,
            "reason": str(exc),
        }
    payload = {
        "cost_band_available": True,
        "cost_band_platform": prediction.platform,
        "cost_band_gpu": prediction.gpu,
        "cost_band_epochs": prediction.epochs,
        "cost_band_n_anchors": prediction.n_anchors,
        "cost_band_confidence_tag": prediction.confidence_tag,
        "cost_band_p10_cost_usd": prediction.p10_cost_usd,
        "cost_band_p50_cost_usd": prediction.p50_cost_usd,
        "cost_band_p90_cost_usd": prediction.p90_cost_usd,
        "candidate_cost_vs_p50_ratio": (
            candidate.estimated_dispatch_cost_usd / prediction.p50_cost_usd
            if prediction.p50_cost_usd > 0 else None
        ),
    }
    return prediction.p50_cost_usd, prediction.confidence_tag, payload


def make_dispatch_halt_event(
    candidate: CandidateRow,
    *,
    requires_approval_classes: frozenset[EventClass],
    blockers: list[str] | None = None,
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
) -> HaltEvent:
    """Construct a HALT event for one dispatch decision.

    Per CLAUDE.md "operator-gate non-negotiable": when ``EventClass.DISPATCH``
    is in ``requires_approval_classes``, ``requires_approval=True`` UNLESS the
    operator-authorized le-$5/individual mode is engaged AND every precondition
    holds for THIS candidate. In that case the event is tagged
    ``[autopilot-claude-le-5-dollar]`` and ``requires_approval`` is set to
    False so the loop does not wait for the operator on this row.

    The dual-gate check (CLI flag + env-var) lives entirely inside this
    function; callers cannot bypass it. When ``env_authorized`` is None the
    real env-var is consulted; tests inject ``env_authorized=True/False``
    directly.
    """
    requires = EventClass.DISPATCH in requires_approval_classes
    halt_blockers = list(blockers or [])
    autopilot_authorized = False
    autopilot_tag = ""
    autopilot_reason = ""
    autopilot_refused = ""

    if auth_mode is not None and auth_mode.enabled:
        env_ok = _env_authorizes_mode() if env_authorized is None else bool(env_authorized)
        if not env_ok:
            autopilot_refused = (
                f"env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
                f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED} is missing; CLI "
                "flag alone is insufficient (defense-in-depth)"
            )
        else:
            ok, reason = auth_mode.can_authorize(candidate)
            if ok:
                autopilot_authorized = True
                autopilot_tag = AUTOPILOT_AUTHORIZED_TAG
                autopilot_reason = (
                    f"per-dispatch cost ${candidate.estimated_dispatch_cost_usd:.4f} "
                    f"<= cap ${auth_mode.per_dispatch_cap_usd:.4f}; "
                    f"cumulative ${auth_mode.cumulative_spent_usd + candidate.estimated_dispatch_cost_usd:.4f} "
                    f"<= envelope ${auth_mode.cumulative_cap_usd:.4f}"
                )
                # Reserve cost in the per-process counter so the next candidate
                # in this iteration sees the updated cumulative.
                auth_mode.record_authorization(candidate)
                requires = False
            else:
                autopilot_refused = reason

    return HaltEvent(
        event_class=EventClass.DISPATCH,
        candidate_id=candidate.candidate_id,
        reason=(
            f"Dispatch decision for candidate {candidate.candidate_id} "
            f"(family={candidate.family}, predicted_score_delta="
            f"{candidate.predicted_score_delta:+.6f}) — "
            + (
                "autopilot self-authorized (le-$5/individual operator-set mode)"
                if autopilot_authorized
                else "operator decision required."
            )
        ),
        predicted_score_delta=candidate.predicted_score_delta,
        estimated_cost_usd=candidate.estimated_dispatch_cost_usd,
        requires_approval=requires,
        halt_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        blockers=halt_blockers,
        autopilot_authorized=autopilot_authorized,
        autopilot_tag=autopilot_tag,
        autopilot_authorized_reason=autopilot_reason,
        autopilot_refused_reason=autopilot_refused,
    )


def append_autopilot_journal_row(
    journal_path: Path,
    event: HaltEvent,
    *,
    iteration: int,
) -> None:
    """Append one structured JSONL row recording an autopilot-authorized dispatch.

    Per CLAUDE.md "Forbidden /tmp paths": callers must point ``journal_path`` at
    a durable location (``reports/`` or ``.omx/state/``); this helper does not
    pick a default location.

    Per CLAUDE.md "Subagent commits MUST use serializer": this writes a JSONL
    row that the operator can later commit via the canonical serializer; the
    helper itself never invokes git.
    """
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "tac_cathedral_autopilot_authorized_journal_v1",
        "iteration": iteration,
        "candidate_id": event.candidate_id,
        "predicted_score_delta": event.predicted_score_delta,
        "estimated_cost_usd": event.estimated_cost_usd,
        "halt_at_utc": event.halt_at_utc,
        "autopilot_authorized": event.autopilot_authorized,
        "autopilot_tag": event.autopilot_tag,
        "autopilot_authorized_reason": event.autopilot_authorized_reason,
        "autopilot_refused_reason": event.autopilot_refused_reason,
        "blockers": list(event.blockers),
        "claude_md_compliance_tags": [
            "operator_authorized_le_5_dollar_mode",
            "halt_and_ask_preserved_above_cap",
            "no_kill_verdict",
            "dispatch_claim_check_done",
        ],
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


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
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
    continual_posterior: Any | None = None,
    continual_posterior_path: Path | None = None,
    auto_load_continual_posterior: bool = False,
) -> LoopIterationReport:
    """Run one cycle: rank → dispatch-claim check → halt-event emission.

    Per CLAUDE.md "operator-gate non-negotiable": this never actually
    dispatches. It surfaces HALT events with ``requires_approval=True`` for
    operator decision.

    Per CLAUDE.md "race-mode rigor inversion": when ``race_mode=True`` the
    loop trims the candidate set to the ones with smallest predicted cost
    AND non-trivial predicted_score_delta (the "smallest credible bolt-on"
    pattern from the May 4 race postmortem).

    W/I/A I-1 wire-in (2026-05-12): when ``continual_posterior`` is provided
    (or ``auto_load_continual_posterior=True``) the rank step applies the
    family-keyed correction factor from :mod:`tac.continual_learning`. The
    raw predicted_score_delta on each CandidateRow is unchanged; the
    correction biases ranking order only. Iteration notes record the loaded
    posterior anchor counts for operator visibility.
    """
    started = dt.datetime.now(dt.UTC).isoformat()
    notes: list[str] = []

    # W/I/A I-1: optionally auto-load continual-learning posterior so the
    # loop's rank step applies empirical-anchor reweighting. Tests inject
    # ``continual_posterior=`` directly to skip the file load.
    if continual_posterior is None and auto_load_continual_posterior:
        continual_posterior, posterior_context = load_planner_posterior_for_loop(
            continual_posterior_path=continual_posterior_path,
        )
        if posterior_context.get("loaded"):
            notes.append(
                f"continual-learning posterior loaded "
                f"(accepted_anchors={posterior_context.get('accepted_anchor_count', 0)}, "
                f"track_corrections={posterior_context.get('track_correction_count', 0)})"
            )
        else:
            notes.append(
                f"continual-learning posterior unavailable: "
                f"{posterior_context.get('reason', 'unknown')}"
            )

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
    ranked = (
        rank_candidates(
            candidates,
            rank_axis=rank_axis,
            continual_posterior=continual_posterior,
        )
        if candidates else []
    )

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
            auth_mode=auth_mode,
            env_authorized=env_authorized,
        )
        halt_events.append(event)
        n_ranked += 1
        # Journal the authorization if the autopilot self-authorized.
        if (
            event.autopilot_authorized
            and auth_mode is not None
            and auth_mode.journal_path is not None
        ):
            append_autopilot_journal_row(
                auth_mode.journal_path, event, iteration=iteration
            )
            notes.append(
                f"autopilot self-authorized candidate {cand.candidate_id!r} "
                f"(cumulative ${auth_mode.cumulative_spent_usd:.4f} / cap "
                f"${auth_mode.cumulative_cap_usd:.4f})"
            )

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
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
    continual_posterior: Any | None = None,
    continual_posterior_path: Path | None = None,
    auto_load_continual_posterior: bool = False,
) -> list[LoopIterationReport]:
    """Run the continuous loop for ``iterations`` cycles.

    Each iteration calls ``candidate_source()`` to refresh the queue. The
    operator decision callback is invoked on every HALT event that has
    ``requires_approval=True``; if no callback is supplied, decisions are
    DEFER by default (the safe choice).

    W/I/A I-1 wire-in (2026-05-12): when ``auto_load_continual_posterior``
    is True, the continual-learning posterior is loaded ONCE at the start
    of the loop and passed to every iteration. This is the canonical
    "newly-appended anchor changes next ranking pass" path — the loop's
    candidate_source produces fresh rows each iteration; the posterior is
    re-read implicitly if the candidate_source itself appends to the
    posterior between calls (the load is fast / cached in memory by
    ``tac.continual_learning``).

    Returns the list of per-iteration reports.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0; got {iterations}")
    reports: list[LoopIterationReport] = []
    for i in range(1, iterations + 1):
        candidates = candidate_source()
        # Per-iteration posterior reload: each iteration sees the most recent
        # anchor state. The explicit ``continual_posterior`` arg lets callers
        # inject a stable posterior for deterministic testing.
        iter_posterior = continual_posterior
        iter_auto_load = auto_load_continual_posterior and continual_posterior is None
        report = run_one_loop_iteration(
            candidates,
            iteration=i,
            rank_axis=rank_axis,
            requires_approval_on=requires_approval_on,
            claims_path=claims_path,
            race_mode=race_mode,
            max_dispatch_recommendations=max_dispatch_recommendations,
            auth_mode=auth_mode,
            env_authorized=env_authorized,
            continual_posterior=iter_posterior,
            continual_posterior_path=continual_posterior_path,
            auto_load_continual_posterior=iter_auto_load,
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


# ── Substrate composition matrix ranking integration ──────────────────────


SUBSTRATE_COMPOSITION_RANKING_SCHEMA = "tac_autopilot_dispatch_ranking_v1"


def load_candidates_from_substrate_composition_ranking(
    path: Path,
    *,
    only_in_envelope: bool = True,
    only_fits_per_dispatch_cap: bool = True,
) -> list[CandidateRow]:
    """Load candidates from QQ's substrate composition ranking JSON.

    Per CLAUDE.md "race-mode + parallel-dispatch first" + the substrate
    composition matrix landing memo (`feedback_substrate_composition_
    matrix_autopilot_ranking_theoretical_floor_v2_landed_20260511.md`),
    QQ's ranker emits an artifact with schema
    ``tac_autopilot_dispatch_ranking_v1`` containing ``ranked_dispatches``
    each carrying ``candidate_id``, ``family``, ``predicted_score_delta``,
    ``expected_information_gain``, ``estimated_dispatch_cost_usd``,
    ``substrate_ids`` (the substrates participating in this dispatch),
    ``composition_notes`` (the rationale), ``fits_per_dispatch_cap``,
    and ``fits_cumulative_envelope`` flags.

    This loader converts each ``ranked_dispatch`` into a ``CandidateRow``
    that the autopilot loop consumes via the existing HALT-and-ASK gate.
    Per CLAUDE.md "Forbidden score claims" the loaded rows carry
    ``predicted_score_delta`` tagged ``[predicted; substrate composition
    matrix v1]`` in their ``notes`` field.

    Filtering rules (defaults match QQ's envelope discipline):

    - ``only_in_envelope=True`` drops dispatches that QQ marked as
      out-of-cumulative-envelope (``fits_cumulative_envelope=False``).
    - ``only_fits_per_dispatch_cap=True`` drops dispatches that QQ marked
      as out-of-per-dispatch-cap (``fits_per_dispatch_cap=False``).

    The loader REFUSES to load rows whose ``score_claim`` field is True or
    whose ``ready_for_exact_eval_dispatch`` field is True — those would
    violate the planning-only invariant.
    """
    if not path.is_file():
        raise FileNotFoundError(f"substrate composition ranking JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema") or payload.get("matrix_schema")
    if "ranked_dispatches" not in payload:
        raise ValueError(
            f"substrate composition ranking JSON missing 'ranked_dispatches' "
            f"key (schema={schema!r}); got top-level keys: "
            f"{sorted(payload.keys())}"
        )
    if payload.get("score_claim", False):
        raise ValueError(
            "substrate composition ranking JSON has score_claim=True; "
            "the autopilot ranker must remain planning-only "
            "(per CLAUDE.md 'Forbidden score claims')"
        )
    rows: list[CandidateRow] = []
    for raw in payload["ranked_dispatches"]:
        if raw.get("score_claim", False):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has score_claim=True; "
                "refuse to consume score-claimed planning rows"
            )
        if raw.get("ready_for_exact_eval_dispatch", False):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has "
                "ready_for_exact_eval_dispatch=True; refuse to consume in autopilot "
                "(operator-gated promotion path required)"
            )
        if only_fits_per_dispatch_cap and not raw.get("fits_per_dispatch_cap", True):
            continue
        if only_in_envelope and not raw.get("fits_cumulative_envelope", True):
            continue
        notes_lines = [
            "[predicted; substrate composition matrix v1]",
            f"composition_notes: {raw.get('composition_notes', '')}",
            f"substrate_ids: {raw.get('substrate_ids', [])!r}",
        ]
        rows.append(CandidateRow(
            candidate_id=str(raw["candidate_id"]),
            family=str(raw["family"]),
            predicted_score_delta=float(raw["predicted_score_delta"]),
            expected_information_gain=float(raw["expected_information_gain"]),
            estimated_dispatch_cost_usd=float(raw["estimated_dispatch_cost_usd"]),
            blockers=list(raw.get("blockers", [])),
            notes="\n".join(notes_lines),
        ))
    return rows


def candidate_substrate_ids_from_ranking(path: Path) -> dict[str, tuple[str, ...]]:
    """Map ``candidate_id`` -> tuple of substrate ids participating in the dispatch.

    Used by the composition-constraint enforcer to reason about which
    substrates a candidate touches without re-parsing the full ranking JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "ranked_dispatches" not in payload:
        raise ValueError(
            "substrate composition ranking JSON missing 'ranked_dispatches'"
        )
    out: dict[str, tuple[str, ...]] = {}
    for raw in payload["ranked_dispatches"]:
        cid = str(raw["candidate_id"])
        substrates = tuple(str(s) for s in raw.get("substrate_ids", ()))
        out[cid] = substrates
    return out


def filter_composition_incompatible_dispatches(
    candidates: list[CandidateRow],
    *,
    candidate_substrate_ids: dict[str, tuple[str, ...]],
) -> tuple[list[CandidateRow], list[tuple[str, str]]]:
    """Walk candidates in order; refuse any whose substrates conflict with
    a substrate already chosen by an earlier candidate in the SAME loop
    iteration (per QQ matrix's REPLACEMENT/INCOMPATIBLE classes).

    Returns ``(kept, dropped_with_reasons)``. ``dropped_with_reasons`` is a
    list of ``(candidate_id, reason)`` pairs.

    Per QQ matrix lesson 5 (HNeRV parity discipline) and the
    `substrate vs codec composition meta-pattern` memo: two
    RENDERER_REPLACEMENT substrates cannot coexist in the same archive,
    and two REPLACEMENT-classed cells anywhere produce an
    archive-grammar conflict at byte-level. Composition matrix is
    consulted ONLY through the participating-substrate sets; the loader
    does not need to import the full matrix here (kept lightweight).

    The composition constraint enforced here is a SAME-DISPATCH-CHAIN
    constraint: it does NOT prevent the operator from running two separate
    autopilot iterations each picking ONE renderer-replacement candidate.
    The matrix governs which substrates can be in the same archive bytes,
    not which substrates can be considered across runs.
    """
    try:
        from tac.optimization.substrate_composition_matrix import (
            Composability,
            build_composition_matrix,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "filter_composition_incompatible_dispatches requires "
            "tac.optimization.substrate_composition_matrix; install or "
            "import path setup before calling."
        ) from exc

    matrix = build_composition_matrix()
    kept: list[CandidateRow] = []
    chosen_substrates: set[str] = set()
    dropped: list[tuple[str, str]] = []
    incompatible_classes = {
        Composability.REPLACEMENT,
        Composability.INCOMPATIBLE,
        Composability.ANTAGONISTIC,
    }
    for cand in candidates:
        substrates = candidate_substrate_ids.get(cand.candidate_id, ())
        conflict_reason: str | None = None
        for s in substrates:
            for prior in chosen_substrates:
                if s == prior:
                    continue
                try:
                    cell = matrix.get(s, prior)
                except (KeyError, ValueError):
                    continue
                if cell.composability in incompatible_classes:
                    conflict_reason = (
                        f"substrate {s!r} composes as "
                        f"{cell.composability.value} with already-chosen "
                        f"substrate {prior!r}; refuse same-iteration "
                        "dispatch (per substrate composition matrix v1)"
                    )
                    break
            if conflict_reason is not None:
                break
        if conflict_reason is not None:
            dropped.append((cand.candidate_id, conflict_reason))
            continue
        kept.append(cand)
        for s in substrates:
            chosen_substrates.add(s)
    return kept, dropped


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates-jsonl", type=Path, default=None,
                        help="Path to JSONL file of CandidateRow rows. Mutually "
                             "exclusive with --use-substrate-composition-matrix-ranking.")
    parser.add_argument(
        "--use-substrate-composition-matrix-ranking",
        type=Path,
        default=None,
        help=(
            "Path to QQ's substrate composition matrix ranking JSON "
            "(schema=tac_autopilot_dispatch_ranking_v1, e.g. "
            "experiments/results/cathedral_autopilot_dispatch_ranking_*/ranking.json). "
            "When set, candidates are loaded from the ranking and "
            "filter_composition_incompatible_dispatches enforces the matrix's "
            "REPLACEMENT/INCOMPATIBLE/ANTAGONISTIC constraints across the dispatch "
            "queue. Mutually exclusive with --candidates-jsonl."
        ),
    )
    parser.add_argument(
        "--include-out-of-envelope-ranking-candidates",
        action="store_true",
        help=(
            "When --use-substrate-composition-matrix-ranking is set, also load "
            "ranking rows that QQ marked as out-of-envelope. Default OFF — "
            "the autopilot honors QQ's envelope discipline."
        ),
    )
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
    parser.add_argument(
        "--operator-authorized-le-5-dollar-mode",
        action="store_true",
        help=(
            "OPT-IN: enable operator-authorized le-$5/individual mode (default "
            "OFF). Per CLAUDE.md operator-gate non-negotiable + dual-gated by "
            f"env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
            f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED}."
        ),
    )
    parser.add_argument(
        "--per-dispatch-cap-usd",
        type=float,
        default=DEFAULT_PER_DISPATCH_CAP_USD,
        help=(
            f"Per-dispatch hard cap when authorized mode is on (default "
            f"${DEFAULT_PER_DISPATCH_CAP_USD:.2f})."
        ),
    )
    parser.add_argument(
        "--cumulative-cap-usd",
        type=float,
        default=DEFAULT_CUMULATIVE_CAP_USD,
        help=(
            f"Cumulative spend envelope when authorized mode is on (default "
            f"${DEFAULT_CUMULATIVE_CAP_USD:.2f})."
        ),
    )
    parser.add_argument(
        "--canonical-helper-script",
        type=Path,
        default=None,
        help=(
            "Path to the canonical dispatch-claim helper "
            f"(default {CANONICAL_HELPER_SCRIPT_RELPATH} under the repo root)."
        ),
    )
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=None,
        help=(
            "Where to append authorized-dispatch rows (JSONL). Required when "
            "--operator-authorized-le-5-dollar-mode is set."
        ),
    )
    # W/I/A I-1 wire-in (2026-05-12): continual-learning posterior knobs.
    parser.add_argument(
        "--continual-posterior-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the continual-learning posterior JSONL. Default "
            "uses tac.continual_learning.DEFAULT_POSTERIOR_PATH. Loaded when "
            "--load-continual-posterior is set."
        ),
    )
    parser.add_argument(
        "--load-continual-posterior",
        action="store_true",
        help=(
            "OPT-IN: load tac.continual_learning posterior and reweight "
            "predicted_score_delta by family-keyed correction factor before "
            "ranking. Per CLAUDE.md 'Subagent coherence-by-default' the "
            "wire-in is the structural fix; the per-candidate correction "
            "never auto-promotes nor auto-kills."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.iterations <= 0:
            raise ValueError("--iterations must be > 0")
        # Exactly one of --candidates-jsonl OR --use-substrate-composition-matrix-ranking
        # must be supplied. They are mutually exclusive.
        sources_supplied = sum(
            1 for x in (
                args.candidates_jsonl,
                args.use_substrate_composition_matrix_ranking,
            ) if x is not None
        )
        if sources_supplied != 1:
            raise ValueError(
                "exactly one of --candidates-jsonl or "
                "--use-substrate-composition-matrix-ranking must be supplied "
                f"(got {sources_supplied})"
            )
        if args.candidates_jsonl is not None and not args.candidates_jsonl.is_file():
            raise FileNotFoundError(args.candidates_jsonl)
        if (
            args.use_substrate_composition_matrix_ranking is not None
            and not args.use_substrate_composition_matrix_ranking.is_file()
        ):
            raise FileNotFoundError(args.use_substrate_composition_matrix_ranking)
        approval_set = _parse_approval_flags(
            args.require_operator_approval_on or ["dispatch"]
        )
        # Refuse to activate authorized mode without a journal path.
        if args.operator_authorized_le_5_dollar_mode and args.journal_path is None:
            raise ValueError(
                "--operator-authorized-le-5-dollar-mode requires --journal-path "
                "for the structured-row JSONL ledger (per CLAUDE.md no-/tmp-path)"
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
        return 2

    auth_mode: OperatorAuthorizedModeConfig | None = None
    if args.operator_authorized_le_5_dollar_mode:
        helper_path = args.canonical_helper_script or (
            REPO_ROOT / CANONICAL_HELPER_SCRIPT_RELPATH
        )
        auth_mode = OperatorAuthorizedModeConfig(
            enabled=True,
            per_dispatch_cap_usd=args.per_dispatch_cap_usd,
            cumulative_cap_usd=args.cumulative_cap_usd,
            canonical_helper_script=helper_path,
            journal_path=args.journal_path,
        )
        # Defense-in-depth env-var check.
        if not _env_authorizes_mode():
            print(
                "cathedral_autopilot_autonomous_loop: authorized mode CLI flag "
                f"is set but env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
                f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED} is missing; "
                "loop will run but no candidate will self-authorize",
                file=sys.stderr,
            )

    composition_substrate_map: dict[str, tuple[str, ...]] = {}
    composition_dropped: list[tuple[str, str]] = []

    if args.use_substrate_composition_matrix_ranking is not None:
        composition_substrate_map = candidate_substrate_ids_from_ranking(
            args.use_substrate_composition_matrix_ranking
        )

        def _source() -> list[CandidateRow]:
            base = load_candidates_from_substrate_composition_ranking(
                args.use_substrate_composition_matrix_ranking,
                only_in_envelope=not args.include_out_of_envelope_ranking_candidates,
                only_fits_per_dispatch_cap=not args.include_out_of_envelope_ranking_candidates,
            )
            kept, dropped = filter_composition_incompatible_dispatches(
                base, candidate_substrate_ids=composition_substrate_map,
            )
            composition_dropped.clear()
            composition_dropped.extend(dropped)
            return kept
    else:
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
        auth_mode=auth_mode,
        continual_posterior_path=args.continual_posterior_path,
        auto_load_continual_posterior=args.load_continual_posterior,
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
            "operator_authorized_le_5_dollar_mode_dual_gated",
            "substrate_composition_matrix_constraints_enforced"
                if args.use_substrate_composition_matrix_ranking is not None
                else "candidates_jsonl_source",
        ],
        "iterations_run": len(reports),
        "race_mode": args.race_mode,
        "operator_authorized_mode": {
            "enabled": bool(auth_mode and auth_mode.enabled),
            "per_dispatch_cap_usd": auth_mode.per_dispatch_cap_usd if auth_mode else None,
            "cumulative_cap_usd": auth_mode.cumulative_cap_usd if auth_mode else None,
            "cumulative_spent_usd": auth_mode.cumulative_spent_usd if auth_mode else 0.0,
            "env_authorized": _env_authorizes_mode(),
            "journal_path": str(auth_mode.journal_path) if auth_mode and auth_mode.journal_path else None,
        },
        "substrate_composition_ranking": (
            {
                "ranking_path": str(args.use_substrate_composition_matrix_ranking),
                "include_out_of_envelope": bool(
                    args.include_out_of_envelope_ranking_candidates
                ),
                "n_dropped_by_composition_constraint": len(composition_dropped),
                "dropped_with_reasons": [
                    {"candidate_id": cid, "reason": reason}
                    for (cid, reason) in composition_dropped
                ],
            }
            if args.use_substrate_composition_matrix_ranking is not None
            else None
        ),
        "reports": [serialize_report(r) for r in reports],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # allow_nan=False refuses Infinity/NaN emission per RFC 8259. The
        # eig_per_dollar source fix already enforces this; adding it here
        # as defense-in-depth so any future numeric field that goes
        # non-finite fails loud at the serializer boundary, not silently
        # in the consumer.
        args.output.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    print(json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
