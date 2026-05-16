#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
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
cap must first record a lane claim through :mod:`tools.claim_lane_dispatch`.
Only after the claim helper succeeds is the candidate tagged
``[autopilot-claude-le-5-dollar]`` with ``requires_approval=False`` and
written to the configured journal path. Candidates crossing either cap, or
whose claim helper invocation fails, remain HALT-and-ASK as before. The
cumulative-envelope counter is per-process (the loop's state); the canonical
persistent ledger is :mod:`tools.claim_lane_dispatch`.

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
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

try:  # Reuse the canonical parser/conflict semantics from the claim helper.
    from tools import claim_lane_dispatch as _claim_lane_dispatch
except ModuleNotFoundError:  # pragma: no cover - direct tools/ import mode
    import claim_lane_dispatch as _claim_lane_dispatch  # type: ignore[no-redef]

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
    )
    from tac.continual_learning import (
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
AUTOPILOT_CONTEST_TARGET_MODE = "contest_exact_eval"
PLANNING_ONLY_SOURCE_BLOCKER = "planning_only_source_requires_operator_dispatch_packet"
AUTOPILOT_CLAIM_PLATFORM = "cathedral_autopilot"
AUTOPILOT_CLAIM_STATUS = "active_autopilot_authorized_dispatch"
AUTOPILOT_CLAIM_AGENT = "cathedral_autopilot_autonomous_loop"
AUTOPILOT_CLAIM_TTL_HOURS = 24.0
EXACT_READY_QUEUE_SCHEMA = "optimizer_candidate_exact_eval_ready_queue_v1"
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_sha256_hex(value: object) -> bool:
    """Return True only for concrete 64-hex SHA-256 strings."""
    return bool(_SHA256_HEX_RE.fullmatch(str(value or "").strip()))


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_authorized_journal_path(journal_path: Path, *, repo_root: Path = REPO_ROOT) -> None:
    """Refuse transient paths for self-authorized dispatch journals."""
    forbidden_roots = (
        Path("/tmp"),
        Path("/private/tmp"),
        Path("/var/tmp"),
        Path(tempfile.gettempdir()),
    )
    if any(_path_is_relative_to(journal_path, root) for root in forbidden_roots):
        raise ValueError(
            "--journal-path for authorized mode must be durable and repo-local "
            "(.omx/state/ or reports/); refusing transient path "
            f"{str(journal_path)!r}"
        )
    allowed_roots = (repo_root / ".omx" / "state", repo_root / "reports")
    if any(_path_is_relative_to(journal_path, root) for root in allowed_roots):
        return
    raise ValueError(
        "--journal-path for authorized mode must be under repo-local .omx/state/ "
        f"or reports/; got {str(journal_path)!r}"
    )


def _require_candidate_dispatch_cost(candidate: CandidateRow) -> float:
    return _require_finite_positive_float(
        candidate.estimated_dispatch_cost_usd,
        field="estimated_dispatch_cost_usd",
        context=f"candidate {candidate.candidate_id!r}",
    )


def _require_candidate_planning_cost(candidate: CandidateRow) -> float:
    return _require_finite_nonnegative_float(
        candidate.estimated_dispatch_cost_usd,
        field="estimated_dispatch_cost_usd",
        context=f"candidate {candidate.candidate_id!r}",
    )


def validate_authorized_mode_config(
    auth_mode: OperatorAuthorizedModeConfig | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Validate authorized-mode config before ranking or dispatch side effects."""
    if auth_mode is None or not auth_mode.enabled:
        return
    _require_finite_positive_float(
        auth_mode.per_dispatch_cap_usd,
        field="per_dispatch_cap_usd",
        context="operator-authorized mode",
    )
    _require_finite_positive_float(
        auth_mode.cumulative_cap_usd,
        field="cumulative_cap_usd",
        context="operator-authorized mode",
    )
    try:
        spent = float(auth_mode.cumulative_spent_usd)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "operator-authorized mode has non-numeric cumulative_spent_usd="
            f"{auth_mode.cumulative_spent_usd!r}"
        ) from exc
    if not math.isfinite(spent) or spent < 0.0:
        raise ValueError(
            "operator-authorized mode must carry finite non-negative "
            f"cumulative_spent_usd; got {auth_mode.cumulative_spent_usd!r}"
        )
    if auth_mode.journal_path is None:
        raise ValueError(
            "operator-authorized mode requires a durable journal_path before "
            "ranking or dispatch authorization"
        )
    validate_authorized_journal_path(auth_mode.journal_path, repo_root=repo_root)


def _authorized_mode_config_blocker(
    auth_mode: OperatorAuthorizedModeConfig | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    try:
        validate_authorized_mode_config(auth_mode, repo_root=repo_root)
    except ValueError as exc:
        return str(exc)
    return ""


# ── Events / decisions / verdicts ──────────────────────────────────────────


class EventClass(StrEnum):
    """Operator-decision event classes."""

    DISPATCH = "dispatch"
    KILL = "kill"  # NEVER auto-kill per CLAUDE.md; operator-only
    PROMOTE = "promote"
    POSTERIOR_REWEIGHT = "posterior_reweight"
    RACE_MODE_TOGGLE = "race_mode_toggle"


class OperatorDecision(StrEnum):
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

    Z1 empirical revision fields (2026-05-14, decision #4 per
    ``feedback_z1_mdl_ablation_landed_20260514.md``):

      - ``mdl_density`` — measured scorer-conditional MDL density (0-1) per
        the Z1 ablation tool. None when the candidate has no ablation
        evidence yet (don't penalize lack-of-evidence). Density >0.95
        indicates within-class trap; 0.90-0.95 trending; <0.90 across-class
        promising.
      - ``lane_class`` — declared substrate class ("substrate_class_shift",
        "substrate_engineering", "research_substrate", etc.) used to apply
        the class-shift reward. None = unknown / not-declared.
      - ``literature_anchor`` — citation / family name surfacing the
        substrate-class lineage (e.g. "cooperative-receiver",
        "Tishby-Zaslavsky", "Wyner-Ziv"). Used by
        :func:`adjust_score_for_class_shift` to reward known class-shift
        primitives.
      - ``source_supports`` / ``paper_claim_scope`` / ``pact_must_prove`` /
        ``decode_complexity_evidence`` — source-fidelity scope fields carried
        from the composition matrix so literature anchors cannot silently become
        empirical Pact score claims.
    """

    candidate_id: str
    family: str  # e.g. 'hnerv_lc_v2', 'balle_scale_hyperprior'
    predicted_score_delta: float  # negative = improvement
    expected_information_gain: float
    estimated_dispatch_cost_usd: float
    blockers: list[str] = field(default_factory=list)
    notes: str = ""
    timing_smoke_command: str = ""
    # Z1 empirical revision wire-in (2026-05-14):
    mdl_density: float | None = None
    lane_class: str | None = None
    literature_anchor: str = ""
    source_supports: str = ""
    paper_claim_scope: str = ""
    pact_must_prove: str = ""
    decode_complexity_evidence: str = ""
    # Catalog #227 Tier C empirical revision wire-in (2026-05-14):
    # Tier C-derived substrate-class density estimate (0..1). HIGH (>= 0.70) =
    # within-class; LOW (<= 0.30) = across-class. The signal OVERRIDES Tier A
    # for substrate-class discrimination because Tier A is brotli-saturated at
    # the byte layer (any fp16-weight + brotli archive sits at Tier A density
    # ~0.99) and structurally CANNOT discriminate. Per the C6 5ep empirical
    # anchor `feedback_mdl_ablation_tier_c_ibps1_landed_20260514.md`. None =
    # no Tier C evidence (don't penalize lack-of-evidence).
    mdl_tier_c_density: float | None = None
    # Catalog #227 composition matrix wire-in (2026-05-14):
    # Substrate composition additivity factor alpha (0..1+). HIGH (> 0.7) =
    # ADDITIVE stacking (compound predicted_score_delta savings); MEDIUM
    # (0.3-0.7) = SUB-ADDITIVE (halve savings); LOW (<= 0.3) = SATURATING
    # (single-substrate dominates; floor predicted_score_delta near zero).
    # Sourced from `.omx/state/substrate_composition_matrix.json` per T1-F
    # `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`.
    # None = no composition evidence (single-substrate candidate or untested
    # pair); the candidate is ranked without composition adjustment.
    composition_alpha: float | None = None
    license_ok: bool = True
    inflate_dep_count: int = 0
    sideinfo_consumed: bool | None = None
    exact_duplicate: bool = False
    context_order: int = 0
    # OP-3 predicted_dispatch_risk wire-in (2026-05-15, codex chunk 6 finding):
    # SLIM (Sparse Linear Integer Model) preflight risk score per
    # `tac.preflight_rudin_daubechies.slim_risk_scorer.PreflightSLIMRiskScorer`.
    # Range nominally 0..200 with the canonical refusal threshold at
    # ``DISPATCH_RISK_REFUSAL_THRESHOLD = 50`` (see slim_risk_scorer.py:111).
    # Bands consumed by :func:`adjust_predicted_delta_for_predicted_dispatch_risk`:
    #   risk >= 50 → REFUSE (floor predicted_score_delta at 0)
    #   risk 25-50 → MODERATE (halve predicted savings)
    #   risk < 25  → LOW (no adjustment)
    # None = no preflight evidence (don't penalize lack-of-evidence per the
    # sister Z1 / Tier C / composition_alpha conventions).
    predicted_dispatch_risk: float | None = None
    # Dispatch authority / custody fields. These are intentionally absent from
    # read-only planning sources; operator-authorized self-dispatch requires a
    # real dispatch packet instead of inferred readiness from rank/cost.
    lane_id: str = ""
    claim_keys: list[str] = field(default_factory=list)
    target_modes: list[str] = field(default_factory=list)
    dispatch_packet_ready: bool = False
    dispatch_packet_sha256: str = ""
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def eig_per_dollar(self) -> float:
        cost = _require_candidate_planning_cost(self)
        if cost == 0.0:
            return 0.0
        return self.expected_information_gain / cost

    def dispatch_claim_keys(self) -> list[str]:
        """Return candidate/lane/claim identifiers for conflict checks."""
        keys = [self.candidate_id, self.lane_id, *self.claim_keys]
        deduped: list[str] = []
        for key in keys:
            s = str(key or "").strip()
            if s and s not in deduped:
                deduped.append(s)
        return deduped

    def dispatch_authority_blockers(self) -> list[str]:
        """Return blockers that prevent autonomous dispatch authorization.

        Planning rows are useful for ranking, but they are not executable
        packets. The le-$5 autopilot path may only self-authorize rows that
        already carry lane identity, contest target metadata, and a hash of the
        dispatch packet or exact archive/runtime packet being launched.
        """
        blockers: list[str] = []
        if self.score_claim:
            blockers.append("score_claim_true_requires_operator_review")
        if self.promotion_eligible:
            blockers.append("promotion_eligible_true_requires_operator_review")
        if not self.dispatch_packet_ready:
            blockers.append("dispatch_packet_ready_false")
        if not self.lane_id.strip():
            blockers.append("lane_id_required_for_dispatch_packet")
        contest_exact_target = AUTOPILOT_CONTEST_TARGET_MODE in set(self.target_modes)
        if not contest_exact_target:
            blockers.append("contest_exact_eval_target_mode_required")
        has_dispatch_packet_hash = _is_sha256_hex(self.dispatch_packet_sha256)
        has_archive_hash = _is_sha256_hex(self.archive_sha256)
        has_runtime_hash = _is_sha256_hex(self.runtime_tree_sha256)
        has_exact_packet_hashes = has_archive_hash and has_runtime_hash
        if self.dispatch_packet_sha256.strip() and not has_dispatch_packet_hash:
            blockers.append("dispatch_packet_sha256_malformed")
        if self.archive_sha256.strip() and not has_archive_hash:
            blockers.append("archive_sha256_malformed")
        if self.runtime_tree_sha256.strip() and not has_runtime_hash:
            blockers.append("runtime_tree_sha256_malformed")
        if contest_exact_target and not has_exact_packet_hashes:
            blockers.append("contest_exact_eval_requires_archive_and_runtime_hash")
        if contest_exact_target and self.ready_for_exact_eval_dispatch is not True:
            blockers.append("contest_exact_eval_requires_ready_for_exact_eval_dispatch")
        elif not (has_dispatch_packet_hash or has_exact_packet_hashes):
            blockers.append("dispatch_packet_or_archive_runtime_hash_required")
        if self.ready_for_exact_eval_dispatch and not has_exact_packet_hashes:
            blockers.append(
                "ready_for_exact_eval_dispatch_requires_archive_and_runtime_hash"
            )
        return blockers


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

    def can_authorize(self, candidate: CandidateRow) -> tuple[bool, str]:
        """Return ``(authorized, reason)``.

        Authorization requires every precondition to hold. The first failing
        precondition's reason is returned; on success the reason is empty.
        """
        if not self.enabled:
            return False, "operator-authorized mode is OFF (default safe HALT-and-ASK)"
        if not math.isfinite(self.per_dispatch_cap_usd) or self.per_dispatch_cap_usd <= 0.0:
            return False, (
                f"per-dispatch cap {self.per_dispatch_cap_usd!r} is not finite-positive; "
                "operator must fix authorized-mode configuration"
            )
        if not math.isfinite(self.cumulative_cap_usd) or self.cumulative_cap_usd <= 0.0:
            return False, (
                f"cumulative cap {self.cumulative_cap_usd!r} is not finite-positive; "
                "operator must fix authorized-mode configuration"
            )
        if (
            not math.isfinite(candidate.estimated_dispatch_cost_usd)
            or candidate.estimated_dispatch_cost_usd <= 0.0
        ):
            return False, (
                f"candidate cost {candidate.estimated_dispatch_cost_usd!r} is "
                "not finite-positive; refuse to authorize a malformed estimate"
            )
        if not math.isfinite(self.cumulative_spent_usd) or self.cumulative_spent_usd < 0.0:
            return False, (
                f"cumulative spent {self.cumulative_spent_usd!r} is malformed; "
                "operator must reset or audit authorized-mode state"
            )
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
        authority_blockers = candidate.dispatch_authority_blockers()
        if authority_blockers:
            return False, (
                f"candidate is not a dispatch-authority packet; unresolved "
                f"authority blockers {authority_blockers!r}; operator must "
                "adjudicate before any autonomous dispatch"
            )
        if not self.canonical_helper_script or not self.canonical_helper_script.is_file():
            return False, (
                f"canonical helper script {self.canonical_helper_script!r} does not "
                "exist; operator must point --canonical-helper-script at a real file"
            )
        return True, ""

    def record_authorization(self, candidate: CandidateRow) -> None:
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
    lane_id: str = ""
    claim_keys: list[str] = field(default_factory=list)
    target_modes: list[str] = field(default_factory=list)
    dispatch_packet_sha256: str = ""
    archive_sha256: str = ""
    runtime_tree_sha256: str = ""
    timing_smoke_command: str = ""
    ready_for_exact_eval_dispatch: bool = False
    literature_anchor: str = ""
    source_supports: str = ""
    paper_claim_scope: str = ""
    pact_must_prove: str = ""
    decode_complexity_evidence: str = ""
    decision: OperatorDecision | None = None
    decision_at_utc: str | None = None
    decision_notes: str = ""
    # Operator-authorized le-$5/individual mode fields (2026-05-11).
    autopilot_authorized: bool = False
    autopilot_tag: str = ""
    autopilot_authorized_reason: str = ""
    autopilot_refused_reason: str = ""
    autopilot_claim_recorded: bool = False
    autopilot_claim_instance_job_id: str = ""
    autopilot_claim_reason: str = ""


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


# ── Z1 empirical revision: MDL-density penalty + class-shift reward ────────
#
# Per `feedback_z1_mdl_ablation_landed_20260514.md` operator decision #4
# (2026-05-14): update the cathedral autopilot ranker to penalize within-
# HNeRV-class candidates (high MDL density = class-saturated) and reward
# predictive-receiver / cooperative-receiver / foveation / class-shift
# candidates. The Z1 ablation empirically established 0.90+ MDL density as
# the within-class trap threshold.
#
# Per CLAUDE.md "Forbidden score claims": these adjustments operate on
# PREDICTED scores only. They never claim or alter measured scores. The
# adjustment is a ranking re-weight, not a score promotion.
#
# Per CLAUDE.md "Subagent coherence-by-default" hook #4 (cathedral autopilot
# dispatch hook): these functions are the canonical hook for the autopilot
# ranker to honor the Z1 empirical revision automatically.

# MDL-density thresholds (per Z1 empirical revision 2026-05-14):
#   density > 0.95  -> within-class; cap predicted_score_delta at -0.005
#                       (floor effective predicted improvement near zero)
#   density > 0.90  -> within-class trending; 50% penalty
#   density < 0.90  -> across-class promising; no penalty
#   density unknown -> no penalty (don't punish lack-of-evidence)
MDL_DENSITY_WITHIN_CLASS_SATURATED_THRESHOLD = 0.95
MDL_DENSITY_WITHIN_CLASS_TRENDING_THRESHOLD = 0.90
MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR = -0.005  # negligible
MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR = 0.5  # halve predicted ΔS

# Class-shift reward tokens. Candidates whose lane_class or literature_anchor
# match any of these get a predicted_score_delta bonus per the Z1 council's
# decision #4 (reward cooperative-receiver / predictive-receiver / foveation
# / class-shift candidates). The reward is small (additive ~0.01-0.02) and
# subtractive (lower = better in score-delta space).
_CLASS_SHIFT_LANE_CLASS_TOKENS = (
    "substrate_class_shift",
    "predictive_receiver",
    "cooperative_receiver",
    "foveation",
)

_CLASS_SHIFT_LITERATURE_TOKENS = (
    "cooperative-receiver",
    "cooperative_receiver",
    "predictive-coding",
    "predictive_coding",
    "predictive-receiver",
    "predictive_receiver",
    "foveation",
    "Tishby-Zaslavsky",
    "Tishby Zaslavsky",
    "Atick-Redlich",
    "Atick Redlich",
    "Rao-Ballard",
    "Rao Ballard",
    "Wyner-Ziv",
    "Wyner Ziv",
    "Information Bottleneck",
    "information_bottleneck",
    "MDL-IBPS",
    "Slepian-Wolf",
    "Slepian Wolf",
    "world_model",
    "world-model",
    "time_traveler",
    "time-traveler",
    # 2026-05-14 C1 council "RETAIN" decision per
    # `project_c1_world_model_revision_SUPERSEDED_by_council_unfair_probe_finding_20260514.md`:
    # the C1 world-model architecture revision was DEFERRED-pending-fair-probe-v2,
    # NOT KILLED. The literature anchor for the world-model class IS
    # Ha-Schmidhuber 2018 + Hafner DreamerV3 2023; both must be retained as
    # autopilot-ranker class-shift reward tokens so the probe-v2 evidence loop
    # (when it lands) inherits the canonical priority. Per CLAUDE.md
    # "Forbidden premature KILL without research exhaustion".
    "Ha-Schmidhuber",
    "Ha Schmidhuber",
    "Hafner",
    "DreamerV3",
    "Dreamer V3",
    "Dreamer-V3",
    "balle_2018",
    "Ballé",
    "Balle",
    "scale-hyperprior",
    "scale_hyperprior",
)

CLASS_SHIFT_LANE_CLASS_REWARD = 0.02
CLASS_SHIFT_LITERATURE_ANCHOR_REWARD = 0.01

# Per the 2026-05-14 grand-council reconvening
# (`feedback_c1_council_reconvene_post_probe_v2_landed_20260514.md` Decision 6
# HALF-MEASURE; council vote 4/11 explicit Contrarian + Quantizr + MacKay +
# Time-Traveler peer): probe v2 (FAIR Hafner DreamerV3 RSSM at matched-DOF +
# matched-bit-budget) reports world-model loses 99.98-100% margin in
# feature-space proxy regime. The verdict is NOT a class falsification per
# CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden premature KILL"
# — the regime distinction (FP4 quantization, SegNet+PoseNet preprocess
# invariances, 1200-frame temporal scaling) preserves a bidirectional posterior
# Δ ∈ [-0.04, +0.05] — but the strong-prior signal warrants halving the
# autopilot-ranker C1-class literature-anchor reward as a conditional revision
# pending the dispositive Z5 dispatch (Decision 1 Option β).
#
# RETAIN: the literature anchor stays in `_CLASS_SHIFT_LITERATURE_TOKENS` so
# the C1 lane is NOT closed (Decision δ DROP REJECTED 11/11 — Contrarian
# SUPER-VETO eligible).
# HALVE: when the matched literature token is one of the C1-class tokens
# below, the literature-anchor reward contribution is HALVED (0.01 -> 0.005).
# Combined with the lane_class reward (0.02), the effective C1-class candidate
# reward drops from ~0.025 stacked to ~0.0125 stacked — the 50% reduction the
# council ledger Decision 6 specifies.
#
# REVERT: this halve is a conditional revision. Update to full reward IF Z5
# (`lane_z5_predictive_coding_world_model_step3_20260514`) dispatch returns
# dispositive positive evidence; revert to zero IF Z5 + Decision 1 alpha
# (contest-scale C1) dispatch are jointly dispositive negative.
_C1_HALVED_LITERATURE_TOKENS = (
    "Ha-Schmidhuber",
    "Ha Schmidhuber",
    "Hafner",
    "DreamerV3",
    "Dreamer V3",
    "Dreamer-V3",
)
_C1_LITERATURE_ANCHOR_HALVE_FACTOR = 0.5


def adjust_predicted_delta_for_mdl_density(
    base_delta: float,
    mdl_density: float | None,
) -> float:
    """Penalize within-HNeRV-class candidates (high MDL density = class-
    saturated) per the Z1 empirical revision 2026-05-14.

    Per Z1 council decision #4:
      - density > 0.95: within-class; floor predicted_score_delta at
        ``MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR`` (effectively
        zero improvement)
      - density 0.90-0.95: within-class trending; apply 50% penalty to
        predicted savings (less-negative becomes more-positive)
      - density < 0.90: across-class promising; no penalty
      - density unknown: no penalty

    Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only;
    measured anchors are untouched.

    Returns the (possibly penalized) predicted_score_delta. Lower is better
    in the score-delta convention (negative = improvement).
    """
    if mdl_density is None:
        return base_delta
    try:
        d = float(mdl_density)
    except (TypeError, ValueError):
        return base_delta
    if d > MDL_DENSITY_WITHIN_CLASS_SATURATED_THRESHOLD:
        # Floor at near-zero. If the candidate predicted a strong negative
        # delta, the within-class density says that prediction is empirically
        # unrealistic; cap at the conservative floor.
        return max(base_delta, MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR)
    if d > MDL_DENSITY_WITHIN_CLASS_TRENDING_THRESHOLD:
        # Halve the predicted savings (only affects negative deltas; positive
        # deltas stay the same magnitude since we multiply).
        return base_delta * MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR
    return base_delta


def adjust_predicted_delta_for_class_shift(
    base_delta: float,
    *,
    lane_class: str | None = None,
    literature_anchor: str = "",
    notes: str = "",
) -> float:
    """Reward substrate-class-shift candidates per Z1 council decision #4.

    Acceptance:
      - ``lane_class`` matches any of ``_CLASS_SHIFT_LANE_CLASS_TOKENS``
        adds ``CLASS_SHIFT_LANE_CLASS_REWARD`` (subtracted from base delta;
        lower = better in score-delta convention)
      - ``literature_anchor`` or ``notes`` contains any of
        ``_CLASS_SHIFT_LITERATURE_TOKENS`` adds
        ``CLASS_SHIFT_LITERATURE_ANCHOR_REWARD``

    Both bonuses stack independently. Per CLAUDE.md "Forbidden score claims":
    this is a PREDICTED ΔS reweight, not a score promotion.
    """
    bonus = 0.0
    if isinstance(lane_class, str) and lane_class:
        for tok in _CLASS_SHIFT_LANE_CLASS_TOKENS:
            if tok in lane_class:
                bonus += CLASS_SHIFT_LANE_CLASS_REWARD
                break
    haystacks: list[str] = []
    if isinstance(literature_anchor, str):
        haystacks.append(literature_anchor)
    if isinstance(notes, str):
        haystacks.append(notes)
    for hay in haystacks:
        if not hay:
            continue
        matched_token = next(
            (tok for tok in _CLASS_SHIFT_LITERATURE_TOKENS if tok in hay),
            None,
        )
        if matched_token is None:
            continue
        # Per Decision 6 HALF-MEASURE 2026-05-14: halve the literature-anchor
        # reward when the matched token is a C1-class token (Hafner /
        # DreamerV3 / Ha-Schmidhuber). All other class-shift tokens
        # (cooperative-receiver, predictive-coding, foveation, MDL-IBPS, ...)
        # keep the full literature-anchor reward.
        if matched_token in _C1_HALVED_LITERATURE_TOKENS:
            bonus += (
                CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
                * _C1_LITERATURE_ANCHOR_HALVE_FACTOR
            )
        else:
            bonus += CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
        break
    # Lower-is-better in score-delta convention; subtract the bonus to make
    # this candidate look better in the ranking.
    return base_delta - bonus


def apply_z1_empirical_revision_to_candidate_delta(c: CandidateRow) -> float:
    """Return the rank-key predicted_score_delta after Z1 empirical revision
    adjustments. This is the canonical composition: first apply MDL-density
    penalty (Tier A), then Tier C density penalty/reward (Catalog #227),
    then class-shift reward, then composition matrix factor (Catalog #227).

    Per CLAUDE.md "Subagent coherence-by-default" (Z1 wire-in hook #4): this
    helper is the single source of truth for "what does the autopilot
    *actually* believe about this candidate's predicted_score_delta after
    integrating the Z1 ablation evidence?"

    The ORIGINAL ``c.predicted_score_delta`` is preserved on the row; this
    function returns a transient sort-key value, never mutates the row.
    """
    d = adjust_predicted_delta_for_mdl_density(
        c.predicted_score_delta, c.mdl_density
    )
    # Catalog #227 Tier C wire-in: substrate-class signal overrides Tier A
    # for class discrimination. Apply BEFORE the class-shift reward so the
    # Tier C verdict can either reinforce (across-class density LOW + bonus)
    # or contradict (across-class evidence already captured by Tier C
    # alone, even without a literature_anchor).
    d = adjust_predicted_delta_for_mdl_tier_c_density(d, c.mdl_tier_c_density)
    d = adjust_predicted_delta_for_class_shift(
        d,
        lane_class=c.lane_class,
        literature_anchor=c.literature_anchor,
        notes=c.notes,
    )
    # Catalog #227 composition matrix wire-in: stacking of two substrates
    # has an additivity factor alpha in [0, 1+] per the Z3xC6 probe-disambiguator
    # (`tools/probe_z3_x_c6_composition_disambiguator.py`). The factor maps
    # to a predicted_score_delta scaling per the additivity verdict bands.
    d = adjust_predicted_delta_for_composition_alpha(d, c.composition_alpha)
    # OP-3 predicted_dispatch_risk wire-in (codex chunk 6 finding,
    # 2026-05-15): demote candidates whose RUDIN-DAUBECHIES preflight SLIM
    # risk score crosses the canonical refusal threshold. Applied LAST in
    # the chain so it can floor a candidate whose score-axis adjustments
    # (Tier A / Tier C / class-shift / composition-alpha) would otherwise
    # promote it past safer peers — preflight risk is a STRUCTURAL refusal,
    # not a score-axis penalty, so it overrides the score-axis stack.
    d = adjust_predicted_delta_for_predicted_dispatch_risk(
        d, c.predicted_dispatch_risk
    )
    return d


# ── Tier C substrate-class density (Catalog #227 wire-in, 2026-05-14) ─────
#
# Per `feedback_mdl_ablation_tier_c_ibps1_landed_20260514.md` operator
# decision #4 (RECOMMENDED-LAND-NEXT): wire Tier C decoder/latent curve-knee
# into the autopilot ranker. Tier A is brotli-saturated at the byte layer and
# CANNOT discriminate substrate class; Tier C is the dispositive disambiguator
# per Z1 deep-math §3.5 + Tishby-Zaslavsky 2015.
#
# Thresholds (per Z1 council band math + C6 5ep anchor density ≈ 0.13):
#   density >= 0.70 (within-class) → cap at floor (effective zero
#                                    improvement; sister of Tier A 0.95 cap)
#   density >= 0.50 (within-class trending) → 50% penalty
#   density <= 0.30 (across-class) → 0.01 ΔS bonus (subtract → more-negative
#                                    = better in score-delta convention).
#                                    Sister of CLASS_SHIFT_LITERATURE_ANCHOR_
#                                    REWARD but applied only when EMPIRICAL
#                                    Tier C evidence backs the across-class
#                                    claim, not just lineage.
#   density unknown (None) → no adjustment (don't punish lack-of-evidence).
#
# The Tier C signal is STRONGER than the Tier A signal for substrate-class
# discrimination, but Tier A still captures within-class encoder/codec
# saturation which Tier C alone does not. The composition in
# `apply_z1_empirical_revision_to_candidate_delta` applies BOTH so a
# within-Tier-A candidate gets penalized AND a within-Tier-C candidate is
# further capped at the floor.
MDL_TIER_C_WITHIN_CLASS_SATURATED_THRESHOLD = 0.70
MDL_TIER_C_WITHIN_CLASS_TRENDING_THRESHOLD = 0.50
MDL_TIER_C_ACROSS_CLASS_THRESHOLD = 0.30
MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR = -0.005  # sister of Tier A
MDL_TIER_C_WITHIN_CLASS_TRENDING_PENALTY_FACTOR = 0.5
MDL_TIER_C_ACROSS_CLASS_BONUS = 0.01  # subtract from delta (lower = better)


def adjust_predicted_delta_for_mdl_tier_c_density(
    base_delta: float,
    mdl_tier_c_density: float | None,
) -> float:
    """Apply substrate-class-discrimination Tier C penalty/reward to the
    predicted_score_delta.

    Per Catalog #227 / Tier C wire-in (`feedback_mdl_ablation_tier_c_ibps1_
    landed_20260514.md` operator decision #4):

      - density >= 0.70: within-class (Tier C confirms); floor delta at
        ``MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR``.
      - density >= 0.50: within-class trending; 50% penalty.
      - density <= 0.30: across-class (Tier C confirms); apply
        ``MDL_TIER_C_ACROSS_CLASS_BONUS`` subtraction (more-negative = better).
      - 0.30 < density < 0.50: indeterminate; no adjustment.
      - density unknown (None) or non-numeric: no adjustment.

    Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only.
    Returns the adjusted predicted_score_delta. Lower is better.
    """
    if mdl_tier_c_density is None:
        return base_delta
    try:
        d = float(mdl_tier_c_density)
    except (TypeError, ValueError):
        return base_delta
    if d >= MDL_TIER_C_WITHIN_CLASS_SATURATED_THRESHOLD:
        return max(base_delta, MDL_TIER_C_WITHIN_CLASS_SATURATED_DELTA_FLOOR)
    if d >= MDL_TIER_C_WITHIN_CLASS_TRENDING_THRESHOLD:
        return base_delta * MDL_TIER_C_WITHIN_CLASS_TRENDING_PENALTY_FACTOR
    if d <= MDL_TIER_C_ACROSS_CLASS_THRESHOLD:
        # Subtract bonus → more-negative → better-ranked in score-delta
        # convention. Sister of the class-shift literature bonus but
        # backed by Tier C empirical evidence rather than lineage tokens.
        return base_delta - MDL_TIER_C_ACROSS_CLASS_BONUS
    # 0.30 < density < 0.50 → indeterminate; no adjustment.
    return base_delta


# ── Substrate composition matrix factor (Catalog #227 wire-in, 2026-05-14) ─
#
# Per `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`:
# the Z3xC6 composition probe-disambiguator returns an additivity factor alpha
# in the [0, 1+] band:
#
#   alpha > 0.7   (ADDITIVE) -> stacking realizes additive savings; predicted_
#                          score_delta scaled by 1.0 (no penalty)
#   alpha 0.3-0.7 (SUB-ADDITIVE) -> marginal stacking; halve predicted savings
#   alpha <= 0.3  (SATURATING) -> single-substrate dominates; cap at floor
#
# The matrix is the canonical posterior surface
# `.omx/state/substrate_composition_matrix.json` populated by every probe
# invocation. The ranker reads the candidate's `composition_alpha` field;
# upstream loaders (`load_candidates_from_substrate_composition_ranking`)
# populate it when QQ's substrate composition ranker emits the field.
#
# Per CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator
# pattern": the probe IS the arbitration; the ranker consumes the verdict.
COMPOSITION_ALPHA_ADDITIVE_THRESHOLD = 0.7  # >0.7 = additive, no penalty
COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD = 0.3  # 0.3-0.7 = sub-additive
COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR = 0.5  # halve savings
COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR = -0.005  # sister of MDL-density


def adjust_predicted_delta_for_composition_alpha(
    base_delta: float,
    composition_alpha: float | None,
) -> float:
    """Apply substrate composition additivity factor to predicted_score_delta.

    Per Catalog #227 composition matrix wire-in
    (`feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`):

      - alpha > 0.7  (ADDITIVE): no adjustment (full additive savings)
      - alpha 0.3-0.7 (SUB-ADDITIVE): 50% penalty on predicted savings
      - alpha <= 0.3 (SATURATING): floor at -0.005 (single-substrate dominates)
      - alpha unknown (None): no adjustment (single-substrate candidate or
        composition evidence not yet collected)

    Returns the (possibly adjusted) predicted_score_delta. Lower is better.
    """
    if composition_alpha is None:
        return base_delta
    try:
        a = float(composition_alpha)
    except (TypeError, ValueError):
        return base_delta
    if a > COMPOSITION_ALPHA_ADDITIVE_THRESHOLD:
        # ADDITIVE: full additive savings realized; no adjustment.
        return base_delta
    if a > COMPOSITION_ALPHA_SUB_ADDITIVE_THRESHOLD:
        # SUB-ADDITIVE: halve the predicted savings.
        return base_delta * COMPOSITION_ALPHA_SUB_ADDITIVE_PENALTY_FACTOR
    # SATURATING (alpha <= 0.3): single-substrate dominates; cap at floor.
    return max(base_delta, COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR)


# ── Predicted dispatch risk adjuster (OP-3 wire-in, 2026-05-15) ───────────
#
# Per codex chunk 6 finding (`.omx/research/codex_chunked_full_codebase_review_
# 20260515.md`): the RUDIN-DAUBECHIES preflight composite computes a SLIM
# (Sparse Linear Integer Model) `predicted_dispatch_risk` score over the
# preflight-gate verdict panel (see ``tac.preflight_rudin_daubechies.
# slim_risk_scorer.PreflightSLIMRiskScorer``) but the cathedral autopilot
# ranker did NOT consume it — the continual-learning loop closed on the
# RANKER side but left the PREFLIGHT-RISK signal stranded outside the
# rank_candidates composition chain.
#
# This adjuster wires the SLIM preflight risk into the same
# `apply_z1_empirical_revision_to_candidate_delta` chain that already stacks
# Tier A / Tier C / class-shift / composition-alpha so high-preflight-risk
# candidates are demoted in the autopilot ordering BEFORE any operator-
# authorized dispatch fires. The risk bands match the canonical
# `DISPATCH_RISK_REFUSAL_THRESHOLD = 50` from
# ``tac.preflight_rudin_daubechies.slim_risk_scorer`` (the SLIM scorer's own
# refusal threshold) plus a halve-band at 25 for moderate-risk candidates so
# the ranker degrades gracefully rather than cliff-edging at 50.
#
# Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only;
# measured anchors are untouched. The original ``predicted_score_delta`` on
# the row is preserved.
PREDICTED_DISPATCH_RISK_REFUSAL_THRESHOLD = 50.0  # mirrors slim_risk_scorer
PREDICTED_DISPATCH_RISK_MODERATE_THRESHOLD = 25.0  # halve-band lower edge
PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR = 0.0  # cliff at zero (no improvement)
PREDICTED_DISPATCH_RISK_MODERATE_PENALTY_FACTOR = 0.5  # halve predicted savings


def adjust_predicted_delta_for_predicted_dispatch_risk(
    base_delta: float,
    predicted_dispatch_risk: float | None,
) -> float:
    """Demote high-preflight-risk candidates per the OP-3 SLIM wire-in.

    Per codex chunk 6 finding (2026-05-15): the RUDIN-DAUBECHIES preflight
    SLIM scorer (``tac.preflight_rudin_daubechies.slim_risk_scorer``) emits
    a per-candidate ``predicted_dispatch_risk`` denominated in the same
    integer-coefficient space as the preflight gate panel (Tier-1 = +25 per
    violation, META-meta = +50 per violation, refusal threshold = 50.0). The
    ranker treats the score in three bands matching the SLIM scorer's own
    semantics:

      - risk >= ``PREDICTED_DISPATCH_RISK_REFUSAL_THRESHOLD`` (50.0):
        REFUSE — floor at ``PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR``
        (effectively zero predicted improvement; the candidate is structurally
        unsafe to dispatch even if the predicted ΔS is strongly negative).
      - risk 25-50 (``PREDICTED_DISPATCH_RISK_MODERATE_THRESHOLD`` <= risk <
        refusal): MODERATE — halve predicted savings so the candidate falls
        below clean-preflight peers in the ordering but is still rankable.
      - risk < 25: LOW — no adjustment (gate-discipline is on the safe side
        of the SLIM threshold).
      - risk unknown (None) or non-numeric: no adjustment (don't penalize
        lack-of-evidence per the sister Z1 / Tier C / composition_alpha
        conventions).

    Per CLAUDE.md "Forbidden score claims": this adjusts PREDICTED ΔS only.
    Returns the (possibly demoted) predicted_score_delta. Lower is better in
    the score-delta convention (negative = improvement).

    [verified-against:
     ``tac.preflight_rudin_daubechies.slim_risk_scorer.DISPATCH_RISK_REFUSAL_
     THRESHOLD = 50.0``]
    """
    if predicted_dispatch_risk is None:
        return base_delta
    try:
        r = float(predicted_dispatch_risk)
    except (TypeError, ValueError):
        return base_delta
    if r >= PREDICTED_DISPATCH_RISK_REFUSAL_THRESHOLD:
        # Refuse: floor at zero so no improvement is ranked from this row,
        # regardless of how strongly negative its raw predicted_score_delta
        # was. The SLIM scorer has flagged the candidate as structurally
        # unsafe; the operator must clear the preflight panel first.
        return max(base_delta, PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR)
    if r >= PREDICTED_DISPATCH_RISK_MODERATE_THRESHOLD:
        # Moderate: halve the predicted savings (only affects negative
        # deltas; positive deltas stay the same magnitude under multiply).
        return base_delta * PREDICTED_DISPATCH_RISK_MODERATE_PENALTY_FACTOR
    # LOW (risk < 25): no adjustment.
    return base_delta


def rank_candidates(
    candidates: list[CandidateRow],
    *,
    rank_axis: str = "eig_per_dollar",
    continual_posterior: Any | None = None,
    apply_z1_empirical_revision: bool = True,
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

    When ``apply_z1_empirical_revision=True`` (default), the Z1 empirical
    revision adjustments are applied BEFORE the continual-posterior
    correction: MDL-density penalty (within-class trap) + class-shift
    reward (cooperative-receiver / predictive-receiver / foveation). Per
    `feedback_z1_mdl_ablation_landed_20260514.md` operator decision #4. The
    ORIGINAL ``predicted_score_delta`` on the row is preserved; the
    adjustment is applied transiently for sort-key purposes only.

    Per CLAUDE.md "Forbidden /tmp paths": no temp paths used; pure in-memory.
    """
    for candidate in candidates:
        _require_candidate_planning_cost(candidate)

    def _effective_delta(c: CandidateRow) -> float:
        if _candidate_prediction_band_rank_reward_suppressed(c):
            return 0.0
        if apply_z1_empirical_revision:
            return apply_z1_empirical_revision_to_candidate_delta(c)
        return c.predicted_score_delta

    def _effective_eig_per_dollar(c: CandidateRow) -> float:
        if _candidate_prediction_band_rank_reward_suppressed(c):
            return 0.0
        base = c.eig_per_dollar()
        if not apply_z1_empirical_revision:
            return base

        # Preserve the existing Tier-A MDL-density EIG modifier so saturated
        # within-class rows stay down-ranked even when their raw score delta
        # already sits on the Z1 floor.
        tier_a_eig = base
        if c.mdl_density is None:
            tier_a_delta = c.predicted_score_delta
        else:
            try:
                d = float(c.mdl_density)
            except (TypeError, ValueError):
                tier_a_delta = c.predicted_score_delta
            else:
                tier_a_delta = adjust_predicted_delta_for_mdl_density(
                    c.predicted_score_delta, d
                )
                if d > MDL_DENSITY_WITHIN_CLASS_SATURATED_THRESHOLD:
                    tier_a_eig = base * 0.10  # 90% penalty
                elif d > MDL_DENSITY_WITHIN_CLASS_TRENDING_THRESHOLD:
                    tier_a_eig = (
                        base * MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR
                    )

        # The default autopilot axis is EIG/$, so score-delta-only Z1 signals
        # would otherwise be invisible in normal operation. Reweight the EIG
        # prior by the same non-Tier-A score-delta adjustment chain used by the
        # explicit predicted-score axis: Tier C, class-shift literature, and
        # composition alpha.
        full_delta = apply_z1_empirical_revision_to_candidate_delta(c)
        tier_a_gain = max(0.0, -tier_a_delta)
        full_gain = max(0.0, -full_delta)
        if tier_a_gain > 0.0:
            return tier_a_eig * (full_gain / tier_a_gain)
        if full_gain > 0.0 and c.estimated_dispatch_cost_usd > 0.0:
            return full_gain / c.estimated_dispatch_cost_usd
        return tier_a_eig

    if rank_axis == "eig_per_dollar":
        if continual_posterior is None:
            return sorted(candidates, key=_effective_eig_per_dollar, reverse=True)
        # Reweight EIG/$ by posterior correction. EIG itself is unchanged; the
        # cost-effective dispatch ordering still reflects empirical bias.
        def _eig_key(c: CandidateRow) -> float:
            factor, _, _ = _posterior_correction_factor(c, continual_posterior)
            return _effective_eig_per_dollar(c) * factor
        return sorted(candidates, key=_eig_key, reverse=True)
    if rank_axis == "predicted_score_delta":
        if continual_posterior is None:
            return sorted(candidates, key=_effective_delta)
        # Reweight predicted_score_delta by posterior correction. Most-negative
        # first ordering preserved.
        def _delta_key(c: CandidateRow) -> float:
            factor, _, _ = _posterior_correction_factor(c, continual_posterior)
            return _effective_delta(c) * factor
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
    roots = (
        [REPO_ROOT / "experiments" / "results"]
        if search_dirs is None
        else list(search_dirs)
    )
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


def _claim_token(value: str, *, fallback: str, max_len: int = 96) -> str:
    """Return a claim-helper-safe token with no whitespace."""
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip())
    token = token.strip("._:-")
    return (token[:max_len] or fallback)


def _claim_note_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("|", "/")).strip()


def _autopilot_claim_instance_job_id(candidate: CandidateRow) -> str:
    slug = _claim_token(
        candidate.candidate_id or candidate.lane_id,
        fallback="candidate",
        max_len=80,
    )
    packet_hash = (
        candidate.dispatch_packet_sha256
        or candidate.archive_sha256
        or candidate.runtime_tree_sha256
    )
    suffix = _claim_token(packet_hash[:12], fallback="nohash", max_len=16)
    return f"cathedral_autopilot_{slug}_{suffix}"


def _record_autopilot_dispatch_claim(
    candidate: CandidateRow,
    *,
    auth_mode: OperatorAuthorizedModeConfig,
    claims_path: Path,
) -> tuple[bool, str, str]:
    """Claim the lane before self-authorization can become non-blocking."""
    helper = auth_mode.canonical_helper_script
    instance_job_id = _autopilot_claim_instance_job_id(candidate)
    if helper is None or not helper.is_file():
        return (
            False,
            f"canonical claim helper {helper!r} is unavailable",
            instance_job_id,
        )

    notes = "; ".join(
        part for part in (
            "Cathedral autopilot self-authorization claim before requires_approval=false",
            f"candidate_id={_claim_note_value(candidate.candidate_id)}",
            f"dispatch_packet_sha256={_claim_note_value(candidate.dispatch_packet_sha256)}",
            f"archive_sha256={_claim_note_value(candidate.archive_sha256)}",
            f"runtime_tree_sha256={_claim_note_value(candidate.runtime_tree_sha256)}",
            f"estimated_cost_usd={candidate.estimated_dispatch_cost_usd:.4f}",
            "remote_provider_job_spawned=false",
        ) if part
    )
    cmd = [
        sys.executable,
        str(helper),
        "claim",
        "--lane-id",
        candidate.lane_id,
        "--platform",
        AUTOPILOT_CLAIM_PLATFORM,
        "--instance-job-id",
        instance_job_id,
        "--agent",
        AUTOPILOT_CLAIM_AGENT,
        "--status",
        AUTOPILOT_CLAIM_STATUS,
        "--notes",
        notes,
        "--ttl-hours",
        str(AUTOPILOT_CLAIM_TTL_HOURS),
    ]
    cmd.extend(["--claims-path", str(claims_path)])
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - filesystem/interpreter failure
        return (
            False,
            f"dispatch claim helper invocation failed: {type(exc).__name__}: {exc}",
            instance_job_id,
        )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        detail = stderr or stdout or "no output"
        return (
            False,
            f"dispatch claim helper refused claim with rc={result.returncode}: {detail}",
            instance_job_id,
        )
    return True, stdout or "dispatch claim recorded", instance_job_id


def make_dispatch_halt_event(
    candidate: CandidateRow,
    *,
    requires_approval_classes: frozenset[EventClass],
    blockers: list[str] | None = None,
    auth_mode: OperatorAuthorizedModeConfig | None = None,
    env_authorized: bool | None = None,
    claims_path: Path | None = None,
) -> HaltEvent:
    """Construct a HALT event for one dispatch decision.

    Per CLAUDE.md "operator-gate non-negotiable": when ``EventClass.DISPATCH``
    is in ``requires_approval_classes``, ``requires_approval=True`` UNLESS the
    operator-authorized le-$5/individual mode is engaged AND every precondition
    holds for THIS candidate. In that case the event is tagged
    ``[autopilot-claude-le-5-dollar]`` only after
    :mod:`tools.claim_lane_dispatch` records a lane claim. ``requires_approval``
    is set to False only after that claim succeeds.

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
    autopilot_claim_recorded = False
    autopilot_claim_instance_job_id = ""
    autopilot_claim_reason = ""

    if auth_mode is not None and auth_mode.enabled:
        config_blocker = _authorized_mode_config_blocker(auth_mode, repo_root=REPO_ROOT)
        if config_blocker:
            requires = True
            autopilot_refused = config_blocker
            halt_blockers.append("operator_authorized_mode_config_invalid")
        else:
            env_ok = (
                _env_authorizes_mode()
                if env_authorized is None
                else bool(env_authorized)
            )
            if not env_ok:
                requires = True
                autopilot_refused = (
                    f"env-var {OPERATOR_AUTHORIZED_MODE_ENV_VAR}="
                    f"{OPERATOR_AUTHORIZED_MODE_ENV_VALUE_ENABLED} is missing; CLI "
                    "flag alone is insufficient (defense-in-depth)"
                )
            else:
                ok, reason = auth_mode.can_authorize(candidate)
                if ok:
                    if claims_path is None:
                        autopilot_claim_reason = (
                            "claims_path is required before self-authorization so "
                            "tests and direct callers cannot silently write an "
                            "implicit dispatch claim"
                        )
                    else:
                        (
                            autopilot_claim_recorded,
                            autopilot_claim_reason,
                            autopilot_claim_instance_job_id,
                        ) = _record_autopilot_dispatch_claim(
                            candidate,
                            auth_mode=auth_mode,
                            claims_path=claims_path,
                        )
                    if autopilot_claim_recorded:
                        prospective = (
                            auth_mode.cumulative_spent_usd
                            + candidate.estimated_dispatch_cost_usd
                        )
                        autopilot_authorized = True
                        autopilot_tag = AUTOPILOT_AUTHORIZED_TAG
                        autopilot_reason = (
                            f"per-dispatch cost ${candidate.estimated_dispatch_cost_usd:.4f} "
                            f"<= cap ${auth_mode.per_dispatch_cap_usd:.4f}; "
                            f"cumulative ${prospective:.4f} "
                            f"<= envelope ${auth_mode.cumulative_cap_usd:.4f}; "
                            f"dispatch claim recorded as {autopilot_claim_instance_job_id}"
                        )
                        # Reserve cost in the per-process counter so the next
                        # candidate in this iteration sees the updated cumulative.
                        auth_mode.record_authorization(candidate)
                        requires = False
                    else:
                        requires = True
                        autopilot_refused = (
                            "dispatch claim is required before self-authorization; "
                            f"{autopilot_claim_reason}"
                        )
                        halt_blockers.append(
                            "dispatch_claim_required_for_self_authorization"
                        )
                else:
                    requires = True
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
        lane_id=candidate.lane_id,
        claim_keys=candidate.dispatch_claim_keys(),
        target_modes=list(candidate.target_modes),
        dispatch_packet_sha256=candidate.dispatch_packet_sha256,
        archive_sha256=candidate.archive_sha256,
        runtime_tree_sha256=candidate.runtime_tree_sha256,
        timing_smoke_command=candidate.timing_smoke_command,
        ready_for_exact_eval_dispatch=candidate.ready_for_exact_eval_dispatch,
        literature_anchor=candidate.literature_anchor,
        source_supports=candidate.source_supports,
        paper_claim_scope=candidate.paper_claim_scope,
        pact_must_prove=candidate.pact_must_prove,
        decode_complexity_evidence=candidate.decode_complexity_evidence,
        autopilot_authorized=autopilot_authorized,
        autopilot_tag=autopilot_tag,
        autopilot_authorized_reason=autopilot_reason,
        autopilot_refused_reason=autopilot_refused,
        autopilot_claim_recorded=autopilot_claim_recorded,
        autopilot_claim_instance_job_id=autopilot_claim_instance_job_id,
        autopilot_claim_reason=autopilot_claim_reason,
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
        "lane_id": event.lane_id,
        "claim_keys": list(event.claim_keys),
        "target_modes": list(event.target_modes),
        "dispatch_packet_sha256": event.dispatch_packet_sha256,
        "archive_sha256": event.archive_sha256,
        "runtime_tree_sha256": event.runtime_tree_sha256,
        "ready_for_exact_eval_dispatch": event.ready_for_exact_eval_dispatch,
        "literature_anchor": event.literature_anchor,
        "source_supports": event.source_supports,
        "paper_claim_scope": event.paper_claim_scope,
        "pact_must_prove": event.pact_must_prove,
        "decode_complexity_evidence": event.decode_complexity_evidence,
        "predicted_score_delta": event.predicted_score_delta,
        "estimated_cost_usd": event.estimated_cost_usd,
        "halt_at_utc": event.halt_at_utc,
        "autopilot_authorized": event.autopilot_authorized,
        "autopilot_tag": event.autopilot_tag,
        "autopilot_authorized_reason": event.autopilot_authorized_reason,
        "autopilot_refused_reason": event.autopilot_refused_reason,
        "autopilot_claim_recorded": event.autopilot_claim_recorded,
        "autopilot_claim_instance_job_id": event.autopilot_claim_instance_job_id,
        "autopilot_claim_reason": event.autopilot_claim_reason,
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
    claim_keys: list[str] | tuple[str, ...] | None = None,
    claims_path: Path | None = None,
    now_utc: dt.datetime | None = None,
    ttl_hours: float = AUTOPILOT_CLAIM_TTL_HOURS,
) -> tuple[bool, str]:
    """Check the active-lane-dispatch-claims registry for a conflicting claim.

    Returns ``(has_conflict, reason)``. Returns ``(False, "")`` if no claim
    file exists yet (cold start) or no conflicting claim is found.

    The claims file is the markdown registry referenced in CLAUDE.md
    "CROSS-AGENT DISPATCH COORDINATION". This helper uses the canonical parsed
    claim-row semantics from :mod:`tools.claim_lane_dispatch`: exact
    ``lane_id`` matching only, latest row per ``(lane_id, instance/job_id)``,
    and terminal rows close older nonterminal rows for the same job.
    """
    p = claims_path or (
        REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    if not p.is_file():
        return False, ""
    keys: list[str] = []
    for key in [candidate_id, *(claim_keys or [])]:
        s = str(key or "").strip()
        if s and s not in keys:
            keys.append(s)
    if not keys:
        return False, ""
    try:
        text = p.read_text(encoding="utf-8")
        claims = _claim_lane_dispatch._parse_claims(text)
        latest_claims = _claim_lane_dispatch._latest_claims_by_job(claims)
    except Exception as exc:
        return True, (
            f"could not parse active-lane-dispatch-claims registry at {p} "
            f"with tools/claim_lane_dispatch.py ({type(exc).__name__}: {exc}); "
            "fail closed before dispatch"
        )
    now = now_utc or dt.datetime.now(dt.UTC)
    ttl = dt.timedelta(hours=ttl_hours)
    for claim in latest_claims.values():
        if claim.lane_id not in keys:
            continue
        if _claim_lane_dispatch._is_terminal(claim.status):
            continue
        stale = _claim_lane_dispatch._claim_is_stale_nonterminal(
            claim, now_utc=now, ttl=ttl
        )
        state = "stale nonterminal" if stale else "active"
        return True, (
            f"{state} dispatch claim for exact lane_id {claim.lane_id!r} "
            f"(candidate_id {candidate_id!r}, job {claim.instance_job_id!r}, "
            f"status {claim.status!r}) is present at {p}. "
            "Close or supersede it with tools/claim_lane_dispatch.py before dispatch."
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
    validate_authorized_mode_config(auth_mode, repo_root=REPO_ROOT)
    for candidate in candidates:
        _require_candidate_planning_cost(candidate)

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
        # smallest credible bolt-on: post-gate predicted_delta < 0 AND lowest cost
        candidates = sorted(
            [
                c for c in candidates
                if _candidate_has_effective_negative_delta_for_race_mode(
                    c,
                    continual_posterior=continual_posterior,
                )
            ],
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
    effective_claims_path = claims_path or (
        REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )

    for cand in ranked:
        if max_dispatch_recommendations is not None and n_ranked >= max_dispatch_recommendations:
            break
        # Cross-agent dispatch claim check
        has_conflict, reason = check_dispatch_claim_conflict(
            cand.candidate_id,
            claim_keys=cand.dispatch_claim_keys(),
            claims_path=effective_claims_path,
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
            claims_path=effective_claims_path,
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


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _require_finite_positive_float(
    value: object,
    *,
    field: str,
    context: str,
) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has non-numeric {field}={value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(
            f"{context} must carry finite positive {field}; got {value!r}"
        )
    return out


def _require_finite_nonnegative_float(
    value: object,
    *,
    field: str,
    context: str,
) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has non-numeric {field}={value!r}") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(
            f"{context} must carry finite nonnegative {field}; got {value!r}"
        )
    return out


def _coerce_str_list(value: object) -> list[str]:
    """Return a normalized string list for JSONL/JSON candidate metadata."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if s and s not in out:
            out.append(s)
    return out


def _json_bool_field(
    raw: dict[str, Any],
    key: str,
    *,
    default: bool = False,
    context: str,
) -> bool:
    """Read an authority-bearing JSON bool without Python truthiness coercion."""

    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{context} has non-boolean {key}={value!r}; expected JSON true/false"
    )


def _json_optional_bool_field(
    raw: dict[str, Any],
    key: str,
    *,
    context: str,
) -> bool | None:
    """Read an optional JSON bool while preserving explicit null as unknown."""

    if key not in raw or raw[key] is None:
        return None
    return _json_bool_field(raw, key, default=False, context=context)


def _require_planning_only_flag(raw: dict[str, Any], key: str, *, context: str) -> None:
    if _json_bool_field(raw, key, default=False, context=context):
        raise ValueError(
            f"{context} has {key}=True; refusing to consume it as planning-only "
            "autopilot input"
        )


def _audit_exact_ready_queue(
    path: Path,
    *,
    repo_root: Path,
    dispatch_claims_path: Path,
) -> dict[str, Any]:
    from tac.optimizer.exact_ready_audit import audit_exact_ready_queue

    return audit_exact_ready_queue(
        path,
        repo_root=repo_root,
        dispatch_claims_path=dispatch_claims_path,
    )


def _prediction_band_allows_rank_reward(raw: dict[str, Any]) -> bool:
    """Return whether a raw planning row may keep nonzero EIG rank reward."""

    verdict = raw.get("prediction_band_verdict")
    if isinstance(verdict, dict) and "valid_for_rank_reward" in verdict:
        return verdict.get("valid_for_rank_reward") is True
    notes = str(raw.get("notes", "")).lower()
    if raw.get("prediction_band") is not None:
        return False
    return not ("[prediction" in notes or "[predicted" in notes)


def _candidate_prediction_band_rank_reward_suppressed(c: CandidateRow) -> bool:
    """Return true when prediction-band custody already suppressed rank reward."""

    blockers = set(c.blockers or [])
    notes = str(c.notes or "")
    return (
        "prediction_band_rank_reward_suppressed" in blockers
        or "prediction_band_rank_reward_suppressed" in notes
    )


def _candidate_has_effective_negative_delta_for_race_mode(
    candidate: CandidateRow,
    *,
    continual_posterior: Any | None = None,
) -> bool:
    """Return true only for candidates with a post-gate negative prediction."""

    if _candidate_prediction_band_rank_reward_suppressed(candidate):
        return False
    delta = apply_z1_empirical_revision_to_candidate_delta(candidate)
    if continual_posterior is not None:
        factor, _, _ = _posterior_correction_factor(candidate, continual_posterior)
        delta *= factor
    return delta < 0.0


def _candidate_row_from_raw(
    raw: dict[str, Any],
    *,
    context: str,
    allow_dispatch_authority_flags: bool = False,
) -> CandidateRow:
    authority_flags = ["score_claim", "promotion_eligible"]
    if not allow_dispatch_authority_flags:
        authority_flags.append("ready_for_exact_eval_dispatch")
    for flag in authority_flags:
        _require_planning_only_flag(raw, flag, context=context)
    mdl_density = _coerce_optional_float(raw.get("mdl_density"))
    mdl_tier_c_density = _coerce_optional_float(raw.get("mdl_tier_c_density"))
    composition_alpha = _coerce_optional_float(raw.get("composition_alpha"))
    predicted_dispatch_risk = _coerce_optional_float(
        raw.get("predicted_dispatch_risk")
    )
    lane_class_raw = raw.get("lane_class")
    lane_class: str | None = (
        str(lane_class_raw) if lane_class_raw is not None else None
    )
    blockers = list(raw.get("blockers", []))
    notes = str(raw.get("notes", ""))
    expected_information_gain = float(raw["expected_information_gain"])
    if (
        expected_information_gain > 0.0
        and not _prediction_band_allows_rank_reward(raw)
    ):
        expected_information_gain = 0.0
        if "prediction_band_rank_reward_suppressed" not in blockers:
            blockers.append("prediction_band_rank_reward_suppressed")
        notes += "; prediction_band_rank_reward_suppressed"
    return CandidateRow(
        candidate_id=raw["candidate_id"],
        family=raw["family"],
        predicted_score_delta=float(raw["predicted_score_delta"]),
        expected_information_gain=expected_information_gain,
        estimated_dispatch_cost_usd=_require_finite_positive_float(
            raw["estimated_dispatch_cost_usd"],
            field="estimated_dispatch_cost_usd",
            context=context,
        ),
        blockers=blockers,
        notes=notes,
        timing_smoke_command=str(raw.get("timing_smoke_command", "")),
        mdl_density=mdl_density,
        lane_class=lane_class,
        literature_anchor=str(raw.get("literature_anchor", "")),
        source_supports=str(raw.get("source_supports", "")),
        paper_claim_scope=str(raw.get("paper_claim_scope", "")),
        pact_must_prove=str(raw.get("pact_must_prove", "")),
        decode_complexity_evidence=str(raw.get("decode_complexity_evidence", "")),
        mdl_tier_c_density=mdl_tier_c_density,
        composition_alpha=composition_alpha,
        license_ok=_json_bool_field(
            raw,
            "license_ok",
            default=True,
            context=context,
        ),
        inflate_dep_count=int(raw.get("inflate_dep_count", 0) or 0),
        sideinfo_consumed=_json_optional_bool_field(
            raw,
            "sideinfo_consumed",
            context=context,
        ),
        exact_duplicate=_json_bool_field(
            raw,
            "exact_duplicate",
            default=False,
            context=context,
        ),
        context_order=int(raw.get("context_order", 0) or 0),
        predicted_dispatch_risk=predicted_dispatch_risk,
        lane_id=str(raw.get("lane_id", "")),
        claim_keys=_coerce_str_list(raw.get("claim_keys")),
        target_modes=_coerce_str_list(raw.get("target_modes")),
        dispatch_packet_ready=_json_bool_field(
            raw,
            "dispatch_packet_ready",
            default=False,
            context=context,
        ),
        dispatch_packet_sha256=str(raw.get("dispatch_packet_sha256", "")),
        archive_sha256=str(raw.get("archive_sha256", "")),
        runtime_tree_sha256=str(raw.get("runtime_tree_sha256", "")),
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=(
            _json_bool_field(
                raw,
                "ready_for_exact_eval_dispatch",
                default=False,
                context=context,
            )
            if allow_dispatch_authority_flags
            else False
        ),
    )


def load_candidates_from_jsonl(path: Path) -> list[CandidateRow]:
    """Load planning-only CandidateRow objects from a JSONL file."""

    rows: list[CandidateRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            raw = json.loads(s)
            context = (
                f"{path}:{line_no} candidate "
                f"{raw.get('candidate_id', '<missing-candidate-id>')!r}"
            )
            rows.append(_candidate_row_from_raw(raw, context=context))
    return rows


def load_candidates_from_exact_ready_queue(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dispatch_claims_path: Path | None = None,
) -> list[CandidateRow]:
    """Load exact-ready rows only after the canonical live custody audit passes."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exact-ready queue must be a JSON object")
    if payload.get("schema") != EXACT_READY_QUEUE_SCHEMA:
        raise ValueError(
            "exact-ready queue schema unsupported:"
            f"{payload.get('schema')!r}; expected {EXACT_READY_QUEUE_SCHEMA!r}"
        )
    claims_path = dispatch_claims_path or (
        repo_root / ".omx" / "state" / "active_lane_dispatch_claims.md"
    )
    audit = _audit_exact_ready_queue(
        path,
        repo_root=repo_root,
        dispatch_claims_path=claims_path,
    )
    stale_rows = audit.get("stale_ready_rows")
    if isinstance(stale_rows, list) and stale_rows:
        first = stale_rows[0] if isinstance(stale_rows[0], dict) else {}
        blockers = first.get("blockers") if isinstance(first, dict) else None
        raise ValueError(
            "exact-ready queue failed live custody audit:"
            f"{first.get('candidate_id') if isinstance(first, dict) else '<unknown>'}:"
            f"{blockers}"
        )
    raw_rows = payload.get("dispatch_ready")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("exact-ready queue has no dispatch_ready rows")
    rows: list[CandidateRow] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"exact-ready queue row {index} is not an object")
        if raw.get("ready_for_exact_eval_dispatch") is not True:
            raise ValueError(
                "exact-ready queue dispatch_ready row lacks "
                f"ready_for_exact_eval_dispatch=True:{raw.get('candidate_id')!r}"
            )
        context = (
            f"{path}:dispatch_ready[{index}] candidate "
            f"{raw.get('candidate_id', '<missing-candidate-id>')!r}"
        )
        rows.append(
            _candidate_row_from_raw(
                raw,
                context=context,
                allow_dispatch_authority_flags=True,
            )
        )
    return rows


# ── Substrate composition matrix ranking integration ──────────────────────


SUBSTRATE_COMPOSITION_RANKING_SCHEMA = "tac_autopilot_dispatch_ranking_v1"


def _require_substrate_composition_ranking_schema(
    payload: dict[str, Any],
    *,
    context: str,
) -> None:
    schema = payload.get("schema")
    if schema != SUBSTRATE_COMPOSITION_RANKING_SCHEMA:
        raise ValueError(
            f"{context} schema unsupported: {schema!r}; expected "
            f"{SUBSTRATE_COMPOSITION_RANKING_SCHEMA!r}. Legacy ranking "
            "artifacts need an explicit legacy blocker path before autopilot load."
        )


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
    if not isinstance(payload, dict):
        raise ValueError("substrate composition ranking JSON must be an object")
    _require_substrate_composition_ranking_schema(
        payload,
        context="substrate composition ranking JSON",
    )
    if "ranked_dispatches" not in payload:
        raise ValueError(
            f"substrate composition ranking JSON missing 'ranked_dispatches' "
            f"key (schema={payload.get('schema')!r}); got top-level keys: "
            f"{sorted(payload.keys())}"
        )
    if _json_bool_field(
        payload,
        "score_claim",
        default=False,
        context="substrate composition ranking JSON",
    ):
        raise ValueError(
            "substrate composition ranking JSON has score_claim=True; "
            "the autopilot ranker must remain planning-only "
            "(per CLAUDE.md 'Forbidden score claims')"
        )
    rows: list[CandidateRow] = []
    for raw in payload["ranked_dispatches"]:
        context = (
            "substrate composition ranked dispatch "
            f"{raw.get('candidate_id')!r}"
        )
        if _json_bool_field(raw, "score_claim", default=False, context=context):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has score_claim=True; "
                "refuse to consume score-claimed planning rows"
            )
        if _json_bool_field(
            raw,
            "promotion_eligible",
            default=False,
            context=context,
        ):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has "
                "promotion_eligible=True; refuse to consume promotion-claimed "
                "planning rows"
            )
        if _json_bool_field(
            raw,
            "ready_for_exact_eval_dispatch",
            default=False,
            context=context,
        ):
            raise ValueError(
                f"ranked dispatch {raw.get('candidate_id')!r} has "
                "ready_for_exact_eval_dispatch=True; refuse to consume in autopilot "
                "(operator-gated promotion path required)"
            )
        fits_per_dispatch_cap = _json_bool_field(
            raw,
            "fits_per_dispatch_cap",
            default=True,
            context=context,
        )
        fits_cumulative_envelope = _json_bool_field(
            raw,
            "fits_cumulative_envelope",
            default=True,
            context=context,
        )
        if only_fits_per_dispatch_cap and not fits_per_dispatch_cap:
            continue
        if only_in_envelope and not fits_cumulative_envelope:
            continue
        notes_lines = [
            "[predicted; substrate composition matrix v1]",
            f"composition_notes: {raw.get('composition_notes', '')}",
            f"substrate_ids: {raw.get('substrate_ids', [])!r}",
        ]
        if raw.get("lane_class"):
            notes_lines.append(f"lane_class: {raw.get('lane_class')}")
        if raw.get("literature_anchor"):
            notes_lines.append(f"literature_anchor: {raw.get('literature_anchor')}")
        if raw.get("source_supports"):
            notes_lines.append(f"source_supports: {raw.get('source_supports')}")
        if raw.get("paper_claim_scope"):
            notes_lines.append(f"paper_claim_scope: {raw.get('paper_claim_scope')}")
        if raw.get("pact_must_prove"):
            notes_lines.append(f"pact_must_prove: {raw.get('pact_must_prove')}")
        if raw.get("decode_complexity_evidence"):
            notes_lines.append(
                f"decode_complexity_evidence: {raw.get('decode_complexity_evidence')}"
            )
        if raw.get("campaign_metadata"):
            notes_lines.append(f"campaign_metadata: {raw.get('campaign_metadata')!r}")
        lane_class_raw = raw.get("lane_class")
        row_blockers = list(raw.get("blockers", []))
        if PLANNING_ONLY_SOURCE_BLOCKER not in row_blockers:
            row_blockers.append(PLANNING_ONLY_SOURCE_BLOCKER)
        expected_information_gain = float(raw["expected_information_gain"])
        if (
            expected_information_gain > 0.0
            and not _prediction_band_allows_rank_reward(raw)
        ):
            expected_information_gain = 0.0
            if "prediction_band_rank_reward_suppressed" not in row_blockers:
                row_blockers.append("prediction_band_rank_reward_suppressed")
            notes_lines.append("prediction_band_rank_reward_suppressed")
        # Catalog #227 wire-in: read optional composition_alpha if present
        # (the canonical Z3xC6 probe-disambiguator emits this).
        composition_alpha_raw = raw.get("composition_alpha")
        composition_alpha: float | None = None
        if composition_alpha_raw is not None:
            try:
                composition_alpha = float(composition_alpha_raw)
            except (TypeError, ValueError):
                composition_alpha = None
        rows.append(CandidateRow(
            candidate_id=str(raw["candidate_id"]),
            family=str(raw["family"]),
            predicted_score_delta=float(raw["predicted_score_delta"]),
            expected_information_gain=expected_information_gain,
            estimated_dispatch_cost_usd=_require_finite_nonnegative_float(
                raw["estimated_dispatch_cost_usd"],
                field="estimated_dispatch_cost_usd",
                context=context,
            ),
            blockers=row_blockers,
            notes="\n".join(notes_lines),
            mdl_density=_coerce_optional_float(raw.get("mdl_density")),
            lane_class=str(lane_class_raw) if lane_class_raw is not None else None,
            literature_anchor=str(raw.get("literature_anchor", "")),
            source_supports=str(raw.get("source_supports", "")),
            paper_claim_scope=str(raw.get("paper_claim_scope", "")),
            pact_must_prove=str(raw.get("pact_must_prove", "")),
            decode_complexity_evidence=str(
                raw.get("decode_complexity_evidence", "")
            ),
            composition_alpha=composition_alpha,
            license_ok=_json_bool_field(raw, "license_ok", default=True, context=context),
            inflate_dep_count=int(raw.get("inflate_dep_count", 0) or 0),
            sideinfo_consumed=_json_optional_bool_field(
                raw,
                "sideinfo_consumed",
                context=context,
            ),
            exact_duplicate=_json_bool_field(
                raw,
                "exact_duplicate",
                default=False,
                context=context,
            ),
            context_order=int(raw.get("context_order", 0) or 0),
            score_claim=False,
            promotion_eligible=False,
            ready_for_exact_eval_dispatch=False,
        ))
    return rows


# ── Canonical substrate composition matrix consumer (Catalog #227) ────────
#
# Per `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md` the
# canonical posterior surface for substrate composition is
# `.omx/state/substrate_composition_matrix.json`. The probe-disambiguator
# appends one entry per probe invocation; multiple entries per pair are
# possible (most-recent wins).
#
# This loader maps the matrix into a `(substrate_pair_key, alpha)` dict so
# the autopilot can look up the composition factor for any candidate that
# carries substrate_ids. Substrate pair key is the canonical
# "<substrate_id_a>__x__<substrate_id_b>" form used by T1-F.
#
# Per CLAUDE.md "Forbidden score claims" the matrix entries are PREDICTED
# composition signals, never measured scores; the loader refuses any entry
# with score_claim=True.
SUBSTRATE_COMPOSITION_MATRIX_PATH = (
    REPO_ROOT / ".omx" / "state" / "substrate_composition_matrix.json"
)


def load_substrate_composition_alpha_index(
    path: Path | None = None,
) -> dict[str, float]:
    """Return {"<substrate_a>__x__<substrate_b>": alpha} from the canonical
    substrate composition matrix posterior surface.

    For each pair key in the matrix, the loader returns the alpha of the
    MOST RECENTLY WRITTEN entry (by ``written_at_utc``). Per CLAUDE.md
    "Forbidden score claims" any entry with ``score_claim=True`` is REFUSED
    (the loader raises ``ValueError``).

    Returns an empty dict when the matrix file is absent.
    """
    p = path if path is not None else SUBSTRATE_COMPOSITION_MATRIX_PATH
    if not p.is_file():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"substrate composition matrix exists but is unreadable or invalid JSON: {p}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"substrate composition matrix must be a JSON object: {p}")
    if "entries" not in payload:
        raise ValueError(f"substrate composition matrix missing 'entries': {p}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(f"substrate composition matrix 'entries' must be an object: {p}")
    out: dict[str, float] = {}
    for pair_key, rows in entries.items():
        if not isinstance(rows, list):
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                "must be a list"
            )
        if not rows:
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(
                    f"substrate composition matrix entry for pair {pair_key!r} "
                    f"row {i} must be an object"
                )
        # Pick most-recent by written_at_utc (string lexicographic compare
        # works because all timestamps use canonical UTC ISO format).
        latest = max(
            rows,
            key=lambda r: r.get("written_at_utc", ""),
            default=None,
        )
        if latest is None:
            continue
        context = f"substrate composition matrix entry for pair {pair_key!r}"
        if _json_bool_field(latest, "score_claim", default=False, context=context):
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                "has score_claim=True; refuse to consume score-claimed "
                "composition rows (per CLAUDE.md 'Forbidden score claims')"
            )
        alpha_raw = latest.get("alpha")
        if alpha_raw is None:
            continue
        try:
            alpha = float(alpha_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                f"has non-numeric alpha={alpha_raw!r}"
            ) from exc
        if not math.isfinite(alpha):
            raise ValueError(
                f"substrate composition matrix entry for pair {pair_key!r} "
                f"has non-finite alpha={alpha_raw!r}"
            )
        out[str(pair_key)] = alpha
    return out


def _substrate_pair_ids_from_alpha_key(pair_key: str) -> tuple[str, str] | None:
    parts = pair_key.split("__x__")
    if len(parts) != 2:
        return None
    left, right = (p.strip() for p in parts)
    if not left or not right:
        return None
    return left, right


def apply_substrate_composition_matrix_to_candidates(
    candidates: list[CandidateRow],
    *,
    substrate_ids_by_candidate: dict[str, tuple[str, ...]] | None = None,
    matrix_path: Path | None = None,
) -> list[CandidateRow]:
    """Populate ``candidate.composition_alpha`` from the canonical
    substrate composition matrix.

    For each candidate, if ``substrate_ids_by_candidate`` carries a tuple of
    exactly TWO substrate ids, the canonical key
    ``<a>__x__<b>`` (alphabetically sorted) OR ``<b>__x__<a>`` is looked up
    in the matrix. Single-substrate candidates and substrate triples are not
    matched (composition_alpha stays None).

    Returns the SAME list (mutated in place) — convenient for chaining.
    """
    if not candidates:
        return candidates
    alpha_index = load_substrate_composition_alpha_index(matrix_path)
    if not alpha_index:
        return candidates
    if substrate_ids_by_candidate is None:
        substrate_ids_by_candidate = {}
    for cand in candidates:
        sids = substrate_ids_by_candidate.get(cand.candidate_id)
        if not sids or len(sids) != 2:
            continue
        a, b = sorted(sids)
        key_a = f"{a}__x__{b}"
        key_b = f"{b}__x__{a}"
        alpha = alpha_index.get(key_a)
        if alpha is None:
            alpha = alpha_index.get(key_b)
        if alpha is None:
            # Fallback: scan exact pair keys (the probe may emit keys in
            # non-sorted form). Substring checks can cross-match unrelated ids.
            for k, v in alpha_index.items():
                pair = _substrate_pair_ids_from_alpha_key(k)
                if pair is not None and set(pair) == {a, b}:
                    alpha = v
                    break
        if alpha is not None:
            cand.composition_alpha = alpha
    return candidates


# ── Probe-disambiguator read-only autopilot-row consumer ──────────────────


def load_candidates_from_probe_disambiguator_output(path: Path) -> list[CandidateRow]:
    """Load read-only ``autopilot_rows`` from a probe-disambiguator JSON artifact.

    Probe-disambiguators arbitrate design interpretations; they are NOT
    dispatch packets or score evidence. This consumer intentionally accepts
    only rows whose top-level and per-row safety flags remain fail-closed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"probe-disambiguator JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"probe-disambiguator JSON must be an object: {path}")
    for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        _require_planning_only_flag(payload, key, context="probe-disambiguator JSON")
    if payload.get("dispatch_attempted", False):
        raise ValueError(
            "probe-disambiguator JSON has dispatch_attempted=True; refusing "
            "to consume it as an autopilot planning source"
        )
    raw_rows = payload.get("autopilot_rows")
    if not isinstance(raw_rows, list):
        raise ValueError(
            "probe-disambiguator JSON missing list field 'autopilot_rows'"
        )

    rows: list[CandidateRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("probe-disambiguator autopilot_rows entries must be objects")
        cid = raw.get("candidate_id")
        if not cid:
            raise ValueError("probe-disambiguator row missing candidate_id")
        for key in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
            _require_planning_only_flag(
                raw, key, context=f"probe-disambiguator row {cid!r}"
            )
        if raw.get("dispatch_attempted", False):
            raise ValueError(
                f"probe-disambiguator row {cid!r} has dispatch_attempted=True; "
                "refusing to consume it as an autopilot planning row"
            )
        lane_class_raw = raw.get("lane_class")
        row_blockers = list(raw.get("blockers", []))
        if PLANNING_ONLY_SOURCE_BLOCKER not in row_blockers:
            row_blockers.append(PLANNING_ONLY_SOURCE_BLOCKER)
        expected_information_gain = float(raw.get("expected_information_gain", 0.0))
        if (
            expected_information_gain > 0.0
            and not _prediction_band_allows_rank_reward(raw)
        ):
            expected_information_gain = 0.0
            if "prediction_band_rank_reward_suppressed" not in row_blockers:
                row_blockers.append("prediction_band_rank_reward_suppressed")
            raw_notes = str(raw.get("notes", ""))
            raw = {
                **raw,
                "notes": (
                    f"{raw_notes}; prediction_band_rank_reward_suppressed"
                    if raw_notes
                    else "prediction_band_rank_reward_suppressed"
                ),
            }
        notes = "\n".join(
            [
                "[probe-disambiguator; read-only planning]",
                f"source_path: {path}",
                f"source_schema: {payload.get('schema')!r}",
                f"source_tool: {payload.get('tool')!r}",
                f"row_notes: {raw.get('notes', '')}",
            ]
        )
        rows.append(
            CandidateRow(
                candidate_id=str(cid),
                family=str(raw.get("family", "probe_disambiguator")),
                predicted_score_delta=float(raw.get("predicted_score_delta", 0.0)),
                expected_information_gain=expected_information_gain,
                estimated_dispatch_cost_usd=_require_finite_nonnegative_float(
                    raw.get("estimated_dispatch_cost_usd", 0.0),
                    field="estimated_dispatch_cost_usd",
                    context=f"probe-disambiguator row {cid!r}",
                ),
                blockers=row_blockers,
                notes=notes,
                mdl_density=_coerce_optional_float(raw.get("mdl_density")),
                lane_class=str(lane_class_raw) if lane_class_raw is not None else None,
                literature_anchor=str(raw.get("literature_anchor", "")),
                source_supports=str(raw.get("source_supports", "")),
                paper_claim_scope=str(raw.get("paper_claim_scope", "")),
                pact_must_prove=str(raw.get("pact_must_prove", "")),
                decode_complexity_evidence=str(
                    raw.get("decode_complexity_evidence", "")
                ),
                mdl_tier_c_density=_coerce_optional_float(
                    raw.get("mdl_tier_c_density")
                ),
                composition_alpha=_coerce_optional_float(
                    raw.get("composition_alpha")
                ),
                license_ok=_json_bool_field(
                    raw,
                    "license_ok",
                    default=True,
                    context=f"probe-disambiguator row {cid!r}",
                ),
                inflate_dep_count=int(raw.get("inflate_dep_count", 0) or 0),
                sideinfo_consumed=_json_optional_bool_field(
                    raw,
                    "sideinfo_consumed",
                    context=f"probe-disambiguator row {cid!r}",
                ),
                exact_duplicate=_json_bool_field(
                    raw,
                    "exact_duplicate",
                    default=False,
                    context=f"probe-disambiguator row {cid!r}",
                ),
                context_order=int(raw.get("context_order", 0) or 0),
            )
        )
    return rows


def candidate_substrate_ids_from_ranking(path: Path) -> dict[str, tuple[str, ...]]:
    """Map ``candidate_id`` -> tuple of substrate ids participating in the dispatch.

    Used by the composition-constraint enforcer to reason about which
    substrates a candidate touches without re-parsing the full ranking JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("substrate composition ranking JSON must be an object")
    _require_substrate_composition_ranking_schema(
        payload,
        context="substrate composition ranking JSON",
    )
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


# ── macOS-CPU advisory proxy ranking integration ────────────────────────


MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG = "macos_cpu_advisory"


def load_candidates_from_macos_cpu_advisory_manifest(
    path: Path,
    *,
    default_estimated_dispatch_cost_usd: float = 0.0,
    default_expected_information_gain: float = 0.0,
) -> list[CandidateRow]:
    """Load CandidateRow rows from a macOS-CPU advisory-signal manifest.

    Operator routing 2026-05-13 ("training is the real roadblock; we can
    prepare and run things on macos and cpu"). Per CLAUDE.md PR107
    empirical calibration (|Δ| ≤ 6e-6 vs GHA Linux x86_64 contest-CPU on
    the same exact archive), macOS-CPU is a free first-class advisory
    proxy that lets the autopilot RANK candidates BEFORE any GPU spend.

    The loaded rows participate in EIG-per-dollar ranking BUT carry:
      - notes prefixed with ``[macOS-CPU advisory; ranking-only]``
      - blockers extended with the manifest's ``dispatch_blockers`` so
        the operator sees every reason a row cannot promote
      - ``predicted_score_delta`` derived from ``projected_contest_cpu_score_p50``
        when present, else from ``score_macos_cpu`` directly.

    Per CLAUDE.md Catalog #127 (`check_authoritative_tag_requires_custody_metadata`)
    the manifest's evidence_tag ``[macOS-CPU advisory only]`` is already
    routed to ``refused_class="macos_substrate"`` by the custody validator.
    Promotion requires a paired ``[contest-CPU GHA Linux x86_64]`` anchor;
    the loader DOES NOT lift that gate.
    """
    if not path.is_file():
        raise FileNotFoundError(f"macOS-CPU advisory manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    expected_schema_prefix = "macos_cpu_advisory_signal_manifest"
    schema = str(payload.get("schema") or "")
    if not schema.startswith(expected_schema_prefix):
        raise ValueError(
            f"macOS-CPU advisory manifest at {path!s} has unexpected schema "
            f"{schema!r}; expected schema starting with {expected_schema_prefix!r}"
        )

    # Per CLAUDE.md "Forbidden score claims" + Catalog #192: refuse manifests
    # claiming promotability. The autopilot ranker never lifts these gates.
    for flag in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        if payload.get(flag, False):
            raise ValueError(
                f"macOS-CPU advisory manifest has {flag}=True; refusing to "
                "consume in autopilot ranker (per CLAUDE.md Catalog #192 + "
                "'Forbidden score claims')"
            )

    base_blockers = list(payload.get("dispatch_blockers", []))
    rows: list[CandidateRow] = []
    for raw in payload.get("rows", []):
        if raw.get("score_claim", False):
            raise ValueError(
                f"manifest row {raw.get('variant_id')!r} has score_claim=True; "
                "refuse to consume"
            )
        if raw.get("promotion_eligible", False) or raw.get(
            "ready_for_exact_eval_dispatch", False
        ):
            raise ValueError(
                f"manifest row {raw.get('variant_id')!r} has "
                "promotion_eligible or ready_for_exact_eval_dispatch True; refuse"
            )
        # Predicted score delta: prefer the projected contest-CPU score
        # band's p50 anchor (calibrated). Otherwise use score_macos_cpu
        # directly. Either way the row's notes record that the prediction
        # is non-authoritative.
        projected_p50 = raw.get("projected_contest_cpu_score_p50")
        score_macos_cpu = raw.get("score_macos_cpu")
        if projected_p50 is not None:
            predicted_score = float(projected_p50)
        elif score_macos_cpu is not None:
            predicted_score = float(score_macos_cpu)
        else:
            # No score → can't rank by score; skip with a notes record.
            continue

        family = str(raw.get("family") or "")
        variant_id = str(raw.get("variant_id") or "")
        if not family or not variant_id:
            continue

        # Treat the projected score as the predicted_score_delta absolute
        # value. Most-negative-is-best ranking still works since smaller
        # contest scores are better.
        row_blockers = list(base_blockers) + list(raw.get("dispatch_blockers", []))
        # Dedup blockers preserving insertion order.
        seen: set[str] = set()
        deduped_blockers: list[str] = []
        for b in row_blockers:
            if b in seen:
                continue
            seen.add(b)
            deduped_blockers.append(b)

        archive_sha = str(raw.get("archive_sha256") or "")
        archive_bytes = int(raw.get("archive_bytes") or 0)
        band_low = raw.get("projected_contest_cpu_score_low")
        band_high = raw.get("projected_contest_cpu_score_high")
        notes_lines = [
            "[macOS-CPU advisory; ranking-only]",
            f"proxy_evidence: {MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG}",
            f"hardware_substrate: {raw.get('hardware_substrate') or payload.get('hardware_substrate')!r}",
            f"score_macos_cpu: {score_macos_cpu!r}",
            f"projected_contest_cpu_score: p50={projected_p50!r} "
            f"low={band_low!r} high={band_high!r}",
            f"archive_bytes: {archive_bytes}",
            f"archive_sha256: {archive_sha or '(missing)'}",
            "promotion_blocked: requires paired [contest-CPU GHA Linux x86_64] anchor",
        ]
        rows.append(
            CandidateRow(
                candidate_id=f"macos_cpu_advisory__{family}__{variant_id}",
                family=family,
                predicted_score_delta=predicted_score,
                expected_information_gain=default_expected_information_gain,
                estimated_dispatch_cost_usd=default_estimated_dispatch_cost_usd,
                blockers=deduped_blockers,
                notes="\n".join(notes_lines),
            )
        )
    return rows


def tag_halt_events_with_proxy_evidence(
    halt_events: list[HaltEvent],
    *,
    candidates: list[CandidateRow],
) -> list[HaltEvent]:
    """Annotate halt events whose source candidate carries macOS-CPU advisory notes.

    Per operator routing 2026-05-13: when a halt event's underlying candidate
    came from the macOS-CPU advisory manifest, surface that fact in the halt
    event's decision_notes so the operator can see at a glance which
    rankings depend on the proxy. The autopilot's dispatch journal will
    therefore tag the entry with ``proxy_evidence="macos_cpu_advisory"``.

    Mutates the halt events in place AND returns them for chaining.
    """
    cid_to_proxy: dict[str, str] = {}
    for c in candidates:
        if "[macOS-CPU advisory; ranking-only]" in (c.notes or ""):
            cid_to_proxy[c.candidate_id] = MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG
    for evt in halt_events:
        tag = cid_to_proxy.get(evt.candidate_id)
        if tag is None:
            continue
        marker = f"proxy_evidence={tag}"
        if marker not in evt.decision_notes:
            sep = "; " if evt.decision_notes else ""
            evt.decision_notes = f"{evt.decision_notes}{sep}{marker}"
    return halt_events


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


# ─────────────────────────────────────────────────────────────────────────
# Rudin-Daubechies autopilot ranker integration (opt-in, 2026-05-15)
# ─────────────────────────────────────────────────────────────────────────
#
# Per `feedback_rudin_daubechies_recommendations_for_completing_cathedral_autopilot_nervous_system_20260515.md`
# the canonical ranker stack lives in `tac.autopilot_rudin_daubechies`.
# This integration helper consumes the package's :class:`SLIMRanker` /
# :class:`RashomonEnsembleRanker` to enrich each :class:`CandidateRow` with
# a Rudin-interpretable rule chain explanation BEFORE the autopilot's
# canonical Z1 empirical-revision chain runs.
#
# The helper is OPT-IN: the autopilot's existing rank_candidates path is
# unchanged. Operators / agent partners who want the interpretable
# pre-dispatch ranking surface call ``rerank_candidates_via_rudin_daubechies``
# explicitly.

DEFAULT_RUDIN_DAUBECHIES_SLIM_STORE = Path(".omx/state/rudin_daubechies_slim_anchors.jsonl")


def rerank_candidates_via_rudin_daubechies(
    candidates: list[CandidateRow],
    *,
    slim_store_path: Path | None = None,
    use_rashomon_ensemble: bool = False,
    rashomon_ensemble_size: int = 8,
) -> list[tuple[CandidateRow, float, str]]:
    """Rerank candidates through the Rudin-Daubechies SLIM ranker.

    Returns a list of ``(candidate, predicted_score, explanation)`` tuples
    sorted ascending by predicted_score (lower predicted score = better
    candidate). The explanation is the Rudin rule-chain readback per
    :func:`tac.autopilot_rudin_daubechies.explain_slim_prediction`.

    When ``use_rashomon_ensemble=True`` the K=8 ensemble's consensus
    prediction is used and a disagreement std-dev is appended to the
    explanation as the operator-facing ideation signal.

    Per CLAUDE.md "Subagent coherence-by-default" wire-in hook 4 (cathedral
    autopilot dispatch hook): the helper is the canonical operator-facing
    transparency layer; calling it produces an auditable ranking decision.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the predicted
    score carries the SLIMRanker's ``confidence_tag()`` in the explanation
    so the operator distinguishes first-principles bounds from N-anchor
    posteriors.
    """
    # Lazy import to avoid hard dep at module import time.
    from tac.autopilot_rudin_daubechies import (
        ProxyPanel,
        RashomonEnsembleRanker,
        SLIMRanker,
        explain_slim_prediction,
    )

    store = slim_store_path or DEFAULT_RUDIN_DAUBECHIES_SLIM_STORE
    if use_rashomon_ensemble:
        ranker = RashomonEnsembleRanker(
            ensemble_size=rashomon_ensemble_size,
            store_path=store,
        )
    else:
        ranker = SLIMRanker(store_path=store)

    out: list[tuple[CandidateRow, float, str]] = []
    for c in candidates:
        # Map the CandidateRow to a minimal ProxyPanel. Per the Taylor
        # decomposition memo this is the integration point for the
        # forthcoming `tac.autopilot_proxies` package; for now we use the
        # already-available signals on CandidateRow.
        panel = ProxyPanel(
            candidate_id=c.candidate_id,
            panel_axis="macos_cpu_advisory",
        )
        if use_rashomon_ensemble:
            consensus, disagreement = ranker.predict_with_disagreement(panel)
            pred = consensus
            tag = ranker.confidence_tag()
            expl = (
                f"{tag} consensus={consensus:g} disagreement_stddev={disagreement:g}"
            )
        else:
            pred = ranker.predict(panel)
            expl = explain_slim_prediction(ranker, panel)
        out.append((c, pred, expl))
    out.sort(key=lambda t: t[1])
    return out


def update_rudin_daubechies_from_dispatch_outcome(
    candidate: CandidateRow,
    observed_score: float,
    *,
    axis: str = "contest_cuda",
    slim_store_path: Path | None = None,
) -> None:
    """Closes the continual-learning loop: dispatch outcome -> SLIM update.

    Per operator directive 2026-05-15: every empirical anchor flows through
    this helper so the SLIM ranker's coefficients refit and the next
    candidate evaluation is materially smarter.

    The helper is fcntl-locked per Catalog #128/#131 sister discipline; safe
    to call from concurrent harvesters.

    Per CLAUDE.md "Apples-to-apples evidence discipline" the caller MUST
    pass the correct ``axis`` (``contest_cuda`` / ``contest_cpu`` /
    ``macos_cpu_advisory``); the helper does not infer it from context.
    """
    from tac.autopilot_rudin_daubechies import ProxyPanel, SLIMRanker

    store = slim_store_path or DEFAULT_RUDIN_DAUBECHIES_SLIM_STORE
    ranker = SLIMRanker(store_path=store)
    panel = ProxyPanel(candidate_id=candidate.candidate_id, panel_axis=axis)
    ranker.update_from_anchor(observed_score, panel, axis=axis)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates-jsonl", type=Path, default=None,
                        help="Path to JSONL file of CandidateRow rows. Mutually "
                             "exclusive with the other candidate-source flags.")
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
            "queue. Mutually exclusive with the other candidate-source flags."
        ),
    )
    parser.add_argument(
        "--probe-disambiguator-json",
        type=Path,
        default=None,
        help=(
            "Path to a probe-disambiguator JSON artifact carrying read-only "
            "autopilot_rows. Mutually exclusive with the other candidate-source "
            "flags; rows must be planning-only and fail closed on score/promotion/"
            "exact-eval dispatch flags."
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
        # Exactly one candidate source must be supplied. They are mutually exclusive.
        sources_supplied = sum(
            1 for x in (
                args.candidates_jsonl,
                args.use_substrate_composition_matrix_ranking,
                args.probe_disambiguator_json,
            ) if x is not None
        )
        if sources_supplied != 1:
            raise ValueError(
                "exactly one of --candidates-jsonl, "
                "--use-substrate-composition-matrix-ranking, or "
                "--probe-disambiguator-json must be supplied "
                f"(got {sources_supplied})"
            )
        if args.candidates_jsonl is not None and not args.candidates_jsonl.is_file():
            raise FileNotFoundError(args.candidates_jsonl)
        if (
            args.use_substrate_composition_matrix_ranking is not None
            and not args.use_substrate_composition_matrix_ranking.is_file()
        ):
            raise FileNotFoundError(args.use_substrate_composition_matrix_ranking)
        if (
            args.probe_disambiguator_json is not None
            and not args.probe_disambiguator_json.is_file()
        ):
            raise FileNotFoundError(args.probe_disambiguator_json)
        approval_set = _parse_approval_flags(
            args.require_operator_approval_on or ["dispatch"]
        )
        # Refuse to activate authorized mode without a journal path.
        if args.operator_authorized_le_5_dollar_mode and args.journal_path is None:
            raise ValueError(
                "--operator-authorized-le-5-dollar-mode requires --journal-path "
                "for the structured-row JSONL ledger (per CLAUDE.md no-/tmp-path)"
            )
        if args.operator_authorized_le_5_dollar_mode and args.journal_path is not None:
            _require_finite_positive_float(
                args.per_dispatch_cap_usd,
                field="--per-dispatch-cap-usd",
                context="operator-authorized CLI",
            )
            _require_finite_positive_float(
                args.cumulative_cap_usd,
                field="--cumulative-cap-usd",
                context="operator-authorized CLI",
            )
            validate_authorized_journal_path(args.journal_path, repo_root=REPO_ROOT)
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
    elif args.probe_disambiguator_json is not None:
        def _source() -> list[CandidateRow]:
            return load_candidates_from_probe_disambiguator_output(
                args.probe_disambiguator_json
            )
    else:
        def _source() -> list[CandidateRow]:
            if (
                args.operator_authorized_le_5_dollar_mode
                and _env_authorizes_mode()
            ):
                try:
                    payload = json.loads(args.candidates_jsonl.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and payload.get("schema") == EXACT_READY_QUEUE_SCHEMA:
                    return load_candidates_from_exact_ready_queue(
                        args.candidates_jsonl,
                        dispatch_claims_path=args.claims_path,
                    )
                if isinstance(payload, dict):
                    raise ValueError(
                        "operator-authorized exact dispatch requires an "
                        f"{EXACT_READY_QUEUE_SCHEMA!r} queue, not schema "
                        f"{payload.get('schema')!r}"
                    )
            return load_candidates_from_jsonl(args.candidates_jsonl)

    try:
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
    except (ValueError, FileNotFoundError) as exc:
        print(f"cathedral_autopilot_autonomous_loop: {exc}", file=sys.stderr)
        return 2

    if args.use_substrate_composition_matrix_ranking is not None:
        source_tag = "substrate_composition_matrix_constraints_enforced"
    elif args.probe_disambiguator_json is not None:
        source_tag = "probe_disambiguator_read_only_source"
    else:
        source_tag = "candidates_jsonl_source"

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
            source_tag,
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
        "probe_disambiguator_source": (
            {
                "path": str(args.probe_disambiguator_json),
                "read_only_consumer": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            if args.probe_disambiguator_json is not None
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
