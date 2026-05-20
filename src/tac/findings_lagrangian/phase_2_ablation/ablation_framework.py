# SPDX-License-Identifier: MIT
"""tac.findings_lagrangian.phase_2_ablation.ablation_framework — canonical helper.

Per WAVE-3-DIM-1-PHASE-2-START (operator blanket approval 2026-05-20).

Build a paired-comparison ablation apparatus that replaces hand-derived
``adjust_predicted_delta_for_*`` adjusters with solver-derived dual variables
from ``tac.findings_lagrangian``. Phase 2 START covers 3 highest-leverage
adjusters; sister subagents extend to the remaining 7.

The 3 selected adjusters (HIGH EV x LOW-MEDIUM difficulty):
  1. mdl_density (Tier A MDL density; 3-band cascade)
  2. predicted_dispatch_risk (SLIM preflight; 3-band cascade with refusal floor)
  3. composition_alpha_v2 (composition additivity; 4-band cascade)

For each adjuster, this module implements a closed-form solver-derived equivalent
that:
  1. Takes the candidate's signal value (mdl_density / predicted_dispatch_risk /
     composition_alpha) and observation noise sigma.
  2. Constructs a 1-dim Gaussian posterior over the underlying signal via
     ``tac.findings_lagrangian.posterior_update_from_anchors`` keyed on a
     canonical equation_id per adjuster.
  3. Maps the posterior mean to an adjustment factor via the canonical
     band-boundary semantics (within-class / trending / across-class /
     additive / sub-additive / saturating).
  4. Returns the dual-variable-derived adjusted delta for paired comparison
     against the hand-derived adjuster.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
this is OBSERVABILITY-ONLY for at least 30 days of paired-comparison data.
Promotion to ``solver_derived`` default requires (a) >= 30 days of paired
comparison; (b) max regression vs hand-derived < $1 estimated EV; (c) sister
T3 council deliberation per Catalog #300.
"""
from __future__ import annotations

import datetime
import enum
import fcntl
import json
import os
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from tac.findings_lagrangian.posterior import (
    GaussianPosterior,
    PosteriorInvalidError,
    posterior_update_from_anchors,
)


# ----------------------------------------------------------------------------
# Canonical paths + constants
# ----------------------------------------------------------------------------

PHASE_2_ABLATION_POSTERIOR_PATH: Final[Path] = Path(
    ".omx/state/phase_2_ablation_posterior.jsonl"
)
"""Canonical fcntl-locked JSONL posterior for paired-comparison rows.

Sister of ``.omx/state/modal_call_id_ledger.jsonl`` (Catalog #245) +
``.omx/state/probe_outcomes.jsonl`` (Catalog #313) +
``.omx/state/canonical_equations_registry.jsonl`` (Catalog #344).
"""

PHASE_2_ABLATION_POSTERIOR_LOCK_PATH: Final[Path] = Path(
    ".omx/state/phase_2_ablation_posterior.lock"
)
"""fcntl lock file co-located with the posterior."""

PHASE_2_ABLATION_SCHEMA: Final[str] = "phase_2_ablation_posterior_v1_20260520"
"""Canonical schema id for paired-comparison rows."""


SUPPORTED_ADJUSTERS: Final[tuple[str, ...]] = (
    "mdl_density",
    "predicted_dispatch_risk",
    "composition_alpha_v2",
)
"""The 3 highest-leverage adjusters covered by Phase 2 START.

Sister subagents (Wave-4/5/6) extend to the remaining 7:
  - class_shift
  - mdl_tier_c_density
  - venn_classification_v2
  - per_pair_sister_817_sidecars
  - per_pair_difficulty_atlas
  - cable_d_consumers_7_14_sidecars
  - realistic_stacking_correction
"""


# Phase 2 START canonical equation_ids (one per adjuster). These are placeholder
# IDs scoped to the ablation surface; sister Catalog #344 canonical equation
# entries are queued for Phase 2 completion (each adjuster gets its own
# CanonicalEquation row codifying the band-boundary semantics).
_CANONICAL_EQUATION_IDS: Final[dict[str, str]] = {
    "mdl_density": "phase_2_ablation_mdl_density_v1",
    "predicted_dispatch_risk": "phase_2_ablation_predicted_dispatch_risk_v1",
    "composition_alpha_v2": "phase_2_ablation_composition_alpha_v2_v1",
}


# Promotion thresholds per the operator-routable Phase 2-N decision (recorded
# in landing memo). Hand-derived stays authoritative until BOTH conditions:
PROMOTION_THRESHOLD_MIN_ANCHORS: Final[int] = 30
"""Minimum paired-comparison anchors required before flipping default to
``solver_derived``. 30 days at 1 per day OR 30 distinct candidates.
"""

PROMOTION_THRESHOLD_MAX_REGRESSION_USD: Final[float] = 1.0
"""Maximum dollar-EV regression allowed for solver-derived vs hand-derived
before flipping default. Conservative: $1 covers typical smoke-dispatch costs.
"""

DEFAULT_DIVERGENCE_SIGMA_BOUND: Final[float] = 2.0
"""Default 2-sigma bound for divergence verdicts (within-tolerance vs
material-divergence). Mirrors the master memo's
'solver-derived matches within ±2σ' acceptance criterion.
"""


# ----------------------------------------------------------------------------
# Mode enum
# ----------------------------------------------------------------------------

class AblationMode(str, enum.Enum):
    """Phase 2 ablation runtime mode.

    Per the master design memo Dim 1 Phase 2: the cathedral autopilot
    ranker uses EXACTLY ONE of these per iteration; paired_comparison fires
    BOTH adjusters and emits divergence observability while still using
    HAND_DERIVED as the authoritative ranker.
    """

    HAND_DERIVED = "hand_derived"
    """Pre-Phase-2 baseline. Ranker uses ONLY the hand-derived adjuster.

    The solver-derived adjuster is not computed in this mode (saves cycles).
    Useful as a fallback if the solver causes performance regression.
    """

    SOLVER_DERIVED = "solver_derived"
    """Promotion-target mode. Ranker uses ONLY the solver-derived adjuster.

    Only safe to flip the default after PROMOTION_THRESHOLD_MIN_ANCHORS days
    of paired_comparison data AND max regression vs hand_derived <
    PROMOTION_THRESHOLD_MAX_REGRESSION_USD.
    """

    PAIRED_COMPARISON = "paired_comparison"
    """Default safety-rail mode. Both adjusters fire; ranker uses HAND_DERIVED
    as authoritative; divergence emitted to the posterior.

    This is the safety-rail default per the operator's standing "no fake
    reward" + "apples-to-apples evidence discipline" directives. The solver
    is being measured against the hand-derived baseline; promotion to
    solver_derived requires the operator-routable threshold criterion.
    """


DEFAULT_ABLATION_MODE: Final[AblationMode] = AblationMode.PAIRED_COMPARISON
"""Default mode. Safety-rail: hand-derived is authoritative; solver is
measured.
"""


# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------

class AblationError(Exception):
    """Raised when an ablation invariant is violated."""


# ----------------------------------------------------------------------------
# Provenance helpers (Catalog #323 sister discipline)
# ----------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _reject_placeholder_rationale(rationale: str | None, field_name: str) -> str:
    """Reject placeholder rationales per Catalog #287 sister discipline.

    Empty / None / placeholder literals (``<rationale>`` / ``<reason>``) are
    refused so the gate's docstring example cannot self-waive.
    """
    if rationale is None:
        return ""
    if not isinstance(rationale, str):
        raise AblationError(
            f"{field_name} must be str, got {type(rationale).__name__}"
        )
    stripped = rationale.strip()
    if not stripped:
        return ""
    placeholder_literals = {
        "<rationale>",
        "<reason>",
        "<rationale_here>",
        "<reason_here>",
    }
    if stripped.lower() in {p.lower() for p in placeholder_literals}:
        raise AblationError(
            f"{field_name} placeholder literal {stripped!r} rejected per "
            "Catalog #287 sister discipline"
        )
    if len(stripped) < 4:
        raise AblationError(
            f"{field_name} rationale must be >= 4 chars; got "
            f"{stripped!r} (Catalog #287 sister discipline)"
        )
    return stripped


# ----------------------------------------------------------------------------
# Solver-derived dual-variable computation per adjuster
# ----------------------------------------------------------------------------

def _solver_mdl_density_dual_variable(
    base_delta: float,
    mdl_density: float | None,
    *,
    sigma_obs: float = 1.0,
) -> tuple[float, float]:
    """Solver-derived equivalent of ``adjust_predicted_delta_for_mdl_density``.

    Maps the hand-derived 3-band cascade
    (saturated/trending/across-class) to a Gaussian posterior over the MDL
    density signal:

      - Prior: mu=0.5, sigma=0.5 (uniform-ish over [0, 1])
      - Observation: mdl_density (clipped to [0, 1] if outside)
      - Posterior mean drives the band lookup; posterior sigma drives the
        uncertainty weight (Q7 binding decision: 1/(1+sigma) downweight).
      - Within-class saturated (mu > 0.95): floor at MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR
      - Within-class trending (0.90 < mu <= 0.95): scale by 0.5 (penalty)
      - Across-class (mu <= 0.90): passthrough

    Returns (adjusted_delta, posterior_sigma).

    The dual variable IS the inverse-uncertainty weight times the band
    boundary. Phase 2 closed-form is a fixed-point of the canonical 3-band
    cascade; Phase 3 will lift this to a learned Bayesian band-boundary
    optimization via the 4-term findings Lagrangian.
    """
    if mdl_density is None:
        return base_delta, float("nan")

    try:
        d = float(mdl_density)
    except (TypeError, ValueError):
        return base_delta, float("nan")

    # Clip to [0, 1] for numerical stability of the posterior.
    d_clipped = max(0.0, min(1.0, d))

    # Per ``tac.findings_lagrangian.posterior.posterior_update_from_anchors``
    # contract: ``anchor_residuals`` are OFFSETS from prior_mu (not raw
    # observations). The signal we model is the MDL density itself; convert
    # observation -> residual = (observed - prior_mu).
    prior_mu_val = 0.5
    residual = d_clipped - prior_mu_val

    try:
        posterior = posterior_update_from_anchors(
            _CANONICAL_EQUATION_IDS["mdl_density"],
            prior_mu=(prior_mu_val,),
            prior_sigma_diagonal=(0.5,),
            anchor_residuals=[residual],
            sigma_obs=sigma_obs,
        )
    except PosteriorInvalidError:
        return base_delta, float("nan")

    mu = float(posterior.mu[0])
    sigma = float(posterior.posterior_sigma_per_term[0])

    # Band-boundary semantics mirror the hand-derived constants. Import the
    # canonical constants lazily to avoid coupling at module-load time.
    saturated_threshold = 0.95
    trending_threshold = 0.90
    saturated_floor = -0.005  # MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR
    trending_factor = 0.5     # MDL_DENSITY_WITHIN_CLASS_TRENDING_PENALTY_FACTOR

    if mu > saturated_threshold:
        adjusted = max(base_delta, saturated_floor)
    elif mu > trending_threshold:
        adjusted = base_delta * trending_factor
    else:
        adjusted = base_delta

    return adjusted, sigma


def _solver_predicted_dispatch_risk_dual_variable(
    base_delta: float,
    predicted_dispatch_risk: float | None,
    *,
    sigma_obs: float = 1.0,
) -> tuple[float, float]:
    """Solver-derived equivalent of ``adjust_predicted_delta_for_predicted_dispatch_risk``.

    Maps the hand-derived 3-band cascade (refusal/moderate/low) to a
    Gaussian posterior over the SLIM risk score:

      - Prior: mu=12.5, sigma=12.5 (centered on the LOW-MODERATE boundary)
      - Observation: predicted_dispatch_risk
      - Posterior mean drives the band lookup
      - Refusal (mu >= 50.0): floor at PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR
      - Moderate (25.0 <= mu < 50.0): scale by 0.5 (halve savings)
      - Low (mu < 25.0): passthrough

    Returns (adjusted_delta, posterior_sigma).
    """
    if predicted_dispatch_risk is None:
        return base_delta, float("nan")

    try:
        r = float(predicted_dispatch_risk)
    except (TypeError, ValueError):
        return base_delta, float("nan")

    # SLIM scores are positive integers in practice; clip negative to 0.
    r_clipped = max(0.0, r)

    # Residual = (observed - prior_mu) per posterior contract.
    prior_mu_val = 12.5
    residual = r_clipped - prior_mu_val

    try:
        posterior = posterior_update_from_anchors(
            _CANONICAL_EQUATION_IDS["predicted_dispatch_risk"],
            prior_mu=(prior_mu_val,),
            prior_sigma_diagonal=(12.5,),
            anchor_residuals=[residual],
            sigma_obs=sigma_obs,
        )
    except PosteriorInvalidError:
        return base_delta, float("nan")

    mu = float(posterior.mu[0])
    sigma = float(posterior.posterior_sigma_per_term[0])

    refusal_threshold = 50.0
    moderate_threshold = 25.0
    refusal_floor = 0.0  # PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR
    moderate_factor = 0.5  # PREDICTED_DISPATCH_RISK_MODERATE_PENALTY_FACTOR

    if mu >= refusal_threshold:
        adjusted = max(base_delta, refusal_floor)
    elif mu >= moderate_threshold:
        adjusted = base_delta * moderate_factor
    else:
        adjusted = base_delta

    return adjusted, sigma


def _solver_composition_alpha_v2_dual_variable(
    base_delta: float,
    composition_alpha: float | None,
    *,
    sigma_obs: float = 1.0,
) -> tuple[float, float]:
    """Solver-derived equivalent of ``adjust_predicted_delta_for_composition_alpha_v2``.

    Maps the hand-derived 4-band cascade (super_additive/additive/
    sub_additive/saturating) to a Gaussian posterior over the composition
    additivity factor:

      - Prior: mu=0.7, sigma=0.5 (centered on the ADDITIVE boundary)
      - Observation: composition_alpha (clipped to [0, 2.5])
      - Posterior mean drives the band lookup
      - SUPER_ADDITIVE (mu > 1.05): reward = clamp(mu, 1.0, 2.0)
      - ADDITIVE (0.7 < mu <= 1.05): passthrough
      - SUB_ADDITIVE (0.3 < mu <= 0.7): scale by 0.5
      - SATURATING (mu <= 0.3): floor at -0.005

    Returns (adjusted_delta, posterior_sigma).
    """
    if composition_alpha is None:
        return base_delta, float("nan")

    try:
        a = float(composition_alpha)
    except (TypeError, ValueError):
        return base_delta, float("nan")

    # Clip to [0, 2.5] for posterior stability. SUPER_ADDITIVE cap at 2.0
    # already bounds the reward; we allow observation up to 2.5 to avoid
    # truncating the posterior tail.
    a_clipped = max(0.0, min(2.5, a))

    # Residual = (observed - prior_mu) per posterior contract.
    prior_mu_val = 0.7
    residual = a_clipped - prior_mu_val

    try:
        posterior = posterior_update_from_anchors(
            _CANONICAL_EQUATION_IDS["composition_alpha_v2"],
            prior_mu=(prior_mu_val,),
            prior_sigma_diagonal=(0.5,),
            anchor_residuals=[residual],
            sigma_obs=sigma_obs,
        )
    except PosteriorInvalidError:
        return base_delta, float("nan")

    mu = float(posterior.mu[0])
    sigma = float(posterior.posterior_sigma_per_term[0])

    super_additive_threshold = 1.05
    additive_threshold = 0.7
    sub_additive_threshold = 0.3
    saturating_floor = -0.005
    sub_additive_factor = 0.5
    super_additive_reward_floor = 1.0
    super_additive_reward_cap = 2.0

    if mu > super_additive_threshold:
        reward_factor = max(
            super_additive_reward_floor, min(super_additive_reward_cap, mu)
        )
        # In score-delta convention, lower is better. Multiplying a negative
        # delta by a factor > 1 makes it more negative (better-ranked).
        adjusted = base_delta * reward_factor
    elif mu > additive_threshold:
        adjusted = base_delta
    elif mu > sub_additive_threshold:
        adjusted = base_delta * sub_additive_factor
    else:
        adjusted = max(base_delta, saturating_floor)

    return adjusted, sigma


# Canonical dispatcher mapping adjuster name -> solver-derived dual variable
# computer. Kept module-private so the public surface is the
# ``compute_solver_dual_variable_for_adjuster`` function.
_SOLVER_DUAL_VARIABLE_DISPATCH: Final[dict[str, Any]] = {
    "mdl_density": _solver_mdl_density_dual_variable,
    "predicted_dispatch_risk": _solver_predicted_dispatch_risk_dual_variable,
    "composition_alpha_v2": _solver_composition_alpha_v2_dual_variable,
}


def compute_solver_dual_variable_for_adjuster(
    adjuster_name: str,
    base_delta: float,
    signal_value: float | None,
    *,
    sigma_obs: float = 1.0,
) -> tuple[float, float]:
    """Canonical entry-point for the solver-derived dual-variable computation.

    Args:
        adjuster_name: one of ``SUPPORTED_ADJUSTERS``.
        base_delta: the candidate's raw predicted_score_delta (or the running
            cascade value from the prior adjuster).
        signal_value: the adjuster's input signal (mdl_density /
            predicted_dispatch_risk / composition_alpha). ``None`` means the
            signal was not measured; returns ``(base_delta, NaN)`` per the
            sister hand-derived "unknown → passthrough" convention.
        sigma_obs: observation noise std-dev for the Gaussian posterior.

    Returns:
        ``(adjusted_delta, posterior_sigma)`` tuple. The posterior_sigma is
        NaN when the signal is None or non-numeric (matches the hand-derived
        "unknown → passthrough" semantics; no posterior was updated).

    Raises:
        AblationError: if ``adjuster_name`` is not in ``SUPPORTED_ADJUSTERS``.
    """
    if adjuster_name not in _SOLVER_DUAL_VARIABLE_DISPATCH:
        raise AblationError(
            f"adjuster_name={adjuster_name!r} not in SUPPORTED_ADJUSTERS="
            f"{SUPPORTED_ADJUSTERS}"
        )
    fn = _SOLVER_DUAL_VARIABLE_DISPATCH[adjuster_name]
    return fn(base_delta, signal_value, sigma_obs=sigma_obs)


# ----------------------------------------------------------------------------
# Paired-comparison context + verdict dataclasses
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AdjusterAblationContext:
    """Canonical context for one paired-comparison invocation.

    Per CLAUDE.md "Beauty, simplicity, and developer experience": frozen
    dataclass; explicit fields; no hidden state.
    """

    adjuster_name: str
    """One of ``SUPPORTED_ADJUSTERS``."""

    mode: AblationMode
    """Runtime mode (hand_derived / solver_derived / paired_comparison)."""

    base_delta: float
    """Candidate's raw predicted_score_delta entering this adjuster."""

    signal_value: float | None
    """The adjuster's input signal value (mdl_density / dispatch_risk / alpha)."""

    candidate_id: str
    """Identifier of the candidate being adjusted (for cite-chain audit)."""

    family: str = ""
    """Candidate family (e.g. nerv_family / hnerv_family); optional."""

    sigma_obs: float = 1.0
    """Observation noise std-dev for the Gaussian posterior."""

    panel_axis: str = "contest_cpu"
    """Score panel axis (passthrough for downstream consumers)."""

    def __post_init__(self) -> None:
        if self.adjuster_name not in SUPPORTED_ADJUSTERS:
            raise AblationError(
                f"adjuster_name={self.adjuster_name!r} not in "
                f"SUPPORTED_ADJUSTERS={SUPPORTED_ADJUSTERS}"
            )
        if not isinstance(self.mode, AblationMode):
            raise AblationError(
                f"mode must be AblationMode, got {type(self.mode).__name__}"
            )
        # base_delta NaN guard
        if self.base_delta != self.base_delta:
            raise AblationError("base_delta must not be NaN")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise AblationError("candidate_id must be a non-empty string")


@dataclass(frozen=True)
class AdjusterAblationVerdict:
    """Canonical verdict from a paired-comparison invocation.

    Carries per-candidate observability: hand-derived adjustment + solver-
    derived adjustment + divergence statistics + canonical Provenance.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
    every field is observability-only; ``score_claim``, ``promotable``,
    ``axis_tag`` are pinned to non-promotable values.
    """

    context: AdjusterAblationContext
    """The context that produced this verdict."""

    hand_derived_delta: float
    """The hand-derived adjusted delta."""

    solver_derived_delta: float
    """The solver-derived adjusted delta. Equal to hand_derived_delta when
    the mode is HAND_DERIVED (solver not invoked).
    """

    solver_posterior_sigma: float
    """The Gaussian posterior sigma for the solver-derived computation.
    NaN when the signal was missing OR when mode is HAND_DERIVED.
    """

    divergence: float
    """``solver_derived_delta - hand_derived_delta``. Negative = solver
    predicts a stronger improvement than hand-derived; positive = solver is
    more conservative.
    """

    divergence_absolute: float
    """``abs(divergence)``. Convenience for sorting / aggregation."""

    sign_flip: bool
    """True iff the solver and hand-derived adjusted deltas have opposite
    signs (one positive, one negative). A sign flip indicates the two
    adjusters fundamentally disagree on the candidate's direction.
    """

    within_tolerance: bool
    """True iff ``divergence_absolute`` is within ``DEFAULT_DIVERGENCE_SIGMA_BOUND
    × solver_posterior_sigma`` (or always True when mode is HAND_DERIVED).
    """

    authoritative_delta: float
    """The delta the ranker WILL use. In HAND_DERIVED + PAIRED_COMPARISON
    mode this is hand_derived_delta; in SOLVER_DERIVED mode this is
    solver_derived_delta. The PAIRED_COMPARISON default makes hand-derived
    authoritative as a safety rail.
    """

    captured_at_utc: str
    """ISO-8601 UTC timestamp."""

    schema: str = PHASE_2_ABLATION_SCHEMA
    """Canonical schema id pinned at class level."""

    # Canonical Provenance markers per Catalog #287/#323/#341 sister discipline.
    # These are constants by construction; the dataclass surfaces them so
    # downstream consumers can audit without re-deriving.
    score_claim: bool = False
    promotable: bool = False
    axis_tag: str = "[predicted]"

    def __post_init__(self) -> None:
        # Defensive: enforce canonical non-promotable markers per Catalog
        # #341. These should NEVER be flipped True for this verdict surface;
        # the ablation is OBSERVABILITY-ONLY per the master design memo.
        if self.score_claim is not False:
            raise AblationError("score_claim MUST be False (Catalog #341)")
        if self.promotable is not False:
            raise AblationError("promotable MUST be False (Catalog #341)")
        if self.axis_tag != "[predicted]":
            raise AblationError(
                "axis_tag MUST be '[predicted]' (Catalog #341)"
            )

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialize for JSONL persistence. NaN values are encoded as null."""

        def _nan_to_none(x: float) -> float | None:
            return None if x != x else float(x)

        return {
            "schema": self.schema,
            "captured_at_utc": self.captured_at_utc,
            "adjuster_name": self.context.adjuster_name,
            "mode": self.context.mode.value,
            "candidate_id": self.context.candidate_id,
            "family": self.context.family,
            "base_delta": _nan_to_none(self.context.base_delta),
            "signal_value": (
                None if self.context.signal_value is None
                else _nan_to_none(self.context.signal_value)
            ),
            "panel_axis": self.context.panel_axis,
            "sigma_obs": _nan_to_none(self.context.sigma_obs),
            "hand_derived_delta": _nan_to_none(self.hand_derived_delta),
            "solver_derived_delta": _nan_to_none(self.solver_derived_delta),
            "solver_posterior_sigma": _nan_to_none(self.solver_posterior_sigma),
            "divergence": _nan_to_none(self.divergence),
            "divergence_absolute": _nan_to_none(self.divergence_absolute),
            "sign_flip": bool(self.sign_flip),
            "within_tolerance": bool(self.within_tolerance),
            "authoritative_delta": _nan_to_none(self.authoritative_delta),
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "axis_tag": self.axis_tag,
        }


# ----------------------------------------------------------------------------
# Hand-derived adjuster dispatch
# ----------------------------------------------------------------------------

def _invoke_hand_derived_adjuster(
    adjuster_name: str,
    base_delta: float,
    signal_value: float | None,
) -> float:
    """Dispatch to the hand-derived adjuster in ``cathedral_autopilot_autonomous_loop``.

    Lazy import to avoid coupling at module load time. The cathedral autopilot
    module imports findings_lagrangian, not the other way around.
    """
    # Lazy import - the cathedral autopilot module is the canonical source.
    if adjuster_name == "mdl_density":
        from tools.cathedral_autopilot_autonomous_loop import (  # noqa: PLC0415
            adjust_predicted_delta_for_mdl_density,
        )
        return adjust_predicted_delta_for_mdl_density(base_delta, signal_value)

    if adjuster_name == "predicted_dispatch_risk":
        from tools.cathedral_autopilot_autonomous_loop import (  # noqa: PLC0415
            adjust_predicted_delta_for_predicted_dispatch_risk,
        )
        return adjust_predicted_delta_for_predicted_dispatch_risk(
            base_delta, signal_value
        )

    if adjuster_name == "composition_alpha_v2":
        from tools.cathedral_autopilot_autonomous_loop import (  # noqa: PLC0415
            adjust_predicted_delta_for_composition_alpha_v2,
        )
        return adjust_predicted_delta_for_composition_alpha_v2(
            base_delta, signal_value
        )

    raise AblationError(
        f"adjuster_name={adjuster_name!r} not in SUPPORTED_ADJUSTERS="
        f"{SUPPORTED_ADJUSTERS}"
    )


# ----------------------------------------------------------------------------
# Paired comparison entry point
# ----------------------------------------------------------------------------

def paired_comparison_against_hand_derived(
    context: AdjusterAblationContext,
) -> AdjusterAblationVerdict:
    """Run paired comparison: hand-derived adjuster vs solver-derived dual.

    Always returns a verdict (no exceptions for missing signal — the
    sister hand-derived adjusters all passthrough on None signal).

    Per the 3-mode contract:
      - HAND_DERIVED mode: solver is NOT invoked; ``solver_derived_delta`` ==
        hand_derived_delta; ``solver_posterior_sigma`` is NaN; divergence
        is 0.0; authoritative_delta is hand_derived_delta.
      - SOLVER_DERIVED mode: both adjusters fire; authoritative_delta is
        solver_derived_delta.
      - PAIRED_COMPARISON mode (default): both adjusters fire;
        authoritative_delta is hand_derived_delta (safety rail).
    """
    hand_derived_delta = _invoke_hand_derived_adjuster(
        context.adjuster_name, context.base_delta, context.signal_value
    )

    if context.mode is AblationMode.HAND_DERIVED:
        # Solver not invoked. Authoritative = hand.
        return AdjusterAblationVerdict(
            context=context,
            hand_derived_delta=hand_derived_delta,
            solver_derived_delta=hand_derived_delta,
            solver_posterior_sigma=float("nan"),
            divergence=0.0,
            divergence_absolute=0.0,
            sign_flip=False,
            within_tolerance=True,
            authoritative_delta=hand_derived_delta,
            captured_at_utc=_utc_now_iso(),
        )

    # SOLVER_DERIVED or PAIRED_COMPARISON: invoke the solver.
    solver_derived_delta, solver_sigma = compute_solver_dual_variable_for_adjuster(
        context.adjuster_name,
        context.base_delta,
        context.signal_value,
        sigma_obs=context.sigma_obs,
    )

    divergence = solver_derived_delta - hand_derived_delta
    divergence_absolute = abs(divergence)

    # Sign flip: both deltas have non-zero opposite signs.
    sign_flip = (
        (hand_derived_delta > 0 > solver_derived_delta)
        or (hand_derived_delta < 0 < solver_derived_delta)
    )

    # Within-tolerance: divergence_absolute <= DEFAULT_DIVERGENCE_SIGMA_BOUND *
    # solver_posterior_sigma. When sigma is NaN (signal was None), tolerance
    # is True by definition (no comparison to make).
    if solver_sigma != solver_sigma:  # NaN
        within_tolerance = True
    else:
        sigma_bound = DEFAULT_DIVERGENCE_SIGMA_BOUND * abs(solver_sigma)
        within_tolerance = divergence_absolute <= sigma_bound

    if context.mode is AblationMode.SOLVER_DERIVED:
        authoritative_delta = solver_derived_delta
    else:
        # PAIRED_COMPARISON: hand-derived is authoritative (safety rail).
        authoritative_delta = hand_derived_delta

    return AdjusterAblationVerdict(
        context=context,
        hand_derived_delta=hand_derived_delta,
        solver_derived_delta=solver_derived_delta,
        solver_posterior_sigma=solver_sigma,
        divergence=divergence,
        divergence_absolute=divergence_absolute,
        sign_flip=sign_flip,
        within_tolerance=within_tolerance,
        authoritative_delta=authoritative_delta,
        captured_at_utc=_utc_now_iso(),
    )


# ----------------------------------------------------------------------------
# fcntl-locked JSONL APPEND-ONLY persistence (Catalog #131/#138/#245 pattern)
# ----------------------------------------------------------------------------

def _ensure_state_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_paired_comparison_row(
    verdict: AdjusterAblationVerdict,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """Append a paired-comparison verdict to the JSONL posterior.

    Atomic write per Catalog #131/#138/#245 canonical pattern:
      1. Acquire fcntl LOCK_EX on a sibling lock file.
      2. Write the JSON line under the lock.
      3. Release the lock.

    Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact":
    canonical path is ``.omx/state/phase_2_ablation_posterior.jsonl``.
    """
    posterior = posterior_path or PHASE_2_ABLATION_POSTERIOR_PATH
    lock = lock_path or PHASE_2_ABLATION_POSTERIOR_LOCK_PATH
    _ensure_state_dir(posterior)
    _ensure_state_dir(lock)

    payload = verdict.to_jsonl_dict()
    # Add row-write metadata to satisfy sister forensic disciplines.
    payload["written_pid"] = os.getpid()
    payload["written_at_utc"] = _utc_now_iso()
    payload["row_uuid"] = uuid.uuid4().hex[:12]

    line = json.dumps(payload, sort_keys=True, allow_nan=False)

    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with posterior.open("a", encoding="utf-8") as pf:
                pf.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def load_paired_comparison_rows_lenient(
    posterior_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read every paired-comparison row, skipping malformed lines.

    Tolerant load: malformed JSON lines are silently skipped (the caller
    can detect drift by comparing line count vs row count).
    """
    posterior = posterior_path or PHASE_2_ABLATION_POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = posterior.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def load_paired_comparison_rows_strict(
    posterior_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read every paired-comparison row; raise on ANY malformed JSON.

    Per Catalog #138 strict-load discipline: a corrupt posterior is a
    fail-closed condition; the caller must explicitly opt-in to lenient
    loading via ``load_paired_comparison_rows_lenient`` when working with
    partially-written posteriors.
    """
    posterior = posterior_path or PHASE_2_ABLATION_POSTERIOR_PATH
    if not posterior.exists():
        return []
    out: list[dict[str, Any]] = []
    text = posterior.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AblationError(
                f"phase_2_ablation_posterior corrupt at line {lineno}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise AblationError(
                f"phase_2_ablation_posterior row at line {lineno} is not a "
                f"dict (got {type(row).__name__})"
            )
        out.append(row)
    return out
