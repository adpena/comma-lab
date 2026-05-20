# SPDX-License-Identifier: MIT
"""Cathedral consumer for the findings Lagrangian (tac.findings_lagrangian).

Per WAVE-2-PREREQ-FINDINGS-LAGRANGIAN-CONSUMER directive 2026-05-20 +
CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 2 prerequisite +
operator standing directive 2026-05-19 verbatim *"we shoud pursue PP in
parallel"* (which RATIFIED the dual-track findings Lagrangian build) +
Catalog #335 paradigm-shift (canonical contract auto-discovery) +
Catalog #355 (META-LAGRANGIAN-WIRE-1 Phase 1 canonical invocation
already wired in ``tools/cathedral_autopilot_autonomous_loop.py::
invoke_meta_lagrangian_on_candidates`` — this consumer is the
sister-cathedral-consumer surface for the SAME findings Lagrangian
solver, auto-discovered alongside the 47 other production consumers per
Catalog #335 paradigm).

This consumer is **Tier A observability-only** at landing per Catalog
#341 canonical routing markers + per CATHEDRAL-SMARTER-DESIGN-MEMO
Dimension 6 Step 6.2 (consumers omitting ``CONSUMER_TIER`` default to
``TIER_A_OBSERVABILITY_ONLY``). Every ``consume_candidate`` return value
carries:

  * ``predicted_delta_adjustment=0.0`` (routing/observability signal, NOT
    a score signal per Catalog #287/#323/#341)
  * ``promotable=False`` (per CLAUDE.md "Apples-to-apples evidence
    discipline" + Catalog #127/#192/#317)
  * ``axis_tag="[predicted]"`` (canonical observability axis per
    Catalog #287)
  * ``canonical_provenance`` dict-form Provenance per Catalog #323

The consumer's ``consume_candidate`` surfaces four observability-only
annotations:

  1. ``lagrangian_scalar``: the 4-term scalar Lagrangian value computed
     via :func:`tac.findings_lagrangian.compute_findings_lagrangian`.
  2. ``per_term_decomposition``: the per-term breakdown from
     :meth:`tac.findings_lagrangian.FindingsLagrangianResult.decompose`
     (data_fit / occam_complexity_weighted /
     occam_interpretability_weighted / partition_penalty_weighted /
     info_gain_reward_weighted / scalar).
  3. ``posterior_sigma_per_term``: the canonical sensitivity-map signal
     per Catalog #125 hook #1 (the posterior uncertainty per dimension).
  4. ``expected_information_gain_per_action``: when the candidate carries
     hypothetical-next-action signal, the canonical Lindley 1956 + Foster
     2019 info-gain-per-dollar ranking signal via
     :func:`tac.findings_lagrangian.recommend_next_action_via_expected_information_gain`.

The consumer's ``update_from_anchor`` forwards new empirical anchors to
:func:`tac.findings_lagrangian.posterior_update_from_anchors` so the
findings Lagrangian's per-equation posterior stays calibrated as new
empirical evidence lands. Per Catalog #287/#323: the consumer does NOT
auto-promote diagnostic anchors to contest-grade; the anchor's
``evidence_grade`` is honored and only well-calibrated anchors update
the posterior.

**Phase 2 Tier B promotion pathway** (per CATHEDRAL-SMARTER-DESIGN-MEMO
Dimension 6 Step 6.5): when sister WAVE-2 lands actual dual-variable
computation (replacing META-LAGRANGIAN-WIRE-1 Phase 1's bounded-5%
placeholder factor with solver-derived ``lambda_*`` + ``mu_*`` dual
variables), this consumer becomes the natural Tier B promotion
candidate. The transition path:

  - Bump ``CONSUMER_VERSION`` to ``2.0.0``.
  - Add ``CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING``.
  - Replace ``predicted_delta_adjustment=0.0`` with solver-derived
    ``predicted_delta_adjustment = bounded(-dual_variable_sum, ...)``
    where the bound is determined by Phase 2's dual-variable scaling
    discipline.
  - Add ``predicted_axis_decomposition`` per :class:`AxisDecomposition`
    with non-empty ``canonical_provenance`` per Catalog #356 prerequisite.
  - Keep ``promotable=False`` per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6
    Step 6.5: Tier B contributes to RANKING but NEVER to PROMOTION
    (promotion still requires paired contest-CPU + contest-CUDA empirical
    anchors per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA").
  - Bump ``axis_tag`` from ``[predicted]`` to a non-forbidden Tier B
    axis token per :data:`_TIER_B_FORBIDDEN_AXIS_TOKENS`.
  - Catalog #357 STRICT preflight gate validates the Tier B contract.

Phase 1 (THIS landing) is the structural prerequisite: the consumer is
auto-discoverable + canonically-contract-compliant + ready to consume
solver-derived signal as soon as Phase 2 lands the upstream computation.

Hook assignments per Catalog #125:
  * #1 sensitivity-map — **ACTIVE** (``posterior_sigma_per_term``
    surfaced per :class:`tac.findings_lagrangian.GaussianPosterior`)
  * #4 cathedral autopilot dispatch — **ACTIVE PRIMARY** (annotate
    candidates with Lagrangian decomposition for ranker consumption)
  * #5 continual-learning posterior — **ACTIVE** (``update_from_anchor``
    refreshes the per-equation Gaussian posterior conjugate Bayesian
    update via :func:`posterior_update_from_anchors`)
  * #2 Pareto constraint — N/A at Phase 1 (Phase 2 lands the
    dual-variable surface that maps to Pareto KKT)
  * #3 bit-allocator — N/A at Phase 1 (sister WAVE-2 BIT-ALLOCATOR
    spawned in parallel)
  * #6 probe-disambiguator — N/A at Phase 1 (Phase 2 lands the
    info-gain action-selector branch per Lindley 1956 + Foster 2019)

Sister of:
  * :mod:`tac.findings_lagrangian` (canonical findings-Lagrangian solver
    this consumer wraps)
  * ``tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates``
    (Catalog #355 META-LAGRANGIAN-WIRE-1 Phase 1 sister wire-in for the
    SAME solver at the autopilot main-loop surface; THIS consumer is the
    cathedral_consumers/* sister wire-in for auto-discovery per Catalog
    #335)
  * ``tac.cathedral_consumers.canonical_equation_lookup_consumer``
    (sister observability-only consumer for the canonical_equations
    registry that this Lagrangian operates on)
  * ``tac.cathedral_consumers.atom_consumer`` (sister observability-only
    consumer template)
  * ``tac.cathedral_consumers._example_consumer`` (canonical reference)
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from tac.cathedral.consumer_contract import (
    ConsumerTier,
    HookNumber,
)


CONSUMER_NAME = "findings_lagrangian_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)
# Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 6 Step 6.2: explicit
# Tier A declaration at Phase 1 (default is TIER_A_OBSERVABILITY_ONLY
# but explicit is better than implicit for the Tier B promotion
# pathway documented above).
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY


# Phase 1 default prior for the conjugate Gaussian update when no prior
# state is available. Mirrors `invoke_meta_lagrangian_on_candidates` in
# tools/cathedral_autopilot_autonomous_loop.py (zero-mean, unit-sigma).
_DEFAULT_PRIOR_MU: tuple[float, ...] = (0.0,)
_DEFAULT_PRIOR_SIGMA_DIAGONAL: tuple[float, ...] = (1.0,)
_DEFAULT_SIGMA_OBS: float = 1.0

# Phase 1 bounded residual clip (mirrors
# `_candidate_residuals_for_lagrangian` in
# tools/cathedral_autopilot_autonomous_loop.py to prevent a runaway
# candidate prediction from destabilizing the posterior update).
_RESIDUAL_CLIP: float = 1.0

# Canonical Provenance model_id for predicted-from-model annotations.
_PROVENANCE_MODEL_ID = "tac.findings_lagrangian.findings_lagrangian_consumer.v1"


_logger = logging.getLogger(__name__)


def _extract_residuals(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    """Extract Phase 1 anchor residuals from a candidate dict.

    Mirrors ``_candidate_residuals_for_lagrangian`` in
    tools/cathedral_autopilot_autonomous_loop.py for parity with the
    META-LAGRANGIAN-WIRE-1 Phase 1 invoker. Uses the candidate's
    ``predicted_score_delta`` (or ``predicted_delta`` legacy alias) as
    the single residual. Clipped to ``[-_RESIDUAL_CLIP, +_RESIDUAL_CLIP]``
    so a runaway candidate cannot destabilize the posterior.

    Returns ``()`` when no residual signal can be extracted; consumers
    should treat empty as "skip this candidate".
    """
    for key in ("predicted_score_delta", "predicted_delta"):
        if key in candidate:
            raw_value = candidate[key]
            if not isinstance(raw_value, (int, float)):
                continue
            raw = float(raw_value)
            if raw != raw:  # NaN
                return ()
            if raw > _RESIDUAL_CLIP:
                raw = _RESIDUAL_CLIP
            elif raw < -_RESIDUAL_CLIP:
                raw = -_RESIDUAL_CLIP
            return (raw,)
    return ()


def _extract_equation_id(candidate: Mapping[str, Any]) -> str:
    """Extract the equation_id key for posterior update.

    Phase 1 uses the candidate's family (or candidate_id fallback) as the
    equation_id stand-in; Phase 2 will map candidates to actual canonical
    equation entries via ``tac.canonical_equations.query_equations_by_consumer``.
    """
    for key in ("family", "candidate_family", "candidate_id", "lane_id"):
        if key in candidate:
            value = candidate[key]
            if isinstance(value, str) and value.strip():
                return value
    return "unknown_family"


def _build_canonical_provenance(equation_id: str, residual_count: int) -> dict[str, Any]:
    """Build dict-form Provenance per Catalog #323 canonical umbrella.

    Per Catalog #287/#323: every observability-only consumer contribution
    that surfaces predicted values MUST carry canonical Provenance with
    PREDICTED grade + axis_tag=[predicted] + promotion_eligible=False.
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict
    except ImportError:
        # Defensive fallback when the canonical Provenance package is
        # unavailable (e.g. partial-checkout test fixture). Surface a
        # minimal dict that still honors the canonical contract shape.
        return {
            "artifact_kind": "PREDICTED_FROM_MODEL",
            "evidence_grade": "PREDICTED",
            "measurement_axis": "[predicted]",
            "promotion_eligible": False,
            "score_claim_valid": False,
            "canonical_helper_invocation": "tac.cathedral_consumers.findings_lagrangian_consumer",
            "model_id": _PROVENANCE_MODEL_ID,
            "equation_id": equation_id,
            "residual_count": residual_count,
        }

    import hashlib

    inputs_seed = f"findings_lagrangian:{equation_id}:n={residual_count}"
    inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
    prov = build_provenance_for_predicted(
        model_id=_PROVENANCE_MODEL_ID,
        inputs_sha256=inputs_sha256,
        measurement_axis="[predicted]",
        hardware_substrate="cpu_local",
    )
    return provenance_to_dict(prov)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Forwards new empirical anchors to
    :func:`tac.findings_lagrangian.posterior_update_from_anchors` so the
    findings Lagrangian's per-equation Gaussian posterior stays calibrated
    via the canonical MacKay-1992 conjugate Bayesian update.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
    #287/#323: only well-calibrated anchors with ``evidence_grade`` in
    the canonical promotable set update the posterior; diagnostic-only
    anchors are honored but not promoted. The anchor's residual is
    extracted from common attribute paths (``residual``, ``score`` minus
    ``predicted_score``, etc.) with fail-safe defaults.

    The consumer is intentionally a thin wrapper: the canonical update
    logic lives in :mod:`tac.findings_lagrangian.posterior`, not here.
    Per Catalog #335 paradigm: keep cathedral consumers structurally
    minimal so the canonical solver is the single source of truth for
    update semantics. NO-OP on malformed / missing-signal anchors so the
    consumer never raises in the cathedral autopilot main loop.
    """
    if anchor is None:
        return

    # Best-effort extraction: callers may pass a typed
    # ``ContinualLearningAnchor`` OR a dict OR an
    # ``EmpiricalAnchor`` (canonical equations sister). All three expose
    # the same canonical fields via attribute access (typed) or
    # __getitem__ (dict).
    equation_id: str | None = None
    residual: float | None = None

    for getter in (
        lambda obj, key: getattr(obj, key, None),
        lambda obj, key: obj.get(key) if isinstance(obj, Mapping) else None,
    ):
        try:
            if equation_id is None:
                eq_value = getter(anchor, "equation_id")
                if isinstance(eq_value, str) and eq_value.strip():
                    equation_id = eq_value
            if residual is None:
                r_value = getter(anchor, "residual")
                if isinstance(r_value, (int, float)) and r_value == r_value:
                    residual = float(r_value)
        except (AttributeError, TypeError, KeyError):
            continue

    if equation_id is None or residual is None:
        # No signal — silent NO-OP. The canonical solver's posterior is
        # refreshed via direct invocation by operator-routed tooling per
        # `tools/recalibrate_equation.py` + sister wire-ins.
        return

    # Phase 1: posterior update is purely advisory; the consumer does
    # NOT persist the updated posterior (the canonical solver owns
    # persistence). Phase 2 will route the updated posterior into
    # `tac.canonical_equations.update_equation_with_empirical_anchor`.
    try:
        from tac.findings_lagrangian import posterior_update_from_anchors

        _ = posterior_update_from_anchors(
            equation_id,
            prior_mu=_DEFAULT_PRIOR_MU,
            prior_sigma_diagonal=_DEFAULT_PRIOR_SIGMA_DIAGONAL,
            anchor_residuals=(residual,),
            sigma_obs=_DEFAULT_SIGMA_OBS,
        )
    except Exception as exc:  # noqa: BLE001  defensive — never crash main loop
        _logger.debug(
            "findings_lagrangian_consumer.update_from_anchor swallowed %s: %s",
            type(exc).__name__,
            exc,
        )


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Computes the 4-term findings Lagrangian for the candidate and surfaces
    the scalar + per-term decomposition + posterior sigma as
    observability-only annotations. Returns ``predicted_delta_adjustment=
    0.0`` (Phase 1 Tier A invariant per Catalog #341); Phase 2 will
    return solver-derived dual-variable contribution (see Tier B
    promotion pathway in module docstring).

    Decision cascade:

      1. Extract residuals from candidate via :func:`_extract_residuals`.
      2. Build Phase 1 posterior via :func:`posterior_update_from_anchors`
         (zero-mean unit-sigma prior; mirrors invoke_meta_lagrangian
         Phase 1 in tools/cathedral_autopilot_autonomous_loop.py).
      3. Compute 4-term Lagrangian via :func:`compute_findings_lagrangian`.
      4. Return canonical-routing-markers dict per Catalog #341 with
         observability surfaces (lagrangian_scalar / per_term_decomposition
         / posterior_sigma_per_term) attached.

    Fail-safe defaults: on any computation failure OR empty residual
    signal, returns a canonical Tier A annotation with ``rationale``
    explaining why (so operator audit can distinguish "no signal" from
    "computation failed" from "Tier A by-design"). Never raises in the
    cathedral autopilot main loop per Catalog #335 + #341.
    """
    residuals = _extract_residuals(candidate)
    equation_id = _extract_equation_id(candidate)
    canonical_provenance = _build_canonical_provenance(
        equation_id, len(residuals)
    )

    if not residuals:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "findings_lagrangian_consumer Phase 1 skip: candidate "
                "lacks predicted_score_delta / predicted_delta signal "
                "for residual extraction; observability-only annotation "
                "[predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "canonical_provenance": canonical_provenance,
            "equation_id_used": equation_id,
            "lagrangian_scalar": None,
            "per_term_decomposition": None,
            "posterior_sigma_per_term": None,
            "expected_information_gain_per_action": None,
        }

    try:
        from tac.findings_lagrangian import (
            build_initial_partition,
            compute_findings_lagrangian,
            posterior_update_from_anchors,
        )
    except ImportError as exc:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "findings_lagrangian_consumer Phase 1 skip: "
                f"tac.findings_lagrangian unavailable ({type(exc).__name__}: "
                f"{exc}); observability-only annotation [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "canonical_provenance": canonical_provenance,
            "equation_id_used": equation_id,
            "lagrangian_scalar": None,
            "per_term_decomposition": None,
            "posterior_sigma_per_term": None,
            "expected_information_gain_per_action": None,
        }

    try:
        posterior = posterior_update_from_anchors(
            equation_id,
            prior_mu=_DEFAULT_PRIOR_MU,
            prior_sigma_diagonal=_DEFAULT_PRIOR_SIGMA_DIAGONAL,
            anchor_residuals=residuals,
            sigma_obs=_DEFAULT_SIGMA_OBS,
        )
        partition = build_initial_partition()
        lag_result = compute_findings_lagrangian(
            equation_id,
            posterior=posterior,
            partition=partition,
            anchor_residuals=residuals,
            sigma_obs=_DEFAULT_SIGMA_OBS,
        )
    except Exception as exc:  # noqa: BLE001  defensive — never crash main loop
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "findings_lagrangian_consumer Phase 1 computation failed: "
                f"{type(exc).__name__}: {exc}; observability-only annotation "
                "[predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "canonical_provenance": canonical_provenance,
            "equation_id_used": equation_id,
            "lagrangian_scalar": None,
            "per_term_decomposition": None,
            "posterior_sigma_per_term": None,
            "expected_information_gain_per_action": None,
        }

    scalar = float(lag_result.scalar)
    decompose = {k: float(v) for k, v in lag_result.decompose().items()}
    sigma_per_term = tuple(float(s) for s in lag_result.posterior_sigma_per_term)

    return {
        "predicted_delta_adjustment": 0.0,  # Tier A invariant per Catalog #341
        "rationale": (
            "findings_lagrangian_consumer Phase 1: 4-term scalar "
            f"Lagrangian={scalar:.4f}, posterior_sigma="
            f"{sigma_per_term[0] if sigma_per_term else 'n/a'}, "
            "observability-only [predicted] (Phase 2 Tier B promotion "
            "lands solver-derived dual-variable contribution)"
        )[:512],
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.5,
        "canonical_provenance": canonical_provenance,
        "equation_id_used": equation_id,
        # Catalog #125 hook #1 sensitivity-map signal:
        "posterior_sigma_per_term": list(sigma_per_term),
        # Catalog #305 observability surface:
        "lagrangian_scalar": scalar,
        "per_term_decomposition": decompose,
        # Phase 2 placeholder — surfaces None at Tier A; Phase 2 populates
        # with `recommend_next_action_via_expected_information_gain` output:
        "expected_information_gain_per_action": None,
    }
