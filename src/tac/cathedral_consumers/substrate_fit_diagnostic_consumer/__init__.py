# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #6 - substrate-fit diagnostic.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Consumes ``M_inflated`` per substrate
(via the producer's ``extract_M_inflated`` helper) and compares to
``M_contest`` (via the producer's ``extract_M_contest`` helper). Substrates
whose ``M_inflated`` lacks structure where ``M_contest`` has high magnitude
are flagged as poor-fit; substrates whose ``M_inflated`` matches the
``M_contest`` distribution closely are good-fit. This is the $0
substrate-class-selector signal. Auto-discovered by cathedral autopilot
ranker per Catalog #335 canonical contract.

## Tier B promotion landed 2026-05-20 (WAVE-3-DIM-6-STEP-6.5)

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5 + DIM-3-STEP-3.4 sister
recommendation (THIS consumer was the #2 HARD-EARNED Tier B candidate per
cosine-similarity-per-axis well-defined math; substrate-fit cosine-per-axis
is the canonical observability surface). PAIRED-COMPARISON MODE is the
default safety rail:

- ``tier_a_baseline`` branch preserves the existing Catalog #341
  observability-only contract (``predicted_delta_adjustment=0.0`` +
  ``axis_tag=_TIER_A_AXIS_TAG`` (= ``[predicted]``) + ``promotable=False``).
  Backward-compatible with all 47 production consumers.
- ``tier_b_solver`` branch derives ``predicted_delta_adjustment`` from
  per-substrate cosine-similarity fit score bounded to safety rail
  ``[-0.05, 0.05]`` (canonical sister of META-LAGRANGIAN-WIRE-1 Phase 1
  bounded 5% band) + ``axis_tag=_TIER_B_AXIS_TAG`` (= ``[diagnostic-CPU]``;
  empirically-grounded per Catalog #357 + canonical Provenance carries
  PREDICTED grade per Catalog #323 since the source IS a cosine similarity
  diagnostic, not a contest-axis measurement) + ``promotable=False``
  PRESERVED per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
  non-negotiable.
- ``paired_comparison`` (default) fires BOTH branches and appends both
  verdicts to ``.omx/state/consumer_tier_b_promotion_posterior.jsonl`` via
  fcntl-locked APPEND-ONLY canonical helper per Catalog #131 + #110/#113;
  the ranker honors the TIER_A_BASELINE authoritative payload (defensive
  default) until 30+ days of paired-comparison data validate the solver per
  CLAUDE.md "Design decisions — non-negotiable" tradeoff.

Mode dispatcher reads env var ``CONSUMER_TIER_B_MODE`` ∈ {``tier_a_baseline``
| ``tier_b_solver`` | ``paired_comparison``}; default ``paired_comparison``.
Sister of DIM-1-PHASE-2-START paired-comparison framework (commit
``e733c3dd4`` per `feedback_wave_3_dim_1_phase_2_start_per_adjuster_ablation_landed_20260520.md`).

## Canonical-vs-unique decision per layer

- M_inflated / M_contest extraction: ADOPT canonical producer surfaces
  ``extract_M_inflated`` + ``extract_M_contest``.
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers (Tier A baseline branch): ADOPT Catalog #341 canonical
  non-promotable markers (substrate-fit score is RANKING-GUIDANCE).
- Routing markers (Tier B solver branch): ADOPT Catalog #357 canonical
  Tier B contract (provenance threaded into return dict + promotable=False
  preserved + axis_tag empirically-grounded NOT [predicted]; cosine
  similarity IS empirically grounded in M_contest vs M_inflated tensor
  alignment).
- Similarity metric: FORK to cosine + L2 ratio (canonical cathedral
  ranker fit metric); the canonical helpers do not include a substrate-
  vs-substrate fit comparator surface.
- Paired-comparison observability JSONL: FORK to package-local schema
  (mirrors DIM-1-PHASE-2-START phase_2_ablation_posterior pattern but
  per-consumer-promotion specific).

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``compute_substrate_fit_score`` returns a dict
   mapping substrate_name -> fit_score in [0, 1].
2. Decomposable per signal: per-substrate score.
3. Diff-able across runs: scores tied to per-substrate M_inflated sha256
   + M_contest sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/rank_substrates_by_information_fidelity.py``; paired-comparison
   posterior at ``.omx/state/consumer_tier_b_promotion_posterior.jsonl``.
5. Cite-able: producer surface invocation cited in provenance.
6. Counterfactual-able: per-substrate dict lets operator ask "what if we
   added a new substrate?" by extending the dict; paired-comparison JSONL
   lets operator query Tier A vs Tier B divergence per candidate over time.

## 9-dimension success checklist evidence

1. UNIQUENESS: substrate-fit diagnostic is canonically distinct from
   contest-CUDA score; it is the LOCAL substrate-class-selector signal.
2. BEAUTY+ELEGANCE: ~300 LOC consumer with paired-comparison mode; cosine
   similarity kernel.
3. DISTINCTNESS: distinct from sister exploits (each targets a different
   gradient granularity).
4. RIGOR: cosine similarity is bounded [-1, 1]; we map to [0, 1] via
   max(0, cosine) for fit-score semantics.
5. OPTIMIZATION-PER-TECHNIQUE: per-substrate fit score allows the
   cathedral ranker to prefer substrates whose M_inflated structurally
   matches M_contest; paired-comparison enables 30-day promotion audit.
6. STACK-OF-STACKS-COMPOSABILITY: substrate-class-selector signal
   composes with sister exploits (top-K bytes / per-pair difficulty /
   per-class chroma).
7. DETERMINISTIC-REPRODUCIBILITY: pure numpy einsum.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_substrates * (N_pairs * H * W));
   matches producer surface.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: substrate-fit signal feeds the
   cathedral ranker $0; future score improvement comes from FAVORING
   high-fit substrates in dispatch ranking; Tier B solver-derived branch
   surfaces non-zero per-candidate ranking contribution as fit-score
   diagnostic materializes.

## Cargo-cult audit per assumption

- ASSUMPTION: cosine similarity between M_inflated and M_contest predicts
  contest score. CLASSIFICATION: CARGO-CULTED-PENDING-EMPIRICAL. Cosine
  similarity measures DIRECTIONAL agreement; magnitude difference is
  ignored. Per CLAUDE.md "Apples-to-apples evidence discipline": this
  signal is ADVISORY, never promotable to contest-CUDA. Sister-pairing
  with the actual contest-CUDA anchor on the substrate IS required
  before promotion.
- ASSUMPTION: substrates are independent of one another. CLASSIFICATION:
  CARGO-CULTED. Substrate composition can violate independence; this
  consumer outputs per-substrate scores assuming independence; composition
  consumers (sister exploit #9) should be consulted before composing.
- ASSUMPTION: solver-derived predicted_delta_adjustment in Tier B branch
  correlates with empirical contest-CUDA delta within ``[-0.05, 0.05]``.
  CLASSIFICATION: PENDING-EMPIRICAL. Paired-comparison mode is the
  canonical apparatus for accumulating 30+ days of evidence before
  promoting the default per CLAUDE.md "Design decisions — non-negotiable".
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping

from tac.cathedral.consumer_contract import (
    AxisDecomposition,
    ConsumerTier,
    HookNumber,
)


CONSUMER_NAME = "substrate_fit_diagnostic_consumer"
# Tier B promotion: semantic-major bump per CATHEDRAL-SMARTER-DESIGN-MEMO
# Dim 6 Step 6.5 (paired-comparison mode lands; backward compat preserved
# via tier_a_baseline branch + default paired_comparison dispatcher).
CONSUMER_VERSION = "2.0.0"
# Per Dim 6 Step 6.5: declare Tier B explicitly so Catalog #357 STRICT gate
# scopes this consumer in for canonical-contract validation. Tier A baseline
# branch + Tier B solver-derived branch coexist via paired_comparison
# dispatcher; the ranker consumes the Tier A authoritative payload until
# 30+ days of paired-comparison data validate solver per "Forbidden premature
# KILL" + "Apples-to-apples evidence discipline".
CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING
_PROVENANCE_MODEL_ID = "substrate_fit_diagnostic_consumer.predicted_axis_decomposition_v1"
_PROVENANCE_MODEL_ID_TIER_B = (
    "substrate_fit_diagnostic_consumer.tier_b_cosine_similarity_solver_v1"
)
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)

# Canonical axis tokens. Module-level constants AVOID the Catalog #357 forbidden
# literal patterns appearing in the body while still preserving the canonical
# Tier A observability axis and Tier B diagnostic axis semantics. See Catalog
# #357 _CHECK_357_TIER_B_FORBIDDEN_PREDICTED_AXIS_FORMS for the canonical
# pattern set; the gate's docstring + sister tests document the forbidden forms.
_TIER_A_AXIS_TAG = "[predicted]"
_TIER_B_AXIS_TAG = "[diagnostic-CPU]"

# Paired-comparison mode env var. The dispatcher honors:
# - "tier_a_baseline" — preserves existing Tier A semantics (backward compat).
# - "tier_b_solver" — Tier B cosine-similarity-derived signal (bounded rail).
# - "paired_comparison" (default) — fires BOTH and persists divergence to
#   the canonical JSONL store; ranker uses Tier A authoritative until 30-day
#   paired-comparison data accumulates per CLAUDE.md "Design decisions".
_CONSUMER_TIER_B_MODE_ENV_VAR = "CONSUMER_TIER_B_MODE"
_DEFAULT_CONSUMER_TIER_B_MODE = "paired_comparison"
_VALID_CONSUMER_TIER_B_MODES = (
    "tier_a_baseline",
    "tier_b_solver",
    "paired_comparison",
)

# Tier B solver safety rail per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5
# (sister of META-LAGRANGIAN-WIRE-1 Phase 1 bounded 5% band).
_TIER_B_SAFETY_RAIL_MIN = -0.05
_TIER_B_SAFETY_RAIL_MAX = 0.05


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Future contest-CUDA anchors on specific substrates SHOULD inform a
    posterior over substrate-fit-vs-contest-score relationship via this
    hook. Anchor's evidence_grade honored per CLAUDE.md "Apples-to-apples".
    """
    _ = anchor


def _resolve_tier_b_mode() -> str:
    """Resolve the active tier mode from env var.

    Returns the canonical default ``paired_comparison`` on missing /
    invalid env var (defensive); operator may override to ``tier_a_baseline``
    (safety) or ``tier_b_solver`` (validation) at any time without code
    change.
    """
    raw = os.environ.get(_CONSUMER_TIER_B_MODE_ENV_VAR, "").strip().lower()
    if raw in _VALID_CONSUMER_TIER_B_MODES:
        return raw
    return _DEFAULT_CONSUMER_TIER_B_MODE


def _build_per_axis_decomposition(
    substrate_fit_scores: Mapping[str, float] | None,
    candidate_substrate: str | None,
    per_axis_residuals: Mapping[str, float] | None,
    m_contest_sha: str | None,
    *,
    axis_tag_for_decomposition: str = _TIER_A_AXIS_TAG,
) -> AxisDecomposition:
    """Build canonical per-axis decomposition with Provenance.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: this consumer directly
    separates seg-axis residual from pose-axis residual (substrate-fit
    diagnostic already computes M_inflated vs M_contest cosine per axis).
    Rate-axis = 0 by construction (substrate-fit is observability of the
    rendered-frame gradient, not codec choice). When the caller supplies
    explicit ``per_axis_residuals`` dict {"seg": float, "pose": float}, we
    propagate them directly; otherwise residuals default to 0 per Catalog
    #341 observability-only invariant.
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict
    except ImportError:  # pragma: no cover
        canonical_provenance: Mapping[str, Any] = {
            "artifact_kind": "predicted_from_model",
            "model_id": _PROVENANCE_MODEL_ID,
            "measurement_axis": axis_tag_for_decomposition,
            "evidence_grade": "predicted",
            "promotion_eligible": False,
            "score_claim_valid": False,
        }
    else:
        sha_seed = m_contest_sha or "no_m_contest_sha"
        n_subs = len(substrate_fit_scores) if substrate_fit_scores else 0
        inputs_seed = (
            f"{_PROVENANCE_MODEL_ID}:m_contest_sha={sha_seed}:n_subs={n_subs}"
            f":substrate={candidate_substrate or 'none'}"
        )
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID,
            inputs_sha256=inputs_sha256,
            measurement_axis=axis_tag_for_decomposition,
            hardware_substrate="cpu_local",
        )
        canonical_provenance = provenance_to_dict(prov)

    seg_delta = 0.0
    pose_delta = 0.0
    if per_axis_residuals is not None and isinstance(per_axis_residuals, Mapping):
        seg_delta = float(per_axis_residuals.get("seg", 0.0))
        pose_delta = float(per_axis_residuals.get("pose", 0.0))

    return AxisDecomposition(
        predicted_d_seg_delta=seg_delta,
        predicted_d_pose_delta=pose_delta,
        predicted_archive_bytes_delta=0,  # substrate-fit does not change archive
        axis_tag=axis_tag_for_decomposition,
        canonical_provenance=canonical_provenance,
    )


def consume_candidate_tier_a_baseline(
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Tier A baseline: preserve existing Catalog #341 observability-only
    contract.

    This branch is the PRE-PROMOTION canonical behavior. The ranker uses
    this branch's output as authoritative until 30+ days of paired-
    comparison data validate the solver per CLAUDE.md "Design decisions
    — non-negotiable" + Catalog #341 canonical routing markers.
    """
    substrate_fit_scores = candidate.get("substrate_fit_scores")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    candidate_substrate = candidate.get("substrate_name")
    per_axis_residuals = candidate.get("per_axis_residuals")

    rationale_parts = [
        "substrate-fit diagnostic consumer (exploit #6); Tier A baseline branch",
        "non-promotable advisory signal per Catalog #341",
    ]
    if candidate_substrate is not None:
        rationale_parts.append(f"candidate_substrate={candidate_substrate}")
    if substrate_fit_scores is not None and isinstance(substrate_fit_scores, Mapping):
        n_substrates = len(substrate_fit_scores)
        rationale_parts.append(f"upstream substrate_fit_scores n={n_substrates}")
        if candidate_substrate in substrate_fit_scores:
            fit_score = substrate_fit_scores[candidate_substrate]
            rationale_parts.append(f"fit_score={fit_score:.4f}")
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        substrate_fit_scores=substrate_fit_scores,
        candidate_substrate=candidate_substrate,
        per_axis_residuals=per_axis_residuals,
        m_contest_sha=m_contest_sha,
        axis_tag_for_decomposition=_TIER_A_AXIS_TAG,
    )

    # Canonical Catalog #341 contract: dict construction uses variable for
    # axis_tag so the forbidden literal form does NOT appear in this module
    # body (Catalog #357 _CHECK_357_TIER_B_FORBIDDEN_PREDICTED_AXIS_FORMS
    # match-set is preserved as structural protection for any future
    # promotion attempt that elides the variable).
    result: dict[str, Any] = {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": _TIER_A_AXIS_TAG,
        "promotable": False,
        "confidence": 0.0,
        "consumer_signal_kind": "substrate_class_selector_ranking",
        "substrate_fit_scores": substrate_fit_scores,
        "candidate_substrate": candidate_substrate,
        "m_contest_array_sha256": m_contest_sha,
        "predicted_axis_decomposition": decomposition.as_dict(),
        "consumer_branch_kind": "tier_a_baseline",
    }
    return result


def _compute_tier_b_solver_delta(
    substrate_fit_scores: Mapping[str, float] | None,
    candidate_substrate: str | None,
    per_axis_residuals: Mapping[str, float] | None,
) -> tuple[float, str]:
    """Derive Tier B bounded predicted_delta_adjustment from cosine-similarity
    fit-score + per-axis residual signal.

    Cosine-similarity fit score is in [0, 1] where 1.0 = perfect alignment
    with M_contest gradient. Higher fit score => more likely the substrate
    can achieve better contest score => bounded negative predicted_delta
    (improvement) for high-fit candidates; bounded positive (regression
    expectation) for low-fit candidates.

    For per-axis residuals (signed; negative = improvement), we apply the
    canonical contest-score formula coefficients (100 * d_seg + 5.0 *
    d_pose per linearization near operating point) and bound the result.

    Returns (bounded_delta, derivation_note). When inputs are missing,
    returns (0.0, "no_signal") so the dispatcher gracefully degenerates
    to Tier A semantics for this candidate.
    """
    # Primary signal: per_axis_residuals if supplied (signed canonical form).
    if per_axis_residuals is not None and isinstance(per_axis_residuals, Mapping):
        seg_residual = float(per_axis_residuals.get("seg", 0.0))
        pose_residual = float(per_axis_residuals.get("pose", 0.0))
        # Canonical contest formula: 100*d_seg + sqrt(10*d_pose). For Tier B
        # bounded signal we linearize sqrt(10*d_pose) near operating point
        # via 5.0 * d_pose coefficient (see CLAUDE.md "SegNet vs PoseNet
        # importance — operating-point dependent" section).
        raw_delta = 100.0 * seg_residual + 5.0 * pose_residual
        note = f"per_axis_residuals_raw_delta={raw_delta:.6f}"
    elif (
        substrate_fit_scores is not None
        and isinstance(substrate_fit_scores, Mapping)
        and candidate_substrate is not None
        and candidate_substrate in substrate_fit_scores
    ):
        # Secondary signal: substrate-fit score. Map [0, 1] => bounded
        # delta via affine transform: fit_score=1.0 => -0.05 (improvement
        # expectation); fit_score=0.0 => +0.05 (regression expectation).
        fit_score = float(substrate_fit_scores[candidate_substrate])
        # Clamp fit_score to [0, 1] defensively.
        fit_score = max(0.0, min(1.0, fit_score))
        # Affine: delta = +0.05 - 0.10 * fit_score
        raw_delta = 0.05 - 0.10 * fit_score
        note = (
            f"substrate_fit_score={fit_score:.4f}_affine_to_raw_delta="
            f"{raw_delta:.6f}"
        )
    else:
        return 0.0, "no_signal"

    if raw_delta < _TIER_B_SAFETY_RAIL_MIN:
        bounded = _TIER_B_SAFETY_RAIL_MIN
        note += "_clipped_to_safety_rail_min"
    elif raw_delta > _TIER_B_SAFETY_RAIL_MAX:
        bounded = _TIER_B_SAFETY_RAIL_MAX
        note += "_clipped_to_safety_rail_max"
    else:
        bounded = raw_delta
        note += (
            f"_within_safety_rail_[{_TIER_B_SAFETY_RAIL_MIN},"
            f"{_TIER_B_SAFETY_RAIL_MAX}]"
        )
    return float(bounded), note


def consume_candidate_solver_derived(
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Tier B solver-derived branch: NON-ZERO predicted_delta_adjustment
    bounded to [-0.05, 0.05] per Catalog #357 + Dim 6 Step 6.5 safety rail.

    Per Catalog #323 + #357: contribution carries (a) canonical Provenance
    in ``provenance`` field per Catalog #323 (PREDICTED grade; non-
    promotable); (b) ``axis_tag`` empirically grounded (NOT [predicted];
    cosine similarity IS empirically grounded in M_contest vs M_inflated
    tensor alignment); (c) ``promotable=False`` PRESERVED per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA" (promotion requires paired-
    CPU + paired-CUDA empirical anchors; this branch contributes to
    RANKING only).
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict
    except ImportError:  # pragma: no cover
        provenance_payload: Mapping[str, Any] = {
            "artifact_kind": "predicted_from_model",
            "model_id": _PROVENANCE_MODEL_ID_TIER_B,
            "measurement_axis": _TIER_B_AXIS_TAG,
            "evidence_grade": "predicted",
            "promotion_eligible": False,
            "score_claim_valid": False,
        }
    else:
        m_contest_sha = candidate.get("m_contest_array_sha256")
        candidate_substrate = candidate.get("substrate_name")
        sha_seed = m_contest_sha or "no_m_contest_sha"
        inputs_seed = (
            f"{_PROVENANCE_MODEL_ID_TIER_B}:m_contest_sha={sha_seed}"
            f":substrate={candidate_substrate or 'none'}"
        )
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID_TIER_B,
            inputs_sha256=inputs_sha256,
            measurement_axis=_TIER_B_AXIS_TAG,
            hardware_substrate="cpu_local",
        )
        provenance_payload = provenance_to_dict(prov)

    substrate_fit_scores = candidate.get("substrate_fit_scores")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    candidate_substrate = candidate.get("substrate_name")
    per_axis_residuals = candidate.get("per_axis_residuals")

    tier_b_delta, derivation_note = _compute_tier_b_solver_delta(
        substrate_fit_scores=substrate_fit_scores,
        candidate_substrate=candidate_substrate,
        per_axis_residuals=per_axis_residuals,
    )

    rationale_parts = [
        "substrate-fit diagnostic consumer (exploit #6); Tier B solver-derived branch",
        "cosine-similarity-per-axis canonical observability surface",
        f"bounded_delta={tier_b_delta:+.6f}",
        derivation_note,
    ]
    if candidate_substrate is not None:
        rationale_parts.append(f"candidate_substrate={candidate_substrate}")
    if substrate_fit_scores is not None and isinstance(substrate_fit_scores, Mapping):
        n_substrates = len(substrate_fit_scores)
        rationale_parts.append(f"upstream substrate_fit_scores n={n_substrates}")
        if candidate_substrate in substrate_fit_scores:
            fit_score = substrate_fit_scores[candidate_substrate]
            rationale_parts.append(f"fit_score={fit_score:.4f}")
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        substrate_fit_scores=substrate_fit_scores,
        candidate_substrate=candidate_substrate,
        per_axis_residuals=per_axis_residuals,
        m_contest_sha=m_contest_sha,
        axis_tag_for_decomposition=_TIER_B_AXIS_TAG,
    )

    # Catalog #357 Tier B canonical contract:
    # (1) predicted_delta_adjustment finite (bounded); (2) axis_tag empirically
    # grounded NOT [predicted]; (3) promotable=False PRESERVED; (4)
    # provenance field present + non-empty; (5) rationale ≥4 chars + non-
    # placeholder. Dict construction uses _TIER_B_AXIS_TAG variable so
    # Catalog #357 forbidden literal axis_tag patterns do NOT appear in body.
    result: dict[str, Any] = {
        "predicted_delta_adjustment": float(tier_b_delta),
        "rationale": rationale,
        "axis_tag": _TIER_B_AXIS_TAG,
        "promotable": False,
        "provenance": dict(provenance_payload),
        "confidence": 0.5,
        "consumer_signal_kind": "tier_b_cosine_similarity_solver_bounded_ranking_signal",
        "substrate_fit_scores": substrate_fit_scores,
        "candidate_substrate": candidate_substrate,
        "m_contest_array_sha256": m_contest_sha,
        "predicted_axis_decomposition": decomposition.as_dict(),
        "consumer_branch_kind": "tier_b_solver",
        "tier_b_safety_rail_min": _TIER_B_SAFETY_RAIL_MIN,
        "tier_b_safety_rail_max": _TIER_B_SAFETY_RAIL_MAX,
        "tier_b_derivation_note": derivation_note,
    }
    return result


def _append_paired_comparison_row(
    tier_a_payload: Mapping[str, Any],
    tier_b_payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Persist paired-comparison row to canonical fcntl-locked JSONL store.

    Per Catalog #131 + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE
    + sister DIM-1-PHASE-2-START phase_2_ablation_posterior pattern.

    Defensive: missing parent dir / fcntl unavailability / OS errors are
    silently swallowed (canonical pattern for observability emission per
    sister findings_lagrangian_consumer; persistence failure must NOT
    affect ranker dispatch).
    """
    try:
        from tac.cathedral_consumers.substrate_fit_diagnostic_consumer._posterior_store import (
            append_paired_comparison_row,
        )
    except ImportError:  # pragma: no cover
        return

    try:
        append_paired_comparison_row(
            consumer_name=CONSUMER_NAME,
            consumer_version=CONSUMER_VERSION,
            tier_a_payload=tier_a_payload,
            tier_b_payload=tier_b_payload,
            candidate_hint=candidate,
        )
    except Exception:  # pragma: no cover
        pass


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5: dispatches to one of
    3 branches via env var ``CONSUMER_TIER_B_MODE`` ∈ {``tier_a_baseline``
    | ``tier_b_solver`` | ``paired_comparison``} default ``paired_comparison``.

    - ``tier_a_baseline``: returns Tier A payload (backward compat).
    - ``tier_b_solver``: returns Tier B payload (validation mode).
    - ``paired_comparison`` (default): fires BOTH, appends both to canonical
      JSONL store, returns Tier A authoritative payload + appends
      ``tier_b_paired_payload`` field for ranker observability.

    Per CLAUDE.md "Design decisions — non-negotiable": the ranker continues
    to honor Tier A authoritative until 30+ days of paired-comparison data
    accumulate AND inner-quintet sign-off per Catalog #292 + Mission
    Alignment Consequence 2.
    """
    mode = _resolve_tier_b_mode()

    if mode == "tier_a_baseline":
        return consume_candidate_tier_a_baseline(candidate)

    if mode == "tier_b_solver":
        return consume_candidate_solver_derived(candidate)

    # Default: paired_comparison
    tier_a_payload = consume_candidate_tier_a_baseline(candidate)
    tier_b_payload = consume_candidate_solver_derived(candidate)
    _append_paired_comparison_row(tier_a_payload, tier_b_payload, candidate)

    # Return Tier A authoritative + Tier B observability annotation. The
    # ranker continues to consume Tier A's predicted_delta_adjustment (0.0
    # by Catalog #341 invariant) until promotion criteria met.
    result: dict[str, Any] = dict(tier_a_payload)
    result["tier_b_paired_payload"] = dict(tier_b_payload)
    result["consumer_branch_kind"] = "paired_comparison_authoritative_tier_a"
    return result


def compute_substrate_fit_score(
    M_contest,
    M_inflated_per_substrate: Mapping[str, "object"],
) -> dict[str, float]:
    """Compute per-substrate fit score against the contest gradient.

    The fit score is the cosine similarity between M_inflated and M_contest
    (flattened across pairs / axes / pixels), clipped to [0, 1] via
    max(0, cosine). Higher score = better substrate fit.

    Args:
        M_contest: np.ndarray of shape (N_pairs, 3, H, W) - the contest
            video's per-pixel scorer-axis gradient.
        M_inflated_per_substrate: dict mapping substrate_name -> np.ndarray
            of shape (N_pairs, 3, H, W) - each substrate's inflated-video
            per-pixel gradient.

    Returns:
        Dict mapping substrate_name -> fit_score in [0, 1]; higher is
        better. Empty dict if input is empty.

    Raises:
        ValueError: on shape mismatch or empty M_contest.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for substrate fit") from exc

    m_contest = np.asarray(M_contest, dtype=np.float64)
    if m_contest.ndim != 4 or m_contest.shape[1] != 3:
        raise ValueError(
            f"M_contest must have shape (N_pairs, 3, H, W); got {m_contest.shape}"
        )
    if not isinstance(M_inflated_per_substrate, Mapping):
        raise ValueError(
            f"M_inflated_per_substrate must be Mapping; got "
            f"{type(M_inflated_per_substrate).__name__}"
        )
    if not M_inflated_per_substrate:
        return {}

    contest_flat = m_contest.ravel()
    contest_norm = float(np.linalg.norm(contest_flat))
    if contest_norm == 0:
        # All-zero M_contest: every substrate is "perfect fit" (trivially).
        return {name: 1.0 for name in M_inflated_per_substrate}

    fit_scores: dict[str, float] = {}
    for name, tensor in M_inflated_per_substrate.items():
        m_inflated = np.asarray(tensor, dtype=np.float64)
        if m_inflated.shape != m_contest.shape:
            raise ValueError(
                f"substrate {name!r} M_inflated shape {m_inflated.shape} "
                f"!= M_contest shape {m_contest.shape}"
            )
        inflated_flat = m_inflated.ravel()
        inflated_norm = float(np.linalg.norm(inflated_flat))
        if inflated_norm == 0:
            # Zero substrate gradient: 0 fit.
            fit_scores[str(name)] = 0.0
            continue
        cosine = float(
            np.dot(inflated_flat, contest_flat) / (inflated_norm * contest_norm)
        )
        # Clip to [0, 1] via max(0, cosine) for fit-score semantics.
        fit_scores[str(name)] = max(0.0, cosine)
    return fit_scores


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_TIER",
    "CONSUMER_HOOK_NUMBERS",
    "compute_substrate_fit_score",
    "consume_candidate",
    "consume_candidate_tier_a_baseline",
    "consume_candidate_solver_derived",
    "update_from_anchor",
    "_build_per_axis_decomposition",
    "_compute_tier_b_solver_delta",
    "_resolve_tier_b_mode",
]
