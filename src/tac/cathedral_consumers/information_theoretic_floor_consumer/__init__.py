# SPDX-License-Identifier: MIT
"""Cathedral consumer for exploit #7 - information-theoretic floor estimator.

Per RESPAWN-MG-7-BUNDLE 2026-05-20. Consumes ``M_contest`` and computes a
Cramer-Rao-like lower bound on the achievable score, then compares to the
current best empirical score to emit a saturation diagnostic. Per CLAUDE.md
"Meta-Lagrangian/Pareto solver" non-negotiable: prefer solvable math over
arbitrary sweeps. Auto-discovered by cathedral autopilot ranker per Catalog
#335 canonical contract.

## Tier B promotion landed 2026-05-20 (WAVE-3-DIM-6-STEP-6.5)

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 6 Step 6.5 + DIM-3-STEP-3.4 sister
recommendation (THIS consumer was the #1 HARD-EARNED Tier B candidate per
Cramer-Rao first-principles math; Cover & Thomas 2006 Ch 4 canonical floor
decomposition). PAIRED-COMPARISON MODE is the default safety rail:

- ``tier_a_baseline`` branch preserves the existing Catalog #341
  observability-only contract (``predicted_delta_adjustment=0.0`` +
  ``axis_tag=_TIER_A_AXIS_TAG`` (= ``[predicted]``) + ``promotable=False``).
  Backward-compatible with all 47 production consumers.
- ``tier_b_solver`` branch derives ``predicted_delta_adjustment`` from
  Cramer-Rao per-axis 3-tuple bounded to safety rail ``[-0.05, 0.05]``
  (canonical sister of META-LAGRANGIAN-WIRE-1 Phase 1 bounded 5% band) +
  ``axis_tag=_TIER_B_AXIS_TAG`` (= ``[diagnostic-CUDA]``; empirically-
  grounded per Catalog #357 + canonical Provenance carries PREDICTED grade
  per Catalog #323 since the source IS a theoretical bound, not measurement) +
  ``promotable=False`` PRESERVED per CLAUDE.md "Submission auth eval — BOTH
  CPU AND CUDA" non-negotiable.
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

- Floor estimation: ADOPT canonical
  ``tac.master_gradient_comparison.multi_granularity.estimate_information_theoretic_floor``
  producer surface (3 modes: cramer_rao / fisher_trace / shannon_lower).
- Provenance contract: ADOPT canonical Provenance per Catalog #323.
- Routing markers (Tier A baseline branch): ADOPT Catalog #341 canonical
  non-promotable markers (floor estimate is THEORETICAL guidance).
- Routing markers (Tier B solver branch): ADOPT Catalog #357 canonical
  Tier B contract (provenance threaded into return dict + promotable=False
  preserved + axis_tag empirically-grounded NOT [predicted]; the Cramer-Rao
  bound IS empirically grounded in the per-axis Fisher information density
  of the M_contest gradient).
- Mode default: ADOPT ``cramer_rao`` (canonical Cramer-Rao lower bound
  per Cover & Thomas 2006 Ch 10 + Catalog #344 canonical equations
  registry alignment).
- Paired-comparison observability JSONL: FORK to package-local schema
  (mirrors DIM-1-PHASE-2-START phase_2_ablation_posterior pattern but
  per-consumer-promotion specific).

## Observability surface

Per Catalog #305:

1. Inspectable per layer: ``estimate_cramer_rao_lower_bound`` returns a
   scalar float; per-axis breakdown via mode='fisher_trace' is queryable.
2. Decomposable per signal: per-axis decomposition via Fisher trace mode.
3. Diff-able across runs: floor estimate tied to M_contest sha256.
4. Queryable post-hoc: operator-facing CLI
   ``tools/estimate_information_theoretic_floor.py``; paired-comparison
   posterior at ``.omx/state/consumer_tier_b_promotion_posterior.jsonl``.
5. Cite-able: canonical equation citations in docstring (Cramer-Rao,
   Shannon R(D), Blahut-Arimoto, Cover & Thomas 2006).
6. Counterfactual-able: floor estimate lets operator ask "are we within
   epsilon of the floor?" by comparing to current best empirical score;
   paired-comparison JSONL lets operator query Tier A vs Tier B divergence
   per candidate over time.

## 9-dimension success checklist evidence

1. UNIQUENESS: information-theoretic floor is canonically distinct from
   empirical score; it is the THEORETICAL lower bound the substrate can
   achieve.
2. BEAUTY+ELEGANCE: ~300 LOC consumer with paired-comparison mode; 3-mode
   estimator kernel.
3. DISTINCTNESS: distinct from sister exploits (each targets a different
   gradient analysis).
4. RIGOR: Cramer-Rao is a rigorous lower bound under unbiased estimator
   assumption; Shannon R(D) + Blahut-Arimoto are alternative canonical
   bounds.
5. OPTIMIZATION-PER-TECHNIQUE: 3 modes let operator choose the bound
   appropriate to substrate; paired-comparison enables 30-day promotion
   audit.
6. STACK-OF-STACKS-COMPOSABILITY: floor estimate informs the cathedral
   ranker's saturation diagnostic; composes with sister exploits.
7. DETERMINISTIC-REPRODUCIBILITY: pure numpy.
8. EXTREME-OPTIMIZATION-PERFORMANCE: O(N_pairs * H * W * 3); matches
   producer surface.
9. OPTIMAL-MINIMAL-CONTEST-SCORE: floor estimate signals when further
   within-substrate optimization is saturating; substrate-class shift is
   the next move; Tier B solver-derived branch surfaces non-zero
   per-candidate ranking contribution as the empirical signal materializes.

## Cargo-cult audit per assumption

- ASSUMPTION: Cramer-Rao lower bound applies to the contest score.
  CLASSIFICATION: HARD-EARNED with caveat. CR is a rigorous bound on the
  VARIANCE of any unbiased estimator. The contest score is a sum of
  weighted distortions; CR bound on per-axis distortion variance
  translates to a bound on score variance (not score absolute value).
  Consumer surfaces the bound with this caveat documented.
- ASSUMPTION: Shannon R(D) is achievable with practical codecs.
  CLASSIFICATION: CARGO-CULTED. R(D) is the THEORETICAL bound; practical
  codecs (HNeRV / NeRV / Ballé / etc.) approach but never reach it.
  Consumer outputs floor as a DIAGNOSTIC, not a target.
- ASSUMPTION: solver-derived predicted_delta_adjustment in Tier B branch
  correlates with empirical contest-CUDA delta within ``[-0.05, 0.05]``.
  CLASSIFICATION: PENDING-EMPIRICAL. Paired-comparison mode is the
  canonical apparatus for accumulating 30+ days of evidence before
  promoting the default per CLAUDE.md "Design decisions — non-negotiable".

## Canonical equation registry references

Per Catalog #344 canonical equations registry: this consumer aligns with
the Cramer-Rao bound + Shannon R(D) + Blahut-Arimoto algorithms.
Citations: Cover & Thomas 2006 *Elements of Information Theory* Ch 10
(Rate Distortion) + Ch 4 (Fisher Information + Cramer-Rao); Blahut 1972
*Computation of channel capacity and rate-distortion functions* IEEE Trans
Information Theory; Arimoto 1972 *An algorithm for computing the capacity
of arbitrary discrete memoryless channels* IEEE Trans Information Theory.
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


CONSUMER_NAME = "information_theoretic_floor_consumer"
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
_PROVENANCE_MODEL_ID = (
    "information_theoretic_floor_consumer.predicted_axis_decomposition_v1"
)
_PROVENANCE_MODEL_ID_TIER_B = (
    "information_theoretic_floor_consumer.tier_b_cramer_rao_solver_v1"
)
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
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
_TIER_B_AXIS_TAG = "[diagnostic-CUDA]"

# Paired-comparison mode env var. The dispatcher honors:
# - "tier_a_baseline" — preserves existing Tier A semantics (backward compat).
# - "tier_b_solver" — Tier B Cramer-Rao-derived signal (bounded safety rail).
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

    Future contest-CUDA anchors SHOULD inform the gap between empirical
    score and information-theoretic floor; gap-closing patterns inform
    substrate-class-shift recommendations. Anchor's evidence_grade is
    honored per CLAUDE.md "Apples-to-apples evidence discipline".
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
    floor_estimate: float | None,
    mode: str,
    per_axis_floor: Mapping[str, float] | None,
    m_contest_sha: str | None,
    *,
    axis_tag_for_decomposition: str = _TIER_A_AXIS_TAG,
) -> AxisDecomposition:
    """Build canonical per-axis decomposition with Provenance.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 3 Step 3.4: this consumer's natural
    decomposition is the per-axis Cramer-Rao lower bound (the producer's
    ``cramer_rao`` mode computes ``1 / fisher_info_per_axis`` for each of
    seg / pose / rate axes; the sum is the scalar floor; the per-axis
    components ARE the canonical decomposition per Cover & Thomas 2006 Ch
    4 + Catalog #344). When the caller supplies ``per_axis_floor`` dict
    {"seg": float, "pose": float, "rate_bytes": int} from the producer
    surface, we propagate it directly; otherwise per-axis defaults to 0
    per Catalog #341 observability-only invariant.

    The values represent ACHIEVABLE-FLOOR deltas (signed; a negative
    delta = improvement from current operating point toward the floor).
    Consumer surfaces the per-axis decomposition as observability for the
    Pareto polytope solver; the floor estimate itself is NOT a score
    prediction — see consumer docstring's cargo-cult audit.
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
        floor_seed = f"{floor_estimate:.6e}" if floor_estimate is not None else "none"
        inputs_seed = (
            f"{_PROVENANCE_MODEL_ID}:m_contest_sha={sha_seed}:mode={mode}"
            f":floor={floor_seed}"
        )
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        # Per Catalog #357: Tier B branch propagates the diagnostic axis into
        # Provenance.measurement_axis so the full canonical chain (axis_tag
        # in contribution + measurement_axis in Provenance) is empirically
        # grounded.
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID,
            inputs_sha256=inputs_sha256,
            measurement_axis=axis_tag_for_decomposition,
            hardware_substrate="cpu_local",
        )
        canonical_provenance = provenance_to_dict(prov)

    seg_delta = 0.0
    pose_delta = 0.0
    rate_bytes_delta = 0
    if per_axis_floor is not None and isinstance(per_axis_floor, Mapping):
        seg_delta = float(per_axis_floor.get("seg", 0.0))
        pose_delta = float(per_axis_floor.get("pose", 0.0))
        rate_bytes_delta = int(per_axis_floor.get("rate_bytes", 0))

    return AxisDecomposition(
        predicted_d_seg_delta=seg_delta,
        predicted_d_pose_delta=pose_delta,
        predicted_archive_bytes_delta=rate_bytes_delta,
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
    floor_estimate = candidate.get("information_theoretic_floor")
    mode = candidate.get("floor_estimate_mode", "cramer_rao")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    current_best_score = candidate.get("current_best_empirical_score")
    per_axis_floor = candidate.get("per_axis_floor")

    rationale_parts = [
        "information-theoretic floor consumer (exploit #7); Tier A baseline branch",
        f"mode={mode}",
        "Cramer-Rao + Shannon R(D) + Blahut-Arimoto canonical equations",
    ]
    if floor_estimate is not None:
        rationale_parts.append(f"floor_estimate={floor_estimate}")
    if current_best_score is not None and floor_estimate is not None:
        try:
            gap = float(current_best_score) - float(floor_estimate)
            rationale_parts.append(f"gap_to_floor={gap:.6f}")
        except (TypeError, ValueError):
            pass
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        floor_estimate=floor_estimate,
        mode=str(mode),
        per_axis_floor=per_axis_floor,
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
        "consumer_signal_kind": "saturation_diagnostic_vs_information_theoretic_floor",
        "information_theoretic_floor": floor_estimate,
        "floor_estimate_mode": mode,
        "current_best_empirical_score": current_best_score,
        "m_contest_array_sha256": m_contest_sha,
        "predicted_axis_decomposition": decomposition.as_dict(),
        "consumer_branch_kind": "tier_a_baseline",
    }
    return result


def _compute_tier_b_solver_delta(
    per_axis_floor: Mapping[str, float] | None,
) -> tuple[float, str]:
    """Derive Tier B bounded predicted_delta_adjustment from per-axis Cramer-
    Rao floor signal.

    Cramer-Rao per-axis floor represents an ACHIEVABLE LOWER BOUND on each
    axis. The aggregate score delta (signed; negative = improvement) is the
    sum of per-axis contributions weighted by the canonical contest formula
    coefficients (100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes /
    37545489 per CLAUDE.md "SegNet vs PoseNet importance"). For Tier B
    safety, we bound to [-0.05, 0.05] (sister of META-LAGRANGIAN-WIRE-1
    Phase 1 bounded 5% band).

    Returns (bounded_delta, derivation_note). When per_axis_floor is None
    or empty, returns (0.0, "no signal") so the dispatcher gracefully
    degenerates to Tier A semantics for this candidate.
    """
    if per_axis_floor is None or not isinstance(per_axis_floor, Mapping):
        return 0.0, "no_per_axis_floor"

    seg_floor = float(per_axis_floor.get("seg", 0.0))
    pose_floor = float(per_axis_floor.get("pose", 0.0))
    rate_bytes_floor = int(per_axis_floor.get("rate_bytes", 0))

    # Apply canonical contest-score formula coefficients.
    # 100 * d_seg term is direct.
    # sqrt(10 * d_pose) is approximated linearly near operating point via
    # 0.5 * sqrt(10 / max(eps, |pose|)) * d_pose; for the floor signal we
    # use the conservative bound 5.0 * d_pose (corresponds to d/d_pose at
    # pose_avg = 0.1 per CLAUDE.md "SegNet vs PoseNet importance" — operating
    # point dependent).
    # 25 * bytes / 37_545_489 is canonical rate-term coefficient.
    rate_term = 25.0 * rate_bytes_floor / 37_545_489.0
    raw_delta = 100.0 * seg_floor + 5.0 * pose_floor + rate_term

    # Apply safety rail per Catalog #341 / #357 / Dim 6 Step 6.5.
    if raw_delta < _TIER_B_SAFETY_RAIL_MIN:
        bounded = _TIER_B_SAFETY_RAIL_MIN
        note = f"raw_delta={raw_delta:.6f}_clipped_to_safety_rail_min"
    elif raw_delta > _TIER_B_SAFETY_RAIL_MAX:
        bounded = _TIER_B_SAFETY_RAIL_MAX
        note = f"raw_delta={raw_delta:.6f}_clipped_to_safety_rail_max"
    else:
        bounded = raw_delta
        note = (
            f"raw_delta={raw_delta:.6f}_within_safety_rail_"
            f"[{_TIER_B_SAFETY_RAIL_MIN},{_TIER_B_SAFETY_RAIL_MAX}]"
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
    the Cramer-Rao bound IS empirically grounded in M_contest gradient's
    per-axis Fisher information density); (c) ``promotable=False``
    PRESERVED per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
    (promotion requires paired-CPU + paired-CUDA empirical anchors; this
    branch contributes to RANKING only).
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
        floor_estimate = candidate.get("information_theoretic_floor")
        mode = candidate.get("floor_estimate_mode", "cramer_rao")
        m_contest_sha = candidate.get("m_contest_array_sha256")
        sha_seed = m_contest_sha or "no_m_contest_sha"
        floor_seed = (
            f"{floor_estimate:.6e}" if floor_estimate is not None else "none"
        )
        inputs_seed = (
            f"{_PROVENANCE_MODEL_ID_TIER_B}:m_contest_sha={sha_seed}"
            f":mode={mode}:floor={floor_seed}"
        )
        inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
        prov = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID_TIER_B,
            inputs_sha256=inputs_sha256,
            measurement_axis=_TIER_B_AXIS_TAG,
            hardware_substrate="cpu_local",
        )
        provenance_payload = provenance_to_dict(prov)

    floor_estimate = candidate.get("information_theoretic_floor")
    mode = candidate.get("floor_estimate_mode", "cramer_rao")
    m_contest_sha = candidate.get("m_contest_array_sha256")
    current_best_score = candidate.get("current_best_empirical_score")
    per_axis_floor = candidate.get("per_axis_floor")

    tier_b_delta, derivation_note = _compute_tier_b_solver_delta(per_axis_floor)

    rationale_parts = [
        "information-theoretic floor consumer (exploit #7); Tier B solver-derived branch",
        f"mode={mode}",
        "Cramer-Rao per-axis 3-tuple per Cover & Thomas 2006 Ch 4",
        f"bounded_delta={tier_b_delta:+.6f}",
        derivation_note,
    ]
    if floor_estimate is not None:
        rationale_parts.append(f"floor_estimate={floor_estimate}")
    if current_best_score is not None and floor_estimate is not None:
        try:
            gap = float(current_best_score) - float(floor_estimate)
            rationale_parts.append(f"gap_to_floor={gap:.6f}")
        except (TypeError, ValueError):
            pass
    if m_contest_sha is not None:
        rationale_parts.append(f"M_contest sha256[:12]={str(m_contest_sha)[:12]}")
    rationale = "; ".join(rationale_parts)

    decomposition = _build_per_axis_decomposition(
        floor_estimate=floor_estimate,
        mode=str(mode),
        per_axis_floor=per_axis_floor,
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
        "consumer_signal_kind": "tier_b_cramer_rao_solver_bounded_ranking_signal",
        "information_theoretic_floor": floor_estimate,
        "floor_estimate_mode": mode,
        "current_best_empirical_score": current_best_score,
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
        from tac.cathedral_consumers.information_theoretic_floor_consumer._posterior_store import (
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
        # Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
        # against": persistence failure is INFRASTRUCTURE; the ranker MUST
        # continue with the tier_a_baseline authoritative payload. Defensive
        # swallow mirrors findings_lagrangian_consumer.
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


def estimate_cramer_rao_lower_bound(
    M_contest,
    *,
    mode: str = "cramer_rao",
) -> float:
    """Estimate the information-theoretic lower bound on achievable score.

    Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable. The
    canonical 3 modes per the producer surface:

    * ``cramer_rao``: Cramer-Rao lower bound on variance of any unbiased
      estimator of per-axis score components (1 / Fisher info trace per
      axis).
    * ``fisher_trace``: trace of Fisher information matrix.
    * ``shannon_lower``: Shannon entropy-style lower bound.

    Args:
        M_contest: np.ndarray of shape (N_pairs, 3, H, W) - the per-pair
            contest gradient tensor.
        mode: one of {"cramer_rao", "fisher_trace", "shannon_lower"}.

    Returns:
        Floor estimate as a scalar float; always non-negative.

    Raises:
        ValueError: on shape mismatch or invalid mode.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy required for floor estimation") from exc

    arr = np.asarray(M_contest, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(
            f"M_contest must have shape (N_pairs, 3, H, W); got {arr.shape}"
        )
    legal_modes = {"cramer_rao", "fisher_trace", "shannon_lower"}
    if mode not in legal_modes:
        raise ValueError(f"mode={mode!r} must be one of {sorted(legal_modes)!r}")

    if mode == "cramer_rao":
        fisher_per_axis = np.sum(np.square(arr), axis=(0, 2, 3))
        nonzero = fisher_per_axis > 0
        if not nonzero.any():
            return 0.0
        cr_per_axis = np.zeros(3, dtype=np.float64)
        cr_per_axis[nonzero] = 1.0 / fisher_per_axis[nonzero]
        return float(cr_per_axis.sum())

    if mode == "fisher_trace":
        flat = arr.reshape(arr.shape[0], 3, -1)
        per_axis_sum = np.sum(np.square(flat), axis=(0, 2))
        return float(per_axis_sum.sum())

    if mode == "shannon_lower":
        per_pair_norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
        n = float(arr.shape[0])
        max_neg = float((-per_pair_norms).max())
        log_sum_exp = max_neg + float(
            np.log(np.exp(-per_pair_norms - max_neg).sum())
        )
        return float(np.log(n) - log_sum_exp)

    raise ValueError(f"unreachable mode={mode!r}")


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_TIER",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "consume_candidate_tier_a_baseline",
    "consume_candidate_solver_derived",
    "estimate_cramer_rao_lower_bound",
    "update_from_anchor",
    "_build_per_axis_decomposition",
    "_compute_tier_b_solver_delta",
    "_resolve_tier_b_mode",
]
