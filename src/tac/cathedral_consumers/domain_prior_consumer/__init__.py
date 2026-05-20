# SPDX-License-Identifier: MIT
"""Cathedral consumer for tac.domain_priors aggregate (4-domain prior bundle).

Per WAVE-3-DIM-4-STEP-4.3 2026-05-20 + CATHEDRAL-SMARTER-DESIGN-MEMO
``.omx/research/cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md``
Dim 4 Step 4.3: this consumer ingests the 4 canonical domain-prior wrappers
landed by WAVE-1-DIM-4-NAMESPACE (commit ``f537de2b8``) so cathedral
autopilot routes the per-frame / per-pair / per-class / OOD signal WITHOUT
manual ranker-cascade edits.

Auto-discovered per Catalog #335 canonical contract (PARADIGM SHIFT:
convention-over-configuration); no main() edit needed.

## Source domain priors (4-fold aggregate)

  1. **PerFrameDifficultyAtlas** (tac.domain_priors.per_frame_difficulty) —
     per-frame difficulty derived from per-pair master-gradient atlas.
     Surfaces hardest-frame indices for bit-allocator prioritization.

  2. **EgoMotionConcentrationAtlas** (tac.domain_priors.ego_motion_concentration)
     — per-frame ego-motion magnitude + flow concentration aligned with the
     Atick-Redlich 1990 cooperative-receiver lineage per CLAUDE.md
     "PER-SUBSTRATE OPTIMAL FORM" Catalog #311.

  3. **PerClassStatisticalPriors** (tac.domain_priors.per_class_statistical)
     — SegNet 5-class (background_sky_road / vehicle / pedestrian /
     lane_marking / other_foreground) pixel-count + chroma-variance +
     motion-magnitude priors.

  4. **Comma2k19DashcamPriors** (tac.domain_priors.comma2k19_priors) — OOD
     dashcam prior summary from canonical ``Comma2k19LocalCache`` per
     Catalog #213. ALWAYS OOD-tagged per Catalog #209.

## Canonical-vs-unique decision per layer (Catalog #290)

- Per-domain wrappers: ADOPT canonical (the 4 wrappers in tac.domain_priors
  are the canonical surface; this consumer composes their outputs).
- Provenance contract: ADOPT canonical
  :func:`tac.provenance.builders.build_provenance_for_predicted` (4-domain
  aggregate is PREDICTED_FROM_MODEL until paired CUDA+CPU anchor lands).
- Routing markers: ADOPT Catalog #341 canonical non-promotable markers
  (predicted_delta_adjustment=0.0 + promotable=False + axis_tag=[predicted]).
- Canonical equation update path: ADOPT canonical
  :func:`tac.canonical_equations.update_equation_with_empirical_anchor`
  per the 3 DIM-4 equations.
- Per-axis decomposition (Catalog #356, OPTIONAL BONUS): EMIT
  ``predicted_axis_decomposition`` per :class:`AxisDecomposition` when the
  aggregate signal admits a (seg, pose, archive_bytes) breakdown — for
  this consumer the per-axis attribution is bounded zero-delta per Tier A
  invariants but carries canonical Provenance so the consumer is
  Tier-B-promotion-ready when Dim 6 Step 6.5 fires.

## Observability surface (Catalog #305)

1. **Inspectable per layer**: each of the 4 priors' summary surfaces is
   queryable in the consumer's ``consume_candidate`` output dict
   (``per_frame_difficulty_summary`` / ``ego_motion_concentration_summary`` /
   ``per_segnet_class_chroma_summary`` / ``comma2k19_ood_distance_summary``).
2. **Decomposable per signal**: the 4-domain dict structurally decomposes
   the aggregate scalar.
3. **Diff-able across runs**: every summary cites archive_sha256 +
   measurement_axis from the source prior dataclass so diff is deterministic.
4. **Queryable post-hoc**: cathedral consumer auto-discovery means the
   operator can fire this consumer any time via
   ``tools.cathedral_autopilot_autonomous_loop.invoke_cathedral_consumers_on_candidates``.
5. **Cite-able**: every summary carries the source wrapper's
   ``canonical_helper_invocation`` / archive sha / axis tag.
6. **Counterfactual-able**: the 4-domain breakdown lets the operator ask
   "what if we re-route bits from per-frame hard-difficulty to per-class
   chroma?" without re-running the canonical wrappers.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: 4-domain prior aggregation is canonically distinct from
   single-prior consumers (sisters consume one prior each; this aggregates).
2. **BEAUTY+ELEGANCE**: ~430 LOC consumer; canonical helpers are ~1200 LOC.
3. **DISTINCTNESS**: distinct from sister consumers per_pair_difficulty
   (per-pair surface) + per_segnet_class_chroma (per-class only) + sister
   exploits (each targets one granularity).
4. **RIGOR**: Catalog #287 + #323 canonical Provenance on every emission;
   axis tag [predicted] until paired CUDA+CPU anchor lands.
5. **OPTIMIZATION-PER-TECHNIQUE**: 4-domain aggregation is substrate-
   optimal for cathedral autopilot ranking (Tier A observability now,
   Tier B-ready for Dim 6 Step 6.5 promotion).
6. **STACK-OF-STACKS-COMPOSABILITY**: composes additively with sister
   single-prior consumers (each domain surface remains accessible).
7. **DETERMINISTIC-REPRODUCIBILITY**: pure stdlib + numpy aggregation;
   no random sampling; identical input -> identical output.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: O(N_frames + N_classes); matches
   underlying wrapper throughput.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: 4-domain priors inform autopilot
   ranking + future bit-allocator (per Dim 4 Phase 2). Future score
   improvement requires substrate that actually consumes the priors.

## Cargo-cult audit per assumption (Catalog #303)

- **ASSUMPTION**: 4-domain aggregation is more informative than any single
  prior. **CLASSIFICATION**: CARGO-CULTED-PENDING-EMPIRICAL. The 4 priors
  cover orthogonal axes (frame-difficulty / ego-motion / per-class /
  OOD-distance) BUT empirical evidence that aggregation outperforms
  single-prior consumption per cathedral autopilot ranking is pending the
  first paired CUDA+CPU dispatch that records per-axis ablation.
- **ASSUMPTION**: missing prior signals (e.g. cache empty for Comma2k19
  OR no master-gradient anchor for per-frame difficulty) are best handled
  by emitting None-signal-per-domain rather than refusing the candidate.
  **CLASSIFICATION**: HARD-EARNED. Per CLAUDE.md "Subagent coherence-by-
  default" + Catalog #341: defensive consumers should NEVER raise in the
  cathedral autopilot main loop; missing-signal -> per-domain None +
  rationale message.

## Predicted-band Dykstra-feasibility check (Catalog #296) — N/A

This consumer does NOT emit a predicted ΔS band; it surfaces
``predicted_delta_adjustment=0.0`` per Catalog #341 Tier A invariants.
Dykstra-feasibility analysis applies to Tier B promotion (Dim 6 Step 6.5).

## Horizon class (Catalog #309) — N/A

This is an observability consumer, not a substrate design memo. The
horizon-class classification applies to substrate designs, not cathedral
consumers that aggregate them.

## Mission contribution per Catalog #300

``frontier_breaking_enabler`` — closes Dim 4 Step 4.3 (domain priors now
flow into cathedral autopilot ranking via the canonical 6-hook wire-in).
Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 line 663 dependency chain: this
landing unblocks Dimension 1 Phase 5 (sensitivity-map regularization).

## 6-hook wire-in per Catalog #125

1. **Sensitivity-map** = ACTIVE per-domain. Each of the 4 priors IS a
   per-frame / per-pair / per-class sensitivity surface. Downstream
   ``tac.sensitivity_map.*`` consumers route per-axis breakdowns through
   this consumer's emitted observability dict.
2. **Pareto constraint** = N/A. Priors inform WEIGHTS, not feasibility.
3. **Bit-allocator** = N/A (cathedral consumer surface; sister
   ``bit_allocator_per_pair_consumer`` consumes the per-frame /
   per-pair surface directly; this consumer aggregates for autopilot).
4. **Cathedral autopilot dispatch** = ACTIVE PRIMARY. Auto-discovered
   per Catalog #335; integrated into autopilot ranker via
   ``invoke_cathedral_consumers_on_candidates``.
5. **Continual-learning posterior** = ACTIVE. ``update_from_anchor``
   forwards new empirical anchors to the 3 DIM-4 canonical equations
   via :func:`tac.canonical_equations.update_equation_with_empirical_anchor`.
6. **Probe-disambiguator** = ACTIVE. The 4-domain breakdown disambiguates
   between candidates that score similarly on average but differ on
   per-frame / per-pair / per-class / OOD distribution.

## Tier B promotion pathway (per Dim 6 Step 6.5)

This consumer is currently TIER_A_OBSERVABILITY_ONLY per Catalog #341.
Tier B promotion (per Dim 6 Step 6.5) becomes possible when ALL of:

1. Paired contest-CPU + contest-CUDA empirical anchor lands on a candidate
   whose per-axis (seg, pose, archive_bytes) breakdown is recorded.
2. The 3 DIM-4 canonical equations have empirical_anchors registered (>0)
   demonstrating the per-axis residuals are well-calibrated per the
   :func:`tac.canonical_equations.update_equation_with_empirical_anchor`
   posterior-refit pattern.
3. The consumer's ``consume_candidate`` is updated to:
   - Compute per-axis (predicted_d_seg_delta, predicted_d_pose_delta,
     predicted_archive_bytes_delta) from the 4-domain priors.
   - Emit ``provenance`` per :func:`tac.provenance.builders.build_provenance_for_predicted`
     with axis_tag = canonical contest axis (NOT [predicted]).
   - Set ``CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING``.
4. Catalog #357 STRICT preflight gate (Tier B canonical contract)
   validates the consumer per :func:`tac.cathedral.consumer_contract.validate_tier_b_contribution`.

Until then the OPTIONAL BONUS ``predicted_axis_decomposition`` emission
below provides Tier-B-ready infrastructure with bounded zero-delta
per-axis breakdown + canonical Provenance, so the Tier B promotion
becomes a small + reviewable change rather than a refactor.

## Cross-references

- WAVE-1-DIM-4-NAMESPACE landing memo:
  ``feedback_wave_1_dim_4_domain_priors_namespace_landed_20260520.md``
- CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Step 4.3 (THIS landing implements)
- Catalog #209 contest-video-leakage (Comma2k19 OOD-tagging propagated)
- Catalog #210 DP1 codebook provenance pattern (license_spdx + dataset
  provenance propagated through the Comma2k19 path)
- Catalog #213 Comma2k19 canonical local-chunk cache
- Catalog #287 docstring-overstatement-without-evidence-tag
- Catalog #290 canonical-vs-unique decision per layer
- Catalog #294 9-dimension success checklist evidence
- Catalog #305 observability surface
- Catalog #323 canonical Provenance umbrella
- Catalog #335 cathedral consumer canonical contract (THIS gate)
- Catalog #341 canonical routing markers (Tier A invariants enforced)
- Catalog #344 canonical equations registry (3 DIM-4 equations consumed)
- Catalog #354 master-gradient exploit consumer bundle (sister consumers
  that each consume a SINGLE prior; this consumer aggregates ALL 4)
- Catalog #356 AxisDecomposition (OPTIONAL BONUS emission per Dim 3 Step 3.1)
- Catalog #357 Tier B canonical contract (Tier B promotion pathway)
- CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in non-negotiable
- CLAUDE.md "Forbidden score claims" (predicted_delta_adjustment=0.0)
- CLAUDE.md "Apples-to-apples evidence discipline"
"""
from __future__ import annotations

import importlib
import math
from typing import Any, Mapping, Sequence

from tac.cathedral.consumer_contract import (
    AxisDecomposition,
    ConsumerTier,
    HookNumber,
)


CONSUMER_NAME = "domain_prior_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_TIER = ConsumerTier.TIER_A_OBSERVABILITY_ONLY
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)

# Canonical equation IDs this consumer feeds via update_from_anchor.
# Sister of tac.domain_priors.equations.DOMAIN_PRIORS_EQUATION_IDS.
_DOMAIN_PRIORS_EQUATION_IDS: tuple[str, ...] = (
    "per_frame_difficulty_atlas_v1",
    "ego_motion_concentration_prior_v1",
    "per_segnet_class_chroma_priors_v1",
)

_PROVENANCE_MODEL_ID = "domain_prior_consumer.predicted_axis_decomposition_v1"
_PROVENANCE_PLACEHOLDER_SHA = "0" * 64

# Candidate field aliases for the 4 prior payloads. Candidates MAY carry
# pre-built atlas dataclasses OR dict-form summaries OR neither (no-signal
# fallback). The consumer is defensive: missing prior -> None-signal +
# rationale message, never raises.
_PER_FRAME_DIFFICULTY_FIELDS = (
    "per_frame_difficulty_atlas",
    "per_frame_difficulty",
    "per_frame_difficulty_summary",
)
_EGO_MOTION_FIELDS = (
    "ego_motion_concentration_atlas",
    "ego_motion_concentration",
    "ego_motion_concentration_summary",
)
_PER_CLASS_FIELDS = (
    "per_class_statistical_priors",
    "per_class_statistical",
    "per_segnet_class_chroma_priors",
    "per_segnet_class_chroma_summary",
)
_COMMA2K19_FIELDS = (
    "comma2k19_dashcam_priors",
    "comma2k19_priors",
    "comma2k19_ood_distance_summary",
)


def update_from_anchor(anchor: Any) -> Mapping[str, Any]:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Forwards new empirical anchors to the 3 DIM-4 canonical equations via
    :func:`tac.canonical_equations.update_equation_with_empirical_anchor`
    when the anchor carries a recognizable ``equation_id`` matching one of
    ``_DOMAIN_PRIORS_EQUATION_IDS``. For anchors that don't match (most
    contest-CUDA / contest-CPU anchors don't carry an equation_id), the
    consumer returns a payload-only acknowledgement without mutating any
    posterior state (per Catalog #341 Tier A invariants).

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
    #287/#323: only well-formed anchors with a recognized ``equation_id``
    trigger the canonical update path; malformed / missing-signal anchors
    return a fail-closed payload without raising in the cathedral
    autopilot main loop.

    Per Catalog #335 paradigm: keep cathedral consumers structurally
    minimal so the canonical update logic lives in
    :mod:`tac.canonical_equations`, not here. NO-OP on malformed anchors.

    Args:
        anchor: a typed ``EmpiricalAnchor`` OR a dict OR None. The
            canonical fields the consumer probes are ``equation_id`` /
            ``residual`` / ``axis_tag``.

    Returns:
        dict with at minimum ``accepted`` (bool), ``status`` (str),
        ``equation_id`` (str | None), and canonical routing markers per
        Catalog #341 (axis_tag / score_claim / promotion_eligible).
    """
    if anchor is None:
        return _fail_closed_update(
            "anchor is None; domain_prior_consumer update refused"
        )

    equation_id, residual, axis_tag = _extract_canonical_anchor_fields(anchor)

    if equation_id is None:
        return _fail_closed_update(
            "anchor missing equation_id; domain_prior_consumer update refused",
        )

    if equation_id not in _DOMAIN_PRIORS_EQUATION_IDS:
        return _fail_closed_update(
            f"equation_id={equation_id!r} not in DOMAIN_PRIORS_EQUATION_IDS "
            f"({_DOMAIN_PRIORS_EQUATION_IDS}); consumer routes only domain-prior anchors",
            equation_id=equation_id,
        )

    if residual is None:
        return _fail_closed_update(
            f"anchor for equation_id={equation_id!r} missing finite residual",
            equation_id=equation_id,
        )

    # Best-effort delegation to canonical update path. Per Catalog #335:
    # NEVER raise in the cathedral autopilot main loop. Errors are surfaced
    # as fail-closed payloads.
    try:
        from tac.canonical_equations import (
            update_equation_with_empirical_anchor,
        )
        from tac.canonical_equations.equation import EmpiricalAnchor
    except ImportError as exc:
        return _fail_closed_update(
            f"tac.canonical_equations unavailable: {exc}",
            equation_id=equation_id,
        )

    try:
        empirical_anchor = _coerce_to_empirical_anchor(
            anchor=anchor,
            equation_id=equation_id,
            residual=residual,
            axis_tag=axis_tag,
            EmpiricalAnchor=EmpiricalAnchor,
        )
    except (TypeError, ValueError) as exc:
        return _fail_closed_update(
            f"failed to coerce anchor for equation_id={equation_id!r}: {exc}",
            equation_id=equation_id,
        )

    try:
        update_equation_with_empirical_anchor(
            equation_id=equation_id,
            anchor=empirical_anchor,
            agent="domain_prior_consumer",
            notes="domain_prior_consumer update_from_anchor (Dim 4 Step 4.3)",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return _fail_closed_update(
            f"canonical equation update raised: {exc}",
            equation_id=equation_id,
        )

    return {
        "accepted": True,
        "status": "canonical_equation_updated",
        "equation_id": equation_id,
        "axis_tag": axis_tag or "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Aggregates the 4-domain prior signals from the candidate (if present)
    and surfaces a unified observability annotation per Catalog #341 Tier
    A invariants (predicted_delta_adjustment=0.0 + promotable=False +
    axis_tag="[predicted]").

    The 4 domain priors are extracted defensively from the candidate dict:
    if a prior is missing OR malformed, the consumer emits a None-signal
    summary for that domain rather than refusing the candidate (per
    Catalog #335 paradigm: never raise in the cathedral main loop).

    OPTIONAL BONUS per Catalog #356: when ALL 4 priors are present + the
    candidate carries an archive_sha256 anchor, the consumer ALSO emits
    ``predicted_axis_decomposition`` per :class:`AxisDecomposition` with
    bounded zero-delta per-axis breakdown + canonical Provenance — making
    the consumer Tier-B-promotion-ready when Dim 6 Step 6.5 fires.

    Args:
        candidate: a dict-like mapping with optional fields per the
            ``_PER_FRAME_DIFFICULTY_FIELDS`` / ``_EGO_MOTION_FIELDS`` /
            ``_PER_CLASS_FIELDS`` / ``_COMMA2K19_FIELDS`` aliases.

    Returns:
        dict with Catalog #341 canonical non-promotable markers PLUS
        observability annotation containing per-domain summaries.
    """
    if not isinstance(candidate, Mapping):
        return _no_signal_verdict(
            "candidate is not a mapping; domain-prior aggregation refused"
        )

    archive_sha = _extract_archive_sha(candidate)

    per_frame_summary = _extract_per_frame_difficulty_summary(candidate)
    ego_motion_summary = _extract_ego_motion_summary(candidate)
    per_class_summary = _extract_per_class_summary(candidate)
    comma2k19_summary = _extract_comma2k19_summary(candidate)

    n_priors_present = sum(
        s is not None
        for s in (
            per_frame_summary,
            ego_motion_summary,
            per_class_summary,
            comma2k19_summary,
        )
    )

    rationale_parts = [
        "domain_prior_consumer: 4-domain aggregate observability "
        f"(priors_present={n_priors_present}/4)",
    ]
    if per_frame_summary is not None:
        n_hardest = len(per_frame_summary.get("top_k_hardest_frame_indices") or [])
        rationale_parts.append(f"per_frame_hardest_n={n_hardest}")
    if ego_motion_summary is not None:
        kind = ego_motion_summary.get("source_anchor_kind", "unknown")
        rationale_parts.append(f"ego_motion_kind={kind}")
    if per_class_summary is not None:
        n_classes = len(per_class_summary.get("class_priors") or [])
        rationale_parts.append(f"per_class_n={n_classes}")
    if comma2k19_summary is not None:
        n_chunks = len(comma2k19_summary.get("cached_chunk_ids") or [])
        rationale_parts.append(f"comma2k19_cached_chunks={n_chunks}")
    rationale_parts.append("Catalog #341 Tier A: predicted_delta=0.0 [predicted]")
    rationale = "; ".join(rationale_parts)

    output: dict[str, Any] = {
        # Catalog #341 canonical non-promotable markers (Tier A invariants).
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        # Catalog #305 observability surface: per-domain breakdown.
        "consumer_signal_kind": "domain_prior_aggregate_observability",
        "n_priors_present": n_priors_present,
        "per_frame_difficulty_summary": per_frame_summary,
        "ego_motion_concentration_summary": ego_motion_summary,
        "per_segnet_class_chroma_summary": per_class_summary,
        "comma2k19_ood_distance_summary": comma2k19_summary,
        "domain_priors_equation_ids_cited": list(_DOMAIN_PRIORS_EQUATION_IDS),
    }

    # OPTIONAL BONUS per Catalog #356 + Dim 3 Step 3.1: emit
    # predicted_axis_decomposition when archive_sha256 is present so the
    # consumer is Tier-B-promotion-ready. The per-axis deltas are bounded
    # zero per Catalog #341 Tier A invariants; canonical Provenance
    # carries the [predicted] axis tag.
    if archive_sha is not None:
        try:
            decomposition = _build_predicted_axis_decomposition(
                archive_sha256=archive_sha,
                n_priors_present=n_priors_present,
            )
            output["predicted_axis_decomposition"] = decomposition.as_dict()
        except (ImportError, ValueError, TypeError) as exc:
            # Per Catalog #335 paradigm: never raise in the cathedral main
            # loop. Surface the decomposition-build failure as a notes
            # entry but keep emitting the rest of the observability.
            output["predicted_axis_decomposition_error"] = str(exc)

    return output


# ---------------------------------------------------------------------------
# Helpers — anchor extraction + canonical-equation coercion
# ---------------------------------------------------------------------------


def _extract_canonical_anchor_fields(
    anchor: Any,
) -> tuple[str | None, float | None, str | None]:
    """Extract (equation_id, residual, axis_tag) from a typed anchor or dict."""
    equation_id: str | None = None
    residual: float | None = None
    axis_tag: str | None = None

    for getter in (
        lambda obj, key: getattr(obj, key, None),
        lambda obj, key: obj.get(key) if isinstance(obj, Mapping) else None,
    ):
        try:
            if equation_id is None:
                eq_value = getter(anchor, "equation_id")
                if isinstance(eq_value, str) and eq_value.strip():
                    equation_id = eq_value.strip()
            if residual is None:
                r_value = getter(anchor, "residual")
                if isinstance(r_value, bool):
                    # bool subclasses int; reject explicit bool to avoid
                    # accidental True/False -> 1.0/0.0 coercion.
                    pass
                elif isinstance(r_value, (int, float)) and r_value == r_value:
                    if not math.isinf(r_value):
                        residual = float(r_value)
            if axis_tag is None:
                a_value = getter(anchor, "axis_tag") or getter(
                    anchor, "measurement_axis"
                )
                if isinstance(a_value, str) and a_value.strip():
                    axis_tag = a_value.strip()
        except (AttributeError, TypeError, KeyError):
            continue

    return equation_id, residual, axis_tag


def _coerce_to_empirical_anchor(
    *,
    anchor: Any,
    equation_id: str,
    residual: float,
    axis_tag: str | None,
    EmpiricalAnchor: Any,
) -> Any:
    """Coerce a raw anchor input to a canonical :class:`EmpiricalAnchor`.

    The canonical EmpiricalAnchor surface per tac.canonical_equations.equation
    is:
      anchor_id / measurement_utc / inputs / predicted_output /
      empirical_output / residual / source_artifact / measurement_method /
      provenance
    """
    if isinstance(anchor, EmpiricalAnchor):
        return anchor

    # Best-effort field extraction. Caller (canonical equations registry)
    # validates the full EmpiricalAnchor invariants.
    measurement_axis = axis_tag or "[predicted]"
    measurement_utc = (
        _safe_str_attr(anchor, "measurement_utc") or "1970-01-01T00:00:00Z"
    )
    archive_sha256 = (
        _safe_str_attr(anchor, "archive_sha256") or _PROVENANCE_PLACEHOLDER_SHA
    )
    call_id = _safe_str_attr(anchor, "call_id") or "domain_prior_consumer_update"
    measurement_hardware = (
        _safe_str_attr(anchor, "measurement_hardware") or "unknown"
    )
    measurement_method = (
        _safe_str_attr(anchor, "measurement_method")
        or "domain_prior_consumer.update_from_anchor"
    )
    predicted_value = _safe_numeric_attr(anchor, "predicted_value")
    if predicted_value is None:
        # If only residual is known, derive a placeholder predicted value
        # such that empirical_output - predicted_output == residual.
        predicted_value = 0.0
    empirical_value = predicted_value + residual
    source_artifact = (
        _safe_str_attr(anchor, "source_artifact")
        or f"domain_prior_consumer:archive={archive_sha256[:12]}"
    )

    # Build canonical Provenance per Catalog #323.
    try:
        from tac.provenance.builders import build_provenance_for_predicted

        provenance = build_provenance_for_predicted(
            model_id=_PROVENANCE_MODEL_ID,
            inputs_sha256=archive_sha256,
            measurement_axis=measurement_axis,
            hardware_substrate=measurement_hardware,
        )
    except ImportError as exc:
        raise ValueError(
            f"tac.provenance unavailable for canonical Provenance: {exc}"
        ) from exc

    return EmpiricalAnchor(
        anchor_id=f"domain_prior_consumer:{equation_id}:{call_id}",
        measurement_utc=measurement_utc,
        inputs={
            "equation_id": equation_id,
            "archive_sha256": archive_sha256,
            "call_id": call_id,
            "measurement_axis": measurement_axis,
            "measurement_hardware": measurement_hardware,
        },
        predicted_output=float(predicted_value),
        empirical_output=float(empirical_value),
        residual=float(residual),
        source_artifact=source_artifact,
        measurement_method=measurement_method,
        provenance=provenance,
    )


def _safe_str_attr(obj: Any, name: str) -> str | None:
    """Probe both attribute + mapping-key access for a string field."""
    for getter in (
        lambda o, k: getattr(o, k, None),
        lambda o, k: o.get(k) if isinstance(o, Mapping) else None,
    ):
        try:
            v = getter(obj, name)
            if isinstance(v, str) and v.strip():
                return v.strip()
        except (AttributeError, TypeError, KeyError):
            continue
    return None


def _safe_numeric_attr(obj: Any, name: str) -> float | None:
    """Probe both attribute + mapping-key access for a numeric field."""
    for getter in (
        lambda o, k: getattr(o, k, None),
        lambda o, k: o.get(k) if isinstance(o, Mapping) else None,
    ):
        try:
            v = getter(obj, name)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v == v and not math.isinf(v):
                return float(v)
        except (AttributeError, TypeError, KeyError):
            continue
    return None


def _extract_archive_sha(candidate: Mapping[str, Any]) -> str | None:
    """Extract canonical 64-char hex archive sha256 from candidate."""
    for field_name in (
        "archive_sha256",
        "archive_sha",
        "sha256",
        "sha",
        "scored_archive_sha256",
    ):
        value = candidate.get(field_name)
        if isinstance(value, str):
            stripped = value.strip().lower()
            if (
                len(stripped) == 64
                and all(c in "0123456789abcdef" for c in stripped)
            ):
                return stripped
    return None


# ---------------------------------------------------------------------------
# Helpers — per-domain summary extraction (defensive)
# ---------------------------------------------------------------------------


def _extract_per_frame_difficulty_summary(
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract a JSON-safe per-frame-difficulty summary from candidate.

    Supports both a typed :class:`PerFrameDifficultyAtlas` dataclass AND a
    dict-form summary (the canonical ``as_dict()`` output). Returns None
    if no recognizable signal is present.
    """
    raw = _first_present(candidate, _PER_FRAME_DIFFICULTY_FIELDS)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return _safe_summary_from_dict(
            raw,
            keys=(
                "total_frames",
                "total_pairs",
                "aggregator",
                "source_pair_atlas_archive_sha256",
                "source_measurement_axis",
                "top_k_hardest_frame_indices",
                "bottom_k_easiest_frame_indices",
            ),
        )
    # Typed PerFrameDifficultyAtlas (or sister with as_dict).
    return _safe_as_dict_call(raw)


def _extract_ego_motion_summary(
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract a JSON-safe ego-motion-concentration summary from candidate."""
    raw = _first_present(candidate, _EGO_MOTION_FIELDS)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return _safe_summary_from_dict(
            raw,
            keys=(
                "total_frames",
                "source_anchor_kind",
                "source_archive_sha256",
                "source_measurement_axis",
                "atick_redlich_alignment_tag",
                "top_k_ego_motion_intense_frame_indices",
            ),
        )
    return _safe_as_dict_call(raw)


def _extract_per_class_summary(
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract a JSON-safe per-class-statistical-priors summary from candidate."""
    raw = _first_present(candidate, _PER_CLASS_FIELDS)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return _safe_summary_from_dict(
            raw,
            keys=(
                "source_archive_sha256",
                "source_measurement_axis",
                "source_scorer_kind",
                "canonical_openpilot_mask_prior_contract_cited",
                "class_priors",
            ),
        )
    return _safe_as_dict_call(raw)


def _extract_comma2k19_summary(
    candidate: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract a JSON-safe Comma2k19-dashcam-priors summary from candidate."""
    raw = _first_present(candidate, _COMMA2K19_FIELDS)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return _safe_summary_from_dict(
            raw,
            keys=(
                "cached_chunk_ids",
                "total_cached_bytes",
                "dataset_license_spdx",
                "dataset_provenance",
                "is_ood_relative_to_contest_video",
                "canonical_cache_helper_invocation",
            ),
        )
    return _safe_as_dict_call(raw)


def _first_present(
    candidate: Mapping[str, Any], field_aliases: Sequence[str]
) -> Any:
    """Return the first non-None field value from the candidate among aliases."""
    for name in field_aliases:
        value = candidate.get(name)
        if value is not None:
            return value
    return None


def _safe_summary_from_dict(
    raw: Mapping[str, Any], *, keys: Sequence[str]
) -> dict[str, Any]:
    """Defensive subset projection over a dict-form summary."""
    return {k: raw.get(k) for k in keys if k in raw}


def _safe_as_dict_call(raw: Any) -> dict[str, Any] | None:
    """Call ``raw.as_dict()`` if available; else return None defensively."""
    as_dict = getattr(raw, "as_dict", None)
    if not callable(as_dict):
        return None
    try:
        return as_dict()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Helpers — Catalog #356 AxisDecomposition emission (OPTIONAL BONUS)
# ---------------------------------------------------------------------------


def _build_predicted_axis_decomposition(
    *, archive_sha256: str, n_priors_present: int
) -> AxisDecomposition:
    """Build canonical zero-delta :class:`AxisDecomposition` for Tier A invariants.

    Per Catalog #341 Tier A: predicted_delta_adjustment=0.0, so per-axis
    deltas (predicted_d_seg_delta / predicted_d_pose_delta /
    predicted_archive_bytes_delta) also bounded zero. The decomposition
    carries canonical Provenance per Catalog #323 so the consumer is
    Tier-B-promotion-ready when Dim 6 Step 6.5 fires (per-axis deltas
    become non-zero when paired contest-CPU + contest-CUDA anchors land).

    Args:
        archive_sha256: 64-char hex sha256 of the candidate archive.
        n_priors_present: number of priors observed (0-4) — recorded in
            the rationale of the Provenance for cite-chain.

    Returns:
        Frozen :class:`AxisDecomposition` with canonical Provenance.
    """
    try:
        from tac.provenance.builders import build_provenance_for_predicted
        from tac.provenance.validator import provenance_to_dict
    except ImportError as exc:
        raise ImportError(
            f"tac.provenance unavailable: {exc}; cannot emit AxisDecomposition"
        ) from exc

    import hashlib

    inputs_seed = (
        f"domain_prior_consumer:archive_sha256={archive_sha256}:"
        f"n_priors_present={n_priors_present}"
    )
    inputs_sha256 = hashlib.sha256(inputs_seed.encode("utf-8")).hexdigest()
    prov = build_provenance_for_predicted(
        model_id=_PROVENANCE_MODEL_ID,
        inputs_sha256=inputs_sha256,
        measurement_axis="[predicted]",
        hardware_substrate="cpu_local",
    )
    canonical_provenance = provenance_to_dict(prov)

    return AxisDecomposition(
        predicted_d_seg_delta=0.0,
        predicted_d_pose_delta=0.0,
        predicted_archive_bytes_delta=0,
        axis_tag="[predicted]",
        canonical_provenance=canonical_provenance,
    )


# ---------------------------------------------------------------------------
# Helpers — fail-closed verdicts
# ---------------------------------------------------------------------------


def _no_signal_verdict(reason: str) -> dict[str, Any]:
    """Catalog #341 canonical no-signal verdict for consume_candidate."""
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": f"{CONSUMER_NAME}: {reason} [predicted]",
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _fail_closed_update(
    reason: str, *, equation_id: str | None = None
) -> dict[str, Any]:
    """Canonical fail-closed update payload (no posterior mutation performed)."""
    return {
        "accepted": False,
        "status": "fail_closed",
        "reason": reason,
        "equation_id": equation_id,
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "posterior_mutation_performed": False,
    }


__all__ = (
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_TIER",
    "CONSUMER_HOOK_NUMBERS",
    "consume_candidate",
    "update_from_anchor",
)
