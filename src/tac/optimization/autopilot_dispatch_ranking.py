# SPDX-License-Identifier: MIT
"""Cathedral autopilot dispatch ranking using substrate composition matrix.

Per operator directive 2026-05-11 ("wiring and integration" + "autopilot")
+ MM autopilot activation landing (`feedback_cathedral_autopilot_activation_
phase2_probes_integration_audit_v2_landed_20260511.md`), this module bridges
the substrate composition matrix (Deliverable 1) into the cathedral autopilot
autonomous loop (`tools/cathedral_autopilot_autonomous_loop.py`).

The ranker takes the 24-substrate inventory + Hinton-distilled L2 encoders
landing + per-substrate dispatch cost estimates and produces a ranked list
of `CandidateRow`-compatible dispatches the autopilot consumes via the
existing HALT-and-ASK pattern.

**Ranking is a planning artifact, not a dispatch authorization.** Every
ranked candidate carries `score_claim=False`, `promotion_eligible=False`,
`ready_for_exact_eval_dispatch=False`. The autopilot's existing
operator-authorized le-$5/individual mode is the ONLY path that actually
authorizes a dispatch; this ranker just orders the queue the operator sees.

**Composition constraints applied (per Deliverable 1 matrix):**

1. Drop redundant substrates with strictly lower EV/$ (Pareto-frontier
   filter on (axis, class) projection).
2. Refuse multi-substrate dispatches that include >1 RENDERER_REPLACEMENT
   (mutually exclusive at archive level).
3. Prefer ORTHOGONAL pairs (alpha=1.0) over REDUNDANT/ANTAGONISTIC.
4. Honor `--per-dispatch-cap-usd` and `--cumulative-cap-usd` envelopes.
5. SELF_COMPRESSION + RENDERER_REPLACEMENT ranks above bare RENDERER_REPLACEMENT
   (cascade gives diminishing-but-positive returns).

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — Deliverable 1
  (matrix consumed here)
- :mod:`tools.cathedral_autopilot_autonomous_loop` — autopilot loop that
  consumes :class:`RankedDispatchCandidate` rows
- :mod:`tools.theoretical_floor_solver_v2` — Deliverable 3 (refresh
  consumes the same matrix)
- :mod:`tac.continual_learning` — posterior updates feed back into the
  per-substrate predicted delta band on next loop iteration

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``halt_and_ask_default_on``
- ``no_tmp_paths``
- ``substrate_composition_matrix_consumed``
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.substrate_composition_matrix import (
    Composability,
    CompositionMatrix,
    ParetoRow,
    build_composition_matrix,
    filter_pareto_dominated,
    per_substrate_pareto_rows,
    predicted_composite_delta,
)

SCHEMA_VERSION = "tac_autopilot_dispatch_ranking_v1"

# Default envelope budgets per CLAUDE.md "Cathedral autopilot activation"
# (operator-authorized le-$5/individual mode):
#   per_dispatch_cap_usd  = $5.00
#   cumulative_cap_usd    = $20.00
# These are mirrored from the autopilot loop's defaults so the ranker and
# loop agree on which candidates fit the envelope.
DEFAULT_PER_DISPATCH_CAP_USD: float = 5.00
DEFAULT_CUMULATIVE_CAP_USD: float = 20.00


@dataclass(frozen=True)
class RankedDispatchCandidate:
    """One ranked dispatch candidate.

    Compatible with :class:`tools.cathedral_autopilot_autonomous_loop.CandidateRow`
    via the :meth:`as_candidate_row_kwargs` helper.

    Per CLAUDE.md "Forbidden score claims": ``predicted_score_delta`` is
    explicitly tagged as `[predicted; substrate composition matrix v1]`.
    """

    candidate_id: str
    family: str  # substrate_class value (e.g., "renderer_replacement").
    substrate_ids: tuple[str, ...]  # 1+ substrates participating in this dispatch.
    predicted_score_delta: float  # negative = improvement; from composite delta.  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
    expected_information_gain: float
    estimated_dispatch_cost_usd: float
    eig_per_dollar: float
    composition_notes: str
    lane_class: str = ""
    literature_anchor: str = ""
    source_supports: str = ""
    paper_claim_scope: str = ""
    pact_must_prove: str = ""
    decode_complexity_evidence: str = ""
    source_fidelity_metadata: tuple[str, ...] = ()
    campaign_metadata: tuple[str, ...] = ()
    prediction_band: dict[str, Any] | None = None
    prediction_band_verdict: dict[str, Any] | None = None
    blockers: tuple[str, ...] = ()
    license_ok: bool = True
    inflate_dep_count: int = 0
    sideinfo_consumed: bool | None = None
    exact_duplicate: bool = False
    context_order: int = 0
    fits_per_dispatch_cap: bool = True
    fits_cumulative_envelope: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_candidate_row_kwargs(self) -> dict[str, Any]:
        """Return kwargs for `tools.cathedral_autopilot_autonomous_loop.CandidateRow`."""
        return {  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
            "candidate_id": self.candidate_id,
            "family": self.family,
            "predicted_score_delta": self.predicted_score_delta,
            "expected_information_gain": self.expected_information_gain,
            "estimated_dispatch_cost_usd": self.estimated_dispatch_cost_usd,
            "blockers": list(self.blockers),
            "notes": self.composition_notes,
            "lane_class": self.lane_class or None,
            "literature_anchor": self.literature_anchor,
            "source_supports": self.source_supports,
            "paper_claim_scope": self.paper_claim_scope,
            "pact_must_prove": self.pact_must_prove,
            "decode_complexity_evidence": self.decode_complexity_evidence,
            "license_ok": self.license_ok,
            "inflate_dep_count": self.inflate_dep_count,
            "sideinfo_consumed": self.sideinfo_consumed,
            "exact_duplicate": self.exact_duplicate,
            "context_order": self.context_order,
        }


@dataclass(frozen=True)
class RankingResult:
    """Aggregated ranking output.

    Includes the composition matrix used (so consumers can audit which
    substrates were considered) plus the ranked dispatches plus the
    envelope diagnostic.
    """

    schema: str
    generated_at_utc: str
    matrix_schema: str
    n_substrates_considered: int
    per_dispatch_cap_usd: float
    cumulative_cap_usd: float
    cumulative_estimated_spend_usd: float
    n_ranked_dispatches: int
    n_filtered_dropped: int
    ranked_dispatches: list[RankedDispatchCandidate]
    composition_constraints_applied: tuple[str, ...]
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


# ── Ranking algorithm ────────────────────────────────────────────────────


def _source_fidelity_metadata(row: ParetoRow, *, prefix: str = "") -> tuple[str, ...]:
    return tuple(
        part
        for part in (
            f"{prefix}source_supports={row.source_supports}" if row.source_supports else "",
            f"{prefix}paper_claim_scope={row.paper_claim_scope}" if row.paper_claim_scope else "",
            f"{prefix}pact_must_prove={row.pact_must_prove}" if row.pact_must_prove else "",
            (
                f"{prefix}decode_complexity_evidence={row.decode_complexity_evidence}"
                if row.decode_complexity_evidence
                else ""
            ),
        )
        if part
    )


def _prediction_band_verdict_allows_rank_reward(
    verdict: dict[str, Any] | None,
) -> bool:
    """Prediction-band rank reward requires a typed literal JSON true."""

    if verdict is None:
        return True
    return verdict.get("valid_for_rank_reward") is True


def _build_singleton_dispatch_candidates(
    pareto_rows: list[ParetoRow],
    matrix: CompositionMatrix,
) -> list[RankedDispatchCandidate]:
    """One RankedDispatchCandidate per substrate (the simplest dispatch unit)."""
    out: list[RankedDispatchCandidate] = []
    for r in pareto_rows:
        source_fidelity_metadata = _source_fidelity_metadata(r)
        campaign_metadata = tuple(
            part for part in (
                f"lane_id={r.lane_id}" if r.lane_id else "",
                f"campaign_id={r.campaign_id}" if r.campaign_id else "",
                f"campaign_stage={r.campaign_stage}" if r.campaign_stage else "",
                f"campaign_priority={r.campaign_priority}" if r.campaign_priority else "",
                f"lane_class={r.lane_class}" if r.lane_class else "",
                f"literature_anchor={r.literature_anchor}" if r.literature_anchor else "",
                *source_fidelity_metadata,
                f"license_ok={r.license_ok}",
                f"inflate_dep_count={r.inflate_dep_count}",
                f"sideinfo_consumed={r.sideinfo_consumed}",
                f"exact_duplicate={r.exact_duplicate}",
                f"context_order={r.context_order}",
            )
            if part
        )
        metadata_note = (
            "; " + "; ".join(campaign_metadata)
            if campaign_metadata else ""
        )
        rank_reward_note = (
            "; prediction_band_rank_reward_suppressed"
            if not _prediction_band_verdict_allows_rank_reward(
                r.prediction_band_verdict
            )
            else ""
        )
        expected_information_gain = (
            r.expected_information_gain if not rank_reward_note else 0.0
        )
        eig_per_dollar = r.eig_per_dollar if not rank_reward_note else 0.0
        out.append(
            RankedDispatchCandidate(
                candidate_id=f"singleton__{r.substrate_id}",
                family=r.substrate_class.value,
                substrate_ids=(r.substrate_id,),
                predicted_score_delta=r.predicted_delta_alone_midpoint,  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
                expected_information_gain=expected_information_gain,
                estimated_dispatch_cost_usd=r.estimated_dispatch_cost_usd,
                eig_per_dollar=eig_per_dollar,
                composition_notes=(
                    f"[predicted; substrate composition matrix v1] "
                    f"singleton dispatch of {r.substrate_id}; "
                    f"axis={r.target_axis.value}, class={r.substrate_class.value}"
                    f"{metadata_note}"
                    f"{rank_reward_note}"
                ),
                lane_class=r.lane_class,
                literature_anchor=r.literature_anchor,
                source_supports=r.source_supports,
                paper_claim_scope=r.paper_claim_scope,
                pact_must_prove=r.pact_must_prove,
                decode_complexity_evidence=r.decode_complexity_evidence,
                source_fidelity_metadata=source_fidelity_metadata,
                campaign_metadata=campaign_metadata,
                prediction_band=r.prediction_band,
                prediction_band_verdict=r.prediction_band_verdict,
                blockers=r.dispatch_blockers,
                license_ok=r.license_ok,
                inflate_dep_count=r.inflate_dep_count,
                sideinfo_consumed=r.sideinfo_consumed,
                exact_duplicate=r.exact_duplicate,
                context_order=r.context_order,
            )
        )
    return out


def _build_orthogonal_pair_candidates(
    pareto_rows: list[ParetoRow],
    matrix: CompositionMatrix,
    *,
    per_dispatch_cap_usd: float,
) -> list[RankedDispatchCandidate]:
    """Build candidates for orthogonal pairs that fit the per-dispatch cap.

    Per CLAUDE.md "Bayesian experimental design": orthogonal pairs deliver
    additive expected information gain at the joint-dispatch cost. The
    per-dispatch cap is applied to the SUM of the pair's costs.

    Filtering rule: only emit pairs whose joint cost <= per_dispatch_cap_usd.
    Pairs above cap remain visible at the singleton level.
    """
    out: list[RankedDispatchCandidate] = []
    n = len(pareto_rows)
    for i in range(n):
        for j in range(i + 1, n):
            ri = pareto_rows[i]
            rj = pareto_rows[j]
            cell = matrix.get(ri.substrate_id, rj.substrate_id)
            if cell.composability != Composability.ORTHOGONAL:
                continue
            joint_cost = ri.estimated_dispatch_cost_usd + rj.estimated_dispatch_cost_usd
            if joint_cost > per_dispatch_cap_usd:
                continue
            joint_eig = ri.expected_information_gain + rj.expected_information_gain
            # Cost-zero is treated as cost-unknown (missing estimation), NOT
            # free. Emit eig_per_dollar=0.0 (sorts LAST) so the row's existing
            # `cost_estimation_required` blocker (propagated from singletons)
            # surfaces the gap. float("inf") would (a) violate RFC 8259 when
            # JSON-serialized and (b) trivially dominate the sort, masking
            # real signal.
            joint_eig_per_dollar = (
                joint_eig / joint_cost if joint_cost > 0 else 0.0
            )
            composite = predicted_composite_delta(
                [ri.substrate_id, rj.substrate_id], matrix=matrix
            )
            lane_class = "+".join(
                part for part in (ri.lane_class, rj.lane_class) if part
            )
            literature_anchor = "; ".join(
                part for part in (ri.literature_anchor, rj.literature_anchor) if part
            )
            source_supports = "; ".join(
                f"{row.substrate_id}: {row.source_supports}"
                for row in (ri, rj)
                if row.source_supports
            )
            paper_claim_scope = "; ".join(
                f"{row.substrate_id}: {row.paper_claim_scope}"
                for row in (ri, rj)
                if row.paper_claim_scope
            )
            pact_must_prove = "; ".join(
                f"{row.substrate_id}: {row.pact_must_prove}"
                for row in (ri, rj)
                if row.pact_must_prove
            )
            decode_complexity_evidence = "; ".join(
                f"{row.substrate_id}: {row.decode_complexity_evidence}"
                for row in (ri, rj)
                if row.decode_complexity_evidence
            )
            source_fidelity_metadata = tuple(
                part
                for row in (ri, rj)
                for part in _source_fidelity_metadata(
                    row,
                    prefix=f"{row.substrate_id}:",
                )
            )
            blockers = tuple(dict.fromkeys(
                list(ri.dispatch_blockers) + list(rj.dispatch_blockers)
            ))
            campaign_metadata = tuple(
                part
                for row in (ri, rj)
                for part in (
                    f"{row.substrate_id}:lane_id={row.lane_id}" if row.lane_id else "",
                    f"{row.substrate_id}:campaign_id={row.campaign_id}" if row.campaign_id else "",
                    f"{row.substrate_id}:campaign_stage={row.campaign_stage}" if row.campaign_stage else "",
                    f"{row.substrate_id}:campaign_priority={row.campaign_priority}" if row.campaign_priority else "",
                    f"{row.substrate_id}:lane_class={row.lane_class}" if row.lane_class else "",
                    f"{row.substrate_id}:literature_anchor={row.literature_anchor}" if row.literature_anchor else "",
                    *_source_fidelity_metadata(row, prefix=f"{row.substrate_id}:"),
                    f"{row.substrate_id}:license_ok={row.license_ok}",
                    f"{row.substrate_id}:inflate_dep_count={row.inflate_dep_count}",
                    f"{row.substrate_id}:sideinfo_consumed={row.sideinfo_consumed}",
                    f"{row.substrate_id}:exact_duplicate={row.exact_duplicate}",
                    f"{row.substrate_id}:context_order={row.context_order}",
                )
                if part
            )
            prediction_band_verdicts = tuple(
                verdict
                for verdict in (ri.prediction_band_verdict, rj.prediction_band_verdict)
                if verdict is not None
            )
            rank_reward_allowed = all(
                _prediction_band_verdict_allows_rank_reward(verdict)
                for verdict in prediction_band_verdicts
            )
            if not rank_reward_allowed:
                joint_eig = 0.0
                joint_eig_per_dollar = 0.0
            metadata_note = (
                f"; metadata={list(campaign_metadata)!r}"
                if campaign_metadata else ""
            )
            rank_reward_note = (
                "; prediction_band_rank_reward_suppressed"
                if not rank_reward_allowed
                else ""
            )
            out.append(
                RankedDispatchCandidate(
                    candidate_id=f"orthogonal_pair__{ri.substrate_id}__{rj.substrate_id}",
                    family=f"{ri.substrate_class.value}+{rj.substrate_class.value}",
                    substrate_ids=(ri.substrate_id, rj.substrate_id),
                    predicted_score_delta=composite["predicted_composite_delta"],  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
                    expected_information_gain=joint_eig,
                    estimated_dispatch_cost_usd=joint_cost,
                    eig_per_dollar=joint_eig_per_dollar,
                    composition_notes=(
                        f"[predicted; substrate composition matrix v1] "
                        f"orthogonal pair ({ri.substrate_id} + {rj.substrate_id}); "
                        f"alpha={cell.expected_alpha:.2f}; "
                        f"rationale={cell.rationale}"
                        f"{metadata_note}"
                        f"{rank_reward_note}"
                    ),
                    lane_class=lane_class,
                    literature_anchor=literature_anchor,
                    source_supports=source_supports,
                    paper_claim_scope=paper_claim_scope,
                    pact_must_prove=pact_must_prove,
                    decode_complexity_evidence=decode_complexity_evidence,
                    source_fidelity_metadata=source_fidelity_metadata,
                    campaign_metadata=campaign_metadata,
                    prediction_band_verdict={
                        "components": list(prediction_band_verdicts),
                    },
                    blockers=blockers,
                    license_ok=ri.license_ok and rj.license_ok,
                    inflate_dep_count=ri.inflate_dep_count + rj.inflate_dep_count,
                    sideinfo_consumed=(
                        (ri.sideinfo_consumed or False)
                        or (rj.sideinfo_consumed or False)
                    ),
                    exact_duplicate=ri.exact_duplicate or rj.exact_duplicate,
                    context_order=max(ri.context_order, rj.context_order),
                )
            )
    return out


def rank_dispatches(
    *,
    matrix: CompositionMatrix | None = None,
    per_dispatch_cap_usd: float = DEFAULT_PER_DISPATCH_CAP_USD,
    cumulative_cap_usd: float = DEFAULT_CUMULATIVE_CAP_USD,
    include_orthogonal_pairs: bool = True,
    drop_redundant_dominated: bool = True,
    max_total: int | None = None,
) -> RankingResult:
    """Rank dispatches by EV/$ subject to composition constraints + envelopes.

    Per CLAUDE.md "Cathedral autopilot dispatch hook" + the May-2026 race
    postmortem: the autopilot's PRIMARY signal is EV/$. The composition
    matrix is the secondary signal that enforces (a) no double-dispatch
    of redundant siblings, (b) no two-renderer-replacement dispatches.

    Parameters
    ----------
    matrix
        The composition matrix (defaults to canonical 24-substrate inventory).
    per_dispatch_cap_usd
        Hard cap per single dispatch (operator-set; default $5.00).
    cumulative_cap_usd
        Hard cap for cumulative spend across this ranking session (default $20.00).
    include_orthogonal_pairs
        If True, also emit 2-substrate joint dispatches when their joint
        cost fits per_dispatch_cap_usd. Default True.
    drop_redundant_dominated
        If True, drop substrates dominated by a redundant sibling with
        strictly higher EV/$ (Pareto-frontier filter). Default True.
    max_total
        Optional cap on the number of ranked dispatches returned.

    Returns
    -------
    RankingResult bundle with the ranked list + envelope diagnostic.
    """
    matrix = matrix or build_composition_matrix()
    base_rows = per_substrate_pareto_rows(matrix=matrix)
    n_pre_filter = len(base_rows)
    if drop_redundant_dominated:
        base_rows = filter_pareto_dominated(base_rows, matrix=matrix)
    n_dropped = n_pre_filter - len(base_rows)

    # Singleton dispatches.
    singletons = _build_singleton_dispatch_candidates(base_rows, matrix)

    # Orthogonal pair dispatches that fit per-dispatch cap.
    pairs: list[RankedDispatchCandidate] = []
    if include_orthogonal_pairs:
        pairs = _build_orthogonal_pair_candidates(
            base_rows, matrix, per_dispatch_cap_usd=per_dispatch_cap_usd
        )

    # Apply per-dispatch cap to singletons (annotate, don't drop).
    annotated_singletons: list[RankedDispatchCandidate] = []
    for c in singletons:
        fits = c.estimated_dispatch_cost_usd <= per_dispatch_cap_usd
        annotated_singletons.append(
            dataclasses.replace(c, fits_per_dispatch_cap=fits)
        )

    # Combine singletons + pairs; rank by EV/$ descending.
    all_dispatches = list(annotated_singletons) + list(pairs)
    all_dispatches.sort(key=lambda c: c.eig_per_dollar, reverse=True)

    # Apply cumulative envelope: walk down the list, mark any candidate that
    # would push us over cumulative_cap_usd as fits_cumulative_envelope=False.
    cumulative_spend = 0.0
    enveloped: list[RankedDispatchCandidate] = []
    for c in all_dispatches:
        prospective = cumulative_spend + c.estimated_dispatch_cost_usd
        fits_envelope = prospective <= cumulative_cap_usd
        enveloped.append(
            dataclasses.replace(c, fits_cumulative_envelope=fits_envelope)
        )
        if fits_envelope and c.fits_per_dispatch_cap:
            cumulative_spend = prospective

    # Optional cap.
    if max_total is not None:
        enveloped = enveloped[:max_total]

    return RankingResult(
        schema=SCHEMA_VERSION,
        generated_at_utc=dt.datetime.now(dt.UTC).isoformat(),
        matrix_schema=matrix.schema_version,
        n_substrates_considered=matrix.n_substrates(),
        per_dispatch_cap_usd=per_dispatch_cap_usd,
        cumulative_cap_usd=cumulative_cap_usd,
        cumulative_estimated_spend_usd=cumulative_spend,
        n_ranked_dispatches=len(enveloped),
        n_filtered_dropped=n_dropped,
        ranked_dispatches=enveloped,
        composition_constraints_applied=(
            "drop_redundant_dominated" if drop_redundant_dominated else "no_drop",
            "renderer_replacement_mutually_exclusive",
            "format_id_collision_check",
            f"per_dispatch_cap_usd={per_dispatch_cap_usd}",
            f"cumulative_cap_usd={cumulative_cap_usd}",
        ),
    )


# ── Hinton-distilled L2 encoder integration ─────────────────────────────


def synthetic_l2_encoder_dispatch_candidates(
    *,
    encoders: tuple[str, ...] = ("c3_residual", "wavelet_residual", "cool_chic_residual"),
    cost_per_encoder_usd: float = 0.30,
    predicted_delta_per_encoder: tuple[float, float] = (-0.0030, -0.0008),
) -> list[RankedDispatchCandidate]:
    """Construct candidates representing Hinton-distilled L2 encoder dispatches.

    Per LL landing (`feedback_hinton_distilled_scorer_saliency_masked_l2_
    encoders_landed_20260511.md`), the L2 sparse-aware encoders gain a
    ``--use-hinton-distilled-scorer`` + ``--use-saliency-masking`` flag pair
    that breaks the YUV6 MSE proxy dominance (603.78 -> 0.64 on smoke).

    These candidates are special: they are L1-band rate-residual encoders
    BUT the LL upgrade promises stronger empirical Δ score per byte than the
    bare encoder. Per LL operator-decision item 2 ("First L2 + Hinton +
    saliency Modal T4 dispatch ($0.30) — proceed?"), the cost is ~$0.30/family.

    Per CLAUDE.md "Forbidden score claims": every candidate carries
    `score_claim=False` and is tagged `[predicted; LL Hinton-distilled L2
    encoder upgrade]`. Promotion requires `[contest-CUDA]` anchor.
    """
    out: list[RankedDispatchCandidate] = []
    delta_low, delta_high = predicted_delta_per_encoder
    delta_mid = 0.5 * (delta_low + delta_high)
    eig = abs(delta_mid)
    for enc in encoders:
        eig_per_dollar = eig / cost_per_encoder_usd if cost_per_encoder_usd > 0 else 0.0
        out.append(
            RankedDispatchCandidate(
                candidate_id=f"l2_hinton_saliency__{enc}",
                family="residual_l2_hinton_distilled",
                substrate_ids=(enc,),
                predicted_score_delta=delta_mid,  # DUAL_AXIS_RANKING_WAIVED: planning-only single-axis prediction; dual-axis CPU/CUDA companion lives at empirical-anchor / posterior_update_locked layer per CLAUDE.md auth-eval-everywhere
                expected_information_gain=eig,
                estimated_dispatch_cost_usd=cost_per_encoder_usd,
                eig_per_dollar=eig_per_dollar,
                composition_notes=(
                    f"[predicted; LL Hinton-distilled L2 encoder upgrade] "
                    f"{enc} + --use-hinton-distilled-scorer "
                    f"--use-saliency-masking; "
                    f"YUV6 proxy dominance broken empirically (603.78 -> 0.64); "
                    f"per LL operator-decision item 2 ($0.30/family)"
                ),
            )
        )
    return out


# ── Serialization ────────────────────────────────────────────────────────


def serialize_candidate(c: RankedDispatchCandidate) -> dict[str, Any]:
    d = dataclasses.asdict(c)
    d["substrate_ids"] = list(c.substrate_ids)
    d["blockers"] = list(c.blockers)
    d["campaign_metadata"] = list(c.campaign_metadata)
    d["source_fidelity_metadata"] = list(c.source_fidelity_metadata)
    return d


def serialize_ranking(result: RankingResult) -> dict[str, Any]:
    """JSON-safe serialization of a RankingResult."""
    return {
        "schema": result.schema,
        "generated_at_utc": result.generated_at_utc,
        "matrix_schema": result.matrix_schema,
        "n_substrates_considered": result.n_substrates_considered,
        "per_dispatch_cap_usd": result.per_dispatch_cap_usd,
        "cumulative_cap_usd": result.cumulative_cap_usd,
        "cumulative_estimated_spend_usd": result.cumulative_estimated_spend_usd,
        "n_ranked_dispatches": result.n_ranked_dispatches,
        "n_filtered_dropped": result.n_filtered_dropped,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_autopilot_dispatch_ranking_v1",
        "ranked_dispatches": [serialize_candidate(c) for c in result.ranked_dispatches],
        "composition_constraints_applied": list(result.composition_constraints_applied),
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_default_on",
            "no_tmp_paths",
            "substrate_composition_matrix_consumed",
        ],
    }


def write_ranking_json(result: RankingResult, path: str) -> None:
    """Write the ranking as pretty-printed JSON.

    Per CLAUDE.md "Forbidden /tmp paths": refuses /tmp/var/tmp/private/tmp paths.
    """
    if path.startswith("/tmp/") or "/private/tmp/" in path or "/var/tmp/" in path:
        raise ValueError(f"refusing to write to forbidden /tmp path: {path!r}")
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_ranking(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_CUMULATIVE_CAP_USD",
    "DEFAULT_PER_DISPATCH_CAP_USD",
    "SCHEMA_VERSION",
    "RankedDispatchCandidate",
    "RankingResult",
    "rank_dispatches",
    "serialize_candidate",
    "serialize_ranking",
    "synthetic_l2_encoder_dispatch_candidates",
    "write_ranking_json",
]
