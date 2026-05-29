# SPDX-License-Identifier: MIT
"""Cathedral consumer for the Wave N+48 substrate-family L1-L42 hygiene-EV audit.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable +
Catalog #335 paradigm-shift (canonical contract auto-discovery) + Slot M
2026-05-29 Wave N+48 audit + canonical equation
``wave_n_plus_48_l1_l42_hygiene_ev_decay_predicts_pr95_parity_gap_v1``
registered via Slot O Phase 1 (canonical_equations_registry.jsonl row 326).

Wires the orphan-signal closure for the Wave N+48 substrate-family hygiene
audit so the cathedral autopilot ranker sees per-substrate hygiene-EV
annotations on every candidate that maps to a registered substrate family.

This consumer is observability-only (predicted_delta_adjustment=0.0; per
CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323/#341).
Hygiene-EV annotations are NEVER promoted to score adjustments — they
surface as ``[predicted]`` annotations that future paired-CUDA dispatches
can compare against to refresh the equation's
``predicted_vs_empirical_residual`` via Catalog #371 auto-recalibrator.

Per the Slot M audit (`.omx/research/wave_n_plus_48_audit_canonical_re_run_l1_l42_expanded_lesson_set_20260529T060912Z.md`):
- TOP-2 (HYG-EV 0.80; paired-CUDA-eligible): PR101_FAMILY + FRAME_EXPLOIT_HFV
- MID-TIER (HYG-EV 0.59-0.71): PR110_OPT / PR103_HNERV_LC_AC / PR95_FAMILY_DIRECT
- BOTTOM-TIER (HYG-EV 0.49; L14-L42 backfill required): sane_hnerv / PR106 / A1
- LOW-confidence-HIGH-leverage (Tier 3 $0 MLX-LOCAL): TIME_TRAVELER_Z_FAMILY

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE PRIMARY (annotate candidates
    with per-substrate hygiene-EV from registered canonical equation)
  * #5 continual-learning posterior — ACTIVE (auto-recalibration via
    Catalog #371 fires when 3+ NEW substrate audits land against the
    L1-L42 baseline)
  * #1, #2, #3, #6 — N/A (observability-only annotation; no
    sensitivity-map / Pareto / bit-allocator / probe-disambiguator
    contribution at this consumer)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "wave_n_plus_48_hygiene_lookup_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


# Canonical equation id this consumer surfaces; registered via Slot O Phase 1
# at .omx/state/canonical_equations_registry.jsonl row 326 (2026-05-29).
WAVE_N_PLUS_48_EQUATION_ID = (
    "wave_n_plus_48_l1_l42_hygiene_ev_decay_predicts_pr95_parity_gap_v1"
)


# Canonical 25-family hygiene-EV scoring matrix from Wave N+48 audit
# (.omx/research/wave_n_plus_48_audit_canonical_re_run_l1_l42_expanded_lesson_set_20260529T060912Z.md
# Phase B per-family scoring). KEY = substrate family token (lower-case);
# VALUE = (hygiene_ev_score, tier_label, mission_contribution_hint).
#
# Tier labels:
#   - "TOP_2_PAIRED_CUDA_ELIGIBLE" — HYG-EV >= 0.80; canonical-frontier-positioned
#   - "MID_TIER_CLOSE_REVIEW_BASELINE" — HYG-EV 0.59-0.71; queue-able
#   - "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED" — HYG-EV 0.49; substrate-engineering needed
#   - "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL" — HYG-EV ~0.05; class-shift candidate
#   - "FEC_BOLT_ON_STACK" — HYG-EV 0.34; off-the-shelf rate-attack workflow
#   - "FORENSIC_PROBE_BASELINE" — N/A scoring; probe-only lanes
#   - "OTHER_NON_PR95_FAMILY" — HYG-EV 0.05-0.10; non-HNeRV-class
_WAVE_N_PLUS_48_FAMILY_SCORES: dict[str, tuple[float, str, str]] = {
    # TOP-2 paired-CUDA-eligible (HYG-EV 0.80)
    "pr101_family": (0.80, "TOP_2_PAIRED_CUDA_ELIGIBLE", "frontier_protecting"),
    "pr101_lc_v2_clone": (0.80, "TOP_2_PAIRED_CUDA_ELIGIBLE", "frontier_protecting"),
    "pr101_ft_microcodec": (0.80, "TOP_2_PAIRED_CUDA_ELIGIBLE", "frontier_protecting"),
    "frame_exploit_hfv": (0.80, "TOP_2_PAIRED_CUDA_ELIGIBLE", "frontier_breaking_enabler"),
    "frame_exploit_selector": (0.80, "TOP_2_PAIRED_CUDA_ELIGIBLE", "frontier_breaking_enabler"),
    # MID-TIER close-review baseline (HYG-EV 0.59-0.71)
    "pr110_opt": (0.71, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_protecting"),
    "pr103_hnerv_lc_ac": (0.68, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_breaking"),
    "hnerv_lc_ac": (0.68, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_breaking"),
    "pr95_family_direct": (0.59, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_breaking_enabler"),
    "pr95_hnerv": (0.59, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_breaking_enabler"),
    "pr95_lora_dora": (0.59, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_breaking_enabler"),
    "hnerv_muon": (0.59, "MID_TIER_CLOSE_REVIEW_BASELINE", "frontier_breaking_enabler"),
    # BOTTOM-TIER L14-L42 backfill required (HYG-EV 0.49)
    "sane_hnerv": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    "pr100_hnerv_lc_v2": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    "hnerv_lc_v2": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    "pr106_latent_sidecar": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    "pr106": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    "a1_sister": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    "a1": (0.49, "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED", "apparatus_maintenance"),
    # FEC bolt-on stack (HYG-EV 0.34; off-the-shelf workflow)
    "fec_rate_family": (0.34, "FEC_BOLT_ON_STACK", "frontier_breaking_enabler"),
    "fec6": (0.34, "FEC_BOLT_ON_STACK", "frontier_breaking_enabler"),
    "fec8": (0.34, "FEC_BOLT_ON_STACK", "frontier_breaking_enabler"),
    "fec10": (0.34, "FEC_BOLT_ON_STACK", "frontier_breaking_enabler"),
    # LOW-confidence-HIGH-leverage class-shift (Tier 3 MLX-LOCAL)
    "time_traveler_z_family": (0.05, "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL", "frontier_breaking_enabler"),
    "time_traveler_l5": (0.05, "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL", "frontier_breaking_enabler"),
    "z5_predictive_coding": (0.05, "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL", "frontier_breaking_enabler"),
    "z6_v2": (0.05, "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL", "frontier_breaking_enabler"),
    "z7_mamba_2": (0.05, "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL", "frontier_breaking_enabler"),
    # OTHER non-PR95-family
    "hnerv_non_pr95_family": (0.07, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "factorized_hnerv_v1": (0.07, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "cascade_c_frame_1": (0.10, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "nerv_family": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "cool_chic_c3_siren_coord_mlp": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "balle_hyperprior": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "wavelet_mallat": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "self_compress_block_fp": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "grayscale_lut_nscs": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "wunderkind_atw_dp1": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "wyner_ziv_family": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "magic_codec_hessian": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "substrate_composition_stacks": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
    "d1_polytope_overlay": (0.05, "OTHER_NON_PR95_FAMILY", "apparatus_maintenance"),
}


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    When a new empirical anchor lands in the canonical continual-learning
    posterior, the canonical equations registry's
    ``auto_recalibrate_from_continual_learning_posterior`` (Catalog #371)
    is the canonical refresh path. This consumer is structurally NO-OP
    here — the canonical refresh is operator-triggered via
    ``tools/recalibrate_equation.py`` because automatic refit requires
    explicit signed measurement provenance per Catalog #287/#323.

    Sister of the canonical_equation_lookup_consumer pattern; the Slot O
    Wave N+48 equation auto-recalibrates when 3+ NEW substrate audits
    land against the L1-L42 baseline per the equation's declared
    ``next_recalibration_trigger=when_3_or_more_new_empirical_anchors_in_domain``.
    """
    _ = anchor


def _lookup_substrate_family(candidate: Mapping[str, Any]) -> tuple[str, float, str, str] | None:
    """Best-effort match: walk candidate values for substrate-family token.

    Returns (matched_family_token, hygiene_ev, tier_label, mission_contribution)
    for the highest-scoring match, or None if no family token matches.
    """
    candidate_text = " ".join(
        f"{k}={v}" for k, v in candidate.items() if isinstance(v, (str, int, float))
    ).lower()

    best_match: tuple[str, float, str, str] | None = None
    for family_token, (hyg_ev, tier, mission) in _WAVE_N_PLUS_48_FAMILY_SCORES.items():
        if family_token in candidate_text:
            # Prefer highest hygiene-EV match (PR101_FAMILY > PR106 > A1 etc.)
            if best_match is None or hyg_ev > best_match[1]:
                best_match = (family_token, hyg_ev, tier, mission)
    return best_match


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with Wave N+48 hygiene-EV.

    Looks up the candidate's substrate-family token against the canonical
    Wave N+48 scoring matrix and surfaces:
    - per-substrate hygiene_ev score (0.05-0.80 per Wave N+48 audit)
    - tier label (TOP_2 / MID_TIER / BOTTOM_TIER / LOW_CONFIDENCE_HIGH_LEVERAGE / FEC / OTHER)
    - mission_contribution hint per Catalog #300 enum
    - operator-routable cascade routing recommendation

    Returns ``predicted_delta_adjustment=0.0`` always — annotations are
    observability-only per Catalog #287/#323/#341. Promotion requires
    paired-CUDA + paired-CPU empirical anchors per CLAUDE.md "Submission
    auth eval — BOTH CPU AND CUDA" non-negotiable.
    """
    match = _lookup_substrate_family(candidate)
    if match is None:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "no Wave N+48 substrate-family token matches this candidate; "
                "observability-only annotation [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
            "matched_family": None,
            "hygiene_ev": None,
            "tier_label": None,
            "mission_contribution_hint": None,
            "routing_recommendation": None,
            "canonical_equation_id": WAVE_N_PLUS_48_EQUATION_ID,
        }

    family_token, hyg_ev, tier, mission = match

    # Operator-routable cascade routing per Wave N+48 Phase C trichotomy
    if tier == "TOP_2_PAIRED_CUDA_ELIGIBLE":
        routing = (
            "TIER_1_PAIRED_CUDA_ELIGIBLE — canonical-frontier-protected per Catalog #343; "
            "no new dispatch unless operator launches NEW bolt-on candidates"
        )
    elif tier == "MID_TIER_CLOSE_REVIEW_BASELINE":
        routing = (
            "TIER_2_PAIRED_CUDA_QUEUE_ELIGIBLE — $0.50 paired CPU+CUDA dispatch "
            "for close-review-class baseline anchoring"
        )
    elif tier == "BOTTOM_TIER_L14_L42_BACKFILL_REQUIRED":
        routing = (
            "TIER_4_APPARATUS_MAINTENANCE — L14-L42 substrate-engineering backfill "
            "required per HNeRV parity discipline L7 BEFORE paired-CUDA dispatch"
        )
    elif tier == "LOW_CONFIDENCE_HIGH_LEVERAGE_MLX_LOCAL":
        routing = (
            "TIER_3_MLX_LOCAL_FANOUT — $0 MLX-LOCAL training per "
            "mlx-first-numpy-portable standing directive; class-shift candidate "
            "preserves dispatch_enabled=false per Catalog #325 until per-substrate symposium PROCEED"
        )
    elif tier == "FEC_BOLT_ON_STACK":
        routing = (
            "FEC_OFF_THE_SHELF_STACKING — per off-the-shelf rate-attack workflow "
            "standing directive; queue per FEC family progression"
        )
    else:
        routing = (
            "OTHER_NON_PR95_FAMILY — apparatus-maintenance class; bolt-on demonstrations; "
            "NOT structurally promising for frontier improvement"
        )

    rationale = (
        f"Wave N+48 L1-L42 hygiene-EV={hyg_ev:.2f} for substrate-family={family_token!r} "
        f"(tier={tier}); canonical equation={WAVE_N_PLUS_48_EQUATION_ID}; "
        f"routing={routing} [predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_family": family_token,
        "hygiene_ev": hyg_ev,
        "tier_label": tier,
        "mission_contribution_hint": mission,
        "routing_recommendation": routing,
        "canonical_equation_id": WAVE_N_PLUS_48_EQUATION_ID,
    }
