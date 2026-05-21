# SPDX-License-Identifier: MIT
"""Cathedral consumer for Venn rank-composition predicted-dispatch-risk guard.

Per CODEX CROSS-POLLINATION audit `aafac7c84` §10.1 + operator blanket approval
2026-05-20 (WAVE-3-CATHEDRAL-CONSUMER-REGISTRATION-CODEX-AUDIT-CANDIDATES).

Source codex memo: `.omx/research/venn_risk_composition_bugfix_20260517_codex.md`
documenting the 2026-05-17 Venn reweighting wire-in bugfix: the new
``master_gradient_consumers`` sidecar-driven per-archive rank adjustment must
NOT bypass or replace the older OP-3 ``predicted_dispatch_risk`` structural
refusal hook. The canonical composition order is now explicit:

  1. Apply score-axis rank adjustments.
  2. Apply ``adjust_predicted_delta_for_predicted_dispatch_risk``.
  3. Apply ``adjust_predicted_delta_for_venn_classification`` to the
     already-risk-adjusted delta.

A candidate with ``predicted_dispatch_risk >= 50`` must floor its effective
score delta at ``0.0`` even if a Venn sidecar reports HIGH PAIR_INVARIANT
byte mass. Venn classification is planning evidence, NOT a dispatch-safety
override. Regression test:
``test_venn_reweight_does_not_replace_predicted_dispatch_risk_refusal``.

This consumer annotates candidates that resemble Venn / predicted-dispatch-
risk composition work (venn / predicted_dispatch_risk / pair_invariant /
master_gradient_consumers tokens in candidate text) with the canonical
composition-order reminder so the cathedral autopilot ranker can surface the
canonical risk-floor verdict per Catalog #319 (sister Wyner-Ziv venn-reweight
gate) + Catalog #287 canonical Provenance.

Tier A (observability-only) per Catalog #341 canonical-routing-markers:

- ``predicted_delta_adjustment=0.0`` (NEVER mutates score signal; the actual
  risk-floor logic lives in ``adjust_predicted_delta_for_predicted_dispatch_risk``
  in the autopilot main loop)
- ``promotable=False`` (per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA")
- ``axis_tag="[predicted]"`` (per Catalog #287 canonical Provenance umbrella)

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch - ACTIVE (annotate candidates with
    canonical composition-order reminder)
  * #5 continual-learning posterior - ACTIVE (NO-OP refresh path; venn
    classification anchors flow through ``tac.continual_learning.posterior_update_locked``)
  * #1 sensitivity-map - N/A (defensive annotation only)
  * #2 Pareto constraint - N/A
  * #3 bit-allocator - N/A
  * #6 probe-disambiguator - ACTIVE (the canonical composition-order IS the
    disambiguator between Venn-as-rank-signal vs Venn-as-safety-override
    semantics)

Sister of:
  * ``wr01_static_packet_custody_consumer`` (static packet custody validation)
  * ``xray_cuda_score_input_hardening_consumer`` (CUDA-axis input hardening)
  * ``tt5l_sideinfo_consumer`` (per-pair sideinfo consumption proof)
  * Catalog #319 ``check_substrate_wyner_ziv_reweight_has_deliverability_proof``
    (the sister Wyner-Ziv venn-reweight strict gate this consumer mirrors at
    the canonical-consumer surface)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "venn_risk_composition_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Canonical Venn rank-composition tokens lifted from the codex memo's
# "Finding" + "Fix" sections. A candidate whose text overlaps any of these
# tokens is treated as a Venn rank-composition candidate and annotated with
# the canonical composition-order reminder.
_VENN_COMPOSITION_TOKENS: tuple[str, ...] = (
    "venn",
    "predicted_dispatch_risk",
    "pair_invariant",
    "high_pair_invariant",
    "master_gradient_consumers",
    "venn_classification",
    "venn_reweight",
)


# Canonical composition-order per the codex memo's "Fix" section. The
# 3-step ordering is the structural protection that Venn rank signal cannot
# override the dispatch-safety refusal hook.
_CANONICAL_COMPOSITION_ORDER: tuple[str, ...] = (
    "step_1_apply_score_axis_rank_adjustments_first",
    "step_2_apply_adjust_predicted_delta_for_predicted_dispatch_risk_second",
    "step_3_apply_adjust_predicted_delta_for_venn_classification_third_to_already_risk_adjusted_delta",
)


# Canonical risk-floor invariant per the codex memo's "Finding" section.
# When ``predicted_dispatch_risk >= 50`` the effective score delta floors
# at ``0.0`` even if Venn reports HIGH PAIR_INVARIANT byte mass.
_CANONICAL_RISK_FLOOR_INVARIANT: tuple[str, ...] = (
    "predicted_dispatch_risk_ge_50_floors_effective_delta_at_zero",
    "venn_classification_is_planning_evidence_not_dispatch_safety_override",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    NO-OP refresh path. Venn classification anchors flow through the
    canonical posterior surface (``tac.continual_learning.posterior_update_locked``
    per Catalog #128/#131 fcntl-locked discipline); this consumer does not
    maintain in-memory state because composition-order verdicts are per-
    candidate static-text matches against the canonical Venn token set.
    """
    _ = anchor  # explicit acknowledgment; no in-memory state to refresh


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Annotates candidates whose text overlaps the canonical Venn rank-
    composition token set with composition-order + risk-floor invariant
    reminders. Returns ``predicted_delta_adjustment=0.0`` always (Tier A
    observability-only per Catalog #341 canonical-routing-markers).

    Hook #6 probe-disambiguator semantics: the canonical composition-order
    IS the disambiguator between Venn-as-rank-signal (Tier A planning
    evidence; safe to apply AFTER predicted_dispatch_risk refusal hook) vs
    Venn-as-safety-override semantics (FORBIDDEN; would bypass dispatch-
    safety floor). The consumer surfaces both invariants so operators can
    audit which composition path a candidate is on without re-reading the
    source.
    """
    candidate_text = " ".join(
        f"{k}={v}"
        for k, v in candidate.items()
        if isinstance(v, (str, int, float))
    ).lower()

    matched_tokens = [
        token for token in _VENN_COMPOSITION_TOKENS if token in candidate_text
    ]

    if not matched_tokens:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "no Venn rank-composition token match; consumer inactive "
                "for this candidate [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    matched_summary = ",".join(matched_tokens[:3])
    rationale = (
        f"Venn rank-composition candidate detected (matched: "
        f"{matched_summary}); canonical composition order: "
        f"{','.join(_CANONICAL_COMPOSITION_ORDER)}; risk-floor invariant: "
        f"{','.join(_CANONICAL_RISK_FLOOR_INVARIANT)} [predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_tokens": tuple(matched_tokens),
        "canonical_composition_order": _CANONICAL_COMPOSITION_ORDER,
        "canonical_risk_floor_invariant": _CANONICAL_RISK_FLOOR_INVARIANT,
    }
