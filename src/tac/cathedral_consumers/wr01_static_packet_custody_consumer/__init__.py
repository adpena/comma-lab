# SPDX-License-Identifier: MIT
"""Cathedral consumer for WR01-style static packet custody validation.

Per CODEX CROSS-POLLINATION audit `aafac7c84` §10.1 + operator blanket approval
2026-05-20 (WAVE-3-CATHEDRAL-CONSUMER-REGISTRATION-CODEX-AUDIT-CANDIDATES).

Source codex memo: `.omx/research/wr01_static_packet_custody_20260506_codex.md`
documenting the WR01 ``hnerv_wavelet_apply_transform_pr106x_1_2`` byte-custody
exact-eval candidate (archive_bytes=186222 / archive_sha256=d2208ffa.../
byte_delta=-9 / byte_only_expected_score_delta=-5.99e-6) with explicit gate
state (``static_packet_ready=true`` / ``byte_custody_exact_eval_candidate_ready=true``
/ ``runtime_decode_gate_ready=true``) and remaining structural blockers
(``missing_lightning_environment`` / ``missing_active_lane_dispatch_claim`` /
``adversarial_priority_review_prioritizes_rate_only_candidate``).

This consumer annotates candidates that resemble static-packet custody work
(byte-custody-exact-eval / static_packet_ready / runtime_decode_gate_ready
tokens in candidate text) with structural-blocker surfacing so the cathedral
autopilot ranker can surface the canonical readiness verdict per Catalog #287
canonical Provenance discipline.

Tier A (observability-only) per Catalog #341 canonical-routing-markers:

- ``predicted_delta_adjustment=0.0`` (NEVER mutates score signal)
- ``promotable=False`` (per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA")
- ``axis_tag="[predicted]"`` (per Catalog #287 canonical Provenance umbrella)

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch - ACTIVE (annotate candidates with
    structural-blocker readiness surface)
  * #5 continual-learning posterior - ACTIVE (NO-OP refresh path; canonical
    custody anchors flow through ``tac.continual_learning.posterior_update_locked``
    per Catalog #128/#131 fcntl-locked discipline)
  * #1 sensitivity-map - N/A (defensive annotation only)
  * #2 Pareto constraint - N/A
  * #3 bit-allocator - N/A
  * #6 probe-disambiguator - N/A

Sister of:
  * ``xray_cuda_score_input_hardening_consumer`` (CUDA-axis input hardening)
  * ``tt5l_sideinfo_consumer`` (per-pair sideinfo consumption proof)
  * ``venn_risk_composition_consumer`` (Venn rank-composition guard)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "wr01_static_packet_custody_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


# Canonical readiness tokens lifted from the WR01 codex memo's "Gate State"
# section. A candidate whose text overlaps any of these tokens is treated as
# a static-packet custody candidate and annotated with the canonical
# structural-blocker reminder set.
_STATIC_PACKET_CUSTODY_TOKENS: tuple[str, ...] = (
    "static_packet_ready",
    "byte_custody_exact_eval_candidate_ready",
    "runtime_decode_gate_ready",
    "candidate_static_preflight_ready",
    "wr01",
    "hnerv_wavelet_apply_transform",
    "exact_eval_packet",
)


# Canonical structural-blocker reminder set per WR01 codex memo "Gate State"
# remaining blockers list. Surfaced as observability annotations so the
# cathedral autopilot ranker can flag custody candidates that haven't cleared
# the canonical readiness checklist.
_CANONICAL_STRUCTURAL_BLOCKERS: tuple[str, ...] = (
    "missing_lightning_environment",
    "missing_active_lane_dispatch_claim",
    "adversarial_priority_review_prioritizes_rate_only_candidate",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    NO-OP refresh path. Static packet custody anchors flow through the
    canonical posterior surface (``tac.continual_learning.posterior_update_locked``
    per Catalog #128/#131 fcntl-locked discipline); this consumer does not
    maintain in-memory state because custody verdicts are per-candidate
    static-text matches against the canonical readiness token set.
    """
    _ = anchor  # explicit acknowledgment; no in-memory state to refresh


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Annotates candidates whose text overlaps the canonical static-packet
    custody token set with structural-blocker readiness reminders. Returns
    ``predicted_delta_adjustment=0.0`` always (Tier A observability-only per
    Catalog #341 canonical-routing-markers).
    """
    candidate_text = " ".join(
        f"{k}={v}"
        for k, v in candidate.items()
        if isinstance(v, (str, int, float))
    ).lower()

    matched_tokens = [
        token
        for token in _STATIC_PACKET_CUSTODY_TOKENS
        if token in candidate_text
    ]

    if not matched_tokens:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "no static-packet-custody token match; consumer inactive "
                "for this candidate [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    matched_summary = ",".join(matched_tokens[:3])
    rationale = (
        f"static-packet custody candidate detected (matched: {matched_summary}); "
        f"verify canonical structural blockers cleared: "
        f"{','.join(_CANONICAL_STRUCTURAL_BLOCKERS)} [predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_tokens": tuple(matched_tokens),
        "canonical_structural_blockers": _CANONICAL_STRUCTURAL_BLOCKERS,
    }
