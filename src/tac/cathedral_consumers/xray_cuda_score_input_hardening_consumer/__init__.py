# SPDX-License-Identifier: MIT
"""Cathedral consumer for X-ray CUDA-score input hardening discipline.

Per CODEX CROSS-POLLINATION audit `aafac7c84` §10.1 + operator blanket approval
2026-05-20 (WAVE-3-CATHEDRAL-CONSUMER-REGISTRATION-CODEX-AUDIT-CANDIDATES).

Source codex memo: `.omx/research/xray_cuda_score_input_hardening_20260511_codex.md`
documenting the hardening of ``tools/xray_cpu_cuda_drift_per_arch_class.py`` to
refuse bare ``--cuda-score <float>`` inputs (which made it too easy to feed
rounded / stale / CPU / MPS / proxy / chat-derived numbers into the CPU/CUDA
drift predictor and treat the prediction as artifact-backed CUDA evidence).
The fix routes the canonical path through ``--cuda-auth-eval-json`` parsed via
``tools.auth_eval_records.parse_auth_eval_payload`` and refuses inputs unless
``score_axis=contest_cuda`` + ``n_samples=600`` + ``gpu_t4_match=true``.

This consumer annotates candidates that resemble CPU/CUDA drift analysis work
(cuda-score / xray / cpu_cuda_drift tokens in candidate text) with the
canonical custody-routing reminder so the cathedral autopilot ranker can
surface the canonical input-hardening verdict per Catalog #287/#127 +
CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

Tier A (observability-only) per Catalog #341 canonical-routing-markers:

- ``predicted_delta_adjustment=0.0`` (NEVER mutates score signal)
- ``promotable=False`` (per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA")
- ``axis_tag="[predicted]"`` (per Catalog #287 canonical Provenance umbrella)

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch - ACTIVE (annotate candidates with
    canonical input-hardening custody reminder)
  * #5 continual-learning posterior - ACTIVE (NO-OP refresh path; drift
    custody anchors flow through ``tac.continual_learning.posterior_update_locked``)
  * #1 sensitivity-map - N/A (defensive annotation only)
  * #2 Pareto constraint - N/A
  * #3 bit-allocator - N/A
  * #6 probe-disambiguator - ACTIVE (the canonical input-hardening verdict
    IS the disambiguator between artifact-backed CUDA evidence vs free-
    floating numeric inputs)

Sister of:
  * ``wr01_static_packet_custody_consumer`` (static packet custody validation)
  * ``tt5l_sideinfo_consumer`` (per-pair sideinfo consumption proof)
  * ``venn_risk_composition_consumer`` (Venn rank-composition guard)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "xray_cuda_score_input_hardening_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Canonical X-ray CUDA-drift analysis tokens lifted from the codex memo's
# "Fix" + "Score-lowering relevance" sections. A candidate whose text overlaps
# any of these tokens is treated as a CPU/CUDA drift analysis candidate and
# annotated with the canonical input-hardening custody reminder.
_XRAY_CUDA_DRIFT_TOKENS: tuple[str, ...] = (
    "xray_cpu_cuda_drift",
    "cpu_cuda_drift",
    "cuda_score_source",
    "manual_cuda_score_diagnostic",
    "drift_per_arch_class",
    "cuda_auth_eval_json",
    "unknown_uncalibrated",
)


# Canonical input-hardening requirements per the codex memo's "Fix" section.
# Surfaced as observability annotations so the cathedral autopilot ranker
# can flag drift candidates that don't route through the canonical custody
# path.
_CANONICAL_INPUT_HARDENING_REQUIREMENTS: tuple[str, ...] = (
    "cuda_auth_eval_json_required",
    "score_axis_contest_cuda_required",
    "n_samples_600_required",
    "gpu_t4_match_true_required",
    "allow_unknown_architecture_class_explicit_required_if_unknown_uncalibrated",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    NO-OP refresh path. CUDA drift custody anchors flow through the canonical
    posterior surface (``tac.continual_learning.posterior_update_locked`` per
    Catalog #128/#131 fcntl-locked discipline); this consumer does not
    maintain in-memory state because input-hardening verdicts are per-
    candidate static-text matches against the canonical drift token set.
    """
    _ = anchor  # explicit acknowledgment; no in-memory state to refresh


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Annotates candidates whose text overlaps the canonical X-ray CUDA-drift
    token set with input-hardening custody reminders. Returns
    ``predicted_delta_adjustment=0.0`` always (Tier A observability-only per
    Catalog #341 canonical-routing-markers).

    Hook #6 probe-disambiguator semantics: the canonical input-hardening
    requirements ARE the disambiguator between artifact-backed CUDA evidence
    (parseable JSON with score_axis=contest_cuda + n_samples=600 +
    gpu_t4_match=true) vs free-floating numeric inputs (manual --cuda-score
    diagnostic escape hatch). The consumer surfaces the requirement set so
    operators can audit which path a candidate is on without re-reading the
    source.
    """
    candidate_text = " ".join(
        f"{k}={v}"
        for k, v in candidate.items()
        if isinstance(v, (str, int, float))
    ).lower()

    matched_tokens = [
        token for token in _XRAY_CUDA_DRIFT_TOKENS if token in candidate_text
    ]

    if not matched_tokens:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "no X-ray CUDA-drift token match; consumer inactive for this "
                "candidate [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    matched_summary = ",".join(matched_tokens[:3])
    rationale = (
        f"X-ray CUDA-drift candidate detected (matched: {matched_summary}); "
        f"verify canonical input-hardening requirements: "
        f"{','.join(_CANONICAL_INPUT_HARDENING_REQUIREMENTS)} [predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "matched_tokens": tuple(matched_tokens),
        "canonical_input_hardening_requirements": (
            _CANONICAL_INPUT_HARDENING_REQUIREMENTS
        ),
    }
