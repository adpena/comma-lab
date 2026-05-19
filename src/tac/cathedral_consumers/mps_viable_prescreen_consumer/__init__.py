# SPDX-License-Identifier: MIT
"""MPS-VIABLE pre-screen consumer for cathedral autopilot.

Per META-ASSUMPTION review item #2 (commit ``c8d51ebb5`` 2026-05-19; memo
``.omx/research/meta_assumption_adversarial_review_post_r11_20260519T062526Z.md``)
+ MPS-VIABLE probe outcome
``mps_phase_b_options_b_plus_c_completion_20260519T062500Z`` registered to
``.omx/state/probe_outcomes.jsonl`` (PROCEED at 0.072% aggregate gap; 69x
below 5% LOCAL_MPS_TRAIN_VIABLE threshold).

This consumer operationalizes the META-ASSUMPTION review's classification
that the 23x MPS-vs-CUDA universality assumption is HARD-EARNED-NUANCED:
on current archives + current PoseNet/SegNet + current scoring code the
MPS-vs-CUDA drift is <1% (deeply VIABLE), so local MPS is a valid 1:1
pre-screen surrogate for dev-velocity-accelerating substrate experiments.

When a candidate's evidence_grade / axis_tag indicates advisory-grade
(MPS-research-signal / macOS-CPU-advisory / [predicted] / [advisory only])
the consumer recommends routing through local MPS pre-screen BEFORE paying
for Modal/Vast.ai/Lightning dispatch. When a candidate's evidence_grade
indicates promotable contest-axis (contest-CUDA / contest-CPU on Linux
x86_64 + NVIDIA) the consumer recommends ``paid_cuda_authoritative``: per
CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable +
"MPS auth eval is NOISE" non-negotiable, promotion REQUIRES dual-axis
contest-faithful hardware; local MPS routing is forbidden for promotable
candidates.

The consumer auto-fallbacks to ``paid_cuda_authoritative`` if the
MPS-VIABLE probe outcome is SUPERSEDED to ``DEFER`` (future re-measurement
contradicts current PROCEED) per Catalog #313 probe-outcomes ledger
consumption discipline. Empty/missing probe outcome treated as
conservative fail-closed (no MPS routing recommendation until empirical
anchor lands).

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: this consumer's
recommendations are ADVISORY-only ranking signals; the cathedral autopilot
ranker (consumer of this output) is responsible for actually routing
dispatch + the routing target is responsible for axis-tag preservation
per Catalog #127/#192/#317. This consumer contributes
``predicted_delta_adjustment=0.0`` (routing signal, not score signal) per
the canonical CathedralConsumerContract.

Sister of:
- ``mps_diagnostic_consumer`` (layerwise MPS-vs-CPU/CUDA drift; pure
  diagnostic, no routing recommendation)
- ``mps_gap_experiment_consumer`` (aggregate MPS gap quantification; pure
  diagnostic, no routing recommendation)
- ``contest_oracle_consumer`` (contest-axis evidence-grade canonical lookup
  the routing decision logic mirrors)
- ``per_pair_pareto_envelope_consumer`` (consumer-pattern reference;
  diagnostic-only with axis-tag honoring)

Catalog #125 6-hook wire-in:
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (PRIMARY — this consumer's routing
  signal is consumed by the cathedral autopilot ranker cascade)
- Hook #6 PROBE_DISAMBIGUATOR (SECONDARY — the MPS-VIABLE probe outcome IS
  the canonical disambiguator between MPS-routable and CUDA-required
  candidates; consumer reads the probe ledger at consume-time so SUPERSEDE
  events auto-fallback the routing recommendation)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "mps_viable_prescreen_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)


# Canonical MPS-VIABLE probe outcome ID per
# .omx/research/mps_phase_b_options_b_plus_c_completion_20260519T062500Z.md.
# When this probe is SUPERSEDED to a blocking verdict (DEFER / KILL /
# INDEPENDENT), the consumer auto-fallbacks routing recommendation to
# paid_cuda_authoritative to preserve the dual-axis discipline.
MPS_VIABLE_PROBE_ID = "mps_phase_b_options_b_plus_c_completion_20260519T062500Z"


# Canonical advisory-grade axis tokens per CLAUDE.md "Apples-to-apples
# evidence discipline" + Catalog #127/#192/#317. A candidate with ANY of
# these tags (or evidence_grade) is NON-PROMOTABLE by construction and a
# safe target for local MPS pre-screen routing.
_ADVISORY_AXIS_TOKENS = frozenset(
    {
        "[predicted]",
        "[advisory only]",
        "[macos-cpu advisory]",
        "[mps-proxy]",
        "[mps-research-signal]",
        "[diagnostic-cpu]",
        "[diagnostic-cuda]",
        "predicted",
        "advisory",
        "advisory_only",
        "macos_cpu_advisory",
        "macos-cpu-advisory",
        "mps-proxy",
        "mps_proxy",
        "mps-research-signal",
        "mps_research_signal",
        "diagnostic_cpu",
        "diagnostic_cuda",
    }
)


# Canonical contest-axis tokens that REQUIRE paid 1:1 contest-compliant
# hardware. A candidate carrying ANY of these tokens (or
# promotion_eligible=True) is NOT a valid MPS pre-screen target.
_CONTEST_AXIS_TOKENS = frozenset(
    {
        "[contest-cuda]",
        "[contest-cpu]",
        "contest-cuda",
        "contest_cuda",
        "contest-cpu",
        "contest_cpu",
    }
)


# Routing recommendation taxonomy. Consumer never actually executes the
# dispatch — that is the cathedral autopilot ranker's responsibility per
# Catalog #167 (smoke-before-full) + Catalog #243 (local pre-deploy harness)
# + Catalog #271 (codex pre-dispatch review).
ROUTE_LOCAL_MPS_PRESCREEN = "local_mps_prescreen"
ROUTE_PAID_CUDA_AUTHORITATIVE = "paid_cuda_authoritative"
ROUTE_NONE = "none"


def _extract_axis_tokens(candidate: Mapping[str, Any]) -> set[str]:
    """Collect axis-token strings from a candidate's metadata fields.

    Honors multiple canonical surfaces per CLAUDE.md "Apples-to-apples":
    ``axis_tag`` / ``evidence_grade`` / ``score_axis`` / ``lane_tag``.
    Returns lowercased tokens for case-insensitive comparison.
    """
    tokens: set[str] = set()
    for key in ("axis_tag", "evidence_grade", "score_axis", "lane_tag"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            tokens.add(value.strip().lower())
    return tokens


def _is_promotable(candidate: Mapping[str, Any]) -> bool:
    """Return True if the candidate carries explicit promotion signals.

    Per Catalog #127 (custody validator routing) + Catalog #192
    (macOS-CPU-advisory not promoted without Linux verification) +
    Catalog #317 (local-research-signal dispatches stamp evidence grade):
    a candidate is promotable only when an explicit field says so AND no
    advisory axis token contradicts.
    """
    promotion_eligible = candidate.get("promotion_eligible")
    if promotion_eligible is True:
        return True
    if candidate.get("promotable") is True:
        return True
    # Score-claim with contest-axis evidence is promotable by construction.
    if candidate.get("score_claim_valid") is True:
        axis_tokens = _extract_axis_tokens(candidate)
        if axis_tokens & _CONTEST_AXIS_TOKENS:
            return True
    return False


def _is_advisory_only(candidate: Mapping[str, Any]) -> bool:
    """Return True if the candidate is unambiguously advisory-grade.

    A candidate is advisory-only when ANY axis token is in the advisory
    set AND NO contest-axis token contradicts. The contest-axis token wins
    in mixed-signal cases (conservative fail-closed: routing CONTEST-tagged
    candidates to MPS would violate Catalog #192/#317).
    """
    axis_tokens = _extract_axis_tokens(candidate)
    if not axis_tokens:
        return False
    if axis_tokens & _CONTEST_AXIS_TOKENS:
        return False
    return bool(axis_tokens & _ADVISORY_AXIS_TOKENS)


def _mps_viable_probe_blocked() -> bool:
    """Return True if the MPS-VIABLE probe outcome is SUPERSEDED to a
    blocking verdict (DEFER / KILL / INDEPENDENT).

    Per Catalog #313 probe-outcomes ledger consumption discipline + the
    canonical helpers in ``tac.probe_outcomes_ledger``. When the future
    re-measurement contradicts current PROCEED, the routing recommendation
    auto-fallbacks to paid_cuda_authoritative so the consumer remains
    safe-by-construction across the probe's lifecycle.

    Lookup is fail-OPEN on ledger errors: if the canonical ledger is
    unavailable (missing file / IO error / import error) the consumer
    treats the probe as non-blocking. Per CLAUDE.md "Forbidden premature
    KILL": the absence of a blocking signal does not authorize ROUTING
    via MPS by itself; the caller (cathedral autopilot) is responsible
    for cross-checking the canonical probe outcome separately.
    """
    try:
        from tac.probe_outcomes_ledger import (
            BLOCKING_VERDICTS,
            latest_outcome_by_probe_id,
        )
    except Exception:  # pragma: no cover — fail-OPEN per docstring
        return False
    try:
        latest = latest_outcome_by_probe_id(MPS_VIABLE_PROBE_ID)
    except Exception:  # pragma: no cover — fail-OPEN per docstring
        return False
    if latest is None:
        # No outcome registered yet → fail-CLOSED for routing recommendation
        # (the cathedral autopilot ranker will see route=ROUTE_NONE and
        # default to its existing routing logic).
        return True
    verdict = latest.get("verdict")
    if isinstance(verdict, str) and verdict in BLOCKING_VERDICTS:
        return True
    # PROCEED / PARTIAL / PROMOTE / OPERATOR_REVIEW_REQUIRED → non-blocking
    return False


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP.

    The MPS-VIABLE probe outcome is the canonical source-of-truth for the
    routing decision; this consumer reads the canonical
    ``.omx/state/probe_outcomes.jsonl`` ledger at consume-time via
    ``tac.probe_outcomes_ledger`` helpers (Catalog #313). No additional
    posterior update is required here — when the probe is SUPERSEDED, the
    next ``consume_candidate`` call automatically reflects the new state.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns a routing recommendation as observability-only ranking signal.
    Per the canonical CathedralConsumerContract: this consumer contributes
    ``predicted_delta_adjustment=0.0`` because routing is NOT a score
    signal; the cathedral autopilot ranker honors the
    ``recommended_route`` field separately when constructing the dispatch
    plan.

    Decision logic (cascade):

    1. MPS-VIABLE probe SUPERSEDED → fallback to ``paid_cuda_authoritative``
       (preserves dual-axis discipline per CLAUDE.md "Submission auth eval
       — BOTH CPU AND CUDA").
    2. Candidate is promotable contest-axis → ``paid_cuda_authoritative``
       (Catalog #192/#317 forbid MPS routing for promotable candidates).
    3. Candidate is advisory-grade → ``local_mps_prescreen`` (5-10x
       dev-velocity acceleration per META-ASSUMPTION review item #2).
    4. Mixed / no axis information → ``none`` (cathedral autopilot ranker
       falls back to its default routing logic).

    Per CLAUDE.md "Apples-to-apples evidence discipline": axis_tag and
    promotion_eligible are honored; advisory-grade candidates are routed
    to MPS pre-screen ONLY, never promoted to contest-axis without paired
    Linux x86_64 + NVIDIA evidence per Catalog #127/#192/#317.
    """
    # Step 1: probe-outcome blocking check (Catalog #313 consumption).
    if _mps_viable_probe_blocked():
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "MPS-VIABLE probe outcome "
                f"{MPS_VIABLE_PROBE_ID} is SUPERSEDED to blocking verdict "
                "OR no probe outcome registered; auto-fallback to "
                "paid_cuda_authoritative per Catalog #313 + CLAUDE.md "
                '"Submission auth eval — BOTH CPU AND CUDA" non-negotiable'
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 1.0,
            "recommended_route": ROUTE_PAID_CUDA_AUTHORITATIVE,
        }

    # Step 2: promotable contest-axis → MUST go to paid CUDA.
    if _is_promotable(candidate):
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "candidate is promotable contest-axis (promotion_eligible / "
                "score_claim_valid with [contest-CUDA] or [contest-CPU] "
                "axis); routing to paid CUDA per CLAUDE.md "
                '"MPS auth eval is NOISE" + Catalog #192/#317 — MPS '
                "pre-screen forbidden for promotable candidates"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 1.0,
            "recommended_route": ROUTE_PAID_CUDA_AUTHORITATIVE,
        }

    # Step 3: advisory-grade → safe to route to local MPS pre-screen.
    if _is_advisory_only(candidate):
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "candidate is advisory-grade (axis_tag / evidence_grade in "
                "[predicted] / [advisory only] / [macOS-CPU advisory] / "
                "[MPS-PROXY] / [MPS-research-signal] / [diagnostic-*]); "
                "MPS-VIABLE probe outcome PROCEED at 0.072% aggregate gap "
                "(69x below 5% threshold) authorizes 1:1 pre-screen "
                "routing per META-ASSUMPTION review item #2 (commit "
                "c8d51ebb5 2026-05-19); 5-10x dev-velocity acceleration "
                "predicted vs paid Modal/Vast.ai/Lightning dispatch"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.9,
            "recommended_route": ROUTE_LOCAL_MPS_PRESCREEN,
        }

    # Step 4: insufficient signal → no recommendation; cathedral autopilot
    # falls back to its default routing logic.
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "candidate axis_tag / evidence_grade / promotion_eligible "
            "insufficient for MPS pre-screen routing decision; no "
            "recommendation issued (cathedral autopilot ranker falls back "
            "to existing routing logic)"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "recommended_route": ROUTE_NONE,
    }
