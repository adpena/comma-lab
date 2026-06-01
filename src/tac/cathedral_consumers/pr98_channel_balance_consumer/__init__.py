# SPDX-License-Identifier: MIT
"""Cathedral consumer for L28 PR98 zero-byte decode-side channel-balance bolt-on.

Slot LL landing per Slot DD canonical highest-EV-shortest-WC RANK 1 finding
(L28 PR #98 third-prize zero-byte canonical trick; -0.0001 to -0.0005 score
points + 0 archive bytes per PR101 hnerv_ft_microcodec inflate.py:49-51).

Per Catalog #335 canonical contract auto-discovery + Catalog #341 Tier A
canonical-routing markers + Catalog #287 placeholder-rationale rejection.

Consumer surfaces L28 applicability prediction on every candidate that maps
to a known PR-95-family substrate (V14-V2 DQS1, fec6, PR106, NSCS06, sister
HNeRV/HNeRV-class decoder substrates). Per-candidate prediction is
observability-only (predicted_delta_adjustment=0.0 + axis_tag=[predicted] +
promotable=False) per Catalog #341. Promotion to Tier B per Catalog #357
requires paired-CUDA RATIFICATION empirical anchor per Catalog #246.

Per Slot DD canonical: L28 applies to ANY current frontier candidate; this
consumer's role is to ANNOTATE candidates with the canonical L28
applicability + estimated score delta band so the cathedral autopilot
ranker + operator-routable next-step queue see the bolt-on as a structural
RANK 1 immediate-application opportunity.

Hook assignments per Catalog #125:
  * #4 cathedral autopilot dispatch — ACTIVE (annotate candidates)
  * #5 continual-learning posterior — ACTIVE (refresh canonical equation
    candidate when paired-CUDA RATIFICATION anchors land)
  * #1, #2, #3, #6 — N/A (observability-only annotation; no
    sensitivity-map / Pareto / bit-allocator / probe-disambiguator
    contribution from THIS consumer; per-axis decomposition is surfaced
    via the canonical helper's build_axis_decomposition_for_pr98_bolt_on)

Per Slot DD operator-routable #5: this consumer enables the operator to
see L28 applicability across the cathedral autopilot's per-candidate
ranking surface without requiring per-substrate manual review.

Per operator binding META directive #3: INTEGRATE + WIRE into existing
cathedral_consumers/ namespace via canonical Catalog #335 auto-discovery
(NOT parallel build).

Cross-references:
  * tac.codec.pr98_channel_balance_zero_byte_bolt_on (canonical L28 helper)
  * Slot DD canonical: .omx/research/cross_pr_family_canonical_techniques_mining_L14_L70_20260529T075244Z.md
  * CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L28
  * Catalog #335 canonical Protocol contract
  * Catalog #341 Tier A canonical-routing markers (this consumer IS Tier A)
  * Catalog #287 placeholder-rationale rejection (sister discipline)
  * Catalog #344 canonical equations registry (predicted_equation_candidate)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "pr98_channel_balance_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


# Canonical PR-95-family substrate tokens this consumer recognizes. Per Slot
# DD canonical highest-EV-shortest-WC RANK 1 finding: L28 applies to ANY
# HNeRV-class decoder substrate (PR95-family canonical 600 × 28-d latent
# grid + PixelShuffle + bilinear-skip + sin per L18). The token set below
# enumerates known current-frontier candidates per
# list_candidate_substrates_for_l28_application + extends to canonical
# sister substrates per HNeRV parity discipline.
_PR98_L28_APPLICABLE_SUBSTRATE_TOKENS: frozenset[str] = frozenset({
    # Current canonical frontier candidates per Catalog #343
    "v14_v2_dqs1",
    "fec6",
    "pr106_format0d",
    "pr106",
    "nscs06_v8",
    "nscs06",
    "dqs1",
    # PR-95-family canonical decoders (all share PR101 inflate.py:49-51 sister structure)
    "pr95",
    "pr95_family",
    "pr100",
    "pr101",
    "pr103",
    "hnerv",
    "hnerv_lc_v2",
    "hnerv_lc_ac",
    "hnerv_ft_microcodec",
    "hnerv_muon",
    # Frame-exploit-selector sister family
    "frame_exploit",
    "fec",
    "fec10",
    # PR110 + sister substitution-stacking
    "pr110",
    "pr110_opt",
})


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    When a paired-CUDA RATIFICATION empirical anchor lands on a substrate
    where L28 was applied, the canonical equation candidate
    `pr98_zero_byte_decode_side_channel_balance_score_savings_v1` accumulates
    a new EmpiricalAnchor per Catalog #344. Auto-recalibration per Catalog
    #371 fires `when_3+_new_empirical_anchors_in_domain`.

    This consumer is structurally NO-OP at this surface — refresh happens
    via the canonical `tac.canonical_equations.update_equation_with_empirical_anchor`
    path. The consumer's role is per-candidate annotation, not refit.
    """
    _ = anchor  # explicit acknowledgment; refit lives at the canonical equation surface


def _candidate_substrate_token(candidate: Mapping[str, Any]) -> str:
    """Extract canonical substrate token from a cathedral candidate dict.

    Per the cathedral autopilot main loop's canonical candidate schema,
    inspect common fields where the substrate / lane / archive_family
    token is surfaced.
    """
    for field in ("substrate_id", "substrate", "lane_id", "archive_family", "id", "name"):
        value = candidate.get(field)
        if isinstance(value, str) and value:
            return value.lower()
    return ""


def _l28_applicable_to_substrate(substrate_token: str) -> bool:
    """Return True when the substrate token matches a known PR-95-family decoder."""
    if not substrate_token:
        return False
    substrate_lower = substrate_token.lower()
    return any(
        marker in substrate_lower
        for marker in _PR98_L28_APPLICABLE_SUBSTRATE_TOKENS
    )


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — annotate candidate with L28 applicability.

    Per Slot DD canonical: surfaces L28 applicability + estimated score
    delta band as observability-only annotation. Caller (cathedral
    autopilot ranker) can use the annotation to prioritize substrates where
    L28 is structurally applicable for paired-CUDA RATIFICATION dispatch.

    Per Catalog #341 Tier A canonical-routing markers contract:
      * predicted_delta_adjustment=0.0 (observability-only; never a score signal)
      * axis_tag='[predicted]' (per Catalog #287/#323 canonical Provenance)
      * promotable=False (promotion requires paired-CUDA empirical anchor)

    Returns:
        Canonical Tier A contribution dict per Catalog #341 with
        L28-applicability rationale + estimated score delta band + canonical
        equation candidate reference.
    """
    substrate_token = _candidate_substrate_token(candidate)
    applicable = _l28_applicable_to_substrate(substrate_token)

    if not applicable:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"L28 PR98 channel-balance not structurally applicable to"
                f" substrate token={substrate_token!r}; canonical L28 sister"
                " surface is PR-95-family HNeRV-class decoder substrates"
                " (see _PR98_L28_APPLICABLE_SUBSTRATE_TOKENS) [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    # L28 IS applicable; surface canonical applicability annotation.
    try:
        from tac.codec.pr98_channel_balance_zero_byte_bolt_on import (
            CANONICAL_EQUATION_CANDIDATE_ID,
            PR98_L28_ARCHIVE_BYTES_DELTA,
            PR98_L28_CANONICAL_SOURCE_LINE_RANGE,
            PR98_L28_EXPECTED_SCORE_DELTA_BAND,
        )
    except ImportError:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"L28 canonical helper unavailable; substrate={substrate_token}"
                " [predicted]"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }

    rationale = (
        f"L28 PR98 zero-byte decode-side channel-balance applicable to"
        f" substrate={substrate_token};"
        f" canonical source=PR101 hnerv_ft_microcodec inflate.py:"
        f"{PR98_L28_CANONICAL_SOURCE_LINE_RANGE};"
        f" estimated score delta band={PR98_L28_EXPECTED_SCORE_DELTA_BAND};"
        f" archive bytes delta={PR98_L28_ARCHIVE_BYTES_DELTA} (zero-byte);"
        f" canonical equation candidate={CANONICAL_EQUATION_CANDIDATE_ID};"
        " paired-CUDA RATIFICATION required for promotion per Catalog #246 [predicted]"
    )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.5,  # mid-confidence per Slot DD canonical -0.0001 to -0.0005 band
    }


__all__ = [
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
    "update_from_anchor",
    "consume_candidate",
]
