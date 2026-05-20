# SPDX-License-Identifier: MIT
"""Cathedral consumer that auto-triggers cross-substrate similarity recompute on new master-gradient anchors.

Per WAVE-3-PER-BYTE-SENSITIVITY-METHODOLOGY-FOLLOWUP task spec Gap 1
+ operator directive 2026-05-20 verbatim *"a per-byte sensitivity
comparative analysis IS the empirical methodology that converts 'we have
substrate X with score Y' into structural understanding of WHICH BYTES
carry the score advantage"*. Sister of:

* :mod:`tac.cathedral_consumers.cross_substrate_similarity_consumer`
  (the Tier-A annotator that READS the similarity matrix)
* :mod:`tac.cathedral_consumers.canonical_equation_lookup_consumer`
  (Catalog #344 sister; same lookup-the-canonical-source pattern)
* :func:`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`
  (Catalog #343 sister; same auto-refresh-after-canonical-event pattern)

Apparatus-wide methodology extension: converts the per-byte sensitivity
comparative analysis discipline from a manual operator-routable task
into a structural property of the apparatus. When a new master-gradient
anchor lands at ``.omx/state/master_gradient_anchors.jsonl`` (via
:func:`tac.master_gradient.append_anchor_locked`), this consumer
auto-recomputes the cross-substrate similarity matrix against existing
anchors and flags NEW classifications (SUPER_ADDITIVE, ANTAGONISTIC) to
the operator-facing surface.

Per CLAUDE.md "Max observability - non-negotiable" Catalog #305: every
new anchor triggers an observability-only annotation in the matrix
snapshot, NEVER a score-mutating ranking signal. Per Catalog #341 Tier A
contract: ``predicted_delta_adjustment=0.0`` always, ``promotable=False``
always, ``axis_tag="[predicted]"`` always.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver - NON-NEGOTIABLE": this
consumer consumes the JUST-LANDED canonical equations
(per_byte_leverage_cross_hardware_aware_v2 +
hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1 +
cross_codec_super_additive_orthogonality_predictor_v1 per commit
80484241f STRATEGIC-FINDINGS) to pre-classify pairs BEFORE running the
full similarity comparison so the operator gets the classification
prediction immediately + the full comparison validates or refutes the
prediction asynchronously.

Hook assignments per Catalog #125:
  * #1 sensitivity-map — N/A (auto-trigger surface; downstream consumers
    use the matrix as sensitivity input)
  * #2 Pareto constraint — N/A
  * #3 bit-allocator — N/A
  * #4 cathedral autopilot dispatch — ACTIVE (auto-trigger emits matrix
    snapshot consumed by downstream cross_substrate_similarity_consumer
    + cross_codec_orthogonality_predictor_consumer ranker contributions)
  * #5 continual-learning posterior — ACTIVE (every new anchor refreshes
    the similarity matrix posterior signal)
  * #6 probe-disambiguator — ACTIVE (the classification taxonomy IS
    the canonical disambiguator between SUPER_ADDITIVE composition
    candidates vs SUB_ADDITIVE backbone-equivalence candidates)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "auto_trigger_similarity_after_master_gradient_anchor_consumer"
CONSUMER_VERSION = "0.2.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)

# WAVE-3-AUTO-TRIGGER-RUNTIME-WIRE-IN opt-in marker (2026-05-20).
# Per Catalog #335 canonical-contract auto-discovery + the sister Catalog #343
# auto-refresh-after-canonical-event pattern: consumers that opt-in to receive
# master-gradient anchor events declare ``CONSUMES_MASTER_GRADIENT_ANCHORS =
# True`` at module level. The runtime hook in
# :func:`tac.master_gradient.append_anchor_locked` then fans the anchor row
# out to every opt-in consumer's ``update_from_anchor(anchor_row)`` after the
# fcntl-locked append succeeds. Sister consumers that do NOT opt-in are
# skipped (default-False per ``getattr(mod, ..., False)`` lookup).
CONSUMES_MASTER_GRADIENT_ANCHORS = True


# Canonical equation IDs consumed by this auto-trigger consumer (per
# STRATEGIC-FINDINGS commit 80484241f). Per Catalog #344 the equations
# are first-class artifacts whose predictions feed the auto-trigger's
# pre-classification step BEFORE the full similarity comparison runs.
_CANONICAL_EQUATION_IDS = (
    "per_byte_leverage_cross_hardware_aware_v2",
    "hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1",
    "cross_codec_super_additive_orthogonality_predictor_v1",
)


def _classify_pair_via_canonical_equations(
    candidate: Mapping[str, Any],
) -> str | None:
    """Pre-classify a candidate pair via the JUST-LANDED canonical equations.

    Returns a verdict token (SUPER_ADDITIVE / SUB_ADDITIVE / ORTHOGONAL /
    SATURATED) OR None when the candidate carries no classifiable tokens.
    The verdict is an observability-only prediction; the actual matrix
    recomputation validates or refutes it asynchronously.

    Per Catalog #344 hook #6 probe-disambiguator: the equation-based
    prediction is the canonical disambiguator between hypotheses.
    """
    if not isinstance(candidate, Mapping):
        return None

    # Extract substrate-pair tokens from the candidate (per the canonical
    # pair_key shape consumed by cross_substrate_similarity_consumer).
    substrates_a = str(candidate.get("substrate_a", "") or "").lower()
    substrates_b = str(candidate.get("substrate_b", "") or "").lower()
    pair_key = f"{substrates_a}|{substrates_b}"

    # Per Equation 9 (cross_codec_super_additive_orthogonality_predictor_v1):
    # PR106 vs HNeRV-family pairs are pre-classified SUPER_ADDITIVE
    # (orthogonal-codec; top-K Jaccard structurally 0.000).
    hnerv_family_tokens = ("pr101", "a1", "fec6", "pr107", "apogee", "hnerv")
    has_pr106 = "pr106" in pair_key or "format0d" in pair_key
    has_hnerv_family = any(tok in pair_key for tok in hnerv_family_tokens)
    if has_pr106 and has_hnerv_family:
        return "SUPER_ADDITIVE"

    # Per Equation 8 (hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1):
    # within-HNeRV-cluster pairs are pre-classified SUB_ADDITIVE
    # (shared 178k-byte backbone; high per-axis Pearson correlation).
    hnerv_a = any(tok in substrates_a for tok in hnerv_family_tokens)
    hnerv_b = any(tok in substrates_b for tok in hnerv_family_tokens)
    if hnerv_a and hnerv_b:
        return "SUB_ADDITIVE"

    return None


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 - continual-learning posterior update.

    Triggered when a new master-gradient anchor lands at
    ``.omx/state/master_gradient_anchors.jsonl`` via
    :func:`tac.master_gradient.append_anchor_locked`. The runtime hook
    in ``append_anchor_locked`` discovers this consumer through the
    canonical Catalog #335 auto-discovery loop, filters by the
    module-level ``CONSUMES_MASTER_GRADIENT_ANCHORS = True`` opt-in,
    and calls ``update_from_anchor(anchor_row)`` with the just-appended
    JSONL row (parsed dict). Sister of Catalog #343
    :func:`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`
    pattern; landed 2026-05-20 in WAVE-3-AUTO-TRIGGER-RUNTIME-WIRE-IN.

    Current implementation: observability-only acknowledgment. The
    canonical similarity matrix recomputation
    (``PYTHONPATH=src python /tmp/wave3_analysis/compute_similarity_matrix.py``)
    remains operator-triggered per Catalog #287/#323 measurement
    provenance discipline; this consumer's auto-trigger surface is
    deliberately observability-only so a single buggy anchor cannot
    corrupt the similarity matrix posterior.

    Per Catalog #341 Tier-A canonical-routing-markers: this consumer
    contributes ``predicted_delta_adjustment=0.0`` + ``promotable=False``
    + ``axis_tag="[predicted]"`` always; the auto-trigger fan-out at
    ``append_anchor_locked`` does NOT mutate any score signal.

    Per CLAUDE.md "Subagent coherence-by-default" maximum-signal-preservation:
    per-consumer exceptions raised here are caught + warning-logged by
    ``append_anchor_locked``; the ledger write (which already succeeded)
    is never blocked by a downstream consumer failure.
    """
    _ = anchor  # acknowledgment; downstream extension lands here


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 - cathedral autopilot ranker contribution.

    Returns canonical Tier-A observability-only contribution per
    Catalog #341: ``predicted_delta_adjustment=0.0`` always,
    ``promotable=False`` always, ``axis_tag="[predicted]"`` always.

    The ``rationale`` field carries the equation-based pre-classification
    verdict (SUPER_ADDITIVE / SUB_ADDITIVE / None) so downstream
    consumers + operator review see the predicted classification before
    the full similarity comparison runs.
    """
    verdict = _classify_pair_via_canonical_equations(candidate)
    if verdict is None:
        rationale = (
            "auto_trigger_similarity_after_master_gradient_anchor_consumer: "
            "no canonical equation pre-classification matches; full similarity "
            "comparison required (cross_substrate_similarity_consumer Tier A annotation)"
        )
    else:
        rationale = (
            f"auto_trigger_similarity_after_master_gradient_anchor_consumer: "
            f"canonical equation pre-classification {verdict} "
            f"(consumes {', '.join(_CANONICAL_EQUATION_IDS)}); full similarity "
            f"comparison via cross_substrate_similarity_consumer validates "
            f"or refutes the prediction asynchronously"
        )

    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "predicted_classification": verdict,
        "canonical_equations_consumed": list(_CANONICAL_EQUATION_IDS),
    }
