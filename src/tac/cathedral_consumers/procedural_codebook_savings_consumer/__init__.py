# SPDX-License-Identifier: MIT
"""Cathedral consumer for procedural-codebook savings prediction.

Sister of :mod:`tac.cathedral_consumers.procedural_codebook_generator_consumer`
(the AUTHORITY-PACKET routing surface per Catalog #329; this consumer is
the SAVINGS-PREDICTION surface per canonical equation
``procedural_codebook_from_seed_compression_savings_v1``).

Both consumers serve complementary roles in the cathedral autopilot's
procedural-codebook routing pipeline:

* **procedural_codebook_generator_consumer** (sister; Catalog #329 +
  ProvenanceKind PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED) — surfaces
  authority packet metadata (archive-seed inclusion + procedural seed
  mode classification + denied-uses contract). Hook surfaces: #4 + #6.

* **procedural_codebook_savings_consumer** (THIS module; Catalog #344
  canonical equation
  ``procedural_codebook_from_seed_compression_savings_v1``) — surfaces
  predicted ΔS from canonical-formula savings calculation per memo §4
  ``25 * bytes_saved / 37_545_489``. Hook surfaces: #1 + #3 + #4 + #5.

Per WAVE-3-PROCEDURAL-CODEBOOK-GENERATOR-BUILD task spec 2026-05-20 +
``.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md``
Top-3 op-routable #1 (Q5 follow-on of canonical investigation
``.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
Q4 STRUCTURALLY COMPLIANT verdict).

Operationalizes the canonical helper
:func:`tac.procedural_codebook_generator.derive_codebook_from_seed` as
the cathedral autopilot routing surface for substrates with deterministic
codebook constants (>2 KB) that can be replaced with seed-derived bytes
per the null-exploit class.

Per Catalog #335 canonical Protocol contract + Catalog #341 Tier A
canonical routing markers (``predicted_delta_adjustment=0.0``,
``promotable=False``, ``axis_tag="[predicted]"``) + Catalog #357 Tier
classification (Tier A observability-only by construction; Tier B
score-contributing requires per-substrate symposium per Catalog #325 +
post-training Tier-C validation per Catalog #324 BEFORE elevation).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287
canonical Provenance discipline: every savings prediction is
``predicted`` evidence-grade (not contest-CUDA / contest-CPU /
empirical); promotion requires per-substrate paired Linux x86_64 +
NVIDIA contest-CUDA empirical anchors per CLAUDE.md "Submission auth
eval — BOTH CPU AND CUDA" non-negotiable.

Hook assignments per Catalog #125:

* #1 sensitivity-map = ACTIVE (seed-derived codebook size + predicted
  bytes-saved contribute to the bit-allocator's "where can I take bytes
  from?" surface)
* #2 Pareto constraint = N/A (observability routing; no Pareto-feasible-
  region constraint added)
* #3 bit-allocator = ACTIVE (canonical formula ``25 * bytes_saved /
  37_545_489`` per-substrate prediction informs the bit-allocator)
* #4 cathedral autopilot dispatch = ACTIVE PRIMARY (auto-discovered per
  Catalog #335 + emits Tier A routing markers per Catalog #341)
* #5 continual-learning posterior = ACTIVE (auto-trigger via
  ``CONSUMES_MASTER_GRADIENT_ANCHORS = True``; canonical equation
  extends via
  :func:`tac.canonical_equations.update_equation_with_empirical_anchor`
  when per-substrate smokes land)
* #6 probe-disambiguator = N/A (sister consumer
  procedural_codebook_generator_consumer is the canonical disambiguator
  via authority packet kind taxonomy)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "procedural_codebook_savings_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)

# WAVE-3-AUTO-TRIGGER-RUNTIME-WIRE-IN opt-in marker (per sister
# auto_trigger_similarity_after_master_gradient_anchor_consumer pattern).
# Per Catalog #335 + #354 sister discipline + canonical
# CONSUMES_MASTER_GRADIENT_ANCHORS pattern from sister
# null_byte_codebook_candidate_consumer.
CONSUMES_MASTER_GRADIENT_ANCHORS = True

# Canonical contest rate-term per CLAUDE.md "Submission auth eval" +
# upstream/evaluate.py line 63.
_CANONICAL_RATE_DENOM_BYTES = 37_545_489
_CANONICAL_RATE_MULTIPLIER = 25.0

# Threshold below which a candidate is "too small to be worth inflate.py
# LOC budget" per memo §4 (FEC6 selector_payload 249 B saves
# ~-0.000166 ΔS which is below operator-attention threshold).
_MIN_BYTES_SAVED_FOR_ACTIONABLE_CANDIDATE = 256

# Candidate payload keys this consumer recognizes (typically populated
# by a sister probe tool that emits seed-derivation candidate metadata).
_PROCEDURAL_CANDIDATE_KEYS = (
    "procedural_codebook_savings_candidate",
    "procedural_codebook_candidate",
    "seed_derived_codebook_candidate",
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update entrypoint.

    A new master-gradient anchor lands → this consumer's next
    ``consume_candidate`` call has access to the freshest per-byte
    sensitivity surface (via the candidate row's
    ``null_byte_probe_json`` / inline payload). There is no consumer-
    local cache to mutate when an anchor lands; the next
    ``consume_candidate`` re-derives against canonical sources.
    """
    _ = anchor


def _no_signal(reason: str) -> Mapping[str, Any]:
    """Tier A canonical no-signal return (per Catalog #341)."""
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"procedural-codebook savings consumer: {reason} [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_signal_kind": "procedural_codebook_savings_absent",
    }


def _load_candidate_payload(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Locate the procedural-codebook savings candidate payload."""
    for key in _PROCEDURAL_CANDIDATE_KEYS:
        value = candidate.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _compute_predicted_savings(
    n_codebook_bytes: int,
    k_seed_bytes: int,
) -> dict[str, Any]:
    """Per canonical equation procedural_codebook_from_seed_compression_savings_v1.

    Returns the canonical predicted savings dict per
    :mod:`tac.canonical_equations.procedural_codebook_savings`.
    """
    bytes_saved = max(0, n_codebook_bytes - k_seed_bytes)
    rate_term_decrease = (
        _CANONICAL_RATE_MULTIPLIER * bytes_saved / _CANONICAL_RATE_DENOM_BYTES
    )
    # Score is lower-is-better; removing bytes DECREASES score
    predicted_delta_s = -rate_term_decrease
    return {
        "n_codebook_bytes": n_codebook_bytes,
        "k_seed_bytes": k_seed_bytes,
        "bytes_saved": bytes_saved,
        "rate_term_decrease": rate_term_decrease,
        "predicted_delta_s": predicted_delta_s,
        "actionable": bytes_saved >= _MIN_BYTES_SAVED_FOR_ACTIONABLE_CANDIDATE,
        "canonical_equation_id": "procedural_codebook_from_seed_compression_savings_v1",
    }


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — surface procedural-codebook savings prediction.

    Returns Tier A canonical-routing markers per Catalog #341 — the
    savings prediction is OBSERVABILITY ONLY; the actual score-mutating
    mechanism is the per-substrate substrate-trainer + inflate-runtime
    integration, which remains gated by Catalog #325 per-substrate
    symposium + Catalog #324 post-training Tier-C validation per
    CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY".
    """
    if not isinstance(candidate, Mapping):
        return _no_signal("candidate is not a mapping")
    payload = _load_candidate_payload(candidate)
    if payload is None:
        return _no_signal(
            "no procedural_codebook_savings_candidate payload on candidate row"
        )

    substrate_id = payload.get("substrate_id", "unknown_substrate")
    n_codebook_bytes = payload.get("n_codebook_bytes")
    k_seed_bytes = payload.get("k_seed_bytes")
    generator_kind = payload.get("generator_kind", "pcg64")

    if not isinstance(n_codebook_bytes, int) or n_codebook_bytes <= 0:
        return _no_signal(
            f"payload missing/invalid n_codebook_bytes (got {n_codebook_bytes!r})"
        )
    if not isinstance(k_seed_bytes, int) or k_seed_bytes <= 0:
        return _no_signal(
            f"payload missing/invalid k_seed_bytes (got {k_seed_bytes!r})"
        )

    savings = _compute_predicted_savings(
        n_codebook_bytes=n_codebook_bytes,
        k_seed_bytes=k_seed_bytes,
    )
    rationale = (
        f"procedural-codebook savings prediction for substrate={substrate_id}; "
        f"N_codebook={n_codebook_bytes} -> K_seed={k_seed_bytes} "
        f"(generator={generator_kind}); predicted_delta_s="
        f"{savings['predicted_delta_s']:+.6f}; "
        f"actionable={savings['actionable']}; "
        "promotion gated by Catalog #325 + #324 + #272 [predicted]"
    )

    return {
        # Catalog #341 Tier A canonical routing markers (observability-only)
        "predicted_delta_adjustment": 0.0,
        "rationale": rationale,
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_signal_kind": "procedural_codebook_savings_routing",
        # Operator-facing summary surfaces
        "substrate_id": substrate_id,
        "n_codebook_bytes": n_codebook_bytes,
        "k_seed_bytes": k_seed_bytes,
        "bytes_saved": savings["bytes_saved"],
        "predicted_delta_s_per_canonical_equation": savings["predicted_delta_s"],
        "actionable_above_min_bytes_saved_threshold": savings["actionable"],
        "generator_kind": generator_kind,
        # Cite-chain (Catalog #305 observability surface)
        "canonical_equation_id": savings["canonical_equation_id"],
        "canonical_producer": "tac.procedural_codebook_generator.derive_codebook_from_seed",
        "compliance_citation_chain": (
            "upstream/evaluate.py:63_rate_charges_archive_zip_bytes_only + "
            "memo_q4_structurally_compliant_verdict_2026_05_18 + "
            "catalog_213_comma2k19_canonical_helper_sister_pattern + "
            "catalog_272_byte_mutation_smoke_required_for_promotion + "
            "catalog_318_master_gradient_raw_byte_authority_guard"
        ),
        "promotion_gates": (
            "catalog_325_per_substrate_symposium + "
            "catalog_324_post_training_tier_c_validation + "
            "catalog_272_distinguishing_feature_byte_mutation_smoke"
        ),
    }
