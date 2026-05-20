# SPDX-License-Identifier: MIT
"""Canonical equation: procedural-codebook from-seed compression savings (v1).

Builder for ``procedural_codebook_from_seed_compression_savings_v1``, the
formal predictor for the *score-improvement-per-byte-saved when replacing
N codebook bytes with K seed bytes (K << N) via deterministic procedural
derivation*. Sister of
``master_gradient_null_space_byte_fraction_v1`` (the upstream identifier
of WHERE bytes can be replaced; this equation predicts the score
contribution of replacing them).

**Empirical anchor**: predicted-only at landing 2026-05-20; first
empirical anchor pending operator-routable NSCS06 v8 chroma-LUT paired
smoke per Catalog #325 (per-substrate symposium) + Catalog #324
(post-training Tier-C validation) + Catalog #272 (distinguishing-feature
integration contract — byte-mutation smoke MUST verify seed-derived
bytes affect rendered frames).

**Derivation rationale**: replacing N codebook bytes with K seed bytes
(K << N) saves ``N - K`` bytes from archive.zip. The canonical contest
formula (per CLAUDE.md "Submission auth eval" + ``upstream/evaluate.py``
line 63) charges:

    rate_term = 25 * archive_bytes / 37_545_489

so the score improvement is:

    ΔS = -25 * (N - K) / 37_545_489

(negative because the contest score is lower-is-better; removing bytes
DECREASES the rate term hence DECREASES the score).

Compliance via ``tac.procedural_codebook_generator.derive_codebook_from_seed``
keeps the seed bytes charged inside archive.zip while the generic derivation
routine remains ordinary inflate runtime code, per memo Q4 STRUCTURALLY
COMPLIANT verdict (distinguishable from rejected loophole-class pattern
PR #36/#38/#68/#69/#78/#87).

**Aggregate prediction** (per memo Top-3 #2): ~0.00264 ΔS per 4 KB
hoisted. Aggregate across 5 substrates with deterministic constants
(NSCS06 v8 chroma LUT / ATW V2 codec / TT5L / DP1 / sister) ≈ ΔS -0.013
(predicted; empirical anchor pending per-substrate smoke).

Producer/consumer wiring per CLAUDE.md "Subagent coherence-by-default"
6-hook discipline:

- Producer: ``tac.procedural_codebook_generator.derive_codebook_from_seed``
- Consumer #1: ``tac.cathedral_consumers.procedural_codebook_generator_consumer``
  (Tier A observability-only routing per Catalog #341; surfaces
  procedural-codebook candidate routing metadata)
- Consumer #2: per-substrate procedural-replacement smokes (NSCS06 v8
  chroma LUT + ATW V2 codec + TT5L + DP1 + sister)
- Consumer #3: ``tac.cathedral_consumers.null_byte_codebook_candidate_consumer``
  (sister consumer — identifies WHERE bytes can be replaced; this
  equation predicts the score contribution of replacing them)
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CanonicalEquation,
    DomainOfValidityViolation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.provenance.builders import build_provenance_for_predicted


# Canonical contest rate-term denominator per CLAUDE.md "Submission auth eval"
# + upstream/evaluate.py line 63. Bytes-inside-archive ARE charged at the rate
# `25 * archive_bytes / 37_545_489`.
CANONICAL_RATE_DENOM_BYTES = 37_545_489
CANONICAL_RATE_MULTIPLIER = 25.0

# Per-substrate predicted bytes-saved enumeration per memo §4
# (NSCS06 v8 chroma LUT ~4 KB → 32-byte seed = ~3.968 KB saved).
_NSCS06_V8_BYTES_SAVED = 4096 - 32
_NSCS06_V8_PREDICTED_DELTA_S = (
    -CANONICAL_RATE_MULTIPLIER * _NSCS06_V8_BYTES_SAVED / CANONICAL_RATE_DENOM_BYTES
)
_AGGREGATE_PREDICTED_DELTA_S = -0.013  # per memo Top-3 #2 quote

# ---------------------------------------------------------------------------
# Domain-of-validity refinement per WAVE-3-CANONICAL-EQUATION-26-DOMAIN-
# REFINEMENT 2026-05-20 (sister DWT-DETAIL-SUBBAND CPU smoke commit
# `f25f8cc1b`; KL=1.638 nats / 3.28σ empirical vindication that direct
# procedural-codebook byte substitution on DWT detail subbands corrupts
# inverse DWT). Catalog #344 sister discipline + Catalog #110/#113
# APPEND-ONLY HISTORICAL_PROVENANCE.
# ---------------------------------------------------------------------------

# Contexts where the canonical equation #26 producer
# (`derive_codebook_from_seed`) IS valid — the codebook is consumed at an
# INTERMEDIATE transform position (quantizer table / dequantizer LUT / class
# anchor) rather than directly substituted into pixel/coefficient bytes.
_INCLUDED_CONTEXTS = (
    "intermediate_transform_quantizer",
    "intermediate_transform_dequantizer",
    "procedural_codebook_as_lookup_table",
    "comma2k19_ood_derived_basis_replacement",
    "chroma_lut_replacement",
    "class_anchor_replacement",
    "nscs06_v8_chroma_lut",
    "atw_v2_codec_quantizer_lut",
    "tt5l_transformer_tokens",
    "dp1_codebook_bytes",
    "deterministic_constants_codebook_replacement",
)

# Contexts EXPLICITLY excluded — direct byte substitution on transform
# coefficients (DWT detail subbands / DCT coefficients / etc.) where the
# uniform-PRNG distributional mismatch (KL=1.638 nats / 3.28σ on the DWT
# anchor) corrupts the downstream inverse transform.
_EXCLUDED_CONTEXTS = (
    "direct_dwt_detail_subband_byte_substitution",
    "direct_byte_substitution_on_wavelet_decomposition_coefficients",
)

# Default context applied to legacy callers that don't pass an explicit
# context kwarg. Maps to the original IMPLICIT assumption from the equation's
# pre-refinement (registered 2026-05-20T22:37:45Z): intermediate-transform
# codebook substitution per memo §4.
_DEFAULT_CONTEXT = "intermediate_transform_quantizer"


def validate_context_is_in_domain(
    context: str | None = None,
    *,
    raise_on_excluded: bool = True,
) -> bool:
    """Verify a substrate context is within canonical equation #26's domain.

    Per WAVE-3-CANONICAL-EQUATION-26-DOMAIN-REFINEMENT 2026-05-20 + the
    DWT-DETAIL-SUBBAND CPU SMOKE empirical anchor (commit ``f25f8cc1b``;
    KL=1.638 nats / 3.28σ proving direct procedural-codebook substitution
    on DWT detail subbands corrupts inverse DWT). Equation
    ``procedural_codebook_from_seed_compression_savings_v1`` is valid in
    :data:`_INCLUDED_CONTEXTS` (intermediate-transform codebook positions)
    and EXPLICITLY invalid in :data:`_EXCLUDED_CONTEXTS` (direct byte
    substitution on transform coefficients).

    Args:
        context: a canonical context token (string). If ``None`` or empty,
            the default :data:`_DEFAULT_CONTEXT` (``intermediate_transform_quantizer``)
            is applied for backward-compat with legacy callers that
            predate this refinement.
        raise_on_excluded: when True (default), raises
            :class:`DomainOfValidityViolation` if the context is in
            :data:`_EXCLUDED_CONTEXTS`. When False, returns ``False``
            instead so callers can route the candidate to a non-promotable
            advisory surface per Catalog #341.

    Returns:
        True if ``context`` is in :data:`_INCLUDED_CONTEXTS`; False if the
        context is unknown (neither included nor excluded) OR if the
        context is excluded AND ``raise_on_excluded=False``.

    Raises:
        DomainOfValidityViolation: when ``context`` is in
            :data:`_EXCLUDED_CONTEXTS` AND ``raise_on_excluded=True``.
    """
    if context is None or not str(context).strip():
        context = _DEFAULT_CONTEXT
    ctx = str(context).strip().lower()
    if ctx in _EXCLUDED_CONTEXTS:
        if raise_on_excluded:
            raise DomainOfValidityViolation(
                f"context={context!r} is EXPLICITLY excluded from canonical "
                "equation procedural_codebook_from_seed_compression_savings_v1 "
                "domain_of_validity per WAVE-3 DWT-DETAIL-SUBBAND CPU smoke "
                "(commit f25f8cc1b; KL=1.638 nats / 3.28σ > 2σ threshold proves "
                "direct procedural-codebook substitution on DWT detail subbands "
                "corrupts inverse DWT). Re-scope the consumer to an "
                "INTERMEDIATE-TRANSFORM context (e.g., chroma_lut_replacement) "
                "or use a sister canonical equation. See "
                ".omx/research/dwt_bind_rescope_intermediate_transform_path_design_20260520.md"
            )
        return False
    if ctx in _INCLUDED_CONTEXTS:
        return True
    return False  # unknown context: not refused, not endorsed


def build_procedural_codebook_from_seed_compression_savings_v1() -> CanonicalEquation:
    """Equation: ΔS = -25 * (N_codebook - K_seed) / 37_545_489 per substrate.

    Per Catalog #344 canonical-equations-registry sister discipline +
    Catalog #318 (master-gradient raw-byte-authority guard — typed
    CandidateModificationSpec discipline preserved) +
    ``.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md``
    Q4 STRUCTURALLY COMPLIANT verdict.

    The equation has NO empirical anchor at landing — it is a
    *predicted-only* equation registered as a hypothesis pending the
    first NSCS06 v8 chroma-LUT paired smoke per memo §9 op-routable #3.
    The next operator-routable subagent (NSCS06 v8 substrate build)
    extends this equation via
    ``tac.canonical_equations.update_equation_with_empirical_anchor``
    once the smoke lands.

    The aggregate ΔS prediction of -0.013 across 5 substrates is captured
    as a hypothesis-anchor with ``residual=None`` (no empirical pairing
    yet); subsequent per-substrate anchors will populate the residual
    field as they land.
    """
    # The initial anchor captures the PREDICTED-ONLY hypothesis (per
    # CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"); the
    # `predicted_output` field holds the numerical prediction, and
    # `empirical_output` is intentionally empty to signal "pending".
    aggregate_hypothesis_anchor = EmpiricalAnchor(
        anchor_id="aggregate_5_substrate_hypothesis_pending_empirical_20260520",
        measurement_utc="2026-05-20T22:00:00Z",
        inputs={
            "substrate_class": "deterministic_constants_>2kb_per_archive",
            "substrate_count": 5,
            "candidate_substrates": [
                "nscs06_v8_chroma_lut",
                "atw_v2_codec",
                "tt5l_transformer_tokens",
                "dp1_codebook_bytes",  # NOTE: DIFFERENT class per memo §4 (OOD-derived)
                "sister_substrate_pending_identification",
            ],
            "predicted_seed_size_bytes": 32,
            "predicted_codebook_size_per_substrate_bytes_lower": 2048,
            "predicted_codebook_size_per_substrate_bytes_upper": 6144,
            "canonical_rate_denom_bytes": CANONICAL_RATE_DENOM_BYTES,
            "canonical_rate_multiplier": CANONICAL_RATE_MULTIPLIER,
            "axis_tag": "[predicted]",
            "evidence_grade": "predicted",
        },
        predicted_output={
            "aggregate_predicted_delta_s": _AGGREGATE_PREDICTED_DELTA_S,
            "per_substrate_predicted_delta_s_nscs06_v8": _NSCS06_V8_PREDICTED_DELTA_S,
            "per_substrate_predicted_bytes_saved_nscs06_v8": _NSCS06_V8_BYTES_SAVED,
            "hypothesis_status": "predicted_only_pending_empirical_anchor",
        },
        empirical_output={},  # intentionally empty — first anchor pending
        # `residual=0.0` per equation schema invariant (must be numeric +
        # non-negative). For this predicted-only anchor the residual
        # convention is 0.0 = "no empirical pairing yet — residual will
        # populate once first per-substrate smoke lands".
        residual=0.0,
        source_artifact=(
            ".omx/research/procedural_codebook_generator_null_exploit_design_20260520.md"
        ),
        measurement_method="canonical_contest_formula_25_times_bytes_saved_div_37545489_predicted_only",
        provenance=build_provenance_for_predicted(
            model_id="procedural_codebook_from_seed_compression_savings_predictor.v1",
            inputs_sha256="a1afce293533fbe1c1be67b626db9e532700e4ed66d84c62ed6d0bb67d15a1bc",
            measurement_axis="[predicted]",
            hardware_substrate="not_applicable_predicted_only_no_dispatch",
        ),
    )

    return CanonicalEquation(
        equation_id="procedural_codebook_from_seed_compression_savings_v1",
        name="Procedural-codebook from-seed compression savings",
        one_line_summary=(
            "Predicted ΔS = -25 * (N_codebook - K_seed) / 37_545_489 per substrate "
            "when replacing N codebook bytes with K seed bytes via deterministic "
            "procedural derivation per tac.procedural_codebook_generator."
        ),
        latex_form=(
            r"\Delta S_{\text{procedural}} = "
            r"-\frac{25 \cdot (N_{\text{codebook}} - K_{\text{seed}})}{37{,}545{,}489}"
        ),
        python_callable_module_path=(
            "tac.procedural_codebook_generator:derive_codebook_from_seed"
        ),
        domain_of_validity={
            "substrate_classes": [
                "nscs06_v8_chroma_lut",
                "atw_v2_codec",
                "tt5l_transformer_tokens",
                "dp1_codebook_bytes",
                "deterministic_constants_>2kb",
            ],
            "codebook_size_bytes_range": [256, 65536],
            "seed_size_bytes_range": [8, 256],
            "generator_kinds": ["xorshift", "lcg", "pcg64"],
            "measurement_axes": ["[contest-CUDA]", "[contest-CPU]"],
            "empirical_anchor_status": "predicted_only_at_landing_2026_05_20",
            # WAVE-3-CANONICAL-EQUATION-26-DOMAIN-REFINEMENT 2026-05-20:
            # canonical refinement fields populated at builder construction
            # so the FIRST `registered` event already carries the refined
            # surface. Sister DWT-DETAIL-SUBBAND CPU smoke (commit f25f8cc1b)
            # empirically vindicated the EXCLUDED contexts list.
            "domain_of_validity_included": list(_INCLUDED_CONTEXTS),
            "domain_of_validity_excluded": list(_EXCLUDED_CONTEXTS),
            "default_context_when_legacy_caller_omits": _DEFAULT_CONTEXT,
        },
        units_in={
            "n_codebook_bytes": "int_archive_member_bytes_count",
            "k_seed_bytes": "int_archive_member_bytes_count",
            "substrate_id": "string_canonical_substrate_identifier",
            "generator_kind": "string_xorshift_lcg_pcg64",
        },
        units_out={
            "predicted_delta_s_signed_float": "float_score_axis_delta_lower_is_better",
            "predicted_bytes_saved": "int_archive_member_bytes_count",
            "predicted_rate_term_decrease": "float_25x_bytes_saved_div_denom",
        },
        empirical_anchors=(aggregate_hypothesis_anchor,),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-20T22:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.cathedral_consumers.procedural_codebook_generator_consumer",
            "tac.cathedral_consumers.null_byte_codebook_candidate_consumer",
            "per_substrate_procedural_replacement_smokes_nscs06_v8_atw_v2_tt5l_dp1_sister",
        ),
        canonical_producers=(
            "tac.procedural_codebook_generator.derive_codebook_from_seed",
        ),
        provenance=build_provenance_for_predicted(
            model_id="procedural_codebook_from_seed_compression_savings_predictor.v1",
            inputs_sha256="a1afce293533fbe1c1be67b626db9e532700e4ed66d84c62ed6d0bb67d15a1bc",
            measurement_axis="[predicted]",
            hardware_substrate="not_applicable_predicted_only_no_dispatch",
        ),
    )
