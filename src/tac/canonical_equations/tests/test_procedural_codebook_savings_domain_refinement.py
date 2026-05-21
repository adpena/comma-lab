# SPDX-License-Identifier: MIT
"""Tests for WAVE-3-CANONICAL-EQUATION-26-DOMAIN-REFINEMENT 2026-05-20.

Covers canonical equation #26 (``procedural_codebook_from_seed_compression_savings_v1``)
domain-of-validity refinement landing per the sister DWT-DETAIL-SUBBAND CPU
SMOKE empirical anchor (commit ``f25f8cc1b``; KL=1.638 nats / 3.28σ proving
direct procedural-codebook substitution on DWT detail subbands corrupts
inverse DWT).

Test surfaces:

* ``validate_context_is_in_domain`` happy path (intermediate transform).
* ``validate_context_is_in_domain`` forbidden path (DWT detail subband)
  raises :class:`DomainOfValidityViolation`.
* ``_INCLUDED_CONTEXTS`` + ``_EXCLUDED_CONTEXTS`` constants present + populated.
* Builder return value carries the refined fields.
* Cathedral consumer rejects out-of-domain candidate with canonical
  Catalog #341 markers + ``[predicted_domain_violated]`` axis tag.
* ``domain_refined`` event_type round-trip via
  :func:`tac.canonical_equations.load_registry_events_lenient`.
* Backward-compat: legacy callers without explicit context still work.
* Sister regression: existing equation #26 + consumer tests still PASS.

Cross-refs:
  * CLAUDE.md "Canonical equations + models registry" non-negotiable.
  * Catalog #344 sister discipline (canonical equation registry).
  * Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.
  * Catalog #341 Tier A canonical routing markers.
  * Catalog #287 placeholder-rationale rejection.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.canonical_equations import (
    DomainOfValidityViolation,
    EVENT_DOMAIN_REFINED,
    InvalidEquationError,
    VALID_EVENT_TYPES,
    get_equation_by_id,
    load_registry_events_lenient,
    register_canonical_equation,
    update_equation_with_domain_refinement,
)
from tac.canonical_equations.procedural_codebook_savings import (
    _DEFAULT_CONTEXT,
    _EXCLUDED_CONTEXTS,
    _INCLUDED_CONTEXTS,
    build_procedural_codebook_from_seed_compression_savings_v1,
    validate_context_is_in_domain,
)


EQUATION_ID = "procedural_codebook_from_seed_compression_savings_v1"


# ---------------------------------------------------------------------------
# Canonical constants present + populated
# ---------------------------------------------------------------------------


def test_event_domain_refined_present_in_valid_event_types() -> None:
    """``domain_refined`` event-type is part of the canonical taxonomy."""
    assert EVENT_DOMAIN_REFINED == "domain_refined"
    assert EVENT_DOMAIN_REFINED in VALID_EVENT_TYPES


def test_included_contexts_constant_populated() -> None:
    """``_INCLUDED_CONTEXTS`` covers the canonical intermediate-transform set."""
    assert isinstance(_INCLUDED_CONTEXTS, tuple)
    assert len(_INCLUDED_CONTEXTS) >= 5
    assert "intermediate_transform_quantizer" in _INCLUDED_CONTEXTS
    assert "chroma_lut_replacement" in _INCLUDED_CONTEXTS
    assert "nscs06_v8_chroma_lut" in _INCLUDED_CONTEXTS


def test_excluded_contexts_constant_populated_with_dwt_anchor() -> None:
    """``_EXCLUDED_CONTEXTS`` covers the DWT detail-subband empirical anchor."""
    assert isinstance(_EXCLUDED_CONTEXTS, tuple)
    assert "direct_dwt_detail_subband_byte_substitution" in _EXCLUDED_CONTEXTS
    assert (
        "direct_byte_substitution_on_wavelet_decomposition_coefficients"
        in _EXCLUDED_CONTEXTS
    )


def test_default_context_is_intermediate_transform_quantizer() -> None:
    """Backward-compat default for legacy callers per builder docstring."""
    assert _DEFAULT_CONTEXT == "intermediate_transform_quantizer"
    assert _DEFAULT_CONTEXT in _INCLUDED_CONTEXTS


# ---------------------------------------------------------------------------
# validate_context_is_in_domain helper
# ---------------------------------------------------------------------------


def test_validate_context_intermediate_transform_returns_true() -> None:
    assert validate_context_is_in_domain("intermediate_transform_quantizer") is True
    assert validate_context_is_in_domain("chroma_lut_replacement") is True
    assert validate_context_is_in_domain("nscs06_v8_chroma_lut") is True


def test_validate_context_dwt_detail_subband_raises() -> None:
    """Excluded context raises :class:`DomainOfValidityViolation`."""
    with pytest.raises(DomainOfValidityViolation, match="EXPLICITLY excluded"):
        validate_context_is_in_domain(
            "direct_dwt_detail_subband_byte_substitution"
        )


def test_validate_context_dwt_detail_subband_does_not_raise_when_raise_false() -> None:
    """``raise_on_excluded=False`` returns False instead of raising."""
    result = validate_context_is_in_domain(
        "direct_dwt_detail_subband_byte_substitution",
        raise_on_excluded=False,
    )
    assert result is False


def test_validate_context_unknown_returns_false_without_raise() -> None:
    """Unknown context (neither included nor excluded) returns False."""
    result = validate_context_is_in_domain("unknown_future_context_v99")
    assert result is False


def test_validate_context_none_applies_default() -> None:
    """``None`` / empty context applies :data:`_DEFAULT_CONTEXT` (legacy compat)."""
    assert validate_context_is_in_domain(None) is True
    assert validate_context_is_in_domain("") is True
    assert validate_context_is_in_domain("   ") is True


def test_validate_context_case_insensitive() -> None:
    """Context comparison normalizes case (canonical lowercase)."""
    assert validate_context_is_in_domain("CHROMA_LUT_REPLACEMENT") is True
    assert validate_context_is_in_domain("Chroma_Lut_Replacement") is True


def test_validate_context_excluded_raises_with_anchor_citation() -> None:
    """Excluded context error message cites empirical anchor commit + memo."""
    with pytest.raises(DomainOfValidityViolation) as excinfo:
        validate_context_is_in_domain(
            "direct_byte_substitution_on_wavelet_decomposition_coefficients"
        )
    msg = str(excinfo.value)
    assert "f25f8cc1b" in msg
    assert "1.638" in msg
    assert "dwt_bind_rescope_intermediate_transform_path_design" in msg


# ---------------------------------------------------------------------------
# Builder return value
# ---------------------------------------------------------------------------


def test_builder_populates_refined_domain_fields() -> None:
    """Builder return value carries refined ``domain_of_validity_*`` fields."""
    eq = build_procedural_codebook_from_seed_compression_savings_v1()
    dov = eq.domain_of_validity
    assert "domain_of_validity_included" in dov
    assert "domain_of_validity_excluded" in dov
    assert "default_context_when_legacy_caller_omits" in dov
    assert "direct_dwt_detail_subband_byte_substitution" in dov[
        "domain_of_validity_excluded"
    ]
    assert "intermediate_transform_quantizer" in dov["domain_of_validity_included"]
    # Legacy fields preserved.
    assert "substrate_classes" in dov
    assert "generator_kinds" in dov


def test_builder_canonical_helper_callable_path() -> None:
    """Builder still declares canonical producer."""
    eq = build_procedural_codebook_from_seed_compression_savings_v1()
    assert (
        eq.python_callable_module_path
        == "tac.procedural_codebook_generator:derive_codebook_from_seed"
    )


# ---------------------------------------------------------------------------
# update_equation_with_domain_refinement helper
# ---------------------------------------------------------------------------


def _register_minimal_eq(path: Path, lock: Path) -> None:
    """Helper: register a minimal equation in a tmp registry for testing."""
    from tac.canonical_equations import (
        CanonicalEquation,
        RECALIBRATE_ON_NEW_ANCHORS,
    )
    from tac.provenance.builders import build_provenance_for_predicted

    eq = CanonicalEquation(
        equation_id="test_domain_refinement_v1",
        name="Test domain refinement",
        one_line_summary="Test equation for domain refinement helper",
        latex_form=r"y = mx + b",
        python_callable_module_path="tac.foo:bar",
        domain_of_validity={"axis": "test"},
        units_in={"x": "float"},
        units_out={"y": "float"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-05-20T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.foo",),
        canonical_producers=(),
        provenance=build_provenance_for_predicted(
            model_id="test.v1",
            inputs_sha256="0" * 64,
        ),
    )
    register_canonical_equation(
        eq, path=path, lock_path=lock, agent="test", subagent_id="test"
    )


def test_update_equation_with_domain_refinement_appends_new_event(tmp_path: Path) -> None:
    """Helper appends a ``domain_refined`` event preserving prior rows."""
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    _register_minimal_eq(path, lock)

    updated = update_equation_with_domain_refinement(
        "test_domain_refinement_v1",
        domain_of_validity_extension={
            "domain_of_validity_included": ["context_a", "context_b"],
            "domain_of_validity_excluded": ["context_z"],
        },
        rationale="empirical anchor proves context_z is excluded; KL=1.0 nats",
        path=path,
        lock_path=lock,
    )
    assert updated.domain_of_validity["domain_of_validity_included"] == [
        "context_a",
        "context_b",
    ]
    assert updated.domain_of_validity["domain_of_validity_excluded"] == ["context_z"]
    # Original axis field preserved.
    assert updated.domain_of_validity["axis"] == "test"
    # Rationale recorded.
    assert (
        updated.domain_of_validity["last_domain_refinement_rationale"]
        == "empirical anchor proves context_z is excluded; KL=1.0 nats"
    )

    rows = load_registry_events_lenient(path)
    assert len(rows) == 2  # registered + domain_refined
    assert rows[0]["event_type"] == "registered"
    assert rows[1]["event_type"] == EVENT_DOMAIN_REFINED
    # APPEND-ONLY: prior registered row's payload preserved verbatim.
    assert rows[0]["equation_payload"]["domain_of_validity"] == {"axis": "test"}


def test_update_equation_with_domain_refinement_rejects_placeholder_rationale(
    tmp_path: Path,
) -> None:
    """Placeholder rationale literal rejected per Catalog #287."""
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    _register_minimal_eq(path, lock)

    for placeholder in ("<rationale>", "<reason>", "<rationale_here>"):
        with pytest.raises(InvalidEquationError, match="placeholder literal"):
            update_equation_with_domain_refinement(
                "test_domain_refinement_v1",
                domain_of_validity_extension={"k": "v"},
                rationale=placeholder,
                path=path,
                lock_path=lock,
            )


def test_update_equation_with_domain_refinement_rejects_short_rationale(
    tmp_path: Path,
) -> None:
    """Rationale shorter than 4 chars rejected."""
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    _register_minimal_eq(path, lock)

    with pytest.raises(InvalidEquationError, match="substantive non-placeholder"):
        update_equation_with_domain_refinement(
            "test_domain_refinement_v1",
            domain_of_validity_extension={"k": "v"},
            rationale="abc",
            path=path,
            lock_path=lock,
        )


def test_update_equation_with_domain_refinement_rejects_unknown_equation(
    tmp_path: Path,
) -> None:
    """Unknown equation_id raises InvalidEquationError."""
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    _register_minimal_eq(path, lock)

    with pytest.raises(InvalidEquationError, match="not found in registry"):
        update_equation_with_domain_refinement(
            "nonexistent_equation_v1",
            domain_of_validity_extension={"k": "v"},
            rationale="this rationale is long enough to pass validation",
            path=path,
            lock_path=lock,
        )


def test_update_equation_with_domain_refinement_requires_mapping_extension(
    tmp_path: Path,
) -> None:
    """``domain_of_validity_extension`` must be a Mapping."""
    path = tmp_path / "registry.jsonl"
    lock = path.with_suffix(path.suffix + ".lock")
    _register_minimal_eq(path, lock)

    with pytest.raises(InvalidEquationError, match="must be a Mapping"):
        update_equation_with_domain_refinement(
            "test_domain_refinement_v1",
            domain_of_validity_extension=["not", "a", "mapping"],  # type: ignore[arg-type]
            rationale="this rationale is long enough to pass validation",
            path=path,
            lock_path=lock,
        )


# ---------------------------------------------------------------------------
# Live-repo regression guards
# ---------------------------------------------------------------------------


def test_live_registry_has_domain_refined_event_for_eq26() -> None:
    """The persisted registry has the canonical equation #26 ``domain_refined`` event."""
    rows = load_registry_events_lenient()
    eq26 = [
        r for r in rows if r.get("equation_id") == EQUATION_ID
    ]
    domain_events = [
        r for r in eq26 if r.get("event_type") == EVENT_DOMAIN_REFINED
    ]
    assert len(domain_events) >= 1, (
        f"live registry must have at least 1 domain_refined event for {EQUATION_ID}; "
        f"found {len(domain_events)} (eq26 rows total: {len(eq26)})"
    )
    # Latest domain_refined payload carries the DWT exclusion.
    latest = domain_events[-1]
    dov = latest["equation_payload"]["domain_of_validity"]
    assert "direct_dwt_detail_subband_byte_substitution" in dov.get(
        "domain_of_validity_excluded", []
    )


def test_live_registry_latest_eq26_payload_carries_refinement_fields() -> None:
    """Latest payload (via ``get_equation_by_id``) carries refined fields."""
    eq = get_equation_by_id(EQUATION_ID)
    assert eq is not None
    dov = eq.domain_of_validity
    assert "domain_of_validity_included" in dov
    assert "domain_of_validity_excluded" in dov
    assert "direct_dwt_detail_subband_byte_substitution" in dov[
        "domain_of_validity_excluded"
    ]
    assert "intermediate_transform_quantizer" in dov["domain_of_validity_included"]
    # APPEND-ONLY discipline: anchor count preserved (3 anchors → wait,
    # actually we have 2 anchors at registration time; no anchors lost
    # during domain_refined event).
    assert len(eq.empirical_anchors) >= 2


# ---------------------------------------------------------------------------
# Cathedral consumer integration
# ---------------------------------------------------------------------------


def test_consumer_rejects_dwt_detail_subband_candidate_with_canonical_markers() -> None:
    """Out-of-domain candidate gets ``[predicted_domain_violated]`` + non-promotable markers."""
    from tac.cathedral_consumers import (
        procedural_codebook_savings_consumer as M,
    )

    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "dwt_hnerv_ll_l0_scaffold",
                "n_codebook_bytes": 4096,
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
                "substrate_context": "direct_dwt_detail_subband_byte_substitution",
            }
        }
    )
    # Canonical Catalog #341 non-promotable markers
    assert row["predicted_delta_adjustment"] == 0.0
    assert row["promotable"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    # Augmented axis tag
    assert row["axis_tag"] == "[predicted_domain_violated]"
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_domain_violated"
    # Operator-facing surface
    assert row["substrate_id"] == "dwt_hnerv_ll_l0_scaffold"
    assert (
        row["context_attempted"]
        == "direct_dwt_detail_subband_byte_substitution"
    )
    assert (
        row["canonical_equation_id"]
        == "procedural_codebook_from_seed_compression_savings_v1"
    )
    # Cite-chain back to empirical anchor + re-scope design memo
    assert "dwt_detail_subband_procedural_smoke" in row["empirical_anchor_citation"]
    assert (
        "dwt_bind_rescope_intermediate_transform_path_design"
        in row["re_scope_design_memo"]
    )


def test_consumer_accepts_intermediate_transform_candidate() -> None:
    """In-domain candidate gets canonical ``[predicted]`` axis + classification surface."""
    from tac.cathedral_consumers import (
        procedural_codebook_savings_consumer as M,
    )

    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "nscs06_v8_chroma_lut",
                "n_codebook_bytes": 4096,
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
                "substrate_context": "chroma_lut_replacement",
            }
        }
    )
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_routing"
    assert row["axis_tag"] == "[predicted]"
    assert row["domain_of_validity_classification"]["domain_classification"] == (
        "in_included_contexts"
    )


def test_consumer_unknown_context_routes_to_domain_uncertain() -> None:
    """Unknown context falls back to canonical formula with uncertain tag."""
    from tac.cathedral_consumers import (
        procedural_codebook_savings_consumer as M,
    )

    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "exotic_substrate",
                "n_codebook_bytes": 4096,
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
                "substrate_context": "exotic_unknown_context_v99",
            }
        }
    )
    # Still produces savings prediction (canonical formula applied)
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_routing"
    # ... but tagged uncertain
    assert row["axis_tag"] == "[predicted_domain_uncertain]"
    assert row["domain_of_validity_classification"]["domain_classification"] == (
        "uncertain_not_in_included_or_excluded"
    )


def test_consumer_legacy_caller_without_context_defaults_to_included() -> None:
    """Legacy caller omitting substrate_context applies default = intermediate_transform_quantizer."""
    from tac.cathedral_consumers import (
        procedural_codebook_savings_consumer as M,
    )

    # Note: no substrate_context key
    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "legacy_substrate",
                "n_codebook_bytes": 4096,
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
            }
        }
    )
    # Legacy default = intermediate_transform_quantizer → IN INCLUDED
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_routing"
    assert row["axis_tag"] == "[predicted]"
    assert (
        row["domain_of_validity_classification"]["context_attempted"]
        == "intermediate_transform_quantizer"
    )


def test_consumer_rejects_sister_excluded_token() -> None:
    """Sister excluded token also refused with canonical markers."""
    from tac.cathedral_consumers import (
        procedural_codebook_savings_consumer as M,
    )

    row = M.consume_candidate(
        {
            "procedural_codebook_savings_candidate": {
                "substrate_id": "dwt_hl_subband_substrate",
                "n_codebook_bytes": 4096,
                "k_seed_bytes": 32,
                "generator_kind": "pcg64",
                "substrate_context": (
                    "direct_byte_substitution_on_wavelet_decomposition_coefficients"
                ),
            }
        }
    )
    assert row["axis_tag"] == "[predicted_domain_violated]"
    assert row["consumer_signal_kind"] == "procedural_codebook_savings_domain_violated"


# ---------------------------------------------------------------------------
# RATIFY-4 / WAVE-3-ATW-V2-CDF-TABLE-BLOB-RECONCILIATION 2026-05-21
# Catalog #344 NEW EXCLUDED context registration
# `direct_byte_substitution_on_decode_opaque_raw_sections`.
# Sister memo: atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md §5
# Codex empirical anchor: commit 057130de4 max_abs_raw_byte_delta=0 across 2,560 bytes
# ---------------------------------------------------------------------------


def test_ratify_4_decode_opaque_excluded_context_present_in_source() -> None:
    """Source-level `_EXCLUDED_CONTEXTS` includes the RATIFY-4 NEW context."""
    assert (
        "direct_byte_substitution_on_decode_opaque_raw_sections"
        in _EXCLUDED_CONTEXTS
    )


def test_ratify_4_decode_opaque_context_raises_via_canonical_helper() -> None:
    """The NEW EXCLUDED context refuses canonical equation #26 routing."""
    with pytest.raises(DomainOfValidityViolation, match="EXPLICITLY excluded"):
        validate_context_is_in_domain(
            "direct_byte_substitution_on_decode_opaque_raw_sections"
        )


def test_ratify_4_decode_opaque_context_returns_false_when_raise_false() -> None:
    """`raise_on_excluded=False` returns False for the NEW EXCLUDED context."""
    result = validate_context_is_in_domain(
        "direct_byte_substitution_on_decode_opaque_raw_sections",
        raise_on_excluded=False,
    )
    assert result is False


def test_ratify_4_decode_opaque_context_case_insensitive() -> None:
    """Canonical lowercase comparison applies to the NEW EXCLUDED context."""
    with pytest.raises(DomainOfValidityViolation):
        validate_context_is_in_domain(
            "DIRECT_BYTE_SUBSTITUTION_ON_DECODE_OPAQUE_RAW_SECTIONS"
        )
    with pytest.raises(DomainOfValidityViolation):
        validate_context_is_in_domain(
            "Direct_Byte_Substitution_On_Decode_Opaque_Raw_Sections"
        )


def test_ratify_4_decode_opaque_context_distinct_from_parser_safe_score_affecting() -> None:
    """The NEW context is DISTINCT from sister
    `direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections`.

    Both refuse equation #26 but for different reasons:
      * parser_safe_but_score_affecting: bytes ARE consumed at decode time
        but contribute to score via the canonical scorer's input pipeline
        (e.g. PR101 magic header).
      * decode_opaque: bytes are parser-visible (static parse-safe per the
        canonical helper) but CANNOT influence the rendered frame because
        the runtime never consumes them (codex byte-mutation smoke
        max_abs_raw_byte_delta=0 across all mutated bytes).
    """
    assert (
        "direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections"
        in _EXCLUDED_CONTEXTS
    )
    assert (
        "direct_byte_substitution_on_decode_opaque_raw_sections"
        in _EXCLUDED_CONTEXTS
    )
    # Verify the two contexts are stored as DISTINCT tuple entries (no
    # accidental alias / typo collapsing them).
    decode_opaque_count = sum(
        1
        for c in _EXCLUDED_CONTEXTS
        if c == "direct_byte_substitution_on_decode_opaque_raw_sections"
    )
    parser_safe_count = sum(
        1
        for c in _EXCLUDED_CONTEXTS
        if c
        == "direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections"
    )
    assert decode_opaque_count == 1
    assert parser_safe_count == 1


def test_ratify_4_builder_propagates_new_excluded_context() -> None:
    """Builder's `domain_of_validity_excluded` list includes RATIFY-4 context."""
    eq = build_procedural_codebook_from_seed_compression_savings_v1()
    excluded = eq.domain_of_validity.get("domain_of_validity_excluded", [])
    assert "direct_byte_substitution_on_decode_opaque_raw_sections" in excluded
