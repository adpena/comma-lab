# SPDX-License-Identifier: MIT
"""Tests for META-LIFT-4 UNIWARD canonical-application-surface invariant enumerator.

Per CLAUDE.md "Subagent coherence-by-default" Catalog #125 6-hook wire-in +
the 11th standing directive ORDER discipline. Covers:

  - Synthetic enumeration with known canonical-application surfaces
  - Per-surface canonical-formula-grounded verification (Sallee 2003 +
    Fridrich 2007 + Holub-Fridrich-Denemark 2014 references)
  - Ranking determinism (same registry produces same DESC-by-bound ordering)
  - Provenance + non-promotable markers verified per Catalog #341
  - Cathedral consumer contract compliance (Catalog #335)
  - CLI subprocess test (exit codes 0 / 1 / 2)
  - Live-repo regression guard (canonical registry has expected anchor)
  - Integration consuming today's UNIWARD 7th-order substrate artifacts

Sister of:
  - tests for META-LIFT-1 cross_substrate_master_gradient_analyzer
  - tests for META-LIFT-2 pareto_polytope_unified_solver
  - tests for UNIWARD 7th-order substrate integration
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.uniward_invariant_enumerator import (
    CANONICAL_EQUATION_ID,
    PREDICTED_AXIS_TAG,
    SCHEMA_VERSION,
    UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH,
    VALID_AXIS_LABELS,
    VALID_INVARIANT_VERDICTS,
    VALID_SURFACE_KINDS,
    RankedUniwardSurfaces,
    UniwardApplicabilityVerdict,
    UniwardCanonicalApplicationSurface,
    UniwardInvariantEnumeration,
    UniwardInvariantEnumerationCorruptError,
    append_enumeration_locked,
    enumerate_uniward_canonical_application_surfaces,
    load_enumerations_strict,
    rank_uniward_applicable_surfaces_by_predicted_delta_s,
    verify_uniward_applicability,
)


# ---------------------------------------------------------------------------
# Module-level constants + schema validation
# ---------------------------------------------------------------------------


def test_schema_version_pinned():
    """SCHEMA_VERSION canonical value pinned."""
    assert SCHEMA_VERSION == "uniward_invariant_enumeration_v1"


def test_canonical_equation_id_pinned():
    """Canonical equation id pinned per Catalog #344."""
    assert CANONICAL_EQUATION_ID == (
        "uniward_canonical_application_surface_invariant_enumeration_v1"
    )


def test_predicted_axis_tag_pinned():
    """Catalog #341 routing marker pinned."""
    assert PREDICTED_AXIS_TAG == "[predicted]"


def test_valid_axis_labels_canonical_three():
    """Per Catalog #356 per-axis decomposition: exactly 3 axes (seg/pose/rate)."""
    assert VALID_AXIS_LABELS == frozenset({"seg", "pose", "rate"})


def test_valid_surface_kinds_canonical_taxonomy():
    """Canonical surface-kind taxonomy per Fridrich literature has 10 kinds."""
    assert len(VALID_SURFACE_KINDS) == 10
    # Canonical anchors per Holub-Fridrich-Denemark 2014 + sister references.
    assert "chroma_lut_quantized_codebook" in VALID_SURFACE_KINDS
    assert "dct_quantized_coefficient_blob" in VALID_SURFACE_KINDS
    assert "arithmetic_coded_symbol_stream" in VALID_SURFACE_KINDS


def test_valid_invariant_verdicts_canonical_seven():
    """7-verdict taxonomy: 2 APPLICABLE + 4 INAPPLICABLE + 1 UNKNOWN."""
    assert len(VALID_INVARIANT_VERDICTS) == 7
    applicable = {v for v in VALID_INVARIANT_VERDICTS if v.startswith("APPLICABLE_")}
    inapplicable = {v for v in VALID_INVARIANT_VERDICTS if v.startswith("INAPPLICABLE_")}
    assert len(applicable) == 2
    assert len(inapplicable) == 4


# ---------------------------------------------------------------------------
# UniwardCanonicalApplicationSurface dataclass invariants
# ---------------------------------------------------------------------------


def _canonical_test_surface():
    """Sister test fixture: NSCS06 v8 chroma LUT canonical surface descriptor."""
    return UniwardCanonicalApplicationSurface(
        surface_id="test_nscs06_v8_chroma_lut",
        surface_kind="chroma_lut_quantized_codebook",
        substrate_id="test_substrate",
        entropy_coded_axis="brotli",
        quantization_axis="uint8",
        per_symbol_routable_axis="direct_per_symbol",
        canonical_formula_reference=(
            "Holub-Fridrich-Denemark 2014 universal distortion"
        ),
        n_symbols_estimated=240,
        architecture_layer="both",
        notes="Test fixture for canonical surface invariants",
    )


def test_surface_canonical_construction():
    """Canonical surface descriptor constructs cleanly."""
    s = _canonical_test_surface()
    assert s.surface_id == "test_nscs06_v8_chroma_lut"


def test_surface_rejects_empty_surface_id():
    with pytest.raises(ValueError, match="surface_id must be non-empty"):
        UniwardCanonicalApplicationSurface(
            surface_id="",
            surface_kind="chroma_lut_quantized_codebook",
            substrate_id="test",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="Holub-Fridrich-Denemark 2014",
            n_symbols_estimated=240,
            architecture_layer="both",
            notes="Test fixture",
        )


def test_surface_rejects_invalid_surface_kind():
    with pytest.raises(ValueError, match="surface_kind must be in"):
        UniwardCanonicalApplicationSurface(
            surface_id="test",
            surface_kind="bogus_kind",
            substrate_id="test",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="Holub-Fridrich-Denemark 2014",
            n_symbols_estimated=240,
            architecture_layer="both",
            notes="Test fixture",
        )


def test_surface_rejects_placeholder_canonical_reference():
    """Per Catalog #287 placeholder rejection."""
    with pytest.raises(ValueError, match="canonical_formula_reference"):
        UniwardCanonicalApplicationSurface(
            surface_id="test",
            surface_kind="chroma_lut_quantized_codebook",
            substrate_id="test",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="<rationale>",
            n_symbols_estimated=240,
            architecture_layer="both",
            notes="Test fixture",
        )


def test_surface_rejects_short_canonical_reference():
    """Per Catalog #287 min 4 chars."""
    with pytest.raises(ValueError, match="canonical_formula_reference"):
        UniwardCanonicalApplicationSurface(
            surface_id="test",
            surface_kind="chroma_lut_quantized_codebook",
            substrate_id="test",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="abc",  # 3 chars; rejected
            n_symbols_estimated=240,
            architecture_layer="both",
            notes="Test fixture",
        )


def test_surface_rejects_non_positive_n_symbols():
    with pytest.raises(ValueError, match="n_symbols_estimated must be positive"):
        UniwardCanonicalApplicationSurface(
            surface_id="test",
            surface_kind="chroma_lut_quantized_codebook",
            substrate_id="test",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="Holub-Fridrich-Denemark 2014",
            n_symbols_estimated=0,
            architecture_layer="both",
            notes="Test fixture",
        )


def test_surface_as_dict_round_trip():
    s = _canonical_test_surface()
    d = s.as_dict()
    assert d["surface_id"] == s.surface_id
    assert d["n_symbols_estimated"] == 240


# ---------------------------------------------------------------------------
# verify_uniward_applicability — per-condition test correctness
# ---------------------------------------------------------------------------


def test_verify_applicable_canonical_fridrich_natural():
    """A surface with all 4 conditions PASS + direct_per_symbol = CANONICAL."""
    s = _canonical_test_surface()
    v = verify_uniward_applicability(s)
    assert v.verdict == "APPLICABLE_CANONICAL_FRIDRICH_NATURAL"
    assert v.condition_1_entropy_coded is True
    assert v.condition_2_quantized is True
    assert v.condition_3_per_symbol_routable is True
    assert v.condition_4_canonical_formula_grounded is True
    assert v.all_conditions_pass() is True


def test_verify_applicable_variant_per_pair_routable():
    """A surface with all 4 conditions PASS + per_pair_routable = VARIANT."""
    s = UniwardCanonicalApplicationSurface(
        surface_id="test_per_pair",
        surface_kind="wyner_ziv_codec_layer",
        substrate_id="test",
        entropy_coded_axis="ans",
        quantization_axis="int16",
        per_symbol_routable_axis="per_pair_routable",
        canonical_formula_reference="Wyner-Ziv 1976 side-information",
        n_symbols_estimated=4800,
        architecture_layer="sidecar",
        notes="Per-pair routable variant",
    )
    v = verify_uniward_applicability(s)
    assert v.verdict == "APPLICABLE_VARIANT_REQUIRES_FORMULA_ADAPTER"


def test_verify_inapplicable_no_entropy_coding():
    """A surface with entropy_coded_axis='none' fails condition #1."""
    s = UniwardCanonicalApplicationSurface(
        surface_id="test_raw",
        surface_kind="dct_quantized_coefficient_blob",
        substrate_id="test",
        entropy_coded_axis="none",
        quantization_axis="uint8",
        per_symbol_routable_axis="direct_per_symbol",
        canonical_formula_reference="Holub-Fridrich-Denemark 2014",
        n_symbols_estimated=240,
        architecture_layer="both",
        notes="Raw bytes no entropy coding",
    )
    v = verify_uniward_applicability(s)
    assert v.verdict == "INAPPLICABLE_NO_ENTROPY_CODING"
    assert v.condition_1_entropy_coded is False


def test_verify_inapplicable_raw_float_domain():
    """A surface with quantization_axis='none' fails condition #2.

    The canonical 5th + 6th-order PARADIGM-NULL failure mode (raw-RGB
    application domain mismatch).
    """
    s = UniwardCanonicalApplicationSurface(
        surface_id="test_float",
        surface_kind="dct_quantized_coefficient_blob",
        substrate_id="test",
        entropy_coded_axis="arithmetic",
        quantization_axis="none",
        per_symbol_routable_axis="direct_per_symbol",
        canonical_formula_reference="Holub-Fridrich-Denemark 2014",
        n_symbols_estimated=240,
        architecture_layer="both",
        notes="Raw float domain",
    )
    v = verify_uniward_applicability(s)
    assert v.verdict == "INAPPLICABLE_RAW_FLOAT_DOMAIN"


def test_verify_inapplicable_no_per_symbol_routability():
    """A surface with per_symbol_routable_axis='none' fails condition #3."""
    s = UniwardCanonicalApplicationSurface(
        surface_id="test_no_route",
        surface_kind="dct_quantized_coefficient_blob",
        substrate_id="test",
        entropy_coded_axis="brotli",
        quantization_axis="uint8",
        per_symbol_routable_axis="none",
        canonical_formula_reference="Holub-Fridrich-Denemark 2014",
        n_symbols_estimated=240,
        architecture_layer="both",
        notes="No per-symbol routability",
    )
    v = verify_uniward_applicability(s)
    assert v.verdict == "INAPPLICABLE_NO_PER_SYMBOL_ROUTABILITY"


# ---------------------------------------------------------------------------
# UniwardApplicabilityVerdict dataclass invariants
# ---------------------------------------------------------------------------


def test_verdict_rejects_inconsistent_all_pass_with_inapplicable_verdict():
    """Cross-validation: all 4 conditions PASS with INAPPLICABLE_* verdict rejected."""
    with pytest.raises(ValueError, match="inconsistent"):
        UniwardApplicabilityVerdict(
            surface_id="test",
            verdict="INAPPLICABLE_NO_ENTROPY_CODING",
            condition_1_entropy_coded=True,
            condition_2_quantized=True,
            condition_3_per_symbol_routable=True,
            condition_4_canonical_formula_grounded=True,
            rationale="Test inconsistent verdict",
            canonical_reference_cited="Holub-Fridrich-Denemark 2014",
        )


def test_verdict_rejects_inconsistent_some_fail_with_applicable_verdict():
    """Cross-validation: at least 1 condition FAIL with APPLICABLE_* verdict rejected."""
    with pytest.raises(ValueError, match="inconsistent"):
        UniwardApplicabilityVerdict(
            surface_id="test",
            verdict="APPLICABLE_CANONICAL_FRIDRICH_NATURAL",
            condition_1_entropy_coded=False,
            condition_2_quantized=True,
            condition_3_per_symbol_routable=True,
            condition_4_canonical_formula_grounded=True,
            rationale="Test inconsistent verdict",
            canonical_reference_cited="Holub-Fridrich-Denemark 2014",
        )


def test_verdict_rejects_placeholder_rationale():
    """Per Catalog #287 placeholder rejection."""
    with pytest.raises(ValueError, match="rationale must be substantive"):
        UniwardApplicabilityVerdict(
            surface_id="test",
            verdict="APPLICABLE_CANONICAL_FRIDRICH_NATURAL",
            condition_1_entropy_coded=True,
            condition_2_quantized=True,
            condition_3_per_symbol_routable=True,
            condition_4_canonical_formula_grounded=True,
            rationale="<rationale>",
            canonical_reference_cited="Holub-Fridrich-Denemark 2014",
        )


# ---------------------------------------------------------------------------
# Ranking determinism
# ---------------------------------------------------------------------------


def test_ranking_determinism():
    """Same surfaces produce same DESC-by-bound ordering."""
    surfaces = [_canonical_test_surface()]
    verdicts = [verify_uniward_applicability(s) for s in surfaces]
    r1 = rank_uniward_applicable_surfaces_by_predicted_delta_s(
        surfaces, verdicts, axis="seg"
    )
    r2 = rank_uniward_applicable_surfaces_by_predicted_delta_s(
        surfaces, verdicts, axis="seg"
    )
    assert r1.ranked_surface_ids == r2.ranked_surface_ids
    assert r1.per_surface_predicted_delta_s_upper_bound == r2.per_surface_predicted_delta_s_upper_bound


def test_ranking_desc_invariant():
    """Per __post_init__ DESC invariant verified."""
    # Build 3 synthetic surfaces with predictable bounds.
    surfaces = []
    for i, n in enumerate([100, 400, 900]):
        surfaces.append(
            UniwardCanonicalApplicationSurface(
                surface_id=f"surface_{i}",
                surface_kind="chroma_lut_quantized_codebook",
                substrate_id=f"sub_{i}",
                entropy_coded_axis="brotli",
                quantization_axis="uint8",
                per_symbol_routable_axis="direct_per_symbol",
                canonical_formula_reference="Holub-Fridrich-Denemark 2014",
                n_symbols_estimated=n,
                architecture_layer="both",
                notes=f"Synthetic surface n={n}",
            )
        )
    verdicts = [verify_uniward_applicability(s) for s in surfaces]
    r = rank_uniward_applicable_surfaces_by_predicted_delta_s(
        surfaces, verdicts, axis="seg"
    )
    # DESC: surface_2 (sqrt(900)=30) > surface_1 (sqrt(400)=20) > surface_0 (sqrt(100)=10)
    bounds = r.per_surface_predicted_delta_s_upper_bound
    for i in range(1, len(bounds)):
        assert bounds[i - 1] >= bounds[i]


def test_ranking_inapplicable_last():
    """INAPPLICABLE surfaces rank last (upper_bound=0)."""
    surfaces = [
        _canonical_test_surface(),  # APPLICABLE
        UniwardCanonicalApplicationSurface(
            surface_id="inapplicable",
            surface_kind="dct_quantized_coefficient_blob",
            substrate_id="test",
            entropy_coded_axis="none",  # fails condition #1
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="Holub-Fridrich-Denemark 2014",
            n_symbols_estimated=1000,
            architecture_layer="both",
            notes="Inapplicable surface",
        ),
    ]
    verdicts = [verify_uniward_applicability(s) for s in surfaces]
    r = rank_uniward_applicable_surfaces_by_predicted_delta_s(
        surfaces, verdicts, axis="seg"
    )
    # APPLICABLE first (non-zero bound), INAPPLICABLE last (zero bound).
    assert r.ranked_surface_ids[0] == "test_nscs06_v8_chroma_lut"
    assert r.ranked_surface_ids[1] == "inapplicable"
    assert r.per_surface_predicted_delta_s_upper_bound[1] == 0.0


def test_ranking_with_gradient_norms():
    """Per-axis gradient norms incorporate empirical signal."""
    surfaces = [
        _canonical_test_surface(),
        UniwardCanonicalApplicationSurface(
            surface_id="other",
            surface_kind="chroma_lut_quantized_codebook",
            substrate_id="other_substrate",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference="Holub-Fridrich-Denemark 2014",
            n_symbols_estimated=100,
            architecture_layer="both",
            notes="Other surface for ranking test",
        ),
    ]
    verdicts = [verify_uniward_applicability(s) for s in surfaces]
    # Provide gradient norm that flips structural ranking.
    norms = {
        "test_nscs06_v8_chroma_lut": 100.0,
        "other": 1.0,
    }
    r = rank_uniward_applicable_surfaces_by_predicted_delta_s(
        surfaces,
        verdicts,
        axis="seg",
        per_surface_gradient_l2_norms=norms,
    )
    # test_nscs06_v8_chroma_lut (norm=100, sqrt(240)) >>> other (norm=1, sqrt(100))
    assert r.ranked_surface_ids[0] == "test_nscs06_v8_chroma_lut"


# ---------------------------------------------------------------------------
# enumerate_uniward_canonical_application_surfaces — canonical entry point
# ---------------------------------------------------------------------------


def test_enumerate_canonical_registry_has_expected_anchor():
    """7th-order PARADIGM-VALIDATED ANCHOR present in canonical registry."""
    e = enumerate_uniward_canonical_application_surfaces()
    surface_ids = {s.surface_id for s in e.surfaces}
    # The 7th-order ANCHOR (commit 87bd1c355) MUST be in the canonical registry.
    assert "nscs06_v8_chroma_lut" in surface_ids


def test_enumerate_canonical_registry_at_least_5_applicable():
    """Canonical registry has at least 5 APPLICABLE surfaces."""
    e = enumerate_uniward_canonical_application_surfaces()
    assert e.n_applicable_surfaces >= 5


def test_enumerate_master_gradient_per_byte_inapplicable():
    """Catalog #318 FORBIDDEN master-gradient-per-byte is INAPPLICABLE
    (anti-example for invariant test)."""
    e = enumerate_uniward_canonical_application_surfaces()
    found = False
    for s, v in zip(e.surfaces, e.verdicts):
        if s.surface_id == "master_gradient_per_byte_raw_authority":
            assert v.verdict.startswith("INAPPLICABLE_")
            found = True
            break
    assert found, "master_gradient_per_byte_raw_authority must be in registry as anti-example"


def test_enumerate_per_axis_three_rankings():
    """Per Catalog #356 per-axis decomposition: exactly 3 rankings."""
    e = enumerate_uniward_canonical_application_surfaces()
    assert len(e.rankings_per_axis) == 3
    axes = {r.axis for r in e.rankings_per_axis}
    assert axes == VALID_AXIS_LABELS


def test_enumerate_canonical_provenance_markers():
    """Per Catalog #341 non-promotable routing markers verified."""
    e = enumerate_uniward_canonical_application_surfaces()
    assert e.axis_tag == "[predicted]"
    assert e.score_claim is False
    assert e.promotable is False
    assert e.evidence_grade.startswith("[predicted;")


def test_enumerate_canonical_equation_status_formalization_pending():
    """Per Catalog #344 FORMALIZATION_PENDING until paired-CUDA anchor."""
    e = enumerate_uniward_canonical_application_surfaces()
    assert e.canonical_equation_id == CANONICAL_EQUATION_ID
    assert e.canonical_equation_status == "FORMALIZATION_PENDING"


def test_enumerate_count_consistency():
    """N_applicable + N_inapplicable + N_unknown == N_surfaces."""
    e = enumerate_uniward_canonical_application_surfaces()
    total = e.n_applicable_surfaces + e.n_inapplicable_surfaces + e.n_unknown_surfaces
    assert total == len(e.surfaces)


def test_enumerate_with_custom_surfaces():
    """Custom surfaces sequence accepted for testing."""
    custom = [_canonical_test_surface()]
    e = enumerate_uniward_canonical_application_surfaces(custom_surfaces=custom)
    assert len(e.surfaces) == 1
    assert e.surfaces[0].surface_id == "test_nscs06_v8_chroma_lut"
    assert e.n_applicable_surfaces == 1


# ---------------------------------------------------------------------------
# UniwardInvariantEnumeration dataclass cross-validation
# ---------------------------------------------------------------------------


def test_enumeration_rejects_mismatched_surfaces_verdicts_length():
    """surfaces and verdicts MUST have same length."""
    surfaces = [_canonical_test_surface()]
    verdicts = []
    rankings = [
        RankedUniwardSurfaces(
            axis=a,
            ranked_surface_ids=("test_nscs06_v8_chroma_lut",),
            per_surface_predicted_delta_s_upper_bound=(1.0,),
            per_surface_per_byte_leverage=(0.1,),
            canonical_equation_reference="test",
        )
        for a in ("seg", "pose", "rate")
    ]
    with pytest.raises(ValueError, match="surfaces length"):
        UniwardInvariantEnumeration(
            schema_version=SCHEMA_VERSION,
            enumeration_id="test",
            measurement_utc="2026-05-26T00:00:00",
            surfaces=tuple(surfaces),
            verdicts=tuple(verdicts),
            rankings_per_axis=tuple(rankings),
            n_applicable_surfaces=0,
            n_inapplicable_surfaces=0,
            n_unknown_surfaces=0,
            axis_tag="[predicted]",
            score_claim=False,
            promotable=False,
            evidence_grade="[predicted; test]",
            canonical_helper_invocation="test",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
        )


def test_enumeration_rejects_score_claim_true():
    """Per Catalog #341 score_claim MUST be False."""
    surfaces = (_canonical_test_surface(),)
    verdicts = (verify_uniward_applicability(surfaces[0]),)
    rankings = tuple(
        rank_uniward_applicable_surfaces_by_predicted_delta_s(
            list(surfaces), list(verdicts), axis=a
        )
        for a in ("seg", "pose", "rate")
    )
    with pytest.raises(ValueError, match="score_claim must be False"):
        UniwardInvariantEnumeration(
            schema_version=SCHEMA_VERSION,
            enumeration_id="test",
            measurement_utc="2026-05-26T00:00:00",
            surfaces=surfaces,
            verdicts=verdicts,
            rankings_per_axis=rankings,
            n_applicable_surfaces=1,
            n_inapplicable_surfaces=0,
            n_unknown_surfaces=0,
            axis_tag="[predicted]",
            score_claim=True,  # FORBIDDEN
            promotable=False,
            evidence_grade="[predicted; test]",
            canonical_helper_invocation="test",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
        )


# ---------------------------------------------------------------------------
# Cathedral consumer contract compliance (Catalog #335)
# ---------------------------------------------------------------------------


def test_cathedral_consumer_contract_compliant():
    """Per Catalog #335: consumer module satisfies canonical Protocol contract."""
    from tac.cathedral.consumer_contract import validate_consumer_module
    import tac.cathedral_consumers.uniward_invariant_enumerator_consumer as consumer

    registration = validate_consumer_module(consumer)
    assert registration.contract_compliant is True
    assert registration.consumer_name == "uniward_invariant_enumerator_consumer"
    assert registration.validation_errors == ()


def test_cathedral_consumer_consume_candidate_observability_only():
    """Per Catalog #341 routing markers: ALL return values non-promotable."""
    import tac.cathedral_consumers.uniward_invariant_enumerator_consumer as consumer

    annot = consumer.consume_candidate({"substrate_id": "nscs06_v8_chroma_lut"})
    assert annot["predicted_delta_adjustment"] == 0.0
    assert annot["promotable"] is False
    assert annot["axis_tag"] == "[predicted]"


def test_cathedral_consumer_handles_missing_substrate_id():
    """Defensive: candidate without substrate_id still returns valid annotation."""
    import tac.cathedral_consumers.uniward_invariant_enumerator_consumer as consumer

    annot = consumer.consume_candidate({})
    # Should not crash; should return valid observability-only contribution.
    assert "consumer_name" in annot
    assert annot["promotable"] is False


def test_cathedral_consumer_auto_discoverable():
    """Per Catalog #336 / #337: consumer auto-discovered by cathedral autopilot loop."""
    import sys

    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from cathedral_autopilot_autonomous_loop import discover_compliant_consumer_modules

    discovered = discover_compliant_consumer_modules()
    discovered_names = {getattr(m, "CONSUMER_NAME", m.__name__) for m in discovered}
    assert "uniward_invariant_enumerator_consumer" in discovered_names


# ---------------------------------------------------------------------------
# Canonical fcntl-locked JSONL ledger I/O
# ---------------------------------------------------------------------------


def test_load_enumerations_strict_empty_when_missing(tmp_path):
    """Missing ledger returns empty list (no fail-closed crash)."""
    rows = load_enumerations_strict(tmp_path / "nonexistent.jsonl")
    assert rows == []


def test_load_enumerations_strict_raises_on_corrupt(tmp_path):
    """Per Catalog #138 fail-closed: corrupt JSON raises CorruptError."""
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"valid": "json"}\nNOT_VALID_JSON_LINE\n')
    with pytest.raises(UniwardInvariantEnumerationCorruptError):
        load_enumerations_strict(corrupt)


def test_append_enumeration_locked_round_trip(tmp_path):
    """Append + strict-load round-trip preserves canonical fields."""
    e = enumerate_uniward_canonical_application_surfaces()
    target = tmp_path / "test_ledger.jsonl"
    append_enumeration_locked(e, ledger_path=target)
    rows = load_enumerations_strict(target)
    assert len(rows) == 1
    assert rows[0]["enumeration_id"] == e.enumeration_id
    assert rows[0]["canonical_equation_status"] == "FORMALIZATION_PENDING"
    assert rows[0]["score_claim"] is False
    assert rows[0]["promotable"] is False


# ---------------------------------------------------------------------------
# CLI subprocess test
# ---------------------------------------------------------------------------


def test_cli_default_mode_exit_0():
    """CLI default mode (--enumerate-all) exits 0 on canonical registry."""
    cli = REPO_ROOT / "tools" / "uniward_invariant_enumerator_cli.py"
    result = subprocess.run(
        [sys.executable, str(cli)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "META-LIFT-4" in result.stdout
    assert "OBSERVABILITY-ONLY" in result.stdout


def test_cli_json_mode_emits_valid_json():
    """CLI --json mode emits valid JSON parseable as enumeration."""
    cli = REPO_ROOT / "tools" / "uniward_invariant_enumerator_cli.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--enumerate-all", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    parsed = json.loads(result.stdout)
    assert parsed["canonical_equation_id"] == CANONICAL_EQUATION_ID
    assert parsed["canonical_equation_status"] == "FORMALIZATION_PENDING"
    assert parsed["promotable"] is False


def test_cli_rank_without_axis_exits_2():
    """CLI --rank-by-predicted-delta-s without --contest-axis exits 2."""
    cli = REPO_ROOT / "tools" / "uniward_invariant_enumerator_cli.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--rank-by-predicted-delta-s"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "--contest-axis" in result.stderr


def test_cli_verify_unknown_surface_exits_2():
    """CLI --verify-surface with unknown id exits 2."""
    cli = REPO_ROOT / "tools" / "uniward_invariant_enumerator_cli.py"
    result = subprocess.run(
        [sys.executable, str(cli), "--verify-surface", "nonexistent_surface_id_xyz"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "not found" in result.stderr


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_canonical_registry_anchors_present():
    """Live-repo regression guard: canonical registry has documented anchors."""
    e = enumerate_uniward_canonical_application_surfaces()
    surface_ids = {s.surface_id for s in e.surfaces}
    # 7th-order PARADIGM-VALIDATED ANCHOR (commit 87bd1c355) — MUST be present.
    assert "nscs06_v8_chroma_lut" in surface_ids
    # T3 council #2 + #3 stacking candidates.
    assert "nscs06_grayscale_lut" in surface_ids
    assert "vq_vae_indices_blob" in surface_ids
    # FEC family sister.
    assert "fec_selector_indices_per_frame" in surface_ids
    # Anti-example for invariant test.
    assert "master_gradient_per_byte_raw_authority" in surface_ids


def test_live_repo_7th_order_anchor_applicable_canonical():
    """7th-order ANCHOR is APPLICABLE_CANONICAL_FRIDRICH_NATURAL."""
    e = enumerate_uniward_canonical_application_surfaces()
    for surface, verdict in zip(e.surfaces, e.verdicts):
        if surface.surface_id == "nscs06_v8_chroma_lut":
            assert verdict.verdict == "APPLICABLE_CANONICAL_FRIDRICH_NATURAL"
            assert verdict.all_conditions_pass() is True
            return
    pytest.fail("nscs06_v8_chroma_lut not found in canonical registry")


def test_live_repo_anti_example_inapplicable():
    """Anti-example master_gradient_per_byte is INAPPLICABLE_NO_ENTROPY_CODING."""
    e = enumerate_uniward_canonical_application_surfaces()
    for surface, verdict in zip(e.surfaces, e.verdicts):
        if surface.surface_id == "master_gradient_per_byte_raw_authority":
            assert verdict.verdict == "INAPPLICABLE_NO_ENTROPY_CODING"
            return
    pytest.fail("master_gradient_per_byte_raw_authority not found in canonical registry")


# ---------------------------------------------------------------------------
# Integration with UNIWARD 7th-order substrate artifacts
# ---------------------------------------------------------------------------


def test_integration_nscs06_v8_chroma_lut_surface_descriptor_matches_substrate():
    """Integration: NSCS06 v8 chroma LUT surface descriptor matches actual substrate.

    The 7th-order UNIWARD landing (commit 87bd1c355) integrated UNIWARD into
    the NSCS06 v8 substrate with LUT shape (16 grayscale_levels x 5 segnet_classes
    x 3 RGB) = 240 entries. The canonical surface descriptor MUST match.
    """
    e = enumerate_uniward_canonical_application_surfaces()
    for surface in e.surfaces:
        if surface.surface_id == "nscs06_v8_chroma_lut":
            # 16 x 5 x 3 = 240 per substrate architecture
            assert surface.n_symbols_estimated == 240
            # Per the 7th-order landing memo: brotli + uint8 + direct per-symbol
            assert surface.entropy_coded_axis == "brotli"
            assert surface.quantization_axis == "uint8"
            assert surface.per_symbol_routable_axis == "direct_per_symbol"
            # Per Holub-Fridrich-Denemark 2014 canonical reference
            assert "Holub-Fridrich-Denemark 2014" in surface.canonical_formula_reference
            assert "Sallee 2003" in surface.canonical_formula_reference
            return
    pytest.fail("nscs06_v8_chroma_lut not found in canonical registry")


def test_integration_uniward_substrate_modules_importable():
    """Integration: just-landed UNIWARD 7th-order substrate modules importable."""
    # Per Catalog #230 sister-disjoint: just imports; no mutation.
    from tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration import (
        weight_map_per_lut_index,
        lut_derivation_uniward_weighted,
    )
    # The canonical types referenced in the integration must exist.
    assert hasattr(weight_map_per_lut_index, "PerLutIndexUniwardWeights")
    assert hasattr(lut_derivation_uniward_weighted, "WeightedMedianResult")
    assert hasattr(lut_derivation_uniward_weighted, "build_uniward_weighted_chroma_lut")
