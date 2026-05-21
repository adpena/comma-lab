# SPDX-License-Identifier: MIT
"""Catalog #359 tests — residual-hybrid misapplication detection.

Per WAVE-3-MAGIC-CODEC-PAIR-1-2-ENGINEERING-FIX 2026-05-20 +
CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable. Covers:

1. Canonical helper `is_residual_hybrid_context` pattern detection.
2. Canonical helper `refuse_residual_hybrid_context_misapplication`
   raise-on-residual-hybrid behavior + non-raise opt-out.
3. Catalog #359 STRICT preflight gate end-to-end behavior:
   - clean-repo regression guard (live count == 0)
   - synthetic violation detection
   - waiver semantics (placeholder rejection)
   - strict mode raises with Catalog #359 message
   - cutoff filter exempts pre-cutoff historical anchors
   - missing registry silent skip
4. Catalog #185 sister-callable regression guard
5. Orchestrator wire-in strict=True regression guard
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.equation import DomainOfValidityViolation
from tac.canonical_equations.procedural_codebook_savings import (
    _RESIDUAL_HYBRID_CONTEXT_PATTERNS,
    is_residual_hybrid_context,
    refuse_residual_hybrid_context_misapplication,
)
from tac.preflight import (
    PreflightError,
    check_no_canonical_equation_misapplication_to_residual_hybrid_contexts,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ============================================================================
# Helper unit tests
# ============================================================================


def test_residual_hybrid_pattern_set_is_non_empty():
    assert len(_RESIDUAL_HYBRID_CONTEXT_PATTERNS) >= 6
    assert all(isinstance(p, str) for p in _RESIDUAL_HYBRID_CONTEXT_PATTERNS)


def test_is_residual_hybrid_context_pair_1_anchor():
    """Pair #1 (DWT detail residual) context matches."""
    assert is_residual_hybrid_context(
        "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands"
    )


def test_is_residual_hybrid_context_pair_2_anchor():
    """Pair #2 (fec6 null-byte SRL1 residual) context matches."""
    assert is_residual_hybrid_context(
        "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes"
    )


def test_is_residual_hybrid_context_intermediate_transform_does_not_match():
    """Equation #26 INCLUDED contexts MUST NOT match the residual-hybrid pattern."""
    assert not is_residual_hybrid_context("intermediate_transform_quantizer")
    assert not is_residual_hybrid_context("chroma_lut_replacement")
    assert not is_residual_hybrid_context("nscs06_v8_chroma_lut")


def test_is_residual_hybrid_context_none_and_empty():
    assert not is_residual_hybrid_context(None)
    assert not is_residual_hybrid_context("")
    assert not is_residual_hybrid_context("   ")


def test_is_residual_hybrid_context_case_insensitive():
    assert is_residual_hybrid_context(
        "MAGIC_CODEC_DENSE_STREAMS_RESIDUAL_CORRECTION_ON_DWT"
    )


def test_refuse_residual_hybrid_raises_by_default():
    with pytest.raises(DomainOfValidityViolation) as exc_info:
        refuse_residual_hybrid_context_misapplication(
            "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands"
        )
    assert "Catalog #359" in str(exc_info.value) or "residual-hybrid" in str(exc_info.value)
    assert "REPLACEMENT" in str(exc_info.value)


def test_refuse_residual_hybrid_returns_false_when_opt_out():
    result = refuse_residual_hybrid_context_misapplication(
        "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
        raise_on_residual_hybrid=False,
    )
    assert result is False


def test_refuse_residual_hybrid_proceeds_on_safe_context():
    result = refuse_residual_hybrid_context_misapplication(
        "intermediate_transform_quantizer"
    )
    assert result is True


def test_refuse_residual_hybrid_proceeds_on_none():
    result = refuse_residual_hybrid_context_misapplication(None)
    assert result is True


# ============================================================================
# End-to-end Catalog #359 gate behavior
# ============================================================================


def test_live_repo_regression_guard():
    """Live count MUST be 0 at landing (cutoff exempts historical anchors)."""
    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert len(violations) == 0, (
        f"Live repo Catalog #359 regression — expected 0 violations, got {len(violations)}: "
        + "\n".join(v[:300] for v in violations[:3])
    )


def test_synthetic_post_cutoff_violation_flagged(tmp_path):
    """Post-cutoff residual-hybrid anchor without waiver MUST flag."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"

    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "synthetic_post_cutoff_residual_hybrid_violation",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {
                        "in_domain_context": "_residual_correction_on_some_new_substrate",
                    },
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")

    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "synthetic_post_cutoff_residual_hybrid_violation" in violations[0]
    assert "Catalog #359" in violations[0] or "residual-hybrid" in violations[0]


def test_synthetic_pre_cutoff_anchor_exempt(tmp_path):
    """Pre-cutoff (e.g. pair #1) anchor MUST be exempt."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"

    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-05-20T23:47:07Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "pair_1_pre_cutoff_historical_anchor",
                    "measurement_utc": "2026-05-20T23:47:07Z",
                    "inputs": {
                        "in_domain_context": "magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands",
                    },
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")

    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_synthetic_substantive_waiver_accepted(tmp_path):
    """Post-cutoff anchor with substantive non-placeholder waiver MUST pass."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"

    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "synthetic_waived_anchor",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {
                        "in_domain_context": "_residual_correction_on_some_new_substrate",
                    },
                    "_residual_hybrid_misapplication_waiver": "operator_explicitly_documents_audit_trail_for_sister_equation_design_per_catalog_359",
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")

    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_synthetic_placeholder_waiver_rejected(tmp_path):
    """Placeholder waiver (`<rationale>`) MUST be rejected per Catalog #287 sister discipline."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"

    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "synthetic_placeholder_waiver",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {
                        "in_domain_context": "_residual_correction_on_some_new_substrate",
                    },
                    "_residual_hybrid_misapplication_waiver": "<rationale>",
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")

    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_synthetic_short_waiver_rejected(tmp_path):
    """Waiver shorter than 10 chars MUST be rejected."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"

    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "synthetic_short_waiver",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {"in_domain_context": "_residual_correction_x"},
                    "_residual_hybrid_misapplication_waiver": "ok",
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")

    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_strict_mode_raises_with_catalog_359_message(tmp_path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"
    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "strict_raise_anchor",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {
                        "in_domain_context": "_seed_plus_residual_some_substrate",
                    },
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")

    with pytest.raises(PreflightError) as exc:
        check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
            repo_root=tmp_path, strict=True, verbose=False
        )
    assert "[catalog-359]" in str(exc.value) or "residual-hybrid" in str(exc.value)


def test_missing_registry_silent_skip(tmp_path):
    """Missing registry MUST silent-skip (clean repos without registry are valid)."""
    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_other_equation_anchors_ignored(tmp_path):
    """Anchors for OTHER equations MUST be ignored."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"
    row = {
        "equation_id": "some_other_equation_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "other_equation_anchor",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {
                        "in_domain_context": "_residual_correction_other_equation_safe",
                    },
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")
    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_anchor_without_in_domain_context_ignored(tmp_path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"
    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "no_in_domain_context_anchor",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {"substrate_id": "nscs06_v8"},
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")
    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_safe_context_anchors_not_flagged(tmp_path):
    """In-domain INCLUDED context anchors MUST NOT be flagged."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"
    row = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "safe_anchor",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {"in_domain_context": "chroma_lut_replacement"},
                }
            ]
        },
    }
    registry.write_text(json.dumps(row) + "\n")
    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


def test_only_latest_anchor_appended_row_evaluated(tmp_path):
    """Only the LATEST anchor_appended row's anchor list is evaluated (latest wins)."""
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    registry = state / "canonical_equations_registry.jsonl"
    # First row with bad anchor
    row1 = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-01T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "first_bad_anchor",
                    "measurement_utc": "2026-06-01T00:00:00Z",
                    "inputs": {"in_domain_context": "_residual_correction_x"},
                }
            ]
        },
    }
    # Latest row with only safe anchor (the bad one is gone)
    row2 = {
        "equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "event_type": "anchor_appended",
        "written_at_utc": "2026-06-02T00:00:00Z",
        "equation_payload": {
            "empirical_anchors": [
                {
                    "anchor_id": "latest_safe_anchor",
                    "measurement_utc": "2026-06-02T00:00:00Z",
                    "inputs": {"in_domain_context": "chroma_lut_replacement"},
                }
            ]
        },
    }
    registry.write_text(json.dumps(row1) + "\n" + json.dumps(row2) + "\n")
    violations = check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 0


# ============================================================================
# Sister-callable regression guards (Catalog #185 + #176)
# ============================================================================


def test_catalog_185_sister_callable_via_globals():
    """Catalog #185 META-meta drift guard: gate function MUST be importable from tac.preflight."""
    from tac import preflight as preflight_mod
    fn = getattr(
        preflight_mod,
        "check_no_canonical_equation_misapplication_to_residual_hybrid_contexts",
        None,
    )
    assert fn is not None
    assert callable(fn)


def test_orchestrator_strict_true_regression_guard():
    """Catalog #176 META-meta: orchestrator callsite MUST use strict=True."""
    import inspect
    from tac import preflight as preflight_mod

    src = inspect.getsource(preflight_mod.preflight_all)
    assert "check_no_canonical_equation_misapplication_to_residual_hybrid_contexts" in src
    # Catalog #359 wire-in is inside a _parallel.run() block; the strict=True
    # MUST appear in the lambda body.
    catalog_359_section_start = src.find(
        "check_no_canonical_equation_misapplication_to_residual_hybrid_contexts"
    )
    # Look forward ~15 lines for strict=True
    next_300_chars = src[catalog_359_section_start:catalog_359_section_start + 1000]
    assert "strict=True" in next_300_chars, (
        f"Catalog #359 orchestrator callsite must use strict=True; "
        f"context: {next_300_chars[:500]}"
    )


def test_catalog_359_appears_in_claude_md():
    """Catalog #176 META-meta: STRICT callsites MUST have CLAUDE.md row."""
    claude_md = REPO_ROOT / "CLAUDE.md"
    if not claude_md.exists():
        pytest.skip("CLAUDE.md missing")
    text = claude_md.read_text()
    # The catalog table format is `^N. \`check_<name>\``
    assert "359. `check_no_canonical_equation_misapplication_to_residual_hybrid_contexts`" in text


def test_canonical_equation_26_module_exports_helpers():
    """Canonical helper module MUST export the new public API symbols."""
    from tac.canonical_equations import procedural_codebook_savings as mod

    assert hasattr(mod, "is_residual_hybrid_context")
    assert hasattr(mod, "refuse_residual_hybrid_context_misapplication")
    assert hasattr(mod, "_RESIDUAL_HYBRID_CONTEXT_PATTERNS")
    # Existing helpers preserved (Catalog #110/#113 APPEND-ONLY)
    assert hasattr(mod, "validate_context_is_in_domain")
    assert hasattr(mod, "build_procedural_codebook_from_seed_compression_savings_v1")
