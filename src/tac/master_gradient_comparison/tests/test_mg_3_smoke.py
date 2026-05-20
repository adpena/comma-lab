"""SLOT MG-3 multi-granularity master-gradient comparison surface smoke tests.

Per RECOVERY-MG-1-thru-5 (2026-05-20): the partial-landed MG-3 module
landed with no test files. This minimal smoke test verifies the canonical
producer's public API surface (import + signature stability) so future
refactors do not silently break the chain-rule discipline per Catalog
#318 + the 10-exploit enumeration.

Per CLAUDE.md "Beauty, simplicity, and developer experience": this test
is intentionally minimal — it does NOT exercise the chain-rule numerics
(which require a paired per-pair-gradient sister anchor + inflate Jacobian
that lives outside the recovery scope). The follow-on subagent landing
Catalog #352 (claimed but not yet written) is expected to write the full
numerical regression suite.
"""

from __future__ import annotations

import inspect


def test_mg_3_module_importable():
    """MG-3 module loads + exposes the 10-exploit producer API."""
    import tac.master_gradient_comparison as mg

    expected_symbols = {
        "ArchiveByteGradientTensor",
        "ContestGradientTensor",
        "EquivalenceClass",
        "InflatedGradientTensor",
        "MultiGranularityComparisonError",
        "PerPairDifficulty",
        "PerPixelReconstructionError",
        "cluster_pairs_by_gradient_similarity",
        "compute_per_pair_difficulty_atlas",
        "compute_score_weighted_reconstruction_error",
        "decompose_M_contest_per_segnet_class",
        "estimate_information_theoretic_floor",
        "extract_M_archive_via_chain_rule",
        "extract_M_contest",
        "extract_M_inflated",
        "persist_comparison_artifact",
    }
    actual_symbols = set(mg.__all__)
    missing = expected_symbols - actual_symbols
    assert not missing, f"MG-3 __all__ missing canonical symbols: {missing}"


def test_mg_3_canonical_provenance_kinds_distinct():
    """The 3 Provenance kinds for M_contest / M_archive / M_inflated must be distinct."""
    from tac.master_gradient_comparison import (
        M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND,
        M_CONTEST_PROVENANCE_KIND,
        M_INFLATED_PROVENANCE_KIND,
    )

    kinds = {
        M_CONTEST_PROVENANCE_KIND,
        M_ARCHIVE_VIA_CHAIN_RULE_PROVENANCE_KIND,
        M_INFLATED_PROVENANCE_KIND,
    }
    assert len(kinds) == 3, (
        "All 3 master-gradient Provenance kinds must be distinct so consumers "
        "can disambiguate via Catalog #323 canonical Provenance kind field"
    )


def test_mg_3_chain_rule_extractor_signature():
    """extract_M_archive_via_chain_rule must NOT accept raw byte-FD kwargs per Catalog #318."""
    from tac.master_gradient_comparison import extract_M_archive_via_chain_rule

    sig = inspect.signature(extract_M_archive_via_chain_rule)
    forbidden = {"byte_modifications", "finite_difference_bit_flip", "gradient_array_path"}
    actual = set(sig.parameters.keys())
    intersect = forbidden & actual
    assert not intersect, (
        f"extract_M_archive_via_chain_rule signature MUST NOT accept raw-byte-FD "
        f"kwargs per Catalog #318 master-gradient raw-byte-authority guard; "
        f"found forbidden kwargs: {intersect}"
    )


def test_mg_3_error_class_hierarchy():
    """MultiGranularityComparisonError must inherit from a common base."""
    from tac.master_gradient_comparison import MultiGranularityComparisonError

    # Should be a real exception class, not just a string
    assert issubclass(MultiGranularityComparisonError, Exception)
