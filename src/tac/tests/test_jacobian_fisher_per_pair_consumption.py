# SPDX-License-Identifier: MIT
"""Tests for ``tac.optimization.jacobian_fisher_importance_allocator.allocate_per_pair_fisher_importance``.

MEDIUM gap closure wave 2026-05-17 — `lane_medium_gap_closure_3_optimization_
modules_per_pair_consumption_20260517`. Closes GAP-4 from the comprehensive
wiring + integration audit by computing per-byte Fisher importance DIRECTLY
from the per-pair gradient covariance (NOT weight-domain saliency proxies
which Catalog #123 forbids on score-gradient-trained substrates).

Per CLAUDE.md FORBIDDEN PATTERNS "Forbidden weight-domain saliency on score-
gradient substrate" (Catalog #123): the per-pair gradient is SCORE-GRADIENT-
derived, NOT weight-derived; the function structurally passes the invariant.

Per CLAUDE.md "Apples-to-apples evidence discipline": every outcome carries
`[predicted; ...]` and `score_claim=False`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.jacobian_fisher_importance_allocator import (
    PER_PAIR_FISHER_IMPORTANCE_SCHEMA,
    ImportanceAllocationError,
    PerPairFisherImportanceOutcome,
    allocate_per_pair_fisher_importance,
)


def test_basic_per_byte_fisher_importance_computation() -> None:
    """Per-pair gradient → per-byte Fisher importance with correct shape."""
    rng = np.random.default_rng(42)
    per_pair = rng.standard_normal((10, 4, 3)).astype(np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        top_k=5,
        bottom_k=5,
        auto_load=False,
    )
    assert result.schema == PER_PAIR_FISHER_IMPORTANCE_SCHEMA
    assert result.n_bytes == 10
    assert result.n_pairs == 4
    assert len(result.per_byte_fisher_importance) == 10
    assert all(v >= 0 for v in result.per_byte_fisher_importance.values())


def test_catalog_123_invariant_satisfied() -> None:
    """Per-pair Fisher is SCORE-GRADIENT-derived (not weight-domain) → Catalog #123 PASS."""
    per_pair = np.ones((5, 3, 3), dtype=np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        auto_load=False,
    )
    assert result.catalog_123_invariant_satisfied is True
    # Evidence grade explicitly mentions score-gradient-derived
    assert "score-gradient-derived" in result.evidence_grade
    assert "NOT weight-domain" in result.evidence_grade


def test_score_claim_false_invariant() -> None:
    """Outcome carries score_claim=False per Apples-to-apples discipline."""
    per_pair = np.ones((5, 3, 3), dtype=np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        auto_load=False,
    )
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.rank_or_kill_eligible is False
    assert "[predicted;" in result.evidence_grade


def test_top_k_bottom_k_sorted_descending() -> None:
    """Top-K importance indices are sorted by Fisher score descending."""
    # Build per-pair gradient with strict ordering: byte 0 highest, byte 4 lowest
    per_pair = np.zeros((5, 4, 3), dtype=np.float64)
    per_pair[0, :, :] = 10.0  # byte 0: highest
    per_pair[1, :, :] = 5.0   # byte 1: 2nd-highest
    per_pair[2, :, :] = 2.0
    per_pair[3, :, :] = 1.0
    per_pair[4, :, :] = 0.1   # byte 4: lowest
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        top_k=3,
        bottom_k=3,
        auto_load=False,
    )
    # Byte 0 should be #1 in top-K (highest importance)
    assert result.top_k_byte_indices_by_importance[0] == 0
    # Byte 1 should be #2
    assert result.top_k_byte_indices_by_importance[1] == 1
    # Byte 4 should be #1 in bottom-K (lowest importance)
    assert result.bottom_k_byte_indices_by_importance[0] == 4


def test_per_byte_fisher_matches_canonical_formula() -> None:
    """Per-byte Fisher = sqrt(sum_p sum_a grad^2) / sqrt(N_pairs * N_axes)."""
    # Constant per-pair gradient → known Fisher
    per_pair = np.ones((3, 4, 3), dtype=np.float64) * 2.0
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        auto_load=False,
    )
    # Expected: sqrt(4*3 * 4) / sqrt(4*3) = sqrt(48)/sqrt(12) = sqrt(4) = 2.0
    expected = 2.0
    assert all(
        abs(v - expected) < 1e-9 for v in result.per_byte_fisher_importance.values()
    )


def test_aggregate_norms_consistent() -> None:
    """L1 norm = sum of per-byte fisher; L2 norm = sqrt(sum of squares)."""
    per_pair = np.ones((5, 3, 3), dtype=np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        auto_load=False,
    )
    fisher_values = list(result.per_byte_fisher_importance.values())
    expected_l1 = sum(fisher_values)
    expected_l2 = np.sqrt(sum(v ** 2 for v in fisher_values))
    assert abs(result.aggregate_fisher_l1 - expected_l1) < 1e-9
    assert abs(result.aggregate_fisher_l2 - expected_l2) < 1e-9


def test_invalid_archive_sha_rejected() -> None:
    """Sub-12-char or non-hex sha rejected with ImportanceAllocationError."""
    with pytest.raises(ImportanceAllocationError, match="archive_sha256 must be"):
        allocate_per_pair_fisher_importance(
            archive_sha256="abc",
            per_pair_gradient=np.ones((3, 3, 3)),
            auto_load=False,
        )
    with pytest.raises(ImportanceAllocationError, match="archive_sha256 must be"):
        allocate_per_pair_fisher_importance(
            archive_sha256="not-a-hex-sha",
            per_pair_gradient=np.ones((3, 3, 3)),
            auto_load=False,
        )


def test_negative_top_k_rejected() -> None:
    """Negative top_k or bottom_k rejected."""
    with pytest.raises(ImportanceAllocationError, match="must be non-negative"):
        allocate_per_pair_fisher_importance(
            archive_sha256="deadbeef1234567890abcdef",
            per_pair_gradient=np.ones((3, 3, 3)),
            top_k=-1,
            auto_load=False,
        )
    with pytest.raises(ImportanceAllocationError, match="must be non-negative"):
        allocate_per_pair_fisher_importance(
            archive_sha256="deadbeef1234567890abcdef",
            per_pair_gradient=np.ones((3, 3, 3)),
            bottom_k=-1,
            auto_load=False,
        )


def test_no_gradient_no_auto_load_raises() -> None:
    """per_pair_gradient=None + auto_load=False raises."""
    with pytest.raises(ImportanceAllocationError, match="per_pair_gradient is None"):
        allocate_per_pair_fisher_importance(
            archive_sha256="deadbeef1234567890abcdef",
            per_pair_gradient=None,
            auto_load=False,
        )


def test_wrong_shape_rejected() -> None:
    """Wrong-shape per-pair gradient raises ImportanceAllocationError."""
    bad_gradient = np.ones((10, 4, 2))  # last dim != 3
    with pytest.raises(ImportanceAllocationError, match="shape \\(N_bytes, N_pairs, 3\\)"):
        allocate_per_pair_fisher_importance(
            archive_sha256="deadbeef1234567890abcdef",
            per_pair_gradient=bad_gradient,
            auto_load=False,
        )


def test_top_k_caps_at_n_bytes() -> None:
    """top_k > n_bytes silently caps at n_bytes (no overflow)."""
    per_pair = np.ones((3, 4, 3), dtype=np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        top_k=100,
        bottom_k=100,
        auto_load=False,
    )
    # Only 3 bytes available; top/bottom-k each cap at 3
    assert len(result.top_k_byte_indices_by_importance) == 3
    assert len(result.bottom_k_byte_indices_by_importance) == 3


def test_zero_bottom_k_returns_empty_tuple() -> None:
    """bottom_k=0 should surface no bottom indices, not all byte indices."""
    per_pair = np.ones((4, 3, 3), dtype=np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        top_k=0,
        bottom_k=0,
        auto_load=False,
    )
    assert result.top_k_byte_indices_by_importance == ()
    assert result.bottom_k_byte_indices_by_importance == ()


def test_composability_with_wyner_ziv_reweight() -> None:
    """Per-byte Fisher importance composes with Wyner-Ziv reweight as base signal."""
    # Build per-pair gradient with known structure
    per_pair = np.ones((10, 4, 3), dtype=np.float64)
    # Make bytes 0, 1, 2 have HIGH per-byte importance
    per_pair[:3, :, :] = 5.0
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        auto_load=False,
    )
    # Use Fisher importance as the BASE WEIGHT for Wyner-Ziv reweight composition.
    # This is the canonical "Gap-2 + Gap-4 compose" path the design memo specifies.
    base_weights = result.per_byte_fisher_importance
    assert len(base_weights) == 10
    # Bytes 0-2 should have HIGHER importance than bytes 3-9
    high_importance = sum(base_weights[i] for i in range(3))
    low_importance = sum(base_weights[i] for i in range(3, 10))
    assert high_importance > low_importance


def test_rationale_documents_catalog_123_compliance() -> None:
    """Rationale string explicitly cites Catalog #123 invariant."""
    per_pair = np.ones((3, 3, 3), dtype=np.float64)
    result = allocate_per_pair_fisher_importance(
        archive_sha256="deadbeef1234567890abcdef",
        per_pair_gradient=per_pair,
        auto_load=False,
    )
    assert "Catalog #123" in result.rationale
    assert "SATISFIED" in result.rationale or "score-gradient-derived" in result.rationale
