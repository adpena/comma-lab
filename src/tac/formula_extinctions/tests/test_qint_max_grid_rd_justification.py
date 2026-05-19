# SPDX-License-Identifier: MIT
"""Tests for Row #3 — qint_max grid R-D justification."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.qint_max_grid_rd_justification import (
    CANONICAL_QINT_MAX_GRID,
    QintMaxGridInput,
    canonical_qint_max_grid_rd_proof,
)


def test_canonical_grid_is_all_rd_optimal():
    """The canonical (1,3,7,15,31) grid passes all R-D-optimal verdicts."""
    r = canonical_qint_max_grid_rd_proof()
    assert r.intermediate_values["all_canonical"] is True
    # Bit budgets: log2(2)=1, log2(4)=2, log2(8)=3, log2(16)=4, log2(32)=5
    assert r.intermediate_values["bit_budgets_per_sample"] == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_non_canonical_grid_flagged():
    """Non-power-of-2 grids are flagged as suboptimal."""
    r = canonical_qint_max_grid_rd_proof(QintMaxGridInput(grid=(2, 5, 9)))
    assert r.intermediate_values["all_canonical"] is False


def test_symbol_counts_correct():
    """symbol_counts_no_sign = grid + 1; symbol_counts_with_sign = 2*grid + 1."""
    r = canonical_qint_max_grid_rd_proof()
    assert r.intermediate_values["symbol_counts_no_sign"] == (2, 4, 8, 16, 32)
    assert r.intermediate_values["symbol_counts_with_sign"] == (3, 7, 15, 31, 63)


def test_canonical_grid_constant_pinned():
    """CANONICAL_QINT_MAX_GRID is the empirical optimum."""
    assert CANONICAL_QINT_MAX_GRID == (1, 3, 7, 15, 31)


def test_invalid_inputs_raise():
    """Empty grid + negative entries raise."""
    with pytest.raises(ValueError, match="non-empty"):
        QintMaxGridInput(grid=())
    with pytest.raises(ValueError, match=">= 1"):
        QintMaxGridInput(grid=(0, 3))


def test_citation_cover_thomas():
    """Cover-Thomas Ch.13 citation present."""
    r = canonical_qint_max_grid_rd_proof()
    assert "Cover-Thomas" in r.literature_citation
    assert "R-D" in r.literature_citation or "Rate-Distortion" in r.literature_citation
