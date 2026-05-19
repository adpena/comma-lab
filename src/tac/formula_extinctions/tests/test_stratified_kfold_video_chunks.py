# SPDX-License-Identifier: MIT
"""Tests for Row #2 — Stratified k-fold across video chunks."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.stratified_kfold_video_chunks import (
    ValidationSplitInput,
    canonical_validation_split,
)


def test_round_robin_canonical():
    """10 chunks at 15% -> 2 val chunks stride-5."""
    r = canonical_validation_split(ValidationSplitInput(total_chunks=10))
    assert r.intermediate_values["val_chunk_count"] == 2
    assert r.intermediate_values["round_robin_period_K"] == 5
    assert r.solved_value == (0, 5)


def test_train_indices_disjoint_from_val():
    """coupled_adjustments['train_chunk_indices'] is disjoint from val."""
    r = canonical_validation_split(ValidationSplitInput(total_chunks=10))
    val_set = set(r.solved_value)
    train_set = set(r.coupled_adjustments["train_chunk_indices"])
    assert val_set & train_set == set()
    assert val_set | train_set == set(range(10))


def test_min_val_chunks_clipping():
    """fraction tiny -> min_val_chunks=1 takes over."""
    r = canonical_validation_split(
        ValidationSplitInput(total_chunks=100, fraction_for_val=0.005, min_val_chunks=1)
    )
    assert r.intermediate_values["val_chunk_count"] == 1


def test_invalid_inputs_raise():
    """ValueError for bad inputs."""
    with pytest.raises(ValueError, match="total_chunks"):
        ValidationSplitInput(total_chunks=1)
    with pytest.raises(ValueError, match="fraction_for_val"):
        ValidationSplitInput(total_chunks=10, fraction_for_val=0.0)
    with pytest.raises(ValueError, match="fraction_for_val"):
        ValidationSplitInput(total_chunks=10, fraction_for_val=1.0)


def test_citation_present():
    """Bengio 2012 citation in result."""
    r = canonical_validation_split(ValidationSplitInput(total_chunks=10))
    assert "Bengio 2012" in r.literature_citation


def test_atom_emission():
    """Atom emitted with ResolutionPath.FORMULA."""
    from tac.atom.types import ResolutionPath
    r = canonical_validation_split(
        ValidationSplitInput(total_chunks=10),
        emit_arbitrariness_atom=True,
        substrate_id="sub_test",
    )
    atom = r.coupled_adjustments["atom"]
    assert atom.resolution_path == ResolutionPath.FORMULA
