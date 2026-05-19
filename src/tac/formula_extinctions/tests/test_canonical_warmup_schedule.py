# SPDX-License-Identifier: MIT
"""Tests for Row #1 — Goyal+He warmup schedule."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.canonical_warmup_schedule import (
    FormulaSolveResult,
    WarmupScheduleInput,
    canonical_warmup_steps,
)


def test_goyal_5pct_canonical_recovery():
    """Goyal 2017 §2.2 canonical 5% of 1.2M steps = 60_000 warmup steps."""
    r = canonical_warmup_steps(WarmupScheduleInput(total_steps=1_200_000))
    assert r.solved_value == 60_000
    assert "Goyal et al 2017" in r.literature_citation


def test_he_resnet_10pct_recovery():
    """He et al 2016 ResNet 10% anchor = 1000 warmup steps for 10K total."""
    r = canonical_warmup_steps(
        WarmupScheduleInput(total_steps=10_000, fraction_of_total=0.10)
    )
    assert r.solved_value == 1000


def test_min_warmup_clipping():
    """min_warmup_steps clips raw warmup below the floor."""
    r = canonical_warmup_steps(
        WarmupScheduleInput(total_steps=100, fraction_of_total=0.05, min_warmup_steps=20)
    )
    # 5% of 100 = 5; min=20 clips up to 20
    assert r.solved_value == 20
    assert r.intermediate_values["min_clipped"] is True


def test_invalid_inputs_raise():
    """Invalid inputs raise ValueError."""
    with pytest.raises(ValueError, match="total_steps"):
        WarmupScheduleInput(total_steps=0)
    with pytest.raises(ValueError, match="fraction_of_total"):
        WarmupScheduleInput(total_steps=1000, fraction_of_total=0.15)
    with pytest.raises(ValueError, match="fraction_of_total"):
        WarmupScheduleInput(total_steps=1000, fraction_of_total=0.0)
    with pytest.raises(ValueError, match="min_warmup_steps"):
        WarmupScheduleInput(total_steps=1000, min_warmup_steps=-1)


def test_result_is_frozen_dataclass():
    """FormulaSolveResult is frozen + has canonical fields."""
    r = canonical_warmup_steps(WarmupScheduleInput(total_steps=1000))
    assert isinstance(r, FormulaSolveResult)
    with pytest.raises((AttributeError, TypeError)):
        r.solved_value = 999  # type: ignore[misc]
    assert "canonical_warmup_steps" in r.canonical_helper_invocation


def test_atom_emission():
    """emit_arbitrariness_atom=True attaches a canonical Atom."""
    from tac.atom.atom import Atom
    r = canonical_warmup_steps(
        WarmupScheduleInput(total_steps=1000),
        emit_arbitrariness_atom=True,
        substrate_id="test_sub",
    )
    atom = r.coupled_adjustments["atom"]
    assert isinstance(atom, Atom)
    assert atom.atom_id == "warmup_steps_solved_for_test_sub"
