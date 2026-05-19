# SPDX-License-Identifier: MIT
"""Tests for Row #11 — Smith 2017 warmup init_lr factor."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.lr_warmup_init_smith_canonical import (
    DEFAULT_INIT_LR_FACTOR_SMITH,
    WarmupInitLRInput,
    canonical_lr_warmup_init_lr_factor,
)


def test_smith_canonical_factor_0_1():
    """Smith §3.2 canonical: init_lr_factor = 0.1."""
    assert DEFAULT_INIT_LR_FACTOR_SMITH == 0.1


def test_default_base_lr_5e_4_recovers_5e_5():
    """5e-4 base_lr * 0.1 = 5e-05."""
    r = canonical_lr_warmup_init_lr_factor(WarmupInitLRInput(base_lr=5e-4))
    assert r.solved_value == pytest.approx(5e-5)


def test_custom_factor_overrides_smith():
    """Custom factor (e.g. 0.01) overrides default."""
    r = canonical_lr_warmup_init_lr_factor(
        WarmupInitLRInput(base_lr=1e-3, init_lr_factor=0.01)
    )
    assert r.solved_value == pytest.approx(1e-5)
    assert r.intermediate_values["is_canonical_smith"] is False


def test_canonical_smith_flag_set():
    """is_canonical_smith=True when factor == 0.1."""
    r = canonical_lr_warmup_init_lr_factor(WarmupInitLRInput(base_lr=1e-3))
    assert r.intermediate_values["is_canonical_smith"] is True


def test_invalid_inputs_raise():
    """Bad inputs raise."""
    with pytest.raises(ValueError, match="base_lr"):
        WarmupInitLRInput(base_lr=0.0)
    with pytest.raises(ValueError, match="init_lr_factor"):
        WarmupInitLRInput(base_lr=1e-3, init_lr_factor=0.0)
    with pytest.raises(ValueError, match="init_lr_factor"):
        WarmupInitLRInput(base_lr=1e-3, init_lr_factor=1.0)


def test_citation_smith():
    """Smith 2017 citation present."""
    r = canonical_lr_warmup_init_lr_factor(WarmupInitLRInput(base_lr=1e-3))
    assert "Smith 2017" in r.literature_citation
    assert "arxiv:1506.01186" in r.literature_citation
