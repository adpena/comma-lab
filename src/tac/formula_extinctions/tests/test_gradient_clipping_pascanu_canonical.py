# SPDX-License-Identifier: MIT
"""Tests for Row #9 — Pascanu 2013 gradient-norm clipping."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.gradient_clipping_pascanu_canonical import (
    GradientClipInput,
    canonical_gradient_clipping_norm,
)


def test_rnn_canonical_threshold_1():
    """Pascanu §4.2: RNN canonical clip = 1.0."""
    r = canonical_gradient_clipping_norm(GradientClipInput(architecture_class="rnn"))
    assert r.solved_value == 1.0


def test_cnn_canonical_threshold_5():
    """CNN/feed-forward canonical clip = 5.0."""
    r = canonical_gradient_clipping_norm(GradientClipInput(architecture_class="cnn"))
    assert r.solved_value == 5.0


def test_transformer_canonical_1():
    """Transformer (Pascanu §4.2 lineage) = 1.0."""
    r = canonical_gradient_clipping_norm(GradientClipInput(architecture_class="transformer"))
    assert r.solved_value == 1.0


def test_data_driven_99th_percentile():
    """data_driven=99th percentile of observed norms (nearest-rank method).

    For 200 sorted values with the top 2 = 10.0 and rest = 0.1, the
    nearest-rank 99th percentile is at index ceil(0.99 * 200) - 1 = 197,
    which is still 0.1 (the threshold should sit BELOW the rare outliers
    so clipping fires on them).
    """
    norms = [0.1] * 198 + [10.0, 10.0]
    r = canonical_gradient_clipping_norm(GradientClipInput(
        architecture_class="data_driven",
        observed_gradient_norms=norms,
    ))
    # 99th percentile of 200 sorted values = index 197 -> 0.1 (clip outliers above)
    assert r.solved_value == 0.1
    # 100th percentile = max = 10.0
    r_max = canonical_gradient_clipping_norm(GradientClipInput(
        architecture_class="data_driven",
        observed_gradient_norms=norms,
        percentile=100.0,
    ))
    assert r_max.solved_value == 10.0


def test_invalid_inputs_raise():
    """Empty data_driven + bad architecture + negative norms raise."""
    with pytest.raises(ValueError, match="architecture_class"):
        GradientClipInput(architecture_class="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="data_driven"):
        GradientClipInput(architecture_class="data_driven", observed_gradient_norms=())
    with pytest.raises(ValueError, match="non-negative"):
        GradientClipInput(
            architecture_class="data_driven",
            observed_gradient_norms=[1.0, -0.5],
        )
    with pytest.raises(ValueError, match="percentile"):
        GradientClipInput(
            architecture_class="data_driven",
            observed_gradient_norms=[1.0],
            percentile=49.0,
        )


def test_citation_pascanu():
    """Pascanu+Mikolov+Bengio 2013 citation."""
    r = canonical_gradient_clipping_norm(GradientClipInput(architecture_class="rnn"))
    assert "Pascanu" in r.literature_citation
    assert "arxiv:1211.5063" in r.literature_citation
