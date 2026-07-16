# SPDX-License-Identifier: MIT
"""Isolated tests for segnet_head_rank4_flipdist_20260715 (locked-registry pattern)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    EQUATION_ID,
    HEAD_CENTERED_SINGVALS,
    HEAD_PAIR_NORMS,
    build_segnet_head_rank4_linear_flipdist_v1,
    head_difference_rank,
    head_flip_distance_feature_space,
    head_pair_normals_from_weight,
)

REPO_ROOT = Path(__file__).resolve().parents[3].parent
SEGNET_PATH = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"


def test_flip_distance_law_basic() -> None:
    assert head_flip_distance_feature_space(0.5, 2.0) == pytest.approx(0.25)
    assert head_flip_distance_feature_space(-0.5, 2.0) == pytest.approx(0.25)


def test_flip_distance_rejects_nonpositive_norm() -> None:
    with pytest.raises(ValueError):
        head_flip_distance_feature_space(1.0, 0.0)


def test_head_difference_rank_synthetic() -> None:
    assert head_difference_rank([3.0, 2.0, 2.0, 1.8, 0.0]) == 4
    assert head_difference_rank([1.0, 1e-9]) == 1
    assert head_difference_rank([]) == 0


def test_pair_normals_shape_synthetic() -> None:
    w = np.random.default_rng(0).normal(size=(5, 16, 3, 3))
    normals = head_pair_normals_from_weight(w)
    assert len(normals) == 10
    assert all(v.shape == (144,) for v in normals.values())


def test_equation_builds_and_validates() -> None:
    eq = build_segnet_head_rank4_linear_flipdist_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 2
    assert eq.domain_of_validity["research_only"] is True
    assert eq.domain_of_validity["score_claim"] is False


def test_measured_rank_is_four_constant() -> None:
    assert head_difference_rank(HEAD_CENTERED_SINGVALS) == 4
    # any 5-functional argmax has difference rank <= 4 by algebra
    assert head_difference_rank(HEAD_CENTERED_SINGVALS) <= 4


@pytest.mark.skipif(not SEGNET_PATH.exists(), reason="frozen segnet weights not present")
def test_pair_norms_match_real_frozen_weights() -> None:
    from safetensors import safe_open

    with safe_open(str(SEGNET_PATH), framework="numpy") as f:
        w = f.get_tensor("segmentation_head.0.weight")
    normals = head_pair_normals_from_weight(w)
    classes = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
    for (c, cp), vec in normals.items():
        key = f"{classes[c]}-{classes[cp]}"
        assert float(np.linalg.norm(vec)) == pytest.approx(HEAD_PAIR_NORMS[key], abs=5e-3)
    # exact rank-4 on the real weights
    wf = w.reshape(5, -1)
    sv = np.linalg.svd(wf - wf.mean(axis=0, keepdims=True), compute_uv=False)
    assert head_difference_rank(sv) == 4
