# SPDX-License-Identifier: MIT
"""Tests for tie-aware factor-2 preimage selection and its optimality certificate."""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.tie_aware_preimage import (
    canonical_preimage_fp32_residual,
    enumerate_block_preimages,
    select_tie_aware_factor2_uint8,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
)

CONTEST_KW = dict(camera_h=874, camera_w=1164, scorer_h=384, scorer_w=512)


def _contest_operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(**CONTEST_KW)


def _torch_resize_oracle():
    import torch

    def oracle(frame: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(frame)).float().permute(2, 0, 1)[None]
        y = torch.nn.functional.interpolate(x, size=(384, 512), mode="bilinear")
        return y[0].permute(1, 2, 0).contiguous().numpy()

    return oracle


def _small_plane(op: DisjointResizeOperator, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(op.scorer_h, op.scorer_w, 3), dtype=np.uint8)


# ---- enumerate_block_preimages ------------------------------------------------


def test_enumerate_includes_canonical_and_all_exact():
    # A real contest block: rows (278,490) sum 768, cols (490,278) sum 1024.
    row_num, col_num, y = (278, 490), (490, 278), 137
    d = (row_num[0] + row_num[1]) * (col_num[0] + col_num[1])
    coeff = np.outer(row_num, col_num).reshape(-1)
    ties = enumerate_block_preimages(row_num, col_num, y)
    assert ties.shape[1] == 4
    # every enumerated member is an exact preimage
    sums = ties @ coeff
    assert np.all(sums == y * d)
    assert np.all((ties >= 0) & (ties <= 255))
    # canonical all-y member is present
    assert np.any(np.all(ties == y, axis=1))
    # interior target has a rich (non-singleton) tie set
    assert ties.shape[0] > 1


def test_enumerate_rejects_out_of_range_byte():
    with pytest.raises(Uint8LatticeError):
        enumerate_block_preimages((278, 490), (490, 278), 300)


def test_enumerate_extreme_byte_is_singleton():
    # y at the box corner leaves only the all-y solution.
    ties = enumerate_block_preimages((86, 682), (372, 652), 255)
    assert ties.shape[0] == 1
    assert np.all(ties[0] == 255)


# ---- canonical fp32-exactness theorem (the module's central measured fact) ----


def test_canonical_preimage_is_fp32_exact_contest_geometry():
    pytest.importorskip("torch")
    op = _contest_operator()
    oracle = _torch_resize_oracle()
    for seed in (0, 1, 2):
        y = _small_plane(op, seed)
        res = canonical_preimage_fp32_residual(op, y, oracle)
        assert res.is_fp32_exact
        assert res.max_abs == 0.0
        assert res.nonzero_values == 0
        assert res.total_values == y.size


# ---- selector fast path (optimality certificate) ------------------------------


def test_selector_returns_canonical_with_certificate_on_contest_geometry():
    pytest.importorskip("torch")
    op = _contest_operator()
    oracle = _torch_resize_oracle()
    y = _small_plane(op, 7)
    result = select_tie_aware_factor2_uint8(op, y, oracle)
    assert result.optimal_certificate is True
    assert result.canonical_residual_max == 0.0
    assert result.blocks_searched == 0
    assert result.blocks_improved == 0
    assert result.numerator_exact is True
    # frame is exactly the canonical support-fill (no tie selected)
    canonical = realize_factor2_uint8_scorer_plane(op, y)
    assert np.array_equal(result.frame, canonical)


def test_selector_preserves_scorer_plane_bytes():
    """Byte-identity: the returned preimage reproduces the exact plane numerators."""
    pytest.importorskip("torch")
    op = _contest_operator()
    oracle = _torch_resize_oracle()
    y = _small_plane(op, 11)
    result = select_tie_aware_factor2_uint8(op, y, oracle)
    verification = op.verify_factor2_uint8(result.frame, y)
    assert verification.numerator_exact
    assert verification.certified_exact


def test_selector_frame_is_read_only_copy():
    pytest.importorskip("torch")
    op = _contest_operator()
    y = _small_plane(op, 3)
    result = select_tie_aware_factor2_uint8(op, y, _torch_resize_oracle())
    with pytest.raises(ValueError):
        result.frame[0, 0, 0] = 1


# ---- selector search machinery (non-exact oracle exercises the fallback) -------


def test_selector_search_path_runs_and_stays_exact():
    """A residual-injecting oracle forces the search branch; exactness must hold.

    The mock oracle reports canonical residual > 0 so the fast-path certificate
    does not fire.  The selector must enumerate ties, keep the plane bytes exact,
    and never return a non-exact preimage.  Runs on a small factor-2 operator so
    the (per-block, exhaustive) search is fast.
    """
    op = DisjointResizeOperator.build(camera_h=10, camera_w=10, scorer_h=3, scorer_w=3)

    def biased_oracle(frame: np.ndarray) -> np.ndarray:
        # exact rational resize plus a fixed +0.4 bias => canonical never matches Y,
        # forcing the search branch; the bias is identical across candidates so no
        # tie strictly improves, exercising the "searched but not improved" path.
        return op.apply(frame.astype(np.float64)) + 0.4

    rng = np.random.default_rng(5)
    y = rng.integers(0, 256, size=(3, 3, 2), dtype=np.uint8)
    result = select_tie_aware_factor2_uint8(
        op, y, biased_oracle, max_candidates_per_block=8
    )
    assert result.optimal_certificate is False
    assert result.blocks_searched >= 1
    assert 0 <= result.blocks_improved <= result.blocks_searched
    assert result.numerator_exact is True
    # The load-bearing invariant: whatever tie is chosen, the scorer plane bytes
    # are preserved exactly (byte-identity of the archive by construction).  A tie
    # that differs from the canonical fill is still numerator-exact (same plane),
    # even though it is not the canonical realization.
    verification = op.verify_factor2_uint8(result.frame, y)
    assert verification.numerator_exact
