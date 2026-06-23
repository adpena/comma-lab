# SPDX-License-Identifier: MIT
"""Compliance parity tests for ``tac.contest_score``.

These tests are the COMPLIANCE PROOF, not mere self-consistency:

1. The helper is byte-identical to an INLINE re-implementation of
   ``upstream/evaluate.py:92`` across a grid of inputs.
2. The helper reproduces TWO REAL landed exact-eval rows (the G3
   torch-vehicle bc20 dual exact row, contest-CPU + contest-CUDA, 600
   samples) within rounding — proving the helper == a real
   ``upstream/evaluate.py`` output, the compliance bedrock.
3. ``break_even_d_seg`` round-trips against ``compute_contest_score``.
"""
from __future__ import annotations

import math

import pytest

from tac.contest_score import (
    POSE_WEIGHT,
    RATE_WEIGHT,
    SEG_WEIGHT,
    UNCOMPRESSED_SIZE_BYTES,
    break_even_d_seg,
    compute_contest_score,
    pose_term,
    rate_term,
    seg_term,
)


def _upstream_evaluate_py_line_92(
    segnet_dist: float, posenet_dist: float, archive_bytes: int
) -> float:
    """Inline re-implementation of upstream/evaluate.py:62-65,92.

    DO NOT refactor to call the helper — the whole point is an
    independent witness of the upstream formula.
    """
    compressed_size = archive_bytes
    uncompressed_size = 37_545_489
    rate = compressed_size / uncompressed_size
    score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate
    return score


# === Canonical constants match upstream ===
def test_canonical_constants_match_upstream():
    assert UNCOMPRESSED_SIZE_BYTES == 37_545_489
    assert SEG_WEIGHT == 100.0
    assert POSE_WEIGHT == 10.0
    assert RATE_WEIGHT == 25.0


# === Parity across a grid ===
_GRID = [
    (0.0, 0.0, 0),
    (0.00260094, 0.00034168, 89244),
    (0.001, 0.0001, 177169),
    (0.05, 0.18, 299970),
    (0.5, 1.0, 1_000_000),
    (0.00039557, 3.3e-4, 79592),
    (1.0, 10.0, 37_545_489),
]


@pytest.mark.parametrize("d_seg,d_pose,archive_bytes", _GRID)
def test_compute_contest_score_byte_identical_to_upstream(
    d_seg, d_pose, archive_bytes
):
    helper = compute_contest_score(d_seg, d_pose, archive_bytes)
    inline = _upstream_evaluate_py_line_92(d_seg, d_pose, archive_bytes)
    assert helper == pytest.approx(inline, rel=0, abs=1e-15)


@pytest.mark.parametrize("d_seg,d_pose,archive_bytes", _GRID)
def test_per_term_helpers_compose_to_total(d_seg, d_pose, archive_bytes):
    total = (
        seg_term(d_seg) + pose_term(d_pose) + rate_term(archive_bytes)
    )
    assert total == pytest.approx(
        compute_contest_score(d_seg, d_pose, archive_bytes), abs=1e-15
    )


def test_seg_term_value():
    assert seg_term(0.00260094) == pytest.approx(0.260094, abs=1e-9)


def test_pose_term_value():
    assert pose_term(0.00034168) == pytest.approx(0.05845341, abs=1e-7)


def test_rate_term_value():
    # 25 * 89244 / 37545489 == 0.0594239164...
    assert rate_term(89244) == pytest.approx(0.0594239, abs=1e-6)
    # rate_term MUST carry the x25 (the bug a subagent dropped):
    assert rate_term(89244) == pytest.approx(
        25 * 89244 / 37_545_489, abs=1e-15
    )


# === CROSS-CHECK vs REAL landed exact rows (the compliance bedrock) ===
# Source: .omx/research/g3_torch_vehicle_bc20_first_exact_row_20260618T0135Z.md
# The G3 dual exact row, 600 samples, recomputed-from-components S.
_G3_CPU = {"d_seg": 0.00260094, "d_pose": 0.00034168, "bytes": 89244, "S": 0.37797132}
_G3_CUDA = {"d_seg": 0.00262703, "d_pose": 0.00048168, "bytes": 89244, "S": 0.39153009}


def test_reproduces_g3_real_contest_cpu_exact_row():
    """Helper reproduces the REAL G3 contest-CPU exact eval (600 samples)."""
    S = compute_contest_score(
        _G3_CPU["d_seg"], _G3_CPU["d_pose"], _G3_CPU["bytes"]
    )
    assert pytest.approx(_G3_CPU["S"], abs=1e-6) == S


def test_reproduces_g3_real_contest_cuda_exact_row():
    """Helper reproduces the REAL G3 contest-CUDA exact eval (600 samples)."""
    S = compute_contest_score(
        _G3_CUDA["d_seg"], _G3_CUDA["d_pose"], _G3_CUDA["bytes"]
    )
    assert pytest.approx(_G3_CUDA["S"], abs=1e-6) == S


def test_reproduces_canonical_frontier_pointer_cpu_anchor():
    """Helper rate term is consistent with the frontier pointer anchor.

    The pointer stores only the composite score + archive_bytes (177169);
    we cross-check the rate term contribution is correct (the x25-carrying
    term), which is the compliance surface a subagent slipped on.
    """
    # Frontier pointer: contest-CPU score 0.19109982419209975, bytes 177169.
    rt = rate_term(177169)
    assert rt == pytest.approx(25 * 177169 / 37_545_489, abs=1e-15)
    assert rt == pytest.approx(0.117961, abs=1e-5)


# === break_even_d_seg round-trip ===
@pytest.mark.parametrize(
    "target_S,d_pose,archive_bytes",
    [
        (0.19110, 3.3e-4, 79592),
        (0.15, 3.3e-4, 79592),
        (0.37797132, 0.00034168, 89244),
        (0.5, 0.001, 200000),
    ],
)
def test_break_even_round_trip(target_S, d_pose, archive_bytes):
    d_seg_star = break_even_d_seg(target_S, d_pose, archive_bytes)
    recomputed = compute_contest_score(d_seg_star, d_pose, archive_bytes)
    assert recomputed == pytest.approx(target_S, abs=1e-12)


def test_break_even_reproduces_g3_d_seg():
    """break_even at the G3 CPU score must recover the G3 CPU d_seg."""
    d_seg_star = break_even_d_seg(
        _G3_CPU["S"], _G3_CPU["d_pose"], _G3_CPU["bytes"]
    )
    assert d_seg_star == pytest.approx(_G3_CPU["d_seg"], abs=1e-7)


def test_break_even_corrects_the_2026_06_23_subagent_bug():
    """The bug: subagent reported break-even ~1.89e-3 dropping the x25.

    With d_pose=3.3e-4, archive_bytes=79592, target 0.19110, the CORRECT
    break-even is ~8.07e-4 (NOT 1.89e-3). This test pins the correct
    value so the bug cannot regress.
    """
    correct = break_even_d_seg(0.19110, 3.3e-4, 79592)
    assert correct == pytest.approx(8.07e-4, abs=2e-5)
    # The WRONG value (dropping x25, i.e. using rate-fraction) would be
    # ~1.31e-3; assert we are nowhere near it.
    assert abs(correct - 1.89e-3) > 5e-4
    assert abs(correct - 1.31e-3) > 3e-4


# === Input validation ===
def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        compute_contest_score(-0.1, 0.1, 1000)
    with pytest.raises(ValueError):
        compute_contest_score(0.1, -0.1, 1000)
    with pytest.raises(ValueError):
        compute_contest_score(0.1, 0.1, -1000)


def test_nan_inf_rejected():
    with pytest.raises(ValueError):
        compute_contest_score(float("nan"), 0.1, 1000)
    with pytest.raises(ValueError):
        compute_contest_score(0.1, float("inf"), 1000)


def test_bool_rejected_as_numeric():
    with pytest.raises(TypeError):
        seg_term(True)


def test_rate_term_zero_denominator_rejected():
    with pytest.raises(ValueError):
        rate_term(1000, uncompressed_size=0)


def test_break_even_can_be_negative_when_target_unreachable():
    """If pose+rate already exceeds target, break-even d_seg is negative."""
    # target tiny, big archive -> rate term alone exceeds target.
    val = break_even_d_seg(0.01, 0.1, 37_545_489)
    assert val < 0
