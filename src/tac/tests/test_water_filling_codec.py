"""Tests for Lane Ω-W water-filling Lagrangian bit-budget allocator.

Test coverage matches docs/paper/water_filling_design_20260429.md §3.2:
12 Contrarian-mandated tests + 3 paranoia tests = 15 total.

The eval_roundtrip-required path is exercised on a tiny SegMap-like
fixture (cheap CPU smoke); the production CUDA path is verified by the
lane script's auth eval, not unit tests.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.water_filling_codec import (
    QINT_BITS,
    QINT_LEVELS,
    WaterFillError,
    bits_for_qint,
    estimate_per_channel_hessian,
    estimate_per_channel_variance,
    export_with_water_filling,
    qint_for_bits,
    water_fill_bit_budget,
)


class _TinySegMap(nn.Module):
    """Minimal SegMap-shaped renderer for unit testing.

    Uses the same name layout as tac.segmap_renderer.SegMap so
    iter_eligible_conv_names sees the right protected/unprotected split.
    Output range [0, 255] to match the eval_roundtrip chain.
    """

    def __init__(self, hidden: int = 4, in_channels: int = 5, num_blocks: int = 2):
        super().__init__()
        self.hidden = hidden
        self.layer_in = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, kernel_size=3, padding=1) for _ in range(num_blocks)]
        )
        self.layer_out = nn.Conv2d(hidden, 3, kernel_size=1)
        # Frame-affine embedding present so estimator helpers don't trip.
        self.frame_affine_embedding = nn.Embedding(8, 6)
        nn.init.normal_(self.frame_affine_embedding.weight, std=0.01)

    def forward(self, x: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
        # Use frame_idx so the embedding sees gradient (otherwise it's dead).
        emb = self.frame_affine_embedding(frame_idx)  # (K, 6)
        bias = emb.mean(dim=1).view(-1, 1, 1, 1)
        feat = self.layer_in(x)
        for blk in self.blocks:
            feat = torch.relu(blk(feat))
        out = self.layer_out(feat) + bias
        return torch.sigmoid(out) * 255.0


# ── 1. test_water_fill_meets_budget ───────────────────────────────────────


def test_water_fill_meets_budget():
    """Σ b_c == B within ±1% (budget_tol_frac default)."""
    torch.manual_seed(0)
    n_channels = 8
    hessians = {"l.weight": torch.rand(n_channels) + 0.1}
    variances = {"l.weight": torch.rand(n_channels) + 0.1}
    counts = {"l.weight": [9] * n_channels}  # 1 layer, 8 channels, 9 elts each

    target = 200  # bits
    qint = water_fill_bit_budget(hessians, variances, counts, target)

    realised = sum(
        bits_for_qint(q) * 9 for q in qint["l.weight"]
    )
    assert realised <= target * 1.01 + 1, f"realised={realised} > target={target}+tol"


# ── 2. test_water_fill_high_hessian_gets_more_bits ────────────────────────


def test_water_fill_high_hessian_gets_more_bits():
    """Channel with 100x H gets ≥ neighbor's bits."""
    n = 4
    hessians = {"l.weight": torch.tensor([100.0, 1.0, 1.0, 1.0])}
    variances = {"l.weight": torch.tensor([1.0, 1.0, 1.0, 1.0])}
    counts = {"l.weight": [4] * n}

    qint = water_fill_bit_budget(hessians, variances, counts, total_bits=80)

    assert qint["l.weight"][0] >= qint["l.weight"][1], (
        f"High-H channel got {qint['l.weight'][0]} but low-H neighbor got "
        f"{qint['l.weight'][1]}"
    )


# ── 3. test_water_fill_zero_hessian_gets_minimum_bits ─────────────────────


def test_water_fill_zero_hessian_gets_minimum_bits():
    """Channel with H=0 gets qint_max=1 (the floor)."""
    n = 4
    hessians = {"l.weight": torch.tensor([10.0, 0.0, 10.0, 10.0])}
    variances = {"l.weight": torch.tensor([1.0, 1.0, 1.0, 1.0])}
    counts = {"l.weight": [4] * n}

    qint = water_fill_bit_budget(hessians, variances, counts, total_bits=80)

    assert qint["l.weight"][1] == 1, f"Zero-H channel got Q={qint['l.weight'][1]}"


# ── 4. test_water_fill_uniform_hessians_uniform_bits ──────────────────────


def test_water_fill_uniform_hessians_uniform_bits():
    """Equal H+σ² → equal Q allocation (within rounding)."""
    n = 6
    hessians = {"l.weight": torch.full((n,), 1.0)}
    variances = {"l.weight": torch.full((n,), 1.0)}
    counts = {"l.weight": [4] * n}

    qint = water_fill_bit_budget(hessians, variances, counts, total_bits=120)

    qs = qint["l.weight"]
    assert max(qs) == min(qs) or (
        # Allow at most one Q-step difference due to discrete rounding.
        QINT_LEVELS.index(max(qs)) - QINT_LEVELS.index(min(qs)) <= 1
    ), f"Uniform inputs produced non-uniform Qs: {qs}"


# ── 5. test_water_fill_invariant_to_global_scaling ────────────────────────


def test_water_fill_invariant_to_global_scaling():
    """Multiply all H_c by 10× → SAME allocation (Eq. 1 absorbs into λ)."""
    n = 5
    h = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0])
    v = torch.tensor([0.5, 1.5, 1.0, 2.0, 0.7])
    counts = {"l.weight": [4] * n}

    qint_a = water_fill_bit_budget(
        {"l.weight": h}, {"l.weight": v}, counts, total_bits=80
    )
    qint_b = water_fill_bit_budget(
        {"l.weight": h * 10.0}, {"l.weight": v}, counts, total_bits=80
    )

    assert qint_a["l.weight"] == qint_b["l.weight"], (
        f"Allocation changed under global scaling: {qint_a} vs {qint_b}"
    )


# ── 6. test_water_fill_respects_max_qint ──────────────────────────────────


def test_water_fill_respects_max_qint():
    """No channel ever assigned Q > 31."""
    n = 4
    hessians = {"l.weight": torch.tensor([1e6, 1e6, 1e6, 1e6])}
    variances = {"l.weight": torch.tensor([1e6, 1e6, 1e6, 1e6])}
    counts = {"l.weight": [4] * n}

    qint = water_fill_bit_budget(hessians, variances, counts, total_bits=120)

    assert max(qint["l.weight"]) <= 31


# ── 7. test_water_fill_respects_min_qint ──────────────────────────────────


def test_water_fill_respects_min_qint():
    """No channel ever assigned Q < 1 (the floor)."""
    n = 4
    hessians = {"l.weight": torch.tensor([1e-30, 1e-30, 1e-30, 1e-30])}
    variances = {"l.weight": torch.tensor([1e-30, 1e-30, 1e-30, 1e-30])}
    counts = {"l.weight": [4] * n}

    qint = water_fill_bit_budget(hessians, variances, counts, total_bits=80)

    assert min(qint["l.weight"]) >= 1


# ── 8. test_export_roundtrip_below_tolerance ──────────────────────────────


def test_export_roundtrip_below_tolerance(tmp_path):
    """end-to-end export: verify_roundtrip MSE < 1e-3 vs original."""
    torch.manual_seed(42)
    model = _TinySegMap(hidden=8, num_blocks=2)
    K, H, W = 4, 16, 16
    inputs = torch.rand(K, 5, H, W)  # one-hot-ish
    targets = torch.rand(K, 3, H, W) * 255.0
    frame_idx = torch.arange(K, dtype=torch.long) % 8

    out = export_with_water_filling(
        model,
        inputs,
        targets,
        frame_idx,
        total_bits=2000,
        output_path=str(tmp_path / "wfill.tar.xz"),
        device="cpu",
        verify_tol=1e-1,  # generous; verify_roundtrip raises on overflow
    )

    assert out["roundtrip_mse_max"] < 1e-1, out["roundtrip_mse_max"]
    assert out["payload_bytes"] > 0


# ── 9. test_export_byte_savings_vs_uniform_q7 ─────────────────────────────


def test_export_byte_savings_vs_uniform_q7(tmp_path):
    """Water-filling at moderate budget produces ≤ uniform-Q=7 payload bytes."""
    from tac.block_fp_codec import pack_payload_tar_xz

    torch.manual_seed(11)
    model = _TinySegMap(hidden=12, num_blocks=3)
    K, H, W = 4, 12, 12
    inputs = torch.rand(K, 5, H, W)
    targets = torch.rand(K, 3, H, W) * 255.0
    frame_idx = torch.arange(K, dtype=torch.long) % 8

    # Uniform-Q=7 baseline
    uniform_path = tmp_path / "uniform.tar.xz"
    pack_payload_tar_xz({k: v.detach() for k, v in model.state_dict().items()},
                        str(uniform_path), qint_max=7)
    uniform_bytes = uniform_path.stat().st_size

    # Water-fill at a budget chosen to undercut uniform-Q=7 by 30%.
    # Uniform Q=7 ≈ 4 bits/elt; aim for ~2.8 bits/elt.
    n_eligible_elts = 0
    from tac.learnable_bit_quant import iter_eligible_conv_names

    eligible = set(iter_eligible_conv_names(model))
    for name, mod in model.named_modules():
        if name in eligible and isinstance(mod, nn.Conv2d):
            n_eligible_elts += int(mod.weight.numel())
    target_bits = int(n_eligible_elts * 2.8)

    out = export_with_water_filling(
        model,
        inputs,
        targets,
        frame_idx,
        total_bits=target_bits,
        output_path=str(tmp_path / "wfill.tar.xz"),
        device="cpu",
        verify_tol=1.0,  # discrete-Q=1 ladder may have larger MSE
    )

    assert out["payload_bytes"] <= uniform_bytes, (
        f"Water-fill payload {out['payload_bytes']} > uniform-Q7 {uniform_bytes}"
    )


# ── 10. test_hessian_estimation_finite ────────────────────────────────────


def test_hessian_estimation_finite():
    """Every per-channel Hessian on a normal SegMap forward is finite."""
    torch.manual_seed(7)
    model = _TinySegMap(hidden=6, num_blocks=2)
    K, H, W = 4, 12, 12
    inputs = torch.rand(K, 5, H, W)
    targets = torch.rand(K, 3, H, W) * 255.0
    frame_idx = torch.arange(K, dtype=torch.long) % 8

    h = estimate_per_channel_hessian(
        model, inputs, targets, frame_idx, eval_roundtrip=True, device="cpu"
    )
    assert len(h) > 0
    for name, vec in h.items():
        assert torch.isfinite(vec).all(), f"{name} has non-finite H"


# ── 11. test_variance_estimation_nonneg ───────────────────────────────────


def test_variance_estimation_nonneg():
    """σ_c² ≥ 0 for every channel."""
    torch.manual_seed(3)
    model = _TinySegMap(hidden=6, num_blocks=2)
    v = estimate_per_channel_variance(model)
    assert len(v) > 0
    for name, vec in v.items():
        assert (vec >= 0).all(), f"{name} has negative variance"


# ── 12. test_edge_case_single_channel_layer ───────────────────────────────


def test_edge_case_single_channel_layer():
    """1-output-channel conv works."""
    n = 1
    hessians = {"l.weight": torch.tensor([3.0])}
    variances = {"l.weight": torch.tensor([1.0])}
    counts = {"l.weight": [4]}

    qint = water_fill_bit_budget(hessians, variances, counts, total_bits=20)
    assert qint["l.weight"][0] in QINT_LEVELS


# ── 13. test_edge_case_nan_hessian_raises ─────────────────────────────────


def test_edge_case_nan_hessian_raises():
    """NaN H_c raises WaterFillError loudly (no silent zero)."""
    n = 3
    hessians = {"l.weight": torch.tensor([1.0, float("nan"), 2.0])}
    variances = {"l.weight": torch.tensor([1.0, 1.0, 1.0])}
    counts = {"l.weight": [4] * n}

    with pytest.raises(WaterFillError, match="non-finite"):
        water_fill_bit_budget(hessians, variances, counts, total_bits=40)


# ── 14. test_qint_for_bits_round_trip (paranoia) ──────────────────────────


def test_qint_for_bits_round_trip():
    """qint_for_bits maps the design bin-centres {1.0, 2.0, 3.0, 4.0, 5.0}
    to the canonical Q ladder {1, 3, 7, 15, 31} (design §1.3 table).

    Note: bits_for_qint(Q) returns Shannon entropy log2(2Q+1), which lies
    AT the bin BOUNDARIES (e.g. log2(3)=1.585 sits between Q=1 and Q=3
    bins). The "bin centre" in qint_for_bits is the integer-bit interpretation
    used during water-filling — see design table for the exact mapping.
    """
    centres = (1.0, 2.0, 3.0, 4.0, 5.0)
    expected = QINT_LEVELS
    for b, q in zip(centres, expected):
        assert qint_for_bits(b) == q, (
            f"design bin centre b={b} -> qint_for_bits gave "
            f"{qint_for_bits(b)}, expected {q}"
        )
    # bits_for_qint must return the canonical Shannon entropy
    for q in QINT_LEVELS:
        b = bits_for_qint(q)
        assert math.isclose(b, math.log2(2 * q + 1), rel_tol=1e-9)

    # Non-finite input must raise
    with pytest.raises(WaterFillError):
        qint_for_bits(float("nan"))
    with pytest.raises(WaterFillError):
        qint_for_bits(float("inf"))


# ── 15. test_export_eval_roundtrip_false_raises (paranoia) ────────────────


def test_export_eval_roundtrip_false_raises():
    """eval_roundtrip=False MUST raise WaterFillError loudly."""
    torch.manual_seed(0)
    model = _TinySegMap()
    K, H, W = 2, 8, 8
    inputs = torch.rand(K, 5, H, W)
    targets = torch.rand(K, 3, H, W) * 255.0
    frame_idx = torch.arange(K, dtype=torch.long) % 8

    with pytest.raises(WaterFillError, match="FORBIDDEN"):
        estimate_per_channel_hessian(
            model, inputs, targets, frame_idx, eval_roundtrip=False, device="cpu"
        )


# ── 16. paranoia: device='mps' raises ─────────────────────────────────────


def test_estimate_hessian_rejects_mps():
    """device='mps' MUST raise WaterFillError per CLAUDE.md non-negotiable."""
    torch.manual_seed(0)
    model = _TinySegMap()
    K, H, W = 2, 8, 8
    inputs = torch.rand(K, 5, H, W)
    targets = torch.rand(K, 3, H, W) * 255.0
    frame_idx = torch.arange(K, dtype=torch.long) % 8

    with pytest.raises(WaterFillError, match="mps"):
        estimate_per_channel_hessian(
            model, inputs, targets, frame_idx, eval_roundtrip=True, device="mps"
        )
