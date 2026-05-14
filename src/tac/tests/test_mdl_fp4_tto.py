# SPDX-License-Identifier: MIT
"""Tests for tac.mdl_fp4_tto — MDL/FP4 test-time optimization."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.mdl_fp4_tto import (
    MDLTotalLoss,
    MDLTTOConfig,
    MDLTTOResult,
    encode_byte_count_proxy,
    tto_optimize_mdl,
)


# ── Config validation ─────────────────────────────────────────────────────


def test_config_default_construction():
    cfg = MDLTTOConfig()
    assert cfg.max_iters == 2000
    assert cfg.eval_roundtrip_required is True
    assert cfg.lr > 0
    assert cfg.rate_weight >= 0
    assert cfg.distortion_weight >= 0


def test_config_rejects_zero_max_iters():
    with pytest.raises(ValueError, match="max_iters must be positive"):
        MDLTTOConfig(max_iters=0)


def test_config_rejects_zero_lr():
    with pytest.raises(ValueError, match="lr must be positive"):
        MDLTTOConfig(lr=0)


def test_config_rejects_negative_weights():
    with pytest.raises(ValueError, match="rate/distortion weights must be non-negative"):
        MDLTTOConfig(rate_weight=-1.0)
    with pytest.raises(ValueError, match="rate/distortion weights must be non-negative"):
        MDLTTOConfig(distortion_weight=-0.1)


def test_config_rejects_zero_convergence_window():
    with pytest.raises(ValueError, match="convergence_window must be positive"):
        MDLTTOConfig(convergence_window=0)


# ── MDLTotalLoss forward ───────────────────────────────────────────────────


def test_mdl_total_loss_forward():
    cfg = MDLTTOConfig(rate_weight=2.0, distortion_weight=3.0)
    loss_fn = MDLTotalLoss(cfg)
    rate = torch.tensor(100.0)
    distortion = torch.tensor(50.0)
    total = loss_fn(rate_loss=rate, distortion_loss=distortion)
    assert float(total) == pytest.approx(2.0 * 100.0 + 3.0 * 50.0)


def test_mdl_total_loss_gradient_flows():
    cfg = MDLTTOConfig()
    loss_fn = MDLTotalLoss(cfg)
    rate = torch.tensor(100.0, requires_grad=True)
    distortion = torch.tensor(50.0, requires_grad=True)
    total = loss_fn(rate_loss=rate, distortion_loss=distortion)
    total.backward()
    assert rate.grad is not None
    assert distortion.grad is not None


# ── encode_byte_count_proxy ────────────────────────────────────────────────


def test_byte_count_proxy_basic_arithmetic():
    weights = {"w1": torch.zeros(100), "w2": torch.zeros(200)}
    bits = {"w1": 4.0, "w2": 8.0}
    bc = encode_byte_count_proxy(
        weights=weights,
        bits_per_tensor=bits,
        overhead_bytes=64,
    )
    expected = (100 * 4 + 200 * 8) / 8 + 64  # 50 + 200 + 64 = 314
    assert float(bc) == pytest.approx(expected)


def test_byte_count_proxy_handles_tensor_bits():
    """If bits are tensors (LSQ-learnable step size), gradients flow."""
    weights = {"w1": torch.zeros(100)}
    bits = {"w1": torch.tensor(4.0, requires_grad=True)}
    bc = encode_byte_count_proxy(
        weights=weights,
        bits_per_tensor=bits,
        overhead_bytes=0,
    )
    bc.backward()
    assert bits["w1"].grad is not None
    # d(byte_count) / d(bits) = n_params / 8
    assert float(bits["w1"].grad) == pytest.approx(100.0 / 8.0)


def test_byte_count_proxy_refuses_mismatched_keys():
    weights = {"w1": torch.zeros(100)}
    bits = {"w2": 4.0}
    with pytest.raises(ValueError, match="symmetric difference"):
        encode_byte_count_proxy(weights=weights, bits_per_tensor=bits)


# ── TTO loop ───────────────────────────────────────────────────────────────


class _TinyRenderer(nn.Module):
    """Minimal test substrate (1 linear layer)."""

    def __init__(self, in_dim: int = 4, out_dim: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.linear(latents)


def test_tto_refuses_no_scorer_when_eval_roundtrip_required():
    """If a scorer is provided but eval_roundtrip is not applied, refuse."""
    substrate = _TinyRenderer()
    latents = torch.randn(2, 4)
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}

    def fake_scorer(rendered: torch.Tensor) -> torch.Tensor:
        return rendered.mean()

    cfg = MDLTTOConfig(max_iters=5, eval_roundtrip_required=True)
    with pytest.raises(ValueError, match="eval_roundtrip"):
        tto_optimize_mdl(
            substrate=substrate,
            latents=latents,
            bits_per_tensor=bits,
            scorer_loss_fn=fake_scorer,
            config=cfg,
            eval_roundtrip_applied=False,
        )


def test_tto_runs_when_eval_roundtrip_applied():
    substrate = _TinyRenderer()
    latents = torch.randn(2, 4)
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}

    def fake_scorer(rendered: torch.Tensor) -> torch.Tensor:
        return rendered.mean()

    cfg = MDLTTOConfig(max_iters=5, eval_roundtrip_required=True)
    result = tto_optimize_mdl(
        substrate=substrate,
        latents=latents,
        bits_per_tensor=bits,
        scorer_loss_fn=fake_scorer,
        config=cfg,
        eval_roundtrip_applied=True,
    )
    assert isinstance(result, MDLTTOResult)
    assert result.n_iters_run >= 1
    assert len(result.loss_history) == result.n_iters_run


def test_tto_runs_without_scorer():
    """Rate-only mode: no scorer, no distortion term."""
    substrate = _TinyRenderer()
    latents = torch.randn(2, 4)
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}

    cfg = MDLTTOConfig(max_iters=5)
    result = tto_optimize_mdl(
        substrate=substrate,
        latents=latents,
        bits_per_tensor=bits,
        scorer_loss_fn=None,
        config=cfg,
    )
    assert all(d == 0.0 for d in result.distortion_history)


def test_tto_converges_early_when_loss_plateaus():
    """If the loss plateaus within convergence_window, early-stop fires."""
    substrate = _TinyRenderer()
    latents = torch.randn(2, 4)
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}

    cfg = MDLTTOConfig(
        max_iters=200,
        lr=1e-10,  # essentially zero update — loss should plateau
        convergence_window=10,
        convergence_rel_tol=1e-3,
    )
    result = tto_optimize_mdl(
        substrate=substrate,
        latents=latents,
        bits_per_tensor=bits,
        scorer_loss_fn=None,
        config=cfg,
    )
    assert result.converged is True
    assert result.n_iters_run < 200


def test_tto_refuses_when_no_params_require_grad():
    substrate = _TinyRenderer()
    for p in substrate.parameters():
        p.requires_grad_(False)
    latents = torch.randn(2, 4)  # not requires_grad either
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}

    cfg = MDLTTOConfig(max_iters=5)
    with pytest.raises(ValueError, match="no parameters requires_grad"):
        tto_optimize_mdl(
            substrate=substrate,
            latents=latents,
            bits_per_tensor=bits,
            scorer_loss_fn=None,
            config=cfg,
        )


def test_tto_result_provenance_records_scorer_provided_flag():
    substrate = _TinyRenderer()
    latents = torch.randn(2, 4)
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}
    cfg = MDLTTOConfig(max_iters=3)

    result = tto_optimize_mdl(
        substrate=substrate,
        latents=latents,
        bits_per_tensor=bits,
        scorer_loss_fn=None,
        config=cfg,
    )
    assert result.provenance["scorer_was_provided"] is False
    assert result.provenance["evidence_grade"] == "derivation"


def test_tto_result_records_eval_roundtrip_flag():
    substrate = _TinyRenderer()
    latents = torch.randn(2, 4)
    bits = {n: 4.0 for n, _ in substrate.named_parameters()}

    def fake_scorer(rendered: torch.Tensor) -> torch.Tensor:
        return rendered.mean()

    cfg = MDLTTOConfig(max_iters=3)
    result = tto_optimize_mdl(
        substrate=substrate,
        latents=latents,
        bits_per_tensor=bits,
        scorer_loss_fn=fake_scorer,
        config=cfg,
        eval_roundtrip_applied=True,
    )
    assert result.eval_roundtrip_applied is True
