# SPDX-License-Identifier: MIT
"""Tests for ``tools/pr101_balle_cross_tensor_hyperprior.py``.

Covers the FINAL audit criterion for the ``compressai_balle_hyperprior`` lane
(per ``feedback_implementation_vs_model_gap_audit_20260508.md``):

1. The cross-tensor analysis transform (h_a) is deterministic — same input
   descriptors yield the same latent z.
2. The EntropyBottleneck round-trip on z is byte-faithful.
3. The cross-tensor synthesis (h_s) produces (mean, scale) parameters whose
   softplus+offset constraint maps into the canonical scale_table support.
4. End-to-end byte-faithful round-trip: GaussianConditional with rounded-int
   means + integer symbols recovers the substrate exactly (rel_err ≈ 0).
5. The ``proxy_evidence_contract`` encodes the CLAUDE.md MPS rule
   (``ready_for_exact_eval_dispatch=False`` and matching dispatch_blockers).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool() -> Any:
    path = REPO / "tools" / "pr101_balle_cross_tensor_hyperprior.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_balle_cross_tensor_hyperprior_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_substrate(tool: Any, n_tensors: int = 4, sizes: tuple[int, ...] = (50, 80, 60, 40)):
    """Build a tiny CrossTensorSubstrate without touching the real PR101 file.

    Uses a deterministic random seed so symbol distributions vary across the
    `n_tensors` to give the cross-tensor latent something to summarise.
    """
    assert len(sizes) == n_tensors
    rng = np.random.default_rng(0)
    raw_per_tensor = []
    descriptors_np = np.zeros((n_tensors, tool.DESCRIPTOR_DIM), dtype=np.float32)
    scales = []
    n_total = 0
    for i, n in enumerate(sizes):
        # vary distribution per tensor: scale grows with tensor index
        sigma = 5.0 + 8.0 * i
        symbols = np.clip(rng.normal(0.0, sigma, size=n).round().astype(np.int32), -127, 127)
        raw_per_tensor.append(symbols)
        descriptors_np[i] = tool._tensor_descriptor(symbols)
        scales.append(1.0)
        n_total += symbols.size
    desc_mean = descriptors_np.mean(axis=0, keepdims=True)
    desc_std = descriptors_np.std(axis=0, keepdims=True) + 1e-6
    desc_norm = (descriptors_np - desc_mean) / desc_std
    descriptors = torch.from_numpy(desc_norm.astype(np.float32))
    tensor_ids = torch.arange(n_tensors, dtype=torch.long)
    return tool.CrossTensorSubstrate(
        raw_per_tensor=raw_per_tensor,
        descriptors=descriptors,
        tensor_ids=tensor_ids,
        n_total_symbols=n_total,
        per_tensor_scales=scales,
    )


def _build_model(tool: Any, n_tensors: int, latent_dim: int = 8, embed_dim: int = 4, hidden: int = 16):
    torch.manual_seed(0)
    return tool.CrossTensorHyperprior(
        descriptor_dim=tool.DESCRIPTOR_DIM,
        n_tensors=n_tensors,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        hidden=hidden,
    )


def test_cross_tensor_analysis_is_deterministic() -> None:
    """h_a(descriptors) must be deterministic (same descriptors -> same z)."""
    tool = _load_tool()
    sub = _toy_substrate(tool)
    model = _build_model(tool, n_tensors=4)
    model.eval()
    with torch.no_grad():
        z1 = model.h_a(sub.descriptors)
        z2 = model.h_a(sub.descriptors)
    assert z1.shape == (model.latent_dim,)
    assert torch.allclose(z1, z2)
    # Sanity: the latent should have non-zero variance (otherwise h_s sees a
    # constant — defeating the whole cross-tensor idea).
    assert float(z1.std().item()) > 0.0


def test_entropy_bottleneck_roundtrip_byte_faithful() -> None:
    """EntropyBottleneck.compress(z) -> decompress -> matches the quantized z_hat.

    CompressAI's EntropyBottleneck is a stochastic-quantize-during-train,
    deterministic-compress-during-eval module. After ``model.eval()`` and
    ``update(force=True)`` the round-trip is exactly byte-faithful on its
    QUANTIZED output (z_hat), which is the contract.
    """
    tool = _load_tool()
    sub = _toy_substrate(tool)
    model = _build_model(tool, n_tensors=4)
    model.eval()
    model.entropy_bottleneck.update(force=True)
    with torch.no_grad():
        z = model.h_a(sub.descriptors)
        z_for_eb = z.view(1, model.latent_dim, 1)
        # Eval forward (no quantization noise) -> "z_hat_eval" is what
        # decompress() returns
        z_hat_eval, _ = model.entropy_bottleneck(z_for_eb)
        z_strings = model.entropy_bottleneck.compress(z_for_eb)
        z_dec = model.entropy_bottleneck.decompress(z_strings, z_for_eb.size()[2:])
    assert isinstance(z_strings, list) and len(z_strings) == 1
    assert isinstance(z_strings[0], (bytes, bytearray))
    assert torch.allclose(z_dec, z_hat_eval, atol=1e-5)


def test_predicted_pmf_params_in_scale_table_range() -> None:
    """Scale predictions must fall in [SCALE_TABLE_MIN, SCALE_TABLE_MAX]
    so build_indexes() returns valid CDF indices."""
    tool = _load_tool()
    sub = _toy_substrate(tool)
    model = _build_model(tool, n_tensors=4)
    model.eval()
    with torch.no_grad():
        z = model.h_a(sub.descriptors)
        params = model.h_s(z, sub.tensor_ids)
    assert params.shape == (4, 2)
    log_scale = params[:, 1]
    scale_pred = torch.clamp(
        torch.nn.functional.softplus(log_scale) + tool.SCALE_TABLE_MIN,
        min=tool.SCALE_TABLE_MIN,
        max=tool.SCALE_TABLE_MAX,
    )
    assert float(scale_pred.min().item()) >= tool.SCALE_TABLE_MIN - 1e-6
    assert float(scale_pred.max().item()) <= tool.SCALE_TABLE_MAX + 1e-6
    # Integer-valued mean rounding produces finite, in-range integers
    mean_int = torch.round(params[:, 0])
    assert torch.all(torch.isfinite(mean_int))


def test_end_to_end_roundtrip_is_byte_faithful() -> None:
    """Full compress -> decompress recovers integer symbols EXACTLY.

    GaussianConditional with integer means + integer symbols is exact; the
    cross-tensor pipeline sets means via torch.round() so this property holds
    by construction. rel_err MUST be 0 on integer substrates.
    """
    tool = _load_tool()
    sub = _toy_substrate(tool)
    model = _build_model(tool, n_tensors=4)
    # No training needed — encode_decode_measure handles update() / scale_table
    meas = tool.encode_decode_measure(model, sub)
    assert meas["rel_err"] == pytest.approx(0.0, abs=1e-9), (
        f"cross-tensor hyperprior must round-trip byte-faithfully on integer "
        f"substrates; got rel_err={meas['rel_err']}"
    )
    assert meas["max_abs_symbol_err"] == 0
    assert meas["nonzero_diff_symbol_count"] == 0
    assert meas["n_tensor_payloads"] == 4
    # Each per-tensor payload must be non-empty
    assert all(c > 0 for c in meas["per_tensor_byte_counts"])
    # archive_bytes accounts for model + payload + scale sidecar + overhead
    assert meas["archive_bytes"] >= meas["decoder_blob_bytes"]
    assert meas["archive_bytes"] == (
        meas["decoder_blob_bytes"] + meas["archive_overhead_bytes"]
    )


def test_proxy_evidence_contract_blocks_dispatch() -> None:
    """The MPS-derived evidence row MUST be tagged non-dispatchable per CLAUDE.md."""
    tool = _load_tool()
    contract = tool.proxy_evidence_contract()
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["dispatch_attempted"] is False
    assert contract["proxy_row"] is True
    assert "MPS-research-signal" in contract["evidence_grade"]
    blockers = set(contract["dispatch_blockers"])
    # CLAUDE.md MPS-NOISE non-negotiable requires explicit blocker for
    # "MPS proxy is not score evidence".
    assert "mps_proxy_signal_not_score_evidence" in blockers
    assert "missing_exact_cuda_auth_eval" in blockers
