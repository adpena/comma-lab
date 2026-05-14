# SPDX-License-Identifier: MIT
"""Regression tests for tools/pr101_lossy_int4_gptq.py — audit criterion #4.

Covers:

  * Hessian construction is deterministic + correctly damped + PSD.
  * Cholesky-upper inverse round-trips: U^T @ U ~ H_inv.
  * Sequential GPTQ quantization preserves column-order semantics
    (same calibration + same weights -> same output).
  * Roundtrip is byte-faithful: GPTQ-quantized values land EXACTLY on the
    encoder's int4 grid (codes in {-7,...,+7} after dequantize/scale).
  * Local CPU candidate cannot set ready_for_exact_eval_dispatch=True.
  * Bias tensors take the naive-PTQ path (no calibration available).
  * The dispatch contract sets dispatch_blockers including the
    substrate-mismatch caveat.
  * On a fully-zero weight tensor GPTQ degenerates gracefully (no NaNs).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_tool_module():
    """Load tools/pr101_lossy_int4_gptq.py as a module without executing main."""
    path = REPO / "tools" / "pr101_lossy_int4_gptq.py"
    spec = importlib.util.spec_from_file_location("pr101_lossy_int4_gptq", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gptq_module():
    return _load_tool_module()


def test_dispatch_contract_blockers_include_substrate_caveat(gptq_module) -> None:
    """The local GPTQ dispatch contract must list the substrate-mismatch
    caveat (synthetic activations vs PR106 video latents)."""
    contract = gptq_module.local_gptq_dispatch_contract(cuda_eval_worth_testing=True)
    assert contract["cuda_eval_worth_testing"] is True
    # Per the MPS-falsification + strict-scorer rules, no CPU/MPS run can
    # ever flip these flags True. Only contest-CUDA auth eval can.
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["dispatch_attempted"] is False
    blockers = contract["dispatch_blockers"]
    assert "missing_exact_cuda_auth_eval" in blockers
    assert "byte_closed_int4_candidate_packet_missing" in blockers
    assert "synthetic_activations_substrate_mismatch_vs_pr106_video_latents" in blockers


def test_hessian_construction_deterministic_psd_and_damped(gptq_module) -> None:
    """H = 2 X^T X / n + damping * mean(diag(H)) * I must be:
        - deterministic given inputs
        - positive-definite (Cholesky must succeed)
        - the diagonal damping must show up as a measurable shift."""
    rng = np.random.default_rng(0)
    n_samples, in_features = 128, 32
    activations = rng.normal(size=(n_samples, in_features)).astype(np.float32)

    h_a = gptq_module._build_hessian(activations, damping_fraction=0.01)
    h_b = gptq_module._build_hessian(activations, damping_fraction=0.01)
    np.testing.assert_array_equal(h_a, h_b)  # determinism

    # PSD: Cholesky must succeed
    np.linalg.cholesky(h_a)

    # Damping shows up as a non-zero baseline on the diagonal.
    h_no_damp = gptq_module._build_hessian(activations, damping_fraction=0.0)
    diag_diff = np.diag(h_a) - np.diag(h_no_damp)
    expected_shift = 0.01 * float(np.diag(h_no_damp).mean())
    assert np.allclose(diag_diff, expected_shift, atol=1e-6)


def test_cholesky_upper_inverse_round_trips_to_h_inv(gptq_module) -> None:
    """If H = L L^T and U is the upper Cholesky of H_inv, then U^T @ U
    must equal H_inv numerically."""
    rng = np.random.default_rng(1)
    n_samples, in_features = 64, 16
    activations = rng.normal(size=(n_samples, in_features)).astype(np.float32)
    h = gptq_module._build_hessian(activations, damping_fraction=0.01)
    u = gptq_module._cholesky_inverse_upper(h)
    h_inv_reconstructed = u.T @ u
    h_inv_direct = np.linalg.inv(h)
    # Numerical tolerance is loose because we go through TWO Cholesky steps.
    assert np.allclose(h_inv_reconstructed, h_inv_direct, rtol=1e-5, atol=1e-7)


def test_gptq_quantize_tensor_is_deterministic_for_fixed_inputs(gptq_module) -> None:
    """Same calibration + same weights + same config -> identical output."""
    rng = np.random.default_rng(2)
    out_features, in_features = 8, 16
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = rng.normal(size=(128, in_features)).astype(np.float32)
    cfg = gptq_module.GPTQConfig(
        block_size_encoder=32,
        block_size_gptq=8,
        damping_fraction=0.01,
        actorder=False,
    )
    w_q1, _ = gptq_module.gptq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    w_q2, _ = gptq_module.gptq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    np.testing.assert_array_equal(w_q1, w_q2)
    assert w_q1.shape == weight.shape
    assert w_q1.dtype == np.float32


def test_gptq_quantized_values_lie_on_int4_grid(gptq_module) -> None:
    """Every output element of GPTQ must be representable as code * scale
    with code in {-7..+7} and scale matching the encoder's per-block fp16
    grid. We check by reconstructing the codes and asserting bounds."""
    rng = np.random.default_rng(3)
    out_features, in_features = 4, 16
    block_size = 32  # the per-element scales are computed over this
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = rng.normal(size=(64, in_features)).astype(np.float32)

    cfg = gptq_module.GPTQConfig(
        block_size_encoder=block_size,
        block_size_gptq=8,
        damping_fraction=0.01,
        actorder=False,
    )
    w_q, _ = gptq_module.gptq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )

    # Recompute per-element scales as the encoder would.
    scales_per_elem, _ = gptq_module._per_block_max_abs_scales(
        weight.reshape(out_features, in_features), block_size,
    )
    scales_matrix = scales_per_elem.reshape(out_features, in_features)
    # Reverse-derive the int4 codes
    codes = np.round(w_q / np.where(scales_matrix > 0, scales_matrix, 1.0)).astype(np.int32)
    assert codes.min() >= -gptq_module.INT4_RANGE
    assert codes.max() <= gptq_module.INT4_RANGE
    # And the recon must equal codes * scale within fp32 precision.
    recon = codes.astype(np.float32) * scales_matrix.astype(np.float32)
    np.testing.assert_allclose(w_q, recon, rtol=0, atol=1e-6)


def test_gptq_handles_all_zero_tensor_gracefully(gptq_module) -> None:
    """Degenerate input (all zeros) must produce all-zero output, no NaN/Inf."""
    out_features, in_features = 4, 16
    weight = np.zeros((out_features, in_features), dtype=np.float32)
    rng = np.random.default_rng(4)
    activations = rng.normal(size=(32, in_features)).astype(np.float32)

    cfg = gptq_module.GPTQConfig(
        block_size_encoder=32, block_size_gptq=8, damping_fraction=0.01,
    )
    w_q, _ = gptq_module.gptq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    assert np.all(np.isfinite(w_q))
    assert np.allclose(w_q, 0.0)


def test_classify_cpu_proxy_dominated_baseline_blocks_dispatch(gptq_module) -> None:
    """If archive >= PR101 brotli baseline, candidate is Pareto-dominated
    and not worth CUDA-eval — even with low rel_err."""
    verdict, cuda_worth = gptq_module.classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=0.5,
        archive_bytes=gptq_module.PR101_BROTLI_BASELINE_BYTES + 100,
    )
    assert "DOMINATED" in verdict
    assert cuda_worth is False


def test_classify_cpu_proxy_below_baseline_and_below_threshold_routes_to_cuda(gptq_module) -> None:
    """Beating the baseline AND below 5% rel_err = CONDITIONAL or stronger.
    Below 2% = CUDA-EVAL-WORTH-TESTING. We never set ready_for_exact_eval_dispatch."""
    verdict_strong, cuda_strong = gptq_module.classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=1.5,
        archive_bytes=gptq_module.PR101_BROTLI_BASELINE_BYTES - 1000,
    )
    assert verdict_strong == "CUDA-EVAL-WORTH-TESTING"
    assert cuda_strong is True

    verdict_cond, cuda_cond = gptq_module.classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=4.0,
        archive_bytes=gptq_module.PR101_BROTLI_BASELINE_BYTES - 1000,
    )
    assert verdict_cond == "CONDITIONAL-CUDA-EVAL-WORTH-TESTING"
    assert cuda_cond is True


def test_quantize_pr101_state_routes_biases_through_naive_ptq(gptq_module) -> None:
    """Biases must go through the naive-PTQ path (no calibration available);
    weight tensors with calibration go through GPTQ; weight tensors WITHOUT
    calibration also fall back to naive PTQ."""
    # Build a tiny synthetic state dict matching FIXED_STATE_SCHEMA prefix.
    rng = np.random.default_rng(5)
    fp32_state: dict[str, np.ndarray] = {}
    for name, shape in gptq_module.FIXED_STATE_SCHEMA:
        fp32_state[name] = rng.normal(scale=0.1, size=shape).astype(np.float32)

    # Provide calibration only for stem.weight.
    activations = {
        "stem.weight": rng.normal(size=(32, 28)).astype(np.float32),
    }
    cfg = gptq_module.GPTQConfig(
        block_size_encoder=1024, block_size_gptq=128, damping_fraction=0.01,
    )
    quantized, stats = gptq_module.quantize_pr101_state(
        fp32_state, activations, cfg=cfg,
    )
    # Every tensor present + correct shape.
    for name, shape in gptq_module.FIXED_STATE_SCHEMA:
        assert quantized[name].shape == shape, name
        assert np.all(np.isfinite(quantized[name])), name
    # stem.weight took the GPTQ path.
    assert stats["stem.weight"]["path"] == "gptq"
    # stem.bias took the bias path.
    assert stats["stem.bias"]["path"] == "naive_ptq_bias"
    # blocks.0.weight had no calibration -> naive_ptq fallback.
    assert stats["blocks.0.weight"]["path"] == "naive_ptq_no_calibration"
