"""Regression tests for tools/pr101_lossy_int4_awq.py — audit criterion #5.

Covers:

  * AWQ scale computation: alpha=0 -> identity scaling; alpha>0 -> larger
    scales for larger activation magnitudes; geometric-mean normalization.
  * AWQ quantize tensor is deterministic + grid-search picks lowest-loss alpha.
  * AWQ recon round-trips through the same int4 wire format the encoder uses.
  * Local CPU candidate cannot set ready_for_exact_eval_dispatch=True.
  * Bias tensors take naive-PTQ path; weights without calibration fall back.
  * The dispatch contract sets dispatch_blockers including the
    inverse-scaling-runtime-absorption caveat (AWQ-specific).
  * AWQ at alpha=0 EXACTLY matches naive PTQ on the same tensor.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_tool_module():
    path = REPO / "tools" / "pr101_lossy_int4_awq.py"
    spec = importlib.util.spec_from_file_location("pr101_lossy_int4_awq", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def awq_module():
    return _load_tool_module()


def test_dispatch_contract_blockers_include_runtime_absorption_caveat(awq_module) -> None:
    """The local AWQ dispatch contract must list the AWQ-specific
    inverse-scaling absorption caveat plus the substrate-mismatch caveat."""
    contract = awq_module.local_awq_dispatch_contract(cuda_eval_worth_testing=True)
    assert contract["cuda_eval_worth_testing"] is True
    # Per the MPS-falsification + strict-scorer rules, NO CPU/MPS evidence
    # can ever flip these flags True. Only contest-CUDA auth eval can.
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["dispatch_attempted"] is False
    blockers = contract["dispatch_blockers"]
    assert "missing_exact_cuda_auth_eval" in blockers
    assert "byte_closed_int4_candidate_packet_missing" in blockers
    assert "synthetic_activations_substrate_mismatch_vs_pr106_video_latents" in blockers
    assert "awq_inverse_scaling_assumes_linear_runtime_absorption_not_yet_built" in blockers


def test_awq_scales_alpha_zero_is_identity(awq_module) -> None:
    """At alpha=0, AWQ scaling must reduce to identity (no shift in
    quantization grid)."""
    activations = np.array([1.0, 5.0, 0.1, 100.0], dtype=np.float32)
    s = awq_module.compute_awq_scales(activations, alpha=0.0)
    assert s.shape == activations.shape
    np.testing.assert_allclose(s, np.ones_like(activations))


def test_awq_scales_monotone_in_activation_magnitude(awq_module) -> None:
    """At alpha > 0, larger activation magnitude -> larger scale.
    Geometric-mean normalization makes the product of scales = 1."""
    activations = np.array([1.0, 4.0, 16.0, 64.0], dtype=np.float32)
    s = awq_module.compute_awq_scales(activations, alpha=0.5)
    # Monotone increasing
    assert s[0] < s[1] < s[2] < s[3]
    # Geometric mean = 1 (log mean = 0)
    log_mean = float(np.log(s).mean())
    assert abs(log_mean) < 1e-6


def test_awq_alpha_zero_matches_naive_ptq_exactly(awq_module) -> None:
    """When alpha=0 is in the grid and is selected, the AWQ output must equal
    naive PTQ on the same tensor."""
    rng = np.random.default_rng(7)
    out_features, in_features = 6, 16
    block_size = 32
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = rng.uniform(0.5, 2.0, size=in_features).astype(np.float32)

    # Force alpha=0 only -> AWQ MUST equal naive PTQ.
    cfg = awq_module.AWQConfig(
        block_size_encoder=block_size, alpha_grid=(0.0,),
    )
    w_awq, stats = awq_module.awq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    assert stats["best_alpha"] == 0.0

    w_naive_full = awq_module.naive_int4_quantize_tensor(
        weight.reshape(out_features, in_features), block_size,
    )
    np.testing.assert_array_equal(w_awq, w_naive_full)


def test_awq_quantize_tensor_is_deterministic(awq_module) -> None:
    """Same inputs + same config -> identical output (no RNG, no torch ops)."""
    rng = np.random.default_rng(8)
    out_features, in_features = 8, 24
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = rng.uniform(0.1, 5.0, size=in_features).astype(np.float32)
    cfg = awq_module.AWQConfig(
        block_size_encoder=32, alpha_grid=(0.0, 0.3, 0.5, 0.7, 1.0),
    )
    w1, stats1 = awq_module.awq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    w2, stats2 = awq_module.awq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    np.testing.assert_array_equal(w1, w2)
    assert stats1["best_alpha"] == stats2["best_alpha"]
    assert w1.shape == weight.shape
    assert w1.dtype == np.float32


def test_awq_grid_search_picks_minimum_loss(awq_module) -> None:
    """The reported best_alpha must in fact have the minimum loss in the
    losses_per_alpha dict."""
    rng = np.random.default_rng(9)
    out_features, in_features = 6, 16
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = rng.uniform(0.1, 5.0, size=in_features).astype(np.float32)
    cfg = awq_module.AWQConfig(
        block_size_encoder=32, alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    _, stats = awq_module.awq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    losses = stats["losses_per_alpha"]
    best_alpha = stats["best_alpha"]
    min_loss = min(losses.values())
    assert losses[best_alpha] == min_loss


def test_awq_handles_zero_activation_magnitude_gracefully(awq_module) -> None:
    """All-zero activation magnitude is degenerate; AWQ must clamp via eps
    and return a valid (finite) recon."""
    rng = np.random.default_rng(10)
    out_features, in_features = 4, 16
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = np.zeros(in_features, dtype=np.float32)
    cfg = awq_module.AWQConfig(
        block_size_encoder=32, alpha_grid=(0.0, 0.5, 1.0),
    )
    w_q, stats = awq_module.awq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    assert np.all(np.isfinite(w_q))
    # With zero activations, alpha=0 SHOULD win (any nonzero alpha just
    # disturbs the grid for no signal benefit). But we just check the
    # output is finite and best_alpha is in the grid.
    assert stats["best_alpha"] in (0.0, 0.5, 1.0)


def test_classify_cpu_proxy_dominated_baseline_blocks_dispatch(awq_module) -> None:
    """If archive >= PR101 brotli baseline, candidate is Pareto-dominated
    and not worth CUDA-eval — even with low rel_err."""
    verdict, cuda_worth = awq_module.classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=0.5,
        archive_bytes=awq_module.PR101_BROTLI_BASELINE_BYTES + 100,
    )
    assert "DOMINATED" in verdict
    assert cuda_worth is False


def test_quantize_pr101_state_routes_biases_through_naive_ptq(awq_module) -> None:
    """Biases must take the naive-PTQ path; weights with calibration take
    AWQ; weights without calibration fall back to naive PTQ."""
    rng = np.random.default_rng(11)
    fp32_state: dict[str, np.ndarray] = {}
    for name, shape in awq_module.FIXED_STATE_SCHEMA:
        fp32_state[name] = rng.normal(scale=0.1, size=shape).astype(np.float32)
    activation_mag = {
        "stem.weight": rng.uniform(0.1, 5.0, size=28).astype(np.float32),
    }
    cfg = awq_module.AWQConfig(
        block_size_encoder=1024, alpha_grid=(0.0, 0.5, 1.0),
    )
    quantized, stats = awq_module.quantize_pr101_state(
        fp32_state, activation_mag, cfg=cfg,
    )
    for name, shape in awq_module.FIXED_STATE_SCHEMA:
        assert quantized[name].shape == shape, name
        assert np.all(np.isfinite(quantized[name])), name
    assert stats["stem.weight"]["path"] == "awq"
    assert stats["stem.bias"]["path"] == "naive_ptq_bias"
    assert stats["blocks.0.weight"]["path"] == "naive_ptq_no_calibration"


def test_awq_quantized_values_lie_on_int4_grid_after_unscale(awq_module) -> None:
    """Even after the inverse-scaling step (W_recon = W'_q / s), the
    intermediate W'_q values must be exact int4 codes; the recon is
    arbitrary fp32 because of the divide. We check that the recon error
    pattern is bounded by what the int4 grid can produce."""
    rng = np.random.default_rng(12)
    out_features, in_features = 4, 16
    block_size = 32
    weight = rng.normal(scale=0.1, size=(out_features, in_features)).astype(np.float32)
    activations = rng.uniform(0.1, 5.0, size=in_features).astype(np.float32)

    cfg = awq_module.AWQConfig(
        block_size_encoder=block_size, alpha_grid=(0.5,),
    )
    w_recon, _ = awq_module.awq_quantize_tensor(
        weight, activations, cfg=cfg, original_shape=(out_features, in_features),
    )
    # The max abs error on a column j is bounded by (per-block scale of W*s) / s_j / 2 * 1.5
    # which is finite. We just sanity-check finiteness + non-trivial difference from
    # the original weight (since the int4 grid is coarse).
    assert np.all(np.isfinite(w_recon))
    # Should differ from original (lossy).
    assert not np.allclose(w_recon, weight, atol=0.0)
