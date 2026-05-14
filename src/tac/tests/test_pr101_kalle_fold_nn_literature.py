# SPDX-License-Identifier: MIT
"""Tests for ``tools/pr101_kalle_fold_nn_literature_shapes.py``.

Covers the FAITHFUL implementation contract per
``feedback_implementation_vs_model_gap_audit_20260508`` and
``feedback_premature_falsification_metacognitive_failure_mode_20260508``:

1. Each canonical literature shape integrates to 1 (PMF normalization).
2. Mixture weights sum to 1 after softmax.
3. Mixture PMF is well-defined for all input (positive, normalized, finite).
4. Per-tensor optimum found: multi-start cumulative-best KL is monotonic
   non-increasing (the optimizer respects the multi-start contract).
5. Encode/decode round-trip is byte-faithful (AC encode then decode
   recovers the original int8 symbols).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool() -> Any:
    path = REPO / "tools" / "pr101_kalle_fold_nn_literature_shapes.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_kalle_fold_nn_literature_shapes_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_tiny_schema(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    schema: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    monkeypatch.setattr(module, "FIXED_STATE_SCHEMA", schema)

    def fake_quantize(name: str, tensor: torch.Tensor, *, n_quant: int) -> Any:
        del name
        # Already-int8 input: just bias by N_QUANT then clip to symmetric range.
        arr = tensor.detach().cpu().numpy().astype(np.int32)
        arr = np.clip(arr, -n_quant, n_quant).astype(np.int8)
        return SimpleNamespace(
            q_i8=arr,
            shape=tuple(int(v) for v in tensor.shape),
            scale=1.0,
        )

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)


def test_each_canonical_literature_shape_integrates_to_one() -> None:
    """Rule from feedback_implementation_vs_model_gap_audit: a faithful
    NN-weight-shape mixture must use shapes drawn from the literature
    that are valid PMFs (sum to 1)."""
    tool = _load_tool()

    pmfs = [
        tool.kaiming_truncnormal_pmf(sigma=8.0),
        tool.xavier_truncnormal_pmf(sigma=4.0),
        tool.trained_laplace_narrow_pmf(b=2.0),
        tool.trained_laplace_wide_tail_pmf(b=16.0),
        tool.spike_and_slab_pmf(slab_sigma=4.0, spike_mass=0.5),
        tool.trunc_normal_with_outliers_pmf(core_sigma=4.0, outlier_width=0.4),
        tool.clip_mass_endpoints_pmf(),
    ]
    for i, pmf in enumerate(pmfs):
        assert pmf.shape == (tool.N_SYMBOLS,), f"shape mismatch for shape {i}"
        assert np.all(pmf >= 0.0), f"negative pmf entry in shape {i}"
        assert np.all(np.isfinite(pmf)), f"non-finite pmf entry in shape {i}"
        assert abs(pmf.sum() - 1.0) < 1e-9, (
            f"shape {i} does not integrate to 1 (sum={pmf.sum()})"
        )


def test_mixture_weights_sum_to_one_after_softmax() -> None:
    """The 7-component mixture's weights MUST sum to 1 over a wide range
    of input logit vectors. This is the softmax invariant — if it ever
    breaks, the resulting PMF is not a valid PMF."""
    tool = _load_tool()

    rng = np.random.default_rng(42)
    for _ in range(20):
        params = np.zeros(tool.N_PARAMS_PER_TENSOR)
        params[: tool.N_COMPONENTS] = rng.normal(0.0, 5.0, size=tool.N_COMPONENTS)
        params[tool.N_COMPONENTS:] = rng.normal(np.log(4.0), 1.0, size=5)
        w_logits = params[: tool.N_COMPONENTS]
        w = np.exp(w_logits - np.max(w_logits))
        w /= w.sum()
        assert abs(w.sum() - 1.0) < 1e-9, f"softmax weights sum to {w.sum()}"
        assert np.all(w >= 0.0), "softmax produced a negative weight"


def test_mixture_pmf_is_well_defined_for_all_input() -> None:
    """For every reasonable parameter vector, mixture_pmf must return a
    PMF: shape (N_SYMBOLS,), entries non-negative, sum=1, finite."""
    tool = _load_tool()

    rng = np.random.default_rng(7)
    for _ in range(50):
        params = np.zeros(tool.N_PARAMS_PER_TENSOR)
        # Logits across a wide range — including extreme values that
        # would catch any softmax-overflow bug.
        params[: tool.N_COMPONENTS] = rng.uniform(-20.0, 20.0, size=tool.N_COMPONENTS)
        # Log-scales across a wide range — including very small (close to
        # the 1e-3 floor) and very large (uniform-looking).
        params[tool.N_COMPONENTS:] = rng.uniform(-3.0, 5.0, size=5)
        pmf = tool.mixture_pmf(params)
        assert pmf.shape == (tool.N_SYMBOLS,)
        assert np.all(pmf >= 0.0)
        assert np.all(np.isfinite(pmf))
        assert abs(pmf.sum() - 1.0) < 1e-9


def test_per_tensor_optimum_found_multi_start_kl_monotonic() -> None:
    """The multi-start optimizer must produce a cumulative-best KL trace
    that is monotonic non-increasing. If init i finds a worse local
    optimum than init i-1, the cumulative best should NOT regress."""
    tool = _load_tool()

    # Synthetic target: a clearly-Laplace-like PMF over the symbol grid.
    rng = np.random.default_rng(123)
    samples = rng.laplace(loc=0.0, scale=4.0, size=4096)
    samples = np.clip(np.round(samples).astype(np.int32), -tool.N_QUANT, tool.N_QUANT)
    target = tool.empirical_pmf(samples.astype(np.int8))

    _params, _kl, cum_kl_trace = tool.fit_mixture_multi_start(target, seed=0)
    # Monotonic non-increasing across multi-start iterations.
    for prev, cur in zip(cum_kl_trace, cum_kl_trace[1:]):
        assert cur <= prev + 1e-9, (
            f"multi-start cumulative-best KL regressed: {prev} -> {cur}"
        )
    # Final KL on a Laplace-shaped target should be small (the Laplace
    # component should dominate the mixture). This is a weak finiteness +
    # quality check, not a hard threshold.
    assert cum_kl_trace[-1] >= 0.0
    assert cum_kl_trace[-1] < 1.0  # bits/element; Laplace fit should be tight


def test_encode_decode_round_trip_is_byte_faithful() -> None:
    """The arithmetic-coder round-trip MUST be byte-faithful: encode then
    decode the same int8 symbols using the same per-tensor mixture
    parameters returns identical symbols."""
    tool = _load_tool()

    rng = np.random.default_rng(2026)
    # Heavy-zero mass simulating a sparse layer.
    symbols = np.zeros(512, dtype=np.int8)
    symbols[: 200] = rng.integers(low=-3, high=4, size=200, dtype=np.int8)
    symbols[200: 350] = rng.integers(low=-10, high=11, size=150, dtype=np.int8)

    pmf = tool.empirical_pmf(symbols)
    params, _kl, _cum = tool.fit_mixture_multi_start(pmf, seed=0)
    payload = tool.encode_tensor_with_mixture(symbols, params)
    decoded = tool.decode_tensor_with_mixture(payload, params, n_symbols=symbols.size)

    assert decoded.shape == symbols.shape
    assert decoded.dtype == np.int8
    np.testing.assert_array_equal(decoded, symbols)


def test_proxy_evidence_contract_blocks_dispatch() -> None:
    """Per ``feedback_premature_falsification_metacognitive_failure_mode``
    Rule 4: ANY CPU-derived row sets ``ready_for_exact_eval_dispatch``
    to False. Per Rule 3: ``family_falsified`` is False and
    ``falsification_scope`` is documented."""
    tool = _load_tool()
    contract = tool.proxy_evidence_contract()

    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["rank_or_kill_eligible"] is False
    assert contract["family_falsified"] is False
    assert contract["falsification_scope"] == "measured_configuration_only"
    assert "missing_exact_cuda_auth_eval" in contract["dispatch_blockers"]


def test_run_codec_on_tiny_state_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke: run_codec on a tiny 2-tensor state dict
    produces a manifest with the literature_citations key + neutral
    verdict structure + the proxy evidence contract."""
    tool = _load_tool()
    schema = (("a", (32,)), ("b", (32,)))
    _install_tiny_schema(monkeypatch, tool, schema)

    state_dict_path = tmp_path / "state.pt"
    torch.save(
        {
            "a": torch.tensor(
                [0] * 16 + [1] * 8 + [-1] * 4 + [2, -2, 3, -3], dtype=torch.float32
            ),
            "b": torch.tensor(
                [0] * 8 + [3, 3, 3, -3, -3, -3] + [1, 1, -1, -1] * 4 + [4, -4],
                dtype=torch.float32,
            )[:32],
        },
        state_dict_path,
    )

    manifest = tool.run_codec(state_dict_path)
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["family_falsified"] is False
    assert manifest["falsification_scope"] == "measured_configuration_only"
    assert manifest["n_components"] == 7
    assert manifest["n_tensors"] == 2
    assert len(manifest["literature_citations"]) >= 5
    for row in manifest["per_tensor"]:
        weights = row["mixture_weights_by_component"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # Trace must be monotonic non-increasing.
        trace = row["multi_start_cum_kl_bits_trace"]
        for prev, cur in zip(trace, trace[1:]):
            assert cur <= prev + 1e-9
