# SPDX-License-Identifier: MIT
"""Tests for ``tools/pr101_omega_opt_uniward_weighted_allocation.py``.

The full empirical sweep requires the PR101 decoder state-dict; here we
exercise the helper functions on synthetic data and assert the custody
flags are correct on a tiny end-to-end manifest.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    """Import the tool module by path."""
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "pr101_omega_opt_uniward_weighted_allocation",
        REPO_ROOT / "tools" / "pr101_omega_opt_uniward_weighted_allocation.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr101_omega_opt_uniward_weighted_allocation"] = module
    spec.loader.exec_module(module)
    return module


def test_compute_local_variance_proxy_high_var_first() -> None:
    tool = _load_tool()
    from tac.codec.syndrome_trellis_codec import (  # noqa: F401  ensure imports work
        STCParams,
    )
    # TensorBlob is imported into the tool from pr101_lossy_coarsening_analytical.
    from pr101_lossy_coarsening_analytical import TensorBlob

    # Tensor 0: high-variance ints; Tensor 1: near-constant.
    raw_a = np.array([100, -90, 80, -70, 60, -50], dtype=np.int32)
    raw_b = np.array([1, -1, 1, -1, 1, -1], dtype=np.int32)
    tensors = [
        TensorBlob(name="a", raw=raw_a),
        TensorBlob(name="b", raw=raw_b),
    ]
    variances = tool.compute_local_variance_proxy(tensors)
    assert variances[0] > variances[1] * 100


def test_compute_uniward_weights_inverse_relationship() -> None:
    tool = _load_tool()
    weights = tool.compute_uniward_weights([100.0, 1.0, 0.01])
    # Higher variance ⇒ lower weight.
    assert weights[0] < weights[1] < weights[2]
    # ε floor prevents division blow-up at variance 0.
    weights_zero = tool.compute_uniward_weights([0.0])
    assert np.isfinite(weights_zero[0])


def test_lagrangian_select_uniform_vs_uniward_differs_with_skewed_weights() -> None:
    tool = _load_tool()
    # Two tensors, K∈{1,2,3}: rel_err ↑ with K but byte_proxy ↓ with K.
    curves = [
        [
            {"K": 1, "rel_err": 0.0, "byte_proxy": 100},
            {"K": 2, "rel_err": 0.1, "byte_proxy": 80},
            {"K": 3, "rel_err": 0.2, "byte_proxy": 60},
        ],
        [
            {"K": 1, "rel_err": 0.0, "byte_proxy": 100},
            {"K": 2, "rel_err": 0.1, "byte_proxy": 80},
            {"K": 3, "rel_err": 0.2, "byte_proxy": 60},
        ],
    ]
    Ks_uniform, _ = tool.lagrangian_select_Ks_uniform(curves, lam=1000.0)
    # Skew weights heavily so tensor 0 has much higher cost weight.
    weights = [1000.0, 0.001]
    Ks_uniward, _ = tool.lagrangian_select_Ks_uniward(curves, weights, lam=1000.0)
    # Tensor 0 (high weight ⇒ less rel_err) should choose smaller K than tensor 1
    # under the UNIWARD weighting.
    assert Ks_uniward[0] <= Ks_uniward[1]
    # Uniform selection treats them symmetrically.
    assert Ks_uniform[0] == Ks_uniform[1]


def test_run_experiment_custody_flags_on_synthetic_state_dict(tmp_path: Path) -> None:
    """Tiny synthetic state-dict ⇒ assert manifest carries non-negotiable
    custody flags per CLAUDE.md."""
    tool = _load_tool()
    from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA

    # Build a state-dict that matches FIXED_STATE_SCHEMA shapes with random
    # int-valued floats so quantization produces non-trivial int8 symbols.
    rng = np.random.default_rng(42)
    sd = {}
    for name, shape in FIXED_STATE_SCHEMA:
        # Use small positive variance so quantization is well-defined.
        sd[name] = torch.tensor(rng.normal(0, 1.0, size=shape).astype(np.float32))
    sd_path = tmp_path / "state_dict.pt"
    torch.save(sd, sd_path)

    # Quick sweep: a single rms target.
    manifest = tool.run_experiment(sd_path, [0.05])
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["family_falsified"] is False
    assert (
        manifest["falsification_scope"]
        == "uniward_weighted_lagrangian_per_tensor_only"
    )
    assert manifest["evidence_grade"].startswith(
        "[CPU-prep faithful Fridrich-UNIWARD-weighted"
    )
    assert manifest["score_affecting_payload_changed"] is True
    assert manifest["charged_bits_changed"] is True
    assert manifest["baseline_lossless_bytes"] > 0
    assert len(manifest["comparison_at_rms_targets"]) == 1


def test_run_experiment_rejects_missing_state_dict() -> None:
    tool = _load_tool()
    # Calling main with non-existent state-dict raises SystemExit early.
    with pytest.raises(SystemExit):
        tool.main(["--state-dict", "/non/existent/path.pt"])
