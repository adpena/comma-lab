"""Tests for tools/pr101_tiny_nn_cross_tensor.py.

Validates:
1. Module surface (proxy_evidence_contract, CrossTensorPMFPredictor,
   run_codec, encode_with_cross_tensor, build_cross_tensor_features).
2. proxy_evidence_contract canonical fields & dispatch_blockers.
3. CrossTensorPMFPredictor parameter count in audit range (1K-5K).
4. compute_moments returns 5 expected statistics with stable bounds.
5. build_cross_tensor_features: leave-one-out neighbor average is
   correct (decoder-reconstructible from own_moments alone).
6. End-to-end run_codec on tiny synthetic state_dict produces a
   manifest with side_info_own_moments_bytes accounted for.
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
    path = REPO / "tools" / "pr101_tiny_nn_cross_tensor.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_tiny_nn_cross_tensor_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_tiny_schema(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    monkeypatch.setattr(
        module,
        "FIXED_STATE_SCHEMA",
        (("a", (8,)), ("b", (8,)), ("c", (8,))),
    )
    monkeypatch.setattr(module, "N_TENSORS", 3)

    def fake_quantize(name: str, tensor: torch.Tensor, *, n_quant: int) -> Any:
        del name
        return SimpleNamespace(
            q_i8=(
                tensor.detach().cpu().numpy().astype(np.int32)
                .clip(-n_quant, n_quant)
                .astype(np.int8)
            ),
            shape=tuple(int(v) for v in tensor.shape),
            scale=1.0,
        )

    monkeypatch.setattr(module, "_quantize_tensor", fake_quantize)


def _tiny_state_dict(path: Path) -> Path:
    sd = {
        "a": torch.tensor([0, 1, 1, -1, 0, 1, 0, -1], dtype=torch.float32),
        "b": torch.tensor([2, 2, 0, 0, -2, -2, 1, 1], dtype=torch.float32),
        "c": torch.tensor([5, -5, 0, 0, 0, 5, -5, 0], dtype=torch.float32),
    }
    torch.save(sd, path)
    return path


def test_cross_tensor_module_surface_and_contract() -> None:
    tool = _load_tool()
    assert hasattr(tool, "proxy_evidence_contract")
    assert hasattr(tool, "CrossTensorPMFPredictor")
    assert hasattr(tool, "run_codec")
    assert hasattr(tool, "encode_with_cross_tensor")
    assert hasattr(tool, "build_cross_tensor_features")
    contract = tool.proxy_evidence_contract()
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert (
        contract["evidence_grade"]
        == "[CPU-prep faithful CrossTensor test]"
    )
    assert (
        "missing_exact_cuda_auth_eval"
        in contract["dispatch_blockers"]
    )


def test_cross_tensor_param_count_in_audit_range() -> None:
    tool = _load_tool()
    model = tool.CrossTensorPMFPredictor()
    n = sum(p.numel() for p in model.parameters())
    assert 1_000 <= n <= 5_000, (
        f"CrossTensor params {n} outside audit range 1K-5K"
    )


def test_compute_moments_shape_and_bounds() -> None:
    tool = _load_tool()
    syms = np.array([0, 1, -1, 2, -2, 0, 0, 0], dtype=np.int32)
    m = tool.compute_moments(syms)
    assert m.shape == (tool.N_MOMENT_FEATURES,)
    # Bounded normalized features
    assert -1.0 <= m[0] <= 1.0  # mean / N_QUANT
    assert m[1] >= 0.0  # std / N_QUANT
    assert -3.0 <= m[2] <= 3.0  # clipped skew
    # L0 fraction in [0,1]
    assert 0.0 <= m[4] <= 1.0
    # 4 zeros out of 8 = 0.5
    assert m[4] == pytest.approx(0.5)


def test_build_cross_tensor_features_leave_one_out_correct() -> None:
    """Neighbor moments must be the leave-one-out mean of own_moments.

    This is the contract the decoder will rely on: given own_moments
    side info, it can deterministically reconstruct neighbor_moments.
    """
    tool = _load_tool()
    sym_per_tensor = [
        np.array([0, 1, -1, 0], dtype=np.int32),
        np.array([2, 2, -2, -2], dtype=np.int32),
        np.array([5, -5, 0, 0], dtype=np.int32),
    ]
    own, neighbor = tool.build_cross_tensor_features(sym_per_tensor)
    n = len(sym_per_tensor)
    assert own.shape == (n, tool.N_MOMENT_FEATURES)
    assert neighbor.shape == (n, tool.N_MOMENT_FEATURES)
    total = own.sum(axis=0)
    for i in range(n):
        expected = (total - own[i]) / (n - 1)
        np.testing.assert_allclose(neighbor[i], expected, atol=1e-6)


def test_cross_tensor_run_codec_manifest_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "sd.pt")
    manifest = tool.run_codec(
        state_dict_path,
        hidden=4,
        epochs=1,
        lr=1e-2,
        batch_size=8,
        seed=0,
    )
    assert manifest["schema"] == tool.SCHEMA_VERSION
    assert manifest["tool"] == tool.TOOL_NAME
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["archive_bytes"] > 0
    assert manifest["model_blob_brotli_bytes"] > 0
    assert manifest["ac_payload_brotli_bytes"] > 0
    assert manifest["n_symbols"] == 24
    assert manifest["comparison_brotli_optuna_bytes"] == 178_144
    # 3 tensors * 5 moments * 2 bytes/fp16 = 30 bytes
    assert manifest["side_info_own_moments_bytes"] == 30
    spec = manifest["model_spec_match"]
    assert spec["actual_params"] > 0
