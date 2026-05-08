"""Tests for tools/pr101_tiny_nn_transformer.py.

Validates:
1. Module surface (proxy_evidence_contract, TinyTransformerPredictor,
   run_codec, encode_with_transformer).
2. proxy_evidence_contract canonical fields & dispatch_blockers.
3. TinyTransformerPredictor parameter count in audit range (1K-5K).
4. Causal masking: at position t, output depends only on prev_symbol[<=t].
5. End-to-end run_codec on tiny synthetic state_dict produces a valid
   manifest.
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
    path = REPO / "tools" / "pr101_tiny_nn_transformer.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_tiny_nn_transformer_under_test", path
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


def test_transformer_module_surface_and_contract() -> None:
    tool = _load_tool()
    assert hasattr(tool, "proxy_evidence_contract")
    assert hasattr(tool, "TinyTransformerPredictor")
    assert hasattr(tool, "run_codec")
    assert hasattr(tool, "encode_with_transformer")
    contract = tool.proxy_evidence_contract()
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["evidence_grade"] == "[CPU-prep faithful Transformer test]"
    assert (
        "missing_exact_cuda_auth_eval"
        in contract["dispatch_blockers"]
    )


def test_transformer_param_count_in_audit_range() -> None:
    tool = _load_tool()
    model = tool.TinyTransformerPredictor()
    n = sum(p.numel() for p in model.parameters())
    # Per audit criterion: ~1K-5K params for tiny transformer
    assert 1_000 <= n <= 5_000, (
        f"Transformer params {n} outside audit range 1K-5K"
    )


def test_transformer_causal_mask_blocks_future() -> None:
    """At position t, output must depend only on prev_symbol[0..t]."""
    tool = _load_tool()
    model = tool.TinyTransformerPredictor(d_model=4, n_heads=2, d_ff=8)
    model.eval()
    T = 6
    prev_a = torch.zeros(1, T, dtype=torch.long)
    prev_b = torch.zeros(1, T, dtype=torch.long)
    # Differ only at the LAST position
    prev_a[0, T - 1] = 5
    prev_b[0, T - 1] = 200
    tid = torch.zeros(1, T, dtype=torch.long)
    with torch.no_grad():
        mean_a, ls_a = model(prev_a, tid)
        mean_b, ls_b = model(prev_b, tid)
    # Outputs at positions [0..T-2] must be identical (causal mask)
    np.testing.assert_allclose(
        mean_a[0, : T - 1].cpu().numpy(),
        mean_b[0, : T - 1].cpu().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ls_a[0, : T - 1].cpu().numpy(),
        ls_b[0, : T - 1].cpu().numpy(),
        atol=1e-5,
    )
    # And the last position MAY differ (it can attend to itself)
    # but we don't strictly require it to. Just sanity-check shapes.
    assert mean_a.shape == (1, T)


def test_transformer_run_codec_manifest_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "sd.pt")
    manifest = tool.run_codec(
        state_dict_path,
        d_model=4,
        n_heads=2,
        d_ff=8,
        epochs=1,
        lr=1e-2,
        bptt_window=4,
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
    spec = manifest["model_spec_match"]
    assert spec["actual_params"] > 0


def test_transformer_run_codec_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "sd.pt")

    def _drop_unstable(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "input_state_dict"}

    m1 = tool.run_codec(
        state_dict_path,
        d_model=4,
        n_heads=2,
        d_ff=8,
        epochs=1,
        lr=1e-2,
        bptt_window=4,
        seed=99,
    )
    m2 = tool.run_codec(
        state_dict_path,
        d_model=4,
        n_heads=2,
        d_ff=8,
        epochs=1,
        lr=1e-2,
        bptt_window=4,
        seed=99,
    )
    assert _drop_unstable(m1) == _drop_unstable(m2)
