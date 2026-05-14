# SPDX-License-Identifier: MIT
"""Tests for tools/pr101_tiny_nn_lstm.py.

Validates:
1. Module imports & exposes expected surface (proxy_evidence_contract,
   TinyLSTMPredictor, run_codec, encode_with_lstm).
2. proxy_evidence_contract enforces ready_for_exact_eval_dispatch=False
   and the canonical CPU/MPS dispatch_blockers tuple.
3. TinyLSTMPredictor has parameter count in the audit range (500-2000).
4. encode_with_lstm produces a non-empty AC bitstream that decodes
   back to the original symbols (lossless roundtrip via constriction).
5. End-to-end run_codec on a tiny synthetic state_dict produces a
   manifest with all required schema fields.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool() -> Any:
    path = REPO / "tools" / "pr101_tiny_nn_lstm.py"
    spec = importlib.util.spec_from_file_location(
        "pr101_tiny_nn_lstm_under_test", path
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


def test_lstm_module_surface_and_contract() -> None:
    tool = _load_tool()
    assert hasattr(tool, "proxy_evidence_contract")
    assert hasattr(tool, "TinyLSTMPredictor")
    assert hasattr(tool, "run_codec")
    assert hasattr(tool, "encode_with_lstm")
    contract = tool.proxy_evidence_contract()
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["evidence_grade"] == "[CPU-prep faithful LSTM test]"
    assert (
        "missing_exact_cuda_auth_eval"
        in contract["dispatch_blockers"]
    )
    assert (
        "cpu_proxy_byte_anchor_not_score_evidence"
        in contract["dispatch_blockers"]
    )


def test_lstm_param_count_in_audit_range() -> None:
    tool = _load_tool()
    model = tool.TinyLSTMPredictor()
    n = sum(p.numel() for p in model.parameters())
    # Per audit criterion: ~500-2K params for tiny LSTM
    assert 500 <= n <= 2_000, f"LSTM params {n} outside audit range 500-2000"


def test_lstm_encode_decode_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the AC bitstream decodes back to the original symbols.

    This is the "real bitstream not theoretical NLL" check from
    CLAUDE.md; the codec must be lossless.
    """
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    sym_per_tensor, _scales = tool.collect_symbols(
        _tiny_state_dict(tmp_path / "sd.pt")
    )
    model = tool.TinyLSTMPredictor()
    model.eval()
    payload, stats = tool.encode_with_lstm(sym_per_tensor, model)
    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert stats["n_total_symbols"] == sum(s.size for s in sym_per_tensor)

    # Decode each tensor's stream and verify it matches
    import constriction

    qg = constriction.stream.model.QuantizedGaussian(
        -tool.N_QUANT, tool.N_QUANT
    )
    pos = 0
    streams: list[bytes] = []
    for _ in range(len(sym_per_tensor)):
        (length,) = struct.unpack("<I", payload[pos : pos + 4])
        pos += 4
        streams.append(payload[pos : pos + length])
        pos += length
    assert pos == len(payload)

    prev_seqs, tid_seqs, _ = tool._build_sequences(sym_per_tensor)
    idx_off = 0
    with torch.no_grad():
        for t_idx, syms in enumerate(sym_per_tensor):
            n = syms.size
            if n == 0:
                continue
            prev = prev_seqs[idx_off].unsqueeze(0)
            tid = tid_seqs[idx_off].unsqueeze(0)
            idx_off += 1
            mean, log_scale = model(prev, tid)
            mean_arr = mean.squeeze(0).cpu().numpy().astype(np.float64)
            scale_arr = (
                torch.exp(log_scale.squeeze(0))
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            scale_arr = np.maximum(scale_arr, 0.05)
            stream = streams[t_idx]
            assert len(stream) > 0
            # encoder.get_compressed() returns uint32 array -> bytes() is
            # raw LE u32 bytes; reinterpret as uint32 for the decoder.
            assert len(stream) % 4 == 0
            decoder = constriction.stream.queue.RangeDecoder(
                np.frombuffer(stream, dtype=np.uint32).copy()
            )
            decoded = decoder.decode(qg, mean_arr, scale_arr)
            np.testing.assert_array_equal(
                decoded.astype(np.int32),
                syms.flatten().astype(np.int32),
            )


def test_lstm_run_codec_manifest_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "sd.pt")
    manifest = tool.run_codec(
        state_dict_path,
        embed_dim=2,
        hidden=4,
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
    assert manifest["n_symbols"] == 24  # 8 + 8 + 8
    assert manifest["comparison_brotli_optuna_bytes"] == 178_144
    assert (
        manifest["delta_vs_brotli_optuna"]
        == manifest["archive_bytes"] - 178_144
    )
    spec = manifest["model_spec_match"]
    assert spec["actual_params"] > 0


def test_lstm_run_codec_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()
    _install_tiny_schema(monkeypatch, tool)
    state_dict_path = _tiny_state_dict(tmp_path / "sd.pt")

    def _drop_unstable(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "input_state_dict"}

    m1 = tool.run_codec(
        state_dict_path,
        embed_dim=2,
        hidden=4,
        epochs=1,
        lr=1e-2,
        bptt_window=4,
        seed=42,
    )
    m2 = tool.run_codec(
        state_dict_path,
        embed_dim=2,
        hidden=4,
        epochs=1,
        lr=1e-2,
        bptt_window=4,
        seed=42,
    )
    assert _drop_unstable(m1) == _drop_unstable(m2)
